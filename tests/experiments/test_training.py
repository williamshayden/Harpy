"""Actual parameter updates, persisted reloads, and the training data boundary."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from harpy.experiments.artifacts import load_artifact
from harpy.experiments.training import ppo_training_episode, train
from harpy.learning.pitch_data import pitch_coordinate_split


def test_ppo_reset_sources_use_only_training_partition():
    split = pitch_coordinate_split()
    episodes = tuple(ppo_training_episode(seed=41, index=index) for index in range(500))
    sources = {episode.source_pitch_cents for episode in episodes}
    assert sources <= set(split.training_coordinates)
    assert sources.isdisjoint(split.validation_coordinates)
    assert sources.isdisjoint(split.iid_holdout_coordinates)
    assert episodes == tuple(ppo_training_episode(seed=41, index=index) for index in range(500))
    assert episodes != tuple(ppo_training_episode(seed=42, index=index) for index in range(500))


@pytest.fixture(scope="module", params=["pitch", "ppo"])
def trained_artifact(request, tmp_path_factory):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    trainer = request.param
    initial = {}
    with pytest.MonkeyPatch.context() as patch:
        if trainer == "pitch":
            import harpy.learning.pitch as pitch

            original = pitch._train_one_epoch

            def capture_initial(model, *args, **kwargs):
                if not initial:
                    initial.update(
                        {name: value.detach().clone() for name, value in model.state_dict().items()}
                    )
                return original(model, *args, **kwargs)

            patch.setattr(pitch, "_train_one_epoch", capture_initial)
        else:
            pytest.importorskip("stable_baselines3")
            import harpy.learning.ppo as ppo

            original = ppo.make_ppo_model

            def capture_initial(*args, **kwargs):
                model = original(*args, **kwargs)
                initial.update(
                    {
                        name: value.detach().clone()
                        for name, value in model.policy.state_dict().items()
                    }
                )
                return model

            patch.setattr(ppo, "make_ppo_model", capture_initial)
        artifact = train(
            trainer=trainer,
            profile="smoke",
            seed=3,
            output=tmp_path_factory.mktemp(trainer) / "artifact",
        )
    model = artifact.load_model()
    state = (model if trainer == "pitch" else model.policy).state_dict()
    assert initial and any(not torch.equal(initial[name], value) for name, value in state.items())
    assert all(torch.isfinite(value).all() for value in state.values())
    return artifact


def test_smoke_training_updates_parameters_and_reloads(trained_artifact):
    artifact = trained_artifact
    restored = load_artifact(artifact.root)
    assert restored == artifact
    restored.verify_unchanged()
    metadata = json.loads((artifact.root / "metadata.json").read_bytes())
    if artifact.manifest.trainer == "pitch":
        assert metadata["summary"]["training_examples"] == 256
        assert metadata["summary"]["validation_examples"] == 64
        assert metadata["runtime"]["stable_baselines3"] is None
    else:
        assert metadata["summary"]["completed_environment_steps"] == 2048
        assert (
            metadata["configuration"]["trajectory_scope"]
            == "actions may visit validation and holdout coordinates"
        )
        assert (
            metadata["configuration"]["checkpoint_selection"]
            == "final_training_step_no_performance_gate"
        )
    with pytest.raises(FileExistsError):
        train(trainer=artifact.manifest.trainer, output=artifact.root)


def test_saved_actor_completes_smoke_episodes(trained_artifact, monkeypatch):
    from harpy.envs.models import TerminalReason
    from harpy.experiments import runner
    from harpy.experiments.actors import pitch_actor_spec, ppo_actor_spec
    from harpy.experiments.protocols import smoke_episodes
    from harpy.experiments.results import ExperimentResult

    # Concurrent source edits are unrelated to this model-update integration test.
    source = runner._source()
    monkeypatch.setattr(runner, "_source", lambda: source)
    factory = pitch_actor_spec if trained_artifact.manifest.trainer == "pitch" else ppo_actor_spec
    result = runner.evaluate(smoke_episodes(), [factory(trained_artifact.root)])
    assert len(result.records) == len(smoke_episodes())
    assert all(record.terminal.terminal_reason in TerminalReason for record in result.records)
    assert ExperimentResult.from_document(result.to_document()) == result
    trained_artifact.verify_unchanged()


def test_pitch_lane_never_imports_sb3(tmp_path):
    script = """
import importlib.abc
import sys
from pathlib import Path
class BlockRL(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'stable_baselines3':
            raise AssertionError('pitch imported Stable-Baselines3')
sys.meta_path.insert(0, BlockRL())
from harpy.experiments.training import train
from harpy.learning.dependencies import require_pitch_dependencies
from harpy.learning.pitch_actor import PitchPlannerActor
require_pitch_dependencies().torch.set_num_threads(1)
artifact = train(output=Path(sys.argv[1]))
actor = PitchPlannerActor(artifact.load_model())
assert actor is not None
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "pitch")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "arguments",
    [{"trainer": "bc"}, {"seed": True}, {"seed": -1}, {"profile": "unknown"}, {"device": "gpu"}],
)
def test_invalid_training_request_does_not_create_output(tmp_path, arguments):
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        train(output=output, **arguments)
    assert not output.exists()


def test_invalid_saved_model_never_gets_a_completed_manifest(monkeypatch, tmp_path):
    import harpy.experiments.training as training

    def invalid_model(profile, seed, device, output):
        (output / "model.pt").write_bytes(b"invalid model")
        return {}, {}

    monkeypatch.setattr(training, "_train_pitch", invalid_model)
    output = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="could not be loaded"):
        training.train(output=output)
    assert output.is_dir()
    assert not (output / "manifest.json").exists()
