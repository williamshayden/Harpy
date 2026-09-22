"""Real CPU optimization, portable reloads, and ordinary bounded audio episodes."""

# ruff: noqa: E402 -- optional Torch must be skipped before importing the network.

import shutil
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from harpy.tasks.sound_match.inverse import inverse_actor_spec, load_inverse_artifact, train_inverse
from harpy.tasks.sound_match.inverse_artifacts import state_sha256, training_membership
from harpy.tasks.sound_match.inverse_features import supervision_targets
from harpy.tasks.sound_match.inverse_network import predict_patch, supervised_loss
from harpy.tasks.sound_match.models import EpisodeSpec
from harpy.tasks.sound_match.patch import Patch
from harpy.tasks.sound_match.renderer import render
from harpy.tasks.sound_match.runner import evaluate


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("inverse")
    progress = []
    artifact = train_inverse(root / "quick", seed=0, profile="quick", progress=progress.append)
    yield artifact, progress
    torch.set_num_threads(previous)


def test_quick_performs_real_updates_and_selects_audio_validated_checkpoint(trained):
    artifact, progress = trained
    summary = artifact.summary
    assert summary["training_examples"] == summary["validation_examples"] == 8
    assert summary["total_optimizer_updates"] == 2
    assert summary["initial_state_sha256"] != summary["selected_state_sha256"]
    assert summary["selected_epoch"] == 1
    assert summary["epochs"][0]["mean_training_loss"] > 0
    assert len(summary["epochs"][0]["validation"]["cells"]) == 8
    assert len(progress) == 1
    assert progress[0]["validation"]["macro_distance"] > 0
    assert artifact.training["runtime"]["device"] == "cpu"
    assert artifact.training["runtime"]["torch_threads"] == 1
    assert artifact.manifest["quality_status"] == "experimental"


def test_reloaded_checkpoint_weights_predictions_and_ordinary_results_are_equal(trained):
    artifact, _ = trained
    first = artifact.load_model()
    second = load_inverse_artifact(artifact.root).load_model()
    assert state_sha256(first.state_dict()) == artifact.summary["selected_state_sha256"]
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
    _, _, _, entries = training_membership("quick")
    episodes = tuple(EpisodeSpec.from_dict(entry["episode"]) for entry in entries)
    for episode in episodes:
        audio = render(episode.target_patch)
        assert predict_patch(first, audio, episode.initial_patch, episode.editable_fields) == (
            predict_patch(second, audio, episode.initial_patch, episode.editable_fields)
        )
    actors = (
        inverse_actor_spec(artifact.root, name="inverse"),
        inverse_actor_spec(artifact.root, name="inverse-refine", refine=True),
    )
    for actor in actors:
        assert actor.metadata["inference_torch_version"] == str(torch.__version__)
    result = evaluate(episodes, actors, cache_max_audio_bytes=0)
    standalone_distances = {}
    for record in result.records:
        assert record.status == "submitted"
        assert record.metrics["model_calls"] == 1
        assert record.metrics["invalid_actions"] == 0
        assert record.metrics["evaluations"] <= 32
        assert record.metrics["decisions"] <= 64
        if record.actor_name == "inverse":
            standalone_distances[record.episode_id] = record.metrics["submitted_distance"]
    for record in result.records:
        if record.actor_name == "inverse-refine":
            assert record.metrics["submitted_distance"] <= standalone_distances[record.episode_id]
    assert np.mean(list(standalone_distances.values())) == pytest.approx(
        artifact.summary["selected_validation_macro_distance"], abs=1e-12
    )


def test_same_seed_quick_training_reproduces_selected_tensor_identity(trained):
    artifact, _ = trained
    replay = train_inverse(artifact.root.parent / "repeat", seed=0, profile="quick")
    assert replay.summary["selected_state_sha256"] == artifact.summary["selected_state_sha256"]
    assert replay.summary["epochs"] == artifact.summary["epochs"]


def test_artifact_metadata_reads_without_optional_model_import(trained):
    artifact, _ = trained
    code = """
import sys
from pathlib import Path
from harpy.tasks.sound_match.inverse_artifacts import load_inverse_artifact
artifact = load_inverse_artifact(Path(sys.argv[1]))
assert artifact.summary['total_optimizer_updates'] == 2
assert 'torch' not in sys.modules
assert 'harpy.tasks.sound_match.inverse_network' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code, str(artifact.root)], check=True)


def test_artifact_rejects_changed_weights_and_extra_files(trained, tmp_path):
    artifact, _ = trained
    copied = tmp_path / "copy"
    shutil.copytree(artifact.root, copied)
    loaded = load_inverse_artifact(copied)
    weights = copied / "model.pt"
    weights.write_bytes(weights.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="payload changed"):
        loaded.verify_unchanged()
    with pytest.raises(ValueError, match="immutable identity"):
        load_inverse_artifact(copied)
    shutil.copyfile(artifact.root / "model.pt", weights)
    (copied / "extra").write_text("unexpected")
    with pytest.raises(ValueError, match="exactly"):
        load_inverse_artifact(copied)


def test_actor_respects_no_edit_and_insufficient_propose_submit_budget(trained):
    artifact, _ = trained
    initial = Patch()
    episodes = (
        EpisodeSpec("locked", initial, initial, ()),
        EpisodeSpec("one-render", initial, initial, render_budget=1),
        EpisodeSpec("one-decision", initial, initial, decision_budget=1),
    )
    result = evaluate(episodes, (inverse_actor_spec(artifact.root),), cache_max_audio_bytes=0)
    for record in result.records:
        assert record.status == "submitted"
        assert record.metrics["model_calls"] == 0
        assert record.metrics["evaluations"] == 1
        assert record.submitted_candidate.patch == initial


def test_loss_has_no_gradient_for_locked_or_inactive_heads():
    torch.set_num_threads(1)
    initial = Patch(sustain_db=0)
    episode = EpisodeSpec("loss-mask", replace(initial, attack_ms=80), initial, ("attack_ms",))
    targets = {
        name: torch.from_numpy(value) for name, value in supervision_targets((episode,)).items()
    }
    predictions = {
        "notes": torch.zeros((1, 2401), requires_grad=True),
        "waveforms": torch.zeros((1, 4), requires_grad=True),
        "scalars": torch.zeros((1, 6), requires_grad=True),
    }
    loss = supervised_loss(predictions, targets)
    expected = (0.5 - targets["scalars"][0, 1]).square()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(predictions["notes"].grad) == 0
    assert torch.count_nonzero(predictions["waveforms"].grad) == 0
    assert torch.count_nonzero(predictions["scalars"].grad) == 1
    assert predictions["scalars"].grad[0, 1] != 0


def test_train_rejects_overwrite_or_non_cpu_without_fallback(trained):
    artifact, _ = trained
    with pytest.raises(FileExistsError):
        train_inverse(artifact.root)
    with pytest.raises(ValueError, match="CPU only"):
        train_inverse(artifact.root.parent / "cuda", device="cuda")
