"""Controller comparison holds estimator precision fixed."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from harpy.envs.models import ControlState, ObservationMode, PitchAction
from harpy.experiments import (
    EpisodeSpec,
    evaluate,
    oracle_actor_spec,
    pitch_actor_spec,
    reward_search_actor_spec,
    spectrum_peak_actor_spec,
)
from harpy.experiments.actors import EstimatorPlanner


def observation(controls=None):
    controls = ControlState() if controls is None else controls
    spectrum = np.zeros(1_961, dtype=np.float32)
    spectrum[(6_065 - 1_100) // 5] = 1
    return {
        "spectrum": spectrum,
        "target_note": np.int64(12),
        "controls": np.array(
            [controls.octaves, controls.semitones, controls.cents], dtype=np.int16
        ),
        "steps_remaining": np.int64(64),
    }


def run_perfect_controller(source, *, replan):
    class Perfect:
        def estimate(self, obs):
            controls = ControlState(*(int(value) for value in obs["controls"]))
            return 1_100 + 5 * ((source + controls.offset_cents - 1_100 + 2) // 5)

    actor = EstimatorPlanner(Perfect(), replan=replan)
    controls = ControlState()
    estimates = 0
    for _ in range(64):
        decision = actor.decide(observation(controls))
        estimates += decision.inference_count
        if decision.action is PitchAction.SUBMIT:
            return source + controls.offset_cents - 6_000, estimates
        controls, applied = controls.apply(decision.action)
        assert applied
    pytest.fail("controller exhausted its budget")


def test_committing_preserves_initial_quantized_estimate_instead_of_stopping_at_bin_edge():
    committed = [run_perfect_controller(source, replan=False) for source in range(6_063, 6_068)]
    repeated = [run_perfect_controller(source, replan=True) for source in range(6_063, 6_068)]
    assert [error for error, _ in committed] == [-2, -1, 0, 1, 2]
    assert [error for error, _ in repeated] == [-2] * 5
    assert [count for _, count in committed] == [1] * 5
    assert all(count > 1 for _, count in repeated)


def test_spectrum_actor_ignores_impossible_peak_and_commits_without_reinference():
    actor = spectrum_peak_actor_spec().factory()
    obs = observation()
    obs["spectrum"] *= 0.5
    obs["spectrum"][0] = 1
    first = actor.decide(obs)
    obs["spectrum"][:] = 0
    second = actor.decide(obs)
    assert first.estimated_candidate_cents == 6_065
    assert first.inference_count == 1
    assert second.estimated_candidate_cents is None and second.inference_count == 0


def test_builtin_tracks_and_controller_identities_are_distinct():
    committed = spectrum_peak_actor_spec()
    repeated = spectrum_peak_actor_spec(replan=True)
    assert committed.estimator == repeated.estimator and committed.decoder == repeated.decoder
    assert committed.controller != repeated.controller and committed.name != repeated.name
    assert oracle_actor_spec().observation_mode is ObservationMode.ORACLE
    assert reward_search_actor_spec().observation_mode is ObservationMode.REWARD_ONLY


def test_small_real_baseline_rollout_reports_both_perception_and_control(monkeypatch):
    import harpy.experiments.runner as runner

    monkeypatch.setattr(runner, "_source", lambda: {"package_sha256": "a" * 64})
    result = evaluate(
        (EpisodeSpec("near", 6_006, 12),),
        (spectrum_peak_actor_spec(), spectrum_peak_actor_spec(replan=True)),
    )
    assert result.rows[0]["mean_inferences"] == 1
    assert result.rows[1]["mean_inferences"] > 1
    assert all(row["submitted_success_rate"] == 1 for row in result.rows)
    assert all(row["initial_perception"]["estimates"] == 1 for row in result.rows)


def test_pitch_factory_shares_model_and_keeps_fresh_committed_controllers(monkeypatch, tmp_path):
    import harpy.experiments.actors as actors
    import harpy.experiments.artifacts as artifacts

    model, verified = object(), []
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(trainer="pitch", seed=9),
        verify_unchanged=lambda: verified.append(True),
        load_model=lambda *, device: model,
        provenance=lambda: {"model": "test"},
    )
    monkeypatch.setattr(artifacts, "load_artifact", lambda path: artifact)

    class Estimator:
        def __init__(self, loaded, device):
            assert loaded is model and device == "cpu"

        def estimate(self, observation):
            return 6_065

    monkeypatch.setattr(actors, "_PitchEstimator", Estimator)
    spec = pitch_actor_spec(tmp_path)
    first, second = spec.factory(), spec.factory()
    assert first is not second
    assert first.decide(observation()) == second.decide(observation())
    assert len(verified) == 2


def test_importing_experiments_does_not_load_torch_or_sb3():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, harpy.experiments; assert 'torch' not in sys.modules; "
            "assert 'stable_baselines3' not in sys.modules",
        ],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()


def test_pitch_estimator_has_no_planner_and_uses_only_spectrum(monkeypatch):
    torch = pytest.importorskip("torch")
    import harpy.experiments.actors as actors
    from harpy.learning.pitch_network import PitchEstimatorNetwork

    model = PitchEstimatorNetwork()
    estimator = actors._PitchEstimator(model, "cpu")
    inputs = []

    def forward(spectrum):
        inputs.append(spectrum.clone())
        logits = torch.zeros((1, 1_961), dtype=torch.float32)
        logits[0, 0] = 2
        logits[0, (6_065 - 1_100) // 5] = 1
        return logits

    model.forward = forward
    monkeypatch.setattr(actors, "minimum_action_plan", lambda *a, **kw: pytest.fail("planned"))
    assert estimator.estimate(observation()) == 6_065
    assert len(inputs) == 1 and inputs[0].shape == (1, 1_961)


@pytest.mark.parametrize("bad", ["shape", "dtype", "nonfinite"])
def test_pitch_estimator_rejects_bad_logits_before_decode(bad):
    torch = pytest.importorskip("torch")
    from harpy.experiments.actors import _PitchEstimator
    from harpy.learning.pitch_network import PitchEstimatorNetwork

    model = PitchEstimatorNetwork()
    estimator = _PitchEstimator(model, "cpu")
    logits = torch.zeros((1, 1_961), dtype=torch.float32)
    if bad == "shape":
        logits = logits[0]
    elif bad == "dtype":
        logits = logits.double()
    else:
        logits[0, 0] = float("nan")
    model.forward = lambda _: logits
    with pytest.raises(ValueError, match="logits"):
        estimator.estimate(observation())
