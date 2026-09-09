"""Real small rollouts exercise lifecycle, paired views, and artifact boundaries."""

from dataclasses import replace

import pytest

from harpy.envs.models import ObservationMode, PitchAction
from harpy.envs.robustness import CLEAN_CONDITION, ROBUSTNESS_CONDITIONS
from harpy.experiments import (
    ActorSpec,
    Decision,
    EpisodeSpec,
    evaluate,
    runner,
    spectrum_peak_actor_spec,
)


@pytest.fixture(autouse=True)
def stable_source(monkeypatch):
    # Concurrent source edits are checked separately; tiny rollouts test execution.
    monkeypatch.setattr(runner, "_source", lambda: {"package_sha256": "a" * 64})


def test_fresh_controller_per_episode_and_paired_view_ownership():
    created, seen = [], []

    class Actor:
        def __init__(self):
            self.used = False
            created.append(self)

        def act(self, observation):
            assert not self.used
            self.used = True
            assert set(observation) == {"spectrum", "controls", "target_note", "steps_remaining"}
            seen.append(observation["spectrum"].copy())
            observation["spectrum"][:] = 0
            observation["controls"][:] = 0
            return PitchAction.SUBMIT

    first = ActorSpec("first", Actor, ObservationMode.SPECTRUM, "test", "test", "test")
    second = replace(first, name="second")
    episodes = (EpisodeSpec("one", 6_000, 12, 4), EpisodeSpec("two", 6_001, 12, 5))
    result = evaluate(
        episodes, (first, second), conditions=(CLEAN_CONDITION, ROBUSTNESS_CONDITIONS[5])
    )
    assert len(created) == 8
    assert len({id(item) for item in created}) == 8
    assert all((seen[index] == seen[index + 1]).all() for index in range(0, 8, 2))
    assert [record.actor_name for record in result.records] == ["first", "second"] * 4
    assert all(record.terminal.submitted_success for record in result.records)
    assert len(result.paired_clean) == 2


def test_run_api_preserves_trace_and_explicit_unknown_provenance():
    from harpy.experiments import run

    actor = spectrum_peak_actor_spec()
    result = run(EpisodeSpec("trace", 6064, 12), actor)
    record = result.records[0]
    assert record.terminal.submitted_success
    assert record.trace is not None and len(record.trace) == len(record.terminal.actions)
    assert sum(record.inference_counts) == 1
    assert result.actors[0]["artifact_provenance"]["status"] == "unknown"
    assert result.actors[0]["artifact_provenance"]["reason"]


@pytest.mark.parametrize(
    "mode", [ObservationMode.SPECTRUM, ObservationMode.WAVEFORM, ObservationMode.REWARD_ONLY]
)
def test_only_reward_track_receives_public_feedback(mode):
    feedback = []

    class Actor:
        def __init__(self):
            self.count = 0

        def act(self, observation):
            assert "source_pitch_cents" not in observation
            if mode is ObservationMode.WAVEFORM:
                assert "waveform" in observation and "spectrum" not in observation
            self.count += 1
            return PitchAction.CENT_UP if self.count == 1 else PitchAction.SUBMIT

        def feedback(self, reward, info):
            assert set(info) == {"action_applied"}
            feedback.append((reward, info))

    spec = ActorSpec("custom", Actor, mode, "none", "none", "test")
    result = evaluate((EpisodeSpec("test", 6_000, 12),), (spec,), trace=True)
    assert len(feedback) == int(mode is ObservationMode.REWARD_ONLY)
    assert result.records[0].terminal.final_pitch_cents == 6_001
    assert len(result.records[0].trace) == 2


def test_artifact_changes_during_execution_reject_result(tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"original")

    class Actor:
        def act(self, observation):
            model.write_bytes(b"modified")
            return PitchAction.SUBMIT

    spec = ActorSpec("custom", Actor, ObservationMode.SPECTRUM, "none", "none", "test", (model,))
    with pytest.raises(ValueError, match="artifact files changed"):
        evaluate((EpisodeSpec("test", 6_000, 12),), (spec,))


def test_source_changes_during_execution_reject_result(monkeypatch):
    values = iter(({"revision": 1}, {"revision": 2}))
    monkeypatch.setattr(runner, "_source", lambda: next(values))
    with pytest.raises(ValueError, match="source changed"):
        evaluate((EpisodeSpec("test", 6_000, 12),), (spectrum_peak_actor_spec(),))


@pytest.mark.parametrize("bad", [3, True, Decision])
def test_malformed_actor_action_fails_and_closes_environment(monkeypatch, bad):
    closed = []
    original = runner.make_robustness_env

    def make(*args, **kwargs):
        env = original(*args, **kwargs)
        env.close = lambda: closed.append(True)
        return env

    monkeypatch.setattr(runner, "make_robustness_env", make)

    class Actor:
        def act(self, observation):
            return bad

    spec = ActorSpec("custom", Actor, ObservationMode.SPECTRUM, "none", "none", "test")
    with pytest.raises(ValueError, match="PitchAction"):
        evaluate((EpisodeSpec("test", 6_000, 12),), (spec,))
    assert closed == [True]


def test_preflight_rejects_duplicate_identity_before_factory_runs():
    def factory():
        pytest.fail("factory should not run")

    actor = ActorSpec("a", factory, ObservationMode.SPECTRUM, "none", "none", "test")
    episode = EpisodeSpec("same", 6_000, 12)
    with pytest.raises(ValueError, match="unique"):
        evaluate((episode, episode), (actor,))


def test_estimates_are_optional_but_recorded_inferences_are_explicit():
    class Actor:
        def decide(self, observation):
            return Decision(PitchAction.SUBMIT, 6_000, 2)

    spec = ActorSpec("two-inferences", Actor, ObservationMode.SPECTRUM, "test", "test", "test")
    result = evaluate((EpisodeSpec("test", 6_001, 12),), (spec,))
    row = result.rows[0]
    assert row["mean_inferences"] == 2
    assert row["initial_perception"]["mean_absolute_error_cents"] == 1
    assert result.records[0].trace is None
    assert len(result.records[0].estimates) == 1
