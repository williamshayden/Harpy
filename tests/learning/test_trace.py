"""Complete full-range hands-on learned-actor traces."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import harpy.learning.trace as trace
import harpy.learning.workflows as workflows
from harpy.envs.models import EpisodeResult, PitchAction, TerminalReason
from harpy.envs.sine_pitch import SinePitchEnv, _CandidateEvidence
from harpy.learning.artifacts import LoadedArtifact
from harpy.learning.errors import ArtifactError, LearningContractError, LearningExecutionError
from harpy.learning.models import DeviceName, ProfileName, TrainerKind
from harpy.learning.pitch_artifacts import LoadedPitchArtifact
from harpy.learning.pitch_data import PitchTrainerKind
from harpy.tuning import Tuning


class FastDirectEnv(SinePitchEnv):
    """Real direct state machine with cheap deterministic candidate evidence."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = False
        self.result_reads = 0
        self.reset_calls: list[dict[str, object]] = []
        self.step_calls = 0

    @property
    def episode_result(self) -> EpisodeResult:
        self.result_reads += 1
        if self._episode_result is None:
            raise AssertionError("episode_result was read before terminal truth was available")
        return self._episode_result

    def reset(self, **kwargs: Any):
        self.reset_calls.append(dict(kwargs))
        return super().reset(**kwargs)

    def step(self, action: int):
        self.step_calls += 1
        return super().step(action)

    def close(self) -> None:
        self.closed = True

    def _candidate_evidence(self, candidate_cents: int) -> _CandidateEvidence:
        spectrum = np.full((1_961,), (candidate_cents % 97) / 96.0, dtype=np.float32)
        spectrum.setflags(write=False)
        return _CandidateEvidence(candidate_audio=None, spectrum=spectrum)


class SequenceActor:
    def __init__(self, *actions: object) -> None:
        self._actions = iter(actions)
        self.observations: list[Mapping[str, object]] = []

    def act(self, observation: Mapping[str, object]):
        self.observations.append(observation)
        return next(self._actions)


def _install_env(monkeypatch: pytest.MonkeyPatch, env: FastDirectEnv) -> list[object]:
    calls: list[object] = []

    def make(environment_id: str):
        calls.append(environment_id)
        return env

    monkeypatch.setattr(trace.gymnasium, "make", make)
    return calls


def test_trace_uses_direct_registered_full_range_seed_and_terminal_truth_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = FastDirectEnv()
    make_calls = _install_env(monkeypatch, env)
    actor = SequenceActor(PitchAction.CENT_UP, PitchAction.SUBMIT)

    result = trace.trace_episode(actor, seed=417)

    assert make_calls == ["Harpy/SinePitch-v0"]
    assert env.reset_calls == [{"seed": 417}]
    assert env.result_reads == 1
    assert env.closed is True
    assert tuple(tuple(sorted(observation)) for observation in actor.observations) == (
        ("controls", "spectrum", "steps_remaining", "target_note"),
        ("controls", "spectrum", "steps_remaining", "target_note"),
    )
    assert result.environment_id == "Harpy/SinePitch-v0"
    assert result.distribution_id == trace.FULL_RANGE_DEMONSTRATION_DISTRIBUTION_ID
    assert result.seed == 417
    assert tuple(step.action for step in result.steps) == (
        PitchAction.CENT_UP,
        PitchAction.SUBMIT,
    )
    assert tuple(step.step for step in result.steps) == (1, 2)
    assert result.action_count == 2
    assert result.terminal_reason in {
        TerminalReason.SUBMITTED_SUCCESS,
        TerminalReason.SUBMITTED_FAILURE,
    }
    assert result.total_return == pytest.approx(sum(step.reward for step in result.steps))


def test_trace_maps_target_index_through_existing_tuning_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = FastDirectEnv()
    _install_env(monkeypatch, env)

    result = trace.trace_episode(SequenceActor(PitchAction.SUBMIT), seed=99)

    tuning = Tuning()
    coordinate = 48 + result.target_note_index
    expected = tuning.describe_frequency(tuning.frequency_hz_for_midi_coordinate(coordinate)).name
    assert result.target_note == expected


def test_trace_rejects_non_pitch_action_before_step_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = FastDirectEnv()
    _install_env(monkeypatch, env)

    with pytest.raises(LearningExecutionError, match="PitchAction"):
        trace.trace_episode(SequenceActor(PitchAction.SUBMIT.value), seed=0)

    assert env.step_calls == 0
    assert env.result_reads == 0
    assert env.closed is True


@pytest.mark.parametrize("poison_at", ["reset", "intermediate"])
def test_trace_rejects_hidden_observation_keys_before_they_reach_actor(
    poison_at: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PoisonedEnv(FastDirectEnv):
        def reset(self, **kwargs: Any):
            observation, info = super().reset(**kwargs)
            if poison_at == "reset":
                observation = {**observation, "source_pitch_cents": 4_400}
            return observation, info

        def step(self, action: int):
            observation, reward, terminated, truncated, info = super().step(action)
            if poison_at == "intermediate" and not (terminated or truncated):
                observation = {**observation, "current_pitch_cents": 4_500}
            return observation, reward, terminated, truncated, info

    env = PoisonedEnv()
    _install_env(monkeypatch, env)
    actor = SequenceActor(PitchAction.CENT_UP, PitchAction.SUBMIT)

    with pytest.raises(LearningExecutionError, match="exactly"):
        trace.trace_episode(actor, seed=1)

    expected_calls = 0 if poison_at == "reset" else 1
    assert len(actor.observations) == expected_calls
    assert env.closed is True


def test_trace_reads_each_environment_observation_value_once_before_copying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SingleReadMapping(Mapping[str, object]):
        def __init__(self, values: Mapping[str, object]) -> None:
            self._values = dict(values)
            self.reads = {key: 0 for key in values}

        def __getitem__(self, key: str) -> object:
            self.reads[key] += 1
            if self.reads[key] != 1:
                raise AssertionError(f"observation value {key!r} was read more than once")
            return self._values[key]

        def __iter__(self) -> Iterator[str]:
            return iter(self._values)

        def __len__(self) -> int:
            return len(self._values)

    class SingleReadEnv(FastDirectEnv):
        supplied: SingleReadMapping | None = None

        def reset(self, **kwargs: Any):
            observation, info = super().reset(**kwargs)
            self.supplied = SingleReadMapping(observation)
            return self.supplied, info

    env = SingleReadEnv()
    _install_env(monkeypatch, env)

    result = trace.trace_episode(SequenceActor(PitchAction.SUBMIT), seed=4)

    assert result.action_count == 1
    assert env.supplied is not None
    assert env.supplied.reads == {key: 1 for key in env.supplied}


def test_trace_closes_environment_when_actor_or_reset_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingActor:
        def act(self, observation: Mapping[str, object]) -> PitchAction:
            del observation
            raise RuntimeError("actor sentinel")

    actor_env = FastDirectEnv()
    _install_env(monkeypatch, actor_env)
    with pytest.raises(RuntimeError, match="actor sentinel"):
        trace.trace_episode(FailingActor(), seed=1)
    assert actor_env.closed is True

    class ResetFailureEnv(FastDirectEnv):
        def reset(self, **kwargs: Any):
            del kwargs
            raise RuntimeError("reset sentinel")

    reset_env = ResetFailureEnv()
    _install_env(monkeypatch, reset_env)
    with pytest.raises(RuntimeError, match="reset sentinel"):
        trace.trace_episode(SequenceActor(PitchAction.SUBMIT), seed=1)
    assert reset_env.closed is True


def test_trace_is_frozen_and_rejects_nonfinite_rewards() -> None:
    with pytest.raises(ValueError, match="finite"):
        trace.TraceStep(step=1, action=PitchAction.SUBMIT, reward=float("nan"))

    step = trace.TraceStep(step=1, action=PitchAction.SUBMIT, reward=-1.0)
    with pytest.raises(FrozenInstanceError):
        step.reward = 0.0  # type: ignore[misc]


def test_json_trace_is_strict_complete_repeatable_and_contains_no_hidden_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def capture() -> bytes:
        env = FastDirectEnv()
        _install_env(monkeypatch, env)
        episode = trace.trace_episode(SequenceActor(PitchAction.SUBMIT), seed=2026)
        return trace.trace_json_bytes(episode)

    first = capture()
    second = capture()

    assert first == second
    assert first.endswith(b"\n") and not first.endswith(b"\n\n")
    assert b"NaN" not in first and b"Infinity" not in first
    assert b"source_pitch" not in first
    assert b"initial_" not in first
    assert b"current_pitch" not in first
    assert b"timestamp" not in first
    assert b'"action":3' in first
    assert b'"terminal_reason":"submitted_' in first


def test_human_trace_prints_target_then_actions_then_terminal_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = FastDirectEnv()
    _install_env(monkeypatch, env)
    episode = trace.trace_episode(
        SequenceActor(PitchAction.CENT_DOWN, PitchAction.SUBMIT),
        seed=7,
    )

    rendered = trace.format_human_trace(episode)
    lines = rendered.splitlines()

    assert lines[0].startswith("Target: ")
    assert lines[1].startswith("Distribution: ")
    assert lines[2] == "Seed: 7"
    assert lines[3].startswith("Step 1: Cent Down; reward=")
    assert lines[4].startswith("Step 2: Submit; reward=")
    assert lines[5] == "Terminal:"
    assert lines[-1].startswith("Total return: ")
    assert rendered.endswith("\n")
    assert "source pitch" not in rendered.lower()
    assert "initial error" not in rendered.lower()


@pytest.mark.parametrize("seed", [True, -1, 1.5, "4"])
def test_trace_rejects_invalid_seed_before_environment_construction(
    seed: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trace.gymnasium,
        "make",
        lambda environment_id: pytest.fail(f"invalid seed constructed {environment_id}"),
    )

    with pytest.raises(ValueError, match="seed"):
        trace.trace_episode(SequenceActor(PitchAction.SUBMIT), seed=seed)  # type: ignore[arg-type]


def test_run_artifact_validates_complete_payload_before_actor_and_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    artifact = object.__new__(LoadedArtifact)
    object.__setattr__(artifact, "root", root.resolve())
    object.__setattr__(
        artifact,
        "manifest",
        SimpleNamespace(
            trainer=TrainerKind.BC,
            seed=0,
            profile=ProfileName.SMOKE,
        ),
    )
    events: list[str] = []
    monkeypatch.setattr(workflows, "load_artifact", lambda path: artifact)
    monkeypatch.setattr(
        workflows,
        "_peek_artifact_identity",
        lambda path: (1, ProfileName.SMOKE, TrainerKind.BC.value),
    )

    def validate(value: object) -> None:
        assert value is artifact
        events.append("validate")

    def load_actor(value: object, *, device: DeviceName):
        assert value is artifact and device is DeviceName.CPU
        assert events == ["validate"]
        events.append("actor")
        return object()

    sentinel = object()

    def capture(actor: object, *, seed: int):
        assert events == ["validate", "actor"]
        assert seed == 11
        events.append("trace")
        return sentinel

    monkeypatch.setattr(workflows, "_require_evaluation_device", lambda device: None)
    monkeypatch.setattr(workflows, "_validate_artifact_payload", validate)
    monkeypatch.setattr(workflows, "_load_artifact_actor", load_actor)
    monkeypatch.setattr(trace, "trace_episode", capture)

    assert workflows.run_artifact(root, seed=11) is sentinel
    assert events == ["validate", "actor", "trace"]


def test_run_artifact_rejects_invalid_seed_and_incomplete_artifact_before_actor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    loads: list[Path] = []
    monkeypatch.setattr(
        workflows,
        "load_artifact",
        lambda path: loads.append(path) or (_ for _ in ()).throw(ValueError("incomplete")),
    )
    monkeypatch.setattr(
        workflows,
        "_peek_artifact_identity",
        lambda path: (1, ProfileName.SMOKE, TrainerKind.BC.value),
    )
    monkeypatch.setattr(
        workflows,
        "_load_artifact_actor",
        lambda *args, **kwargs: pytest.fail("invalid artifact reached actor loading"),
    )

    with pytest.raises(LearningContractError, match="seed"):
        workflows.run_artifact(root, seed=-1)
    assert loads == []

    with pytest.raises(ArtifactError, match="incomplete"):
        workflows.run_artifact(root, seed=0)


def test_run_artifact_dispatches_pitch_without_v1_fallthrough(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "manifest.json").write_bytes(
        workflows.canonical_json_bytes(
            {
                "schema_version": 2,
                "profile": ProfileName.SMOKE.value,
                "trainer": PitchTrainerKind.PITCH.value,
            }
        )
    )
    artifact = object.__new__(LoadedPitchArtifact)
    object.__setattr__(artifact, "root", root.resolve())
    object.__setattr__(
        artifact,
        "manifest",
        SimpleNamespace(
            trainer=PitchTrainerKind.PITCH,
            profile=ProfileName.SMOKE,
            seed=0,
        ),
    )
    events: list[str] = []
    monkeypatch.setattr(workflows, "load_artifact", lambda path: artifact)
    monkeypatch.setattr(workflows, "_require_evaluation_device", lambda device: None)
    monkeypatch.setattr(
        workflows,
        "_validate_artifact_payload",
        lambda value: pytest.fail("pitch artifact reached legacy payload validation"),
    )
    monkeypatch.setattr(
        workflows,
        "_load_artifact_actor",
        lambda *args, **kwargs: pytest.fail("pitch artifact reached legacy actor loading"),
    )
    actor = object()

    def load_pitch(value: object, *, device: DeviceName) -> object:
        assert value is artifact and device is DeviceName.CPU
        events.append("pitch-actor")
        return actor

    sentinel = object()

    def capture(value: object, *, seed: int) -> object:
        assert value is actor and seed == 9
        events.append("trace")
        return sentinel

    monkeypatch.setattr(workflows, "_load_pitch_artifact_actor", load_pitch)
    monkeypatch.setattr(trace, "trace_episode", capture)

    assert workflows.run_artifact(root, seed=9) is sentinel
    assert events == ["pitch-actor", "trace"]


def test_trace_model_cross_checks_terminal_action_truth() -> None:
    with pytest.raises(ValueError, match="action_count"):
        trace.EpisodeTrace(
            environment_id="Harpy/SinePitch-v0",
            distribution_id=trace.FULL_RANGE_DEMONSTRATION_DISTRIBUTION_ID,
            seed=0,
            target_note_index=0,
            target_note="C2",
            steps=(trace.TraceStep(1, PitchAction.SUBMIT, -1.0),),
            terminal_reason=TerminalReason.SUBMITTED_FAILURE,
            final_absolute_error_cents=100,
            submitted_success=False,
            action_count=2,
            excess_actions=None,
            total_return=-1.0,
        )
