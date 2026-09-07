"""Complete direct-environment traces for hands-on learned-policy runs."""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass

import gymnasium
import numpy as np

import harpy.envs  # noqa: F401  # Register the frozen public environments.
from harpy.envs.models import (
    MAX_ABSOLUTE_ERROR_CENTS,
    MAX_STEPS,
    SUCCESS_TOLERANCE_CENTS,
    TARGET_MIN_COORDINATE,
    TARGET_NOTE_COUNT,
    EpisodeResult,
    PitchAction,
    TerminalReason,
)
from harpy.learning.actors import Actor
from harpy.learning.artifacts import canonical_json_bytes
from harpy.learning.errors import LearningExecutionError
from harpy.learning.models import ENVIRONMENT_ID
from harpy.learning.observations import preprocess_observation
from harpy.tuning import Tuning

FULL_RANGE_DEMONSTRATION_DISTRIBUTION_ID = "harpy-sine-pitch-full-range-demo-v1"


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        normalized = int(operator.index(value))
    except TypeError as error:
        raise ValueError(f"{field} must be an integer") from error
    if normalized < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return normalized


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be a finite number")
    return normalized


@dataclass(frozen=True, slots=True)
class TraceStep:
    """One actor-selected action and the environment's scalar reward."""

    step: int
    action: PitchAction
    reward: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "step", _integer(self.step, "step", minimum=1))
        if not isinstance(self.action, PitchAction):
            raise ValueError("action must be a PitchAction")
        object.__setattr__(self, "reward", _finite_float(self.reward, "reward"))


@dataclass(frozen=True, slots=True)
class EpisodeTrace:
    """Actor-visible actions paired with terminal-only evaluator truth."""

    environment_id: str
    distribution_id: str
    seed: int
    target_note_index: int
    target_note: str
    steps: tuple[TraceStep, ...]
    terminal_reason: TerminalReason
    final_absolute_error_cents: int
    submitted_success: bool
    action_count: int
    excess_actions: int | None
    total_return: float

    def __post_init__(self) -> None:
        if self.environment_id != ENVIRONMENT_ID:
            raise ValueError(f"environment_id must be {ENVIRONMENT_ID!r}")
        if self.distribution_id != FULL_RANGE_DEMONSTRATION_DISTRIBUTION_ID:
            raise ValueError(
                "distribution_id must identify the full-range demonstration distribution"
            )
        seed = _integer(self.seed, "seed")
        target_index = _integer(self.target_note_index, "target_note_index")
        if target_index >= TARGET_NOTE_COUNT:
            raise ValueError(f"target_note_index must be within 0..{TARGET_NOTE_COUNT - 1}")
        expected_note = _target_note_label(target_index)
        if self.target_note != expected_note:
            raise ValueError("target_note must match target_note_index")
        try:
            steps = tuple(self.steps)
        except TypeError as error:
            raise ValueError("steps must contain TraceStep values") from error
        if not steps or not all(isinstance(step, TraceStep) for step in steps):
            raise ValueError("steps must contain TraceStep values")
        if tuple(step.step for step in steps) != tuple(range(1, len(steps) + 1)):
            raise ValueError("steps must use consecutive one-based numbering")
        action_count = _integer(self.action_count, "action_count", minimum=1)
        if action_count != len(steps):
            raise ValueError("action_count must match the number of trace steps")
        if action_count > MAX_STEPS:
            raise ValueError(f"action_count must be at most {MAX_STEPS}")
        if not isinstance(self.terminal_reason, TerminalReason):
            raise ValueError("terminal_reason must be a TerminalReason")
        final_error = _integer(
            self.final_absolute_error_cents,
            "final_absolute_error_cents",
        )
        if final_error > MAX_ABSOLUTE_ERROR_CENTS:
            raise ValueError(
                f"final_absolute_error_cents must be at most {MAX_ABSOLUTE_ERROR_CENTS}"
            )
        if not isinstance(self.submitted_success, bool):
            raise ValueError("submitted_success must be a bool")
        excess = (
            None if self.excess_actions is None else _integer(self.excess_actions, "excess_actions")
        )
        submitted = steps[-1].action is PitchAction.SUBMIT
        if self.terminal_reason is TerminalReason.SUBMITTED_SUCCESS:
            if (
                not submitted
                or not self.submitted_success
                or final_error > SUCCESS_TOLERANCE_CENTS
                or excess is None
            ):
                raise ValueError("submitted-success trace fields must agree")
        elif self.terminal_reason is TerminalReason.SUBMITTED_FAILURE:
            if (
                submitted is False
                or self.submitted_success
                or final_error <= SUCCESS_TOLERANCE_CENTS
            ):
                raise ValueError("submitted-failure trace fields must agree")
            if excess is not None:
                raise ValueError("failed submissions must not report excess_actions")
        else:
            if submitted or self.submitted_success or action_count != MAX_STEPS:
                raise ValueError("budget exhaustion requires MAX_STEPS non-submit actions")
            if excess is not None:
                raise ValueError("budget exhaustion must not report excess_actions")
        if excess is not None and excess >= action_count:
            raise ValueError("excess_actions must be less than action_count")
        total_return = _finite_float(self.total_return, "total_return")
        step_total = sum(step.reward for step in steps)
        if not math.isclose(total_return, step_total, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("total_return must match the sum of step rewards")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "target_note_index", target_index)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "final_absolute_error_cents", final_error)
        object.__setattr__(self, "action_count", action_count)
        object.__setattr__(self, "excess_actions", excess)
        object.__setattr__(self, "total_return", total_return)


def trace_episode(actor: Actor, *, seed: int) -> EpisodeTrace:
    """Run one direct seeded full-range episode and read truth only after done."""

    normalized_seed = _integer(seed, "seed")
    if not callable(getattr(actor, "act", None)):
        raise ValueError("actor must provide a callable act method")
    env = gymnasium.make(ENVIRONMENT_ID)
    steps: list[TraceStep] = []
    try:
        observation, _ = env.reset(seed=normalized_seed)
        while True:
            actor_observation = _validated_actor_observation(observation)
            action = actor.act(actor_observation)
            if not isinstance(action, PitchAction):
                raise LearningExecutionError("actor must return a PitchAction")
            observation, reward, terminated, truncated, _ = env.step(action)
            try:
                normalized_reward = _finite_float(reward, "reward")
            except ValueError as error:
                raise LearningExecutionError(str(error)) from error
            steps.append(
                TraceStep(
                    step=len(steps) + 1,
                    action=action,
                    reward=normalized_reward,
                )
            )
            if terminated or truncated:
                result = env.unwrapped.episode_result
                if not isinstance(result, EpisodeResult):
                    raise LearningExecutionError(
                        "environment produced no valid EpisodeResult after done"
                    )
                break
        actions = tuple(step.action for step in steps)
        if result.actions != actions:
            raise LearningExecutionError("terminal EpisodeResult actions do not match the trace")
        if len(result.actions) != len(steps):
            raise LearningExecutionError(
                "terminal EpisodeResult action count does not match the trace"
            )
        if not math.isclose(
            result.total_return,
            sum(step.reward for step in steps),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise LearningExecutionError("terminal EpisodeResult return does not match the trace")
        return EpisodeTrace(
            environment_id=ENVIRONMENT_ID,
            distribution_id=FULL_RANGE_DEMONSTRATION_DISTRIBUTION_ID,
            seed=normalized_seed,
            target_note_index=result.target_note_index,
            target_note=_target_note_label(result.target_note_index),
            steps=tuple(steps),
            terminal_reason=result.terminal_reason,
            final_absolute_error_cents=result.final_absolute_error_cents,
            submitted_success=result.submitted_success,
            action_count=len(result.actions),
            excess_actions=result.excess_actions,
            total_return=result.total_return,
        )
    finally:
        env.close()


def _validated_actor_observation(observation: object) -> dict[str, object]:
    """Validate and own-copy one exact raw observation before actor access."""

    if not isinstance(observation, Mapping):
        raise LearningExecutionError("environment observation must be a mapping")
    raw_keys = ("spectrum", "target_note", "controls", "steps_remaining")
    if set(observation) != set(raw_keys):
        raise LearningExecutionError(
            "environment observation must contain exactly spectrum, target_note, controls, "
            "and steps_remaining"
        )
    snapshot: dict[str, object] = {}
    for key in raw_keys:
        value = observation[key]
        snapshot[key] = (
            np.array(value, copy=True, order="C") if isinstance(value, np.ndarray) else value
        )
    try:
        preprocess_observation(snapshot)
    except Exception as error:
        raise LearningExecutionError(f"invalid environment observation: {error}") from error
    return snapshot


def format_human_trace(episode: EpisodeTrace) -> str:
    """Render target and actor-safe steps before terminal-only truth."""

    if not isinstance(episode, EpisodeTrace):
        raise ValueError("trace must be an EpisodeTrace")
    lines = [
        f"Target: {episode.target_note} (index {episode.target_note_index})",
        f"Distribution: {episode.distribution_id} (full-range demonstration)",
        f"Seed: {episode.seed}",
    ]
    lines.extend(
        f"Step {step.step}: {step.action.label}; reward={_format_float(step.reward)}"
        for step in episode.steps
    )
    lines.extend(
        [
            "Terminal:",
            f"Reason: {episode.terminal_reason.value}",
            f"Final absolute error: {episode.final_absolute_error_cents} cents",
            f"Submitted success: {str(episode.submitted_success).lower()}",
            f"Action count: {episode.action_count}",
            (
                "Excess actions: undefined"
                if episode.excess_actions is None
                else f"Excess actions: {episode.excess_actions}"
            ),
            f"Total return: {_format_float(episode.total_return)}",
        ]
    )
    return "\n".join(lines) + "\n"


def trace_json_bytes(episode: EpisodeTrace) -> bytes:
    """Encode one complete finite sorted trace object plus one newline."""

    if not isinstance(episode, EpisodeTrace):
        raise ValueError("trace must be an EpisodeTrace")
    return canonical_json_bytes(
        {
            "environment_id": episode.environment_id,
            "distribution_id": episode.distribution_id,
            "seed": episode.seed,
            "target_note_index": episode.target_note_index,
            "target_note": episode.target_note,
            "steps": [
                {
                    "step": step.step,
                    "action": int(step.action),
                    "reward": step.reward,
                }
                for step in episode.steps
            ],
            "terminal_reason": episode.terminal_reason.value,
            "final_absolute_error_cents": episode.final_absolute_error_cents,
            "submitted_success": episode.submitted_success,
            "action_count": episode.action_count,
            "excess_actions": episode.excess_actions,
            "total_return": episode.total_return,
        }
    )


def trace_from_document(value: object) -> EpisodeTrace:
    """Validate a saved legacy trace for use inside a provenance envelope."""
    if not isinstance(value, Mapping) or set(value) != set(EpisodeTrace.__dataclass_fields__):
        raise ValueError("trace fields must match the complete EpisodeTrace document")
    if not isinstance(value["steps"], list):
        raise ValueError("trace steps must be a list")
    steps = []
    for item in value["steps"]:
        if not isinstance(item, Mapping) or set(item) != {"step", "action", "reward"}:
            raise ValueError("trace step fields must be exactly step, action, and reward")
        if type(item["action"]) is not int:
            raise ValueError("trace action must be an integer")
        steps.append(
            TraceStep(step=item["step"], action=PitchAction(item["action"]), reward=item["reward"])
        )
    fields = dict(value)
    fields["steps"] = tuple(steps)
    fields["terminal_reason"] = TerminalReason(value["terminal_reason"])
    return EpisodeTrace(**fields)


def _target_note_label(target_note_index: int) -> str:
    coordinate = TARGET_MIN_COORDINATE + target_note_index
    tuning = Tuning()
    frequency_hz = tuning.frequency_hz_for_midi_coordinate(coordinate)
    reading = tuning.describe_frequency(frequency_hz)
    if not math.isclose(reading.cents, 0.0, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("integer target coordinate did not map to an exact note")
    return reading.name


def _format_float(value: float) -> str:
    return format(value, ".17g")


__all__ = [
    "FULL_RANGE_DEMONSTRATION_DISTRIBUTION_ID",
    "EpisodeTrace",
    "TraceStep",
    "format_human_trace",
    "trace_episode",
    "trace_from_document",
    "trace_json_bytes",
]
