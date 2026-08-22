"""Immutable musical controls and evaluator-owned episode records."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import IntEnum, StrEnum

SOURCE_MIN_CENTS = 4_800
SOURCE_MAX_CENTS = 7_200
TARGET_MIN_COORDINATE = 48
TARGET_NOTE_COUNT = 25
OCTAVE_MIN, OCTAVE_MAX = -2, 2
SEMITONE_MIN, SEMITONE_MAX = -12, 12
CENT_MIN, CENT_MAX = -100, 100
SUCCESS_TOLERANCE_CENTS = 5
STRICT_TOLERANCE_CENTS = 1
MAX_STEPS = 64
MAX_ABSOLUTE_ERROR_CENTS = 6_100


class PitchAction(IntEnum):
    OCTAVE_DOWN = 0
    SEMITONE_DOWN = 1
    CENT_DOWN = 2
    SUBMIT = 3
    CENT_UP = 4
    SEMITONE_UP = 5
    OCTAVE_UP = 6

    @property
    def label(self) -> str:
        return {
            PitchAction.OCTAVE_DOWN: "Octave Down",
            PitchAction.SEMITONE_DOWN: "Semitone Down",
            PitchAction.CENT_DOWN: "Cent Down",
            PitchAction.SUBMIT: "Submit",
            PitchAction.CENT_UP: "Cent Up",
            PitchAction.SEMITONE_UP: "Semitone Up",
            PitchAction.OCTAVE_UP: "Octave Up",
        }[self]


class ObservationMode(StrEnum):
    SPECTRUM = "spectrum"
    ORACLE = "oracle"
    REWARD_ONLY = "reward_only"


class TerminalReason(StrEnum):
    SUBMITTED_SUCCESS = "submitted_success"
    SUBMITTED_FAILURE = "submitted_failure"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True, slots=True)
class ControlState:
    octaves: int = 0
    semitones: int = 0
    cents: int = 0

    def __post_init__(self) -> None:
        _bounded_integer(self.octaves, "octaves", OCTAVE_MIN, OCTAVE_MAX)
        _bounded_integer(self.semitones, "semitones", SEMITONE_MIN, SEMITONE_MAX)
        _bounded_integer(self.cents, "cents", CENT_MIN, CENT_MAX)

    @property
    def offset_cents(self) -> int:
        return 1_200 * self.octaves + 100 * self.semitones + self.cents

    def apply(self, action: PitchAction) -> tuple[ControlState, bool]:
        if not isinstance(action, PitchAction):
            raise ValueError("action must be a PitchAction")
        if action is PitchAction.SUBMIT:
            raise ValueError("SUBMIT is not a control mutation")

        if action is PitchAction.OCTAVE_DOWN:
            return self._adjust("octaves", -1, OCTAVE_MIN)
        if action is PitchAction.OCTAVE_UP:
            return self._adjust("octaves", 1, OCTAVE_MAX)
        if action is PitchAction.SEMITONE_DOWN:
            return self._adjust("semitones", -1, SEMITONE_MIN)
        if action is PitchAction.SEMITONE_UP:
            return self._adjust("semitones", 1, SEMITONE_MAX)
        if action is PitchAction.CENT_DOWN:
            return self._adjust("cents", -1, CENT_MIN)
        return self._adjust("cents", 1, CENT_MAX)

    def _adjust(self, name: str, delta: int, bound: int) -> tuple[ControlState, bool]:
        value = getattr(self, name)
        if value == bound:
            return self, False
        return replace(self, **{name: value + delta}), True


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    source_pitch_cents: int
    target_note_index: int
    target_pitch_cents: int
    final_pitch_cents: int
    initial_signed_error_cents: int
    initial_absolute_error_cents: int
    final_signed_error_cents: int
    final_absolute_error_cents: int
    submitted_success: bool
    within_5_cents: bool
    within_1_cent: bool
    actions: tuple[PitchAction, ...]
    invalid_action_count: int
    optimal_actions: tuple[PitchAction, ...]
    excess_actions: int | None
    total_return: float
    terminal_reason: TerminalReason

    def __post_init__(self) -> None:
        _bounded_integer(
            self.source_pitch_cents, "source_pitch_cents", SOURCE_MIN_CENTS, SOURCE_MAX_CENTS
        )
        _bounded_integer(self.target_note_index, "target_note_index", 0, TARGET_NOTE_COUNT - 1)
        _integer(self.target_pitch_cents, "target_pitch_cents")
        _integer(self.final_pitch_cents, "final_pitch_cents")
        _integer(self.initial_signed_error_cents, "initial_signed_error_cents")
        _bounded_integer(
            self.initial_absolute_error_cents,
            "initial_absolute_error_cents",
            0,
            MAX_ABSOLUTE_ERROR_CENTS,
        )
        _integer(self.final_signed_error_cents, "final_signed_error_cents")
        _bounded_integer(
            self.final_absolute_error_cents,
            "final_absolute_error_cents",
            0,
            MAX_ABSOLUTE_ERROR_CENTS,
        )
        _integer(self.invalid_action_count, "invalid_action_count", minimum=0)
        if self.excess_actions is not None:
            _integer(self.excess_actions, "excess_actions", minimum=0)
        if not isinstance(self.submitted_success, bool):
            raise ValueError("submitted_success must be a bool")
        if not isinstance(self.within_5_cents, bool):
            raise ValueError("within_5_cents must be a bool")
        if not isinstance(self.within_1_cent, bool):
            raise ValueError("within_1_cent must be a bool")
        if not isinstance(self.terminal_reason, TerminalReason):
            raise ValueError("terminal_reason must be a TerminalReason")

        actions = _action_tuple(self.actions, "actions")
        optimal_actions = _action_tuple(self.optimal_actions, "optimal_actions")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "optimal_actions", optimal_actions)
        _finite_float(self.total_return, "total_return")

        expected_target = 100 * (TARGET_MIN_COORDINATE + self.target_note_index)
        if self.target_pitch_cents != expected_target:
            raise ValueError("target_pitch_cents must match target_note_index")
        if self.initial_signed_error_cents != self.source_pitch_cents - self.target_pitch_cents:
            raise ValueError("initial_signed_error_cents must match source and target")
        if self.initial_absolute_error_cents != abs(self.initial_signed_error_cents):
            raise ValueError("initial_absolute_error_cents must equal absolute signed error")

        controls, invalid_count = _controls_after(actions)
        if self.final_pitch_cents != self.source_pitch_cents + controls.offset_cents:
            raise ValueError("final_pitch_cents must match the action trajectory")
        if self.final_signed_error_cents != self.final_pitch_cents - self.target_pitch_cents:
            raise ValueError("final_signed_error_cents must match final and target pitch")
        if self.final_absolute_error_cents != abs(self.final_signed_error_cents):
            raise ValueError("final_absolute_error_cents must equal absolute signed error")
        if self.invalid_action_count != invalid_count:
            raise ValueError("invalid_action_count must match blocked actions")
        if not optimal_actions or optimal_actions[-1] is not PitchAction.SUBMIT:
            raise ValueError("optimal_actions must end with SUBMIT")
        if PitchAction.SUBMIT in optimal_actions[:-1]:
            raise ValueError("optimal_actions may contain SUBMIT only at the end")
        from harpy.envs.planning import minimum_action_plan

        if optimal_actions != minimum_action_plan(self.initial_signed_error_cents):
            raise ValueError("optimal_actions must be the tolerance-optimal initial path")

        expected_within_5 = self.final_absolute_error_cents <= SUCCESS_TOLERANCE_CENTS
        expected_within_1 = self.final_absolute_error_cents <= STRICT_TOLERANCE_CENTS
        if self.within_5_cents != expected_within_5:
            raise ValueError("within_5_cents must match final error")
        if self.within_1_cent != expected_within_1:
            raise ValueError("within_1_cent must match final error")
        self._validate_terminal_state(actions)
        expected_total_return = _total_return_after(
            self.source_pitch_cents,
            self.target_pitch_cents,
            actions,
        )
        if not math.isclose(
            self.total_return,
            expected_total_return,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("total_return must match the action trajectory")

    @property
    def optimal_total_actions(self) -> int:
        return len(self.optimal_actions)

    def _validate_terminal_state(self, actions: tuple[PitchAction, ...]) -> None:
        if len(actions) > MAX_STEPS:
            raise ValueError("actions must not exceed MAX_STEPS")
        submitted = bool(actions) and actions[-1] is PitchAction.SUBMIT
        if self.terminal_reason is TerminalReason.SUBMITTED_SUCCESS:
            if not submitted or not self.submitted_success or not self.within_5_cents:
                raise ValueError("successful submissions must end with SUBMIT inside tolerance")
            if self.excess_actions != len(actions) - self.optimal_total_actions:
                raise ValueError("successful submissions must report excess_actions")
        elif self.terminal_reason is TerminalReason.SUBMITTED_FAILURE:
            if not submitted or self.submitted_success or self.within_5_cents:
                raise ValueError("failed submissions must end with SUBMIT outside tolerance")
            if self.excess_actions is not None:
                raise ValueError("failed submissions must not report excess_actions")
        else:
            if submitted or self.submitted_success or len(actions) != MAX_STEPS:
                raise ValueError("budget exhaustion requires MAX_STEPS non-submit actions")
            if self.excess_actions is not None:
                raise ValueError("budget exhaustion must not report excess_actions")


def _bounded_integer(value: object, name: str, minimum: int, maximum: int) -> None:
    _integer(value, name)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be within {minimum}..{maximum}")


def _integer(value: object, name: str, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _finite_float(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite float")


def _action_tuple(value: Iterable[PitchAction], name: str) -> tuple[PitchAction, ...]:
    try:
        actions = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an iterable of PitchAction values") from error
    if any(not isinstance(action, PitchAction) for action in actions):
        raise ValueError(f"{name} must contain only PitchAction values")
    return actions


def _controls_after(actions: tuple[PitchAction, ...]) -> tuple[ControlState, int]:
    controls = ZERO_CONTROLS
    invalid_count = 0
    for index, action in enumerate(actions):
        if action is PitchAction.SUBMIT:
            if index != len(actions) - 1:
                raise ValueError("actions may contain SUBMIT only at the end")
            continue
        controls, applied = controls.apply(action)
        if not applied:
            invalid_count += 1
    return controls, invalid_count


def _total_return_after(
    source_pitch_cents: int,
    target_pitch_cents: int,
    actions: tuple[PitchAction, ...],
) -> float:
    controls = ZERO_CONTROLS
    total_return = 0.0
    for step_count, action in enumerate(actions, start=1):
        before_pitch_cents = source_pitch_cents + controls.offset_cents
        before_absolute_error = abs(before_pitch_cents - target_pitch_cents)
        if action is PitchAction.SUBMIT:
            reward = 1.0 if before_absolute_error <= SUCCESS_TOLERANCE_CENTS else -1.0
        else:
            controls, applied = controls.apply(action)
            if applied:
                after_pitch_cents = source_pitch_cents + controls.offset_cents
                after_absolute_error = abs(after_pitch_cents - target_pitch_cents)
                reward = (
                    before_absolute_error - after_absolute_error
                ) / MAX_ABSOLUTE_ERROR_CENTS - 0.00001
            else:
                reward = -0.01
            if step_count == MAX_STEPS:
                reward -= 1.0
        total_return += reward
    return total_return


ZERO_CONTROLS = ControlState()
