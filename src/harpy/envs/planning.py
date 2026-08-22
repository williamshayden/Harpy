"""Pure shortest-path planning over bounded musical controls."""

from __future__ import annotations

from harpy.envs.models import (
    CENT_MAX,
    CENT_MIN,
    OCTAVE_MAX,
    OCTAVE_MIN,
    SEMITONE_MAX,
    SEMITONE_MIN,
    SOURCE_MAX_CENTS,
    SOURCE_MIN_CENTS,
    SUCCESS_TOLERANCE_CENTS,
    TARGET_MIN_COORDINATE,
    TARGET_NOTE_COUNT,
    ZERO_CONTROLS,
    ControlState,
    PitchAction,
)


def minimum_action_plan(
    base_error_cents: int,
    controls: ControlState = ZERO_CONTROLS,
    *,
    tolerance_cents: int = SUCCESS_TOLERANCE_CENTS,
) -> tuple[PitchAction, ...]:
    """Return the shortest bounded musical path that ends with ``SUBMIT``."""

    _validate_base_error(base_error_cents)
    if not isinstance(controls, ControlState):
        raise ValueError("controls must be a ControlState")
    if (
        isinstance(tolerance_cents, bool)
        or not isinstance(tolerance_cents, int)
        or not 0 <= tolerance_cents <= SUCCESS_TOLERANCE_CENTS
    ):
        raise ValueError("tolerance_cents must be an integer within 0..5")

    target = _best_target(base_error_cents, controls, tolerance_cents)
    return (*_actions_to_reach(controls, target), PitchAction.SUBMIT)


def _validate_base_error(base_error_cents: object) -> None:
    max_initial_error = SOURCE_MAX_CENTS - TARGET_MIN_COORDINATE * 100
    min_initial_error = SOURCE_MIN_CENTS - (TARGET_MIN_COORDINATE + TARGET_NOTE_COUNT - 1) * 100
    if (
        isinstance(base_error_cents, bool)
        or not isinstance(base_error_cents, int)
        or not min_initial_error <= base_error_cents <= max_initial_error
    ):
        raise ValueError("base_error_cents must be an integer within -2400..2400")


def _best_target(
    base_error_cents: int, controls: ControlState, tolerance_cents: int
) -> ControlState:
    best_rank: tuple[int, int, int, int, int] | None = None
    best_target: ControlState | None = None
    for octaves in range(OCTAVE_MIN, OCTAVE_MAX + 1):
        for semitones in range(SEMITONE_MIN, SEMITONE_MAX + 1):
            fixed_error = base_error_cents + 1_200 * octaves + 100 * semitones
            cents_min = max(CENT_MIN, -tolerance_cents - fixed_error)
            cents_max = min(CENT_MAX, tolerance_cents - fixed_error)
            if cents_min > cents_max:
                continue
            cents = min(max(controls.cents, cents_min), cents_max)
            target = ControlState(octaves=octaves, semitones=semitones, cents=cents)
            final_error = base_error_cents + target.offset_cents
            rank = (
                abs(target.octaves - controls.octaves)
                + abs(target.semitones - controls.semitones)
                + abs(target.cents - controls.cents),
                abs(final_error),
                target.octaves,
                target.semitones,
                target.cents,
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_target = target
    if best_target is None:
        raise RuntimeError("no bounded control state satisfies the requested tolerance")
    return best_target


def _actions_to_reach(controls: ControlState, target: ControlState) -> tuple[PitchAction, ...]:
    return (
        *_actions_for_delta(
            controls.octaves, target.octaves, PitchAction.OCTAVE_DOWN, PitchAction.OCTAVE_UP
        ),
        *_actions_for_delta(
            controls.semitones,
            target.semitones,
            PitchAction.SEMITONE_DOWN,
            PitchAction.SEMITONE_UP,
        ),
        *_actions_for_delta(
            controls.cents, target.cents, PitchAction.CENT_DOWN, PitchAction.CENT_UP
        ),
    )


def _actions_for_delta(
    current: int, target: int, down: PitchAction, up: PitchAction
) -> tuple[PitchAction, ...]:
    if target >= current:
        return (up,) * (target - current)
    return (down,) * (current - target)
