import pytest

from harpy.envs.models import (
    CENT_MAX,
    CENT_MIN,
    OCTAVE_MAX,
    OCTAVE_MIN,
    SEMITONE_MAX,
    SEMITONE_MIN,
    ControlState,
    PitchAction,
)
from harpy.envs.planning import minimum_action_plan


def test_planner_uses_the_named_hierarchy_and_submit() -> None:
    assert minimum_action_plan(-1_305, tolerance_cents=0) == (
        PitchAction.OCTAVE_UP,
        PitchAction.SEMITONE_UP,
        PitchAction.CENT_UP,
        PitchAction.CENT_UP,
        PitchAction.CENT_UP,
        PitchAction.CENT_UP,
        PitchAction.CENT_UP,
        PitchAction.SUBMIT,
    )


def test_tolerance_can_stop_at_five_cents() -> None:
    plan = minimum_action_plan(-6, tolerance_cents=5)

    assert plan == (PitchAction.CENT_UP, PitchAction.SUBMIT)


def test_exact_plans_reach_zero_for_every_initial_error() -> None:
    for base_error_cents in range(-2_400, 2_401):
        plan = minimum_action_plan(base_error_cents, tolerance_cents=0)
        final_error = _final_error(base_error_cents, ControlState(), plan)

        assert plan[-1] is PitchAction.SUBMIT
        assert final_error == 0
        assert len(plan) - 1 <= 57


def test_default_tolerance_plans_reach_the_success_region_for_every_initial_error() -> None:
    for base_error_cents in range(-2_400, 2_401):
        plan = minimum_action_plan(base_error_cents)
        final_error = _final_error(base_error_cents, ControlState(), plan)

        assert plan[-1] is PitchAction.SUBMIT
        assert abs(final_error) <= 5
        assert len(plan) <= 53


@pytest.mark.parametrize(
    ("base_error_cents", "controls", "tolerance_cents"),
    [
        (-2_400, ControlState(octaves=2, semitones=12, cents=100), 0),
        (2_400, ControlState(octaves=-2, semitones=-12, cents=-100), 0),
        (-1_305, ControlState(octaves=-1, semitones=4, cents=90), 5),
        (1_306, ControlState(octaves=1, semitones=-7, cents=-99), 1),
    ],
)
def test_planner_accounts_for_nonzero_starting_controls(
    base_error_cents: int, controls: ControlState, tolerance_cents: int
) -> None:
    plan = minimum_action_plan(base_error_cents, controls, tolerance_cents=tolerance_cents)

    assert abs(_final_error(base_error_cents, controls, plan)) <= tolerance_cents
    assert plan[-1] is PitchAction.SUBMIT


def test_planner_matches_independent_global_ranking_for_every_initial_error() -> None:
    zero_controls = ControlState()
    zero_reference = _best_state_by_offset(zero_controls)

    for tolerance_cents in range(6):
        for base_error_cents in range(-2_400, 2_401):
            plan = minimum_action_plan(
                base_error_cents,
                zero_controls,
                tolerance_cents=tolerance_cents,
            )

            assert _plan_rank(base_error_cents, zero_controls, plan) == _reference_rank(
                base_error_cents,
                tolerance_cents,
                zero_reference,
            )

    nonzero_controls = ControlState(octaves=-1, semitones=4, cents=90)
    nonzero_reference = _best_state_by_offset(nonzero_controls)
    plan = minimum_action_plan(-1_305, nonzero_controls, tolerance_cents=5)

    assert _plan_rank(-1_305, nonzero_controls, plan) == _reference_rank(
        -1_305,
        5,
        nonzero_reference,
    )


@pytest.mark.parametrize("base_error_cents", [True, False, 0.0, "0", -2_401, 2_401])
def test_planner_rejects_invalid_base_errors(base_error_cents: object) -> None:
    with pytest.raises(ValueError, match="base_error_cents"):
        minimum_action_plan(base_error_cents)  # type: ignore[arg-type]


@pytest.mark.parametrize("tolerance_cents", [True, False, 0.0, "5", -1, 6])
def test_planner_rejects_invalid_tolerances(tolerance_cents: object) -> None:
    with pytest.raises(ValueError, match="tolerance_cents"):
        minimum_action_plan(0, tolerance_cents=tolerance_cents)  # type: ignore[arg-type]


def _final_error(
    base_error_cents: int, controls: ControlState, plan: tuple[PitchAction, ...]
) -> int:
    final_controls = controls
    for action in plan[:-1]:
        final_controls, applied = final_controls.apply(action)
        assert applied
    return base_error_cents + final_controls.offset_cents


def _best_state_by_offset(
    controls: ControlState,
) -> dict[int, tuple[int, int, int, int]]:
    best_by_offset: dict[int, tuple[int, int, int, int]] = {}
    for octaves in range(OCTAVE_MIN, OCTAVE_MAX + 1):
        for semitones in range(SEMITONE_MIN, SEMITONE_MAX + 1):
            for cents in range(CENT_MIN, CENT_MAX + 1):
                offset = 1_200 * octaves + 100 * semitones + cents
                rank = (
                    abs(octaves - controls.octaves)
                    + abs(semitones - controls.semitones)
                    + abs(cents - controls.cents),
                    octaves,
                    semitones,
                    cents,
                )
                if offset not in best_by_offset or rank < best_by_offset[offset]:
                    best_by_offset[offset] = rank
    return best_by_offset


def _reference_rank(
    base_error_cents: int,
    tolerance_cents: int,
    best_by_offset: dict[int, tuple[int, int, int, int]],
) -> tuple[int, int, int, int, int]:
    candidates = []
    for final_error in range(-tolerance_cents, tolerance_cents + 1):
        state_rank = best_by_offset.get(final_error - base_error_cents)
        if state_rank is not None:
            action_count, octaves, semitones, cents = state_rank
            candidates.append((action_count, abs(final_error), octaves, semitones, cents))
    return min(candidates)


def _plan_rank(
    base_error_cents: int,
    controls: ControlState,
    plan: tuple[PitchAction, ...],
) -> tuple[int, int, int, int, int]:
    final_controls = controls
    for action in plan[:-1]:
        final_controls, applied = final_controls.apply(action)
        assert applied
    return (
        len(plan) - 1,
        abs(base_error_cents + final_controls.offset_cents),
        final_controls.octaves,
        final_controls.semitones,
        final_controls.cents,
    )
