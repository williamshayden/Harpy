from dataclasses import FrozenInstanceError, replace
from math import inf, nan

import pytest

from harpy.envs.models import (
    CENT_MAX,
    CENT_MIN,
    MAX_STEPS,
    OCTAVE_MAX,
    OCTAVE_MIN,
    SEMITONE_MAX,
    SEMITONE_MIN,
    ControlState,
    EpisodeResult,
    ObservationMode,
    PitchAction,
    TerminalReason,
)


def test_pitch_actions_have_stable_symmetric_ids() -> None:
    assert [(action.name, action.value) for action in PitchAction] == [
        ("OCTAVE_DOWN", 0),
        ("SEMITONE_DOWN", 1),
        ("CENT_DOWN", 2),
        ("SUBMIT", 3),
        ("CENT_UP", 4),
        ("SEMITONE_UP", 5),
        ("OCTAVE_UP", 6),
    ]


def test_controls_apply_one_named_unit_without_carry() -> None:
    state = ControlState(octaves=1, semitones=12, cents=100)

    next_state, applied = state.apply(PitchAction.OCTAVE_UP)

    assert applied is True
    assert next_state == ControlState(octaves=2, semitones=12, cents=100)
    assert next_state.offset_cents == 3_700


def test_bound_attempt_is_the_same_immutable_state() -> None:
    state = ControlState(octaves=2, semitones=12, cents=100)

    next_state, applied = state.apply(PitchAction.CENT_UP)

    assert applied is False
    assert next_state is state


@pytest.mark.parametrize(
    ("action", "state"),
    [
        (PitchAction.OCTAVE_DOWN, ControlState(octaves=OCTAVE_MIN)),
        (PitchAction.OCTAVE_UP, ControlState(octaves=OCTAVE_MAX)),
        (PitchAction.SEMITONE_DOWN, ControlState(semitones=SEMITONE_MIN)),
        (PitchAction.SEMITONE_UP, ControlState(semitones=SEMITONE_MAX)),
        (PitchAction.CENT_DOWN, ControlState(cents=CENT_MIN)),
        (PitchAction.CENT_UP, ControlState(cents=CENT_MAX)),
    ],
)
def test_each_named_control_has_a_blocked_lower_and_upper_bound(
    action: PitchAction, state: ControlState
) -> None:
    next_state, applied = state.apply(action)

    assert applied is False
    assert next_state is state


@pytest.mark.parametrize(
    ("action", "state", "expected"),
    [
        (PitchAction.OCTAVE_DOWN, ControlState(octaves=OCTAVE_MIN + 1), ControlState(octaves=-2)),
        (PitchAction.OCTAVE_UP, ControlState(octaves=OCTAVE_MAX - 1), ControlState(octaves=2)),
        (
            PitchAction.SEMITONE_DOWN,
            ControlState(semitones=SEMITONE_MIN + 1),
            ControlState(semitones=-12),
        ),
        (
            PitchAction.SEMITONE_UP,
            ControlState(semitones=SEMITONE_MAX - 1),
            ControlState(semitones=12),
        ),
        (PitchAction.CENT_DOWN, ControlState(cents=CENT_MIN + 1), ControlState(cents=-100)),
        (PitchAction.CENT_UP, ControlState(cents=CENT_MAX - 1), ControlState(cents=100)),
    ],
)
def test_each_named_control_can_reach_its_boundary(
    action: PitchAction, state: ControlState, expected: ControlState
) -> None:
    assert state.apply(action) == (expected, True)


@pytest.mark.parametrize("field", ["octaves", "semitones", "cents"])
@pytest.mark.parametrize("value", [True, False, 1.0, "1"])
def test_control_values_must_be_exact_integers(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        ControlState(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("octaves", OCTAVE_MIN - 1),
        ("octaves", OCTAVE_MAX + 1),
        ("semitones", SEMITONE_MIN - 1),
        ("semitones", SEMITONE_MAX + 1),
        ("cents", CENT_MIN - 1),
        ("cents", CENT_MAX + 1),
    ],
)
def test_control_values_must_stay_in_their_named_bounds(field: str, value: int) -> None:
    with pytest.raises(ValueError, match=field):
        ControlState(**{field: value})


@pytest.mark.parametrize("action", [True, False, 0, "CENT_UP"])
def test_apply_rejects_non_enum_actions(action: object) -> None:
    with pytest.raises(ValueError, match="action"):
        ControlState().apply(action)  # type: ignore[arg-type]


def test_submit_is_not_a_control_mutation() -> None:
    with pytest.raises(ValueError, match="SUBMIT"):
        ControlState().apply(PitchAction.SUBMIT)


def test_control_state_is_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        ControlState().cents = 1  # type: ignore[misc]


def test_action_labels_are_actor_facing_musical_names() -> None:
    assert [action.label for action in PitchAction] == [
        "Octave Down",
        "Semitone Down",
        "Cent Down",
        "Submit",
        "Cent Up",
        "Semitone Up",
        "Octave Up",
    ]


def test_observation_modes_and_terminal_reasons_have_stable_values() -> None:
    assert [(mode.name, mode.value) for mode in ObservationMode] == [
        ("SPECTRUM", "spectrum"),
        ("WAVEFORM", "waveform"),
        ("ORACLE", "oracle"),
        ("REWARD_ONLY", "reward_only"),
    ]
    assert [(reason.name, reason.value) for reason in TerminalReason] == [
        ("SUBMITTED_SUCCESS", "submitted_success"),
        ("SUBMITTED_FAILURE", "submitted_failure"),
        ("BUDGET_EXHAUSTED", "budget_exhausted"),
    ]


def _successful_result(**changes: object) -> EpisodeResult:
    values: dict[str, object] = {
        "source_pitch_cents": 6_000,
        "target_note_index": 12,
        "target_pitch_cents": 6_000,
        "final_pitch_cents": 6_000,
        "initial_signed_error_cents": 0,
        "initial_absolute_error_cents": 0,
        "final_signed_error_cents": 0,
        "final_absolute_error_cents": 0,
        "submitted_success": True,
        "within_5_cents": True,
        "within_1_cent": True,
        "actions": (PitchAction.SUBMIT,),
        "invalid_action_count": 0,
        "optimal_actions": (PitchAction.SUBMIT,),
        "excess_actions": 0,
        "total_return": 1.0,
        "terminal_reason": TerminalReason.SUBMITTED_SUCCESS,
    }
    values.update(changes)
    return EpisodeResult(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "case",
    ["immediate_success", "applied_progress", "blocked_then_failed", "timeout"],
)
def test_episode_result_rejects_trajectory_inconsistent_finite_return(case: str) -> None:
    result = _return_replay_result(case)

    with pytest.raises(ValueError, match="total_return"):
        replace(result, total_return=result.total_return + 0.25)


def _return_replay_result(case: str) -> EpisodeResult:
    if case == "immediate_success":
        return _successful_result()
    if case == "applied_progress":
        move_reward = 1 / 6_100 - 0.00001
        return EpisodeResult(
            source_pitch_cents=5_994,
            target_note_index=12,
            target_pitch_cents=6_000,
            final_pitch_cents=5_995,
            initial_signed_error_cents=-6,
            initial_absolute_error_cents=6,
            final_signed_error_cents=-5,
            final_absolute_error_cents=5,
            submitted_success=True,
            within_5_cents=True,
            within_1_cent=False,
            actions=(PitchAction.CENT_UP, PitchAction.SUBMIT),
            invalid_action_count=0,
            optimal_actions=(PitchAction.CENT_UP, PitchAction.SUBMIT),
            excess_actions=0,
            total_return=move_reward + 1.0,
            terminal_reason=TerminalReason.SUBMITTED_SUCCESS,
        )
    if case == "blocked_then_failed":
        away_reward = -1_200 / 6_100 - 0.00001
        total_return = 0.0
        for reward in (away_reward, away_reward, -0.01, -1.0):
            total_return += reward
        return EpisodeResult(
            source_pitch_cents=6_006,
            target_note_index=12,
            target_pitch_cents=6_000,
            final_pitch_cents=8_406,
            initial_signed_error_cents=6,
            initial_absolute_error_cents=6,
            final_signed_error_cents=2_406,
            final_absolute_error_cents=2_406,
            submitted_success=False,
            within_5_cents=False,
            within_1_cent=False,
            actions=(
                PitchAction.OCTAVE_UP,
                PitchAction.OCTAVE_UP,
                PitchAction.OCTAVE_UP,
                PitchAction.SUBMIT,
            ),
            invalid_action_count=1,
            optimal_actions=(PitchAction.CENT_DOWN, PitchAction.SUBMIT),
            excess_actions=None,
            total_return=total_return,
            terminal_reason=TerminalReason.SUBMITTED_FAILURE,
        )

    away_reward = -1_200 / 6_100 - 0.00001
    timeout_reward = 1_200 / 6_100 - 0.00001 - 1.0
    total_return = 0.0
    for reward in (away_reward, away_reward, *([-0.01] * 61), timeout_reward):
        total_return += reward
    return EpisodeResult(
        source_pitch_cents=6_000,
        target_note_index=12,
        target_pitch_cents=6_000,
        final_pitch_cents=7_200,
        initial_signed_error_cents=0,
        initial_absolute_error_cents=0,
        final_signed_error_cents=1_200,
        final_absolute_error_cents=1_200,
        submitted_success=False,
        within_5_cents=False,
        within_1_cent=False,
        actions=(PitchAction.OCTAVE_UP,) * 63 + (PitchAction.OCTAVE_DOWN,),
        invalid_action_count=61,
        optimal_actions=(PitchAction.SUBMIT,),
        excess_actions=None,
        total_return=total_return,
        terminal_reason=TerminalReason.BUDGET_EXHAUSTED,
    )


def test_episode_result_owns_immutable_action_tuples_and_derives_optimal_count() -> None:
    result = _successful_result(actions=[PitchAction.SUBMIT], optimal_actions=[PitchAction.SUBMIT])

    assert result.actions == (PitchAction.SUBMIT,)
    assert result.optimal_actions == (PitchAction.SUBMIT,)
    assert result.optimal_total_actions == 1


def test_episode_result_rejects_a_nonoptimal_initial_path() -> None:
    with pytest.raises(ValueError, match="optimal_actions"):
        _successful_result(
            actions=(PitchAction.CENT_UP, PitchAction.CENT_DOWN, PitchAction.SUBMIT),
            optimal_actions=(PitchAction.CENT_UP, PitchAction.SUBMIT),
            excess_actions=1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_pitch_cents", True),
        ("target_note_index", False),
        ("target_pitch_cents", 6_000.0),
        ("final_pitch_cents", True),
        ("initial_signed_error_cents", 0.0),
        ("initial_absolute_error_cents", False),
        ("final_signed_error_cents", True),
        ("final_absolute_error_cents", 0.0),
        ("invalid_action_count", True),
        ("excess_actions", False),
    ],
)
def test_episode_result_integer_scalars_reject_booleans_and_nonintegers(
    field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=field):
        _successful_result(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_pitch_cents", 4_799),
        ("source_pitch_cents", 7_201),
        ("target_note_index", -1),
        ("target_note_index", 25),
        ("target_pitch_cents", 6_100),
        ("initial_signed_error_cents", 1),
        ("initial_absolute_error_cents", 1),
        ("final_pitch_cents", 6_001),
        ("final_signed_error_cents", 1),
        ("final_absolute_error_cents", 1),
        ("within_5_cents", False),
        ("within_1_cent", False),
        ("invalid_action_count", 1),
        ("excess_actions", 1),
        ("total_return", nan),
        ("total_return", inf),
        ("terminal_reason", TerminalReason.SUBMITTED_FAILURE),
    ],
)
def test_episode_result_rejects_inconsistent_evaluator_truth(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _successful_result(**{field: value})


def test_episode_result_requires_a_complete_failed_submission() -> None:
    result = EpisodeResult(
        source_pitch_cents=6_000,
        target_note_index=12,
        target_pitch_cents=6_000,
        final_pitch_cents=7_202,
        initial_signed_error_cents=0,
        initial_absolute_error_cents=0,
        final_signed_error_cents=1_202,
        final_absolute_error_cents=1_202,
        submitted_success=False,
        within_5_cents=False,
        within_1_cent=False,
        actions=(PitchAction.CENT_UP,) * 2 + (PitchAction.OCTAVE_UP, PitchAction.SUBMIT),
        invalid_action_count=0,
        optimal_actions=(PitchAction.SUBMIT,),
        excess_actions=None,
        total_return=-1.0 - 1_202 / 6_100 - 3 * 0.00001,
        terminal_reason=TerminalReason.SUBMITTED_FAILURE,
    )

    assert result.terminal_reason is TerminalReason.SUBMITTED_FAILURE


def test_episode_result_limits_successful_submissions_to_the_action_budget() -> None:
    boundary_actions = (PitchAction.CENT_UP, PitchAction.CENT_DOWN) * 31 + (
        PitchAction.CENT_UP,
        PitchAction.SUBMIT,
    )
    result = _successful_result(
        final_pitch_cents=6_001,
        final_signed_error_cents=1,
        final_absolute_error_cents=1,
        actions=boundary_actions,
        excess_actions=63,
        total_return=1.0 - 1 / 6_100 - 63 * 0.00001,
    )

    assert len(result.actions) == MAX_STEPS

    over_budget_actions = (PitchAction.CENT_UP, PitchAction.CENT_DOWN) * 32 + (PitchAction.SUBMIT,)
    with pytest.raises(ValueError, match="MAX_STEPS"):
        _successful_result(actions=over_budget_actions, excess_actions=64)


def test_episode_result_limits_failed_submissions_to_the_action_budget() -> None:
    boundary_actions = (PitchAction.CENT_UP,) * 63 + (PitchAction.SUBMIT,)
    result = EpisodeResult(
        source_pitch_cents=6_000,
        target_note_index=12,
        target_pitch_cents=6_000,
        final_pitch_cents=6_063,
        initial_signed_error_cents=0,
        initial_absolute_error_cents=0,
        final_signed_error_cents=63,
        final_absolute_error_cents=63,
        submitted_success=False,
        within_5_cents=False,
        within_1_cent=False,
        actions=boundary_actions,
        invalid_action_count=0,
        optimal_actions=(PitchAction.SUBMIT,),
        excess_actions=None,
        total_return=-1.0 - 63 / 6_100 - 63 * 0.00001,
        terminal_reason=TerminalReason.SUBMITTED_FAILURE,
    )

    assert len(result.actions) == MAX_STEPS

    over_budget_actions = (PitchAction.CENT_UP,) * 64 + (PitchAction.SUBMIT,)
    with pytest.raises(ValueError, match="MAX_STEPS"):
        EpisodeResult(
            source_pitch_cents=6_000,
            target_note_index=12,
            target_pitch_cents=6_000,
            final_pitch_cents=6_064,
            initial_signed_error_cents=0,
            initial_absolute_error_cents=0,
            final_signed_error_cents=64,
            final_absolute_error_cents=64,
            submitted_success=False,
            within_5_cents=False,
            within_1_cent=False,
            actions=over_budget_actions,
            invalid_action_count=0,
            optimal_actions=(PitchAction.SUBMIT,),
            excess_actions=None,
            total_return=-1.0,
            terminal_reason=TerminalReason.SUBMITTED_FAILURE,
        )


def test_episode_result_rejects_success_without_submit() -> None:
    with pytest.raises(ValueError):
        _successful_result(
            final_pitch_cents=6_001,
            final_signed_error_cents=1,
            final_absolute_error_cents=1,
            actions=(PitchAction.CENT_UP,),
        )
