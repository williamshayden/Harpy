from __future__ import annotations

import math

import pytest

from harpy.envs.models import PitchAction, TerminalReason
from harpy.learning.diagnostics import (
    DiagnosticDecisionInput,
    build_diagnostic_report,
    diagnose_episode,
)

_DIGEST_A = "a" * 64


def _decision(
    action: PitchAction, estimated_candidate_cents: int | None = None
) -> DiagnosticDecisionInput:
    return DiagnosticDecisionInput(
        action=action,
        estimated_candidate_cents=estimated_candidate_cents,
    )


def _commuting_episode():
    # True error is -1,300 cents. The canonical planner starts with Octave Up,
    # but Semitone Up commutes and remains on a shortest successful path.
    return diagnose_episode(
        episode_index=0,
        source_pitch_cents=4_900,
        target_note_index=14,
        decisions=(
            _decision(PitchAction.SEMITONE_UP, 4_895),
            _decision(PitchAction.OCTAVE_UP, 4_995),
            _decision(PitchAction.SUBMIT, 6_195),
        ),
    )


def _looping_episode():
    return diagnose_episode(
        episode_index=1,
        source_pitch_cents=4_800,
        target_note_index=0,
        decisions=tuple(
            _decision(action)
            for action in (
                PitchAction.OCTAVE_DOWN,
                PitchAction.OCTAVE_DOWN,
                PitchAction.OCTAVE_DOWN,  # blocked at the lower bound
                PitchAction.OCTAVE_UP,  # returns to a previously visited state
                PitchAction.SUBMIT,
            )
        ),
    )


def test_commuting_action_is_canonical_mismatch_without_shortest_divergence() -> None:
    episode = _commuting_episode()

    assert tuple(decision.step for decision in episode.decisions) == (1, 2, 3)
    first = episode.decisions[0]
    assert first.teacher_action is PitchAction.OCTAVE_UP
    assert first.predicted_action is PitchAction.SEMITONE_UP
    assert first.canonical_match is False
    assert first.in_shortest_action_set is True
    assert episode.first_canonical_mismatch_step == 1
    assert episode.first_canonical_mismatch_action is PitchAction.SEMITONE_UP
    assert episode.first_shortest_divergence_step is None
    assert episode.first_shortest_divergence_action is None
    assert episode.terminal_reason is TerminalReason.SUBMITTED_SUCCESS

    estimate = first.estimate
    assert estimate is not None
    assert estimate.estimated_candidate_cents == 4_895
    assert estimate.signed_error_cents == -5
    assert estimate.absolute_error_cents == 5
    assert _looping_episode().decisions[0].estimate is None


def test_bound_blocks_repeated_visits_and_loop_transitions_are_distinct() -> None:
    episode = _looping_episode()

    assert episode.legal_action_count == 4
    assert episode.bound_blocked_action_count == 1
    assert episode.repeated_state_visit_count == 2
    assert episode.loop_transition_count == 2
    assert episode.first_loop_step == 3
    assert episode.first_shortest_divergence_step == 1
    assert episode.first_shortest_divergence_action is PitchAction.OCTAVE_DOWN
    assert episode.decisions[2].bound_blocked is True
    assert episode.decisions[2].loop_transition is True
    assert episode.decisions[3].repeated_state_visit is True
    assert episode.final_absolute_error_cents == 1_200
    assert episode.terminal_reason is TerminalReason.SUBMITTED_FAILURE


def test_recoverability_includes_submit_and_uses_public_remaining_budget() -> None:
    # Error 1,850 has a long cents component. After two away octaves and repeated
    # semitone loops, the tolerance-five plan no longer fits the public budget.
    episode = diagnose_episode(
        episode_index=0,
        source_pitch_cents=6_650,
        target_note_index=0,
        decisions=tuple(
            _decision(action)
            for action in (
                PitchAction.OCTAVE_UP,
                PitchAction.OCTAVE_UP,
                PitchAction.SEMITONE_UP,
                PitchAction.SEMITONE_DOWN,
                PitchAction.SEMITONE_UP,
                PitchAction.SEMITONE_DOWN,
                PitchAction.SEMITONE_UP,
                PitchAction.SEMITONE_DOWN,
                PitchAction.SEMITONE_UP,
                PitchAction.SEMITONE_DOWN,
                PitchAction.SUBMIT,
            )
        ),
    )

    assert episode.decisions[9].shortest_plan_length == 54
    assert episode.decisions[9].steps_remaining == 55
    assert episode.decisions[9].recoverable is True
    assert episode.decisions[10].shortest_plan_length == 55
    assert episode.decisions[10].steps_remaining == 54
    assert episode.decisions[10].recoverable is False
    assert episode.first_unrecoverable_step == 11


def test_null_events_and_late_submit_or_budget_terminal_rules() -> None:
    exact_submit = diagnose_episode(
        episode_index=0,
        source_pitch_cents=4_800,
        target_note_index=0,
        decisions=(_decision(PitchAction.SUBMIT),),
    )
    assert exact_submit.first_canonical_mismatch_step is None
    assert exact_submit.first_canonical_mismatch_action is None
    assert exact_submit.first_shortest_divergence_step is None
    assert exact_submit.first_shortest_divergence_action is None
    assert exact_submit.first_loop_step is None
    assert exact_submit.first_unrecoverable_step is None

    late_actions = (
        *(
            _decision(PitchAction.CENT_UP if index % 2 == 0 else PitchAction.CENT_DOWN)
            for index in range(63)
        ),
        _decision(PitchAction.SUBMIT),
    )
    late_submit = diagnose_episode(
        episode_index=0,
        source_pitch_cents=4_800,
        target_note_index=0,
        decisions=late_actions,
    )
    assert late_submit.action_count == 64
    assert late_submit.terminal_reason is TerminalReason.SUBMITTED_SUCCESS
    assert late_submit.excess_actions == 63

    exhausted = diagnose_episode(
        episode_index=0,
        source_pitch_cents=4_800,
        target_note_index=0,
        decisions=tuple(_decision(PitchAction.CENT_UP) for _ in range(64)),
    )
    assert exhausted.action_count == 64
    assert exhausted.terminal_reason is TerminalReason.BUDGET_EXHAUSTED
    assert exhausted.final_absolute_error_cents == 64
    assert exhausted.excess_actions is None


def test_submit_error_bands_cover_all_fixed_boundaries() -> None:
    errors = (1, 2, 6, 26, 100)
    episodes = tuple(
        diagnose_episode(
            episode_index=index,
            source_pitch_cents=4_800 + error,
            target_note_index=0,
            decisions=(_decision(PitchAction.SUBMIT),),
        )
        for index, error in enumerate(errors)
    )
    report = build_diagnostic_report(
        seed=0,
        artifact_manifest_sha256=_DIGEST_A,
        actor_semantics="legacy-v1",
        bound_mask=False,
        suite_id="bands-v1",
        suite_digest_sha256=_DIGEST_A,
        episodes=episodes,
    )

    assert tuple((band.label, band.count) for band in report.submit_error_bands) == (
        ("0..1", 1),
        ("2..5", 1),
        ("6..25", 1),
        ("26..99", 1),
        ("100+", 1),
    )


def test_report_pins_confusion_metrics_shortest_accuracy_and_survival() -> None:
    report = build_diagnostic_report(
        seed=0,
        artifact_manifest_sha256=_DIGEST_A,
        actor_semantics="pitch-planner-v1",
        bound_mask=False,
        suite_id="crafted-v1",
        suite_digest_sha256=_DIGEST_A,
        episodes=(_commuting_episode(), _looping_episode()),
    )

    matrix = report.confusion_matrix
    assert len(matrix) == 7 and all(len(row) == 7 for row in matrix)
    assert matrix[PitchAction.SUBMIT][PitchAction.OCTAVE_DOWN] == 1
    assert matrix[PitchAction.SUBMIT][PitchAction.SUBMIT] == 1
    assert matrix[PitchAction.OCTAVE_UP] == (2, 0, 0, 1, 0, 1, 2)

    octave_up = report.per_action_metrics[PitchAction.OCTAVE_UP]
    assert octave_up.precision_numerator == 2
    assert octave_up.precision_denominator == 2
    assert octave_up.precision == 1.0
    assert octave_up.recall_numerator == 2
    assert octave_up.recall_denominator == 6
    assert octave_up.recall == pytest.approx(1 / 3)
    assert octave_up.support == 6

    semitone_down = report.per_action_metrics[PitchAction.SEMITONE_DOWN]
    assert semitone_down.precision is None
    assert semitone_down.recall is None
    assert semitone_down.support == 0
    semitone_up = report.per_action_metrics[PitchAction.SEMITONE_UP]
    assert semitone_up.precision == 0.0
    assert semitone_up.recall is None

    assert report.shortest_action_correct == 4
    assert report.shortest_action_decisions == 8
    assert report.shortest_action_accuracy == 0.5
    assert tuple(
        (entry.step, entry.at_risk, entry.survivors, entry.rate) for entry in report.survival_curve
    ) == (
        (1, 2, 1, 0.5),
        (2, 2, 1, 0.5),
        (3, 2, 1, 0.5),
        (4, 1, 0, 0.0),
        (5, 1, 0, 0.0),
    )
    assert tuple((band.label, band.count) for band in report.submit_error_bands) == (
        ("0..1", 1),
        ("2..5", 0),
        ("6..25", 0),
        ("26..99", 0),
        ("100+", 1),
    )
    assert all(math.isfinite(entry.rate) for entry in report.survival_curve)


@pytest.mark.parametrize(
    "decisions, message",
    [
        ((), "decision"),
        ((_decision(PitchAction.CENT_UP),), "terminate"),
        (
            (_decision(PitchAction.SUBMIT), _decision(PitchAction.CENT_UP)),
            "Submit",
        ),
    ],
)
def test_episode_input_rejects_nonterminal_or_post_submit_sequences(
    decisions: tuple[DiagnosticDecisionInput, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        diagnose_episode(
            episode_index=0,
            source_pitch_cents=4_800,
            target_note_index=0,
            decisions=decisions,
        )
