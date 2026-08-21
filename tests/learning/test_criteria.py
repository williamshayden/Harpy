"""Strict evaluation-file and scientific-criterion contracts."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from harpy.envs.baselines import BaselineKind
from harpy.envs.models import ObservationMode, TerminalReason
from harpy.learning.evaluation import (
    SHUFFLED_SPECTRUM_PROBE,
    ZERO_SPECTRUM_PROBE,
    EvaluationFile,
    EvaluationRow,
    TerminalEpisodeRecord,
    build_evaluation_rows,
    evaluate_bc_criterion,
    evaluate_ppo_criterion,
)
from harpy.learning.models import EpisodeSpec, EpisodeSuite, EvaluationSuiteId, TrainerKind
from harpy.learning.suites import fixed_evaluation_suite, suite_digest


def _criterion_suite() -> EpisodeSuite:
    episodes = (
        EpisodeSpec(target_note_index=12, source_pitch_cents=6_100),
        EpisodeSpec(target_note_index=12, source_pitch_cents=5_900),
    )
    return EpisodeSuite(
        schema_version=1,
        suite_id=EvaluationSuiteId.IID,
        suite_seed=202_608_101,
        episodes=episodes,
        digest_sha256=suite_digest(
            suite_id=EvaluationSuiteId.IID,
            suite_seed=202_608_101,
            episodes=episodes,
        ),
    )


def _records(suite: EpisodeSuite, submitted_success_rate: float):
    success_count = round(submitted_success_rate * len(suite.episodes))
    assert success_count / len(suite.episodes) == submitted_success_rate
    return tuple(
        TerminalEpisodeRecord(
            episode_index=index,
            episode=episode,
            submitted_success=index < success_count,
            within_5_cents=index < success_count,
            within_1_cent=index < success_count,
            final_absolute_error_cents=1 if index < success_count else 100,
            action_count=1,
            excess_actions=0 if index < success_count else None,
            invalid_action_count=0,
            total_return=1.0 if index < success_count else -1.0,
            terminal_reason=(
                TerminalReason.SUBMITTED_SUCCESS
                if index < success_count
                else TerminalReason.SUBMITTED_FAILURE
            ),
        )
        for index, episode in enumerate(suite.episodes)
    )


def _row(
    *,
    rate: float,
    trainer: TrainerKind | None = TrainerKind.BC,
    seed: int | None = 0,
    actor_id: str = "bc-0",
    probe: str | None = None,
    suite: EpisodeSuite | None = None,
) -> EvaluationRow:
    active_suite = _criterion_suite() if suite is None else suite
    return build_evaluation_rows(
        actor_id=actor_id,
        trainer=trainer,
        seed=seed,
        environment_id=(
            BaselineKind.RANDOM.environment_id if trainer is None else "Harpy/SinePitch-v0"
        ),
        observation_mode=ObservationMode.SPECTRUM,
        suite=active_suite,
        records=_records(active_suite, rate),
        probe=probe,
    )[0]


def _ppo_row(seed: int, rate: float) -> EvaluationRow:
    return _row(rate=rate, trainer=TrainerKind.PPO, seed=seed, actor_id=f"ppo-{seed}")


def test_bc_gate_requires_both_declared_thresholds() -> None:
    suite = _criterion_suite()
    expanded_episodes = tuple(
        EpisodeSpec(target_note_index=index, source_pitch_cents=6_000 + index * 10)
        for index in range(4)
    )
    expanded = EpisodeSuite(
        schema_version=1,
        suite_id=suite.suite_id,
        suite_seed=suite.suite_seed,
        episodes=expanded_episodes,
        digest_sha256=suite_digest(
            suite_id=suite.suite_id,
            suite_seed=suite.suite_seed,
            episodes=expanded_episodes,
        ),
    )
    passing = _row(rate=0.75, suite=expanded)

    assert (
        evaluate_bc_criterion(
            heldout_next_action_accuracy=0.90,
            iid_row=passing,
            eligible=True,
        ).criterion_met
        is True
    )
    assert (
        evaluate_bc_criterion(
            heldout_next_action_accuracy=0.8999,
            iid_row=passing,
            eligible=True,
        ).criterion_met
        is False
    )
    assert (
        evaluate_bc_criterion(
            heldout_next_action_accuracy=0.90,
            iid_row=_row(rate=0.5),
            eligible=True,
        ).criterion_met
        is False
    )


def test_ppo_gate_uses_exact_five_seed_median_and_strict_random_beats() -> None:
    rates = [0.5, 0.5, 0.5, 1.0, 1.0]
    result = evaluate_ppo_criterion(
        iid_rows=tuple(_ppo_row(seed=index, rate=rate) for index, rate in enumerate(rates)),
        random_iid_row=_row(
            rate=0.0,
            trainer=None,
            seed=None,
            actor_id=BaselineKind.RANDOM.value,
        ),
        eligible=True,
    )

    assert result.median_iid_submitted_success_rate == 0.5
    assert result.seeds_strictly_beating_random == 5
    assert result.criterion_met is True
    assert result.status == "criterion_met"

    equality = evaluate_ppo_criterion(
        iid_rows=tuple(_ppo_row(seed=index, rate=rate) for index, rate in enumerate(rates)),
        random_iid_row=_row(
            rate=0.5,
            trainer=None,
            seed=None,
            actor_id=BaselineKind.RANDOM.value,
        ),
        eligible=True,
    )
    assert equality.seeds_strictly_beating_random == 2
    assert equality.criterion_met is False
    assert equality.status == "criterion_not_met"


@pytest.mark.parametrize(
    "rows",
    [
        tuple(_ppo_row(seed, 0.5) for seed in range(4)),
        tuple(_ppo_row(seed, 0.5) for seed in (0, 1, 2, 3, 3)),
        tuple(_ppo_row(seed, 0.5) for seed in range(6)),
        (_row(rate=0.5, trainer=TrainerKind.PPO, seed=None, actor_id="ppo-best"),),
    ],
)
def test_ppo_gate_rejects_missing_duplicate_extra_and_best_seed_pooling(
    rows: tuple[EvaluationRow, ...],
) -> None:
    with pytest.raises(ValueError, match="seeds"):
        evaluate_ppo_criterion(
            iid_rows=rows,
            random_iid_row=_row(
                rate=0.0, trainer=None, seed=None, actor_id=BaselineKind.RANDOM.value
            ),
            eligible=True,
        )


def test_gates_reject_mismatched_suite_digest_and_diagnostic_rows() -> None:
    ppo_rows = tuple(_ppo_row(seed, 0.5) for seed in range(5))
    mismatched = replace(ppo_rows[-1], suite_digest_sha256="f" * 64)
    random_row = _row(rate=0.0, trainer=None, seed=None, actor_id=BaselineKind.RANDOM.value)

    with pytest.raises(ValueError, match="suite"):
        evaluate_ppo_criterion(
            iid_rows=(*ppo_rows[:-1], mismatched),
            random_iid_row=random_row,
            eligible=True,
        )
    with pytest.raises(ValueError, match="probe"):
        evaluate_bc_criterion(
            heldout_next_action_accuracy=0.9,
            iid_row=_row(rate=0.5, probe=ZERO_SPECTRUM_PROBE),
            eligible=True,
        )

    ood = EpisodeSuite(
        schema_version=1,
        suite_id=EvaluationSuiteId.REGISTER_OOD,
        suite_seed=1,
        episodes=(
            EpisodeSpec(target_note_index=12, source_pitch_cents=4_900),
            EpisodeSpec(target_note_index=12, source_pitch_cents=7_100),
        ),
        digest_sha256=suite_digest(
            suite_id=EvaluationSuiteId.REGISTER_OOD,
            suite_seed=1,
            episodes=(
                EpisodeSpec(target_note_index=12, source_pitch_cents=4_900),
                EpisodeSpec(target_note_index=12, source_pitch_cents=7_100),
            ),
        ),
    )
    with pytest.raises(ValueError, match="IID"):
        evaluate_bc_criterion(
            heldout_next_action_accuracy=0.9,
            iid_row=_row(rate=0.5, suite=ood),
            eligible=True,
        )


def test_explicit_ineligibility_returns_without_metric_computation() -> None:
    bc = evaluate_bc_criterion(  # type: ignore[arg-type]
        heldout_next_action_accuracy=object(),
        iid_row=object(),
        eligible=False,
    )
    ppo = evaluate_ppo_criterion(  # type: ignore[arg-type]
        iid_rows=object(),
        random_iid_row=object(),
        eligible=False,
    )

    assert (bc.eligible, bc.criterion_met, bc.status) == (False, None, "ineligible")
    assert bc.heldout_next_action_accuracy is None
    assert bc.iid_submitted_success_rate is None
    assert (ppo.eligible, ppo.criterion_met, ppo.status) == (False, None, "ineligible")
    assert ppo.median_iid_submitted_success_rate is None
    assert ppo.seeds_strictly_beating_random is None


def _canonical_file(*, trainer: TrainerKind = TrainerKind.BC) -> EvaluationFile:
    suite = fixed_evaluation_suite(EvaluationSuiteId.SMOKE)
    row = build_evaluation_rows(
        actor_id=f"{trainer.value}-0",
        trainer=trainer,
        seed=0,
        environment_id="Harpy/SinePitch-v0",
        observation_mode=ObservationMode.SPECTRUM,
        suite=suite,
        records=_records(suite, 0.0),
    )[0]
    return EvaluationFile(
        schema_version=1,
        suite_id=suite.suite_id,
        suite_digest_sha256=suite.digest_sha256,
        rows=(row,),
        next_action_accuracy=0.75 if trainer is TrainerKind.BC else None,
    )


def test_evaluation_file_round_trip_is_strict_and_fresh() -> None:
    source = _canonical_file()

    document = source.to_document()
    decoded = EvaluationFile.from_document(document)

    assert decoded == source
    assert document is not source.to_document()
    assert set(document) == {
        "schema_version",
        "suite_id",
        "suite_digest_sha256",
        "rows",
        "next_action_accuracy",
    }
    assert set(document["rows"][0]) == {  # type: ignore[index]
        "actor_id",
        "trainer",
        "seed",
        "environment_id",
        "observation_mode",
        "suite_id",
        "suite_digest_sha256",
        "subset",
        "probe",
        "parameter_count",
        "training_environment_steps",
        "training_examples",
        "training_wall_time_seconds",
        "metrics",
        "episodes",
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda doc: doc.update(extra=True), "fields"),
        (lambda doc: doc.pop("rows"), "fields"),
        (lambda doc: doc.update(suite_id="not-a-suite"), "suite_id"),
        (lambda doc: doc.update(next_action_accuracy=1.01), "next_action_accuracy"),
        (lambda doc: doc["rows"][0]["metrics"].update(mean_return=float("nan")), "finite"),
        (lambda doc: doc["rows"][0]["metrics"].update(episodes=31), "denominator"),
        (lambda doc: doc["rows"][0]["episodes"].reverse(), "ordered episode membership"),
        (lambda doc: doc["rows"][0].update(suite_digest_sha256="f" * 64), "suite"),
    ],
)
def test_evaluation_file_codec_rejects_malformed_documents(mutation, message: str) -> None:
    document = copy.deepcopy(_canonical_file().to_document())
    mutation(document)

    with pytest.raises(ValueError, match=message):
        EvaluationFile.from_document(document)


def test_evaluation_file_rejects_duplicate_rows_and_disallowed_accuracy() -> None:
    document = _canonical_file().to_document()
    document["rows"].append(copy.deepcopy(document["rows"][0]))  # type: ignore[union-attr,index]
    with pytest.raises(ValueError, match="duplicate"):
        EvaluationFile.from_document(document)

    ppo_document = _canonical_file(trainer=TrainerKind.PPO).to_document()
    ppo_document["next_action_accuracy"] = 0.9
    with pytest.raises(ValueError, match="next_action_accuracy"):
        EvaluationFile.from_document(ppo_document)


def test_terminal_record_rejects_tolerance_flags_inconsistent_with_final_error() -> None:
    episode = EpisodeSpec(target_note_index=12, source_pitch_cents=6_100)

    with pytest.raises(ValueError, match="within_5_cents"):
        TerminalEpisodeRecord(
            episode_index=0,
            episode=episode,
            submitted_success=False,
            within_5_cents=False,
            within_1_cent=False,
            final_absolute_error_cents=1,
            action_count=1,
            excess_actions=None,
            invalid_action_count=0,
            total_return=-1.0,
            terminal_reason=TerminalReason.SUBMITTED_FAILURE,
        )


def test_terminal_record_rejects_impossible_failed_submission_and_excess_count() -> None:
    episode = EpisodeSpec(target_note_index=12, source_pitch_cents=6_100)

    with pytest.raises(ValueError, match="failed submissions"):
        TerminalEpisodeRecord(
            episode_index=0,
            episode=episode,
            submitted_success=False,
            within_5_cents=True,
            within_1_cent=True,
            final_absolute_error_cents=1,
            action_count=1,
            excess_actions=None,
            invalid_action_count=0,
            total_return=-1.0,
            terminal_reason=TerminalReason.SUBMITTED_FAILURE,
        )
    with pytest.raises(ValueError, match="excess_actions"):
        TerminalEpisodeRecord(
            episode_index=0,
            episode=episode,
            submitted_success=True,
            within_5_cents=True,
            within_1_cent=True,
            final_absolute_error_cents=1,
            action_count=1,
            excess_actions=1,
            invalid_action_count=0,
            total_return=1.0,
            terminal_reason=TerminalReason.SUBMITTED_SUCCESS,
        )


def test_evaluation_row_rejects_mislabeled_declared_baseline_lane() -> None:
    random_row = _row(
        rate=0.0,
        trainer=None,
        seed=None,
        actor_id=BaselineKind.RANDOM.value,
    )

    with pytest.raises(ValueError, match="declared baseline"):
        replace(random_row, environment_id="unregistered-env")


def test_evaluation_file_codec_rejects_numeric_strings() -> None:
    document = _canonical_file().to_document()
    document["rows"][0]["metrics"]["mean_return"] = "-1.0"  # type: ignore[index]

    with pytest.raises(ValueError, match="finite"):
        EvaluationFile.from_document(document)


def test_probe_rows_are_never_accepted_by_scientific_gates() -> None:
    for probe in (ZERO_SPECTRUM_PROBE, SHUFFLED_SPECTRUM_PROBE):
        with pytest.raises(ValueError, match="probe"):
            evaluate_bc_criterion(
                heldout_next_action_accuracy=0.9,
                iid_row=_row(rate=0.5, probe=probe),
                eligible=True,
            )
