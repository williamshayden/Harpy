"""Milestone E direct-coordinate and closed-loop evaluation contracts."""

from __future__ import annotations

from dataclasses import replace

import gymnasium
import numpy as np
import pytest

from harpy.envs.baselines import BaselineKind
from harpy.envs.models import TARGET_MIN_COORDINATE, ObservationMode, TerminalReason
from harpy.envs.planning import minimum_action_plan
from harpy.learning.evaluation import TerminalEpisodeRecord
from harpy.learning.models import (
    ENVIRONMENT_ID,
    EpisodeSpec,
    PitchCoordinatePartition,
    PitchCoordinateRecord,
)
from harpy.learning.pitch_data import (
    PITCH_DISTRIBUTION_ID,
    PITCH_SPLIT_DIGEST_SHA256,
    PitchEvaluationSuiteId,
    PitchTrainerKind,
    fixed_pitch_evaluation_suite,
    iid_shuffled_spectrum_permutation,
    pitch_coordinate_split,
)
from harpy.learning.pitch_evaluation import (
    PITCH_SHUFFLED_SPECTRUM_PROBE,
    PITCH_ZERO_SPECTRUM_PROBE,
    PitchCoordinateEvaluation,
    PitchEvaluationRow,
    RegisterOODAggregate,
    build_pitch_evaluation_row,
    evaluate_final_pitch_coordinates,
    make_pitch_model_grid_predictor,
    make_pitch_spectrum_probe_factory,
    summarize_pitch_coordinate_records,
    validate_pitch_checkpoint_row_matrix,
    validate_pitch_smoke_row_matrix,
)
from harpy.learning.pitch_network import PitchEstimatorNetwork


def _record(
    *,
    seed: int,
    partition: PitchCoordinatePartition,
    coordinate: int,
    predicted_cents: int | None = None,
) -> PitchCoordinateRecord:
    prediction = coordinate - coordinate % 5 if predicted_cents is None else predicted_cents
    return PitchCoordinateRecord(
        seed=seed,
        distribution_id=PITCH_DISTRIBUTION_ID,
        split_digest_sha256=PITCH_SPLIT_DIGEST_SHA256,
        partition=partition,
        true_coordinate_cents=coordinate,
        predicted_grid_index=(prediction - 1_100) // 5,
        predicted_cents=prediction,
        signed_error_cents=prediction - coordinate,
        absolute_error_cents=abs(prediction - coordinate),
    )


def _coordinate_evaluation(seed: int, *, iid_bad: int = 0, ood_bad: int = 0):
    split = pitch_coordinate_split()
    members = (
        (PitchCoordinatePartition.IID, tuple(sorted(split.iid_holdout_coordinates))),
        (PitchCoordinatePartition.OOD_LOWER, split.ood_lower_coordinates),
        (PitchCoordinatePartition.OOD_UPPER, split.ood_upper_coordinates),
    )
    records: list[PitchCoordinateRecord] = []
    iid_seen = 0
    ood_seen = 0
    for partition, coordinates in members:
        for coordinate in coordinates:
            make_bad = False
            if partition is PitchCoordinatePartition.IID and iid_seen < iid_bad:
                iid_seen += 1
                make_bad = True
            elif partition is not PitchCoordinatePartition.IID and ood_seen < ood_bad:
                ood_seen += 1
                make_bad = True
            predicted = coordinate - coordinate % 5 + (10 if make_bad else 0)
            records.append(
                _record(
                    seed=seed,
                    partition=partition,
                    coordinate=coordinate,
                    predicted_cents=predicted,
                )
            )
    return PitchCoordinateEvaluation.from_records(seed=seed, records=tuple(records))


def _terminal_records(suite_id: PitchEvaluationSuiteId, successes: int):
    suite = fixed_pitch_evaluation_suite(suite_id)
    return tuple(
        TerminalEpisodeRecord(
            episode_index=index,
            episode=EpisodeSpec(episode.target_note_index, episode.source_pitch_cents),
            submitted_success=index < successes,
            within_5_cents=index < successes,
            within_1_cent=index < successes,
            final_absolute_error_cents=0 if index < successes else 100,
            action_count=(
                len(
                    minimum_action_plan(
                        episode.source_pitch_cents
                        - 100 * (TARGET_MIN_COORDINATE + episode.target_note_index)
                    )
                )
                if index < successes
                else 1
            ),
            excess_actions=0 if index < successes else None,
            invalid_action_count=0,
            total_return=1.0 if index < successes else -1.0,
            terminal_reason=(
                TerminalReason.SUBMITTED_SUCCESS
                if index < successes
                else TerminalReason.SUBMITTED_FAILURE
            ),
        )
        for index, episode in enumerate(suite.episodes)
    )


def _learned_row(
    seed: int,
    suite_id: PitchEvaluationSuiteId,
    *,
    probe: str | None = None,
    successes: int | None = None,
) -> PitchEvaluationRow:
    suite = fixed_pitch_evaluation_suite(suite_id)
    return build_pitch_evaluation_row(
        actor_id=f"pitch-{seed}",
        trainer=PitchTrainerKind.PITCH,
        seed=seed,
        environment_id=ENVIRONMENT_ID,
        observation_mode=ObservationMode.SPECTRUM,
        suite=suite,
        records=_terminal_records(
            suite_id,
            len(suite.episodes) if successes is None else successes,
        ),
        probe=probe,
        parameter_count=2_497,
        training_examples=1_400,
        training_wall_time_seconds=1.0,
    )


def _baseline_row(kind: BaselineKind, suite_id: PitchEvaluationSuiteId) -> PitchEvaluationRow:
    suite = fixed_pitch_evaluation_suite(suite_id)
    return build_pitch_evaluation_row(
        actor_id=kind.value,
        trainer=None,
        seed=None,
        environment_id=kind.environment_id,
        observation_mode=kind.observation_mode,
        suite=suite,
        records=_terminal_records(suite_id, len(suite.episodes)),
    )


def _checkpoint_rows() -> tuple[PitchEvaluationRow, ...]:
    rows: list[PitchEvaluationRow] = []
    for seed in (0, 1, 2):
        rows.extend(
            (
                _learned_row(seed, PitchEvaluationSuiteId.IID),
                _learned_row(seed, PitchEvaluationSuiteId.OOD_LOWER),
                _learned_row(seed, PitchEvaluationSuiteId.OOD_UPPER),
                _learned_row(
                    seed,
                    PitchEvaluationSuiteId.IID,
                    probe=PITCH_ZERO_SPECTRUM_PROBE,
                    successes=100,
                ),
                _learned_row(
                    seed,
                    PitchEvaluationSuiteId.IID,
                    probe=PITCH_SHUFFLED_SPECTRUM_PROBE,
                    successes=100,
                ),
            )
        )
    for kind in (
        BaselineKind.RANDOM,
        BaselineKind.REWARD_SEARCH,
        BaselineKind.SPECTRUM_PEAK,
        BaselineKind.ORACLE,
    ):
        rows.extend(
            _baseline_row(kind, suite_id)
            for suite_id in (
                PitchEvaluationSuiteId.IID,
                PitchEvaluationSuiteId.OOD_LOWER,
                PitchEvaluationSuiteId.OOD_UPPER,
            )
        )
    return tuple(rows)


def test_pitch_coordinate_record_rederives_every_prediction_field() -> None:
    record = _record(
        seed=2,
        partition=PitchCoordinatePartition.IID,
        coordinate=5_038,
        predicted_cents=5_040,
    )
    assert record.predicted_grid_index == 788
    assert record.signed_error_cents == 2
    assert record.absolute_error_cents == 2

    for changes, message in (
        ({"predicted_grid_index": 780}, "predicted_cents"),
        ({"signed_error_cents": -2}, "signed_error"),
        ({"absolute_error_cents": 3}, "absolute_error"),
        ({"split_digest_sha256": "f" * 64}, "pinned"),
    ):
        with pytest.raises(ValueError, match=message):
            replace(record, **changes)


def test_coordinate_metrics_use_nearest_rank_and_are_recomputed_from_records() -> None:
    records = tuple(
        _record(
            seed=0,
            partition=PitchCoordinatePartition.OOD_LOWER,
            coordinate=4_800 + error,
            predicted_cents=4_800,
        )
        for error in (0, 1, 2, 5, 6, 10, 20, 30, 40, 50)
    )
    metrics = summarize_pitch_coordinate_records(records)

    assert metrics.count == 10
    assert (metrics.within_one_count, metrics.within_five_count) == (2, 4)
    assert (metrics.within_one_rate, metrics.within_five_rate) == (0.2, 0.4)
    assert metrics.mean_absolute_error_cents == 16.4
    assert (metrics.p50_cents, metrics.p90_cents, metrics.p95_cents, metrics.p99_cents) == (
        6,
        40,
        50,
        50,
    )
    assert metrics.max_cents == 50


def test_complete_coordinate_evaluation_pins_order_register_and_residue_tables() -> None:
    evaluation = _coordinate_evaluation(1)

    assert len(evaluation.records) == 801
    assert tuple(row.partition for row in evaluation.register_rows) == (
        PitchCoordinatePartition.IID,
        PitchCoordinatePartition.OOD_LOWER,
        PitchCoordinatePartition.OOD_UPPER,
    )
    assert tuple(row.residue for row in evaluation.residue_rows) == (0, 1, 2, 3, 4)
    assert sum(row.metrics.count for row in evaluation.residue_rows) == 801
    assert all(row.metrics.within_five_rate == 1.0 for row in evaluation.register_rows)

    with pytest.raises(ValueError, match="ordered final coordinate membership"):
        PitchCoordinateEvaluation.from_records(
            seed=1,
            records=tuple(reversed(evaluation.records)),
        )


def test_direct_evaluator_exposes_only_immutable_spectrum_to_predictor() -> None:
    split = pitch_coordinate_split()
    observed_coordinates: list[int] = []
    observed_spectra: list[np.ndarray] = []

    class Evidence:
        def spectrum_for_cents(self, coordinate: int) -> np.ndarray:
            observed_coordinates.append(coordinate)
            return np.full((1_961,), coordinate / 10_000, dtype=np.float32)

    def predictor(spectrum: np.ndarray) -> int:
        observed_spectra.append(spectrum)
        assert spectrum.flags.writeable is False
        coordinate = round(float(spectrum[0]) * 10_000)
        return (coordinate - 1_100 + 2) // 5

    evaluation = evaluate_final_pitch_coordinates(
        seed=0,
        evidence_provider=Evidence(),
        predict_grid_index=predictor,
    )

    expected = (
        *sorted(split.iid_holdout_coordinates),
        *split.ood_lower_coordinates,
        *split.ood_upper_coordinates,
    )
    assert tuple(observed_coordinates) == expected
    assert len(observed_spectra) == 801
    assert all(row.metrics.within_five_rate == 1.0 for row in evaluation.register_rows)


def test_model_predictor_rejects_non_cpu_cuda_before_mutating_model() -> None:
    model = PitchEstimatorNetwork()

    with pytest.raises(ValueError, match="CPU or CUDA"):
        make_pitch_model_grid_predictor(model, device="meta")

    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


class _OneObservationEnv(gymnasium.Env):
    def __init__(self, observation: dict[str, object]) -> None:
        super().__init__()
        self._observation = observation
        self.observation_space = gymnasium.spaces.Dict({})
        self.action_space = gymnasium.spaces.Discrete(1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._observation, {}


def test_v2_probes_change_only_spectrum_and_use_the_exact_iid_permutation() -> None:
    spectrum = np.arange(1_961, dtype=np.float32) / np.float32(1_960)
    original = {
        "spectrum": spectrum,
        "target_note": np.int64(12),
        "controls": np.array([0, 0, 0], dtype=np.int16),
        "steps_remaining": np.int64(64),
    }

    snapshots: dict[str, dict[str, object]] = {}
    for probe in (PITCH_ZERO_SPECTRUM_PROBE, PITCH_SHUFFLED_SPECTRUM_PROBE):
        env = make_pitch_spectrum_probe_factory(lambda: _OneObservationEnv(original), probe)()
        transformed, _ = env.reset()
        snapshots[probe] = transformed
        assert transformed is not original
        for key in ("target_note", "controls", "steps_remaining"):
            assert transformed[key] is original[key]
        assert transformed["spectrum"] is not spectrum

    assert np.all(snapshots[PITCH_ZERO_SPECTRUM_PROBE]["spectrum"] == 0.0)
    assert np.array_equal(
        snapshots[PITCH_SHUFFLED_SPECTRUM_PROBE]["spectrum"],
        spectrum[np.asarray(iid_shuffled_spectrum_permutation())],
    )
    assert np.array_equal(original["spectrum"], spectrum)


def test_register_ood_aggregate_requires_two_separate_canonical_suites() -> None:
    lower = _learned_row(0, PitchEvaluationSuiteId.OOD_LOWER)
    upper = _learned_row(0, PitchEvaluationSuiteId.OOD_UPPER)
    aggregate = RegisterOODAggregate.from_rows(lower, upper)

    assert aggregate.lower_suite_id is PitchEvaluationSuiteId.OOD_LOWER
    assert aggregate.upper_suite_id is PitchEvaluationSuiteId.OOD_UPPER
    assert aggregate.metrics.episodes == 400
    assert aggregate.metrics.submitted_success_rate == 1.0

    with pytest.raises(ValueError, match="lower and upper"):
        RegisterOODAggregate.from_rows(lower, lower)
    with pytest.raises(ValueError, match="same actor"):
        RegisterOODAggregate.from_rows(lower, _learned_row(1, PitchEvaluationSuiteId.OOD_UPPER))


def test_checkpoint_matrix_is_exact_and_never_synthesizes_combined_ood_row() -> None:
    rows = _checkpoint_rows()
    validated = validate_pitch_checkpoint_row_matrix(rows)

    assert validated == rows
    assert len(rows) == 27
    assert not any("combined" in row.suite_id.value for row in rows)

    with pytest.raises(ValueError, match="exact checkpoint row matrix"):
        validate_pitch_checkpoint_row_matrix(rows[:-1])
    with pytest.raises(ValueError, match="exact checkpoint row matrix"):
        validate_pitch_checkpoint_row_matrix((rows[1], rows[0], *rows[2:]))


def test_smoke_matrix_requires_one_complete_learned_training_identity() -> None:
    rows = (
        _learned_row(7, PitchEvaluationSuiteId.SMOKE),
        _learned_row(
            7,
            PitchEvaluationSuiteId.SMOKE,
            probe=PITCH_ZERO_SPECTRUM_PROBE,
        ),
        _learned_row(
            7,
            PitchEvaluationSuiteId.SMOKE,
            probe=PITCH_SHUFFLED_SPECTRUM_PROBE,
        ),
        *(
            _baseline_row(kind, PitchEvaluationSuiteId.SMOKE)
            for kind in (
                BaselineKind.RANDOM,
                BaselineKind.REWARD_SEARCH,
                BaselineKind.SPECTRUM_PEAK,
                BaselineKind.ORACLE,
            )
        ),
    )

    assert validate_pitch_smoke_row_matrix(rows) == rows
    with pytest.raises(ValueError, match="exact smoke row matrix"):
        validate_pitch_smoke_row_matrix(
            (rows[0], replace(rows[1], training_examples=1_399), *rows[2:])
        )


def test_pitch_evaluation_row_rejects_wrong_membership_and_cross_lane_metadata() -> None:
    row = _learned_row(0, PitchEvaluationSuiteId.IID)
    with pytest.raises(ValueError, match="ordered episode membership"):
        replace(row, episodes=tuple(reversed(row.episodes)))
    with pytest.raises(ValueError, match=r"baseline|learned"):
        replace(row, trainer=None)
    with pytest.raises(ValueError, match="probe"):
        replace(
            row,
            suite_id=PitchEvaluationSuiteId.OOD_LOWER,
            suite_digest_sha256=fixed_pitch_evaluation_suite(
                PitchEvaluationSuiteId.OOD_LOWER
            ).digest_sha256,
            probe=PITCH_ZERO_SPECTRUM_PROBE,
        )
    with pytest.raises(ValueError, match="complete training metadata"):
        replace(row, parameter_count=None)

    baseline = _baseline_row(BaselineKind.RANDOM, PitchEvaluationSuiteId.IID)
    with pytest.raises(ValueError, match="must not contain training metadata"):
        replace(baseline, parameter_count=1)


def test_pitch_rows_rederive_successful_excess_actions_from_raw_episode_identity() -> None:
    row = _learned_row(0, PitchEvaluationSuiteId.IID)
    tampered = replace(row.episodes[0], excess_actions=1)

    with pytest.raises(ValueError, match="excess_actions must be re-derived"):
        replace(row, episodes=(tampered, *row.episodes[1:]))
