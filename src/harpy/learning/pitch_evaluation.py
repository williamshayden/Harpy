"""Schema-v2 perception and closed-loop evaluation for the pitch estimator.

The legacy Milestone D evaluation models remain untouched.  This module owns the
separate pitch trainer/suite matrix, recomputes every direct metric from ordered raw
records, and exposes only spectrum evidence to coordinate predictors.
"""

from __future__ import annotations

import math
import operator
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import InitVar, dataclass
from typing import Any

import gymnasium
import numpy as np

from harpy.envs.baselines import BaselineKind
from harpy.envs.models import TARGET_MIN_COORDINATE, ObservationMode, TerminalReason
from harpy.envs.planning import minimum_action_plan
from harpy.learning.actors import Actor
from harpy.learning.cache import SpectrumEvidenceCache
from harpy.learning.envs import SpectrumEvidenceProvider
from harpy.learning.evaluation import (
    AggregateMetrics,
    TerminalEpisodeRecord,
    aggregate_episode_records,
    evaluate_baseline_suite,
    evaluate_learned_actor,
    make_indexed_spectrum_probe_factory,
)
from harpy.learning.models import (
    ENVIRONMENT_ID,
    EpisodeSpec,
    EpisodeSuite,
    EvaluationSuiteId,
    PitchCoordinatePartition,
    PitchCoordinateRecord,
)
from harpy.learning.pitch_data import (
    PitchEvaluationSuite,
    PitchEvaluationSuiteId,
    PitchTrainerKind,
    fixed_pitch_evaluation_suite,
    iid_shuffled_spectrum_permutation,
    pitch_coordinate_split,
)
from harpy.learning.suites import suite_digest

PITCH_ZERO_SPECTRUM_PROBE = "zero_spectrum"
PITCH_SHUFFLED_SPECTRUM_PROBE = "shuffled_spectrum"
PITCH_SPECTRUM_SHUFFLE_ID = "harpy-sine-pitch-spectrum-shuffle-v1"

_PITCH_PROBES = frozenset({PITCH_ZERO_SPECTRUM_PROBE, PITCH_SHUFFLED_SPECTRUM_PROBE})
_BASELINE_ORDER = (
    BaselineKind.RANDOM,
    BaselineKind.REWARD_SEARCH,
    BaselineKind.SPECTRUM_PEAK,
    BaselineKind.ORACLE,
)
_FINAL_PARTITION_ORDER = (
    PitchCoordinatePartition.IID,
    PitchCoordinatePartition.OOD_LOWER,
    PitchCoordinatePartition.OOD_UPPER,
)
_CRITERION_EVIDENCE_TOKEN = object()


def _integer(
    value: object,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        normalized = int(operator.index(value))
    except TypeError as error:
        raise ValueError(f"{field} must be an integer") from error
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return normalized


def _finite_float(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be a finite number")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return normalized


def _optional_integer(value: object, field: str) -> int | None:
    return None if value is None else _integer(value, field, minimum=0)


def _optional_float(value: object, field: str) -> float | None:
    return None if value is None else _finite_float(value, field, minimum=0.0)


def _typed_tuple[T](
    value: Iterable[T],
    expected_type: type[T],
    field: str,
    *,
    nonempty: bool = True,
) -> tuple[T, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must contain {expected_type.__name__} values")
    try:
        normalized = tuple(value)
    except TypeError as error:
        raise ValueError(f"{field} must contain {expected_type.__name__} values") from error
    if (nonempty and not normalized) or not all(
        isinstance(item, expected_type) for item in normalized
    ):
        raise ValueError(f"{field} must contain {expected_type.__name__} values")
    return normalized


@dataclass(frozen=True, slots=True)
class PitchCoordinateMetrics:
    """Direct pitch-error metrics over one nonempty raw-record denominator."""

    count: int
    within_one_count: int
    within_one_rate: float
    within_five_count: int
    within_five_rate: float
    mean_absolute_error_cents: float
    p50_cents: int
    p90_cents: int
    p95_cents: int
    p99_cents: int
    max_cents: int

    def __post_init__(self) -> None:
        count = _integer(self.count, "count", minimum=1)
        within_one = _integer(
            self.within_one_count,
            "within_one_count",
            minimum=0,
            maximum=count,
        )
        within_five = _integer(
            self.within_five_count,
            "within_five_count",
            minimum=within_one,
            maximum=count,
        )
        within_one_rate = _finite_float(
            self.within_one_rate,
            "within_one_rate",
            minimum=0.0,
            maximum=1.0,
        )
        within_five_rate = _finite_float(
            self.within_five_rate,
            "within_five_rate",
            minimum=0.0,
            maximum=1.0,
        )
        if within_one_rate != within_one / count or within_five_rate != within_five / count:
            raise ValueError("within-one/five rates must be re-derived from their counts")
        percentiles = tuple(
            _integer(getattr(self, field), field, minimum=0)
            for field in ("p50_cents", "p90_cents", "p95_cents", "p99_cents", "max_cents")
        )
        if tuple(sorted(percentiles)) != percentiles:
            raise ValueError("pitch error percentiles and maximum must be nondecreasing")
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "within_one_count", within_one)
        object.__setattr__(self, "within_one_rate", within_one_rate)
        object.__setattr__(self, "within_five_count", within_five)
        object.__setattr__(self, "within_five_rate", within_five_rate)
        object.__setattr__(
            self,
            "mean_absolute_error_cents",
            _finite_float(
                self.mean_absolute_error_cents,
                "mean_absolute_error_cents",
                minimum=0.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class PitchRegisterRow:
    """One exact direct-coordinate register row."""

    partition: PitchCoordinatePartition
    metrics: PitchCoordinateMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.partition, PitchCoordinatePartition):
            raise ValueError("partition must be a PitchCoordinatePartition")
        if not isinstance(self.metrics, PitchCoordinateMetrics):
            raise ValueError("metrics must be PitchCoordinateMetrics")


@dataclass(frozen=True, slots=True)
class PitchResidueRow:
    """One exact true-coordinate modulo-five row across all final registers."""

    residue: int
    metrics: PitchCoordinateMetrics

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "residue",
            _integer(self.residue, "residue", minimum=0, maximum=4),
        )
        if not isinstance(self.metrics, PitchCoordinateMetrics):
            raise ValueError("metrics must be PitchCoordinateMetrics")


def summarize_pitch_coordinate_records(
    records: Sequence[PitchCoordinateRecord],
) -> PitchCoordinateMetrics:
    """Recompute counts, rates, MAE, and exact nearest-rank errors."""

    normalized = _typed_tuple(records, PitchCoordinateRecord, "records")
    errors = tuple(record.absolute_error_cents for record in normalized)
    ordered = tuple(sorted(errors))
    count = len(ordered)

    def nearest_rank(quantile: float) -> int:
        return ordered[math.ceil(quantile * count) - 1]

    within_one = sum(error <= 1 for error in errors)
    within_five = sum(error <= 5 for error in errors)
    return PitchCoordinateMetrics(
        count=count,
        within_one_count=within_one,
        within_one_rate=within_one / count,
        within_five_count=within_five,
        within_five_rate=within_five / count,
        mean_absolute_error_cents=math.fsum(errors) / count,
        p50_cents=nearest_rank(0.50),
        p90_cents=nearest_rank(0.90),
        p95_cents=nearest_rank(0.95),
        p99_cents=nearest_rank(0.99),
        max_cents=ordered[-1],
    )


def _partition_coordinates(partition: PitchCoordinatePartition) -> tuple[int, ...]:
    split = pitch_coordinate_split()
    if partition is PitchCoordinatePartition.IID:
        return tuple(sorted(split.iid_holdout_coordinates))
    if partition is PitchCoordinatePartition.OOD_LOWER:
        return split.ood_lower_coordinates
    return split.ood_upper_coordinates


def _expected_coordinate_membership() -> tuple[tuple[PitchCoordinatePartition, int], ...]:
    return tuple(
        (partition, coordinate)
        for partition in _FINAL_PARTITION_ORDER
        for coordinate in _partition_coordinates(partition)
    )


@dataclass(frozen=True, slots=True)
class PitchCoordinateEvaluation:
    """All 801 final direct predictions and their re-derived breakdown tables."""

    seed: int
    records: tuple[PitchCoordinateRecord, ...]
    register_rows: tuple[PitchRegisterRow, ...]
    residue_rows: tuple[PitchResidueRow, ...]

    def __post_init__(self) -> None:
        seed = _integer(self.seed, "seed", minimum=0)
        records = _typed_tuple(self.records, PitchCoordinateRecord, "records")
        actual_membership = tuple(
            (record.partition, record.true_coordinate_cents) for record in records
        )
        if actual_membership != _expected_coordinate_membership():
            raise ValueError("records must preserve ordered final coordinate membership")
        if any(record.seed != seed for record in records):
            raise ValueError("every coordinate record seed must match the evaluation seed")
        register_rows = _typed_tuple(
            self.register_rows,
            PitchRegisterRow,
            "register_rows",
        )
        expected_register_rows = tuple(
            PitchRegisterRow(
                partition=partition,
                metrics=summarize_pitch_coordinate_records(
                    tuple(record for record in records if record.partition is partition)
                ),
            )
            for partition in _FINAL_PARTITION_ORDER
        )
        if register_rows != expected_register_rows:
            raise ValueError("register_rows must be re-derived from all coordinate records")
        residue_rows = _typed_tuple(self.residue_rows, PitchResidueRow, "residue_rows")
        expected_residue_rows = tuple(
            PitchResidueRow(
                residue=residue,
                metrics=summarize_pitch_coordinate_records(
                    tuple(
                        record for record in records if record.true_coordinate_cents % 5 == residue
                    )
                ),
            )
            for residue in range(5)
        )
        if residue_rows != expected_residue_rows:
            raise ValueError("residue_rows must be re-derived from all coordinate records")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "register_rows", register_rows)
        object.__setattr__(self, "residue_rows", residue_rows)

    @classmethod
    def from_records(
        cls,
        *,
        seed: int,
        records: Sequence[PitchCoordinateRecord],
    ) -> PitchCoordinateEvaluation:
        """Build every derived table from the complete ordered raw records."""

        normalized = _typed_tuple(records, PitchCoordinateRecord, "records")
        return cls(
            seed=seed,
            records=normalized,
            register_rows=tuple(
                PitchRegisterRow(
                    partition=partition,
                    metrics=summarize_pitch_coordinate_records(
                        tuple(record for record in normalized if record.partition is partition)
                    ),
                )
                for partition in _FINAL_PARTITION_ORDER
            ),
            residue_rows=tuple(
                PitchResidueRow(
                    residue=residue,
                    metrics=summarize_pitch_coordinate_records(
                        tuple(
                            record
                            for record in normalized
                            if record.true_coordinate_cents % 5 == residue
                        )
                    ),
                )
                for residue in range(5)
            ),
        )

    def register_metrics(self, partition: PitchCoordinatePartition) -> PitchCoordinateMetrics:
        """Return the exact row for one declared register."""

        if not isinstance(partition, PitchCoordinatePartition):
            raise ValueError("partition must be a PitchCoordinatePartition")
        return next(row.metrics for row in self.register_rows if row.partition is partition)


def _owned_spectrum(value: object) -> np.ndarray[Any, np.dtype[np.float32]]:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float32):
        raise ValueError("spectrum must be a float32 ndarray")
    if value.shape != (1_961,):
        raise ValueError("spectrum must have shape (1961,)")
    if not np.isfinite(value).all():
        raise ValueError("spectrum must contain only finite values")
    if np.any(value < np.float32(0.0)) or np.any(value > np.float32(1.0)):
        raise ValueError("spectrum values must be within 0..1")
    spectrum = np.array(value, dtype=np.float32, copy=True, order="C")
    spectrum.setflags(write=False)
    return spectrum


def evaluate_final_pitch_coordinates(
    *,
    seed: int,
    evidence_provider: object,
    predict_grid_index: Callable[[np.ndarray[Any, np.dtype[np.float32]]], object],
) -> PitchCoordinateEvaluation:
    """Evaluate all disjoint final coordinates through the spectrum-only seam."""

    normalized_seed = _integer(seed, "seed", minimum=0)
    spectrum_for_cents = getattr(evidence_provider, "spectrum_for_cents", None)
    if not callable(spectrum_for_cents):
        raise ValueError("evidence_provider must provide spectrum_for_cents")
    if not callable(predict_grid_index):
        raise ValueError("predict_grid_index must be callable")
    split = pitch_coordinate_split()
    records: list[PitchCoordinateRecord] = []
    for partition, coordinate in _expected_coordinate_membership():
        spectrum = _owned_spectrum(spectrum_for_cents(coordinate))
        predicted_index = _integer(
            predict_grid_index(spectrum),
            "predicted_grid_index",
            minimum=0,
            maximum=1_960,
        )
        predicted_cents = 1_100 + 5 * predicted_index
        signed_error = predicted_cents - coordinate
        records.append(
            PitchCoordinateRecord(
                seed=normalized_seed,
                distribution_id=split.distribution_id,
                split_digest_sha256=split.digest_sha256,
                partition=partition,
                true_coordinate_cents=coordinate,
                predicted_grid_index=predicted_index,
                predicted_cents=predicted_cents,
                signed_error_cents=signed_error,
                absolute_error_cents=abs(signed_error),
            )
        )
    return PitchCoordinateEvaluation.from_records(
        seed=normalized_seed,
        records=tuple(records),
    )


def make_pitch_model_grid_predictor(
    model: object,
    *,
    device: object | None = None,
) -> Callable[[np.ndarray[Any, np.dtype[np.float32]]], int]:
    """Create a strict lazy Torch adapter for direct-coordinate evaluation."""

    from harpy.learning.dependencies import require_training_dependencies
    from harpy.learning.pitch_network import validate_pitch_estimator_model

    stack = require_training_dependencies()
    torch = stack.torch
    validate_pitch_estimator_model(model)
    try:
        target_device = torch.device("cpu" if device is None else device)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("device must identify CPU or CUDA") from error
    if target_device.type not in {"cpu", "cuda"}:
        raise ValueError("device must identify CPU or CUDA")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA evaluation requested but CUDA is unavailable")
    model.to(target_device)
    model.eval()

    def predict(spectrum: np.ndarray[Any, np.dtype[np.float32]]) -> int:
        owned = _owned_spectrum(spectrum)
        tensor = torch.from_numpy(np.array(owned, copy=True)).unsqueeze(0).to(target_device)
        with torch.inference_mode():
            logits = model(tensor)
        if (
            not isinstance(logits, torch.Tensor)
            or logits.shape != (1, 1_961)
            or logits.dtype != torch.float32
            or not torch.isfinite(logits).all()
        ):
            raise RuntimeError("pitch estimator logits must be finite float32 shape (1, 1961)")
        return int(torch.argmax(logits, dim=1).item())

    return predict


def make_pitch_spectrum_probe_factory(
    environment_factory: Callable[[], gymnasium.Env],
    probe: str,
) -> Callable[[], gymnasium.Env]:
    """Wrap an environment with the exact digest-derived schema-v2 IID probe."""

    if probe not in _PITCH_PROBES:
        raise ValueError("probe must be zero_spectrum or shuffled_spectrum")
    return make_indexed_spectrum_probe_factory(
        environment_factory,
        probe,
        shuffle_permutation=iid_shuffled_spectrum_permutation(),
    )


def _legacy_suite_for_rollout(suite: PitchEvaluationSuite) -> EpisodeSuite:
    """Adapt reset membership only, without admitting v2 IDs to schema-v1 codecs."""

    episodes = tuple(
        EpisodeSpec(episode.target_note_index, episode.source_pitch_cents)
        for episode in suite.episodes
    )
    legacy_id = EvaluationSuiteId.SMOKE
    digest = suite_digest(
        suite_id=legacy_id,
        suite_seed=suite.suite_seed,
        episodes=episodes,
    )
    return EpisodeSuite(
        schema_version=1,
        suite_id=legacy_id,
        suite_seed=suite.suite_seed,
        episodes=episodes,
        digest_sha256=digest,
    )


def evaluate_pitch_learned_actor(
    actor: Actor,
    suite: PitchEvaluationSuite,
    *,
    environment_factory: Callable[[], gymnasium.Env],
) -> tuple[TerminalEpisodeRecord, ...]:
    """Run one learned actor on exact v2 membership through the shared runner."""

    canonical = _canonical_pitch_suite(suite)
    return evaluate_learned_actor(
        actor,
        _legacy_suite_for_rollout(canonical),
        environment_factory=environment_factory,
    )


def evaluate_pitch_baseline_suite(
    kind: BaselineKind,
    suite: PitchEvaluationSuite,
    *,
    cache: SpectrumEvidenceCache,
) -> tuple[TerminalEpisodeRecord, ...]:
    """Run one frozen baseline on exact v2 membership."""

    canonical = _canonical_pitch_suite(suite)
    return evaluate_baseline_suite(
        kind,
        _legacy_suite_for_rollout(canonical),
        cache=cache,
    )


def _canonical_pitch_suite(value: object) -> PitchEvaluationSuite:
    if not isinstance(value, PitchEvaluationSuite):
        raise ValueError("suite must be a PitchEvaluationSuite")
    canonical = fixed_pitch_evaluation_suite(value.suite_id)
    if value != canonical:
        raise ValueError("suite must match its canonical v2 identity and membership")
    return value


@dataclass(frozen=True, slots=True)
class PitchEvaluationRow:
    """One schema-v2 actor/suite/probe terminal row."""

    actor_id: str
    trainer: PitchTrainerKind | None
    seed: int | None
    environment_id: str
    observation_mode: ObservationMode
    suite_id: PitchEvaluationSuiteId
    suite_digest_sha256: str
    probe: str | None
    metrics: AggregateMetrics
    episodes: tuple[TerminalEpisodeRecord, ...]
    parameter_count: int | None = None
    training_examples: int | None = None
    training_wall_time_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.actor_id, str) or not self.actor_id:
            raise ValueError("actor_id must be a nonempty string")
        if self.trainer is None:
            try:
                kind = BaselineKind(self.actor_id)
            except ValueError as error:
                raise ValueError("baseline actor_id must name a declared baseline") from error
            if self.seed is not None:
                raise ValueError("baseline rows must not set a seed")
            if any(
                value is not None
                for value in (
                    self.parameter_count,
                    self.training_examples,
                    self.training_wall_time_seconds,
                )
            ):
                raise ValueError("baseline rows must not contain training metadata")
            if (
                self.environment_id != kind.environment_id
                or self.observation_mode is not kind.observation_mode
            ):
                raise ValueError("baseline row must retain its declared capability lane")
        else:
            if self.trainer is not PitchTrainerKind.PITCH:
                raise ValueError("learned rows must use the pitch trainer")
            if self.seed is None:
                raise ValueError("learned rows must set a seed")
            if any(
                value is None
                for value in (
                    self.parameter_count,
                    self.training_examples,
                    self.training_wall_time_seconds,
                )
            ):
                raise ValueError("learned rows require complete training metadata")
            if (
                self.environment_id != ENVIRONMENT_ID
                or self.observation_mode is not ObservationMode.SPECTRUM
            ):
                raise ValueError("learned rows must use the spectrum environment lane")
        seed = None if self.seed is None else _integer(self.seed, "seed", minimum=0)
        if not isinstance(self.suite_id, PitchEvaluationSuiteId):
            raise ValueError("suite_id must be a PitchEvaluationSuiteId")
        suite = fixed_pitch_evaluation_suite(self.suite_id)
        if self.suite_digest_sha256 != suite.digest_sha256:
            raise ValueError("suite_digest_sha256 must match the canonical pitch suite")
        if self.probe is not None:
            if self.probe not in _PITCH_PROBES:
                raise ValueError("probe must be zero_spectrum, shuffled_spectrum, or None")
            if self.suite_id not in (
                PitchEvaluationSuiteId.SMOKE,
                PitchEvaluationSuiteId.IID,
            ):
                raise ValueError("probe rows are allowed only on smoke or IID")
            if self.observation_mode is not ObservationMode.SPECTRUM:
                raise ValueError("probe rows require spectrum observations")
        if not isinstance(self.metrics, AggregateMetrics):
            raise ValueError("metrics must be AggregateMetrics")
        episodes = _typed_tuple(self.episodes, TerminalEpisodeRecord, "episodes")
        expected_membership = tuple(
            (
                index,
                EpisodeSpec(episode.target_note_index, episode.source_pitch_cents),
            )
            for index, episode in enumerate(suite.episodes)
        )
        actual_membership = tuple((record.episode_index, record.episode) for record in episodes)
        if actual_membership != expected_membership:
            raise ValueError("episodes must preserve the canonical ordered episode membership")
        _validate_excess_actions(episodes)
        if self.metrics != aggregate_episode_records(episodes):
            raise ValueError("metrics must be re-derived from terminal episode records")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "episodes", episodes)
        if self.trainer is not None:
            object.__setattr__(
                self,
                "parameter_count",
                _integer(self.parameter_count, "parameter_count", minimum=1),
            )
            object.__setattr__(
                self,
                "training_examples",
                _integer(self.training_examples, "training_examples", minimum=1),
            )
            object.__setattr__(
                self,
                "training_wall_time_seconds",
                _finite_float(
                    self.training_wall_time_seconds,
                    "training_wall_time_seconds",
                    minimum=0.0,
                ),
            )


def _validate_excess_actions(episodes: tuple[TerminalEpisodeRecord, ...]) -> None:
    """Re-derive the evaluator-owned efficiency field from raw episode identity."""

    for record in episodes:
        if not record.submitted_success:
            continue
        target_cents = 100 * (TARGET_MIN_COORDINATE + record.episode.target_note_index)
        optimal_action_count = len(
            minimum_action_plan(record.episode.source_pitch_cents - target_cents)
        )
        expected_excess = record.action_count - optimal_action_count
        if expected_excess < 0 or record.excess_actions != expected_excess:
            raise ValueError(
                "successful excess_actions must be re-derived from episode and action_count"
            )


def build_pitch_evaluation_row(
    *,
    actor_id: str,
    trainer: PitchTrainerKind | None,
    seed: int | None,
    environment_id: str,
    observation_mode: ObservationMode,
    suite: PitchEvaluationSuite,
    records: Sequence[TerminalEpisodeRecord],
    probe: str | None = None,
    parameter_count: int | None = None,
    training_examples: int | None = None,
    training_wall_time_seconds: float | None = None,
) -> PitchEvaluationRow:
    """Build one exact schema-v2 row from ordered raw terminal records."""

    canonical = _canonical_pitch_suite(suite)
    normalized = _typed_tuple(records, TerminalEpisodeRecord, "records")
    return PitchEvaluationRow(
        actor_id=actor_id,
        trainer=trainer,
        seed=seed,
        environment_id=environment_id,
        observation_mode=observation_mode,
        suite_id=canonical.suite_id,
        suite_digest_sha256=canonical.digest_sha256,
        probe=probe,
        metrics=aggregate_episode_records(normalized),
        episodes=normalized,
        parameter_count=parameter_count,
        training_examples=training_examples,
        training_wall_time_seconds=training_wall_time_seconds,
    )


def _row_actor_identity(row: PitchEvaluationRow) -> tuple[object, ...]:
    return (
        row.actor_id,
        row.trainer,
        row.seed,
        row.environment_id,
        row.observation_mode,
        row.parameter_count,
        row.training_examples,
        row.training_wall_time_seconds,
    )


@dataclass(frozen=True, slots=True)
class RegisterOODAggregate:
    """Typed two-register terminal aggregate with no synthetic suite identity."""

    actor_id: str
    trainer: PitchTrainerKind | None
    seed: int | None
    lower_suite_id: PitchEvaluationSuiteId
    lower_suite_digest_sha256: str
    upper_suite_id: PitchEvaluationSuiteId
    upper_suite_digest_sha256: str
    metrics: AggregateMetrics
    episodes: tuple[TerminalEpisodeRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.actor_id, str) or not self.actor_id:
            raise ValueError("actor_id must be a nonempty string")
        if self.trainer is None:
            try:
                BaselineKind(self.actor_id)
            except ValueError as error:
                raise ValueError("baseline aggregate actor_id must name a baseline") from error
            if self.seed is not None:
                raise ValueError("baseline aggregates must not set a seed")
        elif self.trainer is PitchTrainerKind.PITCH:
            object.__setattr__(self, "seed", _integer(self.seed, "seed", minimum=0))
        else:
            raise ValueError("aggregate trainer must be pitch or None")
        if self.lower_suite_id is not PitchEvaluationSuiteId.OOD_LOWER or (
            self.upper_suite_id is not PitchEvaluationSuiteId.OOD_UPPER
        ):
            raise ValueError("aggregate requires separate lower and upper suite identities")
        lower = fixed_pitch_evaluation_suite(PitchEvaluationSuiteId.OOD_LOWER)
        upper = fixed_pitch_evaluation_suite(PitchEvaluationSuiteId.OOD_UPPER)
        if (
            self.lower_suite_digest_sha256 != lower.digest_sha256
            or self.upper_suite_digest_sha256 != upper.digest_sha256
        ):
            raise ValueError("aggregate suite digests must match lower and upper suites")
        episodes = _typed_tuple(self.episodes, TerminalEpisodeRecord, "episodes")
        expected_membership = tuple(
            (
                index,
                EpisodeSpec(episode.target_note_index, episode.source_pitch_cents),
            )
            for suite in (lower, upper)
            for index, episode in enumerate(suite.episodes)
        )
        actual_membership = tuple((record.episode_index, record.episode) for record in episodes)
        if actual_membership != expected_membership:
            raise ValueError("aggregate episodes must concatenate lower and upper membership")
        if self.metrics != aggregate_episode_records(episodes):
            raise ValueError("aggregate metrics must be re-derived from both registers")
        object.__setattr__(self, "episodes", episodes)

    @classmethod
    def from_rows(
        cls,
        lower: PitchEvaluationRow,
        upper: PitchEvaluationRow,
    ) -> RegisterOODAggregate:
        """Build the only valid lower-plus-upper aggregate."""

        if not isinstance(lower, PitchEvaluationRow) or not isinstance(upper, PitchEvaluationRow):
            raise ValueError("lower and upper must be PitchEvaluationRow values")
        if (
            lower.suite_id is not PitchEvaluationSuiteId.OOD_LOWER
            or upper.suite_id is not PitchEvaluationSuiteId.OOD_UPPER
            or lower.probe is not None
            or upper.probe is not None
        ):
            raise ValueError("rows must be separate unperturbed lower and upper suites")
        if _row_actor_identity(lower) != _row_actor_identity(upper):
            raise ValueError("lower and upper rows must describe the same actor")
        episodes = (*lower.episodes, *upper.episodes)
        return cls(
            actor_id=lower.actor_id,
            trainer=lower.trainer,
            seed=lower.seed,
            lower_suite_id=lower.suite_id,
            lower_suite_digest_sha256=lower.suite_digest_sha256,
            upper_suite_id=upper.suite_id,
            upper_suite_digest_sha256=upper.suite_digest_sha256,
            metrics=aggregate_episode_records(episodes),
            episodes=episodes,
        )


def validate_pitch_checkpoint_row_matrix(
    rows: Sequence[PitchEvaluationRow],
) -> tuple[PitchEvaluationRow, ...]:
    """Validate the exact 15 learned plus 12 baseline checkpoint row order."""

    normalized = _typed_tuple(rows, PitchEvaluationRow, "rows")
    expected_lanes: list[tuple[object, ...]] = []
    for seed in (0, 1, 2):
        expected_lanes.extend(
            (
                (PitchTrainerKind.PITCH, seed, PitchEvaluationSuiteId.IID, None),
                (PitchTrainerKind.PITCH, seed, PitchEvaluationSuiteId.OOD_LOWER, None),
                (PitchTrainerKind.PITCH, seed, PitchEvaluationSuiteId.OOD_UPPER, None),
                (
                    PitchTrainerKind.PITCH,
                    seed,
                    PitchEvaluationSuiteId.IID,
                    PITCH_ZERO_SPECTRUM_PROBE,
                ),
                (
                    PitchTrainerKind.PITCH,
                    seed,
                    PitchEvaluationSuiteId.IID,
                    PITCH_SHUFFLED_SPECTRUM_PROBE,
                ),
            )
        )
    for _kind in _BASELINE_ORDER:
        expected_lanes.extend(
            (None, None, suite_id, None)
            for suite_id in (
                PitchEvaluationSuiteId.IID,
                PitchEvaluationSuiteId.OOD_LOWER,
                PitchEvaluationSuiteId.OOD_UPPER,
            )
        )
    actual_lanes = tuple((row.trainer, row.seed, row.suite_id, row.probe) for row in normalized)
    if actual_lanes != tuple(expected_lanes):
        raise ValueError("rows must match the exact checkpoint row matrix")
    for seed, start in enumerate((0, 5, 10)):
        if len({_row_actor_identity(row) for row in normalized[start : start + 5]}) != 1:
            raise ValueError("rows must match the exact checkpoint row matrix")
        if normalized[start].seed != seed:
            raise ValueError("rows must match the exact checkpoint row matrix")
    baseline_rows = normalized[15:]
    for index, kind in enumerate(_BASELINE_ORDER):
        group = baseline_rows[index * 3 : index * 3 + 3]
        if any(row.actor_id != kind.value for row in group):
            raise ValueError("rows must match the exact checkpoint row matrix")
    return normalized


def validate_pitch_smoke_row_matrix(
    rows: Sequence[PitchEvaluationRow],
) -> tuple[PitchEvaluationRow, ...]:
    """Validate one learned base/two probes followed by four smoke baselines."""

    normalized = _typed_tuple(rows, PitchEvaluationRow, "rows")
    if len(normalized) != 7:
        raise ValueError("rows must match the exact smoke row matrix")
    seed = normalized[0].seed
    expected = (
        (PitchTrainerKind.PITCH, seed, None),
        (PitchTrainerKind.PITCH, seed, PITCH_ZERO_SPECTRUM_PROBE),
        (PitchTrainerKind.PITCH, seed, PITCH_SHUFFLED_SPECTRUM_PROBE),
        *((None, None, None) for _ in _BASELINE_ORDER),
    )
    actual = tuple((row.trainer, row.seed, row.probe) for row in normalized)
    if actual != expected or any(
        row.suite_id is not PitchEvaluationSuiteId.SMOKE for row in normalized
    ):
        raise ValueError("rows must match the exact smoke row matrix")
    if len({_row_actor_identity(row) for row in normalized[:3]}) != 1:
        raise ValueError("rows must match the exact smoke row matrix")
    if tuple(row.actor_id for row in normalized[3:]) != tuple(
        kind.value for kind in _BASELINE_ORDER
    ):
        raise ValueError("rows must match the exact smoke row matrix")
    return normalized


@dataclass(frozen=True, slots=True)
class PitchSeedCriterion:
    """Every raw-record-derived threshold input for one checkpoint seed."""

    seed: int
    iid_coordinate_within_five_rate: float
    ood_coordinate_within_five_rate: float
    ood_lower_coordinate_within_five_rate: float
    ood_upper_coordinate_within_five_rate: float
    iid_submitted_success_rate: float
    iid_bound_blocked_actions: int
    iid_truncations: int
    iid_zero_submitted_success_rate: float
    iid_shuffled_submitted_success_rate: float
    register_ood_submitted_success_rate: float
    iid_mean_successful_excess_actions: float | None
    _evidence_token: InitVar[object] = None

    def __post_init__(self, _evidence_token: object) -> None:
        if _evidence_token is not _CRITERION_EVIDENCE_TOKEN:
            raise ValueError("PitchSeedCriterion must be derived from raw evaluation evidence")
        object.__setattr__(self, "seed", _integer(self.seed, "seed", minimum=0, maximum=2))
        for field in (
            "iid_coordinate_within_five_rate",
            "ood_coordinate_within_five_rate",
            "ood_lower_coordinate_within_five_rate",
            "ood_upper_coordinate_within_five_rate",
            "iid_submitted_success_rate",
            "iid_zero_submitted_success_rate",
            "iid_shuffled_submitted_success_rate",
            "register_ood_submitted_success_rate",
        ):
            object.__setattr__(
                self,
                field,
                _finite_float(getattr(self, field), field, minimum=0.0, maximum=1.0),
            )
        object.__setattr__(
            self,
            "iid_bound_blocked_actions",
            _integer(self.iid_bound_blocked_actions, "iid_bound_blocked_actions", minimum=0),
        )
        object.__setattr__(
            self,
            "iid_truncations",
            _integer(self.iid_truncations, "iid_truncations", minimum=0),
        )
        object.__setattr__(
            self,
            "iid_mean_successful_excess_actions",
            _optional_float(
                self.iid_mean_successful_excess_actions,
                "iid_mean_successful_excess_actions",
            ),
        )


@dataclass(frozen=True, slots=True)
class PitchScientificCriterion:
    """The complete three-seed Milestone E scientific verdict."""

    eligible: bool
    seeds: tuple[PitchSeedCriterion, ...]
    median_register_ood_submitted_success_rate: float | None
    failed_gates: tuple[str, ...]
    criterion_met: bool | None
    status: str
    _evidence_token: InitVar[object] = None

    def __post_init__(self, _evidence_token: object) -> None:
        if _evidence_token is not _CRITERION_EVIDENCE_TOKEN:
            raise ValueError(
                "PitchScientificCriterion must be derived from raw evaluation evidence"
            )
        if not isinstance(self.eligible, bool):
            raise ValueError("eligible must be a bool")
        seeds = _typed_tuple(
            self.seeds,
            PitchSeedCriterion,
            "seeds",
            nonempty=self.eligible,
        )
        try:
            failed_gates = tuple(self.failed_gates)
        except TypeError as error:
            raise ValueError("failed_gates must contain unique nonempty strings") from error
        if any(not isinstance(gate, str) or not gate for gate in failed_gates) or len(
            set(failed_gates)
        ) != len(failed_gates):
            raise ValueError("failed_gates must contain unique nonempty strings")
        if self.eligible:
            if tuple(item.seed for item in seeds) != (0, 1, 2):
                raise ValueError("eligible criterion seeds must be exactly 0, 1, and 2")
            median = _finite_float(
                self.median_register_ood_submitted_success_rate,
                "median_register_ood_submitted_success_rate",
                minimum=0.0,
                maximum=1.0,
            )
            if not isinstance(self.criterion_met, bool):
                raise ValueError("eligible criterion requires a bool criterion_met")
            expected_status = "criterion_met" if self.criterion_met else "criterion_not_met"
            if self.status != expected_status:
                raise ValueError(f"eligible criterion status must be {expected_status}")
            expected_median = float(
                statistics.median(seed.register_ood_submitted_success_rate for seed in seeds)
            )
            if median != expected_median:
                raise ValueError(
                    "median_register_ood_submitted_success_rate must be re-derived from seeds"
                )
            expected_failed_gates = _pitch_failed_gates(seeds, median)
            if failed_gates != expected_failed_gates:
                raise ValueError("failed_gates must be re-derived from every seed threshold")
            if self.criterion_met != (not expected_failed_gates):
                raise ValueError("criterion_met must match the complete failed gate inventory")
            object.__setattr__(
                self,
                "median_register_ood_submitted_success_rate",
                median,
            )
        elif (
            seeds
            or self.median_register_ood_submitted_success_rate is not None
            or failed_gates
            or self.criterion_met is not None
            or self.status != "ineligible"
        ):
            raise ValueError("ineligible criterion must not contain computed evidence")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "failed_gates", failed_gates)


def _pitch_failed_gates(
    seeds: tuple[PitchSeedCriterion, ...],
    median_ood: float,
) -> tuple[str, ...]:
    gates = (
        (
            "iid_coordinate_within_five",
            all(seed.iid_coordinate_within_five_rate >= 0.99 for seed in seeds),
        ),
        (
            "ood_coordinate_within_five",
            all(seed.ood_coordinate_within_five_rate >= 0.95 for seed in seeds),
        ),
        (
            "ood_lower_coordinate_within_five",
            all(seed.ood_lower_coordinate_within_five_rate >= 0.90 for seed in seeds),
        ),
        (
            "ood_upper_coordinate_within_five",
            all(seed.ood_upper_coordinate_within_five_rate >= 0.90 for seed in seeds),
        ),
        (
            "iid_submitted_success",
            all(seed.iid_submitted_success_rate >= 0.95 for seed in seeds),
        ),
        (
            "iid_bound_blocked",
            all(seed.iid_bound_blocked_actions == 0 for seed in seeds),
        ),
        (
            "iid_truncations",
            all(seed.iid_truncations == 0 for seed in seeds),
        ),
        (
            "probe_margin",
            all(
                seed.iid_submitted_success_rate - seed.iid_zero_submitted_success_rate >= 0.50
                and seed.iid_submitted_success_rate - seed.iid_shuffled_submitted_success_rate
                >= 0.50
                for seed in seeds
            ),
        ),
        (
            "register_ood_submitted_success",
            median_ood >= 0.90
            and all(seed.register_ood_submitted_success_rate >= 0.80 for seed in seeds),
        ),
        (
            "iid_mean_successful_excess_actions",
            all(
                seed.iid_mean_successful_excess_actions is not None
                and seed.iid_mean_successful_excess_actions <= 4.0
                for seed in seeds
            ),
        ),
    )
    return tuple(name for name, passed in gates if not passed)


def evaluate_pitch_criterion(
    *,
    coordinate_evaluations: Sequence[PitchCoordinateEvaluation],
    terminal_rows: Sequence[PitchEvaluationRow],
    eligible: bool,
) -> PitchScientificCriterion:
    """Recompute all preregistered gates from exact raw coordinate/terminal evidence."""

    if not isinstance(eligible, bool):
        raise ValueError("eligible must be a bool")
    if not eligible:
        return PitchScientificCriterion(
            False,
            (),
            None,
            (),
            None,
            "ineligible",
            _evidence_token=_CRITERION_EVIDENCE_TOKEN,
        )
    evaluations = _typed_tuple(
        coordinate_evaluations,
        PitchCoordinateEvaluation,
        "coordinate_evaluations",
    )
    if tuple(item.seed for item in evaluations) != (0, 1, 2):
        raise ValueError("coordinate_evaluations must contain exact seeds 0, 1, and 2")
    rows = validate_pitch_checkpoint_row_matrix(terminal_rows)
    seeds: list[PitchSeedCriterion] = []
    for seed, evaluation in enumerate(evaluations):
        seed_rows = rows[seed * 5 : seed * 5 + 5]
        iid, lower, upper, zero, shuffled = seed_rows
        register_aggregate = RegisterOODAggregate.from_rows(lower, upper)
        lower_coordinates = tuple(
            record
            for record in evaluation.records
            if record.partition is PitchCoordinatePartition.OOD_LOWER
        )
        upper_coordinates = tuple(
            record
            for record in evaluation.records
            if record.partition is PitchCoordinatePartition.OOD_UPPER
        )
        ood_metrics = summarize_pitch_coordinate_records((*lower_coordinates, *upper_coordinates))
        iid_metrics = evaluation.register_metrics(PitchCoordinatePartition.IID)
        lower_metrics = evaluation.register_metrics(PitchCoordinatePartition.OOD_LOWER)
        upper_metrics = evaluation.register_metrics(PitchCoordinatePartition.OOD_UPPER)
        seeds.append(
            PitchSeedCriterion(
                seed=seed,
                iid_coordinate_within_five_rate=iid_metrics.within_five_rate,
                ood_coordinate_within_five_rate=ood_metrics.within_five_rate,
                ood_lower_coordinate_within_five_rate=lower_metrics.within_five_rate,
                ood_upper_coordinate_within_five_rate=upper_metrics.within_five_rate,
                iid_submitted_success_rate=iid.metrics.submitted_success_rate,
                iid_bound_blocked_actions=sum(
                    record.invalid_action_count for record in iid.episodes
                ),
                iid_truncations=sum(
                    record.terminal_reason is TerminalReason.BUDGET_EXHAUSTED
                    for record in iid.episodes
                ),
                iid_zero_submitted_success_rate=zero.metrics.submitted_success_rate,
                iid_shuffled_submitted_success_rate=shuffled.metrics.submitted_success_rate,
                register_ood_submitted_success_rate=(
                    register_aggregate.metrics.submitted_success_rate
                ),
                iid_mean_successful_excess_actions=(iid.metrics.mean_successful_excess_actions),
                _evidence_token=_CRITERION_EVIDENCE_TOKEN,
            )
        )
    normalized_seeds = tuple(seeds)
    median_ood = float(
        statistics.median(seed.register_ood_submitted_success_rate for seed in normalized_seeds)
    )
    failed = _pitch_failed_gates(normalized_seeds, median_ood)
    met = not failed
    return PitchScientificCriterion(
        eligible=True,
        seeds=normalized_seeds,
        median_register_ood_submitted_success_rate=median_ood,
        failed_gates=failed,
        criterion_met=met,
        status="criterion_met" if met else "criterion_not_met",
        _evidence_token=_CRITERION_EVIDENCE_TOKEN,
    )


def default_pitch_evidence_provider(
    cache: SpectrumEvidenceCache | None = None,
) -> SpectrumEvidenceProvider:
    """Build the real renderer/cache seam for final direct evaluation."""

    if cache is not None and not isinstance(cache, SpectrumEvidenceCache):
        raise ValueError("cache must be a SpectrumEvidenceCache")
    return SpectrumEvidenceProvider(SpectrumEvidenceCache() if cache is None else cache)


__all__ = [
    "PITCH_SHUFFLED_SPECTRUM_PROBE",
    "PITCH_SPECTRUM_SHUFFLE_ID",
    "PITCH_ZERO_SPECTRUM_PROBE",
    "PitchCoordinateEvaluation",
    "PitchCoordinateMetrics",
    "PitchEvaluationRow",
    "PitchRegisterRow",
    "PitchResidueRow",
    "PitchScientificCriterion",
    "PitchSeedCriterion",
    "RegisterOODAggregate",
    "build_pitch_evaluation_row",
    "default_pitch_evidence_provider",
    "evaluate_final_pitch_coordinates",
    "evaluate_pitch_baseline_suite",
    "evaluate_pitch_criterion",
    "evaluate_pitch_learned_actor",
    "make_pitch_model_grid_predictor",
    "make_pitch_spectrum_probe_factory",
    "summarize_pitch_coordinate_records",
    "validate_pitch_checkpoint_row_matrix",
    "validate_pitch_smoke_row_matrix",
]
