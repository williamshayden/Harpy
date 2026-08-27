"""Strict canonical schema-v2 evaluation reports for the pitch estimator."""

from __future__ import annotations

import hashlib
import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from harpy.envs.models import ObservationMode, PitchAction
from harpy.learning.artifacts import (
    PITCH_ARTIFACT_SCHEMA_VERSION,
    canonical_json_bytes,
)
from harpy.learning.evaluation import (
    aggregate_metrics_from_document,
    aggregate_metrics_to_document,
    terminal_episode_record_from_document,
    terminal_episode_record_to_document,
)
from harpy.learning.models import (
    ENVIRONMENT_CONTRACT_ID,
    ENVIRONMENT_ID,
    SPECTRUM_GRID_ID,
    DeviceName,
    JSONValue,
    PitchCoordinatePartition,
    PitchCoordinateRecord,
    ProfileName,
)
from harpy.learning.pitch_data import (
    PITCH_DISTRIBUTION_ID,
    PITCH_SPLIT_DIGEST_SHA256,
    PitchEvaluationSuiteId,
    PitchTrainerKind,
)
from harpy.learning.pitch_evaluation import (
    PitchCoordinateEvaluation,
    PitchCoordinateMetrics,
    PitchEvaluationRow,
    PitchRegisterRow,
    PitchResidueRow,
    PitchScientificCriterion,
    RegisterOODAggregate,
    evaluate_pitch_criterion,
    validate_pitch_checkpoint_row_matrix,
    validate_pitch_smoke_row_matrix,
)

PITCH_PREPROCESSING_SCHEMA_ID = "harpy-sine-pitch-spectrum-float32-v1"
PITCH_POLICY_SEMANTICS_ID = "harpy-sine-pitch-estimator-planner-v1"
PITCH_ESTIMATOR_ARCHITECTURE_ID = "harpy-sine-pitch-estimator-conv-v1"
PITCH_ESTIMATOR_PARAMETER_COUNT = 2_497
_PITCH_ACTION_SCHEMA_ID = "harpy-sine-pitch-actions-v1"
_PITCH_REQUIRED_DEVICE = "cpu"
_PITCH_PROFILE_CONFIGS = MappingProxyType(
    {
        ProfileName.SMOKE: MappingProxyType(
            {
                "training_coordinate_count": 256,
                "validation_coordinate_count": 64,
                "max_epochs": 2,
                "early_stopping_patience": None,
                "batch_size": 64,
                "optimizer": "AdamW",
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "eligible_for_aggregate": False,
            }
        ),
        ProfileName.CHECKPOINT: MappingProxyType(
            {
                "training_coordinate_count": 1_400,
                "validation_coordinate_count": 200,
                "max_epochs": 50,
                "early_stopping_patience": 8,
                "batch_size": 64,
                "optimizer": "AdamW",
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "eligible_for_aggregate": True,
            }
        ),
    }
)
_PITCH_EVALUATION_SUITE_RECORDS = (
    (
        PitchEvaluationSuiteId.SMOKE.value,
        "32a2761392583677d0e0785d27965a8d139a5475117088beecb413bd1a3ca611",
    ),
    (
        PitchEvaluationSuiteId.IID.value,
        "5b91c98d269a7e9b7319e8827b9eea74a89ab76cabf4a0478746ec3601ee73f3",
    ),
    (
        PitchEvaluationSuiteId.OOD_LOWER.value,
        "c8a04e8270e5aa54c17731cd522e0e02b7fa80fa745e5a4c6802480891e73340",
    ),
    (
        PitchEvaluationSuiteId.OOD_UPPER.value,
        "1faa402fc3fdebf6a17c758a0647455c8bc6fab129613b919faf6084d0b4da0d",
    ),
)


def pitch_report_compatibility_sha256(profile: ProfileName) -> str:
    """Re-derive the schema-v2 compatibility identity without training imports."""

    if not isinstance(profile, ProfileName):
        raise ValueError("profile must be a ProfileName")
    configured = _PITCH_PROFILE_CONFIGS[profile]
    document = {
        "schema_version": PITCH_ARTIFACT_SCHEMA_VERSION,
        "trainer": PitchTrainerKind.PITCH.value,
        "profile": profile.value,
        "profile_config": configured,
        "environment_id": ENVIRONMENT_ID,
        "environment_contract_id": ENVIRONMENT_CONTRACT_ID,
        "spectrum_grid_id": SPECTRUM_GRID_ID,
        "action_schema_id": _PITCH_ACTION_SCHEMA_ID,
        "action_order": [{"index": int(action), "name": action.name} for action in PitchAction],
        "architecture_schema_id": PITCH_ESTIMATOR_ARCHITECTURE_ID,
        "preprocessing_schema_id": PITCH_PREPROCESSING_SCHEMA_ID,
        "policy_semantics_id": PITCH_POLICY_SEMANTICS_ID,
        "train_distribution_id": PITCH_DISTRIBUTION_ID,
        "coordinate_split_digest_sha256": PITCH_SPLIT_DIGEST_SHA256,
        "training_coordinate_count": configured["training_coordinate_count"],
        "validation_coordinate_count": configured["validation_coordinate_count"],
        "evaluation_suites": [
            {"suite_id": suite_id, "digest_sha256": digest}
            for suite_id, digest in _PITCH_EVALUATION_SUITE_RECORDS
        ],
        "required_training_device": _PITCH_REQUIRED_DEVICE,
        "required_evaluation_device": _PITCH_REQUIRED_DEVICE,
        "data_loader_num_workers": 0,
        "shuffle_generator_device": "cpu",
    }
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


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


def _optional_integer(value: object, field: str) -> int | None:
    return None if value is None else _integer(value, field, minimum=0)


def _finite_float(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be a finite number")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return normalized


def _optional_float(value: object, field: str) -> float | None:
    return None if value is None else _finite_float(value, field, minimum=0.0)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a string-keyed mapping")
    return value


def _exact_fields(mapping: Mapping[str, object], fields: set[str], field: str) -> None:
    if set(mapping) != fields:
        raise ValueError(f"{field} fields must be exactly {sorted(fields)}")


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _enum[EnumT](enum_type: type[EnumT], value: object, field: str) -> EnumT:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a valid {enum_type.__name__}")
    try:
        return enum_type(value)  # type: ignore[call-arg]
    except ValueError as error:
        raise ValueError(f"{field} must be a valid {enum_type.__name__}") from error


def _list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _typed_tuple[T](value: object, expected_type: type[T], field: str) -> tuple[T, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must contain {expected_type.__name__} values")
    try:
        normalized = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{field} must contain {expected_type.__name__} values") from error
    if not all(isinstance(item, expected_type) for item in normalized):
        raise ValueError(f"{field} must contain {expected_type.__name__} values")
    return normalized


def pitch_evaluation_row_to_document(row: PitchEvaluationRow) -> dict[str, JSONValue]:
    """Encode one fully re-derived terminal row."""

    if not isinstance(row, PitchEvaluationRow):
        raise ValueError("row must be a PitchEvaluationRow")
    return {
        "actor_id": row.actor_id,
        "trainer": None if row.trainer is None else row.trainer.value,
        "seed": row.seed,
        "environment_id": row.environment_id,
        "observation_mode": row.observation_mode.value,
        "suite_id": row.suite_id.value,
        "suite_digest_sha256": row.suite_digest_sha256,
        "probe": row.probe,
        "parameter_count": row.parameter_count,
        "training_examples": row.training_examples,
        "training_wall_time_seconds": row.training_wall_time_seconds,
        "metrics": aggregate_metrics_to_document(row.metrics),
        "episodes": [terminal_episode_record_to_document(record) for record in row.episodes],
    }


def pitch_evaluation_row_from_document(value: object) -> PitchEvaluationRow:
    """Decode one terminal row through Task 5's raw-evidence constructor."""

    mapping = _mapping(value, "pitch evaluation row")
    _exact_fields(
        mapping,
        {
            "actor_id",
            "trainer",
            "seed",
            "environment_id",
            "observation_mode",
            "suite_id",
            "suite_digest_sha256",
            "probe",
            "parameter_count",
            "training_examples",
            "training_wall_time_seconds",
            "metrics",
            "episodes",
        },
        "pitch evaluation row",
    )
    trainer = (
        None
        if mapping["trainer"] is None
        else _enum(PitchTrainerKind, mapping["trainer"], "trainer")
    )
    return PitchEvaluationRow(
        actor_id=_string(mapping["actor_id"], "actor_id"),
        trainer=trainer,
        seed=_optional_integer(mapping["seed"], "seed"),
        environment_id=_string(mapping["environment_id"], "environment_id"),
        observation_mode=_enum(
            ObservationMode,
            mapping["observation_mode"],
            "observation_mode",
        ),
        suite_id=_enum(PitchEvaluationSuiteId, mapping["suite_id"], "suite_id"),
        suite_digest_sha256=_string(
            mapping["suite_digest_sha256"],
            "suite_digest_sha256",
        ),
        probe=None if mapping["probe"] is None else _string(mapping["probe"], "probe"),
        metrics=aggregate_metrics_from_document(mapping["metrics"]),
        episodes=tuple(
            terminal_episode_record_from_document(item)
            for item in _list(mapping["episodes"], "episodes")
        ),
        parameter_count=_optional_integer(mapping["parameter_count"], "parameter_count"),
        training_examples=_optional_integer(
            mapping["training_examples"],
            "training_examples",
        ),
        training_wall_time_seconds=_optional_float(
            mapping["training_wall_time_seconds"],
            "training_wall_time_seconds",
        ),
    )


_COORDINATE_METRIC_FIELDS = (
    "count",
    "within_one_count",
    "within_one_rate",
    "within_five_count",
    "within_five_rate",
    "mean_absolute_error_cents",
    "p50_cents",
    "p90_cents",
    "p95_cents",
    "p99_cents",
    "max_cents",
)


def _coordinate_metrics_to_document(metrics: PitchCoordinateMetrics) -> dict[str, JSONValue]:
    if not isinstance(metrics, PitchCoordinateMetrics):
        raise ValueError("metrics must be PitchCoordinateMetrics")
    return {field: getattr(metrics, field) for field in _COORDINATE_METRIC_FIELDS}


def _coordinate_metrics_from_document(value: object) -> PitchCoordinateMetrics:
    mapping = _mapping(value, "pitch coordinate metrics")
    _exact_fields(mapping, set(_COORDINATE_METRIC_FIELDS), "pitch coordinate metrics")
    return PitchCoordinateMetrics(
        count=_integer(mapping["count"], "count", minimum=1),
        within_one_count=_integer(mapping["within_one_count"], "within_one_count"),
        within_one_rate=_finite_float(mapping["within_one_rate"], "within_one_rate"),
        within_five_count=_integer(mapping["within_five_count"], "within_five_count"),
        within_five_rate=_finite_float(mapping["within_five_rate"], "within_five_rate"),
        mean_absolute_error_cents=_finite_float(
            mapping["mean_absolute_error_cents"],
            "mean_absolute_error_cents",
        ),
        p50_cents=_integer(mapping["p50_cents"], "p50_cents"),
        p90_cents=_integer(mapping["p90_cents"], "p90_cents"),
        p95_cents=_integer(mapping["p95_cents"], "p95_cents"),
        p99_cents=_integer(mapping["p99_cents"], "p99_cents"),
        max_cents=_integer(mapping["max_cents"], "max_cents"),
    )


def _coordinate_record_to_document(record: PitchCoordinateRecord) -> dict[str, JSONValue]:
    if not isinstance(record, PitchCoordinateRecord):
        raise ValueError("record must be a PitchCoordinateRecord")
    return {
        "seed": record.seed,
        "distribution_id": record.distribution_id,
        "split_digest_sha256": record.split_digest_sha256,
        "partition": record.partition.value,
        "true_coordinate_cents": record.true_coordinate_cents,
        "predicted_grid_index": record.predicted_grid_index,
        "predicted_cents": record.predicted_cents,
        "signed_error_cents": record.signed_error_cents,
        "absolute_error_cents": record.absolute_error_cents,
    }


def _coordinate_record_from_document(value: object) -> PitchCoordinateRecord:
    mapping = _mapping(value, "pitch coordinate record")
    fields = {
        "seed",
        "distribution_id",
        "split_digest_sha256",
        "partition",
        "true_coordinate_cents",
        "predicted_grid_index",
        "predicted_cents",
        "signed_error_cents",
        "absolute_error_cents",
    }
    _exact_fields(mapping, fields, "pitch coordinate record")
    return PitchCoordinateRecord(
        seed=_integer(mapping["seed"], "seed"),
        distribution_id=_string(mapping["distribution_id"], "distribution_id"),
        split_digest_sha256=_string(
            mapping["split_digest_sha256"],
            "split_digest_sha256",
        ),
        partition=_enum(PitchCoordinatePartition, mapping["partition"], "partition"),
        true_coordinate_cents=_integer(
            mapping["true_coordinate_cents"],
            "true_coordinate_cents",
        ),
        predicted_grid_index=_integer(
            mapping["predicted_grid_index"],
            "predicted_grid_index",
        ),
        predicted_cents=_integer(mapping["predicted_cents"], "predicted_cents"),
        signed_error_cents=_integer(
            mapping["signed_error_cents"],
            "signed_error_cents",
        ),
        absolute_error_cents=_integer(
            mapping["absolute_error_cents"],
            "absolute_error_cents",
        ),
    )


def pitch_coordinate_evaluation_to_document(
    evaluation: PitchCoordinateEvaluation,
) -> dict[str, JSONValue]:
    """Encode all raw direct predictions and both derived breakdown tables."""

    if not isinstance(evaluation, PitchCoordinateEvaluation):
        raise ValueError("evaluation must be a PitchCoordinateEvaluation")
    return {
        "seed": evaluation.seed,
        "records": [_coordinate_record_to_document(record) for record in evaluation.records],
        "register_rows": [
            {
                "partition": row.partition.value,
                "metrics": _coordinate_metrics_to_document(row.metrics),
            }
            for row in evaluation.register_rows
        ],
        "residue_rows": [
            {
                "residue": row.residue,
                "metrics": _coordinate_metrics_to_document(row.metrics),
            }
            for row in evaluation.residue_rows
        ],
    }


def pitch_coordinate_evaluation_from_document(value: object) -> PitchCoordinateEvaluation:
    """Decode direct predictions and re-derive every aggregate table."""

    mapping = _mapping(value, "pitch coordinate evaluation")
    _exact_fields(
        mapping,
        {"seed", "records", "register_rows", "residue_rows"},
        "pitch coordinate evaluation",
    )
    register_rows = []
    for item in _list(mapping["register_rows"], "register_rows"):
        row = _mapping(item, "pitch register row")
        _exact_fields(row, {"partition", "metrics"}, "pitch register row")
        register_rows.append(
            PitchRegisterRow(
                partition=_enum(
                    PitchCoordinatePartition,
                    row["partition"],
                    "partition",
                ),
                metrics=_coordinate_metrics_from_document(row["metrics"]),
            )
        )
    residue_rows = []
    for item in _list(mapping["residue_rows"], "residue_rows"):
        row = _mapping(item, "pitch residue row")
        _exact_fields(row, {"residue", "metrics"}, "pitch residue row")
        residue_rows.append(
            PitchResidueRow(
                residue=_integer(row["residue"], "residue"),
                metrics=_coordinate_metrics_from_document(row["metrics"]),
            )
        )
    return PitchCoordinateEvaluation(
        seed=_integer(mapping["seed"], "seed"),
        records=tuple(
            _coordinate_record_from_document(item) for item in _list(mapping["records"], "records")
        ),
        register_rows=tuple(register_rows),
        residue_rows=tuple(residue_rows),
    )


def register_ood_aggregate_to_document(
    aggregate: RegisterOODAggregate,
) -> dict[str, JSONValue]:
    """Encode one exact typed lower-plus-upper aggregate."""

    if not isinstance(aggregate, RegisterOODAggregate):
        raise ValueError("aggregate must be a RegisterOODAggregate")
    return {
        "actor_id": aggregate.actor_id,
        "trainer": None if aggregate.trainer is None else aggregate.trainer.value,
        "seed": aggregate.seed,
        "lower_suite_id": aggregate.lower_suite_id.value,
        "lower_suite_digest_sha256": aggregate.lower_suite_digest_sha256,
        "upper_suite_id": aggregate.upper_suite_id.value,
        "upper_suite_digest_sha256": aggregate.upper_suite_digest_sha256,
        "metrics": aggregate_metrics_to_document(aggregate.metrics),
        "episodes": [terminal_episode_record_to_document(record) for record in aggregate.episodes],
    }


def register_ood_aggregate_from_document(value: object) -> RegisterOODAggregate:
    """Decode and re-derive one lower-plus-upper aggregate from raw episodes."""

    mapping = _mapping(value, "register OOD aggregate")
    _exact_fields(
        mapping,
        {
            "actor_id",
            "trainer",
            "seed",
            "lower_suite_id",
            "lower_suite_digest_sha256",
            "upper_suite_id",
            "upper_suite_digest_sha256",
            "metrics",
            "episodes",
        },
        "register OOD aggregate",
    )
    trainer = (
        None
        if mapping["trainer"] is None
        else _enum(PitchTrainerKind, mapping["trainer"], "trainer")
    )
    return RegisterOODAggregate(
        actor_id=_string(mapping["actor_id"], "actor_id"),
        trainer=trainer,
        seed=_optional_integer(mapping["seed"], "seed"),
        lower_suite_id=_enum(
            PitchEvaluationSuiteId,
            mapping["lower_suite_id"],
            "lower_suite_id",
        ),
        lower_suite_digest_sha256=_string(
            mapping["lower_suite_digest_sha256"],
            "lower_suite_digest_sha256",
        ),
        upper_suite_id=_enum(
            PitchEvaluationSuiteId,
            mapping["upper_suite_id"],
            "upper_suite_id",
        ),
        upper_suite_digest_sha256=_string(
            mapping["upper_suite_digest_sha256"],
            "upper_suite_digest_sha256",
        ),
        metrics=aggregate_metrics_from_document(mapping["metrics"]),
        episodes=tuple(
            terminal_episode_record_from_document(item)
            for item in _list(mapping["episodes"], "episodes")
        ),
    )


def pitch_scientific_criterion_to_document(
    criterion: PitchScientificCriterion,
) -> dict[str, JSONValue]:
    """Encode one factory-derived criterion summary."""

    if not isinstance(criterion, PitchScientificCriterion):
        raise ValueError("criterion must be a PitchScientificCriterion")
    return {
        "eligible": criterion.eligible,
        "seeds": [
            {
                "seed": seed.seed,
                "iid_coordinate_within_five_rate": seed.iid_coordinate_within_five_rate,
                "ood_coordinate_within_five_rate": seed.ood_coordinate_within_five_rate,
                "ood_lower_coordinate_within_five_rate": (
                    seed.ood_lower_coordinate_within_five_rate
                ),
                "ood_upper_coordinate_within_five_rate": (
                    seed.ood_upper_coordinate_within_five_rate
                ),
                "iid_submitted_success_rate": seed.iid_submitted_success_rate,
                "iid_bound_blocked_actions": seed.iid_bound_blocked_actions,
                "iid_truncations": seed.iid_truncations,
                "iid_zero_submitted_success_rate": seed.iid_zero_submitted_success_rate,
                "iid_shuffled_submitted_success_rate": (seed.iid_shuffled_submitted_success_rate),
                "register_ood_submitted_success_rate": (seed.register_ood_submitted_success_rate),
                "iid_mean_successful_excess_actions": (seed.iid_mean_successful_excess_actions),
            }
            for seed in criterion.seeds
        ],
        "median_register_ood_submitted_success_rate": (
            criterion.median_register_ood_submitted_success_rate
        ),
        "failed_gates": list(criterion.failed_gates),
        "criterion_met": criterion.criterion_met,
        "status": criterion.status,
    }


def _expected_register_aggregates(
    rows: tuple[PitchEvaluationRow, ...],
) -> tuple[RegisterOODAggregate, ...]:
    return tuple(
        RegisterOODAggregate.from_rows(rows[start], rows[start + 1])
        for start in (1, 6, 11, 16, 19, 22, 25)
    )


def _validate_learned_report_metadata(
    profile: ProfileName,
    rows: tuple[PitchEvaluationRow, ...],
) -> None:
    expected_examples = int(_PITCH_PROFILE_CONFIGS[profile]["training_coordinate_count"])
    for row in rows:
        if (
            row.trainer is not PitchTrainerKind.PITCH
            or row.seed is None
            or row.actor_id != f"pitch-{row.seed}"
            or row.parameter_count != PITCH_ESTIMATOR_PARAMETER_COUNT
            or row.training_examples != expected_examples
        ):
            raise ValueError("learned rows must match the exact pitch actor and profile metadata")


@dataclass(frozen=True, slots=True)
class PitchEvaluationReport:
    """Complete smoke or exact-three checkpoint schema-v2 evidence."""

    schema_version: int
    profile: ProfileName
    evaluation_device: DeviceName
    environment_contract_id: str
    spectrum_grid_id: str
    preprocessing_schema_id: str
    architecture_schema_id: str
    policy_semantics_id: str
    coordinate_distribution_id: str
    coordinate_split_digest_sha256: str
    compatibility_sha256: str
    terminal_rows: tuple[PitchEvaluationRow, ...]
    coordinate_evaluations: tuple[PitchCoordinateEvaluation, ...]
    register_ood_aggregates: tuple[RegisterOODAggregate, ...]
    criterion: PitchScientificCriterion

    def __post_init__(self) -> None:
        version = _integer(self.schema_version, "schema_version", minimum=1)
        if version != PITCH_ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {PITCH_ARTIFACT_SCHEMA_VERSION}")
        if not isinstance(self.profile, ProfileName):
            raise ValueError("profile must be a ProfileName")
        if not isinstance(self.evaluation_device, DeviceName):
            raise ValueError("evaluation_device must be a DeviceName")
        identities = (
            (self.environment_contract_id, ENVIRONMENT_CONTRACT_ID, "environment_contract_id"),
            (self.spectrum_grid_id, SPECTRUM_GRID_ID, "spectrum_grid_id"),
            (
                self.preprocessing_schema_id,
                PITCH_PREPROCESSING_SCHEMA_ID,
                "preprocessing_schema_id",
            ),
            (
                self.architecture_schema_id,
                PITCH_ESTIMATOR_ARCHITECTURE_ID,
                "architecture_schema_id",
            ),
            (self.policy_semantics_id, PITCH_POLICY_SEMANTICS_ID, "policy_semantics_id"),
            (
                self.coordinate_distribution_id,
                PITCH_DISTRIBUTION_ID,
                "coordinate_distribution_id",
            ),
            (
                self.coordinate_split_digest_sha256,
                PITCH_SPLIT_DIGEST_SHA256,
                "coordinate_split_digest_sha256",
            ),
            (
                self.compatibility_sha256,
                pitch_report_compatibility_sha256(self.profile),
                "compatibility_sha256",
            ),
        )
        for actual, expected, field in identities:
            if actual != expected:
                raise ValueError(f"{field} must be {expected!r}")
        rows = _typed_tuple(self.terminal_rows, PitchEvaluationRow, "terminal_rows")
        coordinates = _typed_tuple(
            self.coordinate_evaluations,
            PitchCoordinateEvaluation,
            "coordinate_evaluations",
        )
        aggregates = _typed_tuple(
            self.register_ood_aggregates,
            RegisterOODAggregate,
            "register_ood_aggregates",
        )
        if not isinstance(self.criterion, PitchScientificCriterion):
            raise ValueError("criterion must be a PitchScientificCriterion")
        if self.profile is ProfileName.SMOKE:
            validate_pitch_smoke_row_matrix(rows)
            _validate_learned_report_metadata(self.profile, rows[:3])
            if coordinates or aggregates:
                raise ValueError("smoke reports must not contain final coordinate or OOD evidence")
            expected_criterion = evaluate_pitch_criterion(
                coordinate_evaluations=(),
                terminal_rows=(),
                eligible=False,
            )
        elif self.profile is ProfileName.CHECKPOINT:
            validate_pitch_checkpoint_row_matrix(rows)
            _validate_learned_report_metadata(self.profile, rows[:15])
            if tuple(item.seed for item in coordinates) != (0, 1, 2):
                raise ValueError("checkpoint coordinate evaluations require seeds 0, 1, and 2")
            expected_aggregates = _expected_register_aggregates(rows)
            if aggregates != expected_aggregates:
                raise ValueError("register_ood_aggregates must be re-derived from terminal rows")
            expected_criterion = evaluate_pitch_criterion(
                coordinate_evaluations=coordinates,
                terminal_rows=rows,
                eligible=self.evaluation_device is DeviceName.CPU,
            )
        else:
            raise ValueError("unsupported pitch report profile")
        if self.criterion != expected_criterion:
            raise ValueError("criterion must be re-derived from complete raw report evidence")
        object.__setattr__(self, "terminal_rows", rows)
        object.__setattr__(self, "coordinate_evaluations", coordinates)
        object.__setattr__(self, "register_ood_aggregates", aggregates)
        object.__setattr__(self, "schema_version", version)

    def to_document(self) -> dict[str, JSONValue]:
        """Return a freshly owned canonical schema-v2 report document."""

        return {
            "schema_version": self.schema_version,
            "profile": self.profile.value,
            "evaluation_device": self.evaluation_device.value,
            "environment_contract_id": self.environment_contract_id,
            "spectrum_grid_id": self.spectrum_grid_id,
            "preprocessing_schema_id": self.preprocessing_schema_id,
            "architecture_schema_id": self.architecture_schema_id,
            "policy_semantics_id": self.policy_semantics_id,
            "coordinate_distribution_id": self.coordinate_distribution_id,
            "coordinate_split_digest_sha256": self.coordinate_split_digest_sha256,
            "compatibility_sha256": self.compatibility_sha256,
            "terminal_rows": [pitch_evaluation_row_to_document(row) for row in self.terminal_rows],
            "coordinate_evaluations": [
                pitch_coordinate_evaluation_to_document(item)
                for item in self.coordinate_evaluations
            ],
            "register_ood_aggregates": [
                register_ood_aggregate_to_document(item) for item in self.register_ood_aggregates
            ],
            "criterion": pitch_scientific_criterion_to_document(self.criterion),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, JSONValue]) -> PitchEvaluationReport:
        """Decode every raw section and re-derive all report summaries."""

        mapping = _mapping(document, "pitch evaluation report")
        fields = {
            "schema_version",
            "profile",
            "evaluation_device",
            "environment_contract_id",
            "spectrum_grid_id",
            "preprocessing_schema_id",
            "architecture_schema_id",
            "policy_semantics_id",
            "coordinate_distribution_id",
            "coordinate_split_digest_sha256",
            "compatibility_sha256",
            "terminal_rows",
            "coordinate_evaluations",
            "register_ood_aggregates",
            "criterion",
        }
        _exact_fields(mapping, fields, "pitch evaluation report")
        profile = _enum(ProfileName, mapping["profile"], "profile")
        device = _enum(DeviceName, mapping["evaluation_device"], "evaluation_device")
        rows = tuple(
            pitch_evaluation_row_from_document(item)
            for item in _list(mapping["terminal_rows"], "terminal_rows")
        )
        coordinates = tuple(
            pitch_coordinate_evaluation_from_document(item)
            for item in _list(mapping["coordinate_evaluations"], "coordinate_evaluations")
        )
        aggregates = tuple(
            register_ood_aggregate_from_document(item)
            for item in _list(mapping["register_ood_aggregates"], "register_ood_aggregates")
        )
        expected_criterion = evaluate_pitch_criterion(
            coordinate_evaluations=coordinates,
            terminal_rows=rows if profile is ProfileName.CHECKPOINT else (),
            eligible=profile is ProfileName.CHECKPOINT and device is DeviceName.CPU,
        )
        criterion_document = _mapping(mapping["criterion"], "criterion")
        if criterion_document != pitch_scientific_criterion_to_document(expected_criterion):
            raise ValueError("criterion must be re-derived from complete raw report evidence")
        return cls(
            schema_version=_integer(mapping["schema_version"], "schema_version"),
            profile=profile,
            evaluation_device=device,
            environment_contract_id=_string(
                mapping["environment_contract_id"],
                "environment_contract_id",
            ),
            spectrum_grid_id=_string(mapping["spectrum_grid_id"], "spectrum_grid_id"),
            preprocessing_schema_id=_string(
                mapping["preprocessing_schema_id"],
                "preprocessing_schema_id",
            ),
            architecture_schema_id=_string(
                mapping["architecture_schema_id"],
                "architecture_schema_id",
            ),
            policy_semantics_id=_string(
                mapping["policy_semantics_id"],
                "policy_semantics_id",
            ),
            coordinate_distribution_id=_string(
                mapping["coordinate_distribution_id"],
                "coordinate_distribution_id",
            ),
            coordinate_split_digest_sha256=_string(
                mapping["coordinate_split_digest_sha256"],
                "coordinate_split_digest_sha256",
            ),
            compatibility_sha256=_string(
                mapping["compatibility_sha256"],
                "compatibility_sha256",
            ),
            terminal_rows=rows,
            coordinate_evaluations=coordinates,
            register_ood_aggregates=aggregates,
            criterion=expected_criterion,
        )


__all__ = [
    "PITCH_ESTIMATOR_ARCHITECTURE_ID",
    "PITCH_ESTIMATOR_PARAMETER_COUNT",
    "PITCH_POLICY_SEMANTICS_ID",
    "PITCH_PREPROCESSING_SCHEMA_ID",
    "PitchEvaluationReport",
    "pitch_coordinate_evaluation_from_document",
    "pitch_coordinate_evaluation_to_document",
    "pitch_evaluation_row_from_document",
    "pitch_evaluation_row_to_document",
    "pitch_report_compatibility_sha256",
    "pitch_scientific_criterion_to_document",
    "register_ood_aggregate_from_document",
    "register_ood_aggregate_to_document",
]
