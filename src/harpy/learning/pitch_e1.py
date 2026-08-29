"""Strict E.1 device-cohort reports around frozen schema-v2 pitch evidence."""

from __future__ import annotations

import hashlib
import operator
import re
from collections.abc import Mapping
from dataclasses import dataclass

from harpy.learning.artifacts import (
    PITCH_ARTIFACT_SCHEMA_VERSION,
    RuntimeStatus,
    SourceStatus,
    canonical_json_bytes,
)
from harpy.learning.models import (
    ENVIRONMENT_CONTRACT_ID,
    SPECTRUM_GRID_ID,
    DeviceName,
    JSONValue,
    ProfileName,
)
from harpy.learning.pitch_data import PITCH_DISTRIBUTION_ID, PITCH_SPLIT_DIGEST_SHA256
from harpy.learning.pitch_evaluation import (
    PitchCoordinateEvaluation,
    PitchEvaluationRow,
    PitchScientificCriterion,
    RegisterOODAggregate,
    evaluate_pitch_criterion,
)
from harpy.learning.pitch_reports import (
    PITCH_ESTIMATOR_ARCHITECTURE_ID,
    PITCH_POLICY_SEMANTICS_ID,
    PITCH_PREPROCESSING_SCHEMA_ID,
    PitchEvaluationReport,
    pitch_coordinate_evaluation_from_document,
    pitch_coordinate_evaluation_to_document,
    pitch_evaluation_row_from_document,
    pitch_evaluation_row_to_document,
    pitch_report_compatibility_sha256,
    pitch_scientific_criterion_to_document,
    register_ood_aggregate_from_document,
    register_ood_aggregate_to_document,
)

PITCH_E1_REPORT_SCHEMA_VERSION = 3
PITCH_E1_PROTOCOL_ID = "harpy-sine-pitch-e1-homogeneous-device-cohort-v1"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


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


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a string-keyed mapping")
    return value


def _exact_fields(value: Mapping[str, object], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{field} fields must be exactly {sorted(expected)}")


def _string(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{field} must be a string")
    return value


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a bool")
    return value


def _source_to_document(source: SourceStatus) -> dict[str, JSONValue]:
    return {
        "commit": source.commit,
        "dirty_tree": source.dirty_tree,
        "tracked_diff_sha256": source.tracked_diff_sha256,
        "dependency_lock_sha256": source.dependency_lock_sha256,
        "required_inputs_committed": source.required_inputs_committed,
    }


def _source_from_document(value: object) -> SourceStatus:
    mapping = _mapping(value, "source")
    _exact_fields(
        mapping,
        {
            "commit",
            "dirty_tree",
            "tracked_diff_sha256",
            "dependency_lock_sha256",
            "required_inputs_committed",
        },
        "source",
    )
    return SourceStatus(
        commit=_string(mapping["commit"], "source.commit"),
        dirty_tree=_boolean(mapping["dirty_tree"], "source.dirty_tree"),
        tracked_diff_sha256=_digest(mapping["tracked_diff_sha256"], "source.tracked_diff_sha256"),
        dependency_lock_sha256=_digest(
            mapping["dependency_lock_sha256"], "source.dependency_lock_sha256"
        ),
        required_inputs_committed=_boolean(
            mapping["required_inputs_committed"], "source.required_inputs_committed"
        ),
    )


def _runtime_to_document(runtime: RuntimeStatus) -> dict[str, JSONValue]:
    return {
        "python_version": runtime.python_version,
        "platform": runtime.platform,
        "processor": runtime.processor,
        "numpy_version": runtime.numpy_version,
        "gymnasium_version": runtime.gymnasium_version,
        "torch_version": runtime.torch_version,
        "stable_baselines3_version": runtime.stable_baselines3_version,
        "device": runtime.device.value,
        "device_description": runtime.device_description,
        "cuda_runtime_version": runtime.cuda_runtime_version,
        "cuda_driver_version": runtime.cuda_driver_version,
    }


def _runtime_from_document(value: object) -> RuntimeStatus:
    mapping = _mapping(value, "runtime")
    fields = {
        "python_version",
        "platform",
        "processor",
        "numpy_version",
        "gymnasium_version",
        "torch_version",
        "stable_baselines3_version",
        "device",
        "device_description",
        "cuda_runtime_version",
        "cuda_driver_version",
    }
    _exact_fields(mapping, fields, "runtime")
    try:
        device = DeviceName(_string(mapping["device"], "runtime.device"))
    except ValueError as error:
        raise ValueError("runtime.device must be cpu or cuda") from error
    return RuntimeStatus(
        python_version=_string(mapping["python_version"], "runtime.python_version"),
        platform=_string(mapping["platform"], "runtime.platform"),
        processor=_string(mapping["processor"], "runtime.processor", allow_empty=True),
        numpy_version=_string(mapping["numpy_version"], "runtime.numpy_version"),
        gymnasium_version=_string(mapping["gymnasium_version"], "runtime.gymnasium_version"),
        torch_version=_string(mapping["torch_version"], "runtime.torch_version"),
        stable_baselines3_version=_string(
            mapping["stable_baselines3_version"], "runtime.stable_baselines3_version"
        ),
        device=device,
        device_description=_string(mapping["device_description"], "runtime.device_description"),
        cuda_runtime_version=(
            None
            if mapping["cuda_runtime_version"] is None
            else _string(mapping["cuda_runtime_version"], "runtime.cuda_runtime_version")
        ),
        cuda_driver_version=(
            None
            if mapping["cuda_driver_version"] is None
            else _string(mapping["cuda_driver_version"], "runtime.cuda_driver_version")
        ),
    )


@dataclass(frozen=True, slots=True)
class PitchE1ArtifactProvenance:
    """Immutable manifest, source, and runtime identity for one ordered seed."""

    seed: int
    artifact_manifest_sha256: str
    source: SourceStatus
    runtime: RuntimeStatus
    compatibility_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        object.__setattr__(
            self,
            "artifact_manifest_sha256",
            _digest(self.artifact_manifest_sha256, "artifact_manifest_sha256"),
        )
        if not isinstance(self.source, SourceStatus):
            raise ValueError("source must be a SourceStatus")
        if not isinstance(self.runtime, RuntimeStatus):
            raise ValueError("runtime must be a RuntimeStatus")
        object.__setattr__(
            self,
            "compatibility_sha256",
            _digest(self.compatibility_sha256, "compatibility_sha256"),
        )

    def to_document(self) -> dict[str, JSONValue]:
        return {
            "seed": self.seed,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "source": _source_to_document(self.source),
            "runtime": _runtime_to_document(self.runtime),
            "compatibility_sha256": self.compatibility_sha256,
        }

    @classmethod
    def from_document(cls, value: object) -> PitchE1ArtifactProvenance:
        mapping = _mapping(value, "artifact provenance")
        _exact_fields(
            mapping,
            {
                "seed",
                "artifact_manifest_sha256",
                "source",
                "runtime",
                "compatibility_sha256",
            },
            "artifact provenance",
        )
        return cls(
            seed=_integer(mapping["seed"], "seed"),
            artifact_manifest_sha256=_digest(
                mapping["artifact_manifest_sha256"], "artifact_manifest_sha256"
            ),
            source=_source_from_document(mapping["source"]),
            runtime=_runtime_from_document(mapping["runtime"]),
            compatibility_sha256=_digest(mapping["compatibility_sha256"], "compatibility_sha256"),
        )


def _cohort_identity_document(
    *,
    training_device: DeviceName,
    artifacts: tuple[PitchE1ArtifactProvenance, ...],
    evaluator_source: SourceStatus,
) -> dict[str, JSONValue]:
    return {
        "protocol_id": PITCH_E1_PROTOCOL_ID,
        "training_device": training_device.value,
        "evaluation_device": DeviceName.CPU.value,
        "artifacts": [item.to_document() for item in artifacts],
        "evaluator_source": _source_to_document(evaluator_source),
    }


def pitch_e1_cohort_digest_sha256(
    *,
    training_device: DeviceName,
    artifacts: tuple[PitchE1ArtifactProvenance, ...],
    evaluator_source: SourceStatus,
) -> str:
    """Hash the complete ordered cohort and evaluator identity."""

    return hashlib.sha256(
        canonical_json_bytes(
            _cohort_identity_document(
                training_device=training_device,
                artifacts=artifacts,
                evaluator_source=evaluator_source,
            )
        )
    ).hexdigest()


def _runtime_cohort_identity(runtime: RuntimeStatus) -> tuple[object, ...]:
    return tuple(_runtime_to_document(runtime).values())


@dataclass(frozen=True, slots=True)
class PitchE1RawEvidence:
    """Schema-v3-only raw evidence with no detachable report eligibility claim."""

    terminal_rows: tuple[PitchEvaluationRow, ...]
    coordinate_evaluations: tuple[PitchCoordinateEvaluation, ...]
    register_ood_aggregates: tuple[RegisterOODAggregate, ...]

    def __post_init__(self) -> None:
        rows = tuple(self.terminal_rows)
        coordinates = tuple(self.coordinate_evaluations)
        aggregates = tuple(self.register_ood_aggregates)
        if not all(isinstance(item, PitchEvaluationRow) for item in rows):
            raise ValueError("terminal_rows must contain PitchEvaluationRow values")
        if not all(isinstance(item, PitchCoordinateEvaluation) for item in coordinates):
            raise ValueError("coordinate_evaluations must contain PitchCoordinateEvaluation values")
        if not all(isinstance(item, RegisterOODAggregate) for item in aggregates):
            raise ValueError("register_ood_aggregates must contain RegisterOODAggregate values")
        _validated_schema_v2_evidence(rows, coordinates, aggregates)
        object.__setattr__(self, "terminal_rows", rows)
        object.__setattr__(self, "coordinate_evaluations", coordinates)
        object.__setattr__(self, "register_ood_aggregates", aggregates)

    @classmethod
    def from_pitch_report(cls, report: PitchEvaluationReport) -> PitchE1RawEvidence:
        if not isinstance(report, PitchEvaluationReport):
            raise ValueError("report must be a PitchEvaluationReport")
        return cls(
            terminal_rows=report.terminal_rows,
            coordinate_evaluations=report.coordinate_evaluations,
            register_ood_aggregates=report.register_ood_aggregates,
        )

    def criterion(self) -> PitchScientificCriterion:
        return evaluate_pitch_criterion(
            coordinate_evaluations=self.coordinate_evaluations,
            terminal_rows=self.terminal_rows,
            eligible=True,
        )

    def to_document(self) -> dict[str, JSONValue]:
        return {
            "terminal_rows": [pitch_evaluation_row_to_document(row) for row in self.terminal_rows],
            "coordinate_evaluations": [
                pitch_coordinate_evaluation_to_document(item)
                for item in self.coordinate_evaluations
            ],
            "register_ood_aggregates": [
                register_ood_aggregate_to_document(item) for item in self.register_ood_aggregates
            ],
        }

    @classmethod
    def from_document(cls, value: object) -> PitchE1RawEvidence:
        mapping = _mapping(value, "E.1 raw evidence")
        _exact_fields(
            mapping,
            {"terminal_rows", "coordinate_evaluations", "register_ood_aggregates"},
            "E.1 raw evidence",
        )
        rows_value = mapping["terminal_rows"]
        coordinates_value = mapping["coordinate_evaluations"]
        aggregates_value = mapping["register_ood_aggregates"]
        if not isinstance(rows_value, list):
            raise ValueError("terminal_rows must be a list")
        if not isinstance(coordinates_value, list):
            raise ValueError("coordinate_evaluations must be a list")
        if not isinstance(aggregates_value, list):
            raise ValueError("register_ood_aggregates must be a list")
        return cls(
            terminal_rows=tuple(pitch_evaluation_row_from_document(item) for item in rows_value),
            coordinate_evaluations=tuple(
                pitch_coordinate_evaluation_from_document(item) for item in coordinates_value
            ),
            register_ood_aggregates=tuple(
                register_ood_aggregate_from_document(item) for item in aggregates_value
            ),
        )


def _validated_schema_v2_evidence(
    rows: tuple[PitchEvaluationRow, ...],
    coordinates: tuple[PitchCoordinateEvaluation, ...],
    aggregates: tuple[RegisterOODAggregate, ...],
) -> PitchEvaluationReport:
    """Reuse frozen validation internally without serializing a schema-v2 report."""

    criterion = evaluate_pitch_criterion(
        coordinate_evaluations=coordinates,
        terminal_rows=rows,
        eligible=True,
    )
    return PitchEvaluationReport(
        schema_version=PITCH_ARTIFACT_SCHEMA_VERSION,
        profile=ProfileName.CHECKPOINT,
        evaluation_device=DeviceName.CPU,
        environment_contract_id=ENVIRONMENT_CONTRACT_ID,
        spectrum_grid_id=SPECTRUM_GRID_ID,
        preprocessing_schema_id=PITCH_PREPROCESSING_SCHEMA_ID,
        architecture_schema_id=PITCH_ESTIMATOR_ARCHITECTURE_ID,
        policy_semantics_id=PITCH_POLICY_SEMANTICS_ID,
        coordinate_distribution_id=PITCH_DISTRIBUTION_ID,
        coordinate_split_digest_sha256=PITCH_SPLIT_DIGEST_SHA256,
        compatibility_sha256=pitch_report_compatibility_sha256(ProfileName.CHECKPOINT),
        terminal_rows=rows,
        coordinate_evaluations=coordinates,
        register_ood_aggregates=aggregates,
        criterion=criterion,
    )


@dataclass(frozen=True, slots=True)
class PitchE1EvaluationReport:
    """Schema-v3 E.1 provenance and its exclusively owned criterion claim."""

    schema_version: int
    protocol_id: str
    training_device: DeviceName
    evaluation_device: DeviceName
    artifacts: tuple[PitchE1ArtifactProvenance, ...]
    evaluator_source: SourceStatus
    cohort_digest_sha256: str
    evidence: PitchE1RawEvidence
    criterion: PitchScientificCriterion

    def __post_init__(self) -> None:
        version = _integer(self.schema_version, "schema_version", minimum=1)
        if version != PITCH_E1_REPORT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {PITCH_E1_REPORT_SCHEMA_VERSION}")
        if self.protocol_id != PITCH_E1_PROTOCOL_ID:
            raise ValueError(f"protocol_id must be {PITCH_E1_PROTOCOL_ID!r}")
        if self.training_device is not DeviceName.CUDA:
            raise ValueError("E.1 training_device must be CUDA")
        if self.evaluation_device is not DeviceName.CPU:
            raise ValueError("E.1 final evaluation must use CPU")
        try:
            artifacts = tuple(self.artifacts)
        except TypeError as error:
            raise ValueError("artifacts must contain provenance records") from error
        if (
            tuple(item.seed for item in artifacts if isinstance(item, PitchE1ArtifactProvenance))
            != (0, 1, 2)
            or len(artifacts) != 3
            or not all(isinstance(item, PitchE1ArtifactProvenance) for item in artifacts)
        ):
            raise ValueError("E.1 artifacts must be exact ordered seeds 0, 1, and 2")
        if any(item.runtime.device is not self.training_device for item in artifacts):
            raise ValueError("E.1 artifacts must share the declared training device")
        if len({_runtime_cohort_identity(item.runtime) for item in artifacts}) != 1:
            raise ValueError("E.1 artifacts must share one homogeneous runtime cohort")
        if not isinstance(self.evaluator_source, SourceStatus):
            raise ValueError("evaluator_source must be a SourceStatus")
        if self.evaluator_source.dirty_tree or not self.evaluator_source.required_inputs_committed:
            raise ValueError("E.1 evaluator source must be clean and committed")
        if any(item.source != self.evaluator_source for item in artifacts):
            raise ValueError("E.1 artifacts and evaluator must share exact source provenance")
        if not isinstance(self.evidence, PitchE1RawEvidence):
            raise ValueError("evidence must be PitchE1RawEvidence")
        expected_compatibility = pitch_report_compatibility_sha256(ProfileName.CHECKPOINT)
        if any(item.compatibility_sha256 != expected_compatibility for item in artifacts):
            raise ValueError("E.1 artifact compatibility must match the frozen pitch contract")
        expected_criterion = self.evidence.criterion()
        if self.criterion != expected_criterion:
            raise ValueError("E.1 criterion must be re-derived from raw evidence")
        expected_digest = pitch_e1_cohort_digest_sha256(
            training_device=self.training_device,
            artifacts=artifacts,
            evaluator_source=self.evaluator_source,
        )
        if self.cohort_digest_sha256 != expected_digest:
            raise ValueError("cohort_digest_sha256 must be re-derived from E.1 provenance")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "schema_version", version)

    @classmethod
    def create(
        cls,
        *,
        artifacts: tuple[PitchE1ArtifactProvenance, ...],
        evaluator_source: SourceStatus,
        evidence: PitchE1RawEvidence,
    ) -> PitchE1EvaluationReport:
        normalized = tuple(artifacts)
        if not normalized:
            raise ValueError("artifacts must not be empty")
        training_device = normalized[0].runtime.device
        return cls(
            schema_version=PITCH_E1_REPORT_SCHEMA_VERSION,
            protocol_id=PITCH_E1_PROTOCOL_ID,
            training_device=training_device,
            evaluation_device=DeviceName.CPU,
            artifacts=normalized,
            evaluator_source=evaluator_source,
            cohort_digest_sha256=pitch_e1_cohort_digest_sha256(
                training_device=training_device,
                artifacts=normalized,
                evaluator_source=evaluator_source,
            ),
            evidence=evidence,
            criterion=evidence.criterion(),
        )

    def to_document(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "training_device": self.training_device.value,
            "evaluation_device": self.evaluation_device.value,
            "artifacts": [item.to_document() for item in self.artifacts],
            "evaluator_source": _source_to_document(self.evaluator_source),
            "cohort_digest_sha256": self.cohort_digest_sha256,
            "evidence": self.evidence.to_document(),
            "criterion": pitch_scientific_criterion_to_document(self.criterion),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, JSONValue]) -> PitchE1EvaluationReport:
        mapping = _mapping(document, "E.1 evaluation report")
        _exact_fields(
            mapping,
            {
                "schema_version",
                "protocol_id",
                "training_device",
                "evaluation_device",
                "artifacts",
                "evaluator_source",
                "cohort_digest_sha256",
                "evidence",
                "criterion",
            },
            "E.1 evaluation report",
        )
        artifacts_value = mapping["artifacts"]
        if not isinstance(artifacts_value, list):
            raise ValueError("artifacts must be a list")
        evidence = PitchE1RawEvidence.from_document(mapping["evidence"])
        expected_criterion = evidence.criterion()
        criterion_document = _mapping(mapping["criterion"], "criterion")
        expected_criterion_document = pitch_scientific_criterion_to_document(expected_criterion)
        if canonical_json_bytes(criterion_document) != canonical_json_bytes(
            expected_criterion_document
        ):
            raise ValueError("E.1 criterion must be re-derived from raw evidence")
        try:
            training_device = DeviceName(_string(mapping["training_device"], "training_device"))
            evaluation_device = DeviceName(
                _string(mapping["evaluation_device"], "evaluation_device")
            )
        except ValueError as error:
            raise ValueError("E.1 report devices must be cpu or cuda") from error
        return cls(
            schema_version=_integer(mapping["schema_version"], "schema_version", minimum=1),
            protocol_id=_string(mapping["protocol_id"], "protocol_id"),
            training_device=training_device,
            evaluation_device=evaluation_device,
            artifacts=tuple(
                PitchE1ArtifactProvenance.from_document(item) for item in artifacts_value
            ),
            evaluator_source=_source_from_document(mapping["evaluator_source"]),
            cohort_digest_sha256=_digest(mapping["cohort_digest_sha256"], "cohort_digest_sha256"),
            evidence=evidence,
            criterion=expected_criterion,
        )


__all__ = [
    "PITCH_E1_PROTOCOL_ID",
    "PITCH_E1_REPORT_SCHEMA_VERSION",
    "PitchE1ArtifactProvenance",
    "PitchE1EvaluationReport",
    "PitchE1RawEvidence",
    "pitch_e1_cohort_digest_sha256",
]
