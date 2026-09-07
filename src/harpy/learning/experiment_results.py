"""Portable, explicitly ineligible readouts outside the frozen scientific reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from harpy.envs.baselines import BaselineKind
from harpy.learning.artifacts import (
    SourceProvenance,
    _source_from_document,
    _source_to_document,
    canonical_json_bytes,
    decode_json_bytes,
)
from harpy.learning.diagnostic_codecs import (
    diagnostic_report_from_document,
    diagnostic_report_to_document,
)
from harpy.learning.diagnostics import DiagnosticReport
from harpy.learning.models import DeviceName, ProfileName
from harpy.learning.pitch_data import PitchEvaluationSuiteId, PitchTrainerKind
from harpy.learning.pitch_evaluation import PitchEvaluationRow
from harpy.learning.pitch_reports import (
    PITCH_ESTIMATOR_PARAMETER_COUNT,
    PITCH_POLICY_SEMANTICS_ID,
    pitch_evaluation_row_from_document,
    pitch_evaluation_row_to_document,
)
from harpy.learning.trace import EpisodeTrace, trace_from_document, trace_json_bytes

EXPLORATORY_EVALUATION_SCHEMA_ID = "harpy-pitch-exploration-evaluation-v1"
EXPLORATORY_DIAGNOSTIC_SCHEMA_ID = "harpy-pitch-exploration-diagnostics-v1"
ARTIFACT_RUN_SCHEMA_ID = "harpy-artifact-run-v1"


def _object(value: object, fields: set[str], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{name} fields must be exactly {sorted(fields)}")
    return value


def _hash(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _criterion(reason: str) -> dict[str, object]:
    return {
        "eligible": False,
        "criterion_met": None,
        "status": "ineligible",
        "reason": reason,
    }


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    """Identify the exact trusted artifact and the device used for this readout."""

    artifact_schema_version: int
    manifest_sha256: str
    model_sha256: str
    trainer: str
    training_seed: int
    profile: ProfileName
    training_device: DeviceName
    evaluation_device: DeviceName
    source: SourceProvenance

    def __post_init__(self) -> None:
        if type(self.artifact_schema_version) is not int or self.artifact_schema_version not in (
            1,
            2,
        ):
            raise ValueError("artifact_schema_version must be 1 or 2")
        _hash(self.manifest_sha256, "manifest_sha256")
        _hash(self.model_sha256, "model_sha256")
        expected_trainers = ("bc", "ppo") if self.artifact_schema_version == 1 else ("pitch",)
        if self.trainer not in expected_trainers:
            raise ValueError("trainer must match artifact_schema_version")
        if type(self.training_seed) is not int or self.training_seed < 0:
            raise ValueError("training_seed must be a non-negative integer")
        if not isinstance(self.profile, ProfileName):
            raise ValueError("profile must be a ProfileName")
        if not isinstance(self.training_device, DeviceName) or not isinstance(
            self.evaluation_device, DeviceName
        ):
            raise ValueError("training and evaluation devices must be DeviceName values")
        # Reuse the artifact source codec, including explicit package snapshots.
        if _source_from_document(_source_to_document(self.source)) != self.source:
            raise ValueError("source must be validated artifact provenance")

    def to_document(self) -> dict[str, object]:
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "manifest_sha256": self.manifest_sha256,
            "model_sha256": self.model_sha256,
            "trainer": self.trainer,
            "training_seed": self.training_seed,
            "profile": self.profile.value,
            "training_device": self.training_device.value,
            "evaluation_device": self.evaluation_device.value,
            "source": _source_to_document(self.source),
        }

    @classmethod
    def from_document(cls, value: object) -> ArtifactProvenance:
        document = _object(value, set(cls.__dataclass_fields__), "provenance")
        return cls(
            artifact_schema_version=document["artifact_schema_version"],
            manifest_sha256=document["manifest_sha256"],
            model_sha256=document["model_sha256"],
            trainer=document["trainer"],
            training_seed=document["training_seed"],
            profile=ProfileName(document["profile"]),
            training_device=DeviceName(document["training_device"]),
            evaluation_device=DeviceName(document["evaluation_device"]),
            source=_source_from_document(document["source"]),
        )


@dataclass(frozen=True, slots=True)
class ExploratoryPitchEvaluation:
    """Closed-loop metrics for one model, with no aggregate scientific criterion."""

    provenance: ArtifactProvenance
    terminal_rows: tuple[PitchEvaluationRow, ...]

    def __post_init__(self) -> None:
        _pitch_provenance(self.provenance)
        rows = tuple(self.terminal_rows)
        suites = (
            (PitchEvaluationSuiteId.SMOKE,)
            if self.provenance.profile is ProfileName.SMOKE
            else (
                PitchEvaluationSuiteId.IID,
                PitchEvaluationSuiteId.OOD_LOWER,
                PitchEvaluationSuiteId.OOD_UPPER,
            )
        )
        expected = [(f"pitch-{self.provenance.training_seed}", suite) for suite in suites] + [
            (kind.value, suite) for kind in BaselineKind for suite in suites
        ]
        if (
            not all(isinstance(row, PitchEvaluationRow) for row in rows)
            or [(row.actor_id, row.suite_id) for row in rows] != expected
        ):
            raise ValueError("exploratory rows must contain one model and all matched baselines")
        for index, row in enumerate(rows):
            expected_trainer = PitchTrainerKind.PITCH if index < len(suites) else None
            if row.trainer is not expected_trainer:
                raise ValueError("exploratory row trainer must match its learned or baseline role")
            if row.probe is not None:
                raise ValueError("exploratory terminal rows must be unperturbed")
            if row.trainer is not None and (
                row.trainer is not PitchTrainerKind.PITCH
                or row.seed != self.provenance.training_seed
                or row.parameter_count != PITCH_ESTIMATOR_PARAMETER_COUNT
                or row.training_examples
                != (256 if self.provenance.profile is ProfileName.SMOKE else 1_400)
            ):
                raise ValueError("learned row must match artifact provenance")
        object.__setattr__(self, "terminal_rows", rows)

    def to_document(self) -> dict[str, object]:
        return {
            "schema_id": EXPLORATORY_EVALUATION_SCHEMA_ID,
            "provenance": self.provenance.to_document(),
            "criterion": _criterion("exploratory_single_artifact"),
            "terminal_rows": [pitch_evaluation_row_to_document(row) for row in self.terminal_rows],
        }


@dataclass(frozen=True, slots=True)
class ExploratoryPitchDiagnostics:
    """Detailed single-model diagnostics, explicitly outside cohort eligibility."""

    provenance: ArtifactProvenance
    diagnostics: DiagnosticReport

    def __post_init__(self) -> None:
        _pitch_provenance(self.provenance)
        if not isinstance(self.diagnostics, DiagnosticReport):
            raise ValueError("diagnostics must be a DiagnosticReport")
        if (
            self.diagnostics.seed != self.provenance.training_seed
            or self.diagnostics.artifact_manifest_sha256 != self.provenance.manifest_sha256
            or self.diagnostics.bound_mask
            or self.diagnostics.actor_semantics != PITCH_POLICY_SEMANTICS_ID
            or self.diagnostics.suite_id not in set(PitchEvaluationSuiteId)
        ):
            raise ValueError("diagnostics must match the single artifact provenance")

    def to_document(self) -> dict[str, object]:
        return {
            "schema_id": EXPLORATORY_DIAGNOSTIC_SCHEMA_ID,
            "provenance": self.provenance.to_document(),
            "criterion": _criterion("exploratory_single_artifact"),
            "diagnostics": diagnostic_report_to_document(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class ArtifactRunResult:
    """An unchanged episode trace paired with its exact model provenance."""

    provenance: ArtifactProvenance
    trace: EpisodeTrace

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, ArtifactProvenance):
            raise ValueError("provenance must be ArtifactProvenance")
        if not isinstance(self.trace, EpisodeTrace):
            raise ValueError("trace must be an EpisodeTrace")

    def to_document(self) -> dict[str, object]:
        return {
            "schema_id": ARTIFACT_RUN_SCHEMA_ID,
            "provenance": self.provenance.to_document(),
            "criterion": _criterion("single_episode_demonstration"),
            "trace": decode_json_bytes(trace_json_bytes(self.trace)),
        }


def _pitch_provenance(value: object) -> None:
    if not isinstance(value, ArtifactProvenance) or value.trainer != "pitch":
        raise ValueError("exploration requires pitch artifact provenance")


def readout_bytes(
    result: ExploratoryPitchEvaluation | ExploratoryPitchDiagnostics | ArtifactRunResult,
) -> bytes:
    """Encode a distinct readout without changing historical artifact/report bytes."""
    if not isinstance(
        result, (ExploratoryPitchEvaluation, ExploratoryPitchDiagnostics, ArtifactRunResult)
    ):
        raise ValueError("result must be an exploratory or artifact-run readout")
    return canonical_json_bytes(result.to_document())


def readout_from_bytes(
    content: bytes,
) -> ExploratoryPitchEvaluation | ExploratoryPitchDiagnostics | ArtifactRunResult:
    """Strict-decode a portable readout and forbid promotion to scientific evidence."""
    document = decode_json_bytes(content)
    schema = document.get("schema_id")
    if not isinstance(schema, str):
        raise ValueError("readout schema_id must be a string")
    payload_name = {
        EXPLORATORY_EVALUATION_SCHEMA_ID: "terminal_rows",
        EXPLORATORY_DIAGNOSTIC_SCHEMA_ID: "diagnostics",
        ARTIFACT_RUN_SCHEMA_ID: "trace",
    }.get(schema)
    if payload_name is None:
        raise ValueError("unsupported experiment readout schema_id")
    _object(document, {"schema_id", "provenance", "criterion", payload_name}, "readout")
    reason = (
        "single_episode_demonstration"
        if schema == ARTIFACT_RUN_SCHEMA_ID
        else "exploratory_single_artifact"
    )
    if canonical_json_bytes(document["criterion"]) != canonical_json_bytes(_criterion(reason)):
        raise ValueError("readout criterion must remain explicitly ineligible")
    provenance = ArtifactProvenance.from_document(document["provenance"])
    if schema == EXPLORATORY_EVALUATION_SCHEMA_ID:
        if not isinstance(document["terminal_rows"], list):
            raise ValueError("terminal_rows must be a list")
        result = ExploratoryPitchEvaluation(
            provenance,
            tuple(pitch_evaluation_row_from_document(row) for row in document["terminal_rows"]),
        )
    elif schema == EXPLORATORY_DIAGNOSTIC_SCHEMA_ID:
        result = ExploratoryPitchDiagnostics(
            provenance, diagnostic_report_from_document(document["diagnostics"])
        )
    else:
        result = ArtifactRunResult(provenance, trace_from_document(document["trace"]))
    if content != readout_bytes(result):
        raise ValueError("readout bytes must use canonical JSON encoding")
    return result
