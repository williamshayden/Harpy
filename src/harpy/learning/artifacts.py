"""Strict create-only local artifacts for learned sine policies."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import operator
import os
import re
import signal
import stat
import subprocess
import uuid
import weakref
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from harpy.learning.models import (
    ARCHITECTURE_SCHEMA_ID,
    ENVIRONMENT_CONTRACT_ID,
    ENVIRONMENT_ID,
    PREPROCESSING_SCHEMA_ID,
    PROFILE_CONFIGS,
    SPECTRUM_GRID_ID,
    BCEpochMetrics,
    BCProfile,
    BCTrainingSummary,
    DeviceName,
    EvaluationSuiteId,
    JSONValue,
    PPOProfile,
    PPOTrainingSummary,
    ProfileName,
    TrainerKind,
)

ARTIFACT_SCHEMA_VERSION = 1
PITCH_ARTIFACT_SCHEMA_VERSION = 2
TRAIN_DISTRIBUTION_ID = "harpy-sine-policy-train-v1"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_REQUIRED_SOURCE_INPUTS = (
    "src/harpy/learning/models.py",
    "src/harpy/learning/suites.py",
    "uv.lock",
)


@dataclass(frozen=True, slots=True)
class ArtifactSchemaRegistry:
    """Immutable codec identities owned only by one artifact schema."""

    schema_version: int
    trainers: tuple[str, ...]
    profiles: tuple[str, ...]
    suites: tuple[str, ...]
    architecture_ids: tuple[str, ...]
    preprocessing_ids: tuple[str, ...]


ARTIFACT_SCHEMA_V1_REGISTRY = ArtifactSchemaRegistry(
    schema_version=ARTIFACT_SCHEMA_VERSION,
    trainers=tuple(trainer.value for trainer in TrainerKind),
    profiles=tuple(profile.value for profile in ProfileName),
    suites=tuple(suite.value for suite in EvaluationSuiteId),
    architecture_ids=(ARCHITECTURE_SCHEMA_ID,),
    preprocessing_ids=(PREPROCESSING_SCHEMA_ID,),
)
ARTIFACT_SCHEMA_V1_REGISTRIES = MappingProxyType(
    {ARTIFACT_SCHEMA_VERSION: ARTIFACT_SCHEMA_V1_REGISTRY}
)


class ArtifactStatus(StrEnum):
    """The two monotonic artifact lifecycle states."""

    INCOMPLETE = "incomplete"
    COMPLETE = "complete"


class CriterionStatus(StrEnum):
    """Closed scientific-criterion status for one artifact."""

    INELIGIBLE = "ineligible"
    ELIGIBLE_FOR_AGGREGATE = "eligible_for_aggregate"
    CRITERION_MET = "criterion_met"
    CRITERION_NOT_MET = "criterion_not_met"


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


def _document_float(
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


def _string(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        suffix = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{field} must be {suffix}")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a bool")
    return value


def _enum[EnumT: StrEnum](enum_type: type[EnumT], value: object, field: str) -> EnumT:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a valid {enum_type.__name__}")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"{field} must be a valid {enum_type.__name__}") from error


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a string-keyed mapping")
    return value


def _exact_fields(mapping: Mapping[str, object], expected: set[str], field: str) -> None:
    if set(mapping) != expected:
        raise ValueError(f"{field} fields must be exactly {sorted(expected)}")


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _timestamp(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z") or "T" not in value:
        raise ValueError(f"{field} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 UTC timestamp ending in Z") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{field} must be an ISO-8601 UTC timestamp ending in Z")
    return value


def _relative_path(value: object, field: str = "relative_path") -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{field} must be a safe relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value in {".", ".."}
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field} must be a safe relative POSIX path")
    if path.as_posix() != value:
        raise ValueError(f"{field} must be a safe relative POSIX path")
    return value


def _known_suite_records(profile: ProfileName) -> tuple[tuple[EvaluationSuiteId, str], ...]:
    from harpy.learning.suites import fixed_evaluation_suite

    return tuple(
        (suite_id, fixed_evaluation_suite(suite_id).digest_sha256)
        for suite_id in PROFILE_CONFIGS[profile].evaluation_suites
    )


@dataclass(frozen=True, slots=True)
class FileRecord:
    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", _relative_path(self.relative_path))
        object.__setattr__(self, "size_bytes", _integer(self.size_bytes, "size_bytes", minimum=1))
        object.__setattr__(self, "sha256", _digest(self.sha256, "sha256"))


@dataclass(frozen=True, slots=True)
class SourceStatus:
    commit: str
    dirty_tree: bool
    tracked_diff_sha256: str
    dependency_lock_sha256: str
    required_inputs_committed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.commit, str) or _COMMIT_PATTERN.fullmatch(self.commit) is None:
            raise ValueError("commit must be a full lowercase Git object ID")
        object.__setattr__(self, "dirty_tree", _boolean(self.dirty_tree, "dirty_tree"))
        object.__setattr__(
            self,
            "tracked_diff_sha256",
            _digest(self.tracked_diff_sha256, "tracked_diff_sha256"),
        )
        object.__setattr__(
            self,
            "dependency_lock_sha256",
            _digest(self.dependency_lock_sha256, "dependency_lock_sha256"),
        )
        object.__setattr__(
            self,
            "required_inputs_committed",
            _boolean(self.required_inputs_committed, "required_inputs_committed"),
        )
        if not self.dirty_tree and self.tracked_diff_sha256 != _EMPTY_SHA256:
            raise ValueError("a clean source must have the empty tracked diff SHA-256")


@dataclass(frozen=True, slots=True)
class PackageSourceStatus:
    """Content identity for installed or archived code without checkout provenance.

    This source variant never qualifies for a frozen scientific criterion.  It
    deliberately makes no claim about a Git commit, clean tree, or lockfile.
    """

    distribution_name: str
    distribution_version: str | None
    package_sha256: str

    def __post_init__(self) -> None:
        if self.distribution_name != "harpy-audio":
            raise ValueError("distribution_name must be 'harpy-audio'")
        if self.distribution_version is not None:
            _string(self.distribution_version, "distribution_version")
        _digest(self.package_sha256, "package_sha256")


type SourceProvenance = SourceStatus | PackageSourceStatus


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    python_version: str
    platform: str
    processor: str
    numpy_version: str
    gymnasium_version: str
    torch_version: str
    stable_baselines3_version: str
    device: DeviceName
    device_description: str
    cuda_runtime_version: str | None
    cuda_driver_version: str | None

    def __post_init__(self) -> None:
        for field_name in (
            "python_version",
            "platform",
            "numpy_version",
            "gymnasium_version",
            "torch_version",
            "stable_baselines3_version",
            "device_description",
        ):
            object.__setattr__(
                self,
                field_name,
                _string(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self, "processor", _string(self.processor, "processor", allow_empty=True)
        )
        if not isinstance(self.device, DeviceName):
            raise ValueError("device must be a DeviceName")
        for field_name in ("cuda_runtime_version", "cuda_driver_version"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, _string(value, field_name))


@dataclass(frozen=True, slots=True)
class BCTrainingCounts:
    configured_training_episodes: int
    configured_validation_episodes: int
    training_examples: int | None
    validation_examples: int | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "configured_training_episodes",
            _integer(self.configured_training_episodes, "configured_training_episodes", minimum=1),
        )
        object.__setattr__(
            self,
            "configured_validation_episodes",
            _integer(
                self.configured_validation_episodes,
                "configured_validation_episodes",
                minimum=1,
            ),
        )
        if (self.training_examples is None) != (self.validation_examples is None):
            raise ValueError(
                "training_examples and validation_examples must both be set or both be None"
            )
        if self.training_examples is not None and self.validation_examples is not None:
            training_examples = _integer(self.training_examples, "training_examples", minimum=1)
            validation_examples = _integer(
                self.validation_examples, "validation_examples", minimum=1
            )
            if training_examples < self.configured_training_episodes:
                raise ValueError("training_examples must cover every configured training episode")
            if validation_examples < self.configured_validation_episodes:
                raise ValueError(
                    "validation_examples must cover every configured validation episode"
                )
            object.__setattr__(self, "training_examples", training_examples)
            object.__setattr__(self, "validation_examples", validation_examples)


@dataclass(frozen=True, slots=True)
class PPOTrainingCounts:
    requested_environment_steps: int
    completed_environment_steps: int | None

    def __post_init__(self) -> None:
        requested = _integer(
            self.requested_environment_steps,
            "requested_environment_steps",
            minimum=1,
        )
        object.__setattr__(self, "requested_environment_steps", requested)
        if self.completed_environment_steps is not None:
            completed = _integer(
                self.completed_environment_steps,
                "completed_environment_steps",
                minimum=1,
            )
            if completed != requested:
                raise ValueError(
                    "completed_environment_steps must equal requested_environment_steps"
                )
            object.__setattr__(self, "completed_environment_steps", completed)


type TrainingCounts = BCTrainingCounts | PPOTrainingCounts


@dataclass(frozen=True, slots=True)
class TrainingConfigDocument:
    schema_version: int
    trainer: TrainerKind
    profile: ProfileName
    seed: int
    device: DeviceName
    environment_id: str
    environment_contract_id: str
    train_distribution_id: str
    evaluation_suites: tuple[tuple[EvaluationSuiteId, str], ...]
    spectrum_grid_id: str
    preprocessing_schema_id: str
    architecture_schema_id: str
    profile_config: BCProfile | PPOProfile
    bc_training_digest_sha256: str | None
    bc_validation_digest_sha256: str | None

    def __post_init__(self) -> None:
        schema_version = _integer(self.schema_version, "schema_version", minimum=1)
        if schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {ARTIFACT_SCHEMA_VERSION}")
        object.__setattr__(self, "schema_version", schema_version)
        if not isinstance(self.trainer, TrainerKind):
            raise ValueError("trainer must be a TrainerKind")
        if not isinstance(self.profile, ProfileName):
            raise ValueError("profile must be a ProfileName")
        if not isinstance(self.device, DeviceName):
            raise ValueError("device must be a DeviceName")
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        _require_identity(self.environment_id, ENVIRONMENT_ID, "environment_id")
        _require_identity(
            self.environment_contract_id,
            ENVIRONMENT_CONTRACT_ID,
            "environment_contract_id",
        )
        _require_identity(
            self.train_distribution_id,
            TRAIN_DISTRIBUTION_ID,
            "train_distribution_id",
        )
        _require_identity(self.spectrum_grid_id, SPECTRUM_GRID_ID, "spectrum_grid_id")
        _require_identity(
            self.preprocessing_schema_id,
            PREPROCESSING_SCHEMA_ID,
            "preprocessing_schema_id",
        )
        _require_identity(
            self.architecture_schema_id,
            ARCHITECTURE_SCHEMA_ID,
            "architecture_schema_id",
        )
        suites = _normalize_config_suites(self.evaluation_suites)
        if suites != _known_suite_records(self.profile):
            raise ValueError(
                "evaluation_suites must match the checked-in profile suites and digests"
            )
        object.__setattr__(self, "evaluation_suites", suites)
        expected_profile = PROFILE_CONFIGS[self.profile]
        if self.trainer is TrainerKind.BC:
            if (
                not isinstance(self.profile_config, BCProfile)
                or self.profile_config != expected_profile.bc
            ):
                raise ValueError("profile_config must be the checked-in BC profile")
            if self.bc_training_digest_sha256 is None or self.bc_validation_digest_sha256 is None:
                raise ValueError("BC split digest fields must both be SHA-256 digests")
            object.__setattr__(
                self,
                "bc_training_digest_sha256",
                _digest(self.bc_training_digest_sha256, "bc_training_digest_sha256"),
            )
            object.__setattr__(
                self,
                "bc_validation_digest_sha256",
                _digest(self.bc_validation_digest_sha256, "bc_validation_digest_sha256"),
            )
        else:
            if (
                not isinstance(self.profile_config, PPOProfile)
                or self.profile_config != expected_profile.ppo
            ):
                raise ValueError("profile_config must be the checked-in PPO profile")
            if (
                self.bc_training_digest_sha256 is not None
                or self.bc_validation_digest_sha256 is not None
            ):
                raise ValueError("PPO config must not contain BC split digest fields")

    def to_document(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "trainer": self.trainer.value,
            "profile": self.profile.value,
            "seed": self.seed,
            "device": self.device.value,
            "environment_id": self.environment_id,
            "environment_contract_id": self.environment_contract_id,
            "train_distribution_id": self.train_distribution_id,
            "evaluation_suites": [
                {"suite_id": suite_id.value, "digest_sha256": digest}
                for suite_id, digest in self.evaluation_suites
            ],
            "spectrum_grid_id": self.spectrum_grid_id,
            "preprocessing_schema_id": self.preprocessing_schema_id,
            "architecture_schema_id": self.architecture_schema_id,
            "profile_config": _profile_to_document(self.profile_config),
            "bc_training_digest_sha256": self.bc_training_digest_sha256,
            "bc_validation_digest_sha256": self.bc_validation_digest_sha256,
        }

    @classmethod
    def from_document(cls, document: Mapping[str, JSONValue]) -> TrainingConfigDocument:
        mapping = _mapping(document, "training config")
        _exact_fields(
            mapping,
            {
                "schema_version",
                "trainer",
                "profile",
                "seed",
                "device",
                "environment_id",
                "environment_contract_id",
                "train_distribution_id",
                "evaluation_suites",
                "spectrum_grid_id",
                "preprocessing_schema_id",
                "architecture_schema_id",
                "profile_config",
                "bc_training_digest_sha256",
                "bc_validation_digest_sha256",
            },
            "training config",
        )
        trainer = _enum(TrainerKind, mapping["trainer"], "trainer")
        suites_value = mapping["evaluation_suites"]
        if not isinstance(suites_value, list):
            raise ValueError("evaluation_suites must be a list")
        return cls(
            schema_version=_integer(mapping["schema_version"], "schema_version", minimum=1),
            trainer=trainer,
            profile=_enum(ProfileName, mapping["profile"], "profile"),
            seed=_integer(mapping["seed"], "seed"),
            device=_enum(DeviceName, mapping["device"], "device"),
            environment_id=_string(mapping["environment_id"], "environment_id"),
            environment_contract_id=_string(
                mapping["environment_contract_id"], "environment_contract_id"
            ),
            train_distribution_id=_string(
                mapping["train_distribution_id"], "train_distribution_id"
            ),
            evaluation_suites=tuple(_config_suite_from_document(value) for value in suites_value),
            spectrum_grid_id=_string(mapping["spectrum_grid_id"], "spectrum_grid_id"),
            preprocessing_schema_id=_string(
                mapping["preprocessing_schema_id"], "preprocessing_schema_id"
            ),
            architecture_schema_id=_string(
                mapping["architecture_schema_id"], "architecture_schema_id"
            ),
            profile_config=_profile_from_document(trainer, mapping["profile_config"]),
            bc_training_digest_sha256=(
                None
                if mapping["bc_training_digest_sha256"] is None
                else _digest(mapping["bc_training_digest_sha256"], "bc_training_digest_sha256")
            ),
            bc_validation_digest_sha256=(
                None
                if mapping["bc_validation_digest_sha256"] is None
                else _digest(mapping["bc_validation_digest_sha256"], "bc_validation_digest_sha256")
            ),
        )


@dataclass(frozen=True, slots=True)
class TrainingSummaryDocument:
    schema_version: int
    trainer: TrainerKind
    profile: ProfileName
    seed: int
    summary: BCTrainingSummary | PPOTrainingSummary

    def __post_init__(self) -> None:
        schema_version = _integer(self.schema_version, "schema_version", minimum=1)
        if schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {ARTIFACT_SCHEMA_VERSION}")
        object.__setattr__(self, "schema_version", schema_version)
        if not isinstance(self.trainer, TrainerKind):
            raise ValueError("trainer must be a TrainerKind")
        if not isinstance(self.profile, ProfileName):
            raise ValueError("profile must be a ProfileName")
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        if self.trainer is TrainerKind.BC:
            if not isinstance(self.summary, BCTrainingSummary):
                raise ValueError("summary must be a BCTrainingSummary for trainer bc")
            if len(self.summary.history) > PROFILE_CONFIGS[self.profile].bc.max_epochs:
                raise ValueError("BC summary history exceeds the checked-in max_epochs")
        else:
            if not isinstance(self.summary, PPOTrainingSummary):
                raise ValueError("summary must be a PPOTrainingSummary for trainer ppo")
            if (
                self.summary.requested_environment_steps
                != PROFILE_CONFIGS[self.profile].ppo.total_timesteps
            ):
                raise ValueError("PPO summary steps must match the checked-in profile")

    def to_document(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "trainer": self.trainer.value,
            "profile": self.profile.value,
            "seed": self.seed,
            "summary": _summary_to_document(self.summary),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, JSONValue]) -> TrainingSummaryDocument:
        mapping = _mapping(document, "training summary")
        _exact_fields(
            mapping,
            {"schema_version", "trainer", "profile", "seed", "summary"},
            "training summary",
        )
        trainer = _enum(TrainerKind, mapping["trainer"], "trainer")
        return cls(
            schema_version=_integer(mapping["schema_version"], "schema_version", minimum=1),
            trainer=trainer,
            profile=_enum(ProfileName, mapping["profile"], "profile"),
            seed=_integer(mapping["seed"], "seed"),
            summary=_summary_from_document(trainer, mapping["summary"]),
        )


@dataclass(frozen=True, slots=True)
class ArtifactCompletion:
    completed_at_utc: str
    training_counts: TrainingCounts
    evaluation_device: DeviceName
    bc_criterion_met: bool | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "completed_at_utc",
            _timestamp(self.completed_at_utc, "completed_at_utc"),
        )
        if not isinstance(self.training_counts, (BCTrainingCounts, PPOTrainingCounts)):
            raise ValueError("training_counts must be BCTrainingCounts or PPOTrainingCounts")
        if not isinstance(self.evaluation_device, DeviceName):
            raise ValueError("evaluation_device must be a DeviceName")
        if self.bc_criterion_met is not None:
            object.__setattr__(
                self,
                "bc_criterion_met",
                _boolean(self.bc_criterion_met, "bc_criterion_met"),
            )


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    schema_version: int
    status: ArtifactStatus
    trainer: TrainerKind
    profile: ProfileName
    seed: int
    created_at_utc: str
    completed_at_utc: str | None
    source: SourceProvenance
    runtime: RuntimeStatus
    environment_id: str
    environment_contract_id: str
    train_distribution_id: str
    evaluation_suites: tuple[tuple[str, str], ...]
    spectrum_grid_id: str
    preprocessing_schema_id: str
    architecture_schema_id: str
    parameter_count: int
    training_counts: TrainingCounts
    evaluation_device: DeviceName | None
    criterion_eligible: bool
    criterion_status: CriterionStatus
    criterion_met: bool | None
    files: tuple[FileRecord, ...]

    def __post_init__(self) -> None:
        schema_version = _integer(self.schema_version, "schema_version", minimum=1)
        if schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {ARTIFACT_SCHEMA_VERSION}")
        object.__setattr__(self, "schema_version", schema_version)
        if not isinstance(self.status, ArtifactStatus):
            raise ValueError("status must be an ArtifactStatus")
        if not isinstance(self.trainer, TrainerKind):
            raise ValueError("trainer must be a TrainerKind")
        if not isinstance(self.profile, ProfileName):
            raise ValueError("profile must be a ProfileName")
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        object.__setattr__(
            self, "created_at_utc", _timestamp(self.created_at_utc, "created_at_utc")
        )
        if self.completed_at_utc is not None:
            object.__setattr__(
                self,
                "completed_at_utc",
                _timestamp(self.completed_at_utc, "completed_at_utc"),
            )
        if not isinstance(self.source, SourceStatus | PackageSourceStatus):
            raise ValueError("source must be a SourceStatus or PackageSourceStatus")
        if not isinstance(self.runtime, RuntimeStatus):
            raise ValueError("runtime must be a RuntimeStatus")
        _require_identity(self.environment_id, ENVIRONMENT_ID, "environment_id")
        _require_identity(
            self.environment_contract_id,
            ENVIRONMENT_CONTRACT_ID,
            "environment_contract_id",
        )
        _require_identity(
            self.train_distribution_id,
            TRAIN_DISTRIBUTION_ID,
            "train_distribution_id",
        )
        _require_identity(self.spectrum_grid_id, SPECTRUM_GRID_ID, "spectrum_grid_id")
        _require_identity(
            self.preprocessing_schema_id,
            PREPROCESSING_SCHEMA_ID,
            "preprocessing_schema_id",
        )
        _require_identity(
            self.architecture_schema_id,
            ARCHITECTURE_SCHEMA_ID,
            "architecture_schema_id",
        )
        suites = _normalize_manifest_suites(self.evaluation_suites)
        expected_suites = tuple(
            (suite_id.value, digest) for suite_id, digest in _known_suite_records(self.profile)
        )
        if suites != expected_suites:
            raise ValueError(
                "evaluation_suites must match the checked-in profile suites and digests"
            )
        object.__setattr__(self, "evaluation_suites", suites)
        object.__setattr__(
            self,
            "parameter_count",
            _integer(self.parameter_count, "parameter_count", minimum=1),
        )
        _validate_manifest_counts(self)
        if self.evaluation_device is not None and not isinstance(
            self.evaluation_device, DeviceName
        ):
            raise ValueError("evaluation_device must be a DeviceName or None")
        object.__setattr__(
            self,
            "criterion_eligible",
            _boolean(self.criterion_eligible, "criterion_eligible"),
        )
        if not isinstance(self.criterion_status, CriterionStatus):
            raise ValueError("criterion_status must be a CriterionStatus")
        if self.criterion_met is not None:
            object.__setattr__(self, "criterion_met", _boolean(self.criterion_met, "criterion_met"))
        try:
            files = tuple(self.files)
        except TypeError as error:
            raise ValueError("files must contain FileRecord values") from error
        if not all(isinstance(record, FileRecord) for record in files):
            raise ValueError("files must contain FileRecord values")
        if len({record.relative_path for record in files}) != len(files):
            raise ValueError("files must not contain duplicate relative paths")
        object.__setattr__(self, "files", files)
        _validate_manifest_lifecycle(self)

    def to_document(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "trainer": self.trainer.value,
            "profile": self.profile.value,
            "seed": self.seed,
            "created_at_utc": self.created_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "source": _source_to_document(self.source),
            "runtime": _runtime_to_document(self.runtime),
            "environment_id": self.environment_id,
            "environment_contract_id": self.environment_contract_id,
            "train_distribution_id": self.train_distribution_id,
            "evaluation_suites": [
                {"suite_id": suite_id, "digest_sha256": digest}
                for suite_id, digest in self.evaluation_suites
            ],
            "spectrum_grid_id": self.spectrum_grid_id,
            "preprocessing_schema_id": self.preprocessing_schema_id,
            "architecture_schema_id": self.architecture_schema_id,
            "parameter_count": self.parameter_count,
            "training_counts": _counts_to_document(self.training_counts),
            "evaluation_device": (
                None if self.evaluation_device is None else self.evaluation_device.value
            ),
            "criterion_eligible": self.criterion_eligible,
            "criterion_status": self.criterion_status.value,
            "criterion_met": self.criterion_met,
            "files": [_file_to_document(record) for record in self.files],
        }

    @classmethod
    def from_document(cls, document: Mapping[str, JSONValue]) -> ArtifactManifest:
        mapping = _mapping(document, "artifact manifest")
        _exact_fields(
            mapping,
            {
                "schema_version",
                "status",
                "trainer",
                "profile",
                "seed",
                "created_at_utc",
                "completed_at_utc",
                "source",
                "runtime",
                "environment_id",
                "environment_contract_id",
                "train_distribution_id",
                "evaluation_suites",
                "spectrum_grid_id",
                "preprocessing_schema_id",
                "architecture_schema_id",
                "parameter_count",
                "training_counts",
                "evaluation_device",
                "criterion_eligible",
                "criterion_status",
                "criterion_met",
                "files",
            },
            "artifact manifest",
        )
        trainer = _enum(TrainerKind, mapping["trainer"], "trainer")
        suites_value = mapping["evaluation_suites"]
        files_value = mapping["files"]
        if not isinstance(suites_value, list):
            raise ValueError("evaluation_suites must be a list")
        if not isinstance(files_value, list):
            raise ValueError("files must be a list")
        return cls(
            schema_version=_integer(mapping["schema_version"], "schema_version", minimum=1),
            status=_enum(ArtifactStatus, mapping["status"], "status"),
            trainer=trainer,
            profile=_enum(ProfileName, mapping["profile"], "profile"),
            seed=_integer(mapping["seed"], "seed"),
            created_at_utc=_timestamp(mapping["created_at_utc"], "created_at_utc"),
            completed_at_utc=(
                None
                if mapping["completed_at_utc"] is None
                else _timestamp(mapping["completed_at_utc"], "completed_at_utc")
            ),
            source=_source_from_document(mapping["source"]),
            runtime=_runtime_from_document(mapping["runtime"]),
            environment_id=_string(mapping["environment_id"], "environment_id"),
            environment_contract_id=_string(
                mapping["environment_contract_id"], "environment_contract_id"
            ),
            train_distribution_id=_string(
                mapping["train_distribution_id"], "train_distribution_id"
            ),
            evaluation_suites=tuple(_manifest_suite_from_document(value) for value in suites_value),
            spectrum_grid_id=_string(mapping["spectrum_grid_id"], "spectrum_grid_id"),
            preprocessing_schema_id=_string(
                mapping["preprocessing_schema_id"], "preprocessing_schema_id"
            ),
            architecture_schema_id=_string(
                mapping["architecture_schema_id"], "architecture_schema_id"
            ),
            parameter_count=_integer(mapping["parameter_count"], "parameter_count", minimum=1),
            training_counts=_counts_from_document(trainer, mapping["training_counts"]),
            evaluation_device=(
                None
                if mapping["evaluation_device"] is None
                else _enum(DeviceName, mapping["evaluation_device"], "evaluation_device")
            ),
            criterion_eligible=_boolean(mapping["criterion_eligible"], "criterion_eligible"),
            criterion_status=_enum(
                CriterionStatus, mapping["criterion_status"], "criterion_status"
            ),
            criterion_met=(
                None
                if mapping["criterion_met"] is None
                else _boolean(mapping["criterion_met"], "criterion_met")
            ),
            files=tuple(_file_from_document(value) for value in files_value),
        )


def _require_identity(value: object, expected: str, field: str) -> None:
    if value != expected:
        raise ValueError(f"{field} must be {expected!r}")


def _normalize_config_suites(
    value: object,
) -> tuple[tuple[EvaluationSuiteId, str], ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("evaluation_suites must contain suite ID/digest pairs")
    try:
        suites = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("evaluation_suites must contain suite ID/digest pairs") from error
    normalized: list[tuple[EvaluationSuiteId, str]] = []
    for pair in suites:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("evaluation_suites must contain suite ID/digest pairs")
        suite_id, digest = pair
        if not isinstance(suite_id, EvaluationSuiteId):
            raise ValueError("evaluation_suites must contain EvaluationSuiteId values")
        normalized.append((suite_id, _digest(digest, "evaluation suite digest")))
    if len({suite_id for suite_id, _ in normalized}) != len(normalized):
        raise ValueError("evaluation_suites must not contain duplicates")
    return tuple(normalized)


def _normalize_manifest_suites(value: object) -> tuple[tuple[str, str], ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("evaluation_suites must contain suite ID/digest pairs")
    try:
        suites = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("evaluation_suites must contain suite ID/digest pairs") from error
    normalized: list[tuple[str, str]] = []
    for pair in suites:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("evaluation_suites must contain suite ID/digest pairs")
        raw_suite_id, digest = pair
        suite_id = _enum(EvaluationSuiteId, raw_suite_id, "evaluation suite ID")
        normalized.append((suite_id.value, _digest(digest, "evaluation suite digest")))
    if len({suite_id for suite_id, _ in normalized}) != len(normalized):
        raise ValueError("evaluation_suites must not contain duplicates")
    return tuple(normalized)


def _profile_to_document(profile: BCProfile | PPOProfile) -> dict[str, JSONValue]:
    if isinstance(profile, BCProfile):
        return {
            "train_episodes": profile.train_episodes,
            "validation_episodes": profile.validation_episodes,
            "max_epochs": profile.max_epochs,
            "early_stopping_patience": profile.early_stopping_patience,
            "batch_size": profile.batch_size,
            "optimizer": profile.optimizer,
            "learning_rate": profile.learning_rate,
            "weight_decay": profile.weight_decay,
        }
    return {
        "total_timesteps": profile.total_timesteps,
        "n_steps": profile.n_steps,
        "batch_size": profile.batch_size,
        "n_epochs": profile.n_epochs,
        "learning_rate": profile.learning_rate,
        "gamma": profile.gamma,
        "gae_lambda": profile.gae_lambda,
        "clip_range": profile.clip_range,
        "ent_coef": profile.ent_coef,
        "vf_coef": profile.vf_coef,
    }


def _profile_from_document(trainer: TrainerKind, value: object) -> BCProfile | PPOProfile:
    mapping = _mapping(value, "profile_config")
    if trainer is TrainerKind.BC:
        fields = {
            "train_episodes",
            "validation_episodes",
            "max_epochs",
            "early_stopping_patience",
            "batch_size",
            "optimizer",
            "learning_rate",
            "weight_decay",
        }
        _exact_fields(mapping, fields, "profile_config")
        return BCProfile(
            train_episodes=_integer(mapping["train_episodes"], "train_episodes", minimum=1),
            validation_episodes=_integer(
                mapping["validation_episodes"], "validation_episodes", minimum=1
            ),
            max_epochs=_integer(mapping["max_epochs"], "max_epochs", minimum=1),
            early_stopping_patience=(
                None
                if mapping["early_stopping_patience"] is None
                else _integer(
                    mapping["early_stopping_patience"],
                    "early_stopping_patience",
                    minimum=1,
                )
            ),
            batch_size=_integer(mapping["batch_size"], "batch_size", minimum=1),
            optimizer=_string(mapping["optimizer"], "optimizer"),
            learning_rate=_document_float(mapping["learning_rate"], "learning_rate", minimum=0.0),
            weight_decay=_document_float(mapping["weight_decay"], "weight_decay", minimum=0.0),
        )
    fields = {
        "total_timesteps",
        "n_steps",
        "batch_size",
        "n_epochs",
        "learning_rate",
        "gamma",
        "gae_lambda",
        "clip_range",
        "ent_coef",
        "vf_coef",
    }
    _exact_fields(mapping, fields, "profile_config")
    return PPOProfile(
        total_timesteps=_integer(mapping["total_timesteps"], "total_timesteps", minimum=1),
        n_steps=_integer(mapping["n_steps"], "n_steps", minimum=1),
        batch_size=_integer(mapping["batch_size"], "batch_size", minimum=1),
        n_epochs=_integer(mapping["n_epochs"], "n_epochs", minimum=1),
        learning_rate=_document_float(mapping["learning_rate"], "learning_rate", minimum=0.0),
        gamma=_document_float(mapping["gamma"], "gamma", minimum=0.0, maximum=1.0),
        gae_lambda=_document_float(mapping["gae_lambda"], "gae_lambda", minimum=0.0, maximum=1.0),
        clip_range=_document_float(mapping["clip_range"], "clip_range", minimum=0.0),
        ent_coef=_document_float(mapping["ent_coef"], "ent_coef", minimum=0.0),
        vf_coef=_document_float(mapping["vf_coef"], "vf_coef", minimum=0.0),
    )


def _summary_to_document(summary: BCTrainingSummary | PPOTrainingSummary) -> dict[str, JSONValue]:
    if isinstance(summary, BCTrainingSummary):
        return {
            "history": [
                {
                    "epoch": metric.epoch,
                    "training_loss": metric.training_loss,
                    "validation_loss": metric.validation_loss,
                    "validation_accuracy": metric.validation_accuracy,
                }
                for metric in summary.history
            ],
            "selected_epoch": summary.selected_epoch,
            "training_examples": summary.training_examples,
            "validation_examples": summary.validation_examples,
            "training_wall_time_seconds": summary.training_wall_time_seconds,
        }
    return {
        "requested_environment_steps": summary.requested_environment_steps,
        "completed_environment_steps": summary.completed_environment_steps,
        "training_wall_time_seconds": summary.training_wall_time_seconds,
    }


def _summary_from_document(
    trainer: TrainerKind, value: object
) -> BCTrainingSummary | PPOTrainingSummary:
    mapping = _mapping(value, "summary")
    if trainer is TrainerKind.BC:
        _exact_fields(
            mapping,
            {
                "history",
                "selected_epoch",
                "training_examples",
                "validation_examples",
                "training_wall_time_seconds",
            },
            "summary",
        )
        history_value = mapping["history"]
        if not isinstance(history_value, list):
            raise ValueError("history must be a list")
        return BCTrainingSummary(
            history=tuple(_epoch_from_document(item) for item in history_value),
            selected_epoch=_integer(mapping["selected_epoch"], "selected_epoch", minimum=1),
            training_examples=_integer(
                mapping["training_examples"], "training_examples", minimum=1
            ),
            validation_examples=_integer(
                mapping["validation_examples"], "validation_examples", minimum=1
            ),
            training_wall_time_seconds=_document_float(
                mapping["training_wall_time_seconds"],
                "training_wall_time_seconds",
                minimum=0.0,
            ),
        )
    _exact_fields(
        mapping,
        {
            "requested_environment_steps",
            "completed_environment_steps",
            "training_wall_time_seconds",
        },
        "summary",
    )
    return PPOTrainingSummary(
        requested_environment_steps=_integer(
            mapping["requested_environment_steps"], "requested_environment_steps", minimum=1
        ),
        completed_environment_steps=_integer(
            mapping["completed_environment_steps"], "completed_environment_steps", minimum=1
        ),
        training_wall_time_seconds=_document_float(
            mapping["training_wall_time_seconds"], "training_wall_time_seconds", minimum=0.0
        ),
    )


def _epoch_from_document(value: object) -> BCEpochMetrics:
    mapping = _mapping(value, "epoch metrics")
    _exact_fields(
        mapping,
        {"epoch", "training_loss", "validation_loss", "validation_accuracy"},
        "epoch metrics",
    )
    return BCEpochMetrics(
        epoch=_integer(mapping["epoch"], "epoch", minimum=1),
        training_loss=_document_float(mapping["training_loss"], "training_loss", minimum=0.0),
        validation_loss=_document_float(mapping["validation_loss"], "validation_loss", minimum=0.0),
        validation_accuracy=_document_float(
            mapping["validation_accuracy"],
            "validation_accuracy",
            minimum=0.0,
            maximum=1.0,
        ),
    )


def _config_suite_from_document(value: object) -> tuple[EvaluationSuiteId, str]:
    mapping = _mapping(value, "evaluation suite")
    _exact_fields(mapping, {"suite_id", "digest_sha256"}, "evaluation suite")
    return (
        _enum(EvaluationSuiteId, mapping["suite_id"], "suite_id"),
        _digest(mapping["digest_sha256"], "digest_sha256"),
    )


def _manifest_suite_from_document(value: object) -> tuple[str, str]:
    suite_id, digest = _config_suite_from_document(value)
    return suite_id.value, digest


def _source_to_document(source: SourceProvenance) -> dict[str, JSONValue]:
    if isinstance(source, PackageSourceStatus):
        return {
            "source_kind": "package_snapshot",
            "distribution_name": source.distribution_name,
            "distribution_version": source.distribution_version,
            "package_sha256": source.package_sha256,
        }
    return {
        "commit": source.commit,
        "dirty_tree": source.dirty_tree,
        "tracked_diff_sha256": source.tracked_diff_sha256,
        "dependency_lock_sha256": source.dependency_lock_sha256,
        "required_inputs_committed": source.required_inputs_committed,
    }


def _source_from_document(value: object) -> SourceProvenance:
    mapping = _mapping(value, "source")
    if mapping.get("source_kind") == "package_snapshot":
        _exact_fields(
            mapping,
            {"source_kind", "distribution_name", "distribution_version", "package_sha256"},
            "package source",
        )
        return PackageSourceStatus(
            distribution_name=_string(mapping["distribution_name"], "distribution_name"),
            distribution_version=(
                None
                if mapping["distribution_version"] is None
                else _string(mapping["distribution_version"], "distribution_version")
            ),
            package_sha256=_digest(mapping["package_sha256"], "package_sha256"),
        )
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
        commit=_string(mapping["commit"], "commit"),
        dirty_tree=_boolean(mapping["dirty_tree"], "dirty_tree"),
        tracked_diff_sha256=_digest(mapping["tracked_diff_sha256"], "tracked_diff_sha256"),
        dependency_lock_sha256=_digest(mapping["dependency_lock_sha256"], "dependency_lock_sha256"),
        required_inputs_committed=_boolean(
            mapping["required_inputs_committed"], "required_inputs_committed"
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
    _exact_fields(
        mapping,
        {
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
        },
        "runtime",
    )
    return RuntimeStatus(
        python_version=_string(mapping["python_version"], "python_version"),
        platform=_string(mapping["platform"], "platform"),
        processor=_string(mapping["processor"], "processor", allow_empty=True),
        numpy_version=_string(mapping["numpy_version"], "numpy_version"),
        gymnasium_version=_string(mapping["gymnasium_version"], "gymnasium_version"),
        torch_version=_string(mapping["torch_version"], "torch_version"),
        stable_baselines3_version=_string(
            mapping["stable_baselines3_version"], "stable_baselines3_version"
        ),
        device=_enum(DeviceName, mapping["device"], "device"),
        device_description=_string(mapping["device_description"], "device_description"),
        cuda_runtime_version=(
            None
            if mapping["cuda_runtime_version"] is None
            else _string(mapping["cuda_runtime_version"], "cuda_runtime_version")
        ),
        cuda_driver_version=(
            None
            if mapping["cuda_driver_version"] is None
            else _string(mapping["cuda_driver_version"], "cuda_driver_version")
        ),
    )


def _counts_to_document(counts: TrainingCounts) -> dict[str, JSONValue]:
    if isinstance(counts, BCTrainingCounts):
        return {
            "configured_training_episodes": counts.configured_training_episodes,
            "configured_validation_episodes": counts.configured_validation_episodes,
            "training_examples": counts.training_examples,
            "validation_examples": counts.validation_examples,
        }
    return {
        "requested_environment_steps": counts.requested_environment_steps,
        "completed_environment_steps": counts.completed_environment_steps,
    }


def _counts_from_document(trainer: TrainerKind, value: object) -> TrainingCounts:
    mapping = _mapping(value, "training_counts")
    if trainer is TrainerKind.BC:
        _exact_fields(
            mapping,
            {
                "configured_training_episodes",
                "configured_validation_episodes",
                "training_examples",
                "validation_examples",
            },
            "training_counts",
        )
        return BCTrainingCounts(
            configured_training_episodes=_integer(
                mapping["configured_training_episodes"],
                "configured_training_episodes",
                minimum=1,
            ),
            configured_validation_episodes=_integer(
                mapping["configured_validation_episodes"],
                "configured_validation_episodes",
                minimum=1,
            ),
            training_examples=(
                None
                if mapping["training_examples"] is None
                else _integer(mapping["training_examples"], "training_examples", minimum=1)
            ),
            validation_examples=(
                None
                if mapping["validation_examples"] is None
                else _integer(mapping["validation_examples"], "validation_examples", minimum=1)
            ),
        )
    _exact_fields(
        mapping,
        {"requested_environment_steps", "completed_environment_steps"},
        "training_counts",
    )
    return PPOTrainingCounts(
        requested_environment_steps=_integer(
            mapping["requested_environment_steps"], "requested_environment_steps", minimum=1
        ),
        completed_environment_steps=(
            None
            if mapping["completed_environment_steps"] is None
            else _integer(
                mapping["completed_environment_steps"],
                "completed_environment_steps",
                minimum=1,
            )
        ),
    )


def _file_to_document(record: FileRecord) -> dict[str, JSONValue]:
    return {
        "relative_path": record.relative_path,
        "size_bytes": record.size_bytes,
        "sha256": record.sha256,
    }


def _file_from_document(value: object) -> FileRecord:
    mapping = _mapping(value, "file record")
    _exact_fields(mapping, {"relative_path", "size_bytes", "sha256"}, "file record")
    return FileRecord(
        relative_path=_relative_path(mapping["relative_path"]),
        size_bytes=_integer(mapping["size_bytes"], "size_bytes", minimum=1),
        sha256=_digest(mapping["sha256"], "sha256"),
    )


def _validate_manifest_counts(manifest: ArtifactManifest) -> None:
    profile = PROFILE_CONFIGS[manifest.profile]
    if manifest.trainer is TrainerKind.BC:
        if not isinstance(manifest.training_counts, BCTrainingCounts):
            raise ValueError("training_counts must be BCTrainingCounts for trainer bc")
        if (
            manifest.training_counts.configured_training_episodes != profile.bc.train_episodes
            or manifest.training_counts.configured_validation_episodes
            != profile.bc.validation_episodes
        ):
            raise ValueError(
                "configured_training_episodes and configured_validation_episodes must match profile"
            )
    else:
        if not isinstance(manifest.training_counts, PPOTrainingCounts):
            raise ValueError("training_counts must be PPOTrainingCounts for trainer ppo")
        if manifest.training_counts.requested_environment_steps != profile.ppo.total_timesteps:
            raise ValueError("requested_environment_steps must match profile")


def _criterion_eligibility(manifest: ArtifactManifest) -> bool:
    declared_seed = (
        manifest.seed == 0 if manifest.trainer is TrainerKind.BC else manifest.seed in range(5)
    )
    return (
        manifest.profile is ProfileName.CHECKPOINT
        and declared_seed
        and isinstance(manifest.source, SourceStatus)
        and not manifest.source.dirty_tree
        and manifest.source.required_inputs_committed
        and manifest.runtime.device is DeviceName.CPU
        and manifest.evaluation_device is DeviceName.CPU
    )


def _validate_manifest_lifecycle(manifest: ArtifactManifest) -> None:
    counts = manifest.training_counts
    if manifest.status is ArtifactStatus.INCOMPLETE:
        if (
            manifest.completed_at_utc is not None
            or manifest.evaluation_device is not None
            or manifest.criterion_eligible
            or manifest.criterion_status is not CriterionStatus.INELIGIBLE
            or manifest.criterion_met is not None
            or manifest.files
        ):
            raise ValueError("incomplete manifest must not contain completion claims or files")
        if isinstance(counts, BCTrainingCounts):
            if counts.training_examples is not None or counts.validation_examples is not None:
                raise ValueError("incomplete BC manifest must not contain actual example counts")
        elif counts.completed_environment_steps is not None:
            raise ValueError("incomplete PPO manifest must not contain completed steps")
        return
    if manifest.completed_at_utc is None or manifest.evaluation_device is None:
        raise ValueError("complete manifest requires completion time and evaluation device")
    if datetime.fromisoformat(manifest.completed_at_utc[:-1]) < datetime.fromisoformat(
        manifest.created_at_utc[:-1]
    ):
        raise ValueError("completed_at_utc must not precede created_at_utc")
    if isinstance(counts, BCTrainingCounts):
        if counts.training_examples is None or counts.validation_examples is None:
            raise ValueError("complete BC manifest requires actual example counts")
    elif counts.completed_environment_steps is None:
        raise ValueError("complete PPO manifest requires completed steps")
    expected_paths = required_payload_names(manifest.trainer, manifest.profile)
    if tuple(record.relative_path for record in manifest.files) != expected_paths:
        raise ValueError("complete manifest files must match the exact required inventory")
    expected_eligible = _criterion_eligibility(manifest)
    if manifest.criterion_eligible != expected_eligible:
        raise ValueError("criterion_eligible must be derived from artifact provenance and devices")
    if not expected_eligible:
        if (
            manifest.criterion_status is not CriterionStatus.INELIGIBLE
            or manifest.criterion_met is not None
        ):
            raise ValueError("ineligible artifact must not contain a criterion result")
    elif manifest.trainer is TrainerKind.PPO:
        if (
            manifest.criterion_status is not CriterionStatus.ELIGIBLE_FOR_AGGREGATE
            or manifest.criterion_met is not None
        ):
            raise ValueError("eligible PPO artifact belongs to the later aggregate criterion")
    elif manifest.criterion_met is True:
        if manifest.criterion_status is not CriterionStatus.CRITERION_MET:
            raise ValueError("eligible BC criterion status must match criterion_met")
    elif manifest.criterion_met is False:
        if manifest.criterion_status is not CriterionStatus.CRITERION_NOT_MET:
            raise ValueError("eligible BC criterion status must match criterion_met")
    else:
        raise ValueError("eligible BC artifact requires its single-artifact criterion result")


def required_payload_names(trainer: TrainerKind, profile: ProfileName) -> tuple[str, ...]:
    if not isinstance(trainer, TrainerKind):
        raise ValueError("trainer must be a TrainerKind")
    if not isinstance(profile, ProfileName):
        raise ValueError("profile must be a ProfileName")
    model = "model.pt" if trainer is TrainerKind.BC else "model.zip"
    evaluations = (
        ("evaluation-smoke.json",)
        if profile is ProfileName.SMOKE
        else ("evaluation-iid.json", "evaluation-ood.json")
    )
    return ("training-config.json", "training-summary.json", model, *evaluations)


def _owned_json(value: object, field: str) -> JSONValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field} must contain only finite JSON numbers")
        return value
    if isinstance(value, list):
        return [_owned_json(item, field) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{field} must contain only string-keyed JSON objects")
        return {key: _owned_json(item, field) for key, item in value.items()}
    raise ValueError(f"{field} must contain only JSON values")


def canonical_json_bytes(document: Mapping[str, JSONValue]) -> bytes:
    """Encode one finite JSON object in the artifact's canonical byte form."""

    normalized = _owned_json(document, "document")
    if not isinstance(normalized, dict):
        raise ValueError("document must be a JSON object")
    return (
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _reject_constant(value: str) -> object:
    raise ValueError(f"JSON number {value} must be finite")


def _pairs_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def decode_json_bytes(content: bytes) -> dict[str, JSONValue]:
    """Decode one duplicate-free finite UTF-8 JSON object."""

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("content must be valid UTF-8") from error
    try:
        parsed: object = json.loads(
            text,
            object_pairs_hook=_pairs_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("content must be valid JSON") from error
    normalized = _owned_json(parsed, "JSON document")
    if not isinstance(normalized, dict):
        raise ValueError("JSON document must be an object")
    return normalized


def read_json_document(path: Path) -> dict[str, JSONValue]:
    """Read one artifact JSON document with duplicate and finiteness checks."""

    return decode_json_bytes(path.read_bytes())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_file(path: Path, relative_path: str) -> FileRecord:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact payload {relative_path!r} must be a regular file")
    size = path.stat().st_size
    return FileRecord(
        relative_path=relative_path,
        size_bytes=size,
        sha256=_sha256_file(path),
    )


def _verify_file_record(root: Path, record: FileRecord) -> Path:
    path = _contained_file(root, record.relative_path)
    size = path.stat().st_size
    if size != record.size_bytes:
        raise ValueError(
            f"artifact payload {record.relative_path!r} size does not match its manifest"
        )
    digest = _sha256_file(path)
    if digest != record.sha256:
        raise ValueError(
            f"artifact payload {record.relative_path!r} hash does not match its manifest"
        )
    return path


def _contained_file(root: Path, relative_path: str) -> Path:
    normalized = _relative_path(relative_path)
    unresolved = root / normalized
    if unresolved.is_symlink():
        raise ValueError(f"artifact payload {relative_path!r} must not be a symbolic link")
    try:
        resolved = unresolved.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"artifact payload {relative_path!r} is missing") from error
    if not resolved.is_relative_to(root):
        raise ValueError(f"artifact payload {relative_path!r} escapes the artifact root")
    if not resolved.is_file():
        raise ValueError(f"artifact payload {relative_path!r} must be a regular file")
    return resolved


def _file_from_records(root: Path, records: tuple[FileRecord, ...], relative_path: str) -> Path:
    normalized = _relative_path(relative_path)
    record = next((item for item in records if item.relative_path == normalized), None)
    if record is None:
        raise ValueError(f"artifact payload {normalized!r} is not available in this view")
    return _contained_file(root, record.relative_path)


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    root: Path
    manifest: ArtifactManifest

    def __post_init__(self) -> None:
        root = self.root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("artifact root must be a directory")
        if self.manifest.status is not ArtifactStatus.COMPLETE:
            raise ValueError("LoadedArtifact requires a complete manifest")
        object.__setattr__(self, "root", root)

    def file(self, relative_path: str) -> Path:
        return _file_from_records(self.root, self.manifest.files, relative_path)

    def document(self, relative_path: str) -> dict[str, JSONValue]:
        path = self.file(relative_path)
        if path.suffix != ".json":
            raise ValueError(f"artifact payload {relative_path!r} is not a JSON document")
        return read_json_document(path)


@dataclass(frozen=True, slots=True)
class PendingArtifactView:
    root: Path
    manifest: ArtifactManifest
    files: tuple[FileRecord, ...]
    _owner: weakref.ReferenceType[ArtifactWriter] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        owner = None if self._owner is None else self._owner()
        if owner is None:
            raise ValueError("PendingArtifactView can be created only by its live ArtifactWriter")
        root = self.root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("artifact root must be a directory")
        if self.manifest.status is not ArtifactStatus.INCOMPLETE:
            raise ValueError("PendingArtifactView requires an incomplete manifest")
        if not all(isinstance(record, FileRecord) for record in self.files):
            raise ValueError("files must contain FileRecord values")
        if owner.root != root or owner._bootstrap is not self.manifest or owner._completed:
            raise ValueError("PendingArtifactView must belong to its live ArtifactWriter")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "files", tuple(self.files))

    def _ensure_live(self) -> None:
        owner = None if self._owner is None else self._owner()
        if (
            owner is None
            or owner._completed
            or owner.root != self.root
            or owner._bootstrap is not self.manifest
        ):
            raise RuntimeError("pending artifact view is no longer live")

    def file(self, relative_path: str) -> Path:
        self._ensure_live()
        return _file_from_records(self.root, self.files, relative_path)

    def document(self, relative_path: str) -> dict[str, JSONValue]:
        path = self.file(relative_path)
        if path.suffix != ".json":
            raise ValueError(f"artifact payload {relative_path!r} is not a JSON document")
        return read_json_document(path)


def _temporary_path(final: Path, *, preserve_suffix: bool) -> Path:
    token = uuid.uuid4().hex
    if preserve_suffix:
        return final.with_name(f".{final.stem}.{token}.tmp{final.suffix}")
    return final.with_name(f".{final.name}.{token}.tmp")


_TEMPORARY_TOKEN = r"[0-9a-f]{32}"
_ARTIFACT_TEMPORARY_PATTERN = re.compile(
    rf"(?:"
    rf"\.manifest\.json\.{_TEMPORARY_TOKEN}\.tmp"
    rf"|\.(?:training-config|training-summary|evaluation-smoke|evaluation-iid|evaluation-ood)"
    rf"\.{_TEMPORARY_TOKEN}\.tmp\.json"
    rf"|\.model\.{_TEMPORARY_TOKEN}\.tmp\.(?:pt|zip)"
    rf")"
)


def _is_artifact_temporary(path: Path) -> bool:
    return (
        not path.is_symlink()
        and path.is_file()
        and _ARTIFACT_TEMPORARY_PATTERN.fullmatch(path.name) is not None
    )


def _write_fsynced_file(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace_bytes(path: Path, content: bytes) -> None:
    temporary = _temporary_path(path, preserve_suffix=False)
    try:
        _write_fsynced_file(temporary, content)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _atomic_publish_bytes(path: Path, content: bytes) -> None:
    temporary = _temporary_path(path, preserve_suffix=True)
    try:
        _write_fsynced_file(temporary, content)
        os.link(temporary, path)
        temporary.unlink()
        _fsync_directory(path.parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


@contextmanager
def _deferred_sigint():
    pthread_sigmask = getattr(signal, "pthread_sigmask", None)
    if pthread_sigmask is None:
        yield
        return
    previous = pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        yield
    finally:
        pthread_sigmask(signal.SIG_SETMASK, previous)


def _remove_pre_manifest_residue(root: Path) -> None:
    if not root.is_dir():
        return
    entries = tuple(root.iterdir())
    if any(not _is_artifact_temporary(entry) for entry in entries):
        return
    for entry in entries:
        entry.unlink(missing_ok=True)
    root.rmdir()


def _validate_config_manifest(config: TrainingConfigDocument, manifest: ArtifactManifest) -> None:
    identity_matches = (
        config.trainer is manifest.trainer
        and config.profile is manifest.profile
        and config.seed == manifest.seed
        and config.device is manifest.runtime.device
        and config.environment_id == manifest.environment_id
        and config.environment_contract_id == manifest.environment_contract_id
        and config.train_distribution_id == manifest.train_distribution_id
        and tuple((suite_id.value, digest) for suite_id, digest in config.evaluation_suites)
        == manifest.evaluation_suites
        and config.spectrum_grid_id == manifest.spectrum_grid_id
        and config.preprocessing_schema_id == manifest.preprocessing_schema_id
        and config.architecture_schema_id == manifest.architecture_schema_id
    )
    if not identity_matches:
        raise ValueError("training config identity must match the artifact manifest")


def _validate_summary_manifest(
    summary: TrainingSummaryDocument, manifest: ArtifactManifest
) -> None:
    if (
        summary.trainer is not manifest.trainer
        or summary.profile is not manifest.profile
        or summary.seed != manifest.seed
    ):
        raise ValueError("training summary identity must match the artifact manifest")
    counts = manifest.training_counts
    if manifest.status is ArtifactStatus.INCOMPLETE:
        return
    if isinstance(summary.summary, BCTrainingSummary):
        if not isinstance(counts, BCTrainingCounts):
            raise ValueError("training summary variant must match manifest trainer")
        if counts.training_examples != summary.summary.training_examples:
            raise ValueError("training_examples must match training-summary.json")
        if counts.validation_examples != summary.summary.validation_examples:
            raise ValueError("validation_examples must match training-summary.json")
    else:
        if not isinstance(counts, PPOTrainingCounts):
            raise ValueError("training summary variant must match manifest trainer")
        if counts.requested_environment_steps != summary.summary.requested_environment_steps:
            raise ValueError("requested_environment_steps must match training-summary.json")
        if counts.completed_environment_steps != summary.summary.completed_environment_steps:
            raise ValueError("completed_environment_steps must match training-summary.json")


def read_training_config(
    artifact: LoadedArtifact | PendingArtifactView,
) -> TrainingConfigDocument:
    if isinstance(artifact, (LoadedArtifact, PendingArtifactView)):
        config = TrainingConfigDocument.from_document(artifact.document("training-config.json"))
        _validate_config_manifest(config, artifact.manifest)
        return config

    from harpy.learning.pitch_artifacts import (
        LoadedPitchArtifact,
        PendingPitchArtifactView,
        read_pitch_training_config,
    )

    if isinstance(artifact, (LoadedPitchArtifact, PendingPitchArtifactView)):
        return read_pitch_training_config(artifact)  # type: ignore[return-value]
    raise ValueError("artifact must be a schema-v1 or schema-v2 artifact")


def read_training_summary(
    artifact: LoadedArtifact | PendingArtifactView,
) -> TrainingSummaryDocument:
    if isinstance(artifact, (LoadedArtifact, PendingArtifactView)):
        summary = TrainingSummaryDocument.from_document(artifact.document("training-summary.json"))
        _validate_summary_manifest(summary, artifact.manifest)
        return summary

    from harpy.learning.pitch_artifacts import (
        LoadedPitchArtifact,
        PendingPitchArtifactView,
        read_pitch_training_summary,
    )

    if isinstance(artifact, (LoadedPitchArtifact, PendingPitchArtifactView)):
        return read_pitch_training_summary(artifact)  # type: ignore[return-value]
    raise ValueError("artifact must be a schema-v1 or schema-v2 artifact")


def _validate_disk_inventory(root: Path, expected_payloads: tuple[str, ...]) -> None:
    expected = {"manifest.json", *expected_payloads}
    actual = {entry.name for entry in root.iterdir()}
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"artifact inventory mismatch; missing={missing}, unexpected={unexpected}")
    for name in expected:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact inventory entry {name!r} must be a regular file")


def _decode_all_json_payloads(
    root: Path, expected_payloads: tuple[str, ...]
) -> dict[str, dict[str, JSONValue]]:
    return {
        filename: read_json_document(_contained_file(root, filename))
        for filename in expected_payloads
        if filename.endswith(".json")
    }


def _read_regular_file_bytes(path: Path, field: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{field} is missing or unsafe") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{field} is missing or unsafe")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            return stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def artifact_manifest_from_document(document: Mapping[str, JSONValue]) -> object:
    """Dispatch a manifest only after reading its strict top-level schema ID."""
    mapping = _mapping(document, "artifact manifest")
    schema_version = _integer(mapping.get("schema_version"), "schema_version", minimum=1)
    if schema_version == ARTIFACT_SCHEMA_VERSION:
        return ArtifactManifest.from_document(document)
    if schema_version == PITCH_ARTIFACT_SCHEMA_VERSION:
        from harpy.learning.pitch_artifacts import PitchArtifactManifest

        return PitchArtifactManifest.from_document(document)
    raise ValueError(f"unsupported artifact schema_version {schema_version}")


def artifact_schema_registries() -> Mapping[int, object]:
    """Return the immutable schema-first registry without merging schema ownership."""
    from harpy.learning.pitch_artifacts import PITCH_ARTIFACT_SCHEMA_V2_REGISTRY

    return MappingProxyType(
        {
            ARTIFACT_SCHEMA_VERSION: ARTIFACT_SCHEMA_V1_REGISTRY,
            PITCH_ARTIFACT_SCHEMA_V2_REGISTRY.schema_version: (PITCH_ARTIFACT_SCHEMA_V2_REGISTRY),
        }
    )


def load_artifact(root: Path) -> object:
    """Schema-first load one complete, closed-inventory local artifact."""

    try:
        resolved = root.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("artifact root does not exist") from error
    if not resolved.is_dir():
        raise ValueError("artifact root must be a directory")
    manifest_path = resolved / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        entries = tuple(resolved.iterdir())
        manifest_missing = not manifest_path.exists() and not manifest_path.is_symlink()
        if manifest_missing and (
            not entries or all(_is_artifact_temporary(entry) for entry in entries)
        ):
            raise ValueError(
                "artifact bootstrap did not publish a manifest; directory is safe to remove"
            )
        raise ValueError("artifact manifest.json is missing or unsafe")
    manifest_document = decode_json_bytes(
        _read_regular_file_bytes(manifest_path, "artifact manifest.json")
    )
    schema_version = _integer(
        manifest_document.get("schema_version"),
        "schema_version",
        minimum=1,
    )
    if schema_version != ARTIFACT_SCHEMA_VERSION:
        if schema_version == PITCH_ARTIFACT_SCHEMA_VERSION:
            from harpy.learning.pitch_artifacts import load_pitch_artifact

            return load_pitch_artifact(resolved)
        raise ValueError(f"unsupported artifact schema_version {schema_version}")
    manifest = ArtifactManifest.from_document(manifest_document)
    if manifest.status is not ArtifactStatus.COMPLETE:
        raise ValueError("public artifact loader rejects an incomplete manifest")
    expected_payloads = required_payload_names(manifest.trainer, manifest.profile)
    _validate_disk_inventory(resolved, expected_payloads)
    if tuple(record.relative_path for record in manifest.files) != expected_payloads:
        raise ValueError("manifest file records do not match the exact artifact inventory")
    for record in manifest.files:
        _verify_file_record(resolved, record)
    documents = _decode_all_json_payloads(resolved, expected_payloads)
    config = TrainingConfigDocument.from_document(documents["training-config.json"])
    summary = TrainingSummaryDocument.from_document(documents["training-summary.json"])
    _validate_config_manifest(config, manifest)
    _validate_summary_manifest(summary, manifest)
    return LoadedArtifact(root=resolved, manifest=manifest)


@dataclass(slots=True, weakref_slot=True)
class ArtifactWriter:
    root: Path
    _bootstrap: ArtifactManifest
    _bootstrap_bytes: bytes = field(repr=False)
    _records: dict[str, FileRecord] = field(default_factory=dict, repr=False)
    _completed: bool = field(default=False, repr=False)

    @classmethod
    def begin(cls, output: Path, manifest: ArtifactManifest) -> ArtifactWriter:
        if not isinstance(manifest, ArtifactManifest):
            raise ValueError("manifest must be an ArtifactManifest")
        snapshot = ArtifactManifest.from_document(manifest.to_document())
        if snapshot.status is not ArtifactStatus.INCOMPLETE:
            raise ValueError("ArtifactWriter.begin requires an incomplete manifest")
        manifest_bytes = canonical_json_bytes(snapshot.to_document())
        if output.is_symlink():
            raise FileExistsError(output)
        parent = output.parent.resolve(strict=False)
        root = parent / output.name
        parent.mkdir(parents=True, exist_ok=True)
        created = False
        try:
            with _deferred_sigint():
                root.mkdir(parents=False, exist_ok=False)
                created = True
                _atomic_replace_bytes(root / "manifest.json", manifest_bytes)
        except BaseException:
            if created and not (root / "manifest.json").exists():
                _remove_pre_manifest_residue(root)
            raise
        return cls(root=root, _bootstrap=snapshot, _bootstrap_bytes=manifest_bytes)

    @property
    def file_records(self) -> tuple[FileRecord, ...]:
        return tuple(self._records.values())

    def _ensure_active(self) -> None:
        if self._completed:
            raise RuntimeError("artifact writer is already complete")

    def _validate_publication_name(self, filename: str, *, json_payload: bool) -> str:
        self._ensure_active()
        normalized = _relative_path(filename, "filename")
        if "/" in normalized or normalized not in required_payload_names(
            self._bootstrap.trainer, self._bootstrap.profile
        ):
            raise ValueError("filename must belong to the exact artifact inventory")
        if (normalized.endswith(".json")) != json_payload:
            kind = "JSON" if json_payload else "model"
            raise ValueError(f"filename must be the required {kind} payload")
        if normalized in self._records:
            raise FileExistsError(self.root / normalized)
        return normalized

    def publish_json(self, filename: str, document: Mapping[str, JSONValue]) -> FileRecord:
        normalized = self._validate_publication_name(filename, json_payload=True)
        content = canonical_json_bytes(document)
        path = self.root / normalized
        _atomic_publish_bytes(path, content)
        record = _record_file(path, normalized)
        self._records[normalized] = record
        return record

    def publish_model(self, filename: str, save: Callable[[Path], None]) -> FileRecord:
        normalized = self._validate_publication_name(filename, json_payload=False)
        path = self.root / normalized
        temporary = _temporary_path(path, preserve_suffix=True)
        try:
            save(temporary)
            if temporary.is_symlink() or not temporary.is_file():
                raise ValueError("model saver must create the exact requested temporary file")
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.link(temporary, path)
            temporary.unlink()
            _fsync_directory(self.root)
        finally:
            with suppress(FileNotFoundError):
                temporary.unlink()
        record = _record_file(path, normalized)
        self._records[normalized] = record
        return record

    def pending_view(self, required_names: Sequence[str]) -> PendingArtifactView:
        self._ensure_active()
        if isinstance(required_names, (str, bytes)):
            raise ValueError("required_names must be a sequence of payload names")
        names = tuple(_relative_path(name, "required name") for name in required_names)
        if len(set(names)) != len(names):
            raise ValueError("required_names must not contain duplicates")
        records: list[FileRecord] = []
        for name in names:
            record = self._records.get(name)
            if record is None:
                raise ValueError(f"required payload {name!r} has not been published")
            _verify_file_record(self.root, record)
            records.append(record)
        return PendingArtifactView(
            root=self.root,
            manifest=self._bootstrap,
            files=tuple(records),
            _owner=weakref.ref(self),
        )

    def _verify_bootstrap(self) -> None:
        persisted = (self.root / "manifest.json").read_bytes()
        if persisted != self._bootstrap_bytes:
            raise ValueError("persisted incomplete manifest no longer matches writer bootstrap")

    def _verified_records(self) -> tuple[FileRecord, ...]:
        expected = required_payload_names(self._bootstrap.trainer, self._bootstrap.profile)
        if set(self._records) != set(expected):
            raise ValueError("writer records do not contain the exact required inventory")
        _validate_disk_inventory(self.root, expected)
        records: list[FileRecord] = []
        for name in expected:
            recorded = self._records[name]
            _verify_file_record(self.root, recorded)
            records.append(recorded)
        return tuple(records)

    def complete(self, completion: ArtifactCompletion) -> LoadedArtifact:
        self._ensure_active()
        if not isinstance(completion, ArtifactCompletion):
            raise ValueError("completion must be an ArtifactCompletion")
        self._verify_bootstrap()
        records = self._verified_records()
        pending = self.pending_view(tuple(record.relative_path for record in records))
        for record in records:
            if record.relative_path.endswith(".json"):
                pending.document(record.relative_path)
        config = read_training_config(pending)
        summary = read_training_summary(pending)
        _validate_completion_counts(self._bootstrap, summary, completion)
        if self._bootstrap.trainer is TrainerKind.PPO and completion.bc_criterion_met is not None:
            raise ValueError("PPO completion must not contain a BC criterion result")
        eligible = _completion_is_eligible(self._bootstrap, completion.evaluation_device)
        if eligible and self._bootstrap.trainer is TrainerKind.BC:
            if completion.bc_criterion_met is None:
                raise ValueError("eligible BC completion requires bc_criterion_met")
            criterion_status = (
                CriterionStatus.CRITERION_MET
                if completion.bc_criterion_met
                else CriterionStatus.CRITERION_NOT_MET
            )
            criterion_met = completion.bc_criterion_met
        elif eligible:
            criterion_status = CriterionStatus.ELIGIBLE_FOR_AGGREGATE
            criterion_met = None
        else:
            criterion_status = CriterionStatus.INELIGIBLE
            criterion_met = None
        final_manifest = replace(
            self._bootstrap,
            status=ArtifactStatus.COMPLETE,
            completed_at_utc=completion.completed_at_utc,
            training_counts=completion.training_counts,
            evaluation_device=completion.evaluation_device,
            criterion_eligible=eligible,
            criterion_status=criterion_status,
            criterion_met=criterion_met,
            files=records,
        )
        _validate_config_manifest(config, final_manifest)
        _validate_summary_manifest(summary, final_manifest)
        manifest_path = self.root / "manifest.json"
        temporary = _temporary_path(manifest_path, preserve_suffix=False)
        try:
            _write_fsynced_file(
                temporary,
                canonical_json_bytes(final_manifest.to_document()),
            )
            loaded = LoadedArtifact(root=self.root, manifest=final_manifest)
        except BaseException:
            with suppress(FileNotFoundError):
                temporary.unlink()
            raise
        self._completed = True
        try:
            os.replace(temporary, manifest_path)
        except BaseException:
            try:
                temporary.lstat()
            except FileNotFoundError:
                # An atomic replace consumed its staged source, so publication
                # committed even if an interrupt arrived before the call returned.
                return loaded
            self._completed = False
            with suppress(FileNotFoundError):
                temporary.unlink()
            raise
        return loaded


def _completion_is_eligible(bootstrap: ArtifactManifest, evaluation_device: DeviceName) -> bool:
    declared_seed = (
        bootstrap.seed == 0 if bootstrap.trainer is TrainerKind.BC else bootstrap.seed in range(5)
    )
    return (
        bootstrap.profile is ProfileName.CHECKPOINT
        and declared_seed
        and isinstance(bootstrap.source, SourceStatus)
        and not bootstrap.source.dirty_tree
        and bootstrap.source.required_inputs_committed
        and bootstrap.runtime.device is DeviceName.CPU
        and evaluation_device is DeviceName.CPU
    )


def _validate_completion_counts(
    bootstrap: ArtifactManifest,
    summary: TrainingSummaryDocument,
    completion: ArtifactCompletion,
) -> None:
    if bootstrap.trainer is TrainerKind.BC:
        if not isinstance(completion.training_counts, BCTrainingCounts) or not isinstance(
            summary.summary, BCTrainingSummary
        ):
            raise ValueError("BC completion requires BCTrainingCounts and BCTrainingSummary")
        initial = bootstrap.training_counts
        if not isinstance(initial, BCTrainingCounts):
            raise ValueError("BC bootstrap requires BCTrainingCounts")
        if (
            completion.training_counts.configured_training_episodes
            != initial.configured_training_episodes
            or completion.training_counts.configured_validation_episodes
            != initial.configured_validation_episodes
        ):
            raise ValueError("completion must preserve configured BC episode counts")
        if completion.training_counts.training_examples != summary.summary.training_examples:
            raise ValueError("training_examples must match training-summary.json")
        if completion.training_counts.validation_examples != summary.summary.validation_examples:
            raise ValueError("validation_examples must match training-summary.json")
    else:
        if not isinstance(completion.training_counts, PPOTrainingCounts) or not isinstance(
            summary.summary, PPOTrainingSummary
        ):
            raise ValueError("PPO completion requires PPOTrainingCounts and PPOTrainingSummary")
        initial = bootstrap.training_counts
        if not isinstance(initial, PPOTrainingCounts):
            raise ValueError("PPO bootstrap requires PPOTrainingCounts")
        if (
            completion.training_counts.requested_environment_steps
            != initial.requested_environment_steps
        ):
            raise ValueError("completion must preserve requested_environment_steps")
        if (
            completion.training_counts.requested_environment_steps
            != summary.summary.requested_environment_steps
        ):
            raise ValueError("requested_environment_steps must match training-summary.json")
        if (
            completion.training_counts.completed_environment_steps
            != summary.summary.completed_environment_steps
        ):
            raise ValueError("completed_environment_steps must match training-summary.json")


def write_new_bytes(path: Path, content: bytes) -> None:
    """Atomically publish a new sibling file without a check-then-overwrite race."""

    if not isinstance(content, bytes):
        raise ValueError("content must be bytes")
    parent = path.parent.resolve(strict=True)
    final = parent / path.name
    temporary = _temporary_path(final, preserve_suffix=True)
    linked = False
    try:
        _write_fsynced_file(temporary, content)
        os.link(temporary, final)
        linked = True
        temporary.unlink()
        _fsync_directory(parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    if not linked:
        raise RuntimeError("create-only publication did not link the final path")


def _git(anchor: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(anchor), *arguments],
        check=check,
        capture_output=True,
    )


def capture_source_status(start: Path) -> SourceProvenance:
    """Capture actual checkout or package provenance, independently of the cwd."""
    return _capture_source_status(start, required_inputs=_REQUIRED_SOURCE_INPUTS)


def _capture_source_status(start: Path, *, required_inputs: tuple[str, ...]) -> SourceProvenance:
    resolved = start.resolve(strict=True)
    anchor = resolved if resolved.is_dir() else resolved.parent
    try:
        discovery = _git(anchor, "rev-parse", "--show-toplevel", check=False)
    except FileNotFoundError:
        discovery = None
    root = (
        Path(discovery.stdout.decode("utf-8").strip()).resolve(strict=True)
        if discovery is not None and discovery.returncode == 0
        else None
    )
    # A wheel installed in a checkout's .venv must not inherit that checkout's
    # identity.  Only the actual src/harpy tree (or explicit root anchor) owns it.
    if root is None or not (anchor == root or resolved.is_relative_to(root / "src" / "harpy")):
        return _capture_package_source(resolved)
    commit = _git(root, "rev-parse", "HEAD").stdout.decode("ascii").strip()
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=normal").stdout
    tracked_diff = _git(root, "diff", "--binary", "--no-ext-diff", "HEAD", "--").stdout
    lock_content = (root / "uv.lock").read_bytes()
    required_inputs_committed = all(
        _git(root, "cat-file", "-e", f"HEAD:{relative_path}", check=False).returncode == 0
        for relative_path in required_inputs
    )
    return SourceStatus(
        commit=commit,
        dirty_tree=bool(status),
        tracked_diff_sha256=hashlib.sha256(tracked_diff).hexdigest(),
        dependency_lock_sha256=hashlib.sha256(lock_content).hexdigest(),
        required_inputs_committed=required_inputs_committed,
    )


def _capture_package_source(start: Path) -> PackageSourceStatus:
    candidates = (start, *start.parents)
    package = next(
        (path for path in candidates if path.name == "harpy" and (path / "__init__.py").is_file()),
        None,
    )
    if package is None:
        raise ValueError("source anchor must belong to a Harpy checkout or Python package")
    inventory = []
    for path in sorted(package.rglob("*")):
        relative = path.relative_to(package)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ValueError("package source must not contain symbolic links")
        if path.is_file():
            inventory.append((relative.as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()))
    version = None
    try:
        distribution = importlib.metadata.distribution("harpy-audio")
        installed_init = Path(distribution.locate_file("harpy/__init__.py")).resolve()
        if installed_init == package / "__init__.py":
            version = distribution.version
    except importlib.metadata.PackageNotFoundError:
        pass
    return PackageSourceStatus(
        distribution_name="harpy-audio",
        distribution_version=version,
        package_sha256=hashlib.sha256(
            json.dumps(inventory, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    )


__all__ = [
    "ARTIFACT_SCHEMA_V1_REGISTRIES",
    "ARTIFACT_SCHEMA_V1_REGISTRY",
    "ARTIFACT_SCHEMA_VERSION",
    "PITCH_ARTIFACT_SCHEMA_VERSION",
    "ArtifactCompletion",
    "ArtifactManifest",
    "ArtifactSchemaRegistry",
    "ArtifactStatus",
    "ArtifactWriter",
    "BCTrainingCounts",
    "CriterionStatus",
    "FileRecord",
    "LoadedArtifact",
    "PPOTrainingCounts",
    "PackageSourceStatus",
    "PendingArtifactView",
    "RuntimeStatus",
    "SourceProvenance",
    "SourceStatus",
    "TrainingConfigDocument",
    "TrainingSummaryDocument",
    "artifact_manifest_from_document",
    "artifact_schema_registries",
    "canonical_json_bytes",
    "capture_source_status",
    "decode_json_bytes",
    "load_artifact",
    "read_json_document",
    "read_training_config",
    "read_training_summary",
    "required_payload_names",
    "write_new_bytes",
]
