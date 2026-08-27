"""Strict schema-v2 artifacts for the Milestone E pitch estimator.

Schema v2 is intentionally separate from the frozen Milestone D BC/PPO codecs.  This
module owns the pitch-only registry, compatibility preimage, typed documents, exact
inventory, lifecycle, strict model reload, and exact-three aggregate preflight.
"""

from __future__ import annotations

import hashlib
import operator
import os
import re
import subprocess
import weakref
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final

from harpy.envs.models import ObservationMode, PitchAction
from harpy.learning.artifacts import (
    PITCH_ARTIFACT_SCHEMA_VERSION,
    ArtifactStatus,
    CriterionStatus,
    FileRecord,
    RuntimeStatus,
    SourceStatus,
    _atomic_publish_bytes,
    _atomic_replace_bytes,
    _boolean,
    _contained_file,
    _deferred_sigint,
    _digest,
    _document_float,
    _exact_fields,
    _file_from_document,
    _file_to_document,
    _fsync_directory,
    _mapping,
    _read_regular_file_bytes,
    _record_file,
    _relative_path,
    _runtime_from_document,
    _runtime_to_document,
    _source_from_document,
    _source_to_document,
    _string,
    _temporary_path,
    _timestamp,
    _verify_file_record,
    _write_fsynced_file,
    canonical_json_bytes,
    decode_json_bytes,
    read_json_document,
)
from harpy.learning.errors import PitchArtifactSetError
from harpy.learning.models import (
    ENVIRONMENT_CONTRACT_ID,
    ENVIRONMENT_ID,
    SPECTRUM_GRID_ID,
    DeviceName,
    JSONValue,
    ProfileName,
)
from harpy.learning.pitch import (
    PITCH_PROFILE_CONFIGS,
    PITCH_SHUFFLE_GENERATOR_DEVICE,
    PITCH_TRAINING_NUM_WORKERS,
    PitchDatasetMetrics,
    PitchEpochMetrics,
    PitchTrainingProfile,
    PitchTrainingSummary,
    load_pitch_estimator_model,
)
from harpy.learning.pitch_data import (
    PITCH_DISTRIBUTION_ID,
    PITCH_SPLIT_DIGEST_SHA256,
    PitchEvaluationSuiteId,
    PitchTrainerKind,
)
from harpy.learning.pitch_network import (
    PITCH_ESTIMATOR_ARCHITECTURE_ID,
    PITCH_ESTIMATOR_PARAMETER_COUNT,
)

PITCH_PREPROCESSING_SCHEMA_ID: Final = "harpy-sine-pitch-spectrum-float32-v1"
PITCH_POLICY_SEMANTICS_ID: Final = "harpy-sine-pitch-estimator-planner-v1"
PITCH_ACTION_SCHEMA_ID: Final = "harpy-sine-pitch-actions-v1"
PITCH_REQUIRED_DEVICE: Final = "cpu"
PITCH_REQUIRED_PAYLOAD_NAMES: Final = (
    "training-config.json",
    "training-summary.json",
    "model.pt",
    "evaluation-smoke.json",
    "evaluation-smoke-probes.json",
)
PITCH_EVALUATION_SUITE_RECORDS: Final = (
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

_PITCH_REQUIRED_SOURCE_INPUTS = (
    "src/harpy/envs/baselines.py",
    "src/harpy/envs/models.py",
    "src/harpy/envs/planning.py",
    "src/harpy/envs/sine_pitch.py",
    "src/harpy/envs/spectrum.py",
    "src/harpy/learning/action_masks.py",
    "src/harpy/learning/actors.py",
    "src/harpy/learning/artifacts.py",
    "src/harpy/learning/cache.py",
    "src/harpy/learning/envs.py",
    "src/harpy/learning/evaluation.py",
    "src/harpy/learning/models.py",
    "src/harpy/learning/observations.py",
    "src/harpy/learning/pitch_data.py",
    "src/harpy/learning/pitch_network.py",
    "src/harpy/learning/pitch_actor.py",
    "src/harpy/learning/pitch.py",
    "src/harpy/learning/pitch_artifacts.py",
    "src/harpy/learning/pitch_evaluation.py",
    "src/harpy/learning/suites.py",
    "uv.lock",
)
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_TEMPORARY_TOKEN = r"[0-9a-f]{32}"
_PITCH_TEMPORARY_PATTERN = re.compile(
    rf"(?:\.manifest\.json\.{_TEMPORARY_TOKEN}\.tmp"
    rf"|\.(?:training-config|training-summary|evaluation-smoke|evaluation-smoke-probes)"
    rf"\.{_TEMPORARY_TOKEN}\.tmp\.json"
    rf"|\.model\.{_TEMPORARY_TOKEN}\.tmp\.pt)"
)


@dataclass(frozen=True, slots=True)
class PitchArtifactSchemaRegistry:
    """Immutable closed identities owned only by schema v2."""

    schema_version: int
    trainers: tuple[str, ...]
    profiles: tuple[str, ...]
    suites: tuple[tuple[str, str], ...]
    architecture_ids: tuple[str, ...]
    preprocessing_ids: tuple[str, ...]
    policy_semantics_ids: tuple[str, ...]
    required_payloads: tuple[str, ...]


PITCH_ARTIFACT_SCHEMA_V2_REGISTRY = PitchArtifactSchemaRegistry(
    schema_version=PITCH_ARTIFACT_SCHEMA_VERSION,
    trainers=(PitchTrainerKind.PITCH.value,),
    profiles=tuple(profile.value for profile in ProfileName),
    suites=PITCH_EVALUATION_SUITE_RECORDS,
    architecture_ids=(PITCH_ESTIMATOR_ARCHITECTURE_ID,),
    preprocessing_ids=(PITCH_PREPROCESSING_SCHEMA_ID,),
    policy_semantics_ids=(PITCH_POLICY_SEMANTICS_ID,),
    required_payloads=PITCH_REQUIRED_PAYLOAD_NAMES,
)

# Mapping form is useful to schema-first callers while retaining a single immutable
# v2 registry rather than expanding any v1 global.
PITCH_ARTIFACT_SCHEMA_REGISTRIES = MappingProxyType(
    {PITCH_ARTIFACT_SCHEMA_VERSION: PITCH_ARTIFACT_SCHEMA_V2_REGISTRY}
)


def _integer(
    value: object,
    field_name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    try:
        normalized = int(operator.index(value))
    except TypeError as error:
        raise ValueError(f"{field_name} must be an integer") from error
    if normalized < minimum or (maximum is not None and normalized > maximum):
        if maximum is None:
            raise ValueError(f"{field_name} must be at least {minimum}")
        raise ValueError(f"{field_name} must be within {minimum}..{maximum}")
    return normalized


def _schema_version(value: object) -> int:
    """Return the exact integer schema-v2 identity without numeric coercion."""
    return _integer(
        value,
        "schema_version",
        minimum=PITCH_ARTIFACT_SCHEMA_VERSION,
        maximum=PITCH_ARTIFACT_SCHEMA_VERSION,
    )


def _pitch_profile_to_document(profile: PitchTrainingProfile) -> dict[str, JSONValue]:
    if not isinstance(profile, PitchTrainingProfile):
        raise ValueError("profile_config must be a PitchTrainingProfile")
    return {
        "training_coordinate_count": profile.training_coordinate_count,
        "validation_coordinate_count": profile.validation_coordinate_count,
        "max_epochs": profile.max_epochs,
        "early_stopping_patience": profile.early_stopping_patience,
        "batch_size": profile.batch_size,
        "optimizer": profile.optimizer,
        "learning_rate": profile.learning_rate,
        "weight_decay": profile.weight_decay,
        "eligible_for_aggregate": profile.eligible_for_aggregate,
    }


def _pitch_profile_from_document(value: object) -> PitchTrainingProfile:
    mapping = _mapping(value, "profile_config")
    _exact_fields(
        mapping,
        {
            "training_coordinate_count",
            "validation_coordinate_count",
            "max_epochs",
            "early_stopping_patience",
            "batch_size",
            "optimizer",
            "learning_rate",
            "weight_decay",
            "eligible_for_aggregate",
        },
        "profile_config",
    )
    return PitchTrainingProfile(
        training_coordinate_count=_integer(
            mapping["training_coordinate_count"], "training_coordinate_count", minimum=1
        ),
        validation_coordinate_count=_integer(
            mapping["validation_coordinate_count"], "validation_coordinate_count", minimum=1
        ),
        max_epochs=_integer(mapping["max_epochs"], "max_epochs", minimum=1),
        early_stopping_patience=(
            None
            if mapping["early_stopping_patience"] is None
            else _integer(mapping["early_stopping_patience"], "early_stopping_patience", minimum=1)
        ),
        batch_size=_integer(mapping["batch_size"], "batch_size", minimum=1),
        optimizer=_string(mapping["optimizer"], "optimizer"),
        learning_rate=_document_float(mapping["learning_rate"], "learning_rate", minimum=0.0),
        weight_decay=_document_float(mapping["weight_decay"], "weight_decay", minimum=0.0),
        eligible_for_aggregate=_boolean(
            mapping["eligible_for_aggregate"], "eligible_for_aggregate"
        ),
    )


def _compatibility_document(profile: ProfileName) -> dict[str, JSONValue]:
    if not isinstance(profile, ProfileName):
        raise ValueError("profile must be a ProfileName")
    configured = PITCH_PROFILE_CONFIGS[profile]
    return {
        "schema_version": PITCH_ARTIFACT_SCHEMA_VERSION,
        "trainer": PitchTrainerKind.PITCH.value,
        "profile": profile.value,
        "profile_config": _pitch_profile_to_document(configured),
        "environment_id": ENVIRONMENT_ID,
        "environment_contract_id": ENVIRONMENT_CONTRACT_ID,
        "spectrum_grid_id": SPECTRUM_GRID_ID,
        "action_schema_id": PITCH_ACTION_SCHEMA_ID,
        "action_order": [{"index": int(action), "name": action.name} for action in PitchAction],
        "architecture_schema_id": PITCH_ESTIMATOR_ARCHITECTURE_ID,
        "preprocessing_schema_id": PITCH_PREPROCESSING_SCHEMA_ID,
        "policy_semantics_id": PITCH_POLICY_SEMANTICS_ID,
        "train_distribution_id": PITCH_DISTRIBUTION_ID,
        "coordinate_split_digest_sha256": PITCH_SPLIT_DIGEST_SHA256,
        "training_coordinate_count": configured.training_coordinate_count,
        "validation_coordinate_count": configured.validation_coordinate_count,
        "evaluation_suites": [
            {"suite_id": suite_id, "digest_sha256": digest}
            for suite_id, digest in PITCH_EVALUATION_SUITE_RECORDS
        ],
        "required_training_device": PITCH_REQUIRED_DEVICE,
        "required_evaluation_device": PITCH_REQUIRED_DEVICE,
        "data_loader_num_workers": PITCH_TRAINING_NUM_WORKERS,
        "shuffle_generator_device": PITCH_SHUFFLE_GENERATOR_DEVICE,
    }


def pitch_compatibility_sha256(profile: ProfileName) -> str:
    """Return the seed-neutral canonical compatibility digest for one profile."""
    return hashlib.sha256(canonical_json_bytes(_compatibility_document(profile))).hexdigest()


def _normalize_suite_records(value: object) -> tuple[tuple[str, str], ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("evaluation_suites must contain suite ID/digest pairs")
    try:
        pairs = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("evaluation_suites must contain suite ID/digest pairs") from error
    normalized: list[tuple[str, str]] = []
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("evaluation_suites must contain suite ID/digest pairs")
        suite_id, digest = pair
        if not isinstance(suite_id, str):
            raise ValueError("evaluation suite ID must be a string")
        try:
            normalized_id = PitchEvaluationSuiteId(suite_id).value
        except ValueError as error:
            raise ValueError("evaluation suite ID must be a schema-v2 suite") from error
        normalized.append((normalized_id, _digest(digest, "evaluation suite digest")))
    if tuple(normalized) != PITCH_EVALUATION_SUITE_RECORDS:
        raise ValueError("evaluation_suites must match the complete pinned schema-v2 inventory")
    return tuple(normalized)


def _suite_records_from_document(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise ValueError("evaluation_suites must be a list")
    records: list[tuple[str, str]] = []
    for item in value:
        mapping = _mapping(item, "evaluation suite")
        _exact_fields(mapping, {"suite_id", "digest_sha256"}, "evaluation suite")
        records.append(
            (
                _string(mapping["suite_id"], "suite_id"),
                _digest(mapping["digest_sha256"], "digest_sha256"),
            )
        )
    return _normalize_suite_records(tuple(records))


@dataclass(frozen=True, slots=True)
class PitchTrainingCounts:
    configured_training_coordinates: int
    configured_validation_coordinates: int
    training_examples: int | None
    validation_examples: int | None

    def __post_init__(self) -> None:
        training = _integer(
            self.configured_training_coordinates,
            "configured_training_coordinates",
            minimum=1,
            maximum=1_400,
        )
        validation = _integer(
            self.configured_validation_coordinates,
            "configured_validation_coordinates",
            minimum=1,
            maximum=200,
        )
        if (self.training_examples is None) != (self.validation_examples is None):
            raise ValueError("training_examples and validation_examples must both be set or null")
        if self.training_examples is not None and self.validation_examples is not None:
            training_examples = _integer(self.training_examples, "training_examples", minimum=1)
            validation_examples = _integer(
                self.validation_examples, "validation_examples", minimum=1
            )
            if training_examples != training or validation_examples != validation:
                raise ValueError("pitch examples must exactly match configured coordinates")
            object.__setattr__(self, "training_examples", training_examples)
            object.__setattr__(self, "validation_examples", validation_examples)
        object.__setattr__(self, "configured_training_coordinates", training)
        object.__setattr__(self, "configured_validation_coordinates", validation)


def _counts_to_document(counts: PitchTrainingCounts) -> dict[str, JSONValue]:
    return {
        "configured_training_coordinates": counts.configured_training_coordinates,
        "configured_validation_coordinates": counts.configured_validation_coordinates,
        "training_examples": counts.training_examples,
        "validation_examples": counts.validation_examples,
    }


def _counts_from_document(value: object) -> PitchTrainingCounts:
    mapping = _mapping(value, "training_counts")
    _exact_fields(
        mapping,
        {
            "configured_training_coordinates",
            "configured_validation_coordinates",
            "training_examples",
            "validation_examples",
        },
        "training_counts",
    )
    return PitchTrainingCounts(
        configured_training_coordinates=_integer(
            mapping["configured_training_coordinates"],
            "configured_training_coordinates",
            minimum=1,
        ),
        configured_validation_coordinates=_integer(
            mapping["configured_validation_coordinates"],
            "configured_validation_coordinates",
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


@dataclass(frozen=True, slots=True)
class PitchTrainingConfigDocument:
    schema_version: int
    trainer: PitchTrainerKind
    profile: ProfileName
    seed: int
    device: DeviceName
    environment_id: str
    environment_contract_id: str
    train_distribution_id: str
    coordinate_split_digest_sha256: str
    evaluation_suites: tuple[tuple[str, str], ...]
    spectrum_grid_id: str
    preprocessing_schema_id: str
    architecture_schema_id: str
    policy_semantics_id: str
    parameter_count: int
    profile_config: PitchTrainingProfile
    compatibility_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        if self.trainer is not PitchTrainerKind.PITCH:
            raise ValueError("trainer must be the schema-v2 pitch trainer")
        if not isinstance(self.profile, ProfileName):
            raise ValueError("profile must be a ProfileName")
        if not isinstance(self.device, DeviceName):
            raise ValueError("device must be a DeviceName")
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        identities = (
            (self.environment_id, ENVIRONMENT_ID, "environment_id"),
            (self.environment_contract_id, ENVIRONMENT_CONTRACT_ID, "environment_contract_id"),
            (self.train_distribution_id, PITCH_DISTRIBUTION_ID, "train_distribution_id"),
            (
                self.coordinate_split_digest_sha256,
                PITCH_SPLIT_DIGEST_SHA256,
                "coordinate_split_digest_sha256",
            ),
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
        )
        for actual, expected, field_name in identities:
            if actual != expected:
                raise ValueError(f"{field_name} must be {expected!r}")
        object.__setattr__(
            self,
            "evaluation_suites",
            _normalize_suite_records(self.evaluation_suites),
        )
        if self.parameter_count != PITCH_ESTIMATOR_PARAMETER_COUNT:
            raise ValueError(f"parameter_count must be {PITCH_ESTIMATOR_PARAMETER_COUNT}")
        expected_profile = PITCH_PROFILE_CONFIGS[self.profile]
        if self.profile_config != expected_profile:
            raise ValueError("profile_config must match the checked-in pitch profile")
        expected_compatibility = pitch_compatibility_sha256(self.profile)
        if self.compatibility_sha256 != expected_compatibility:
            raise ValueError("compatibility_sha256 must match the canonical preimage")

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
            "coordinate_split_digest_sha256": self.coordinate_split_digest_sha256,
            "evaluation_suites": [
                {"suite_id": suite_id, "digest_sha256": digest}
                for suite_id, digest in self.evaluation_suites
            ],
            "spectrum_grid_id": self.spectrum_grid_id,
            "preprocessing_schema_id": self.preprocessing_schema_id,
            "architecture_schema_id": self.architecture_schema_id,
            "policy_semantics_id": self.policy_semantics_id,
            "parameter_count": self.parameter_count,
            "profile_config": _pitch_profile_to_document(self.profile_config),
            "compatibility_sha256": self.compatibility_sha256,
        }

    @classmethod
    def from_document(cls, document: Mapping[str, JSONValue]) -> PitchTrainingConfigDocument:
        mapping = _mapping(document, "pitch training config")
        schema_version = _schema_version(mapping.get("schema_version"))
        fields = {
            "schema_version",
            "trainer",
            "profile",
            "seed",
            "device",
            "environment_id",
            "environment_contract_id",
            "train_distribution_id",
            "coordinate_split_digest_sha256",
            "evaluation_suites",
            "spectrum_grid_id",
            "preprocessing_schema_id",
            "architecture_schema_id",
            "policy_semantics_id",
            "parameter_count",
            "profile_config",
            "compatibility_sha256",
        }
        _exact_fields(mapping, fields, "pitch training config")
        try:
            trainer = PitchTrainerKind(_string(mapping["trainer"], "trainer"))
            profile = ProfileName(_string(mapping["profile"], "profile"))
            device = DeviceName(_string(mapping["device"], "device"))
        except ValueError as error:
            raise ValueError("pitch config enum identity is invalid") from error
        return cls(
            schema_version=schema_version,
            trainer=trainer,
            profile=profile,
            seed=_integer(mapping["seed"], "seed"),
            device=device,
            environment_id=_string(mapping["environment_id"], "environment_id"),
            environment_contract_id=_string(
                mapping["environment_contract_id"], "environment_contract_id"
            ),
            train_distribution_id=_string(
                mapping["train_distribution_id"], "train_distribution_id"
            ),
            coordinate_split_digest_sha256=_digest(
                mapping["coordinate_split_digest_sha256"],
                "coordinate_split_digest_sha256",
            ),
            evaluation_suites=_suite_records_from_document(mapping["evaluation_suites"]),
            spectrum_grid_id=_string(mapping["spectrum_grid_id"], "spectrum_grid_id"),
            preprocessing_schema_id=_string(
                mapping["preprocessing_schema_id"], "preprocessing_schema_id"
            ),
            architecture_schema_id=_string(
                mapping["architecture_schema_id"], "architecture_schema_id"
            ),
            policy_semantics_id=_string(mapping["policy_semantics_id"], "policy_semantics_id"),
            parameter_count=_integer(mapping["parameter_count"], "parameter_count", minimum=1),
            profile_config=_pitch_profile_from_document(mapping["profile_config"]),
            compatibility_sha256=_digest(mapping["compatibility_sha256"], "compatibility_sha256"),
        )


def _metrics_to_document(metrics: PitchDatasetMetrics) -> dict[str, JSONValue]:
    return {
        "example_count": metrics.example_count,
        "loss": metrics.loss,
        "mean_absolute_error_cents": metrics.mean_absolute_error_cents,
        "median_absolute_error_cents": metrics.median_absolute_error_cents,
        "within_one_count": metrics.within_one_count,
        "within_one_rate": metrics.within_one_rate,
        "within_five_count": metrics.within_five_count,
        "within_five_rate": metrics.within_five_rate,
    }


def _metrics_from_document(value: object) -> PitchDatasetMetrics:
    mapping = _mapping(value, "pitch dataset metrics")
    fields = {
        "example_count",
        "loss",
        "mean_absolute_error_cents",
        "median_absolute_error_cents",
        "within_one_count",
        "within_one_rate",
        "within_five_count",
        "within_five_rate",
    }
    _exact_fields(mapping, fields, "pitch dataset metrics")
    return PitchDatasetMetrics(
        example_count=_integer(mapping["example_count"], "example_count", minimum=1),
        loss=_document_float(mapping["loss"], "loss", minimum=0.0),
        mean_absolute_error_cents=_document_float(
            mapping["mean_absolute_error_cents"],
            "mean_absolute_error_cents",
            minimum=0.0,
        ),
        median_absolute_error_cents=_document_float(
            mapping["median_absolute_error_cents"],
            "median_absolute_error_cents",
            minimum=0.0,
        ),
        within_one_count=_integer(mapping["within_one_count"], "within_one_count"),
        within_one_rate=_document_float(
            mapping["within_one_rate"], "within_one_rate", minimum=0.0, maximum=1.0
        ),
        within_five_count=_integer(mapping["within_five_count"], "within_five_count"),
        within_five_rate=_document_float(
            mapping["within_five_rate"], "within_five_rate", minimum=0.0, maximum=1.0
        ),
    )


def _pitch_summary_to_document(summary: PitchTrainingSummary) -> dict[str, JSONValue]:
    return {
        "history": [
            {
                "epoch": metrics.epoch,
                "training": _metrics_to_document(metrics.training),
                "validation": _metrics_to_document(metrics.validation),
            }
            for metrics in summary.history
        ],
        "selected_epoch": summary.selected_epoch,
        "training_examples": summary.training_examples,
        "validation_examples": summary.validation_examples,
        "seed": summary.seed,
        "device": summary.device,
        "deterministic_algorithms": summary.deterministic_algorithms,
        "cudnn_benchmark": summary.cudnn_benchmark,
        "cudnn_deterministic": summary.cudnn_deterministic,
        "data_loader_num_workers": summary.data_loader_num_workers,
        "shuffle_generator_device": summary.shuffle_generator_device,
        "training_wall_time_seconds": summary.training_wall_time_seconds,
    }


def _pitch_summary_from_document(value: object) -> PitchTrainingSummary:
    mapping = _mapping(value, "pitch training summary")
    fields = {
        "history",
        "selected_epoch",
        "training_examples",
        "validation_examples",
        "seed",
        "device",
        "deterministic_algorithms",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "data_loader_num_workers",
        "shuffle_generator_device",
        "training_wall_time_seconds",
    }
    _exact_fields(mapping, fields, "pitch training summary")
    history_value = mapping["history"]
    if not isinstance(history_value, list):
        raise ValueError("history must be a list")
    history: list[PitchEpochMetrics] = []
    for value_item in history_value:
        item = _mapping(value_item, "pitch epoch metrics")
        _exact_fields(item, {"epoch", "training", "validation"}, "pitch epoch metrics")
        history.append(
            PitchEpochMetrics(
                epoch=_integer(item["epoch"], "epoch", minimum=1),
                training=_metrics_from_document(item["training"]),
                validation=_metrics_from_document(item["validation"]),
            )
        )
    return PitchTrainingSummary(
        history=tuple(history),
        selected_epoch=_integer(mapping["selected_epoch"], "selected_epoch", minimum=1),
        training_examples=_integer(mapping["training_examples"], "training_examples", minimum=1),
        validation_examples=_integer(
            mapping["validation_examples"], "validation_examples", minimum=1
        ),
        seed=_integer(mapping["seed"], "seed"),
        device=_string(mapping["device"], "device"),
        deterministic_algorithms=_boolean(
            mapping["deterministic_algorithms"], "deterministic_algorithms"
        ),
        cudnn_benchmark=_boolean(mapping["cudnn_benchmark"], "cudnn_benchmark"),
        cudnn_deterministic=_boolean(mapping["cudnn_deterministic"], "cudnn_deterministic"),
        data_loader_num_workers=_integer(
            mapping["data_loader_num_workers"], "data_loader_num_workers"
        ),
        shuffle_generator_device=_string(
            mapping["shuffle_generator_device"], "shuffle_generator_device"
        ),
        training_wall_time_seconds=_document_float(
            mapping["training_wall_time_seconds"],
            "training_wall_time_seconds",
            minimum=0.0,
        ),
    )


@dataclass(frozen=True, slots=True)
class PitchTrainingSummaryDocument:
    schema_version: int
    trainer: PitchTrainerKind
    profile: ProfileName
    seed: int
    summary: PitchTrainingSummary

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        if self.trainer is not PitchTrainerKind.PITCH:
            raise ValueError("trainer must be the schema-v2 pitch trainer")
        if not isinstance(self.profile, ProfileName):
            raise ValueError("profile must be a ProfileName")
        seed = _integer(self.seed, "seed")
        if not isinstance(self.summary, PitchTrainingSummary):
            raise ValueError("summary must be a PitchTrainingSummary")
        configured = PITCH_PROFILE_CONFIGS[self.profile]
        if (
            len(self.summary.history) > configured.max_epochs
            or self.summary.training_examples != configured.training_coordinate_count
            or self.summary.validation_examples != configured.validation_coordinate_count
            or self.summary.seed != seed
        ):
            raise ValueError("summary must match the profile counts, epochs, and seed")
        expected_device = "cpu" if self.summary.device == "cpu" else self.summary.device
        if not expected_device.startswith("cpu") and not expected_device.startswith("cuda"):
            raise ValueError("summary device must identify CPU or CUDA")
        object.__setattr__(self, "seed", seed)

    def to_document(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "trainer": self.trainer.value,
            "profile": self.profile.value,
            "seed": self.seed,
            "summary": _pitch_summary_to_document(self.summary),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, JSONValue]) -> PitchTrainingSummaryDocument:
        mapping = _mapping(document, "pitch training summary document")
        schema_version = _schema_version(mapping.get("schema_version"))
        _exact_fields(
            mapping,
            {"schema_version", "trainer", "profile", "seed", "summary"},
            "pitch training summary document",
        )
        try:
            trainer = PitchTrainerKind(_string(mapping["trainer"], "trainer"))
            profile = ProfileName(_string(mapping["profile"], "profile"))
        except ValueError as error:
            raise ValueError("pitch summary enum identity is invalid") from error
        return cls(
            schema_version=schema_version,
            trainer=trainer,
            profile=profile,
            seed=_integer(mapping["seed"], "seed"),
            summary=_pitch_summary_from_document(mapping["summary"]),
        )


@dataclass(frozen=True, slots=True)
class PitchArtifactManifest:
    schema_version: int
    status: ArtifactStatus
    trainer: PitchTrainerKind
    profile: ProfileName
    seed: int
    created_at_utc: str
    completed_at_utc: str | None
    source: SourceStatus
    runtime: RuntimeStatus
    environment_id: str
    environment_contract_id: str
    train_distribution_id: str
    coordinate_split_digest_sha256: str
    evaluation_suites: tuple[tuple[str, str], ...]
    spectrum_grid_id: str
    preprocessing_schema_id: str
    architecture_schema_id: str
    policy_semantics_id: str
    parameter_count: int
    selected_epoch: int | None
    training_counts: PitchTrainingCounts
    evaluation_device: DeviceName | None
    eligible_for_aggregate: bool
    criterion_status: CriterionStatus
    criterion_met: bool | None
    compatibility_sha256: str
    files: tuple[FileRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        if not isinstance(self.status, ArtifactStatus):
            raise ValueError("status must be an ArtifactStatus")
        if self.trainer is not PitchTrainerKind.PITCH:
            raise ValueError("trainer must be the schema-v2 pitch trainer")
        if not isinstance(self.profile, ProfileName):
            raise ValueError("profile must be a ProfileName")
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        object.__setattr__(
            self,
            "created_at_utc",
            _timestamp(self.created_at_utc, "created_at_utc"),
        )
        if self.completed_at_utc is not None:
            object.__setattr__(
                self,
                "completed_at_utc",
                _timestamp(self.completed_at_utc, "completed_at_utc"),
            )
        if not isinstance(self.source, SourceStatus) or not isinstance(self.runtime, RuntimeStatus):
            raise ValueError("source and runtime must use the shared strict provenance models")
        expected_identities = (
            (self.environment_id, ENVIRONMENT_ID, "environment_id"),
            (self.environment_contract_id, ENVIRONMENT_CONTRACT_ID, "environment_contract_id"),
            (self.train_distribution_id, PITCH_DISTRIBUTION_ID, "train_distribution_id"),
            (
                self.coordinate_split_digest_sha256,
                PITCH_SPLIT_DIGEST_SHA256,
                "coordinate_split_digest_sha256",
            ),
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
        )
        for actual, expected, field_name in expected_identities:
            if actual != expected:
                raise ValueError(f"{field_name} must be {expected!r}")
        object.__setattr__(
            self,
            "evaluation_suites",
            _normalize_suite_records(self.evaluation_suites),
        )
        if self.parameter_count != PITCH_ESTIMATOR_PARAMETER_COUNT:
            raise ValueError(f"parameter_count must be {PITCH_ESTIMATOR_PARAMETER_COUNT}")
        if self.selected_epoch is not None:
            object.__setattr__(
                self,
                "selected_epoch",
                _integer(self.selected_epoch, "selected_epoch", minimum=1),
            )
        if not isinstance(self.training_counts, PitchTrainingCounts):
            raise ValueError("training_counts must be PitchTrainingCounts")
        configured = PITCH_PROFILE_CONFIGS[self.profile]
        if (
            self.training_counts.configured_training_coordinates
            != configured.training_coordinate_count
            or self.training_counts.configured_validation_coordinates
            != configured.validation_coordinate_count
        ):
            raise ValueError("training counts must match the checked-in pitch profile")
        if self.evaluation_device is not None and not isinstance(
            self.evaluation_device, DeviceName
        ):
            raise ValueError("evaluation_device must be a DeviceName or null")
        object.__setattr__(
            self,
            "eligible_for_aggregate",
            _boolean(self.eligible_for_aggregate, "eligible_for_aggregate"),
        )
        if not isinstance(self.criterion_status, CriterionStatus):
            raise ValueError("criterion_status must be a CriterionStatus")
        if self.criterion_met is not None:
            raise ValueError("single pitch artifacts never own the aggregate criterion result")
        expected_compatibility = pitch_compatibility_sha256(self.profile)
        if self.compatibility_sha256 != expected_compatibility:
            raise ValueError("compatibility_sha256 must match the canonical preimage")
        try:
            files = tuple(self.files)
        except TypeError as error:
            raise ValueError("files must contain FileRecord values") from error
        if not all(isinstance(item, FileRecord) for item in files):
            raise ValueError("files must contain FileRecord values")
        if len({item.relative_path for item in files}) != len(files):
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
            "coordinate_split_digest_sha256": self.coordinate_split_digest_sha256,
            "evaluation_suites": [
                {"suite_id": suite_id, "digest_sha256": digest}
                for suite_id, digest in self.evaluation_suites
            ],
            "spectrum_grid_id": self.spectrum_grid_id,
            "preprocessing_schema_id": self.preprocessing_schema_id,
            "architecture_schema_id": self.architecture_schema_id,
            "policy_semantics_id": self.policy_semantics_id,
            "parameter_count": self.parameter_count,
            "selected_epoch": self.selected_epoch,
            "training_counts": _counts_to_document(self.training_counts),
            "evaluation_device": (
                None if self.evaluation_device is None else self.evaluation_device.value
            ),
            "eligible_for_aggregate": self.eligible_for_aggregate,
            "criterion_status": self.criterion_status.value,
            "criterion_met": self.criterion_met,
            "compatibility_sha256": self.compatibility_sha256,
            "files": [_file_to_document(item) for item in self.files],
        }

    @classmethod
    def from_document(cls, document: Mapping[str, JSONValue]) -> PitchArtifactManifest:
        mapping = _mapping(document, "pitch artifact manifest")
        schema_version = _schema_version(mapping.get("schema_version"))
        fields = {
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
            "coordinate_split_digest_sha256",
            "evaluation_suites",
            "spectrum_grid_id",
            "preprocessing_schema_id",
            "architecture_schema_id",
            "policy_semantics_id",
            "parameter_count",
            "selected_epoch",
            "training_counts",
            "evaluation_device",
            "eligible_for_aggregate",
            "criterion_status",
            "criterion_met",
            "compatibility_sha256",
            "files",
        }
        _exact_fields(mapping, fields, "pitch artifact manifest")
        files_value = mapping["files"]
        if not isinstance(files_value, list):
            raise ValueError("files must be a list")
        try:
            status_value = ArtifactStatus(_string(mapping["status"], "status"))
            trainer = PitchTrainerKind(_string(mapping["trainer"], "trainer"))
            profile = ProfileName(_string(mapping["profile"], "profile"))
            criterion_status = CriterionStatus(
                _string(mapping["criterion_status"], "criterion_status")
            )
            evaluation_device = (
                None
                if mapping["evaluation_device"] is None
                else DeviceName(_string(mapping["evaluation_device"], "evaluation_device"))
            )
        except ValueError as error:
            raise ValueError("pitch manifest enum identity is invalid") from error
        return cls(
            schema_version=schema_version,
            status=status_value,
            trainer=trainer,
            profile=profile,
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
            coordinate_split_digest_sha256=_digest(
                mapping["coordinate_split_digest_sha256"],
                "coordinate_split_digest_sha256",
            ),
            evaluation_suites=_suite_records_from_document(mapping["evaluation_suites"]),
            spectrum_grid_id=_string(mapping["spectrum_grid_id"], "spectrum_grid_id"),
            preprocessing_schema_id=_string(
                mapping["preprocessing_schema_id"], "preprocessing_schema_id"
            ),
            architecture_schema_id=_string(
                mapping["architecture_schema_id"], "architecture_schema_id"
            ),
            policy_semantics_id=_string(mapping["policy_semantics_id"], "policy_semantics_id"),
            parameter_count=_integer(mapping["parameter_count"], "parameter_count", minimum=1),
            selected_epoch=(
                None
                if mapping["selected_epoch"] is None
                else _integer(mapping["selected_epoch"], "selected_epoch", minimum=1)
            ),
            training_counts=_counts_from_document(mapping["training_counts"]),
            evaluation_device=evaluation_device,
            eligible_for_aggregate=_boolean(
                mapping["eligible_for_aggregate"], "eligible_for_aggregate"
            ),
            criterion_status=criterion_status,
            criterion_met=mapping["criterion_met"],  # rejected unless null by __post_init__
            compatibility_sha256=_digest(mapping["compatibility_sha256"], "compatibility_sha256"),
            files=tuple(_file_from_document(item) for item in files_value),
        )


def _pitch_artifact_eligible(
    manifest: PitchArtifactManifest,
    *,
    evaluation_device: DeviceName,
) -> bool:
    return (
        manifest.profile is ProfileName.CHECKPOINT
        and manifest.seed in {0, 1, 2}
        and not manifest.source.dirty_tree
        and manifest.source.required_inputs_committed
        and manifest.runtime.device is DeviceName.CPU
        and evaluation_device is DeviceName.CPU
    )


def _validate_manifest_lifecycle(manifest: PitchArtifactManifest) -> None:
    counts = manifest.training_counts
    if manifest.status is ArtifactStatus.INCOMPLETE:
        if (
            manifest.completed_at_utc is not None
            or manifest.selected_epoch is not None
            or counts.training_examples is not None
            or counts.validation_examples is not None
            or manifest.evaluation_device is not None
            or manifest.eligible_for_aggregate
            or manifest.criterion_status is not CriterionStatus.INELIGIBLE
            or manifest.files
        ):
            raise ValueError("incomplete pitch manifest must not contain completion claims")
        return
    if manifest.completed_at_utc is None or manifest.evaluation_device is None:
        raise ValueError("complete pitch manifest requires completion time and evaluation device")
    if datetime.fromisoformat(manifest.completed_at_utc[:-1]) < datetime.fromisoformat(
        manifest.created_at_utc[:-1]
    ):
        raise ValueError("completed_at_utc must not precede created_at_utc")
    if (
        manifest.selected_epoch is None
        or counts.training_examples is None
        or counts.validation_examples is None
    ):
        raise ValueError("complete pitch manifest requires selected epoch and actual counts")
    if tuple(item.relative_path for item in manifest.files) != PITCH_REQUIRED_PAYLOAD_NAMES:
        raise ValueError("complete pitch manifest files must match the exact v2 inventory")
    expected_eligible = _pitch_artifact_eligible(
        manifest,
        evaluation_device=manifest.evaluation_device,
    )
    if manifest.eligible_for_aggregate != expected_eligible:
        raise ValueError("eligible_for_aggregate must be derived from provenance and devices")
    expected_status = (
        CriterionStatus.ELIGIBLE_FOR_AGGREGATE if expected_eligible else CriterionStatus.INELIGIBLE
    )
    if manifest.criterion_status is not expected_status:
        raise ValueError("criterion_status must match individual pitch eligibility")


class PitchSmokePayloadKind(StrEnum):
    """The two exact smoke-evaluation payload roles in each pitch artifact."""

    BASE = "base"
    PROBES = "probes"


def _optional_integer(value: object, field_name: str) -> int | None:
    return None if value is None else _integer(value, field_name)


def _optional_float(value: object, field_name: str) -> float | None:
    return None if value is None else _document_float(value, field_name, minimum=0.0)


def _pitch_row_to_document(row: object) -> dict[str, JSONValue]:
    from harpy.learning.evaluation import _metrics_to_document, _record_to_document
    from harpy.learning.pitch_evaluation import PitchEvaluationRow

    if not isinstance(row, PitchEvaluationRow):
        raise ValueError("rows must contain PitchEvaluationRow values")
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
        "metrics": _metrics_to_document(row.metrics),
        "episodes": [_record_to_document(record) for record in row.episodes],
    }


def _pitch_row_from_document(value: object) -> object:
    from harpy.learning.evaluation import _metrics_from_document, _record_from_document
    from harpy.learning.pitch_evaluation import PitchEvaluationRow

    mapping = _mapping(value, "pitch evaluation row")
    fields = {
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
    }
    _exact_fields(mapping, fields, "pitch evaluation row")
    episodes_value = mapping["episodes"]
    if not isinstance(episodes_value, list):
        raise ValueError("episodes must be a list")
    try:
        trainer = (
            None
            if mapping["trainer"] is None
            else PitchTrainerKind(_string(mapping["trainer"], "trainer"))
        )
        observation_mode = ObservationMode(_string(mapping["observation_mode"], "observation_mode"))
        suite_id = PitchEvaluationSuiteId(_string(mapping["suite_id"], "suite_id"))
    except ValueError as error:
        raise ValueError("pitch evaluation row enum identity is invalid") from error
    return PitchEvaluationRow(
        actor_id=_string(mapping["actor_id"], "actor_id"),
        trainer=trainer,
        seed=_optional_integer(mapping["seed"], "seed"),
        environment_id=_string(mapping["environment_id"], "environment_id"),
        observation_mode=observation_mode,
        suite_id=suite_id,
        suite_digest_sha256=_digest(mapping["suite_digest_sha256"], "suite_digest_sha256"),
        probe=None if mapping["probe"] is None else _string(mapping["probe"], "probe"),
        metrics=_metrics_from_document(mapping["metrics"]),
        episodes=tuple(_record_from_document(item) for item in episodes_value),
        parameter_count=_optional_integer(mapping["parameter_count"], "parameter_count"),
        training_examples=_optional_integer(mapping["training_examples"], "training_examples"),
        training_wall_time_seconds=_optional_float(
            mapping["training_wall_time_seconds"], "training_wall_time_seconds"
        ),
    )


@dataclass(frozen=True, slots=True)
class PitchSmokeEvaluationDocument:
    """One semantically re-derived half of the exact seven-row smoke matrix."""

    schema_version: int
    payload_kind: PitchSmokePayloadKind
    suite_id: PitchEvaluationSuiteId
    suite_digest_sha256: str
    rows: tuple[object, ...]

    def __post_init__(self) -> None:
        from harpy.learning.pitch_evaluation import (
            PITCH_SHUFFLED_SPECTRUM_PROBE,
            PITCH_ZERO_SPECTRUM_PROBE,
            PitchEvaluationRow,
        )

        object.__setattr__(self, "schema_version", _schema_version(self.schema_version))
        if not isinstance(self.payload_kind, PitchSmokePayloadKind):
            raise ValueError("payload_kind must be a PitchSmokePayloadKind")
        if self.suite_id is not PitchEvaluationSuiteId.SMOKE:
            raise ValueError("pitch artifact evaluation may contain only the smoke suite")
        expected_digest = PITCH_EVALUATION_SUITE_RECORDS[0][1]
        if self.suite_digest_sha256 != expected_digest:
            raise ValueError("suite_digest_sha256 must match the pinned smoke suite")
        try:
            rows = tuple(self.rows)
        except TypeError as error:
            raise ValueError("rows must contain PitchEvaluationRow values") from error
        if not all(isinstance(row, PitchEvaluationRow) for row in rows):
            raise ValueError("rows must contain PitchEvaluationRow values")
        if any(
            row.suite_id is not PitchEvaluationSuiteId.SMOKE
            or row.suite_digest_sha256 != expected_digest
            for row in rows
        ):
            raise ValueError("rows must contain only the exact smoke suite")
        baseline_order = ("random", "reward_search", "spectrum_peak", "oracle")
        if self.payload_kind is PitchSmokePayloadKind.BASE:
            if (
                len(rows) != 5
                or rows[0].trainer is not PitchTrainerKind.PITCH
                or rows[0].probe is not None
                or tuple(row.actor_id for row in rows[1:]) != baseline_order
                or any(row.trainer is not None or row.probe is not None for row in rows[1:])
            ):
                raise ValueError("base smoke payload must contain learned base then four baselines")
        elif (
            len(rows) != 2
            or tuple(row.probe for row in rows)
            != (PITCH_ZERO_SPECTRUM_PROBE, PITCH_SHUFFLED_SPECTRUM_PROBE)
            or any(row.trainer is not PitchTrainerKind.PITCH for row in rows)
            or len({(row.actor_id, row.seed) for row in rows}) != 1
        ):
            raise ValueError("probe smoke payload must contain exact zero and shuffled rows")
        object.__setattr__(self, "rows", rows)

    def to_document(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "payload_kind": self.payload_kind.value,
            "suite_id": self.suite_id.value,
            "suite_digest_sha256": self.suite_digest_sha256,
            "rows": [_pitch_row_to_document(row) for row in self.rows],
        }

    @classmethod
    def from_document(cls, document: Mapping[str, JSONValue]) -> PitchSmokeEvaluationDocument:
        mapping = _mapping(document, "pitch smoke evaluation")
        schema_version = _schema_version(mapping.get("schema_version"))
        _exact_fields(
            mapping,
            {
                "schema_version",
                "payload_kind",
                "suite_id",
                "suite_digest_sha256",
                "rows",
            },
            "pitch smoke evaluation",
        )
        rows_value = mapping["rows"]
        if not isinstance(rows_value, list):
            raise ValueError("rows must be a list")
        try:
            payload_kind = PitchSmokePayloadKind(_string(mapping["payload_kind"], "payload_kind"))
            suite_id = PitchEvaluationSuiteId(_string(mapping["suite_id"], "suite_id"))
        except ValueError as error:
            raise ValueError("pitch smoke evaluation enum identity is invalid") from error
        return cls(
            schema_version=schema_version,
            payload_kind=payload_kind,
            suite_id=suite_id,
            suite_digest_sha256=_digest(mapping["suite_digest_sha256"], "suite_digest_sha256"),
            rows=tuple(_pitch_row_from_document(row) for row in rows_value),
        )


def validate_pitch_smoke_evaluation_pair(
    base: PitchSmokeEvaluationDocument,
    probes: PitchSmokeEvaluationDocument,
) -> tuple[object, ...]:
    """Reconstruct and validate the exact Task 5 seven-row smoke matrix."""
    from harpy.learning.pitch_evaluation import validate_pitch_smoke_row_matrix

    if (
        not isinstance(base, PitchSmokeEvaluationDocument)
        or base.payload_kind is not PitchSmokePayloadKind.BASE
        or not isinstance(probes, PitchSmokeEvaluationDocument)
        or probes.payload_kind is not PitchSmokePayloadKind.PROBES
    ):
        raise ValueError("smoke evaluation pair must contain base then probes payloads")
    learned = base.rows[0]
    if any(
        (
            row.actor_id,
            row.seed,
            row.parameter_count,
            row.training_examples,
            row.training_wall_time_seconds,
        )
        != (
            learned.actor_id,
            learned.seed,
            learned.parameter_count,
            learned.training_examples,
            learned.training_wall_time_seconds,
        )
        for row in probes.rows
    ):
        raise ValueError("smoke probe rows must preserve learned actor metadata")
    return validate_pitch_smoke_row_matrix((base.rows[0], *probes.rows, *base.rows[1:]))


@dataclass(frozen=True, slots=True)
class PitchArtifactCompletion:
    completed_at_utc: str
    training_counts: PitchTrainingCounts
    evaluation_device: DeviceName

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "completed_at_utc",
            _timestamp(self.completed_at_utc, "completed_at_utc"),
        )
        if not isinstance(self.training_counts, PitchTrainingCounts):
            raise ValueError("training_counts must be PitchTrainingCounts")
        if not isinstance(self.evaluation_device, DeviceName):
            raise ValueError("evaluation_device must be a DeviceName")


def _validate_config_manifest(
    config: PitchTrainingConfigDocument,
    manifest: PitchArtifactManifest,
) -> None:
    if (
        config.schema_version != manifest.schema_version
        or config.trainer is not manifest.trainer
        or config.profile is not manifest.profile
        or config.seed != manifest.seed
        or config.device is not manifest.runtime.device
        or config.environment_id != manifest.environment_id
        or config.environment_contract_id != manifest.environment_contract_id
        or config.train_distribution_id != manifest.train_distribution_id
        or config.coordinate_split_digest_sha256 != manifest.coordinate_split_digest_sha256
        or config.evaluation_suites != manifest.evaluation_suites
        or config.spectrum_grid_id != manifest.spectrum_grid_id
        or config.preprocessing_schema_id != manifest.preprocessing_schema_id
        or config.architecture_schema_id != manifest.architecture_schema_id
        or config.policy_semantics_id != manifest.policy_semantics_id
        or config.parameter_count != manifest.parameter_count
        or config.compatibility_sha256 != manifest.compatibility_sha256
    ):
        raise ValueError("pitch training config identity must match the artifact manifest")


def _validate_summary_manifest(
    summary: PitchTrainingSummaryDocument,
    manifest: PitchArtifactManifest,
) -> None:
    if (
        summary.trainer is not manifest.trainer
        or summary.profile is not manifest.profile
        or summary.seed != manifest.seed
        or summary.summary.device != manifest.runtime.device.value
    ):
        raise ValueError("pitch training summary identity must match the artifact manifest")
    if manifest.status is ArtifactStatus.COMPLETE:
        counts = manifest.training_counts
        if (
            counts.training_examples != summary.summary.training_examples
            or counts.validation_examples != summary.summary.validation_examples
            or manifest.selected_epoch != summary.summary.selected_epoch
        ):
            raise ValueError("pitch manifest counts and selected epoch must match the summary")


def _validate_evaluation_metadata(
    manifest: PitchArtifactManifest,
    summary: PitchTrainingSummaryDocument,
    base: PitchSmokeEvaluationDocument,
    probes: PitchSmokeEvaluationDocument,
) -> None:
    rows = validate_pitch_smoke_evaluation_pair(base, probes)
    learned_rows = rows[:3]
    expected = (
        f"pitch-{manifest.seed}",
        manifest.seed,
        manifest.parameter_count,
        summary.summary.training_examples,
        summary.summary.training_wall_time_seconds,
    )
    if any(
        (
            row.actor_id,
            row.seed,
            row.parameter_count,
            row.training_examples,
            row.training_wall_time_seconds,
        )
        != expected
        for row in learned_rows
    ):
        raise ValueError("learned smoke rows must match artifact training metadata")


@dataclass(frozen=True, slots=True)
class LoadedPitchArtifact:
    root: Path
    manifest: PitchArtifactManifest

    def __post_init__(self) -> None:
        root = self.root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("artifact root must be a directory")
        if self.manifest.status is not ArtifactStatus.COMPLETE:
            raise ValueError("LoadedPitchArtifact requires a complete manifest")
        object.__setattr__(self, "root", root)

    def file(self, relative_path: str) -> Path:
        normalized = _relative_path(relative_path)
        record = next(
            (item for item in self.manifest.files if item.relative_path == normalized),
            None,
        )
        if record is None:
            raise ValueError(f"artifact payload {normalized!r} is unavailable")
        return _contained_file(self.root, normalized)

    def document(self, relative_path: str) -> dict[str, JSONValue]:
        path = self.file(relative_path)
        if path.suffix != ".json":
            raise ValueError(f"artifact payload {relative_path!r} is not JSON")
        return read_json_document(path)


@dataclass(frozen=True, slots=True)
class PendingPitchArtifactView:
    root: Path
    manifest: PitchArtifactManifest
    files: tuple[FileRecord, ...]
    _owner: weakref.ReferenceType[PitchArtifactWriter] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        owner = None if self._owner is None else self._owner()
        if owner is None:
            raise ValueError("pending view requires its live PitchArtifactWriter")
        root = self.root.resolve(strict=True)
        if (
            not root.is_dir()
            or self.manifest.status is not ArtifactStatus.INCOMPLETE
            or owner.root != root
            or owner._bootstrap is not self.manifest
            or owner._completed
        ):
            raise ValueError("pending view must belong to its live pitch writer")
        if not all(isinstance(item, FileRecord) for item in self.files):
            raise ValueError("files must contain FileRecord values")
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
            raise RuntimeError("pending pitch artifact view is no longer live")

    def file(self, relative_path: str) -> Path:
        self._ensure_live()
        normalized = _relative_path(relative_path)
        if normalized not in {item.relative_path for item in self.files}:
            raise ValueError(f"artifact payload {normalized!r} is unavailable")
        return _contained_file(self.root, normalized)

    def document(self, relative_path: str) -> dict[str, JSONValue]:
        path = self.file(relative_path)
        if path.suffix != ".json":
            raise ValueError(f"artifact payload {relative_path!r} is not JSON")
        return read_json_document(path)


def read_pitch_training_config(
    artifact: LoadedPitchArtifact | PendingPitchArtifactView,
) -> PitchTrainingConfigDocument:
    config = PitchTrainingConfigDocument.from_document(artifact.document("training-config.json"))
    _validate_config_manifest(config, artifact.manifest)
    return config


def read_pitch_training_summary(
    artifact: LoadedPitchArtifact | PendingPitchArtifactView,
) -> PitchTrainingSummaryDocument:
    summary = PitchTrainingSummaryDocument.from_document(artifact.document("training-summary.json"))
    _validate_summary_manifest(summary, artifact.manifest)
    return summary


def _validate_inventory(root: Path) -> None:
    expected = {"manifest.json", *PITCH_REQUIRED_PAYLOAD_NAMES}
    actual = {entry.name for entry in root.iterdir()}
    if actual != expected:
        raise ValueError(
            f"artifact inventory mismatch; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    for name in expected:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"artifact inventory entry {name!r} must be a regular file")


def _semantic_payloads(
    artifact: LoadedPitchArtifact | PendingPitchArtifactView,
) -> tuple[
    PitchTrainingConfigDocument,
    PitchTrainingSummaryDocument,
    PitchSmokeEvaluationDocument,
    PitchSmokeEvaluationDocument,
]:
    config = read_pitch_training_config(artifact)
    summary = read_pitch_training_summary(artifact)
    base, probes = _semantic_evaluation_payloads(artifact, summary)
    load_pitch_estimator_model(artifact.file("model.pt"))
    return config, summary, base, probes


def _semantic_evaluation_payloads(
    artifact: LoadedPitchArtifact | PendingPitchArtifactView,
    summary: PitchTrainingSummaryDocument,
) -> tuple[PitchSmokeEvaluationDocument, PitchSmokeEvaluationDocument]:
    """Decode both smoke payloads and re-derive their cross-file semantics."""
    return _semantic_evaluation_documents(
        artifact,
        summary,
        artifact.document("evaluation-smoke.json"),
        artifact.document("evaluation-smoke-probes.json"),
    )


def _semantic_evaluation_documents(
    artifact: LoadedPitchArtifact | PendingPitchArtifactView,
    summary: PitchTrainingSummaryDocument,
    base_document: Mapping[str, JSONValue],
    probes_document: Mapping[str, JSONValue],
) -> tuple[PitchSmokeEvaluationDocument, PitchSmokeEvaluationDocument]:
    """Re-derive suite-backed semantics from already integrity-decoded documents."""
    base = PitchSmokeEvaluationDocument.from_document(base_document)
    probes = PitchSmokeEvaluationDocument.from_document(probes_document)
    _validate_evaluation_metadata(artifact.manifest, summary, base, probes)
    return base, probes


def _load_pitch_artifact_metadata(
    root: Path,
) -> tuple[
    LoadedPitchArtifact,
    PitchTrainingSummaryDocument,
    dict[str, JSONValue],
    dict[str, JSONValue],
]:
    """Validate metadata and raw JSON without suite or trusted-model construction."""
    if not isinstance(root, Path):
        raise ValueError("artifact root must be a Path")
    try:
        resolved = root.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("artifact root does not exist") from error
    if not resolved.is_dir():
        raise ValueError("artifact root must be a directory")
    manifest_path = resolved / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        entries = tuple(resolved.iterdir())
        if not manifest_path.exists() and (
            not entries or all(_is_pitch_temporary(item) for item in entries)
        ):
            raise ValueError(
                "artifact bootstrap did not publish a manifest; directory is safe to remove"
            )
        raise ValueError("artifact manifest.json is missing or unsafe")
    manifest = PitchArtifactManifest.from_document(
        decode_json_bytes(_read_regular_file_bytes(manifest_path, "artifact manifest.json"))
    )
    if manifest.status is not ArtifactStatus.COMPLETE:
        raise ValueError("public artifact loader rejects an incomplete manifest")
    _validate_inventory(resolved)
    if tuple(item.relative_path for item in manifest.files) != PITCH_REQUIRED_PAYLOAD_NAMES:
        raise ValueError("manifest records do not match the exact v2 inventory")
    for record in manifest.files:
        _verify_file_record(resolved, record)
    loaded = LoadedPitchArtifact(root=resolved, manifest=manifest)
    read_pitch_training_config(loaded)
    summary = read_pitch_training_summary(loaded)
    base_document = loaded.document("evaluation-smoke.json")
    probes_document = loaded.document("evaluation-smoke-probes.json")
    return loaded, summary, base_document, probes_document


def load_pitch_artifact(root: Path) -> LoadedPitchArtifact:
    """Strictly load, hash-check, semantically decode, and reload one v2 artifact."""
    loaded, summary, base_document, probes_document = _load_pitch_artifact_metadata(root)
    _semantic_evaluation_documents(
        loaded,
        summary,
        base_document,
        probes_document,
    )
    load_pitch_estimator_model(loaded.file("model.pt"))
    return loaded


@dataclass(slots=True, weakref_slot=True)
class PitchArtifactWriter:
    """Create-only atomic schema-v2 pitch artifact writer."""

    root: Path
    _bootstrap: PitchArtifactManifest
    _bootstrap_bytes: bytes = field(repr=False)
    _records: dict[str, FileRecord] = field(default_factory=dict, repr=False)
    _completed: bool = field(default=False, repr=False)

    @classmethod
    def begin(
        cls,
        output: Path,
        manifest: PitchArtifactManifest,
    ) -> PitchArtifactWriter:
        if not isinstance(manifest, PitchArtifactManifest):
            raise ValueError("manifest must be a PitchArtifactManifest")
        snapshot = PitchArtifactManifest.from_document(manifest.to_document())
        if snapshot.status is not ArtifactStatus.INCOMPLETE:
            raise ValueError("PitchArtifactWriter.begin requires an incomplete manifest")
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
            raise RuntimeError("pitch artifact writer is already complete")

    def _publication_name(self, filename: str, *, json_payload: bool) -> str:
        self._ensure_active()
        normalized = _relative_path(filename, "filename")
        if "/" in normalized or normalized not in PITCH_REQUIRED_PAYLOAD_NAMES:
            raise ValueError("filename must belong to the exact pitch artifact inventory")
        if normalized.endswith(".json") != json_payload:
            raise ValueError("filename payload kind does not match publication method")
        if normalized in self._records:
            raise FileExistsError(self.root / normalized)
        return normalized

    def publish_json(
        self,
        filename: str,
        document: Mapping[str, JSONValue],
    ) -> FileRecord:
        normalized = self._publication_name(filename, json_payload=True)
        path = self.root / normalized
        _atomic_publish_bytes(path, canonical_json_bytes(document))
        record = _record_file(path, normalized)
        self._records[normalized] = record
        return record

    def publish_model(self, save: object) -> FileRecord:
        normalized = self._publication_name("model.pt", json_payload=False)
        if not callable(save):
            raise ValueError("save must be callable")
        path = self.root / normalized
        temporary = _temporary_path(path, preserve_suffix=True)
        try:
            save(temporary)
            if temporary.is_symlink() or not temporary.is_file():
                raise ValueError("model saver must create the requested temporary file")
            load_pitch_estimator_model(temporary)
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

    def pending_view(self, required_names: Sequence[str]) -> PendingPitchArtifactView:
        self._ensure_active()
        if isinstance(required_names, (str, bytes)):
            raise ValueError("required_names must be a sequence")
        names = tuple(_relative_path(item, "required name") for item in required_names)
        if len(set(names)) != len(names):
            raise ValueError("required_names must not contain duplicates")
        records: list[FileRecord] = []
        for name in names:
            record = self._records.get(name)
            if record is None:
                raise ValueError(f"required payload {name!r} has not been published")
            _verify_file_record(self.root, record)
            records.append(record)
        return PendingPitchArtifactView(
            root=self.root,
            manifest=self._bootstrap,
            files=tuple(records),
            _owner=weakref.ref(self),
        )

    def _verified_records(self) -> tuple[FileRecord, ...]:
        if set(self._records) != set(PITCH_REQUIRED_PAYLOAD_NAMES):
            raise ValueError("writer records do not contain the exact pitch inventory")
        _validate_inventory(self.root)
        records: list[FileRecord] = []
        for name in PITCH_REQUIRED_PAYLOAD_NAMES:
            record = self._records[name]
            _verify_file_record(self.root, record)
            records.append(record)
        return tuple(records)

    def complete(self, completion: PitchArtifactCompletion) -> LoadedPitchArtifact:
        self._ensure_active()
        if not isinstance(completion, PitchArtifactCompletion):
            raise ValueError("completion must be a PitchArtifactCompletion")
        if (self.root / "manifest.json").read_bytes() != self._bootstrap_bytes:
            raise ValueError("persisted incomplete manifest no longer matches bootstrap")
        records = self._verified_records()
        pending = self.pending_view(tuple(item.relative_path for item in records))
        config, summary, base, probes = _semantic_payloads(pending)
        del base, probes
        initial = self._bootstrap.training_counts
        counts = completion.training_counts
        if (
            counts.configured_training_coordinates != initial.configured_training_coordinates
            or counts.configured_validation_coordinates != initial.configured_validation_coordinates
            or counts.training_examples != summary.summary.training_examples
            or counts.validation_examples != summary.summary.validation_examples
        ):
            raise ValueError("completion counts must preserve config and match summary")
        eligible = _pitch_artifact_eligible(
            self._bootstrap,
            evaluation_device=completion.evaluation_device,
        )
        final_manifest = replace(
            self._bootstrap,
            status=ArtifactStatus.COMPLETE,
            completed_at_utc=completion.completed_at_utc,
            selected_epoch=summary.summary.selected_epoch,
            training_counts=counts,
            evaluation_device=completion.evaluation_device,
            eligible_for_aggregate=eligible,
            criterion_status=(
                CriterionStatus.ELIGIBLE_FOR_AGGREGATE if eligible else CriterionStatus.INELIGIBLE
            ),
            files=records,
        )
        _validate_config_manifest(config, final_manifest)
        _validate_summary_manifest(summary, final_manifest)
        temporary = _temporary_path(self.root / "manifest.json", preserve_suffix=False)
        try:
            _write_fsynced_file(temporary, canonical_json_bytes(final_manifest.to_document()))
            loaded = LoadedPitchArtifact(root=self.root, manifest=final_manifest)
        except BaseException:
            with suppress(FileNotFoundError):
                temporary.unlink()
            raise
        self._completed = True
        try:
            os.replace(temporary, self.root / "manifest.json")
        except BaseException:
            try:
                temporary.lstat()
            except FileNotFoundError:
                return loaded
            self._completed = False
            with suppress(FileNotFoundError):
                temporary.unlink()
            raise
        return loaded


def _is_pitch_temporary(path: Path) -> bool:
    return (
        not path.is_symlink()
        and path.is_file()
        and _PITCH_TEMPORARY_PATTERN.fullmatch(path.name) is not None
    )


def _remove_pre_manifest_residue(root: Path) -> None:
    if not root.is_dir():
        return
    entries = tuple(root.iterdir())
    if any(not _is_pitch_temporary(item) for item in entries):
        return
    for item in entries:
        item.unlink(missing_ok=True)
    root.rmdir()


def _validate_pitch_aggregate_identities(
    identities: Sequence[tuple[str, str, bool, str]],
) -> None:
    """Require one source, lock, input status, and compatibility identity."""
    normalized = tuple(identities)
    if len(normalized) != 3 or len(set(normalized)) != 1:
        raise PitchArtifactSetError(
            "pitch aggregate artifacts must share source, lock, and compatibility"
        )


def preflight_pitch_artifacts(
    artifact_paths: Sequence[Path],
) -> tuple[LoadedPitchArtifact, LoadedPitchArtifact, LoadedPitchArtifact]:
    """Validate the canonical checkpoint triple before suite or model construction."""
    if isinstance(artifact_paths, (str, bytes)):
        raise PitchArtifactSetError("artifact_paths must contain exactly three Paths")
    try:
        supplied = tuple(artifact_paths)
    except TypeError as error:
        raise PitchArtifactSetError("artifact_paths must contain exactly three Paths") from error
    if len(supplied) != 3 or not all(isinstance(item, Path) for item in supplied):
        raise PitchArtifactSetError("artifact_paths must contain exactly three Paths")
    roots = tuple(item.resolve(strict=True) for item in supplied)
    if len(set(roots)) != 3:
        raise PitchArtifactSetError("pitch aggregate artifacts must be distinct")

    # This first phase checks complete manifests, exact inventories, every hash,
    # duplicate/nonfinite-decodes every JSON payload, and re-derives config/summary
    # semantics. It deliberately does not construct suite-backed rows or models.
    metadata = tuple(_load_pitch_artifact_metadata(root) for root in roots)
    ordered_pairs = tuple(sorted(metadata, key=lambda item: item[0].manifest.seed))
    ordered = tuple(item[0] for item in ordered_pairs)
    if tuple(item.manifest.seed for item in ordered) != (0, 1, 2):
        raise PitchArtifactSetError("pitch aggregate seeds must be exactly 0, 1, and 2")
    if any(
        item.manifest.profile is not ProfileName.CHECKPOINT
        or not item.manifest.eligible_for_aggregate
        or item.manifest.criterion_status is not CriterionStatus.ELIGIBLE_FOR_AGGREGATE
        or item.manifest.runtime.device is not DeviceName.CPU
        or item.manifest.evaluation_device is not DeviceName.CPU
        or item.manifest.source.dirty_tree
        or not item.manifest.source.required_inputs_committed
        for item in ordered
    ):
        raise PitchArtifactSetError(
            "pitch aggregate artifacts must each be eligible CPU checkpoints"
        )
    identities = tuple(
        (
            item.manifest.source.commit,
            item.manifest.source.dependency_lock_sha256,
            item.manifest.source.required_inputs_committed,
            item.manifest.compatibility_sha256,
        )
        for item in ordered
    )
    _validate_pitch_aggregate_identities(identities)

    # Only a compatible exact triple may instantiate suite-backed evaluation rows.
    # Re-derive every smoke-row semantic before loading any trusted model bytes.
    metadata_by_root = {entry[0].root: entry for entry in metadata}
    for item in ordered:
        entry = metadata_by_root[item.root]
        _semantic_evaluation_documents(item, entry[1], entry[2], entry[3])
    for item in ordered:
        load_pitch_estimator_model(item.file("model.pt"))
    return ordered  # type: ignore[return-value]


def capture_pitch_source_status(start: Path) -> SourceStatus:
    """Capture source provenance against the complete schema-v2 input inventory."""
    resolved = start.resolve(strict=True)
    anchor = resolved if resolved.is_dir() else resolved.parent

    def git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            ["git", "-C", str(anchor), *arguments],
            check=check,
            capture_output=True,
        )

    root = Path(git("rev-parse", "--show-toplevel").stdout.decode("utf-8").strip()).resolve(
        strict=True
    )
    commit = (
        subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
        )
        .stdout.decode("ascii")
        .strip()
    )
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=normal"],
        check=True,
        capture_output=True,
    ).stdout
    tracked_diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
        check=True,
        capture_output=True,
    ).stdout
    required_inputs_committed = all(
        subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"HEAD:{relative_path}"],
            check=False,
            capture_output=True,
        ).returncode
        == 0
        for relative_path in _PITCH_REQUIRED_SOURCE_INPUTS
    )
    return SourceStatus(
        commit=commit,
        dirty_tree=bool(status),
        tracked_diff_sha256=hashlib.sha256(tracked_diff).hexdigest(),
        dependency_lock_sha256=hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        required_inputs_committed=required_inputs_committed,
    )


__all__ = [
    "PITCH_ACTION_SCHEMA_ID",
    "PITCH_ARTIFACT_SCHEMA_REGISTRIES",
    "PITCH_ARTIFACT_SCHEMA_V2_REGISTRY",
    "PITCH_ARTIFACT_SCHEMA_VERSION",
    "PITCH_EVALUATION_SUITE_RECORDS",
    "PITCH_POLICY_SEMANTICS_ID",
    "PITCH_PREPROCESSING_SCHEMA_ID",
    "PITCH_REQUIRED_PAYLOAD_NAMES",
    "LoadedPitchArtifact",
    "PendingPitchArtifactView",
    "PitchArtifactCompletion",
    "PitchArtifactManifest",
    "PitchArtifactSchemaRegistry",
    "PitchArtifactWriter",
    "PitchSmokeEvaluationDocument",
    "PitchSmokePayloadKind",
    "PitchTrainingConfigDocument",
    "PitchTrainingCounts",
    "PitchTrainingSummaryDocument",
    "capture_pitch_source_status",
    "load_pitch_artifact",
    "pitch_compatibility_sha256",
    "preflight_pitch_artifacts",
    "read_pitch_training_config",
    "read_pitch_training_summary",
    "validate_pitch_smoke_evaluation_pair",
]
