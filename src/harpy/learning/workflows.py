"""Strict cross-trainer evaluation over complete local artifacts."""

from __future__ import annotations

import hashlib
import math
import operator
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import gymnasium

from harpy._serialization import canonical_json_bytes as canonical_json_bytes
from harpy.envs.baselines import BaselineKind
from harpy.envs.models import ObservationMode, PitchAction
from harpy.learning.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    PITCH_ARTIFACT_SCHEMA_VERSION,
    ArtifactStatus,
    CriterionStatus,
    LoadedArtifact,
    SourceStatus,
    _read_regular_file_bytes,
    decode_json_bytes,
    load_artifact,
    read_training_summary,
)
from harpy.learning.cache import SpectrumEvidenceCache
from harpy.learning.dependencies import require_training_dependencies
from harpy.learning.envs import make_cached_sine_pitch_env
from harpy.learning.errors import (
    ArtifactError,
    DependencyUnavailableError,
    LearningContractError,
    LearningExecutionError,
    PitchArtifactSetError,
)
from harpy.learning.evaluation import (
    SHUFFLED_SPECTRUM_PROBE,
    ZERO_SPECTRUM_PROBE,
    BCScientificCriterion,
    EvaluationFile,
    EvaluationRow,
    PPOScientificCriterion,
    build_evaluation_rows,
    evaluate_baseline_suite,
    evaluate_bc_criterion,
    evaluate_learned_actor,
    evaluate_ppo_criterion,
    make_spectrum_probe_factory,
)
from harpy.learning.evaluation_reports import (
    EvaluationReport as EvaluationReport,
)
from harpy.learning.evaluation_reports import (
    _actor_row_blocks as _actor_row_blocks,
)
from harpy.learning.evaluation_reports import (
    _base_iid_row as _base_iid_row,
)
from harpy.learning.evaluation_reports import (
    _bc_criterion_document as _bc_criterion_document,
)
from harpy.learning.evaluation_reports import (
    _bc_criterion_from_document as _bc_criterion_from_document,
)
from harpy.learning.evaluation_reports import (
    _expected_row_lanes as _expected_row_lanes,
)
from harpy.learning.evaluation_reports import (
    _ppo_criterion_document as _ppo_criterion_document,
)
from harpy.learning.evaluation_reports import (
    _ppo_criterion_from_document as _ppo_criterion_from_document,
)
from harpy.learning.evaluation_reports import (
    _row_documents as _row_documents,
)
from harpy.learning.evaluation_reports import (
    _rows_from_documents as _rows_from_documents,
)
from harpy.learning.evaluation_reports import (
    _validate_canonical_actor_order as _validate_canonical_actor_order,
)
from harpy.learning.evaluation_reports import (
    _validate_complete_row_matrix as _validate_complete_row_matrix,
)
from harpy.learning.evaluation_reports import (
    _validate_visible_criteria as _validate_visible_criteria,
)
from harpy.learning.evaluation_reports import (
    _visible_actor_seeds as _visible_actor_seeds,
)
from harpy.learning.evaluation_reports import (
    evaluation_report_bytes as evaluation_report_bytes,
)
from harpy.learning.models import (
    ENVIRONMENT_ID,
    PROFILE_CONFIGS,
    BCTrainingSummary,
    DeviceName,
    EvaluationSuiteId,
    PPOTrainingSummary,
    ProfileName,
    TrainerKind,
)
from harpy.learning.suites import fixed_evaluation_suite
from harpy.learning.trace import EpisodeTrace

EVALUATION_REPORT_SCHEMA_VERSION = 1
_TRAIN_DISTRIBUTION_ID = "harpy-sine-policy-train-v1"
_BASELINE_ORDER = (
    BaselineKind.RANDOM,
    BaselineKind.REWARD_SEARCH,
    BaselineKind.SPECTRUM_PEAK,
    BaselineKind.ORACLE,
)
_BASELINE_POSITION = {kind.value: index for index, kind in enumerate(_BASELINE_ORDER)}


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


def _exact_fields(mapping: Mapping[str, object], fields: set[str], field: str) -> None:
    if set(mapping) != fields:
        raise ValueError(f"{field} fields must be exactly {sorted(fields)}")


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _sha256_digest(value: object, field: str) -> str:
    normalized = _string(value, field)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return normalized


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a bool")
    return value


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


def _enum[EnumT](enum_type: type[EnumT], value: object, field: str) -> EnumT:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a valid {enum_type.__name__}")
    try:
        return enum_type(value)  # type: ignore[call-arg]
    except ValueError as error:
        raise ValueError(f"{field} must be a valid {enum_type.__name__}") from error


def evaluation_report_from_bytes(content: bytes) -> object:
    """Compatibility entrypoint retaining the historical strict-decoder hook."""
    from harpy.learning.evaluation_reports import evaluation_report_from_bytes as decode

    return decode(content, decoder=decode_json_bytes)


def evaluate_artifacts(
    artifact_paths: Sequence[Path],
    *,
    device: DeviceName = DeviceName.CPU,
    exploratory: bool = False,
) -> object:
    """Schema-peek a homogeneous set before any full artifact or actor load."""

    if not isinstance(device, DeviceName):
        raise LearningContractError("device must be a DeviceName")
    if not isinstance(exploratory, bool):
        raise LearningContractError("exploratory must be a bool")
    paths = _canonical_artifact_paths(artifact_paths)
    identities = tuple(_peek_artifact_identity(path) for path in paths)
    if exploratory:
        _validate_exploratory_pitch_request(paths, identities)
        artifact = _load_exploratory_pitch_artifact(paths[0], identities[0])
        _require_evaluation_device(device)
        return _evaluate_exploratory_pitch_artifact(artifact, device=device)
    schemas = {identity[0] for identity in identities}
    if len(schemas) != 1:
        raise LearningContractError("mixed artifact schemas are not supported")
    schema_version = next(iter(schemas))
    if schema_version == ARTIFACT_SCHEMA_VERSION:
        return _evaluate_v1_artifacts(paths, device=device)
    if schema_version == PITCH_ARTIFACT_SCHEMA_VERSION:
        return _evaluate_pitch_artifacts(paths, identities, device=device)
    raise LearningContractError(f"unsupported artifact schema_version {schema_version}")


def _validate_exploratory_pitch_request(
    paths: Sequence[Path],
    identities: Sequence[tuple[int, ProfileName, str]],
) -> None:
    if len(paths) != 1 or identities[0][0] != PITCH_ARTIFACT_SCHEMA_VERSION:
        raise LearningContractError("exploratory evaluation requires exactly one pitch artifact")


def _load_exploratory_pitch_artifact(path: Path, identity: tuple[int, ProfileName, str]) -> object:
    """Relax cohort admission only; retain full artifact inventory and semantic validation."""
    from harpy.learning.pitch_artifacts import LoadedPitchArtifact

    artifact = _load_complete_artifact(path)
    if not isinstance(artifact, LoadedPitchArtifact):
        raise LearningContractError("schema-v2 dispatch did not load a pitch artifact")
    if (artifact.manifest.profile, artifact.manifest.trainer.value) != identity[1:]:
        raise LearningContractError("loaded pitch artifact does not match manifest dispatch")
    return artifact


def _artifact_readout_provenance(artifact: object, *, device: DeviceName) -> object:
    """Bind exported results to the exact validated manifest and model payload."""
    from harpy.learning.experiment_results import ArtifactProvenance

    manifest = artifact.manifest
    content = _read_regular_file_bytes(artifact.root / "manifest.json", "artifact manifest.json")
    if decode_json_bytes(content) != manifest.to_document():
        raise ArtifactError("artifact manifest changed while producing its readout")
    model_name = "model.zip" if manifest.trainer.value == "ppo" else "model.pt"
    model_record = next(item for item in manifest.files if item.relative_path == model_name)
    return ArtifactProvenance(
        artifact_schema_version=manifest.schema_version,
        manifest_sha256=hashlib.sha256(content).hexdigest(),
        model_sha256=model_record.sha256,
        trainer=manifest.trainer.value,
        training_seed=manifest.seed,
        profile=manifest.profile,
        training_device=manifest.runtime.device,
        evaluation_device=device,
        source=manifest.source,
    )


def _evaluate_exploratory_pitch_artifact(artifact: object, *, device: DeviceName) -> object:
    """Measure one model and matched baselines without invoking a scientific criterion."""
    from harpy.learning.experiment_results import ExploratoryPitchEvaluation
    from harpy.learning.pitch_artifacts import read_pitch_training_summary
    from harpy.learning.pitch_data import (
        PitchEvaluationSuiteId,
        PitchTrainerKind,
        fixed_pitch_evaluation_suite,
    )
    from harpy.learning.pitch_evaluation import (
        build_pitch_evaluation_row,
        evaluate_pitch_baseline_suite,
        evaluate_pitch_learned_actor,
    )

    provenance = _artifact_readout_provenance(artifact, device=device)
    manifest = artifact.manifest
    summary = read_pitch_training_summary(artifact).summary
    actor = _load_pitch_artifact_actor(artifact, device=device)
    suite_ids = (
        (PitchEvaluationSuiteId.SMOKE,)
        if manifest.profile is ProfileName.SMOKE
        else (
            PitchEvaluationSuiteId.IID,
            PitchEvaluationSuiteId.OOD_LOWER,
            PitchEvaluationSuiteId.OOD_UPPER,
        )
    )
    suites = tuple(fixed_pitch_evaluation_suite(suite_id) for suite_id in suite_ids)
    cache = SpectrumEvidenceCache()
    rows = [
        build_pitch_evaluation_row(
            actor_id=f"pitch-{manifest.seed}",
            trainer=PitchTrainerKind.PITCH,
            seed=manifest.seed,
            environment_id=manifest.environment_id,
            observation_mode=ObservationMode.SPECTRUM,
            suite=suite,
            records=evaluate_pitch_learned_actor(
                actor, suite, environment_factory=lambda: make_cached_sine_pitch_env(cache)
            ),
            parameter_count=manifest.parameter_count,
            training_examples=summary.training_examples,
            training_wall_time_seconds=summary.training_wall_time_seconds,
        )
        for suite in suites
    ]
    for kind in BaselineKind:
        for suite in suites:
            rows.append(
                build_pitch_evaluation_row(
                    actor_id=kind.value,
                    trainer=None,
                    seed=None,
                    environment_id=kind.environment_id,
                    observation_mode=kind.observation_mode,
                    suite=suite,
                    records=evaluate_pitch_baseline_suite(kind, suite, cache=cache),
                )
            )
    return ExploratoryPitchEvaluation(provenance, tuple(rows))


def _evaluate_v1_artifacts(
    paths: tuple[Path, ...],
    *,
    device: DeviceName,
) -> EvaluationReport:
    """Preserve the frozen schema-v1 multi-artifact evaluation behavior."""

    artifacts = tuple(_load_complete_artifact(path) for path in paths)
    _validate_artifact_set(artifacts)
    _require_evaluation_device(device)
    for artifact in artifacts:
        try:
            _validate_artifact_payload(artifact)
        except DependencyUnavailableError:
            raise
        except Exception as error:
            raise ArtifactError(
                f"invalid {artifact.manifest.trainer.value} artifact: {error}"
            ) from error

    canonical_artifacts = tuple(sorted(artifacts, key=_artifact_order_key))
    actors: list[tuple[LoadedArtifact, object]] = []
    for artifact in canonical_artifacts:
        try:
            actor = _load_artifact_actor(artifact, device=device)
        except DependencyUnavailableError:
            raise
        except Exception as error:
            raise ArtifactError(
                f"could not load {artifact.manifest.trainer.value} actor: {error}"
            ) from error
        actors.append((artifact, actor))

    rows: list[EvaluationRow] = []
    try:
        for artifact, actor in actors:
            rows.extend(_evaluate_learned_rows(actor, artifact))
        profile = canonical_artifacts[0].manifest.profile
        for kind in _BASELINE_ORDER:
            rows.extend(_evaluate_baseline_rows(kind, profile=profile))
    except Exception as error:
        raise LearningExecutionError(f"artifact evaluation failed: {error}") from error

    bc_artifacts = tuple(
        artifact for artifact in canonical_artifacts if artifact.manifest.trainer is TrainerKind.BC
    )
    ppo_artifacts = tuple(
        artifact for artifact in canonical_artifacts if artifact.manifest.trainer is TrainerKind.PPO
    )
    bc_criterion = _evaluate_report_bc_criterion(
        bc_artifacts,
        tuple(rows),
        device=device,
    )
    ppo_criterion = _evaluate_report_ppo_criterion(
        ppo_artifacts,
        bc_artifacts,
        tuple(rows),
        device=device,
    )
    reference = canonical_artifacts[0].manifest
    return EvaluationReport(
        schema_version=EVALUATION_REPORT_SCHEMA_VERSION,
        profile=reference.profile,
        evaluation_device=device,
        environment_contract_id=reference.environment_contract_id,
        spectrum_grid_id=reference.spectrum_grid_id,
        preprocessing_schema_id=reference.preprocessing_schema_id,
        architecture_schema_id=reference.architecture_schema_id,
        rows=tuple(rows),
        bc_criterion=bc_criterion,
        ppo_criterion=ppo_criterion,
    )


def _peek_artifact_identity(path: Path) -> tuple[int, ProfileName, str]:
    """Read only the manifest schema/profile/trainer dispatch fields."""

    try:
        document = decode_json_bytes(
            _read_regular_file_bytes(path / "manifest.json", "artifact manifest.json")
        )
        schema_version = _integer(
            document.get("schema_version"),
            "schema_version",
            minimum=1,
        )
    except Exception as error:
        raise ArtifactError(f"invalid artifact manifest dispatch at {path}: {error}") from error
    if schema_version not in (
        ARTIFACT_SCHEMA_VERSION,
        PITCH_ARTIFACT_SCHEMA_VERSION,
    ):
        raise LearningContractError(f"unsupported artifact schema_version {schema_version}")
    try:
        profile = _enum(ProfileName, document.get("profile"), "profile")
        trainer = _string(document.get("trainer"), "trainer")
    except Exception as error:
        raise ArtifactError(f"invalid artifact manifest dispatch at {path}: {error}") from error
    if schema_version == ARTIFACT_SCHEMA_VERSION:
        if trainer not in (TrainerKind.BC.value, TrainerKind.PPO.value):
            raise LearningContractError("schema-v1 artifacts require trainer bc or ppo")
    elif schema_version == PITCH_ARTIFACT_SCHEMA_VERSION and trainer != "pitch":
        raise LearningContractError("schema-v2 artifacts require trainer pitch")
    return schema_version, profile, trainer


def _peek_artifact_seed(path: Path) -> int:
    """Read only the manifest seed needed by dependency-light set preflight."""

    try:
        document = decode_json_bytes(
            _read_regular_file_bytes(path / "manifest.json", "artifact manifest.json")
        )
        return _integer(document.get("seed"), "seed")
    except Exception as error:
        raise ArtifactError(f"invalid artifact manifest seed at {path}: {error}") from error


def _peek_pitch_training_device(path: Path) -> DeviceName:
    """Read only the schema-v2 training device needed for protocol routing."""

    try:
        document = decode_json_bytes(
            _read_regular_file_bytes(path / "manifest.json", "artifact manifest.json")
        )
        runtime = _mapping(document.get("runtime"), "runtime")
        return _enum(DeviceName, runtime.get("device"), "runtime.device")
    except Exception as error:
        raise ArtifactError(f"invalid pitch training-device dispatch at {path}: {error}") from error


def _peek_pitch_aggregate_identity(path: Path) -> tuple[str, str, bool, str, DeviceName]:
    """Read the manifest-only eligibility and compatibility fields for final diagnostics."""

    try:
        document = decode_json_bytes(
            _read_regular_file_bytes(path / "manifest.json", "artifact manifest.json")
        )
        status = _enum(ArtifactStatus, document.get("status"), "status")
        eligible = _boolean(
            document.get("eligible_for_aggregate"),
            "eligible_for_aggregate",
        )
        criterion_status = _enum(
            CriterionStatus,
            document.get("criterion_status"),
            "criterion_status",
        )
        if document.get("criterion_met") is not None:
            raise ValueError("criterion_met must be null for a pitch artifact")
        runtime = _mapping(document.get("runtime"), "runtime")
        training_device = _enum(DeviceName, runtime.get("device"), "runtime.device")
        evaluation_device = _enum(
            DeviceName,
            document.get("evaluation_device"),
            "evaluation_device",
        )
        source = _mapping(document.get("source"), "source")
        if source.get("source_kind") == "package_snapshot":
            raise LearningContractError(
                "package snapshots are ineligible for scientific cohorts; use --exploratory"
            )
        commit = _string(source.get("commit"), "source.commit")
        dirty_tree = _boolean(source.get("dirty_tree"), "source.dirty_tree")
        dependency_lock = _sha256_digest(
            source.get("dependency_lock_sha256"),
            "source.dependency_lock_sha256",
        )
        required_inputs_committed = _boolean(
            source.get("required_inputs_committed"),
            "source.required_inputs_committed",
        )
        compatibility = _sha256_digest(
            document.get("compatibility_sha256"),
            "compatibility_sha256",
        )
    except Exception as error:
        if isinstance(error, LearningContractError):
            raise
        raise ArtifactError(
            f"invalid pitch aggregate manifest dispatch at {path}: {error}"
        ) from error
    if status is not ArtifactStatus.COMPLETE:
        raise ArtifactError(f"pitch aggregate artifact is incomplete: {path}")
    cpu_protocol = (
        training_device is DeviceName.CPU
        and eligible
        and criterion_status is CriterionStatus.ELIGIBLE_FOR_AGGREGATE
    )
    cuda_protocol = (
        training_device is DeviceName.CUDA
        and not eligible
        and criterion_status is CriterionStatus.INELIGIBLE
    )
    if (
        not (cpu_protocol or cuda_protocol)
        or evaluation_device is not DeviceName.CPU
        or dirty_tree
        or not required_inputs_committed
    ):
        raise LearningContractError(
            "invalid pitch checkpoint set: pitch aggregate artifacts must each be "
            "valid CPU checkpoints or prospective E.1 CUDA checkpoints"
        )
    return commit, dependency_lock, required_inputs_committed, compatibility, training_device


def _preflight_pitch_artifacts(
    paths: Sequence[Path],
) -> tuple[object, object, object]:
    """Lazy Task 6 bridge retained as the one exact-three gate call."""

    from harpy.learning.pitch_artifacts import preflight_pitch_artifacts

    return preflight_pitch_artifacts(paths)


def _preflight_pitch_e1_artifacts(paths: Sequence[Path]) -> object:
    """Lazy bridge to the prospective homogeneous-CUDA cohort gate."""

    from harpy.learning.pitch_artifacts import (
        capture_pitch_source_status,
        preflight_pitch_e1_artifacts,
    )

    return preflight_pitch_e1_artifacts(
        paths,
        evaluator_source=capture_pitch_source_status(Path(__file__)),
    )


def _evaluate_pitch_artifacts(
    paths: tuple[Path, ...],
    identities: tuple[tuple[int, ProfileName, str], ...],
    *,
    device: DeviceName,
) -> object:
    """Route smoke or exact-three checkpoint schema-v2 evaluation."""

    profiles = tuple(identity[1] for identity in identities)
    if len(paths) == 1:
        if profiles != (ProfileName.SMOKE,):
            raise LearningContractError(
                "checkpoint pitch evaluation requires exactly three artifacts"
            )
        artifact = _load_complete_artifact(paths[0])
        from harpy.learning.pitch_artifacts import LoadedPitchArtifact

        if not isinstance(artifact, LoadedPitchArtifact):
            raise LearningContractError("schema-v2 dispatch did not load a pitch artifact")
        if (
            artifact.manifest.profile is not profiles[0]
            or artifact.manifest.trainer.value != identities[0][2]
        ):
            raise LearningContractError("loaded pitch artifact does not match manifest dispatch")
        if artifact.manifest.evaluation_device is not device:
            raise LearningContractError(
                "smoke report reconstruction requires the artifact evaluation device"
            )
        _require_evaluation_device(device)
        return _evaluate_pitch_smoke(artifact, device=device)
    if len(paths) != 3 or any(profile is not ProfileName.CHECKPOINT for profile in profiles):
        raise LearningContractError(
            "pitch evaluation requires one smoke artifact or exactly three checkpoints"
        )
    training_devices = tuple(_peek_pitch_training_device(path) for path in paths)
    e1_requested = any(item is DeviceName.CUDA for item in training_devices)
    if e1_requested and device is not DeviceName.CPU:
        raise LearningContractError("E.1 final evaluation must use CPU")
    try:
        admitted = (
            _preflight_pitch_e1_artifacts(paths)
            if e1_requested
            else _preflight_pitch_artifacts(paths)
        )
    except PitchArtifactSetError as error:
        raise LearningContractError(f"invalid pitch checkpoint set: {error}") from error
    except LearningContractError as error:
        raise ArtifactError(f"invalid pitch checkpoint artifact: {error}") from error
    except ValueError as error:
        raise ArtifactError(f"invalid pitch checkpoint artifact: {error}") from error
    _require_evaluation_device(device)
    try:
        if e1_requested:
            return _evaluate_pitch_e1_checkpoint(admitted)
        return _evaluate_pitch_checkpoint(admitted, device=device)
    except DependencyUnavailableError:
        raise
    except (ArtifactError, LearningContractError, LearningExecutionError):
        raise
    except Exception as error:
        raise LearningExecutionError(f"pitch checkpoint evaluation failed: {error}") from error


def _evaluate_pitch_e1_checkpoint(cohort: object) -> object:
    """Evaluate one prospective CUDA cohort on CPU and bind complete provenance."""

    from harpy.learning.pitch_artifacts import PitchE1ArtifactCohort
    from harpy.learning.pitch_e1 import (
        PitchE1ArtifactProvenance,
        PitchE1EvaluationReport,
        PitchE1RawEvidence,
    )

    if not isinstance(cohort, PitchE1ArtifactCohort):
        raise LearningContractError("E.1 evaluation requires an admitted CUDA cohort")
    evaluator_source = cohort.evaluator_source
    provenance = tuple(
        PitchE1ArtifactProvenance(
            seed=artifact.manifest.seed,
            artifact_manifest_sha256=_artifact_manifest_sha256(artifact.root),
            source=artifact.manifest.source,
            runtime=artifact.manifest.runtime,
            compatibility_sha256=artifact.manifest.compatibility_sha256,
        )
        for artifact in cohort.artifacts
    )
    pitch_report = _evaluate_pitch_checkpoint(cohort.artifacts, device=DeviceName.CPU)
    return PitchE1EvaluationReport.create(
        artifacts=provenance,
        evaluator_source=evaluator_source,
        evidence=PitchE1RawEvidence.from_pitch_report(pitch_report),
    )


def _evaluate_pitch_smoke(artifact: object, *, device: DeviceName) -> object:
    """Reconstruct the exact persisted seven-row ineligible smoke report."""

    from harpy.learning.pitch_artifacts import (
        LoadedPitchArtifact,
        PitchSmokeEvaluationDocument,
        validate_pitch_smoke_evaluation_pair,
    )
    from harpy.learning.pitch_evaluation import evaluate_pitch_criterion
    from harpy.learning.pitch_reports import PitchEvaluationReport

    if not isinstance(artifact, LoadedPitchArtifact):
        raise LearningContractError("artifact must be a LoadedPitchArtifact")
    manifest = artifact.manifest
    if manifest.profile is not ProfileName.SMOKE:
        raise LearningContractError("single pitch artifacts may evaluate only smoke")
    if device is not manifest.evaluation_device:
        raise LearningContractError(
            "smoke report reconstruction requires the artifact evaluation device"
        )
    base = PitchSmokeEvaluationDocument.from_document(artifact.document("evaluation-smoke.json"))
    probes = PitchSmokeEvaluationDocument.from_document(
        artifact.document("evaluation-smoke-probes.json")
    )
    rows = validate_pitch_smoke_evaluation_pair(base, probes)
    return PitchEvaluationReport(
        schema_version=PITCH_ARTIFACT_SCHEMA_VERSION,
        profile=manifest.profile,
        evaluation_device=manifest.evaluation_device,
        environment_contract_id=manifest.environment_contract_id,
        spectrum_grid_id=manifest.spectrum_grid_id,
        preprocessing_schema_id=manifest.preprocessing_schema_id,
        architecture_schema_id=manifest.architecture_schema_id,
        policy_semantics_id=manifest.policy_semantics_id,
        coordinate_distribution_id=manifest.train_distribution_id,
        coordinate_split_digest_sha256=manifest.coordinate_split_digest_sha256,
        compatibility_sha256=manifest.compatibility_sha256,
        terminal_rows=rows,
        coordinate_evaluations=(),
        register_ood_aggregates=(),
        criterion=evaluate_pitch_criterion(
            coordinate_evaluations=(),
            terminal_rows=(),
            eligible=False,
        ),
    )


def _evaluate_pitch_checkpoint(
    artifacts: Sequence[object],
    *,
    device: DeviceName,
) -> object:
    """Run final direct and closed-loop gates using one retained model per seed."""

    from harpy.learning.pitch import load_pitch_estimator_model
    from harpy.learning.pitch_actor import PitchPlannerActor
    from harpy.learning.pitch_artifacts import (
        LoadedPitchArtifact,
        read_pitch_training_summary,
    )
    from harpy.learning.pitch_data import (
        PitchEvaluationSuiteId,
        PitchTrainerKind,
        fixed_pitch_evaluation_suite,
    )
    from harpy.learning.pitch_evaluation import (
        PITCH_SHUFFLED_SPECTRUM_PROBE,
        PITCH_ZERO_SPECTRUM_PROBE,
        RegisterOODAggregate,
        build_pitch_evaluation_row,
        default_pitch_evidence_provider,
        evaluate_final_pitch_coordinates,
        evaluate_pitch_baseline_suite,
        evaluate_pitch_criterion,
        evaluate_pitch_learned_actor,
        make_pitch_model_grid_predictor,
        make_pitch_spectrum_probe_factory,
    )
    from harpy.learning.pitch_reports import PitchEvaluationReport

    normalized = tuple(artifacts)
    if len(normalized) != 3 or not all(
        isinstance(item, LoadedPitchArtifact) for item in normalized
    ):
        raise LearningContractError("checkpoint evaluation requires three loaded pitch artifacts")
    if tuple(item.manifest.seed for item in normalized) != (0, 1, 2):
        raise LearningContractError("checkpoint artifacts must be ordered seeds 0, 1, and 2")
    stack = require_training_dependencies()
    torch_device = stack.torch.device(device.value)
    retained = []
    for artifact in normalized:
        model = load_pitch_estimator_model(artifact.file("model.pt"), device=torch_device)
        actor = PitchPlannerActor(model, device=torch_device)
        predictor = make_pitch_model_grid_predictor(model, device=torch_device)
        retained.append((artifact, model, actor, predictor))

    cache = SpectrumEvidenceCache()
    provider = default_pitch_evidence_provider(cache)

    def environment_factory() -> gymnasium.Env:
        return make_cached_sine_pitch_env(cache)

    coordinates = tuple(
        evaluate_final_pitch_coordinates(
            seed=artifact.manifest.seed,
            evidence_provider=provider,
            predict_grid_index=predictor,
        )
        for artifact, _model, _actor, predictor in retained
    )
    suites = {
        suite_id: fixed_pitch_evaluation_suite(suite_id)
        for suite_id in (
            PitchEvaluationSuiteId.IID,
            PitchEvaluationSuiteId.OOD_LOWER,
            PitchEvaluationSuiteId.OOD_UPPER,
        )
    }
    rows = []
    for artifact, _model, actor, _predictor in retained:
        manifest = artifact.manifest
        summary = read_pitch_training_summary(artifact).summary
        common = {
            "actor_id": f"pitch-{manifest.seed}",
            "trainer": PitchTrainerKind.PITCH,
            "seed": manifest.seed,
            "environment_id": manifest.environment_id,
            "observation_mode": ObservationMode.SPECTRUM,
            "parameter_count": manifest.parameter_count,
            "training_examples": summary.training_examples,
            "training_wall_time_seconds": summary.training_wall_time_seconds,
        }
        seed_rows = []
        for suite_id in (
            PitchEvaluationSuiteId.IID,
            PitchEvaluationSuiteId.OOD_LOWER,
            PitchEvaluationSuiteId.OOD_UPPER,
        ):
            suite = suites[suite_id]
            seed_rows.append(
                build_pitch_evaluation_row(
                    suite=suite,
                    records=evaluate_pitch_learned_actor(
                        actor,
                        suite,
                        environment_factory=environment_factory,
                    ),
                    **common,
                )
            )
        iid_suite = suites[PitchEvaluationSuiteId.IID]
        for probe in (PITCH_ZERO_SPECTRUM_PROBE, PITCH_SHUFFLED_SPECTRUM_PROBE):
            seed_rows.append(
                build_pitch_evaluation_row(
                    suite=iid_suite,
                    records=evaluate_pitch_learned_actor(
                        actor,
                        iid_suite,
                        environment_factory=make_pitch_spectrum_probe_factory(
                            environment_factory,
                            probe,
                        ),
                    ),
                    probe=probe,
                    **common,
                )
            )
        rows.extend(seed_rows)
    for kind in _BASELINE_ORDER:
        for suite_id in (
            PitchEvaluationSuiteId.IID,
            PitchEvaluationSuiteId.OOD_LOWER,
            PitchEvaluationSuiteId.OOD_UPPER,
        ):
            suite = suites[suite_id]
            rows.append(
                build_pitch_evaluation_row(
                    actor_id=kind.value,
                    trainer=None,
                    seed=None,
                    environment_id=kind.environment_id,
                    observation_mode=kind.observation_mode,
                    suite=suite,
                    records=evaluate_pitch_baseline_suite(kind, suite, cache=cache),
                )
            )
    terminal_rows = tuple(rows)
    aggregates = tuple(
        RegisterOODAggregate.from_rows(terminal_rows[start], terminal_rows[start + 1])
        for start in (1, 6, 11, 16, 19, 22, 25)
    )
    criterion = evaluate_pitch_criterion(
        coordinate_evaluations=coordinates,
        terminal_rows=terminal_rows,
        eligible=device is DeviceName.CPU,
    )
    reference = normalized[0].manifest
    return PitchEvaluationReport(
        schema_version=PITCH_ARTIFACT_SCHEMA_VERSION,
        profile=ProfileName.CHECKPOINT,
        evaluation_device=device,
        environment_contract_id=reference.environment_contract_id,
        spectrum_grid_id=reference.spectrum_grid_id,
        preprocessing_schema_id=reference.preprocessing_schema_id,
        architecture_schema_id=reference.architecture_schema_id,
        policy_semantics_id=reference.policy_semantics_id,
        coordinate_distribution_id=reference.train_distribution_id,
        coordinate_split_digest_sha256=reference.coordinate_split_digest_sha256,
        compatibility_sha256=reference.compatibility_sha256,
        terminal_rows=terminal_rows,
        coordinate_evaluations=coordinates,
        register_ood_aggregates=aggregates,
        criterion=criterion,
    )


def run_artifact(
    artifact_path: Path,
    *,
    seed: int,
    device: DeviceName = DeviceName.CPU,
) -> EpisodeTrace:
    """Validate one complete artifact, load its actor, and capture one trace."""

    episode, _artifact = _run_artifact_with_context(artifact_path, seed=seed, device=device)
    return episode


def run_artifact_result(
    artifact_path: Path,
    *,
    seed: int,
    device: DeviceName = DeviceName.CPU,
) -> object:
    """Capture a run with portable model/source provenance and explicit non-eligibility."""
    from harpy.learning.experiment_results import ArtifactRunResult

    episode, artifact = _run_artifact_with_context(artifact_path, seed=seed, device=device)
    return ArtifactRunResult(_artifact_readout_provenance(artifact, device=device), episode)


def _run_artifact_with_context(
    artifact_path: Path,
    *,
    seed: int,
    device: DeviceName,
) -> tuple[EpisodeTrace, object]:
    """Share one strict artifact load and execution between legacy and portable outputs."""

    try:
        normalized_seed = _integer(seed, "seed")
    except ValueError as error:
        raise LearningContractError(str(error)) from error
    if not isinstance(device, DeviceName):
        raise LearningContractError("device must be a DeviceName")
    path = _canonical_artifact_paths((artifact_path,))[0]
    identity = _peek_artifact_identity(path)
    artifact = _load_complete_artifact(path)
    _require_evaluation_device(device)
    schema_version, profile, trainer = identity
    if schema_version == ARTIFACT_SCHEMA_VERSION:
        if not isinstance(artifact, LoadedArtifact):
            raise LearningContractError("schema-v1 dispatch did not load a v1 artifact")
        if artifact.manifest.profile is not profile or artifact.manifest.trainer.value != trainer:
            raise LearningContractError("loaded v1 artifact does not match manifest dispatch")
        try:
            _validate_artifact_payload(artifact)
        except DependencyUnavailableError:
            raise
        except Exception as error:
            raise ArtifactError(f"invalid {trainer} artifact: {error}") from error
        try:
            actor = _load_artifact_actor(artifact, device=device)
        except DependencyUnavailableError:
            raise
        except Exception as error:
            raise ArtifactError(f"could not load {trainer} actor: {error}") from error
    elif schema_version == PITCH_ARTIFACT_SCHEMA_VERSION:
        from harpy.learning.pitch_artifacts import LoadedPitchArtifact

        if not isinstance(artifact, LoadedPitchArtifact):
            raise LearningContractError("schema-v2 dispatch did not load a pitch artifact")
        if artifact.manifest.profile is not profile or artifact.manifest.trainer.value != trainer:
            raise LearningContractError("loaded pitch artifact does not match manifest dispatch")
        try:
            actor = _load_pitch_artifact_actor(artifact, device=device)
        except DependencyUnavailableError:
            raise
        except Exception as error:
            raise ArtifactError(f"could not load pitch actor: {error}") from error
    else:
        raise LearningContractError(f"unsupported artifact schema_version {schema_version}")
    try:
        from harpy.learning import trace as trace_module

        episode = trace_module.trace_episode(actor, seed=normalized_seed)  # type: ignore[arg-type]
        return episode, artifact
    except Exception as error:
        if isinstance(error, LearningExecutionError):
            raise
        raise LearningExecutionError(f"artifact run failed: {error}") from error


def diagnose_artifacts(
    artifact_paths: Sequence[Path],
    *,
    suite: str,
    device: DeviceName = DeviceName.CPU,
    bound_mask: bool = False,
    exploratory: bool = False,
) -> object:
    """Run one historical or pitch diagnostic lane after schema-local preflight."""

    paths, identities = _diagnostic_request_context(
        artifact_paths,
        suite=suite,
        device=device,
        bound_mask=bound_mask,
        exploratory=exploratory,
    )
    if exploratory:
        from harpy.learning.experiment_results import ExploratoryPitchDiagnostics

        artifact = _load_exploratory_pitch_artifact(paths[0], identities[0])
        _require_evaluation_device(device)
        bundle = _diagnose_pitch_artifacts(
            (artifact,), suite=suite, device=device, exploratory=True
        )
        return ExploratoryPitchDiagnostics(
            _artifact_readout_provenance(artifact, device=device), bundle
        )
    schema_version = identities[0][0]
    if schema_version == ARTIFACT_SCHEMA_VERSION:
        return _diagnose_v1_artifact(
            paths[0],
            identity=identities[0],
            suite=suite,
            device=device,
            bound_mask=bound_mask,
        )
    if schema_version == PITCH_ARTIFACT_SCHEMA_VERSION:
        if suite == "smoke":
            artifact = _load_complete_artifact(paths[0])
            from harpy.learning.pitch_artifacts import LoadedPitchArtifact

            if not isinstance(artifact, LoadedPitchArtifact):
                raise LearningContractError("schema-v2 dispatch did not load a pitch artifact")
            if (
                artifact.manifest.profile is not identities[0][1]
                or artifact.manifest.trainer.value != identities[0][2]
            ):
                raise LearningContractError(
                    "loaded pitch artifact does not match manifest dispatch"
                )
            _require_evaluation_device(device)
            return _diagnose_pitch_artifacts((artifact,), suite=suite, device=device)
        training_devices = tuple(_peek_pitch_training_device(path) for path in paths)
        e1_requested = any(item is DeviceName.CUDA for item in training_devices)
        if e1_requested and device is not DeviceName.CPU:
            raise LearningContractError("E.1 final diagnostics must use CPU")
        try:
            admitted = (
                _preflight_pitch_e1_artifacts(paths)
                if e1_requested
                else _preflight_pitch_artifacts(paths)
            )
        except PitchArtifactSetError as error:
            raise LearningContractError(f"invalid pitch checkpoint set: {error}") from error
        except LearningContractError as error:
            raise ArtifactError(f"invalid pitch checkpoint artifact: {error}") from error
        except ValueError as error:
            raise ArtifactError(f"invalid pitch checkpoint artifact: {error}") from error
        _require_evaluation_device(device)
        artifacts = admitted.artifacts if e1_requested else admitted
        return _diagnose_pitch_artifacts(artifacts, suite=suite, device=device)
    raise LearningContractError(f"unsupported artifact schema_version {schema_version}")


def preflight_diagnostic_request(
    artifact_paths: Sequence[Path],
    *,
    suite: str,
    device: DeviceName = DeviceName.CPU,
    bound_mask: bool = False,
    exploratory: bool = False,
) -> None:
    """Validate diagnostic paths and closed-contract routing without loading a model."""

    _diagnostic_request_context(
        artifact_paths,
        suite=suite,
        device=device,
        bound_mask=bound_mask,
        exploratory=exploratory,
    )


def _diagnostic_request_context(
    artifact_paths: Sequence[Path],
    *,
    suite: str,
    device: DeviceName,
    bound_mask: bool,
    exploratory: bool = False,
) -> tuple[tuple[Path, ...], tuple[tuple[int, ProfileName, str], ...]]:
    """Return one dependency-light, schema-local diagnostic dispatch context."""

    if not isinstance(suite, str) or suite not in {
        "smoke",
        "iid",
        "ood-lower",
        "ood-upper",
    }:
        raise LearningContractError("suite must be smoke, iid, ood-lower, or ood-upper")
    if not isinstance(device, DeviceName):
        raise LearningContractError("device must be a DeviceName")
    if not isinstance(bound_mask, bool):
        raise LearningContractError("bound_mask must be a bool")
    if not isinstance(exploratory, bool):
        raise LearningContractError("exploratory must be a bool")
    paths = _canonical_artifact_paths(artifact_paths)
    identities = tuple(_peek_artifact_identity(path) for path in paths)
    if exploratory:
        _validate_exploratory_pitch_request(paths, identities)
        if bound_mask:
            raise LearningContractError("bound_mask requires exactly one schema-v1 BC artifact")
        return paths, identities
    schemas = {identity[0] for identity in identities}
    if len(schemas) != 1:
        raise LearningContractError("mixed artifact schemas are not supported")
    schema_version = next(iter(schemas))
    if schema_version == ARTIFACT_SCHEMA_VERSION:
        if len(paths) != 1:
            raise LearningContractError("legacy diagnostics require exactly one artifact")
        if suite in {"ood-lower", "ood-upper"}:
            raise LearningContractError(
                "schema-v1 diagnostics expose only their historical suite identities"
            )
        suite_id = {
            "smoke": EvaluationSuiteId.SMOKE,
            "iid": EvaluationSuiteId.IID,
        }[suite]
        if suite_id not in PROFILE_CONFIGS[identities[0][1]].evaluation_suites:
            raise LearningContractError("diagnostic suite is not declared by the artifact profile")
        if bound_mask and identities[0][2] != TrainerKind.BC.value:
            raise LearningContractError("bound_mask requires exactly one schema-v1 BC artifact")
        return paths, identities
    if schema_version == PITCH_ARTIFACT_SCHEMA_VERSION:
        if bound_mask:
            raise LearningContractError("bound_mask requires exactly one schema-v1 BC artifact")
        profiles = tuple(identity[1] for identity in identities)
        if suite == "smoke":
            if len(paths) != 1 or profiles != (ProfileName.SMOKE,):
                raise LearningContractError(
                    "pitch smoke diagnostics require exactly one smoke artifact"
                )
            return paths, identities
        if len(paths) != 3 or any(profile is not ProfileName.CHECKPOINT for profile in profiles):
            raise LearningContractError(
                "final pitch diagnostics require exactly three checkpoint artifacts"
            )
        seeds = tuple(_peek_artifact_seed(path) for path in paths)
        if tuple(sorted(seeds)) != (0, 1, 2):
            raise LearningContractError("final pitch diagnostics require exact seeds 0, 1, and 2")
        training_devices = tuple(_peek_pitch_training_device(path) for path in paths)
        if len(set(training_devices)) != 1:
            raise LearningContractError(
                "invalid pitch checkpoint set: E.1 artifacts must share one "
                "homogeneous training device"
            )
        aggregate_identities = tuple(_peek_pitch_aggregate_identity(path) for path in paths)
        if len(set(aggregate_identities)) != 1:
            raise LearningContractError(
                "invalid pitch checkpoint set: pitch aggregate artifacts must share source, "
                "lock, and compatibility"
            )
        if (
            any(item is DeviceName.CUDA for item in training_devices)
            and device is not DeviceName.CPU
        ):
            raise LearningContractError("E.1 final diagnostics must use CPU")
        return paths, identities
    raise LearningContractError(f"unsupported artifact schema_version {schema_version}")


def _diagnose_v1_artifact(
    path: Path,
    *,
    identity: tuple[int, ProfileName, str],
    suite: str,
    device: DeviceName,
    bound_mask: bool,
) -> object:
    """Strict-load and diagnose one declared historical schema-v1 suite."""

    suite_id = {
        "smoke": EvaluationSuiteId.SMOKE,
        "iid": EvaluationSuiteId.IID,
    }[suite]
    profile = identity[1]
    if suite_id not in PROFILE_CONFIGS[profile].evaluation_suites:
        raise LearningContractError("diagnostic suite is not declared by the artifact profile")
    artifact = _load_complete_artifact(path)
    if not isinstance(artifact, LoadedArtifact):
        raise LearningContractError("schema-v1 dispatch did not load a v1 artifact")
    manifest = artifact.manifest
    if manifest.profile is not profile or manifest.trainer.value != identity[2]:
        raise LearningContractError("loaded v1 artifact does not match manifest dispatch")
    _require_evaluation_device(device)
    try:
        _validate_artifact_payload(artifact)
        if bound_mask:
            from harpy.learning.bc import load_masked_bc_actor

            actor = load_masked_bc_actor(artifact, device=device)
        else:
            actor = _load_artifact_actor(artifact, device=device)
    except DependencyUnavailableError:
        raise
    except Exception as error:
        raise ArtifactError(f"could not load diagnostic actor: {error}") from error
    from harpy.learning.actors import (
        BC_ACTOR_SEMANTICS_ID,
        MASKED_BC_ACTOR_SEMANTICS_ID,
        PPO_ACTOR_SEMANTICS_ID,
    )
    from harpy.learning.diagnostics import (
        build_diagnostic_bundle,
        build_diagnostic_report,
    )
    from harpy.learning.evaluation import diagnose_actor_suite

    if manifest.trainer is TrainerKind.BC:
        actor_semantics = MASKED_BC_ACTOR_SEMANTICS_ID if bound_mask else BC_ACTOR_SEMANTICS_ID
    elif manifest.trainer is TrainerKind.PPO:
        actor_semantics = PPO_ACTOR_SEMANTICS_ID
    else:
        raise LearningContractError("schema-v1 artifact has an unsupported trainer")
    fixed_suite = fixed_evaluation_suite(suite_id)
    cache = SpectrumEvidenceCache()
    episodes = diagnose_actor_suite(
        decide=_legacy_diagnostic_decider(actor),
        suite=fixed_suite,
        environment_factory=lambda: make_cached_sine_pitch_env(cache),
    )
    report = build_diagnostic_report(
        seed=manifest.seed,
        artifact_manifest_sha256=_artifact_manifest_sha256(artifact.root),
        actor_semantics=actor_semantics,
        bound_mask=bound_mask,
        suite_id=fixed_suite.suite_id.value,
        suite_digest_sha256=fixed_suite.digest_sha256,
        episodes=episodes,
    )
    return build_diagnostic_bundle(
        device=device.value,
        bound_mask=bound_mask,
        reports=(report,),
    )


def _diagnose_pitch_artifacts(
    artifacts: Sequence[object],
    *,
    suite: str,
    device: DeviceName,
    exploratory: bool = False,
) -> object:
    """Diagnose one smoke artifact or the already-preflighted final triple."""

    from harpy.learning.diagnostics import (
        build_diagnostic_bundle,
        build_diagnostic_report,
    )
    from harpy.learning.evaluation import diagnose_actor_suite
    from harpy.learning.pitch_artifacts import LoadedPitchArtifact
    from harpy.learning.pitch_data import (
        PitchEvaluationSuiteId,
        fixed_pitch_evaluation_suite,
    )

    normalized = tuple(artifacts)
    if not normalized or not all(
        isinstance(artifact, LoadedPitchArtifact) for artifact in normalized
    ):
        raise LearningContractError("pitch diagnostics require loaded pitch artifacts")
    suite_id = {
        "smoke": PitchEvaluationSuiteId.SMOKE,
        "iid": PitchEvaluationSuiteId.IID,
        "ood-lower": PitchEvaluationSuiteId.OOD_LOWER,
        "ood-upper": PitchEvaluationSuiteId.OOD_UPPER,
    }[suite]
    if exploratory:
        if len(normalized) != 1:
            raise LearningContractError("exploratory diagnostics require one pitch artifact")
    elif suite_id is PitchEvaluationSuiteId.SMOKE:
        if len(normalized) != 1 or normalized[0].manifest.profile is not ProfileName.SMOKE:
            raise LearningContractError("pitch smoke diagnostics require one smoke artifact")
    elif (
        len(normalized) != 3
        or tuple(artifact.manifest.seed for artifact in normalized) != (0, 1, 2)
        or any(artifact.manifest.profile is not ProfileName.CHECKPOINT for artifact in normalized)
    ):
        raise LearningContractError("final pitch diagnostics require preflighted seeds 0, 1, and 2")
    fixed_suite = fixed_pitch_evaluation_suite(suite_id)
    cache = SpectrumEvidenceCache()
    reports = []
    for artifact in normalized:
        actor = _load_pitch_artifact_actor(artifact, device=device)
        episodes = diagnose_actor_suite(
            decide=_pitch_diagnostic_decider(actor),
            suite=fixed_suite,
            environment_factory=lambda: make_cached_sine_pitch_env(cache),
        )
        reports.append(
            build_diagnostic_report(
                seed=artifact.manifest.seed,
                artifact_manifest_sha256=_artifact_manifest_sha256(artifact.root),
                actor_semantics=artifact.manifest.policy_semantics_id,
                bound_mask=False,
                suite_id=fixed_suite.suite_id.value,
                suite_digest_sha256=fixed_suite.digest_sha256,
                episodes=episodes,
            )
        )
    if exploratory:
        return reports[0]
    return build_diagnostic_bundle(
        device=device.value,
        bound_mask=False,
        reports=tuple(reports),
    )


def _legacy_diagnostic_decider(actor: object):
    """Adapt exactly one legacy ``act`` call to typed diagnostic input."""

    from harpy.learning.diagnostics import DiagnosticDecisionInput

    act = getattr(actor, "act", None)
    if not callable(act):
        raise LearningContractError("legacy diagnostic actor must provide act")

    def decide(observation: Mapping[str, object]) -> DiagnosticDecisionInput:
        action = act(observation)
        if not isinstance(action, PitchAction):
            raise LearningExecutionError("legacy diagnostic actor must return a PitchAction")
        return DiagnosticDecisionInput(action)

    return decide


def _pitch_diagnostic_decider(actor: object):
    """Adapt exactly one pitch ``decide`` call without also calling ``act``."""

    from harpy.learning.diagnostics import DiagnosticDecisionInput
    from harpy.learning.pitch_actor import PitchDecision

    actor_decide = getattr(actor, "decide", None)
    if not callable(actor_decide):
        raise LearningContractError("pitch diagnostic actor must provide decide")

    def decide(observation: Mapping[str, object]) -> DiagnosticDecisionInput:
        decision = actor_decide(observation)
        if not isinstance(decision, PitchDecision):
            raise LearningExecutionError("pitch diagnostic actor must return PitchDecision")
        return DiagnosticDecisionInput(
            decision.action,
            decision.estimated_candidate_cents,
        )

    return decide


def _artifact_manifest_sha256(root: Path) -> str:
    """Hash the exact persisted manifest bytes through the no-follow reader."""

    return hashlib.sha256(
        _read_regular_file_bytes(root / "manifest.json", "artifact manifest.json")
    ).hexdigest()


def _canonical_artifact_paths(artifact_paths: Sequence[Path]) -> tuple[Path, ...]:
    if isinstance(artifact_paths, (str, bytes, Path)):
        raise LearningContractError("artifact_paths must be a non-empty sequence of paths")
    try:
        supplied = tuple(artifact_paths)
    except TypeError as error:
        raise LearningContractError(
            "artifact_paths must be a non-empty sequence of paths"
        ) from error
    if not supplied:
        raise LearningContractError("artifact_paths must not be empty")
    resolved: list[Path] = []
    for path in supplied:
        if not isinstance(path, Path):
            raise LearningContractError("artifact paths must be pathlib.Path values")
        try:
            canonical = path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise LearningContractError(f"artifact path does not exist: {path}") from error
        if not canonical.is_dir():
            raise LearningContractError(f"artifact path must be a directory: {path}")
        resolved.append(canonical)
    if len(set(resolved)) != len(resolved):
        raise LearningContractError("duplicate artifact path after alias resolution")
    return tuple(resolved)


def _load_complete_artifact(path: Path) -> object:
    try:
        return load_artifact(path)
    except Exception as error:
        raise ArtifactError(f"invalid artifact {path}: {error}") from error


def _compatibility_identity(artifact: LoadedArtifact) -> tuple[object, ...]:
    manifest = artifact.manifest
    return (
        manifest.profile,
        manifest.environment_id,
        manifest.environment_contract_id,
        manifest.train_distribution_id,
        manifest.evaluation_suites,
        manifest.spectrum_grid_id,
        manifest.preprocessing_schema_id,
        manifest.architecture_schema_id,
    )


def _validate_artifact_set(artifacts: Sequence[LoadedArtifact]) -> None:
    identities = tuple(
        (artifact.manifest.trainer, artifact.manifest.seed) for artifact in artifacts
    )
    if len(set(identities)) != len(identities):
        raise LearningContractError("duplicate artifact trainer and seed")
    reference = _compatibility_identity(artifacts[0])
    if any(_compatibility_identity(artifact) != reference for artifact in artifacts[1:]):
        raise LearningContractError("artifacts do not share one compatible contract set")


def _require_evaluation_device(device: DeviceName) -> None:
    stack = require_training_dependencies()
    if device is DeviceName.CUDA and not stack.torch.cuda.is_available():
        raise DependencyUnavailableError("CUDA was requested but is unavailable")


def _validate_artifact_payload(artifact: LoadedArtifact) -> None:
    if artifact.manifest.trainer is TrainerKind.BC:
        from harpy.learning.bc import validate_bc_artifact

        validate_bc_artifact(artifact)
    elif artifact.manifest.trainer is TrainerKind.PPO:
        from harpy.learning.ppo import validate_ppo_artifact

        validate_ppo_artifact(artifact)
    else:
        raise LearningContractError("schema-v1 artifact has an unsupported trainer")


def _load_artifact_actor(
    artifact: LoadedArtifact,
    *,
    device: DeviceName,
) -> object:
    if artifact.manifest.trainer is TrainerKind.BC:
        from harpy.learning.bc import load_bc_actor

        return load_bc_actor(artifact, device=device)
    if artifact.manifest.trainer is TrainerKind.PPO:
        from harpy.learning.ppo import load_ppo_actor

        return load_ppo_actor(artifact, device=device)
    raise LearningContractError("schema-v1 artifact has an unsupported trainer")


def _load_pitch_artifact_actor(artifact: object, *, device: DeviceName) -> object:
    """Reload one persisted pitch model and expose only the actor protocol."""

    from harpy.learning.pitch import load_pitch_estimator_model
    from harpy.learning.pitch_actor import PitchPlannerActor
    from harpy.learning.pitch_artifacts import LoadedPitchArtifact

    if not isinstance(artifact, LoadedPitchArtifact):
        raise LearningContractError("artifact must be a LoadedPitchArtifact")
    stack = require_training_dependencies()
    torch_device = stack.torch.device(device.value)
    model = load_pitch_estimator_model(artifact.file("model.pt"), device=torch_device)
    return PitchPlannerActor(model, device=torch_device)


def _artifact_order_key(artifact: LoadedArtifact) -> tuple[int, int]:
    if artifact.manifest.trainer is TrainerKind.BC:
        return 0, artifact.manifest.seed
    if artifact.manifest.trainer is TrainerKind.PPO:
        return 1, artifact.manifest.seed
    raise LearningContractError("schema-v1 artifact has an unsupported trainer")


def _evaluate_learned_rows(
    actor: object,
    artifact: LoadedArtifact,
) -> tuple[EvaluationRow, ...]:
    manifest = artifact.manifest
    summary_document = read_training_summary(artifact)
    summary = summary_document.summary
    cache = SpectrumEvidenceCache()

    def environment_factory() -> gymnasium.Env:
        return make_cached_sine_pitch_env(cache)

    rows: list[EvaluationRow] = []
    for suite_id in PROFILE_CONFIGS[manifest.profile].evaluation_suites:
        suite = fixed_evaluation_suite(suite_id)
        records = evaluate_learned_actor(
            actor,  # type: ignore[arg-type]
            suite,
            environment_factory=environment_factory,
        )
        common: dict[str, Any] = {
            "actor_id": f"{manifest.trainer.value}-{manifest.seed}",
            "trainer": manifest.trainer,
            "seed": manifest.seed,
            "environment_id": ENVIRONMENT_ID,
            "observation_mode": ObservationMode.SPECTRUM,
            "suite": suite,
            "parameter_count": manifest.parameter_count,
            "training_wall_time_seconds": summary.training_wall_time_seconds,
        }
        if isinstance(summary, BCTrainingSummary):
            common["training_examples"] = summary.training_examples
        elif isinstance(summary, PPOTrainingSummary):
            common["training_environment_steps"] = summary.completed_environment_steps
        else:
            raise ValueError("training summary variant does not match the artifact trainer")
        rows.extend(build_evaluation_rows(records=records, **common))
        if suite_id is not EvaluationSuiteId.REGISTER_OOD:
            for probe in (ZERO_SPECTRUM_PROBE, SHUFFLED_SPECTRUM_PROBE):
                probe_records = evaluate_learned_actor(
                    actor,  # type: ignore[arg-type]
                    suite,
                    environment_factory=make_spectrum_probe_factory(
                        environment_factory,
                        probe,
                    ),
                )
                rows.extend(build_evaluation_rows(records=probe_records, probe=probe, **common))
    return tuple(rows)


def _evaluate_baseline_rows(
    kind: BaselineKind,
    *,
    profile: ProfileName,
) -> tuple[EvaluationRow, ...]:
    cache = SpectrumEvidenceCache()
    rows: list[EvaluationRow] = []
    for suite_id in PROFILE_CONFIGS[profile].evaluation_suites:
        suite = fixed_evaluation_suite(suite_id)
        records = evaluate_baseline_suite(kind, suite, cache=cache)
        rows.extend(
            build_evaluation_rows(
                actor_id=kind.value,
                trainer=None,
                seed=None,
                environment_id=kind.environment_id,
                observation_mode=kind.observation_mode,
                suite=suite,
                records=records,
            )
        )
    return tuple(rows)


def _manifest_is_scientifically_eligible(
    artifact: LoadedArtifact,
    *,
    evaluation_device: DeviceName,
) -> bool:
    manifest = artifact.manifest
    if manifest.trainer is TrainerKind.BC:
        declared_seed = manifest.seed == 0
    elif manifest.trainer is TrainerKind.PPO:
        declared_seed = manifest.seed in range(5)
    else:
        raise LearningContractError("schema-v1 artifact has an unsupported trainer")
    return (
        manifest.criterion_eligible
        and isinstance(manifest.source, SourceStatus)
        and manifest.profile is ProfileName.CHECKPOINT
        and declared_seed
        and manifest.runtime.device is DeviceName.CPU
        and manifest.evaluation_device is DeviceName.CPU
        and not manifest.source.dirty_tree
        and manifest.source.required_inputs_committed
        and evaluation_device is DeviceName.CPU
    )


def _bc_heldout_accuracy(artifact: LoadedArtifact) -> float:
    evaluation = EvaluationFile.from_document(artifact.document("evaluation-iid.json"))
    if evaluation.next_action_accuracy is None:
        raise ValueError("BC IID evaluation is missing held-out next-action accuracy")
    return evaluation.next_action_accuracy


def _evaluate_report_bc_criterion(
    artifacts: Sequence[LoadedArtifact],
    rows: tuple[EvaluationRow, ...],
    *,
    device: DeviceName,
) -> BCScientificCriterion | None:
    if not artifacts:
        return None
    eligible = len(artifacts) == 1 and _manifest_is_scientifically_eligible(
        artifacts[0], evaluation_device=device
    )
    if not eligible:
        return evaluate_bc_criterion(  # type: ignore[arg-type]
            heldout_next_action_accuracy=object(),
            iid_row=object(),
            eligible=False,
        )
    artifact = artifacts[0]
    return evaluate_bc_criterion(
        heldout_next_action_accuracy=_bc_heldout_accuracy(artifact),
        iid_row=_base_iid_row(
            rows,
            trainer=TrainerKind.BC,
            seed=artifact.manifest.seed,
            actor_id=f"bc-{artifact.manifest.seed}",
        ),
        eligible=True,
    )


def _evaluate_report_ppo_criterion(
    artifacts: Sequence[LoadedArtifact],
    bc_artifacts: Sequence[LoadedArtifact],
    rows: tuple[EvaluationRow, ...],
    *,
    device: DeviceName,
) -> PPOScientificCriterion | None:
    if not artifacts:
        return None
    seeds = tuple(artifact.manifest.seed for artifact in artifacts)
    bc_allowed = len(bc_artifacts) <= 1 and (
        not bc_artifacts
        or (
            bc_artifacts[0].manifest.seed == 0
            and _manifest_is_scientifically_eligible(bc_artifacts[0], evaluation_device=device)
        )
    )
    eligible = (
        len(artifacts) == 5
        and set(seeds) == set(range(5))
        and all(
            _manifest_is_scientifically_eligible(
                artifact,
                evaluation_device=device,
            )
            for artifact in artifacts
        )
        and bc_allowed
    )
    if not eligible:
        return evaluate_ppo_criterion(  # type: ignore[arg-type]
            iid_rows=object(),
            random_iid_row=object(),
            eligible=False,
        )
    iid_rows = tuple(
        _base_iid_row(
            rows,
            trainer=TrainerKind.PPO,
            seed=seed,
            actor_id=f"ppo-{seed}",
        )
        for seed in seeds
    )
    random_row = _base_iid_row(
        rows,
        trainer=None,
        seed=None,
        actor_id=BaselineKind.RANDOM.value,
    )
    return evaluate_ppo_criterion(
        iid_rows=iid_rows,
        random_iid_row=random_row,
        eligible=True,
    )


__all__ = [
    "EVALUATION_REPORT_SCHEMA_VERSION",
    "EvaluationReport",
    "diagnose_artifacts",
    "evaluate_artifacts",
    "evaluation_report_bytes",
    "evaluation_report_from_bytes",
    "preflight_diagnostic_request",
    "run_artifact",
    "run_artifact_result",
]
