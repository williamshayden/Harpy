"""Strict cross-trainer evaluation over complete local artifacts."""

from __future__ import annotations

import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium

from harpy.envs.baselines import BaselineKind
from harpy.envs.models import ObservationMode
from harpy.learning.artifacts import (
    LoadedArtifact,
    canonical_json_bytes,
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
)
from harpy.learning.evaluation import (
    EVALUATION_SCHEMA_VERSION,
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
from harpy.learning.models import (
    ARCHITECTURE_SCHEMA_ID,
    ENVIRONMENT_CONTRACT_ID,
    ENVIRONMENT_ID,
    PREPROCESSING_SCHEMA_ID,
    PROFILE_CONFIGS,
    SPECTRUM_GRID_ID,
    BCTrainingSummary,
    DeviceName,
    EvaluationSuiteId,
    JSONValue,
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


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """One deterministic compatible multi-artifact evaluation result."""

    schema_version: int
    profile: ProfileName
    evaluation_device: DeviceName
    environment_contract_id: str
    spectrum_grid_id: str
    preprocessing_schema_id: str
    architecture_schema_id: str
    rows: tuple[EvaluationRow, ...]
    bc_criterion: BCScientificCriterion | None
    ppo_criterion: PPOScientificCriterion | None

    def __post_init__(self) -> None:
        version = _integer(self.schema_version, "schema_version", minimum=1)
        if version != EVALUATION_REPORT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {EVALUATION_REPORT_SCHEMA_VERSION}")
        if not isinstance(self.profile, ProfileName):
            raise ValueError("profile must be a ProfileName")
        if not isinstance(self.evaluation_device, DeviceName):
            raise ValueError("evaluation_device must be a DeviceName")
        identities = (
            (self.environment_contract_id, ENVIRONMENT_CONTRACT_ID, "environment_contract_id"),
            (self.spectrum_grid_id, SPECTRUM_GRID_ID, "spectrum_grid_id"),
            (self.preprocessing_schema_id, PREPROCESSING_SCHEMA_ID, "preprocessing_schema_id"),
            (self.architecture_schema_id, ARCHITECTURE_SCHEMA_ID, "architecture_schema_id"),
        )
        for actual, expected, field in identities:
            if actual != expected:
                raise ValueError(f"{field} must be {expected!r}")
        try:
            rows = tuple(self.rows)
        except TypeError as error:
            raise ValueError("rows must contain EvaluationRow values") from error
        if not rows or not all(isinstance(row, EvaluationRow) for row in rows):
            raise ValueError("rows must contain EvaluationRow values")
        allowed_suites = set(PROFILE_CONFIGS[self.profile].evaluation_suites)
        if any(row.suite_id not in allowed_suites for row in rows):
            raise ValueError("rows must use only the evaluation suites declared by the profile")
        row_identities = tuple(
            (
                row.actor_id,
                row.trainer,
                row.seed,
                row.environment_id,
                row.observation_mode,
                row.suite_id,
                row.subset,
                row.probe,
            )
            for row in rows
        )
        if len(set(row_identities)) != len(row_identities):
            raise ValueError("rows must not contain duplicate identities")
        _validate_canonical_actor_order(rows)
        _validate_complete_row_matrix(rows, profile=self.profile)
        has_bc = any(row.trainer is TrainerKind.BC for row in rows)
        has_ppo = any(row.trainer is TrainerKind.PPO for row in rows)
        if has_bc != (self.bc_criterion is not None):
            raise ValueError("bc_criterion must be present exactly when BC rows are present")
        if has_ppo != (self.ppo_criterion is not None):
            raise ValueError("ppo_criterion must be present exactly when PPO rows are present")
        if self.bc_criterion is not None and not isinstance(
            self.bc_criterion, BCScientificCriterion
        ):
            raise ValueError("bc_criterion must be a BCScientificCriterion or None")
        if self.ppo_criterion is not None and not isinstance(
            self.ppo_criterion, PPOScientificCriterion
        ):
            raise ValueError("ppo_criterion must be a PPOScientificCriterion or None")
        _validate_visible_criteria(
            profile=self.profile,
            device=self.evaluation_device,
            rows=rows,
            bc_criterion=self.bc_criterion,
            ppo_criterion=self.ppo_criterion,
        )
        object.__setattr__(self, "schema_version", version)
        object.__setattr__(self, "rows", rows)

    def to_document(self) -> dict[str, JSONValue]:
        """Return a freshly owned exact schema-v1 report document."""

        return {
            "schema_version": self.schema_version,
            "profile": self.profile.value,
            "evaluation_device": self.evaluation_device.value,
            "environment_contract_id": self.environment_contract_id,
            "spectrum_grid_id": self.spectrum_grid_id,
            "preprocessing_schema_id": self.preprocessing_schema_id,
            "architecture_schema_id": self.architecture_schema_id,
            "rows": _row_documents(self.rows),
            "bc_criterion": _bc_criterion_document(self.bc_criterion),
            "ppo_criterion": _ppo_criterion_document(self.ppo_criterion),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, JSONValue]) -> EvaluationReport:
        """Decode one exact finite report document through Task 5 row contracts."""

        mapping = _mapping(document, "evaluation report")
        _exact_fields(
            mapping,
            {
                "schema_version",
                "profile",
                "evaluation_device",
                "environment_contract_id",
                "spectrum_grid_id",
                "preprocessing_schema_id",
                "architecture_schema_id",
                "rows",
                "bc_criterion",
                "ppo_criterion",
            },
            "evaluation report",
        )
        row_values = mapping["rows"]
        if not isinstance(row_values, list):
            raise ValueError("rows must be a list")
        return cls(
            schema_version=_integer(mapping["schema_version"], "schema_version", minimum=1),
            profile=_enum(ProfileName, mapping["profile"], "profile"),
            evaluation_device=_enum(DeviceName, mapping["evaluation_device"], "evaluation_device"),
            environment_contract_id=_string(
                mapping["environment_contract_id"], "environment_contract_id"
            ),
            spectrum_grid_id=_string(mapping["spectrum_grid_id"], "spectrum_grid_id"),
            preprocessing_schema_id=_string(
                mapping["preprocessing_schema_id"], "preprocessing_schema_id"
            ),
            architecture_schema_id=_string(
                mapping["architecture_schema_id"], "architecture_schema_id"
            ),
            rows=_rows_from_documents(row_values),
            bc_criterion=_bc_criterion_from_document(mapping["bc_criterion"]),
            ppo_criterion=_ppo_criterion_from_document(mapping["ppo_criterion"]),
        )


def evaluation_report_bytes(report: EvaluationReport) -> bytes:
    """Encode a report as one sorted finite JSON object plus one newline."""

    if not isinstance(report, EvaluationReport):
        raise ValueError("report must be an EvaluationReport")
    return canonical_json_bytes(report.to_document())


def evaluation_report_from_bytes(content: bytes) -> EvaluationReport:
    """Decode report bytes with Task 6's strict duplicate-free JSON decoder."""

    return EvaluationReport.from_document(decode_json_bytes(content))


def evaluate_artifacts(
    artifact_paths: Sequence[Path],
    *,
    device: DeviceName = DeviceName.CPU,
) -> EvaluationReport:
    """Validate a compatible complete set, then evaluate every actor and baseline."""

    if not isinstance(device, DeviceName):
        raise LearningContractError("device must be a DeviceName")
    paths = _canonical_artifact_paths(artifact_paths)
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


def run_artifact(
    artifact_path: Path,
    *,
    seed: int,
    device: DeviceName = DeviceName.CPU,
) -> EpisodeTrace:
    """Validate one complete artifact, load its actor, and capture one trace."""

    try:
        normalized_seed = _integer(seed, "seed")
    except ValueError as error:
        raise LearningContractError(str(error)) from error
    if not isinstance(device, DeviceName):
        raise LearningContractError("device must be a DeviceName")
    path = _canonical_artifact_paths((artifact_path,))[0]
    artifact = _load_complete_artifact(path)
    _require_evaluation_device(device)
    try:
        _validate_artifact_payload(artifact)
    except DependencyUnavailableError:
        raise
    except Exception as error:
        raise ArtifactError(
            f"invalid {artifact.manifest.trainer.value} artifact: {error}"
        ) from error
    try:
        actor = _load_artifact_actor(artifact, device=device)
    except DependencyUnavailableError:
        raise
    except Exception as error:
        raise ArtifactError(
            f"could not load {artifact.manifest.trainer.value} actor: {error}"
        ) from error
    try:
        from harpy.learning import trace as trace_module

        return trace_module.trace_episode(actor, seed=normalized_seed)  # type: ignore[arg-type]
    except Exception as error:
        if isinstance(error, LearningExecutionError):
            raise
        raise LearningExecutionError(f"artifact run failed: {error}") from error


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


def _load_complete_artifact(path: Path) -> LoadedArtifact:
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
    else:
        from harpy.learning.ppo import validate_ppo_artifact

        validate_ppo_artifact(artifact)


def _load_artifact_actor(
    artifact: LoadedArtifact,
    *,
    device: DeviceName,
) -> object:
    if artifact.manifest.trainer is TrainerKind.BC:
        from harpy.learning.bc import load_bc_actor

        return load_bc_actor(artifact, device=device)
    from harpy.learning.ppo import load_ppo_actor

    return load_ppo_actor(artifact, device=device)


def _artifact_order_key(artifact: LoadedArtifact) -> tuple[int, int]:
    trainer_position = 0 if artifact.manifest.trainer is TrainerKind.BC else 1
    return trainer_position, artifact.manifest.seed


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
    declared_seed = (
        manifest.seed == 0 if manifest.trainer is TrainerKind.BC else manifest.seed in range(5)
    )
    return (
        manifest.criterion_eligible
        and manifest.profile is ProfileName.CHECKPOINT
        and declared_seed
        and manifest.runtime.device is DeviceName.CPU
        and manifest.evaluation_device is DeviceName.CPU
        and not manifest.source.dirty_tree
        and manifest.source.required_inputs_committed
        and evaluation_device is DeviceName.CPU
    )


def _base_iid_row(
    rows: Sequence[EvaluationRow],
    *,
    trainer: TrainerKind | None,
    seed: int | None,
    actor_id: str,
) -> EvaluationRow:
    matches = tuple(
        row
        for row in rows
        if row.trainer is trainer
        and row.seed == seed
        and row.actor_id == actor_id
        and row.suite_id is EvaluationSuiteId.IID
        and row.subset == "combined"
        and row.probe is None
    )
    if len(matches) != 1:
        raise ValueError("report requires one canonical unperturbed combined IID row")
    return matches[0]


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


def _validate_canonical_actor_order(rows: Sequence[EvaluationRow]) -> None:
    group_identities: list[tuple[str, TrainerKind | None, int | None]] = []
    for row in rows:
        identity = row.actor_id, row.trainer, row.seed
        if not group_identities or group_identities[-1] != identity:
            if identity in group_identities:
                raise ValueError("each actor must occupy one contiguous row block")
            group_identities.append(identity)

    def key(identity: tuple[str, TrainerKind | None, int | None]) -> tuple[int, int]:
        actor_id, trainer, seed = identity
        if trainer is TrainerKind.BC:
            if seed is None:
                raise ValueError("BC actor rows require a seed")
            return 0, seed
        if trainer is TrainerKind.PPO:
            if seed is None:
                raise ValueError("PPO actor rows require a seed")
            return 1, seed
        try:
            return 2 + _BASELINE_POSITION[actor_id], 0
        except KeyError as error:
            raise ValueError("baseline rows must use a declared canonical actor") from error

    keys = tuple(key(identity) for identity in group_identities)
    if keys != tuple(sorted(keys)):
        raise ValueError("rows must be ordered BC seeds, PPO seeds, then canonical baselines")


def _actor_row_blocks(
    rows: Sequence[EvaluationRow],
) -> tuple[tuple[tuple[str, TrainerKind | None, int | None], tuple[EvaluationRow, ...]], ...]:
    blocks: list[tuple[tuple[str, TrainerKind | None, int | None], tuple[EvaluationRow, ...]]] = []
    start = 0
    while start < len(rows):
        identity = rows[start].actor_id, rows[start].trainer, rows[start].seed
        end = start + 1
        while end < len(rows):
            candidate = rows[end].actor_id, rows[end].trainer, rows[end].seed
            if candidate != identity:
                break
            end += 1
        blocks.append((identity, tuple(rows[start:end])))
        start = end
    return tuple(blocks)


def _expected_row_lanes(
    profile: ProfileName,
    *,
    learned: bool,
) -> tuple[tuple[EvaluationSuiteId, str, str | None], ...]:
    lanes: list[tuple[EvaluationSuiteId, str, str | None]] = []
    for suite_id in PROFILE_CONFIGS[profile].evaluation_suites:
        if suite_id is EvaluationSuiteId.REGISTER_OOD:
            lanes.extend((suite_id, subset, None) for subset in ("lower", "upper", "combined"))
            continue
        lanes.append((suite_id, "combined", None))
        if learned:
            lanes.extend(
                (suite_id, "combined", probe)
                for probe in (ZERO_SPECTRUM_PROBE, SHUFFLED_SPECTRUM_PROBE)
            )
    return tuple(lanes)


def _validate_complete_row_matrix(
    rows: Sequence[EvaluationRow],
    *,
    profile: ProfileName,
) -> None:
    blocks = _actor_row_blocks(rows)
    learned_blocks = tuple(block for block in blocks if block[0][1] is not None)
    if not learned_blocks:
        raise ValueError("rows must contain at least one learned actor")
    baseline_blocks = tuple(block for block in blocks if block[0][1] is None)
    baseline_actor_ids = tuple(identity[0] for identity, _ in baseline_blocks)
    expected_baselines = tuple(kind.value for kind in _BASELINE_ORDER)
    if baseline_actor_ids != expected_baselines:
        raise ValueError("rows must contain each canonical baseline exactly once")

    learned_lanes = _expected_row_lanes(profile, learned=True)
    baseline_lanes = _expected_row_lanes(profile, learned=False)
    for (actor_id, trainer, seed), actor_rows in blocks:
        if trainer is not None and actor_id != f"{trainer.value}-{seed}":
            raise ValueError("learned actor_id must match its trainer and seed")
        actual_lanes = tuple((row.suite_id, row.subset, row.probe) for row in actor_rows)
        expected_lanes = learned_lanes if trainer is not None else baseline_lanes
        if actual_lanes != expected_lanes:
            raise ValueError(f"actor {actor_id!r} must contain its exact profile row matrix")


def _visible_actor_seeds(
    rows: Sequence[EvaluationRow],
    trainer: TrainerKind,
) -> tuple[int, ...]:
    return tuple(
        seed
        for (_, candidate, seed), _ in _actor_row_blocks(rows)
        if candidate is trainer and seed is not None
    )


def _validate_visible_criteria(
    *,
    profile: ProfileName,
    device: DeviceName,
    rows: Sequence[EvaluationRow],
    bc_criterion: BCScientificCriterion | None,
    ppo_criterion: PPOScientificCriterion | None,
) -> None:
    if bc_criterion is not None and bc_criterion.eligible:
        if profile is not ProfileName.CHECKPOINT or device is not DeviceName.CPU:
            raise ValueError("eligible BC criteria require checkpoint CPU evaluation")
        if _visible_actor_seeds(rows, TrainerKind.BC) != (0,):
            raise ValueError("eligible BC criteria require exactly BC seed 0")
        heldout_accuracy = bc_criterion.heldout_next_action_accuracy
        if heldout_accuracy is None:
            raise ValueError("eligible BC criteria require held-out accuracy")
        expected_bc = evaluate_bc_criterion(
            heldout_next_action_accuracy=heldout_accuracy,
            iid_row=_base_iid_row(
                rows,
                trainer=TrainerKind.BC,
                seed=0,
                actor_id="bc-0",
            ),
            eligible=True,
        )
        if bc_criterion != expected_bc:
            raise ValueError("bc_criterion must match the retained IID evaluation row")

    if ppo_criterion is not None and ppo_criterion.eligible:
        if profile is not ProfileName.CHECKPOINT or device is not DeviceName.CPU:
            raise ValueError("eligible PPO criteria require checkpoint CPU evaluation")
        ppo_seeds = _visible_actor_seeds(rows, TrainerKind.PPO)
        if ppo_seeds != tuple(range(5)):
            raise ValueError("eligible PPO criteria require exactly PPO seeds 0..4")
        bc_seeds = _visible_actor_seeds(rows, TrainerKind.BC)
        if bc_seeds not in ((), (0,)) or (
            bc_seeds == (0,) and (bc_criterion is None or not bc_criterion.eligible)
        ):
            raise ValueError("eligible PPO criteria allow only an eligible BC seed 0")
        expected_ppo = evaluate_ppo_criterion(
            iid_rows=tuple(
                _base_iid_row(
                    rows,
                    trainer=TrainerKind.PPO,
                    seed=seed,
                    actor_id=f"ppo-{seed}",
                )
                for seed in range(5)
            ),
            random_iid_row=_base_iid_row(
                rows,
                trainer=None,
                seed=None,
                actor_id=BaselineKind.RANDOM.value,
            ),
            eligible=True,
        )
        if ppo_criterion != expected_ppo:
            raise ValueError("ppo_criterion must match the retained paired IID rows")


def _row_documents(rows: Sequence[EvaluationRow]) -> list[JSONValue]:
    documents: list[JSONValue] = []
    index = 0
    while index < len(rows):
        row = rows[index]
        if row.suite_id is EvaluationSuiteId.REGISTER_OOD:
            group = tuple(rows[index : index + 3])
            if len(group) != 3:
                raise ValueError("register-OOD rows must form complete three-row groups")
            index += 3
        else:
            group = (row,)
            index += 1
        evaluation = EvaluationFile(
            schema_version=EVALUATION_SCHEMA_VERSION,
            suite_id=row.suite_id,
            suite_digest_sha256=row.suite_digest_sha256,
            rows=group,
            next_action_accuracy=None,
        )
        row_documents = evaluation.to_document()["rows"]
        if not isinstance(row_documents, list):
            raise RuntimeError("Task 5 evaluation codec returned a malformed row list")
        documents.extend(row_documents)
    return documents


def _rows_from_documents(values: Sequence[object]) -> tuple[EvaluationRow, ...]:
    rows: list[EvaluationRow] = []
    index = 0
    while index < len(values):
        mapping = _mapping(values[index], "evaluation row")
        suite_id = _enum(EvaluationSuiteId, mapping.get("suite_id"), "suite_id")
        digest = _string(mapping.get("suite_digest_sha256"), "suite_digest_sha256")
        if suite_id is EvaluationSuiteId.REGISTER_OOD:
            group = list(values[index : index + 3])
            if len(group) != 3:
                raise ValueError("register-OOD rows must form complete three-row groups")
            index += 3
        else:
            group = [values[index]]
            index += 1
        evaluation = EvaluationFile.from_document(
            {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "suite_id": suite_id.value,
                "suite_digest_sha256": digest,
                "rows": group,  # type: ignore[dict-item]
                "next_action_accuracy": None,
            }
        )
        rows.extend(evaluation.rows)
    return tuple(rows)


def _bc_criterion_document(
    criterion: BCScientificCriterion | None,
) -> dict[str, JSONValue] | None:
    if criterion is None:
        return None
    return {
        "eligible": criterion.eligible,
        "heldout_next_action_accuracy": criterion.heldout_next_action_accuracy,
        "iid_submitted_success_rate": criterion.iid_submitted_success_rate,
        "criterion_met": criterion.criterion_met,
        "status": criterion.status,
    }


def _ppo_criterion_document(
    criterion: PPOScientificCriterion | None,
) -> dict[str, JSONValue] | None:
    if criterion is None:
        return None
    return {
        "eligible": criterion.eligible,
        "median_iid_submitted_success_rate": criterion.median_iid_submitted_success_rate,
        "seeds_strictly_beating_random": criterion.seeds_strictly_beating_random,
        "criterion_met": criterion.criterion_met,
        "status": criterion.status,
    }


def _bc_criterion_from_document(value: object) -> BCScientificCriterion | None:
    if value is None:
        return None
    mapping = _mapping(value, "bc_criterion")
    _exact_fields(
        mapping,
        {
            "eligible",
            "heldout_next_action_accuracy",
            "iid_submitted_success_rate",
            "criterion_met",
            "status",
        },
        "bc_criterion",
    )
    return BCScientificCriterion(
        eligible=_boolean(mapping["eligible"], "eligible"),
        heldout_next_action_accuracy=(
            None
            if mapping["heldout_next_action_accuracy"] is None
            else _document_float(
                mapping["heldout_next_action_accuracy"],
                "heldout_next_action_accuracy",
                minimum=0.0,
                maximum=1.0,
            )
        ),
        iid_submitted_success_rate=(
            None
            if mapping["iid_submitted_success_rate"] is None
            else _document_float(
                mapping["iid_submitted_success_rate"],
                "iid_submitted_success_rate",
                minimum=0.0,
                maximum=1.0,
            )
        ),
        criterion_met=(
            None
            if mapping["criterion_met"] is None
            else _boolean(mapping["criterion_met"], "criterion_met")
        ),
        status=_string(mapping["status"], "status"),
    )


def _ppo_criterion_from_document(value: object) -> PPOScientificCriterion | None:
    if value is None:
        return None
    mapping = _mapping(value, "ppo_criterion")
    _exact_fields(
        mapping,
        {
            "eligible",
            "median_iid_submitted_success_rate",
            "seeds_strictly_beating_random",
            "criterion_met",
            "status",
        },
        "ppo_criterion",
    )
    return PPOScientificCriterion(
        eligible=_boolean(mapping["eligible"], "eligible"),
        median_iid_submitted_success_rate=(
            None
            if mapping["median_iid_submitted_success_rate"] is None
            else _document_float(
                mapping["median_iid_submitted_success_rate"],
                "median_iid_submitted_success_rate",
                minimum=0.0,
                maximum=1.0,
            )
        ),
        seeds_strictly_beating_random=(
            None
            if mapping["seeds_strictly_beating_random"] is None
            else _integer(
                mapping["seeds_strictly_beating_random"],
                "seeds_strictly_beating_random",
            )
        ),
        criterion_met=(
            None
            if mapping["criterion_met"] is None
            else _boolean(mapping["criterion_met"], "criterion_met")
        ),
        status=_string(mapping["status"], "status"),
    )


__all__ = [
    "EVALUATION_REPORT_SCHEMA_VERSION",
    "EvaluationReport",
    "evaluate_artifacts",
    "evaluation_report_bytes",
    "evaluation_report_from_bytes",
    "run_artifact",
]
