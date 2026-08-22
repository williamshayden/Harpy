"""Compact oracle data and deterministic behavior-cloning workflows."""

from __future__ import annotations

import math
import operator
import platform as platform_module
import random
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import gymnasium
import numpy as np

from harpy.envs.models import (
    MAX_STEPS,
    TARGET_MIN_COORDINATE,
    ControlState,
    ObservationMode,
    PitchAction,
)
from harpy.envs.planning import minimum_action_plan
from harpy.learning.actors import BCActor
from harpy.learning.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactCompletion,
    ArtifactManifest,
    ArtifactStatus,
    ArtifactWriter,
    BCTrainingCounts,
    CriterionStatus,
    LoadedArtifact,
    PendingArtifactView,
    RuntimeStatus,
    SourceStatus,
    TrainingConfigDocument,
    TrainingSummaryDocument,
    capture_source_status,
    read_training_config,
    read_training_summary,
    required_payload_names,
)
from harpy.learning.cache import SpectrumEvidenceCache
from harpy.learning.dependencies import require_training_dependencies
from harpy.learning.envs import SpectrumEvidenceProvider, make_cached_sine_pitch_env
from harpy.learning.evaluation import (
    EVALUATION_SCHEMA_VERSION,
    SHUFFLED_SPECTRUM_PROBE,
    ZERO_SPECTRUM_PROBE,
    EvaluationFile,
    EvaluationRow,
    build_evaluation_rows,
    evaluate_bc_criterion,
    evaluate_learned_actor,
    make_spectrum_probe_factory,
)
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
    EpisodeSpec,
    EpisodeSuite,
    EvaluationSuiteId,
    ProfileName,
    TrainerKind,
)
from harpy.learning.network import BCPolicyNetwork
from harpy.learning.observations import PolicyObservation, preprocess_observation
from harpy.learning.suites import (
    TRAIN_DISTRIBUTION_ID,
    build_bc_episode_splits,
    fixed_evaluation_suite,
)

_training_stack = require_training_dependencies()
torch = _training_stack.torch

_BC_STATE_SCHEMA: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("encoder.spectrum_encoder.0.weight", (16, 1, 9)),
    ("encoder.spectrum_encoder.0.bias", (16,)),
    ("encoder.spectrum_encoder.2.weight", (32, 16, 7)),
    ("encoder.spectrum_encoder.2.bias", (32,)),
    ("encoder.state_encoder.0.weight", (32, 5)),
    ("encoder.state_encoder.0.bias", (32,)),
    ("encoder.state_encoder.2.weight", (32, 32)),
    ("encoder.state_encoder.2.bias", (32,)),
    ("encoder.combined_encoder.0.weight", (128, 544)),
    ("encoder.combined_encoder.0.bias", (128,)),
    ("action_head.weight", (7, 128)),
    ("action_head.bias", (7,)),
)
_BC_PARAMETER_COUNT = sum(math.prod(shape) for _, shape in _BC_STATE_SCHEMA)
_BC_CORE_PAYLOADS = ("training-config.json", "training-summary.json", "model.pt")
_SEED_ZERO_SPLIT_DIGESTS = {
    ProfileName.SMOKE: (
        "ac43951b718486402500191dcbee7b41ef7844dfe16fbd8fb727611cba6a0833",
        "d52d911cde6b941ff329f0531ae168d2867ef06739f7ce93a4cc9439b55414f0",
    ),
    ProfileName.CHECKPOINT: (
        "f9664dc49d64b9e87e9bf5b0a0e823d7c2e78243b9c380bcbc3e10ecf5bd7a26",
        "3904604026b8738cc27801505340cc6517f0dd452cdcf8965d1926cf8f9e964f",
    ),
}


@dataclass(frozen=True, slots=True)
class BCExample:
    """One compact pre-action oracle label without materialized spectrum evidence."""

    episode: EpisodeSpec
    controls: ControlState
    steps_remaining: int
    action: PitchAction

    def __post_init__(self) -> None:
        if not isinstance(self.episode, EpisodeSpec):
            raise ValueError("episode must be an EpisodeSpec")
        if not isinstance(self.controls, ControlState):
            raise ValueError("controls must be a ControlState")
        if isinstance(self.steps_remaining, bool):
            raise ValueError("steps_remaining must be an integer within 1..64")
        try:
            steps_remaining = operator.index(self.steps_remaining)
        except TypeError as error:
            raise ValueError("steps_remaining must be an integer within 1..64") from error
        if not 1 <= steps_remaining <= MAX_STEPS:
            raise ValueError(f"steps_remaining must be an integer within 1..{MAX_STEPS}")
        if not isinstance(self.action, PitchAction):
            raise ValueError("action must be a PitchAction")
        object.__setattr__(self, "steps_remaining", steps_remaining)


def build_oracle_examples(
    episodes: Sequence[EpisodeSpec],
    *,
    env_factory: Callable[[], gymnasium.Env],
) -> tuple[BCExample, ...]:
    """Step exact minimum plans through one real environment and retain compact labels."""

    normalized = _episode_sequence(episodes)
    if not callable(env_factory):
        raise ValueError("env_factory must be callable")
    if not normalized:
        return ()

    env = env_factory()
    if not all(callable(getattr(env, method, None)) for method in ("reset", "step", "close")):
        raise ValueError("env_factory must return a Gym-compatible environment")
    examples: list[BCExample] = []
    try:
        for episode in normalized:
            observation, _ = env.reset(options=episode.reset_options())
            controls = ControlState()
            plan = minimum_action_plan(
                episode.source_pitch_cents
                - 100 * (TARGET_MIN_COORDINATE + episode.target_note_index),
                controls,
                tolerance_cents=5,
            )
            for step_index, action in enumerate(plan):
                steps_remaining = MAX_STEPS - step_index
                _require_trajectory_observation(
                    observation,
                    episode=episode,
                    controls=controls,
                    steps_remaining=steps_remaining,
                )
                examples.append(
                    BCExample(
                        episode=episode,
                        controls=controls,
                        steps_remaining=steps_remaining,
                        action=action,
                    )
                )
                observation, _, terminated, truncated, info = env.step(action)
                final_action = step_index == len(plan) - 1
                if final_action:
                    if (
                        action is not PitchAction.SUBMIT
                        or not terminated
                        or truncated
                        or not isinstance(info, Mapping)
                        or info.get("submitted") is not True
                        or info.get("submitted_success") is not True
                    ):
                        raise RuntimeError(
                            "planner/environment trajectory mismatch at terminal action"
                        )
                else:
                    if terminated or truncated or action is PitchAction.SUBMIT:
                        raise RuntimeError(
                            "planner/environment trajectory mismatch before terminal action"
                        )
                    controls, applied = controls.apply(action)
                    if not applied:
                        raise RuntimeError(
                            "planner/environment trajectory mismatch at control action"
                        )
    finally:
        env.close()
    return tuple(examples)


class OracleTrajectoryDataset(torch.utils.data.Dataset):
    """Lazily reconstruct actor-visible spectra for compact oracle examples."""

    def __init__(
        self,
        examples: Sequence[BCExample],
        evidence_provider: SpectrumEvidenceProvider,
    ) -> None:
        try:
            normalized = tuple(examples)
        except TypeError as error:
            raise ValueError("examples must contain BCExample records") from error
        if not normalized:
            raise ValueError("examples must not be empty")
        if not all(isinstance(example, BCExample) for example in normalized):
            raise ValueError("examples must contain only BCExample records")
        if not callable(getattr(evidence_provider, "spectrum_for_cents", None)):
            raise ValueError("evidence_provider must provide spectrum_for_cents")
        self._examples = normalized
        self._evidence_provider = evidence_provider

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> tuple[PolicyObservation, np.int64]:
        normalized_index = _dataset_index(index, len(self._examples))
        example = self._examples[normalized_index]
        effective_pitch_cents = example.episode.source_pitch_cents + example.controls.offset_cents
        raw_observation: dict[str, object] = {
            "spectrum": self._evidence_provider.spectrum_for_cents(effective_pitch_cents),
            "target_note": np.int64(example.episode.target_note_index),
            "controls": np.array(
                [
                    example.controls.octaves,
                    example.controls.semitones,
                    example.controls.cents,
                ],
                dtype=np.int16,
            ),
            "steps_remaining": np.int64(example.steps_remaining),
        }
        return preprocess_observation(raw_observation), np.int64(example.action)


def training_class_weights(actions: Sequence[PitchAction]) -> torch.Tensor:
    """Return inverse-frequency seven-class weights from training labels only."""

    try:
        normalized = tuple(actions)
    except TypeError as error:
        raise ValueError("actions must contain PitchAction values") from error
    if not normalized:
        raise ValueError("actions must not be empty")
    if not all(isinstance(action, PitchAction) for action in normalized):
        raise ValueError("actions must contain only PitchAction values")
    counts = Counter(normalized)
    missing = tuple(action for action in PitchAction if counts[action] == 0)
    if missing:
        raise ValueError("every action class must be present in training actions")
    total = len(normalized)
    weights = torch.tensor(
        [total / (len(PitchAction) * counts[action]) for action in PitchAction],
        dtype=torch.float32,
        device="cpu",
    )
    if not torch.isfinite(weights).all():
        raise ValueError("training class weights must be finite")
    return weights


def train_behavior_cloning(
    training_dataset: OracleTrajectoryDataset,
    validation_dataset: OracleTrajectoryDataset,
    *,
    config: BCProfile,
    seed: int,
    device: torch.device,
) -> tuple[BCPolicyNetwork, BCTrainingSummary]:
    """Train one deterministic BC classifier and restore its strict-lowest checkpoint."""

    if not isinstance(training_dataset, OracleTrajectoryDataset):
        raise ValueError("training_dataset must be an OracleTrajectoryDataset")
    if not isinstance(validation_dataset, OracleTrajectoryDataset):
        raise ValueError("validation_dataset must be an OracleTrajectoryDataset")
    if not isinstance(config, BCProfile):
        raise ValueError("config must be a BCProfile")
    normalized_seed = _nonnegative_integer(seed, "seed")
    normalized_device = _torch_device(device)
    _seed_training(normalized_seed, normalized_device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(normalized_seed % (2**63 - 1))
    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    class_weights = training_class_weights(
        tuple(example.action for example in training_dataset._examples)
    ).to(normalized_device)
    model = BCPolicyNetwork().to(normalized_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    started = time.perf_counter()
    history: list[BCEpochMetrics] = []
    selected_epoch = 0
    selected_loss = math.inf
    selected_state: dict[str, torch.Tensor] | None = None
    consecutive_non_improvements = 0
    for epoch in range(1, config.max_epochs + 1):
        training_loss = _train_epoch(
            model,
            training_loader,
            optimizer,
            class_weights,
            normalized_device,
        )
        validation_loss, validation_accuracy = _evaluate_epoch(
            model,
            validation_loader,
            class_weights,
            normalized_device,
        )
        metrics = BCEpochMetrics(
            epoch=epoch,
            training_loss=training_loss,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
        )
        history.append(metrics)
        if validation_loss < selected_loss:
            selected_loss = validation_loss
            selected_epoch = epoch
            selected_state = _cpu_state_dict(model)
            consecutive_non_improvements = 0
        else:
            consecutive_non_improvements += 1
        if (
            config.early_stopping_patience is not None
            and consecutive_non_improvements >= config.early_stopping_patience
        ):
            break

    if selected_state is None or selected_epoch == 0:
        raise RuntimeError("behavior-cloning training produced no selectable checkpoint")
    model.load_state_dict(selected_state, strict=True)
    model.to(normalized_device)
    summary = BCTrainingSummary(
        history=tuple(history),
        selected_epoch=selected_epoch,
        training_examples=len(training_dataset),
        validation_examples=len(validation_dataset),
        training_wall_time_seconds=time.perf_counter() - started,
    )
    return model, summary


@torch.inference_mode()
def next_action_accuracy(
    model: BCPolicyNetwork,
    dataset: OracleTrajectoryDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    """Return exact ordered next-action accuracy over a complete dataset."""

    if not isinstance(model, torch.nn.Module):
        raise ValueError("model must be a torch.nn.Module")
    if not isinstance(dataset, OracleTrajectoryDataset):
        raise ValueError("dataset must be an OracleTrajectoryDataset")
    normalized_batch_size = _positive_integer(batch_size, "batch_size")
    normalized_device = _torch_device(device)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=normalized_batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.to(normalized_device)
    model.eval()
    correct = 0
    examples = 0
    for observations, labels in loader:
        tensors, targets = _batch_to_device(observations, labels, normalized_device)
        logits = _validated_logits(model(tensors), targets.shape[0])
        correct += int((torch.argmax(logits, dim=1) == targets).sum().item())
        examples += targets.shape[0]
    if examples != len(dataset):
        raise RuntimeError("accuracy loader did not consume the complete dataset")
    accuracy = correct / examples
    if not math.isfinite(accuracy):
        raise RuntimeError("next-action accuracy must be finite")
    return accuracy


def save_bc_model(path: Path, model: BCPolicyNetwork) -> None:
    """Persist only the exact finite BC state dictionary."""

    if not isinstance(path, Path):
        raise ValueError("path must be a pathlib.Path")
    if not isinstance(model, BCPolicyNetwork):
        raise ValueError("model must be a BCPolicyNetwork")
    state = _cpu_state_dict(model)
    _validate_bc_state_dict(state, expected_parameter_count=_BC_PARAMETER_COUNT)
    torch.save(state, path)


def validate_bc_artifact(artifact: LoadedArtifact | PendingArtifactView) -> None:
    """Strictly validate one BC artifact view and every payload declared by that view."""

    if not isinstance(artifact, (LoadedArtifact, PendingArtifactView)):
        raise ValueError("artifact must be a LoadedArtifact or PendingArtifactView")
    manifest = artifact.manifest
    if manifest.trainer is not TrainerKind.BC:
        raise ValueError("BC artifact must declare trainer bc")

    if isinstance(artifact, PendingArtifactView):
        declared_names = tuple(record.relative_path for record in artifact.files)
        full_names = required_payload_names(TrainerKind.BC, manifest.profile)
        if declared_names not in {_BC_CORE_PAYLOADS, full_names}:
            raise ValueError(
                "pending BC inventory must be exact config/summary/model or the full inventory"
            )
    else:
        declared_names = tuple(record.relative_path for record in manifest.files)
        full_names = required_payload_names(TrainerKind.BC, manifest.profile)
        if declared_names != full_names:
            raise ValueError("complete BC artifact must declare the full exact inventory")

    config = read_training_config(artifact)
    summary_document = read_training_summary(artifact)
    if config.trainer is not TrainerKind.BC or summary_document.trainer is not TrainerKind.BC:
        raise ValueError("BC config and summary must declare trainer bc")
    if not isinstance(config.profile_config, BCProfile):
        raise ValueError("BC config must contain a BCProfile")
    if not isinstance(summary_document.summary, BCTrainingSummary):
        raise ValueError("BC summary must contain a BCTrainingSummary")
    summary = summary_document.summary

    splits = build_bc_episode_splits(profile=config.profile, run_seed=config.seed)
    actual_digests = (
        splits.training_digest_sha256,
        splits.validation_digest_sha256,
    )
    if config.seed == 0 and actual_digests != _SEED_ZERO_SPLIT_DIGESTS[config.profile]:
        raise ValueError("declared seed-zero BC split digests no longer match the golden contract")
    if actual_digests != (
        config.bc_training_digest_sha256,
        config.bc_validation_digest_sha256,
    ):
        raise ValueError("BC split digests must match the profile and seed")
    if summary.training_examples < config.profile_config.train_episodes:
        raise ValueError("training examples must cover every configured training episode")
    if summary.validation_examples < config.profile_config.validation_episodes:
        raise ValueError("validation examples must cover every configured validation episode")
    if manifest.parameter_count != _BC_PARAMETER_COUNT:
        raise ValueError("artifact parameter_count must match the BC architecture")
    if manifest.architecture_schema_id != ARCHITECTURE_SCHEMA_ID:
        raise ValueError("artifact architecture schema must match the BC architecture")
    if manifest.preprocessing_schema_id != PREPROCESSING_SCHEMA_ID:
        raise ValueError("artifact preprocessing schema must match the BC actor input")

    state = _load_bc_state_dict(artifact.file("model.pt"))
    _validate_bc_state_dict(state, expected_parameter_count=manifest.parameter_count)
    for name in declared_names:
        if not name.startswith("evaluation-"):
            continue
        evaluation = EvaluationFile.from_document(artifact.document(name))
        _validate_evaluation_payload(
            name,
            evaluation,
            profile=manifest.profile,
            seed=manifest.seed,
            parameter_count=manifest.parameter_count,
            summary=summary,
        )


def load_bc_actor(
    artifact: LoadedArtifact | PendingArtifactView,
    *,
    device: DeviceName = DeviceName.CPU,
) -> BCActor:
    """Validate and safely reload a persisted BC model into the narrow actor adapter."""

    if not isinstance(device, DeviceName):
        raise ValueError("device must be a DeviceName")
    if device is DeviceName.CUDA and not torch.cuda.is_available():
        raise ValueError("CUDA actor loading requested but CUDA is unavailable")
    validate_bc_artifact(artifact)
    state = _load_bc_state_dict(artifact.file("model.pt"))
    _validate_bc_state_dict(state, expected_parameter_count=artifact.manifest.parameter_count)
    model = BCPolicyNetwork()
    model.load_state_dict(state, strict=True)
    return BCActor(model, device=torch.device(device.value))


def train_bc_artifact(
    *,
    profile: ProfileName,
    seed: int,
    device: DeviceName,
    output: Path,
) -> LoadedArtifact:
    """Train, evaluate, validate, and atomically complete one BC artifact."""

    normalized_seed = _validate_artifact_inputs(profile, seed, device, output)
    source = capture_source_status(Path(__file__))
    runtime = _capture_runtime_status(device)
    profile_config = PROFILE_CONFIGS[profile]
    suites = tuple(
        fixed_evaluation_suite(suite_id) for suite_id in profile_config.evaluation_suites
    )
    bootstrap = ArtifactManifest(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        status=ArtifactStatus.INCOMPLETE,
        trainer=TrainerKind.BC,
        profile=profile,
        seed=normalized_seed,
        created_at_utc=_utc_now(),
        completed_at_utc=None,
        source=source,
        runtime=runtime,
        environment_id=ENVIRONMENT_ID,
        environment_contract_id=ENVIRONMENT_CONTRACT_ID,
        train_distribution_id=TRAIN_DISTRIBUTION_ID,
        evaluation_suites=tuple((suite.suite_id.value, suite.digest_sha256) for suite in suites),
        spectrum_grid_id=SPECTRUM_GRID_ID,
        preprocessing_schema_id=PREPROCESSING_SCHEMA_ID,
        architecture_schema_id=ARCHITECTURE_SCHEMA_ID,
        parameter_count=_BC_PARAMETER_COUNT,
        training_counts=BCTrainingCounts(
            configured_training_episodes=profile_config.bc.train_episodes,
            configured_validation_episodes=profile_config.bc.validation_episodes,
            training_examples=None,
            validation_examples=None,
        ),
        evaluation_device=None,
        criterion_eligible=False,
        criterion_status=CriterionStatus.INELIGIBLE,
        criterion_met=None,
        files=(),
    )
    writer = ArtifactWriter.begin(output, bootstrap)

    splits = build_bc_episode_splits(profile=profile, run_seed=normalized_seed)
    _verify_split_digests(profile, normalized_seed, splits)
    cache = SpectrumEvidenceCache()

    def environment_factory() -> gymnasium.Env:
        return make_cached_sine_pitch_env(cache)

    training_examples = build_oracle_examples(
        splits.training,
        env_factory=environment_factory,
    )
    validation_examples = build_oracle_examples(
        splits.validation,
        env_factory=environment_factory,
    )
    evidence_provider = SpectrumEvidenceProvider(cache)
    training_dataset = OracleTrajectoryDataset(training_examples, evidence_provider)
    validation_dataset = OracleTrajectoryDataset(validation_examples, evidence_provider)
    model, summary = train_behavior_cloning(
        training_dataset,
        validation_dataset,
        config=profile_config.bc,
        seed=normalized_seed,
        device=torch.device(device.value),
    )

    config_document = TrainingConfigDocument(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        trainer=TrainerKind.BC,
        profile=profile,
        seed=normalized_seed,
        device=device,
        environment_id=ENVIRONMENT_ID,
        environment_contract_id=ENVIRONMENT_CONTRACT_ID,
        train_distribution_id=TRAIN_DISTRIBUTION_ID,
        evaluation_suites=tuple((suite.suite_id, suite.digest_sha256) for suite in suites),
        spectrum_grid_id=SPECTRUM_GRID_ID,
        preprocessing_schema_id=PREPROCESSING_SCHEMA_ID,
        architecture_schema_id=ARCHITECTURE_SCHEMA_ID,
        profile_config=profile_config.bc,
        bc_training_digest_sha256=splits.training_digest_sha256,
        bc_validation_digest_sha256=splits.validation_digest_sha256,
    )
    summary_document = TrainingSummaryDocument(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        trainer=TrainerKind.BC,
        profile=profile,
        seed=normalized_seed,
        summary=summary,
    )
    writer.publish_json("training-config.json", config_document.to_document())
    writer.publish_json("training-summary.json", summary_document.to_document())
    writer.publish_model("model.pt", lambda path: save_bc_model(path, model))

    core_view = writer.pending_view(_BC_CORE_PAYLOADS)
    actor = load_bc_actor(core_view, device=DeviceName.CPU)
    heldout_accuracy = None
    if profile is ProfileName.CHECKPOINT:
        iid_suite = next(suite for suite in suites if suite.suite_id is EvaluationSuiteId.IID)
        heldout_iid_examples = build_oracle_examples(
            iid_suite.episodes,
            env_factory=environment_factory,
        )
        heldout_iid_dataset = OracleTrajectoryDataset(
            heldout_iid_examples,
            evidence_provider,
        )
        heldout_accuracy = next_action_accuracy(
            model,
            heldout_iid_dataset,
            batch_size=profile_config.bc.batch_size,
            device=torch.device("cpu"),
        )

    evaluations = tuple(
        _evaluate_bc_suite(
            actor,
            suite,
            cache=cache,
            seed=normalized_seed,
            summary=summary,
            next_accuracy=(heldout_accuracy if suite.suite_id is EvaluationSuiteId.IID else None),
        )
        for suite in suites
    )
    criterion_row = next(
        evaluation.rows[0]
        for evaluation in evaluations
        if evaluation.suite_id in {EvaluationSuiteId.SMOKE, EvaluationSuiteId.IID}
    )
    eligible = _bc_workflow_eligible(
        profile=profile,
        seed=normalized_seed,
        source=source,
        training_device=device,
        evaluation_device=DeviceName.CPU,
    )
    criterion = evaluate_bc_criterion(
        heldout_next_action_accuracy=(0.0 if heldout_accuracy is None else heldout_accuracy),
        iid_row=criterion_row,
        eligible=eligible,
    )

    for evaluation in evaluations:
        writer.publish_json(
            _evaluation_filename(evaluation.suite_id),
            evaluation.to_document(),
        )
    full_view = writer.pending_view(required_payload_names(TrainerKind.BC, profile))
    validate_bc_artifact(full_view)
    return writer.complete(
        ArtifactCompletion(
            completed_at_utc=_utc_now(),
            training_counts=BCTrainingCounts(
                configured_training_episodes=profile_config.bc.train_episodes,
                configured_validation_episodes=profile_config.bc.validation_episodes,
                training_examples=len(training_dataset),
                validation_examples=len(validation_dataset),
            ),
            evaluation_device=DeviceName.CPU,
            bc_criterion_met=criterion.criterion_met,
        )
    )


def _evaluate_bc_suite(
    actor: BCActor,
    suite: EpisodeSuite,
    *,
    cache: SpectrumEvidenceCache,
    seed: int,
    summary: BCTrainingSummary,
    next_accuracy: float | None,
) -> EvaluationFile:
    def base_factory() -> gymnasium.Env:
        return make_cached_sine_pitch_env(cache)

    records = evaluate_learned_actor(actor, suite, environment_factory=base_factory)
    rows: list[EvaluationRow] = list(
        build_evaluation_rows(
            actor_id=f"bc-{seed}",
            trainer=TrainerKind.BC,
            seed=seed,
            environment_id=ENVIRONMENT_ID,
            observation_mode=ObservationMode.SPECTRUM,
            suite=suite,
            records=records,
            parameter_count=_BC_PARAMETER_COUNT,
            training_examples=summary.training_examples,
            training_wall_time_seconds=summary.training_wall_time_seconds,
        )
    )
    if suite.suite_id is not EvaluationSuiteId.REGISTER_OOD:
        for probe in (ZERO_SPECTRUM_PROBE, SHUFFLED_SPECTRUM_PROBE):
            probe_records = evaluate_learned_actor(
                actor,
                suite,
                environment_factory=make_spectrum_probe_factory(base_factory, probe),
            )
            rows.extend(
                build_evaluation_rows(
                    actor_id=f"bc-{seed}",
                    trainer=TrainerKind.BC,
                    seed=seed,
                    environment_id=ENVIRONMENT_ID,
                    observation_mode=ObservationMode.SPECTRUM,
                    suite=suite,
                    records=probe_records,
                    probe=probe,
                    parameter_count=_BC_PARAMETER_COUNT,
                    training_examples=summary.training_examples,
                    training_wall_time_seconds=summary.training_wall_time_seconds,
                )
            )
    return EvaluationFile(
        schema_version=EVALUATION_SCHEMA_VERSION,
        suite_id=suite.suite_id,
        suite_digest_sha256=suite.digest_sha256,
        rows=tuple(rows),
        next_action_accuracy=next_accuracy,
    )


def _validate_artifact_inputs(
    profile: object,
    seed: object,
    device: object,
    output: object,
) -> int:
    if not isinstance(profile, ProfileName):
        raise ValueError("profile must be a ProfileName")
    normalized_seed = _nonnegative_integer(seed, "seed")
    if not isinstance(device, DeviceName):
        raise ValueError("device must be a DeviceName")
    if not isinstance(output, Path):
        raise ValueError("output must be a pathlib.Path")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if device is DeviceName.CUDA and not torch.cuda.is_available():
        raise ValueError("CUDA training requested but CUDA is unavailable")
    return normalized_seed


def _capture_runtime_status(device: DeviceName) -> RuntimeStatus:
    if device is DeviceName.CPU:
        device_description = platform_module.processor() or "CPU"
        cuda_runtime_version = None
    else:
        device_description = torch.cuda.get_device_name(torch.cuda.current_device())
        cuda_runtime_version = torch.version.cuda
    return RuntimeStatus(
        python_version=platform_module.python_version(),
        platform=platform_module.platform(),
        processor=platform_module.processor(),
        numpy_version=np.__version__,
        gymnasium_version=gymnasium.__version__,
        torch_version=_training_stack.torch_version,
        stable_baselines3_version=_training_stack.stable_baselines3_version,
        device=device,
        device_description=device_description,
        cuda_runtime_version=cuda_runtime_version,
        cuda_driver_version=None,
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _verify_split_digests(
    profile: ProfileName,
    seed: int,
    splits: object,
) -> None:
    if seed != 0:
        return
    actual = (
        splits.training_digest_sha256,
        splits.validation_digest_sha256,
    )
    if actual != _SEED_ZERO_SPLIT_DIGESTS[profile]:
        raise RuntimeError("seed-zero BC splits no longer match the golden digest contract")


def _bc_workflow_eligible(
    *,
    profile: ProfileName,
    seed: int,
    source: SourceStatus,
    training_device: DeviceName,
    evaluation_device: DeviceName,
) -> bool:
    return (
        profile is ProfileName.CHECKPOINT
        and seed == 0
        and not source.dirty_tree
        and source.required_inputs_committed
        and training_device is DeviceName.CPU
        and evaluation_device is DeviceName.CPU
    )


def _evaluation_filename(suite_id: EvaluationSuiteId) -> str:
    return {
        EvaluationSuiteId.SMOKE: "evaluation-smoke.json",
        EvaluationSuiteId.IID: "evaluation-iid.json",
        EvaluationSuiteId.REGISTER_OOD: "evaluation-ood.json",
    }[suite_id]


def _load_bc_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("model.pt must be a safely loadable weights-only state_dict") from error
    if not isinstance(state, Mapping) or any(not isinstance(name, str) for name in state):
        raise ValueError("model.pt must contain a string-keyed state_dict")
    return state


def _validate_bc_state_dict(
    state: Mapping[str, object],
    *,
    expected_parameter_count: int,
) -> None:
    expected_names = tuple(name for name, _ in _BC_STATE_SCHEMA)
    if tuple(state) != expected_names:
        raise ValueError("model state parameter names must exactly match the BC architecture")
    parameter_count = 0
    for name, shape in _BC_STATE_SCHEMA:
        tensor = state[name]
        if type(tensor) is not torch.Tensor:
            raise ValueError(f"model state parameter {name!r} must be a torch.Tensor")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"model state parameter {name!r} has the wrong shape")
        if tensor.dtype != torch.float32:
            raise ValueError(f"model state parameter {name!r} has the wrong dtype")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"model state parameter {name!r} must contain only finite values")
        parameter_count += tensor.numel()
    if parameter_count != _BC_PARAMETER_COUNT or parameter_count != expected_parameter_count:
        raise ValueError("model state parameter_count must exactly match the BC architecture")


def _validate_evaluation_payload(
    filename: str,
    evaluation: EvaluationFile,
    *,
    profile: ProfileName,
    seed: int,
    parameter_count: int,
    summary: BCTrainingSummary,
) -> None:
    expected_suite = {
        "evaluation-smoke.json": EvaluationSuiteId.SMOKE,
        "evaluation-iid.json": EvaluationSuiteId.IID,
        "evaluation-ood.json": EvaluationSuiteId.REGISTER_OOD,
    }[filename]
    if evaluation.suite_id is not expected_suite:
        raise ValueError("evaluation filename must match its fixed suite identity")
    if profile is ProfileName.SMOKE and expected_suite is not EvaluationSuiteId.SMOKE:
        raise ValueError("smoke artifacts may contain only the smoke evaluation suite")
    if profile is ProfileName.CHECKPOINT and expected_suite is EvaluationSuiteId.SMOKE:
        raise ValueError("checkpoint artifacts may contain only IID and OOD evaluation suites")
    if expected_suite is EvaluationSuiteId.IID and evaluation.next_action_accuracy is None:
        raise ValueError("checkpoint IID evaluation requires held-out next-action accuracy")
    if (
        expected_suite is EvaluationSuiteId.REGISTER_OOD
        and evaluation.next_action_accuracy is not None
    ):
        raise ValueError("register-OOD evaluation must not contain next-action accuracy")
    if expected_suite is EvaluationSuiteId.REGISTER_OOD:
        expected_lanes = (
            ("lower", None),
            ("upper", None),
            ("combined", None),
        )
    else:
        expected_lanes = (
            ("combined", None),
            ("combined", ZERO_SPECTRUM_PROBE),
            ("combined", SHUFFLED_SPECTRUM_PROBE),
        )
    if tuple((row.subset, row.probe) for row in evaluation.rows) != expected_lanes:
        raise ValueError("BC evaluation rows must match the exact probe inventory and subsets")
    for row in evaluation.rows:
        if row.actor_id != f"bc-{seed}" or row.trainer is not TrainerKind.BC or row.seed != seed:
            raise ValueError("evaluation rows must match the BC trainer and artifact seed")
        if row.parameter_count != parameter_count:
            raise ValueError("evaluation row parameter_count must match the artifact")
        if row.training_examples != summary.training_examples:
            raise ValueError("evaluation row training_examples must match the summary")
        if row.training_environment_steps is not None:
            raise ValueError("BC evaluation rows must not contain training environment steps")
        if row.training_wall_time_seconds != summary.training_wall_time_seconds:
            raise ValueError("evaluation row wall time must match the training summary")


def _train_epoch(
    model: torch.nn.Module,
    loader: object,
    optimizer: object,
    class_weights: torch.Tensor,
    device: torch.device,
) -> float:
    model.train()
    weighted_losses: list[float] = []
    sample_weights: list[float] = []
    examples = 0
    for observations, labels in loader:  # type: ignore[union-attr]
        tensors, targets = _batch_to_device(observations, labels, device)
        optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]
        logits = _validated_logits(model(tensors), targets.shape[0])
        losses = torch.nn.functional.cross_entropy(
            logits,
            targets,
            weight=class_weights,
            reduction="none",
        )
        denominator = class_weights[targets].sum()
        objective = losses.sum() / denominator
        if not torch.isfinite(losses).all() or not torch.isfinite(objective):
            raise RuntimeError("training loss must contain only finite values")
        objective.backward()
        optimizer.step()  # type: ignore[union-attr]
        weighted_losses.extend(losses.detach().cpu().to(torch.float64).tolist())
        sample_weights.extend(class_weights[targets].detach().cpu().to(torch.float64).tolist())
        examples += targets.shape[0]
    return _dataset_weighted_loss(
        math.fsum(weighted_losses),
        math.fsum(sample_weights),
        examples,
    )


@torch.inference_mode()
def _evaluate_epoch(
    model: torch.nn.Module,
    loader: object,
    class_weights: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    weighted_losses: list[float] = []
    sample_weights: list[float] = []
    correct = 0
    examples = 0
    for observations, labels in loader:  # type: ignore[union-attr]
        tensors, targets = _batch_to_device(observations, labels, device)
        logits = _validated_logits(model(tensors), targets.shape[0])
        losses = torch.nn.functional.cross_entropy(
            logits,
            targets,
            weight=class_weights,
            reduction="none",
        )
        if not torch.isfinite(losses).all():
            raise RuntimeError("validation loss must contain only finite values")
        weighted_losses.extend(losses.cpu().to(torch.float64).tolist())
        sample_weights.extend(class_weights[targets].cpu().to(torch.float64).tolist())
        correct += int((torch.argmax(logits, dim=1) == targets).sum().item())
        examples += targets.shape[0]
    loss = _dataset_weighted_loss(
        math.fsum(weighted_losses),
        math.fsum(sample_weights),
        examples,
    )
    accuracy = correct / examples
    if not math.isfinite(accuracy):
        raise RuntimeError("validation accuracy must be finite")
    return loss, accuracy


def _batch_to_device(
    observations: object,
    labels: object,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if not isinstance(observations, Mapping) or set(observations) != {"spectrum", "state"}:
        raise RuntimeError("data loader must collate spectrum and state tensors")
    spectrum = observations["spectrum"]
    state = observations["state"]
    if not all(isinstance(value, torch.Tensor) for value in (spectrum, state, labels)):
        raise RuntimeError("data loader must collate torch tensors")
    targets = labels
    if targets.dtype != torch.int64 or targets.ndim != 1:
        raise RuntimeError("data loader labels must be one-dimensional int64 tensors")
    return {
        "spectrum": spectrum.to(device),
        "state": state.to(device),
    }, targets.to(device)


def _validated_logits(value: object, batch_size: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != (batch_size, len(PitchAction)):
        raise RuntimeError("BC model must return one seven-action logit row per example")
    if not torch.isfinite(value).all():
        raise RuntimeError("BC model logits must contain only finite values")
    return value


def _dataset_weighted_loss(weighted_loss_sum: float, weight_sum: float, examples: int) -> float:
    if examples < 1 or not math.isfinite(weighted_loss_sum) or not math.isfinite(weight_sum):
        raise RuntimeError("weighted CE aggregation requires finite nonempty totals")
    if weight_sum <= 0.0:
        raise RuntimeError("weighted CE denominator must be positive")
    loss = weighted_loss_sum / weight_sum
    if not math.isfinite(loss) or loss < 0.0:
        raise RuntimeError("weighted CE loss must be finite and non-negative")
    return loss


def _seed_training(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed % (2**63 - 1))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed % (2**63 - 1))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().to(device="cpu", copy=True)
        for name, tensor in model.state_dict().items()
    }


def _torch_device(value: object) -> torch.device:
    if not isinstance(value, torch.device) or value.type not in {"cpu", "cuda"}:
        raise ValueError("device must be a CPU or CUDA torch.device")
    if value.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA training requested but CUDA is unavailable")
    return value


def _nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer")
    try:
        normalized = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{field} must be a non-negative integer") from error
    if normalized < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return normalized


def _positive_integer(value: object, field: str) -> int:
    normalized = _nonnegative_integer(value, field)
    if normalized < 1:
        raise ValueError(f"{field} must be a positive integer")
    return normalized


def _episode_sequence(episodes: Sequence[EpisodeSpec]) -> tuple[EpisodeSpec, ...]:
    if isinstance(episodes, (str, bytes)):
        raise ValueError("episodes must contain EpisodeSpec values")
    try:
        normalized = tuple(episodes)
    except TypeError as error:
        raise ValueError("episodes must contain EpisodeSpec values") from error
    if not all(isinstance(episode, EpisodeSpec) for episode in normalized):
        raise ValueError("episodes must contain only EpisodeSpec values")
    return normalized


def _require_trajectory_observation(
    observation: object,
    *,
    episode: EpisodeSpec,
    controls: ControlState,
    steps_remaining: int,
) -> None:
    if not isinstance(observation, Mapping):
        raise RuntimeError("planner/environment trajectory mismatch: observation is not a mapping")
    target_note = observation.get("target_note")
    raw_controls = observation.get("controls")
    remaining = observation.get("steps_remaining")
    expected_controls = np.array(
        [controls.octaves, controls.semitones, controls.cents],
        dtype=np.int16,
    )
    matches = (
        isinstance(target_note, np.int64)
        and int(target_note) == episode.target_note_index
        and isinstance(raw_controls, np.ndarray)
        and raw_controls.dtype == np.dtype(np.int16)
        and raw_controls.shape == (3,)
        and np.array_equal(raw_controls, expected_controls)
        and isinstance(remaining, np.int64)
        and int(remaining) == steps_remaining
    )
    if not matches:
        raise RuntimeError("planner/environment trajectory mismatch at pre-action state")


def _dataset_index(index: object, length: int) -> int:
    if isinstance(index, bool):
        raise ValueError("dataset index must be an integer")
    try:
        normalized = operator.index(index)
    except TypeError as error:
        raise ValueError("dataset index must be an integer") from error
    if normalized < 0:
        normalized += length
    if not 0 <= normalized < length:
        raise IndexError("dataset index out of range")
    return normalized


__all__ = [
    "BCExample",
    "OracleTrajectoryDataset",
    "build_oracle_examples",
    "load_bc_actor",
    "next_action_accuracy",
    "save_bc_model",
    "train_bc_artifact",
    "train_behavior_cloning",
    "training_class_weights",
    "validate_bc_artifact",
]
