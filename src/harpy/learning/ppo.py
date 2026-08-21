"""Fresh Stable-Baselines3 PPO training for the frozen sine-pitch task."""

from __future__ import annotations

import operator
import platform as platform_module
import random
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import gymnasium
import numpy as np

from harpy.envs.models import MAX_STEPS, EpisodeResult, ObservationMode, TerminalReason
from harpy.learning.actors import PPOActor
from harpy.learning.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactCompletion,
    ArtifactManifest,
    ArtifactStatus,
    ArtifactWriter,
    CriterionStatus,
    LoadedArtifact,
    PendingArtifactView,
    PPOTrainingCounts,
    RuntimeStatus,
    TrainingConfigDocument,
    TrainingSummaryDocument,
    capture_source_status,
    load_artifact,
    read_training_config,
    read_training_summary,
    required_payload_names,
)
from harpy.learning.cache import SpectrumEvidenceCache
from harpy.learning.dependencies import require_training_dependencies
from harpy.learning.envs import make_cached_sine_pitch_env, make_ppo_training_env
from harpy.learning.errors import LearningExecutionError
from harpy.learning.evaluation import (
    EVALUATION_SCHEMA_VERSION,
    SHUFFLED_SPECTRUM_PROBE,
    ZERO_SPECTRUM_PROBE,
    EvaluationFile,
    EvaluationRow,
    build_evaluation_rows,
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
    DeviceName,
    EpisodeSuite,
    EvaluationSuiteId,
    PPOProfile,
    PPOTrainingSummary,
    ProfileName,
    TrainerKind,
)
from harpy.learning.network import HarpySineFeaturesExtractor
from harpy.learning.observations import POLICY_OBSERVATION_SPACE, PolicyObservationWrapper
from harpy.learning.suites import TRAIN_DISTRIBUTION_ID, fixed_evaluation_suite

_training_stack = require_training_dependencies()
torch = _training_stack.torch
PPO = _training_stack.stable_baselines3.PPO
DummyVecEnv = _training_stack.stable_baselines3.common.vec_env.DummyVecEnv
VecEnv = _training_stack.stable_baselines3.common.vec_env.VecEnv
VecNormalize = _training_stack.stable_baselines3.common.vec_env.VecNormalize
VecEnvWrapper = _training_stack.stable_baselines3.common.vec_env.VecEnvWrapper
VecEnvObs = _training_stack.stable_baselines3.common.vec_env.base_vec_env.VecEnvObs
VecEnvStepReturn = _training_stack.stable_baselines3.common.vec_env.base_vec_env.VecEnvStepReturn

_PPO_STATE_SCHEMA: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("features_extractor.encoder.spectrum_encoder.0.weight", (16, 1, 9)),
    ("features_extractor.encoder.spectrum_encoder.0.bias", (16,)),
    ("features_extractor.encoder.spectrum_encoder.2.weight", (32, 16, 7)),
    ("features_extractor.encoder.spectrum_encoder.2.bias", (32,)),
    ("features_extractor.encoder.state_encoder.0.weight", (32, 5)),
    ("features_extractor.encoder.state_encoder.0.bias", (32,)),
    ("features_extractor.encoder.state_encoder.2.weight", (32, 32)),
    ("features_extractor.encoder.state_encoder.2.bias", (32,)),
    ("features_extractor.encoder.combined_encoder.0.weight", (128, 544)),
    ("features_extractor.encoder.combined_encoder.0.bias", (128,)),
    ("action_net.weight", (7, 128)),
    ("action_net.bias", (7,)),
    ("value_net.weight", (1, 128)),
    ("value_net.bias", (1,)),
)
_PPO_PARAMETER_COUNT = 75_816
_PPO_CORE_PAYLOADS = ("training-config.json", "training-summary.json", "model.zip")


class BudgetExhaustionLatch(gymnasium.Wrapper):
    """Retain one validated Harpy budget terminal across a VecEnv auto-reset."""

    def __init__(self, env: gymnasium.Env) -> None:
        super().__init__(env)
        self._budget_exhausted = False

    def step(self, action: object) -> tuple[object, float, bool, bool, dict[str, object]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            result = self._terminal_result_if_available()
            if result is not None and result.terminal_reason is TerminalReason.BUDGET_EXHAUSTED:
                self._validate_budget_transition(
                    result=result,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
                self._budget_exhausted = True
        return observation, reward, terminated, truncated, info

    def consume_budget_exhausted(self) -> bool:
        """Consume and clear the private one-shot budget-exhaustion signal."""

        exhausted = self._budget_exhausted
        self._budget_exhausted = False
        return exhausted

    def _terminal_result_if_available(self) -> EpisodeResult | None:
        try:
            result = self.env.unwrapped.episode_result
        except (AttributeError, RuntimeError):
            return None
        if not isinstance(result, EpisodeResult):
            raise LearningExecutionError("terminal Harpy environment produced no EpisodeResult")
        return result

    @staticmethod
    def _validate_budget_transition(
        *,
        result: EpisodeResult,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, object],
    ) -> None:
        if terminated or not truncated:
            raise LearningExecutionError(
                "budget-exhausted EpisodeResult must use Gym truncation semantics"
            )
        if (
            len(result.actions) != MAX_STEPS
            or info.get("step_count") != MAX_STEPS
            or info.get("steps_remaining") != 0
            or info.get("submitted") is not False
        ):
            raise LearningExecutionError(
                "budget-exhausted transition must match the validated Harpy terminal state"
            )


class NoBudgetBootstrapVecEnv(VecEnvWrapper):
    """Suppress SB3's timeout interpretation only for latched Harpy budget terminals."""

    def __init__(self, venv: VecEnv, budget_latch: BudgetExhaustionLatch) -> None:
        if not isinstance(budget_latch, BudgetExhaustionLatch):
            raise ValueError("budget_latch must be a BudgetExhaustionLatch")
        super().__init__(venv)
        if self.num_envs != 1:
            raise ValueError("PPO requires exactly one synchronous environment")
        self._budget_latch = budget_latch

    def reset(self) -> VecEnvObs:
        return self.venv.reset()

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()
        if self._budget_latch.consume_budget_exhausted():
            if (
                dones.shape != (1,)
                or not bool(dones[0])
                or len(infos) != 1
                or infos[0].get("TimeLimit.truncated") is not True
                or "terminal_observation" not in infos[0]
            ):
                raise LearningExecutionError(
                    "latched Harpy budget exhaustion did not match SB3 terminal metadata"
                )
            infos[0]["TimeLimit.truncated"] = False
        return observations, rewards, dones, infos


def make_ppo_vec_env(
    env_factory: Callable[[], gymnasium.Env],
) -> NoBudgetBootstrapVecEnv:
    """Build the exact one-environment Harpy PPO wrapper stack."""

    if not callable(env_factory):
        raise ValueError("env_factory must be callable")
    env = env_factory()
    if not isinstance(env, gymnasium.Env):
        raise ValueError("env_factory must return a Gymnasium Env")
    budget_latch = BudgetExhaustionLatch(env)
    vec_env = DummyVecEnv([lambda: PolicyObservationWrapper(budget_latch)])
    return NoBudgetBootstrapVecEnv(vec_env, budget_latch)


def make_ppo_model(
    env: VecEnv,
    *,
    config: PPOProfile,
    seed: int,
    device: str,
) -> PPO:
    """Construct PPO with only the checked-in Harpy policy and optimizer settings."""

    if not isinstance(env, VecEnv) or env.num_envs != 1:
        raise ValueError("env must contain exactly one synchronous environment")
    current = env
    while isinstance(current, VecEnvWrapper):
        if isinstance(current, VecNormalize):
            raise ValueError("PPO does not permit observation or reward normalization")
        current = current.venv
    if not isinstance(config, PPOProfile):
        raise ValueError("config must be a PPOProfile")
    normalized_seed = _nonnegative_integer(seed, "seed")
    normalized_device = _device_name(device)
    return PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": HarpySineFeaturesExtractor,
            "net_arch": [],
            "share_features_extractor": True,
            "normalize_images": False,
        },
        gamma=config.gamma,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        seed=normalized_seed,
        device=normalized_device,
    )


def train_ppo(
    env: VecEnv,
    *,
    config: PPOProfile,
    seed: int,
    device: str = "cpu",
) -> tuple[PPO, PPOTrainingSummary]:
    """Seed every training boundary and run one fresh rollout-aligned PPO fit."""

    if not isinstance(env, VecEnv):
        raise ValueError("env must be a Stable-Baselines3 VecEnv")
    if not isinstance(config, PPOProfile):
        raise ValueError("config must be a PPOProfile")
    normalized_seed = _nonnegative_integer(seed, "seed")
    normalized_device = _device_name(device)
    _seed_training(normalized_seed, normalized_device)
    env.seed(normalized_seed)
    model = make_ppo_model(
        env,
        config=config,
        seed=normalized_seed,
        device=normalized_device,
    )
    started = time.perf_counter()
    model.learn(total_timesteps=config.total_timesteps)
    elapsed = time.perf_counter() - started
    if model.num_timesteps != config.total_timesteps:
        raise LearningExecutionError(
            "PPO completed environment steps must equal the rollout-aligned requested budget"
        )
    return model, PPOTrainingSummary(
        requested_environment_steps=config.total_timesteps,
        completed_environment_steps=model.num_timesteps,
        training_wall_time_seconds=elapsed,
    )


def save_ppo_model(path: Path, model: PPO) -> None:
    """Persist one validated PPO policy to the exact requested local SB3 archive."""

    if not isinstance(path, Path):
        raise ValueError("path must be a pathlib.Path")
    if path.suffix != ".zip":
        raise ValueError("PPO model path must use the .zip suffix")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    _validate_ppo_model(model, expected_parameter_count=_PPO_PARAMETER_COUNT)
    model.save(path)
    if path.is_symlink() or not path.is_file():
        raise LearningExecutionError("SB3 did not create the exact requested model.zip archive")


def validate_ppo_artifact(artifact: LoadedArtifact | PendingArtifactView) -> None:
    """Validate one exact PPO artifact view and every payload available through it."""

    if not isinstance(artifact, (LoadedArtifact, PendingArtifactView)):
        raise ValueError("artifact must be a LoadedArtifact or PendingArtifactView")
    manifest = artifact.manifest
    if manifest.trainer is not TrainerKind.PPO:
        raise ValueError("PPO artifact must declare trainer ppo")
    full_names = required_payload_names(TrainerKind.PPO, manifest.profile)
    if isinstance(artifact, PendingArtifactView):
        declared_names = tuple(record.relative_path for record in artifact.files)
        if declared_names not in {_PPO_CORE_PAYLOADS, full_names}:
            raise ValueError(
                "pending PPO inventory must be exact config/summary/model or the full inventory"
            )
    else:
        declared_names = tuple(record.relative_path for record in manifest.files)
        if declared_names != full_names:
            raise ValueError("complete PPO artifact must declare the full exact inventory")

    config = read_training_config(artifact)
    summary_document = read_training_summary(artifact)
    if config.trainer is not TrainerKind.PPO or summary_document.trainer is not TrainerKind.PPO:
        raise ValueError("PPO config and summary must declare trainer ppo")
    if not isinstance(config.profile_config, PPOProfile):
        raise ValueError("PPO config must contain a PPOProfile")
    if not isinstance(summary_document.summary, PPOTrainingSummary):
        raise ValueError("PPO summary must contain a PPOTrainingSummary")
    summary = summary_document.summary
    if manifest.parameter_count != _PPO_PARAMETER_COUNT:
        raise ValueError("artifact parameter_count must match the PPO architecture")
    if manifest.architecture_schema_id != ARCHITECTURE_SCHEMA_ID:
        raise ValueError("artifact architecture schema must match the PPO architecture")
    if manifest.preprocessing_schema_id != PREPROCESSING_SCHEMA_ID:
        raise ValueError("artifact preprocessing schema must match the PPO actor input")
    if (
        summary.requested_environment_steps != config.profile_config.total_timesteps
        or summary.completed_environment_steps != config.profile_config.total_timesteps
    ):
        raise ValueError("PPO training summary must match the rollout-aligned profile budget")

    model = _load_ppo_model(artifact.file("model.zip"), device="cpu")
    _validate_ppo_model(model, expected_parameter_count=manifest.parameter_count)
    _validate_ppo_training_contract(
        model,
        config=config.profile_config,
        seed=manifest.seed,
        completed_environment_steps=summary.completed_environment_steps,
    )
    for name in declared_names:
        if not name.startswith("evaluation-"):
            continue
        evaluation = EvaluationFile.from_document(artifact.document(name))
        _validate_ppo_evaluation_payload(
            name,
            evaluation,
            profile=manifest.profile,
            seed=manifest.seed,
            parameter_count=manifest.parameter_count,
            summary=summary,
        )


def load_ppo_actor(
    artifact: LoadedArtifact | PendingArtifactView,
    *,
    device: DeviceName = DeviceName.CPU,
) -> PPOActor:
    """Validate and reload one persisted trusted-local PPO archive for inference."""

    if not isinstance(device, DeviceName):
        raise ValueError("device must be a DeviceName")
    if device is DeviceName.CUDA and not torch.cuda.is_available():
        raise ValueError("CUDA actor loading requested but CUDA is unavailable")
    validate_ppo_artifact(artifact)
    model = _load_ppo_model(artifact.file("model.zip"), device=device.value)
    _validate_ppo_model(
        model,
        expected_parameter_count=artifact.manifest.parameter_count,
        expected_device=device.value,
    )
    return PPOActor(model)


def train_ppo_artifact(
    *,
    profile: ProfileName,
    seed: int,
    device: DeviceName,
    output: Path,
) -> LoadedArtifact:
    """Train fresh PPO, reload/evaluate its archive, and complete the artifact last."""

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
        trainer=TrainerKind.PPO,
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
        parameter_count=_PPO_PARAMETER_COUNT,
        training_counts=PPOTrainingCounts(
            requested_environment_steps=profile_config.ppo.total_timesteps,
            completed_environment_steps=None,
        ),
        evaluation_device=None,
        criterion_eligible=False,
        criterion_status=CriterionStatus.INELIGIBLE,
        criterion_met=None,
        files=(),
    )
    writer = ArtifactWriter.begin(output, bootstrap)
    cache = SpectrumEvidenceCache()
    training_env = make_ppo_vec_env(
        lambda: make_ppo_training_env(run_seed=normalized_seed, cache=cache)
    )
    try:
        model, summary = train_ppo(
            training_env,
            config=profile_config.ppo,
            seed=normalized_seed,
            device=device.value,
        )
    finally:
        training_env.close()

    config_document = TrainingConfigDocument(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        trainer=TrainerKind.PPO,
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
        profile_config=profile_config.ppo,
        bc_training_digest_sha256=None,
        bc_validation_digest_sha256=None,
    )
    summary_document = TrainingSummaryDocument(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        trainer=TrainerKind.PPO,
        profile=profile,
        seed=normalized_seed,
        summary=summary,
    )
    writer.publish_json("training-config.json", config_document.to_document())
    writer.publish_json("training-summary.json", summary_document.to_document())
    writer.publish_model("model.zip", lambda path: save_ppo_model(path, model))

    core_view = writer.pending_view(_PPO_CORE_PAYLOADS)
    actor = load_ppo_actor(core_view, device=DeviceName.CPU)
    evaluations = tuple(
        _evaluate_ppo_suite(
            actor,
            suite,
            cache=cache,
            seed=normalized_seed,
            summary=summary,
        )
        for suite in suites
    )
    for evaluation in evaluations:
        writer.publish_json(
            _evaluation_filename(evaluation.suite_id),
            evaluation.to_document(),
        )
    full_view = writer.pending_view(required_payload_names(TrainerKind.PPO, profile))
    validate_ppo_artifact(full_view)
    writer.complete(
        ArtifactCompletion(
            completed_at_utc=_utc_now(),
            training_counts=PPOTrainingCounts(
                requested_environment_steps=profile_config.ppo.total_timesteps,
                completed_environment_steps=summary.completed_environment_steps,
            ),
            evaluation_device=DeviceName.CPU,
            bc_criterion_met=None,
        )
    )
    loaded = load_artifact(output)
    validate_ppo_artifact(loaded)
    return loaded


def _evaluate_ppo_suite(
    actor: PPOActor,
    suite: EpisodeSuite,
    *,
    cache: SpectrumEvidenceCache,
    seed: int,
    summary: PPOTrainingSummary,
) -> EvaluationFile:
    def base_factory() -> gymnasium.Env:
        return make_cached_sine_pitch_env(cache)

    records = evaluate_learned_actor(actor, suite, environment_factory=base_factory)
    rows: list[EvaluationRow] = list(
        build_evaluation_rows(
            actor_id=f"ppo-{seed}",
            trainer=TrainerKind.PPO,
            seed=seed,
            environment_id=ENVIRONMENT_ID,
            observation_mode=ObservationMode.SPECTRUM,
            suite=suite,
            records=records,
            parameter_count=_PPO_PARAMETER_COUNT,
            training_environment_steps=summary.completed_environment_steps,
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
                    actor_id=f"ppo-{seed}",
                    trainer=TrainerKind.PPO,
                    seed=seed,
                    environment_id=ENVIRONMENT_ID,
                    observation_mode=ObservationMode.SPECTRUM,
                    suite=suite,
                    records=probe_records,
                    probe=probe,
                    parameter_count=_PPO_PARAMETER_COUNT,
                    training_environment_steps=summary.completed_environment_steps,
                    training_wall_time_seconds=summary.training_wall_time_seconds,
                )
            )
    return EvaluationFile(
        schema_version=EVALUATION_SCHEMA_VERSION,
        suite_id=suite.suite_id,
        suite_digest_sha256=suite.digest_sha256,
        rows=tuple(rows),
        next_action_accuracy=None,
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


def _evaluation_filename(suite_id: EvaluationSuiteId) -> str:
    return {
        EvaluationSuiteId.SMOKE: "evaluation-smoke.json",
        EvaluationSuiteId.IID: "evaluation-iid.json",
        EvaluationSuiteId.REGISTER_OOD: "evaluation-ood.json",
    }[suite_id]


def _load_ppo_model(path: Path, *, device: str) -> PPO:
    try:
        return PPO.load(path, device=device)
    except Exception as error:
        raise ValueError(
            "model.zip must be a loadable trusted-local Stable-Baselines3 PPO archive"
        ) from error


def _validate_ppo_evaluation_payload(
    filename: str,
    evaluation: EvaluationFile,
    *,
    profile: ProfileName,
    seed: int,
    parameter_count: int,
    summary: PPOTrainingSummary,
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
    if evaluation.next_action_accuracy is not None:
        raise ValueError("PPO evaluation must not contain next-action accuracy")
    expected_lanes = (
        (("lower", None), ("upper", None), ("combined", None))
        if expected_suite is EvaluationSuiteId.REGISTER_OOD
        else (
            ("combined", None),
            ("combined", ZERO_SPECTRUM_PROBE),
            ("combined", SHUFFLED_SPECTRUM_PROBE),
        )
    )
    if tuple((row.subset, row.probe) for row in evaluation.rows) != expected_lanes:
        raise ValueError("PPO evaluation rows must match the exact probe inventory and subsets")
    for row in evaluation.rows:
        if row.actor_id != f"ppo-{seed}" or row.trainer is not TrainerKind.PPO or row.seed != seed:
            raise ValueError("evaluation rows must match the PPO trainer and artifact seed")
        if row.parameter_count != parameter_count:
            raise ValueError("evaluation row parameter_count must match the artifact")
        if row.training_environment_steps != summary.completed_environment_steps:
            raise ValueError("evaluation row environment steps must match the summary")
        if row.training_examples is not None:
            raise ValueError("PPO evaluation rows must not contain training examples")
        if row.training_wall_time_seconds != summary.training_wall_time_seconds:
            raise ValueError("evaluation row wall time must match the training summary")


def _validate_ppo_model(
    model: object,
    *,
    expected_parameter_count: int,
    expected_device: str | None = None,
) -> None:
    if not isinstance(model, PPO):
        raise ValueError("model.zip must contain a Stable-Baselines3 PPO model")
    if model.observation_space != POLICY_OBSERVATION_SPACE:
        raise ValueError("PPO observation space must match the Harpy policy contract")
    if not isinstance(model.action_space, gymnasium.spaces.Discrete) or model.action_space.n != 7:
        raise ValueError("PPO action space must contain exactly seven discrete actions")
    policy = model.policy
    if (
        not isinstance(policy.features_extractor, HarpySineFeaturesExtractor)
        or policy.features_extractor is not policy.pi_features_extractor
        or policy.features_extractor is not policy.vf_features_extractor
        or policy.share_features_extractor is not True
        or policy.normalize_images is not False
        or len(policy.mlp_extractor.policy_net) != 0
        or len(policy.mlp_extractor.value_net) != 0
    ):
        raise ValueError("PPO policy topology must use the exact shared Harpy feature extractor")
    if not isinstance(policy.action_net, torch.nn.Linear) or (
        policy.action_net.in_features,
        policy.action_net.out_features,
    ) != (128, 7):
        raise ValueError("PPO policy must use the direct 128-to-7 action head")
    if not isinstance(policy.value_net, torch.nn.Linear) or (
        policy.value_net.in_features,
        policy.value_net.out_features,
    ) != (128, 1):
        raise ValueError("PPO policy must use the direct 128-to-1 value head")
    parameters = tuple(policy.named_parameters())
    if tuple(name for name, _ in parameters) != tuple(name for name, _ in _PPO_STATE_SCHEMA):
        raise ValueError("PPO policy parameter names must exactly match the architecture")
    parameter_count = 0
    for (name, parameter), (_, shape) in zip(parameters, _PPO_STATE_SCHEMA, strict=True):
        if tuple(parameter.shape) != shape:
            raise ValueError(f"PPO policy parameter {name!r} has the wrong shape")
        if parameter.dtype != torch.float32:
            raise ValueError(f"PPO policy parameter {name!r} has the wrong dtype")
        if not torch.isfinite(parameter).all():
            raise ValueError(f"PPO policy parameter {name!r} must contain only finite values")
        if expected_device is not None and parameter.device.type != expected_device:
            raise ValueError("PPO policy parameters must load on the requested device")
        parameter_count += parameter.numel()
    if parameter_count != _PPO_PARAMETER_COUNT or parameter_count != expected_parameter_count:
        raise ValueError("PPO policy parameter_count must exactly match the architecture")


def _validate_ppo_training_contract(
    model: PPO,
    *,
    config: PPOProfile,
    seed: int,
    completed_environment_steps: int,
) -> None:
    try:
        clip_range = float(model.clip_range(1.0))
    except (TypeError, ValueError) as error:
        raise ValueError("model.zip must match the saved PPO training contract") from error
    if (
        model.gamma != config.gamma
        or model.n_steps != config.n_steps
        or model.batch_size != config.batch_size
        or model.n_epochs != config.n_epochs
        or model.learning_rate != config.learning_rate
        or model.gae_lambda != config.gae_lambda
        or clip_range != config.clip_range
        or model.ent_coef != config.ent_coef
        or model.vf_coef != config.vf_coef
        or model.seed != seed
        or model.n_envs != 1
        or model.num_timesteps != completed_environment_steps
    ):
        raise ValueError("model.zip must match the saved PPO training contract")


def _seed_training(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed % (2**63 - 1))
    if device == "cuda":
        torch.cuda.manual_seed_all(seed % (2**63 - 1))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


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


def _device_name(value: object) -> str:
    if not isinstance(value, str) or value not in {"cpu", "cuda"}:
        raise ValueError("device must be 'cpu' or 'cuda'")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA training requested but CUDA is unavailable")
    return value


__all__ = [
    "BudgetExhaustionLatch",
    "NoBudgetBootstrapVecEnv",
    "load_ppo_actor",
    "make_ppo_model",
    "make_ppo_vec_env",
    "save_ppo_model",
    "train_ppo",
    "train_ppo_artifact",
    "validate_ppo_artifact",
]
