"""PPO training, timeout semantics, and artifact workflow contracts."""

from __future__ import annotations

import subprocess
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import gymnasium
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

import harpy.learning.ppo as ppo_module
from harpy.envs.models import (
    MAX_STEPS,
    EpisodeResult,
    ObservationMode,
    PitchAction,
    TerminalReason,
)
from harpy.envs.sine_pitch import SinePitchEnv
from harpy.learning import bc
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
    SourceStatus,
    TrainingConfigDocument,
    TrainingSummaryDocument,
    load_artifact,
    read_json_document,
    required_payload_names,
)
from harpy.learning.evaluation import (
    EVALUATION_SCHEMA_VERSION,
    SHUFFLED_SPECTRUM_PROBE,
    ZERO_SPECTRUM_PROBE,
    EvaluationFile,
    TerminalEpisodeRecord,
    build_evaluation_rows,
)
from harpy.learning.models import (
    ARCHITECTURE_SCHEMA_ID,
    ENVIRONMENT_CONTRACT_ID,
    ENVIRONMENT_ID,
    PREPROCESSING_SCHEMA_ID,
    PROFILE_CONFIGS,
    SPECTRUM_GRID_ID,
    DeviceName,
    EvaluationSuiteId,
    PPOProfile,
    PPOTrainingSummary,
    ProfileName,
    TrainerKind,
)
from harpy.learning.network import HarpySineFeaturesExtractor
from harpy.learning.ppo import (
    BudgetExhaustionLatch,
    load_ppo_actor,
    make_ppo_model,
    make_ppo_vec_env,
    save_ppo_model,
    train_ppo,
    train_ppo_artifact,
    validate_ppo_artifact,
)
from harpy.learning.suites import TRAIN_DISTRIBUTION_ID, fixed_evaluation_suite


def _assert_observations_equal(
    actual: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    assert tuple(actual) == tuple(expected)
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


class _FixedEpisodeSinePitchEnv(SinePitchEnv):
    """Test utility that repeats one raw Harpy episode across VecEnv auto-resets."""

    def reset(self, *, seed=None, options=None):
        return super().reset(
            seed=seed,
            options={"target_note_index": 0, "source_pitch_cents": 7_000},
        )


class _ConstantActionMultiInputPolicy(MultiInputActorCriticPolicy):
    """Malicious archive fixture retaining the declared parameter schema."""

    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic=False,
    ):
        actions, next_state = super().predict(
            observation,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        return np.full_like(actions, int(PitchAction.SUBMIT)), next_state


class _ConstantFeaturesExtractor(HarpySineFeaturesExtractor):
    """Malicious extractor fixture retaining every declared tensor name and shape."""

    def forward(self, observations):
        state = observations["state"]
        return state.new_zeros((state.shape[0], self.features_dim))


def _suite_records(profile: ProfileName) -> tuple[tuple[EvaluationSuiteId, str], ...]:
    return tuple(
        (suite_id, fixed_evaluation_suite(suite_id).digest_sha256)
        for suite_id in PROFILE_CONFIGS[profile].evaluation_suites
    )


def _source_status() -> SourceStatus:
    return SourceStatus(
        commit="1" * 40,
        dirty_tree=True,
        tracked_diff_sha256="a" * 64,
        dependency_lock_sha256="b" * 64,
        required_inputs_committed=True,
    )


def _runtime_status(device: DeviceName = DeviceName.CPU) -> RuntimeStatus:
    return RuntimeStatus(
        python_version="3.12.11",
        platform="Linux-test",
        processor="x86_64",
        numpy_version=np.__version__,
        gymnasium_version=gymnasium.__version__,
        torch_version=torch.__version__,
        stable_baselines3_version="2.9.0",
        device=device,
        device_description="Test CPU" if device is DeviceName.CPU else "Test CUDA",
        cuda_runtime_version=None if device is DeviceName.CPU else "12.8",
        cuda_driver_version=None,
    )


def _ppo_summary(profile: ProfileName) -> PPOTrainingSummary:
    steps = PROFILE_CONFIGS[profile].ppo.total_timesteps
    return PPOTrainingSummary(
        requested_environment_steps=steps,
        completed_environment_steps=steps,
        training_wall_time_seconds=1.25,
    )


def _incomplete_ppo_manifest(
    profile: ProfileName,
    *,
    seed: int = 0,
    device: DeviceName = DeviceName.CPU,
    parameter_count: int = 75_816,
) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        status=ArtifactStatus.INCOMPLETE,
        trainer=TrainerKind.PPO,
        profile=profile,
        seed=seed,
        created_at_utc="2026-08-10T12:00:00Z",
        completed_at_utc=None,
        source=_source_status(),
        runtime=_runtime_status(device),
        environment_id=ENVIRONMENT_ID,
        environment_contract_id=ENVIRONMENT_CONTRACT_ID,
        train_distribution_id=TRAIN_DISTRIBUTION_ID,
        evaluation_suites=tuple(
            (suite_id.value, digest) for suite_id, digest in _suite_records(profile)
        ),
        spectrum_grid_id=SPECTRUM_GRID_ID,
        preprocessing_schema_id=PREPROCESSING_SCHEMA_ID,
        architecture_schema_id=ARCHITECTURE_SCHEMA_ID,
        parameter_count=parameter_count,
        training_counts=PPOTrainingCounts(
            requested_environment_steps=PROFILE_CONFIGS[profile].ppo.total_timesteps,
            completed_environment_steps=None,
        ),
        evaluation_device=None,
        criterion_eligible=False,
        criterion_status=CriterionStatus.INELIGIBLE,
        criterion_met=None,
        files=(),
    )


def _writer_with_ppo_core(
    output: Path,
    *,
    profile: ProfileName = ProfileName.SMOKE,
    seed: int = 0,
    device: DeviceName = DeviceName.CPU,
    parameter_count: int = 75_816,
    model_saver: Callable[[Path], None] | None = None,
) -> tuple[ArtifactWriter, PendingArtifactView]:
    writer = ArtifactWriter.begin(
        output,
        _incomplete_ppo_manifest(
            profile,
            seed=seed,
            device=device,
            parameter_count=parameter_count,
        ),
    )
    writer.publish_json(
        "training-config.json",
        TrainingConfigDocument(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            trainer=TrainerKind.PPO,
            profile=profile,
            seed=seed,
            device=device,
            environment_id=ENVIRONMENT_ID,
            environment_contract_id=ENVIRONMENT_CONTRACT_ID,
            train_distribution_id=TRAIN_DISTRIBUTION_ID,
            evaluation_suites=_suite_records(profile),
            spectrum_grid_id=SPECTRUM_GRID_ID,
            preprocessing_schema_id=PREPROCESSING_SCHEMA_ID,
            architecture_schema_id=ARCHITECTURE_SCHEMA_ID,
            profile_config=PROFILE_CONFIGS[profile].ppo,
            bc_training_digest_sha256=None,
            bc_validation_digest_sha256=None,
        ).to_document(),
    )
    writer.publish_json(
        "training-summary.json",
        TrainingSummaryDocument(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            trainer=TrainerKind.PPO,
            profile=profile,
            seed=seed,
            summary=_ppo_summary(profile),
        ).to_document(),
    )
    if model_saver is None:
        vec_env = make_ppo_vec_env(SinePitchEnv)
        try:
            model = make_ppo_model(
                vec_env,
                config=PROFILE_CONFIGS[profile].ppo,
                seed=seed,
                device="cpu",
            )
            model.num_timesteps = PROFILE_CONFIGS[profile].ppo.total_timesteps
            writer.publish_model("model.zip", lambda path: save_ppo_model(path, model))
        finally:
            vec_env.close()
    else:
        writer.publish_model("model.zip", model_saver)
    core_names = ("training-config.json", "training-summary.json", "model.zip")
    return writer, writer.pending_view(core_names)


def _ppo_evaluation_file(
    suite_id: EvaluationSuiteId,
    *,
    seed: int = 0,
    include_probes: bool = True,
) -> EvaluationFile:
    suite = fixed_evaluation_suite(suite_id)
    profile = ProfileName.SMOKE if suite_id is EvaluationSuiteId.SMOKE else ProfileName.CHECKPOINT
    summary = _ppo_summary(profile)
    records = tuple(
        TerminalEpisodeRecord(
            episode_index=index,
            episode=episode,
            submitted_success=True,
            within_5_cents=True,
            within_1_cent=True,
            final_absolute_error_cents=0,
            action_count=1,
            excess_actions=0,
            invalid_action_count=0,
            total_return=1.0,
            terminal_reason=TerminalReason.SUBMITTED_SUCCESS,
        )
        for index, episode in enumerate(suite.episodes)
    )
    row_arguments = {
        "actor_id": f"ppo-{seed}",
        "trainer": TrainerKind.PPO,
        "seed": seed,
        "environment_id": ENVIRONMENT_ID,
        "observation_mode": ObservationMode.SPECTRUM,
        "suite": suite,
        "records": records,
        "parameter_count": 75_816,
        "training_environment_steps": summary.completed_environment_steps,
        "training_wall_time_seconds": summary.training_wall_time_seconds,
    }
    rows = list(build_evaluation_rows(**row_arguments))  # type: ignore[arg-type]
    if include_probes and suite_id is not EvaluationSuiteId.REGISTER_OOD:
        for probe in (ZERO_SPECTRUM_PROBE, SHUFFLED_SPECTRUM_PROBE):
            rows.extend(
                build_evaluation_rows(
                    **row_arguments,  # type: ignore[arg-type]
                    probe=probe,
                )
            )
    return EvaluationFile(
        schema_version=EVALUATION_SCHEMA_VERSION,
        suite_id=suite_id,
        suite_digest_sha256=suite.digest_sha256,
        rows=tuple(rows),
        next_action_accuracy=None,
    )


def _publish_ppo_evaluations(
    writer: ArtifactWriter,
    profile: ProfileName,
    *,
    seed: int = 0,
) -> None:
    for filename in required_payload_names(TrainerKind.PPO, profile):
        suite_id = {
            "evaluation-smoke.json": EvaluationSuiteId.SMOKE,
            "evaluation-iid.json": EvaluationSuiteId.IID,
            "evaluation-ood.json": EvaluationSuiteId.REGISTER_OOD,
        }.get(filename)
        if suite_id is not None:
            writer.publish_json(
                filename,
                _ppo_evaluation_file(suite_id, seed=seed).to_document(),
            )


def _ppo_completion(profile: ProfileName) -> ArtifactCompletion:
    steps = PROFILE_CONFIGS[profile].ppo.total_timesteps
    return ArtifactCompletion(
        completed_at_utc="2026-08-10T12:01:00Z",
        training_counts=PPOTrainingCounts(
            requested_environment_steps=steps,
            completed_environment_steps=steps,
        ),
        evaluation_device=DeviceName.CPU,
        bc_criterion_met=None,
    )


def _clean_source_status() -> SourceStatus:
    return SourceStatus(
        commit="2" * 40,
        dirty_tree=False,
        tracked_diff_sha256=("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        dependency_lock_sha256="b" * 64,
        required_inputs_committed=True,
    )


def _terminal_records(suite, *, success: bool = True) -> tuple[TerminalEpisodeRecord, ...]:
    return tuple(
        TerminalEpisodeRecord(
            episode_index=index,
            episode=episode,
            submitted_success=success,
            within_5_cents=success,
            within_1_cent=success,
            final_absolute_error_cents=0 if success else 100,
            action_count=1,
            excess_actions=0 if success else None,
            invalid_action_count=0,
            total_return=1.0 if success else -1.0,
            terminal_reason=(
                TerminalReason.SUBMITTED_SUCCESS if success else TerminalReason.SUBMITTED_FAILURE
            ),
        )
        for index, episode in enumerate(suite.episodes)
    )


def _install_fast_ppo_workflow_fakes(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> None:
    def capture_source(path: Path) -> SourceStatus:
        assert path.name == "ppo.py"
        events.append("source")
        return _clean_source_status()

    def capture_runtime(device: DeviceName) -> RuntimeStatus:
        events.append(f"runtime:{device.value}")
        return _runtime_status(device)

    def train(env, *, config, seed, device="cpu"):
        events.append(f"train:{device}")
        model = make_ppo_model(env, config=config, seed=seed, device="cpu")
        with torch.no_grad():
            for parameter in model.policy.parameters():
                parameter.fill_(0.01 * (seed + 1))
        model.num_timesteps = config.total_timesteps
        return model, PPOTrainingSummary(
            requested_environment_steps=config.total_timesteps,
            completed_environment_steps=config.total_timesteps,
            training_wall_time_seconds=1.25,
        )

    def evaluate(actor, suite, *, environment_factory):
        del actor, environment_factory
        events.append(f"evaluate:{suite.suite_id.value}")
        return _terminal_records(suite)

    monkeypatch.setattr(ppo_module, "capture_source_status", capture_source)
    monkeypatch.setattr(ppo_module, "_capture_runtime_status", capture_runtime)
    monkeypatch.setattr(ppo_module, "train_ppo", train)
    monkeypatch.setattr(ppo_module, "evaluate_learned_actor", evaluate)


def _collect_real_ppo_rollout(
    vec_env: VecEnv,
    *,
    rollout_steps: int,
    terminal_value: float,
) -> np.ndarray:
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs={
            "features_extractor_class": HarpySineFeaturesExtractor,
            "net_arch": [],
            "share_features_extractor": True,
            "normalize_images": False,
        },
        gamma=1.0,
        n_steps=rollout_steps,
        batch_size=rollout_steps,
        n_epochs=1,
        seed=0,
        device="cpu",
    )

    def fixed_action_forward(self, observations, deterministic=False):
        batch_size = observations["state"].shape[0]
        return (
            torch.full(
                (batch_size,),
                int(PitchAction.OCTAVE_DOWN),
                dtype=torch.int64,
                device=self.device,
            ),
            torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device),
            torch.zeros(batch_size, dtype=torch.float32, device=self.device),
        )

    def fixed_terminal_value(self, observations):
        batch_size = observations["state"].shape[0]
        return torch.full(
            (batch_size, 1),
            terminal_value,
            dtype=torch.float32,
            device=self.device,
        )

    model.policy.forward = types.MethodType(fixed_action_forward, model.policy)
    model.policy.predict_values = types.MethodType(fixed_terminal_value, model.policy)
    _, callback = model._setup_learn(total_timesteps=rollout_steps)
    assert model.collect_rollouts(
        vec_env,
        callback,
        model.rollout_buffer,
        n_rollout_steps=rollout_steps,
    )
    return model.rollout_buffer.rewards.copy()


def test_budget_latch_preserves_raw_gym_transition_and_exposes_one_shot_signal() -> None:
    """Catch a latch that mutates public Gym data or misses validated exhaustion."""

    options = {"target_note_index": 12, "source_pitch_cents": 6_137}
    direct = SinePitchEnv()
    wrapped = BudgetExhaustionLatch(SinePitchEnv())
    direct.reset(options=options)
    wrapped.reset(options=options)

    for _ in range(MAX_STEPS):
        expected = direct.step(PitchAction.OCTAVE_DOWN)
        actual = wrapped.step(PitchAction.OCTAVE_DOWN)

    actual_observation, actual_reward, actual_terminated, actual_truncated, actual_info = actual
    (
        expected_observation,
        expected_reward,
        expected_terminated,
        expected_truncated,
        expected_info,
    ) = expected
    _assert_observations_equal(actual_observation, expected_observation)
    assert actual_reward == expected_reward == -1.01
    assert actual_terminated is expected_terminated is False
    assert actual_truncated is expected_truncated is True
    assert (
        actual_info
        == expected_info
        == {
            "step_count": 64,
            "steps_remaining": 0,
            "action": "Octave Down",
            "action_applied": False,
            "submitted": False,
            "submitted_success": False,
        }
    )
    assert not any("budget" in key.lower() for key in actual_info)
    result = wrapped.unwrapped.episode_result
    assert isinstance(result, EpisodeResult)
    assert result == direct.episode_result
    assert result.terminal_reason is TerminalReason.BUDGET_EXHAUSTED
    assert len(result.actions) == MAX_STEPS
    assert wrapped.consume_budget_exhausted() is True
    assert wrapped.consume_budget_exhausted() is False


def test_budget_latch_leaves_submit_termination_unchanged_and_unlatched() -> None:
    """Catch a latch that mistakes explicit Harpy termination for budget exhaustion."""

    options = {"target_note_index": 12, "source_pitch_cents": 6_137}
    direct = SinePitchEnv()
    wrapped = BudgetExhaustionLatch(SinePitchEnv())
    direct.reset(options=options)
    wrapped.reset(options=options)

    expected = direct.step(PitchAction.SUBMIT)
    actual = wrapped.step(PitchAction.SUBMIT)

    _assert_observations_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]
    assert actual[1] == -1.0
    assert actual[2] is True
    assert actual[3] is False
    assert wrapped.unwrapped.episode_result.terminal_reason is TerminalReason.SUBMITTED_FAILURE
    assert wrapped.consume_budget_exhausted() is False


def test_ppo_vec_env_suppresses_only_validated_harpy_budget_timeout_copy() -> None:
    """Catch an adapter that leaks a marker or leaves SB3's Harpy timeout flag set."""

    vec_env = make_ppo_vec_env(SinePitchEnv)
    try:
        observation = vec_env.reset()
        assert vec_env.num_envs == 1
        assert tuple(observation) == ("spectrum", "state")
        for _ in range(MAX_STEPS):
            observation, rewards, dones, infos = vec_env.step(
                np.array([int(PitchAction.OCTAVE_DOWN)])
            )
    finally:
        vec_env.close()

    assert rewards.shape == dones.shape == (1,)
    assert rewards[0] == np.float32(-1.01)
    assert dones.tolist() == [True]
    assert infos[0]["TimeLimit.truncated"] is False
    assert tuple(infos[0]["terminal_observation"]) == ("spectrum", "state")
    assert infos[0]["terminal_observation"]["state"][-1] == np.float32(0.0)
    assert observation["state"][0, -1] == np.float32(1.0)
    assert not any("budget" in key.lower() for key in infos[0])


def test_ppo_vec_env_preserves_ordinary_time_limit_bootstrap_flag() -> None:
    """Catch an adapter that suppresses non-Harpy TimeLimit bootstrapping."""

    vec_env = make_ppo_vec_env(
        lambda: gymnasium.wrappers.TimeLimit(SinePitchEnv(), max_episode_steps=1)
    )
    try:
        vec_env.set_options({"target_note_index": 12, "source_pitch_cents": 5_863})
        vec_env.reset()
        _, rewards, dones, infos = vec_env.step(np.array([int(PitchAction.CENT_UP)]))
    finally:
        vec_env.close()

    assert rewards[0] == np.float32(1 / 6_100 - 0.00001)
    assert dones.tolist() == [True]
    assert infos[0]["TimeLimit.truncated"] is True
    assert tuple(infos[0]["terminal_observation"]) == ("spectrum", "state")
    assert not any("budget" in key.lower() for key in infos[0])


def test_real_ppo_rollout_keeps_exact_budget_reward_and_adapter_removal_mutates_it() -> None:
    """Catch timeout-value addition by proving the same rollout fails without the adapter."""

    terminal_value = 7.25
    adapted = make_ppo_vec_env(SinePitchEnv)
    try:
        adapted_rewards = _collect_real_ppo_rollout(
            adapted,
            rollout_steps=MAX_STEPS,
            terminal_value=terminal_value,
        )
    finally:
        adapted.close()

    adapter_removed = make_ppo_vec_env(SinePitchEnv)
    unadapted = adapter_removed.venv
    try:
        unadapted_rewards = _collect_real_ppo_rollout(
            unadapted,
            rollout_steps=MAX_STEPS,
            terminal_value=terminal_value,
        )
    finally:
        unadapted.close()

    raw_terminal_reward = np.float32(-1.01)
    assert adapted_rewards[-1, 0] == raw_terminal_reward
    assert unadapted_rewards[-1, 0] == raw_terminal_reward + np.float32(terminal_value)
    assert unadapted_rewards[-1, 0] != raw_terminal_reward


def test_real_ppo_rollout_still_bootstraps_an_ordinary_time_limit() -> None:
    """Catch broad timeout suppression by exercising SB3's real value-addition branch."""

    terminal_value = 7.25
    vec_env = make_ppo_vec_env(
        lambda: gymnasium.wrappers.TimeLimit(
            _FixedEpisodeSinePitchEnv(),
            max_episode_steps=1,
        )
    )
    try:
        rewards = _collect_real_ppo_rollout(
            vec_env,
            rollout_steps=2,
            terminal_value=terminal_value,
        )
    finally:
        vec_env.close()

    raw_reward = np.float32(1_200 / 6_100 - 0.00001)
    np.testing.assert_array_equal(
        rewards[:, 0],
        np.full(2, raw_reward + np.float32(terminal_value), dtype=np.float32),
    )


def test_make_ppo_model_passes_only_the_exact_pinned_sb3_constructor_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch any inserted MLP, normalization, mask, callback, or unpinned PPO option."""

    config = PROFILE_CONFIGS[ProfileName.SMOKE].ppo
    vec_env = make_ppo_vec_env(SinePitchEnv)
    sentinel = object()
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_ppo(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(ppo_module, "PPO", fake_ppo)
    try:
        actual = make_ppo_model(vec_env, config=config, seed=17, device="cpu")
    finally:
        vec_env.close()

    assert actual is sentinel
    assert calls == [
        (
            ("MultiInputPolicy", vec_env),
            {
                "policy_kwargs": {
                    "features_extractor_class": HarpySineFeaturesExtractor,
                    "net_arch": [],
                    "share_features_extractor": True,
                    "normalize_images": False,
                },
                "gamma": config.gamma,
                "n_steps": config.n_steps,
                "batch_size": config.batch_size,
                "n_epochs": config.n_epochs,
                "learning_rate": config.learning_rate,
                "gae_lambda": config.gae_lambda,
                "clip_range": config.clip_range,
                "ent_coef": config.ent_coef,
                "vf_coef": config.vf_coef,
                "seed": 17,
                "device": "cpu",
            },
        )
    ]


def test_make_ppo_model_has_one_shared_direct_head_policy_with_75816_parameters() -> None:
    """Catch duplicated extractors, default SB3 MLPs, normalization, or topology drift."""

    vec_env = make_ppo_vec_env(SinePitchEnv)
    try:
        model = make_ppo_model(
            vec_env,
            config=PROFILE_CONFIGS[ProfileName.SMOKE].ppo,
            seed=0,
            device="cpu",
        )
        policy = model.policy
        assert model.n_envs == vec_env.num_envs == 1
        assert not isinstance(vec_env, VecNormalize)
        assert not isinstance(vec_env.venv, VecNormalize)
        assert policy.share_features_extractor is True
        assert policy.features_extractor is policy.pi_features_extractor
        assert policy.features_extractor is policy.vf_features_extractor
        assert isinstance(policy.features_extractor, HarpySineFeaturesExtractor)
        assert policy.normalize_images is False
        assert len(policy.mlp_extractor.policy_net) == 0
        assert len(policy.mlp_extractor.value_net) == 0
        assert (policy.action_net.in_features, policy.action_net.out_features) == (128, 7)
        assert (policy.value_net.in_features, policy.value_net.out_features) == (128, 1)
        assert sum(parameter.numel() for parameter in policy.parameters()) == 75_816
    finally:
        vec_env.close()


@pytest.mark.parametrize(
    ("seed", "device"),
    [
        (True, "cpu"),
        (-1, "cpu"),
        (0, "auto"),
    ],
)
def test_make_ppo_model_rejects_invalid_seed_or_device_before_sb3_construction(
    monkeypatch: pytest.MonkeyPatch,
    seed: object,
    device: str,
) -> None:
    """Catch bool/negative seeds or implicit device fallback crossing into SB3."""

    vec_env = make_ppo_vec_env(SinePitchEnv)
    constructed = False

    def forbidden_constructor(*args, **kwargs):
        nonlocal constructed
        del args, kwargs
        constructed = True
        pytest.fail("invalid input must be rejected before PPO construction")

    monkeypatch.setattr(ppo_module, "PPO", forbidden_constructor)
    try:
        with pytest.raises(ValueError):
            make_ppo_model(
                vec_env,
                config=PROFILE_CONFIGS[ProfileName.SMOKE].ppo,
                seed=seed,  # type: ignore[arg-type]
                device=device,
            )
    finally:
        vec_env.close()
    assert constructed is False


def test_make_ppo_model_rejects_a_multi_environment_vecenv() -> None:
    """Catch expansion beyond the pinned one synchronous training environment."""

    config = PROFILE_CONFIGS[ProfileName.SMOKE].ppo
    unadapted = DummyVecEnv(
        [
            lambda: ppo_module.PolicyObservationWrapper(SinePitchEnv()),
            lambda: ppo_module.PolicyObservationWrapper(SinePitchEnv()),
        ]
    )
    try:
        with pytest.raises(ValueError, match=r"one synchronous|synchronous environment"):
            make_ppo_model(unadapted, config=config, seed=0, device="cpu")
    finally:
        unadapted.close()


def test_make_ppo_model_rejects_observation_or_reward_normalization() -> None:
    """Catch a VecNormalize layer changing the frozen observation or reward contract."""

    config = PROFILE_CONFIGS[ProfileName.SMOKE].ppo
    normalized = VecNormalize(make_ppo_vec_env(SinePitchEnv))
    try:
        with pytest.raises(ValueError, match="normalization"):
            make_ppo_model(normalized, config=config, seed=0, device="cpu")
    finally:
        normalized.close()


@pytest.mark.parametrize("profile", tuple(ProfileName))
def test_train_ppo_seeds_before_fresh_construction_and_records_exact_aligned_steps(
    monkeypatch: pytest.MonkeyPatch,
    profile: ProfileName,
) -> None:
    """Catch late seeding, checkpoint reuse, callbacks, or requested/completed count drift."""

    config = PROFILE_CONFIGS[profile].ppo
    vec_env = make_ppo_vec_env(SinePitchEnv)
    events: list[object] = []

    class FreshModel:
        num_timesteps = 0

        def learn(self, **kwargs):
            events.append(("learn", kwargs))
            self.num_timesteps = kwargs["total_timesteps"]
            return self

    model = FreshModel()
    monkeypatch.setattr(ppo_module.random, "seed", lambda seed: events.append(("python", seed)))
    monkeypatch.setattr(
        ppo_module.np.random,
        "seed",
        lambda seed: events.append(("numpy", seed)),
    )
    monkeypatch.setattr(
        ppo_module.torch,
        "manual_seed",
        lambda seed: events.append(("torch", seed)),
    )
    monkeypatch.setattr(
        ppo_module.torch,
        "use_deterministic_algorithms",
        lambda enabled: events.append(("deterministic", enabled)),
    )
    monkeypatch.setattr(
        vec_env,
        "seed",
        lambda seed: events.append(("gym", seed)),
    )

    def fake_make_ppo_model(env, *, config, seed, device):
        events.append(("construct", env, config, seed, device))
        return model

    monkeypatch.setattr(ppo_module, "make_ppo_model", fake_make_ppo_model)
    try:
        actual_model, summary = train_ppo(vec_env, config=config, seed=17)
    finally:
        vec_env.close()

    assert actual_model is model
    assert config.total_timesteps % config.n_steps == 0
    assert summary.requested_environment_steps == config.total_timesteps
    assert summary.completed_environment_steps == config.total_timesteps
    assert summary.training_wall_time_seconds >= 0.0
    assert events == [
        ("python", 17),
        ("numpy", 17),
        ("torch", 17),
        ("deterministic", True),
        ("gym", 17),
        ("construct", vec_env, config, 17, "cpu"),
        ("learn", {"total_timesteps": config.total_timesteps}),
    ]


def test_train_ppo_rejects_an_observed_step_count_different_from_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch silent acceptance of an incomplete or overrun SB3 rollout budget."""

    config = PROFILE_CONFIGS[ProfileName.SMOKE].ppo
    vec_env = make_ppo_vec_env(SinePitchEnv)

    class ShortModel:
        num_timesteps = 0

        def learn(self, *, total_timesteps: int):
            self.num_timesteps = total_timesteps - 1
            return self

    monkeypatch.setattr(ppo_module, "make_ppo_model", lambda *args, **kwargs: ShortModel())
    try:
        with pytest.raises(RuntimeError, match="completed environment steps"):
            train_ppo(vec_env, config=config, seed=0)
    finally:
        vec_env.close()


@pytest.mark.filterwarnings("ignore:CUDA initialization:UserWarning")
def test_real_train_ppo_never_loads_bc_and_uses_stochastic_training_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch any BC warm start or deterministic-inference path entering PPO training."""

    def poisoned_bc_path(*args, **kwargs):
        del args, kwargs
        pytest.fail("PPO must not call a behavior-cloning persistence path")

    for name in (
        "load_artifact",
        "load_bc_actor",
        "save_bc_model",
        "validate_bc_artifact",
        "_load_bc_state_dict",
    ):
        monkeypatch.setattr(bc, name, poisoned_bc_path)

    config = PPOProfile(
        total_timesteps=2,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        learning_rate=3e-4,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
    )
    vec_env = make_ppo_vec_env(_FixedEpisodeSinePitchEnv)
    real_make = ppo_module.make_ppo_model
    stochastic_flags: list[bool] = []
    initial_state: dict[str, torch.Tensor] = {}

    def recording_make(env, *, config, seed, device):
        model = real_make(env, config=config, seed=seed, device=device)
        initial_state.update(
            {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.policy.state_dict().items()
            }
        )
        original_forward = model.policy.forward

        def recording_forward(observations, deterministic=False):
            stochastic_flags.append(deterministic)
            return original_forward(observations, deterministic=deterministic)

        model.policy.forward = recording_forward
        monkeypatch.setattr(model, "predict", poisoned_bc_path)
        return model

    monkeypatch.setattr(ppo_module, "make_ppo_model", recording_make)
    try:
        model, summary = train_ppo(vec_env, config=config, seed=29)
    finally:
        vec_env.close()

    assert summary.requested_environment_steps == summary.completed_environment_steps == 2
    assert initial_state
    assert any(
        not torch.equal(initial_state[name], tensor.detach().cpu())
        for name, tensor in model.policy.state_dict().items()
    )
    assert stochastic_flags and not any(stochastic_flags)


def test_importing_ppo_never_imports_the_behavior_cloning_module() -> None:
    """Catch a static Task 7 dependency even when no BC function is invoked."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import harpy.learning.ppo; "
                "assert 'harpy.learning.bc' not in sys.modules"
            ),
        ],
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.filterwarnings("ignore:CUDA initialization:UserWarning")
def test_real_train_ppo_repeats_weights_for_the_same_fresh_seed() -> None:
    """Catch process-global RNG leakage or reuse of an earlier trained model."""

    config = PPOProfile(
        total_timesteps=2,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        learning_rate=3e-4,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
    )

    def run() -> tuple[dict[str, torch.Tensor], PPOTrainingSummary]:
        vec_env = make_ppo_vec_env(_FixedEpisodeSinePitchEnv)
        try:
            model, summary = train_ppo(vec_env, config=config, seed=73)
            state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.policy.state_dict().items()
            }
            return state, summary
        finally:
            vec_env.close()

    first_state, first_summary = run()
    second_state, second_summary = run()

    assert tuple(first_state) == tuple(second_state)
    assert all(torch.equal(first_state[name], second_state[name]) for name in first_state)
    assert (
        first_summary.requested_environment_steps == second_summary.requested_environment_steps == 2
    )
    assert (
        first_summary.completed_environment_steps == second_summary.completed_environment_steps == 2
    )
    assert first_summary.training_wall_time_seconds >= 0.0
    assert second_summary.training_wall_time_seconds >= 0.0


@pytest.mark.filterwarnings("ignore:CUDA initialization:UserWarning")
def test_save_ppo_model_creates_the_exact_reloadable_sb3_archive(tmp_path: Path) -> None:
    """Catch loose paths, doubled suffixes, or archives that do not preserve policy tensors."""

    vec_env = make_ppo_vec_env(SinePitchEnv)
    try:
        model = make_ppo_model(
            vec_env,
            config=PROFILE_CONFIGS[ProfileName.SMOKE].ppo,
            seed=31,
            device="cpu",
        )
        expected_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.policy.state_dict().items()
        }
        path = tmp_path / "model.zip"
        save_ppo_model(path, model)
    finally:
        vec_env.close()

    assert path.is_file()
    assert not (tmp_path / "model.zip.zip").exists()
    loaded = PPO.load(path, device="cpu")
    actual_state = loaded.policy.state_dict()
    assert tuple(actual_state) == tuple(expected_state)
    assert all(torch.equal(actual_state[name].cpu(), expected_state[name]) for name in actual_state)
    assert sum(parameter.numel() for parameter in loaded.policy.parameters()) == 75_816


@pytest.mark.filterwarnings("ignore:CUDA initialization:UserWarning")
def test_pending_ppo_core_validates_before_actor_exposure_and_defaults_to_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch loose model paths, partial validation, or a non-CPU default actor reload."""

    _, pending = _writer_with_ppo_core(tmp_path / "artifact")
    events: list[str] = []
    real_validate = ppo_module.validate_ppo_artifact
    real_load = ppo_module.PPO.load

    def recording_validate(artifact):
        events.append("validate")
        return real_validate(artifact)

    def recording_load(path, *args, **kwargs):
        events.append(f"load:{kwargs.get('device')}")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(ppo_module, "validate_ppo_artifact", recording_validate)
    monkeypatch.setattr(ppo_module.PPO, "load", recording_load)

    actor = load_ppo_actor(pending)

    assert isinstance(actor, PPOActor)
    assert events[0] == "validate"
    assert events[-1] == "load:cpu"
    assert events.count("load:cpu") >= 2
    assert all(parameter.device.type == "cpu" for parameter in actor._model.policy.parameters())
    observation, _ = SinePitchEnv().reset(
        options={"target_note_index": 12, "source_pitch_cents": 6_137}
    )
    assert isinstance(actor.act(observation), PitchAction)


def test_validate_ppo_artifact_rejects_wrong_count_and_partial_pending_inventory(
    tmp_path: Path,
) -> None:
    """Catch manifest topology drift or acceptance of an undeclared loose archive view."""

    writer, pending = _writer_with_ppo_core(
        tmp_path / "wrong-count",
        parameter_count=75_817,
    )
    with pytest.raises(ValueError, match="parameter_count"):
        validate_ppo_artifact(pending)
    with pytest.raises(ValueError, match="inventory"):
        validate_ppo_artifact(writer.pending_view(("training-config.json", "model.zip")))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("invalid-archive", "trusted-local"),
        ("wrong-topology", "topology"),
        ("nonfinite", "finite"),
    ],
)
@pytest.mark.filterwarnings("ignore:CUDA initialization:UserWarning")
def test_validate_ppo_artifact_rejects_invalid_or_inexact_trusted_local_archives(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    """Catch archive load failures, framework-default MLPs, and unsafe policy tensors."""

    vec_env: VecEnv | None = None
    if mutation == "invalid-archive":

        def saver(path: Path) -> None:
            path.write_bytes(b"not an SB3 archive")

    else:
        vec_env = make_ppo_vec_env(SinePitchEnv)
        if mutation == "wrong-topology":
            model = PPO(
                "MultiInputPolicy",
                vec_env,
                gamma=1.0,
                n_steps=2,
                batch_size=2,
                n_epochs=1,
                seed=0,
                device="cpu",
            )
        else:
            model = make_ppo_model(
                vec_env,
                config=PROFILE_CONFIGS[ProfileName.SMOKE].ppo,
                seed=0,
                device="cpu",
            )
            with torch.no_grad():
                next(model.policy.parameters()).reshape(-1)[0] = float("nan")

        def saver(path: Path) -> None:
            model.save(path)

    try:
        _, pending = _writer_with_ppo_core(
            tmp_path / mutation,
            model_saver=saver,
        )
        with pytest.raises(ValueError, match=message):
            validate_ppo_artifact(pending)
    finally:
        if vec_env is not None:
            vec_env.close()


@pytest.mark.filterwarnings("ignore:CUDA initialization:UserWarning")
def test_validate_ppo_artifact_rejects_policy_subclass_that_overrides_inference(
    tmp_path: Path,
) -> None:
    """Catch a serialized policy subclass overriding predict behind an exact state schema."""

    vec_env = make_ppo_vec_env(SinePitchEnv)
    try:
        model = make_ppo_model(
            vec_env,
            config=PROFILE_CONFIGS[ProfileName.SMOKE].ppo,
            seed=0,
            device="cpu",
        )
        model.num_timesteps = PROFILE_CONFIGS[ProfileName.SMOKE].ppo.total_timesteps
        model.policy.__class__ = _ConstantActionMultiInputPolicy
        model.policy_class = _ConstantActionMultiInputPolicy
        writer, pending = _writer_with_ppo_core(
            tmp_path / "policy-subclass",
            model_saver=lambda path: model.save(path),
        )
    finally:
        vec_env.close()

    raw_model = PPO.load(pending.file("model.zip"), device="cpu")
    raw_env = SinePitchEnv()
    try:
        observation, _ = raw_env.reset(
            options={"target_note_index": 12, "source_pitch_cents": 6_137}
        )
    finally:
        raw_env.close()
    assert writer.root == pending.root
    assert type(raw_model.policy) is _ConstantActionMultiInputPolicy
    assert PPOActor(raw_model).act(observation) is PitchAction.SUBMIT
    with pytest.raises(ValueError, match="exact SB3 MultiInputPolicy"):
        validate_ppo_artifact(pending)
    with pytest.raises(ValueError, match="exact SB3 MultiInputPolicy"):
        load_ppo_actor(pending)


@pytest.mark.filterwarnings("ignore:CUDA initialization:UserWarning")
def test_validate_ppo_artifact_rejects_feature_extractor_subclass_that_overrides_inference(
    tmp_path: Path,
) -> None:
    """Catch an extractor subclass overriding forward behind an exact state schema."""

    vec_env = make_ppo_vec_env(SinePitchEnv)
    try:
        model = make_ppo_model(
            vec_env,
            config=PROFILE_CONFIGS[ProfileName.SMOKE].ppo,
            seed=0,
            device="cpu",
        )
        model.num_timesteps = PROFILE_CONFIGS[ProfileName.SMOKE].ppo.total_timesteps
        model.policy.features_extractor.__class__ = _ConstantFeaturesExtractor
        model.policy_kwargs["features_extractor_class"] = _ConstantFeaturesExtractor
        _, pending = _writer_with_ppo_core(
            tmp_path / "extractor-subclass",
            model_saver=lambda path: model.save(path),
        )
    finally:
        vec_env.close()

    raw_model = PPO.load(pending.file("model.zip"), device="cpu")
    assert type(raw_model.policy) is MultiInputActorCriticPolicy
    assert type(raw_model.policy.features_extractor) is _ConstantFeaturesExtractor
    with pytest.raises(ValueError, match="exact Harpy sine feature extractor"):
        validate_ppo_artifact(pending)
    with pytest.raises(ValueError, match="exact Harpy sine feature extractor"):
        load_ppo_actor(pending)


@pytest.mark.filterwarnings("ignore:CUDA initialization:UserWarning")
def test_validate_ppo_artifact_cross_checks_saved_sb3_training_contract(
    tmp_path: Path,
) -> None:
    """Catch an archive whose algorithm settings disagree with its strict JSON contract."""

    vec_env = make_ppo_vec_env(SinePitchEnv)
    try:
        model = make_ppo_model(
            vec_env,
            config=PROFILE_CONFIGS[ProfileName.SMOKE].ppo,
            seed=0,
            device="cpu",
        )
        model.num_timesteps = PROFILE_CONFIGS[ProfileName.SMOKE].ppo.total_timesteps
        model.gamma = 0.5
        _, pending = _writer_with_ppo_core(
            tmp_path / "wrong-gamma",
            model_saver=lambda path: model.save(path),
        )
        with pytest.raises(ValueError, match="training contract"):
            validate_ppo_artifact(pending)
    finally:
        vec_env.close()


@pytest.mark.parametrize("profile", tuple(ProfileName))
@pytest.mark.filterwarnings("ignore:CUDA initialization:UserWarning")
def test_full_pending_and_complete_ppo_artifacts_validate_exact_evaluation_inventory(
    tmp_path: Path,
    profile: ProfileName,
) -> None:
    """Catch skipped evaluation decoding or a complete manifest with a loose inventory."""

    writer, _ = _writer_with_ppo_core(tmp_path / profile.value, profile=profile)
    _publish_ppo_evaluations(writer, profile)
    full_names = required_payload_names(TrainerKind.PPO, profile)
    full_pending = writer.pending_view(full_names)

    validate_ppo_artifact(full_pending)
    completed = writer.complete(_ppo_completion(profile))
    loaded = load_artifact(completed.root)
    validate_ppo_artifact(loaded)

    assert loaded is not completed
    assert tuple(record.relative_path for record in loaded.manifest.files) == full_names
    assert loaded.manifest.criterion_met is None
    assert loaded.manifest.criterion_status is CriterionStatus.INELIGIBLE
    for filename in full_names:
        if filename.startswith("evaluation-"):
            evaluation = EvaluationFile.from_document(loaded.document(filename))
            assert evaluation.next_action_accuracy is None


@pytest.mark.filterwarnings("ignore:CUDA initialization:UserWarning")
def test_ppo_evaluation_validation_rejects_missing_probes_and_next_action_accuracy(
    tmp_path: Path,
) -> None:
    """Catch incomplete diagnostic lanes or PPO ownership of the BC-only accuracy field."""

    writer, _ = _writer_with_ppo_core(tmp_path / "missing-probes")
    writer.publish_json(
        "evaluation-smoke.json",
        _ppo_evaluation_file(
            EvaluationSuiteId.SMOKE,
            include_probes=False,
        ).to_document(),
    )
    pending = writer.pending_view(required_payload_names(TrainerKind.PPO, ProfileName.SMOKE))
    with pytest.raises(ValueError, match="probe inventory"):
        validate_ppo_artifact(pending)

    other_writer, _ = _writer_with_ppo_core(tmp_path / "accuracy")
    document = _ppo_evaluation_file(EvaluationSuiteId.SMOKE).to_document()
    document["next_action_accuracy"] = 0.5
    other_writer.publish_json("evaluation-smoke.json", document)
    other_pending = other_writer.pending_view(
        required_payload_names(TrainerKind.PPO, ProfileName.SMOKE)
    )
    with pytest.raises(ValueError, match="next_action_accuracy"):
        validate_ppo_artifact(other_pending)


@pytest.mark.parametrize(
    ("profile", "expected_evaluations", "expected_rollouts"),
    [
        (ProfileName.SMOKE, ("evaluation-smoke.json",), 3),
        (
            ProfileName.CHECKPOINT,
            ("evaluation-iid.json", "evaluation-ood.json"),
            4,
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:CUDA initialization:UserWarning")
def test_train_ppo_artifact_uses_exact_atomic_order_and_complete_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: ProfileName,
    expected_evaluations: tuple[str, ...],
    expected_rollouts: int,
) -> None:
    """Catch pre-persistence evaluation, skipped pending validation, or non-final completion."""

    output = tmp_path / profile.value
    events: list[str] = []
    _install_fast_ppo_workflow_fakes(monkeypatch, events)
    completed_views: list[LoadedArtifact] = []

    real_begin = ArtifactWriter.begin.__func__
    real_publish_json = ArtifactWriter.publish_json
    real_pending_view = ArtifactWriter.pending_view
    real_complete = ArtifactWriter.complete
    real_save = ppo_module.save_ppo_model
    real_load_actor = ppo_module.load_ppo_actor
    real_validate = ppo_module.validate_ppo_artifact

    def begin(cls, output: Path, manifest: ArtifactManifest) -> ArtifactWriter:
        events.append("begin")
        return real_begin(cls, output, manifest)

    def publish_json(writer: ArtifactWriter, filename: str, document: object):
        events.append(f"publish:{filename}")
        return real_publish_json(writer, filename, document)  # type: ignore[arg-type]

    def pending_view(writer: ArtifactWriter, names: Sequence[str]) -> PendingArtifactView:
        events.append(f"pending:{','.join(names)}")
        return real_pending_view(writer, names)

    def save(path: Path, model: PPO) -> None:
        events.append("save")
        real_save(path, model)

    def validate(artifact) -> None:
        size: int | str = (
            len(artifact.files) if isinstance(artifact, PendingArtifactView) else "complete"
        )
        events.append(f"validate:{size}")
        real_validate(artifact)

    def load_actor(artifact, *, device: DeviceName = DeviceName.CPU) -> PPOActor:
        events.append(f"load-request:{device.value}")
        actor = real_load_actor(artifact, device=device)
        events.append(f"actor-exposed:{device.value}")
        return actor

    def complete(writer: ArtifactWriter, completion: ArtifactCompletion) -> LoadedArtifact:
        loaded = real_complete(writer, completion)
        completed_views.append(loaded)
        events.append("complete")
        return loaded

    def forbidden_post_completion_load(path: Path) -> LoadedArtifact:
        del path
        events.append("post-completion-load")
        raise AssertionError("no fallible workflow operation may run after completion")

    monkeypatch.setattr(ArtifactWriter, "begin", classmethod(begin))
    monkeypatch.setattr(ArtifactWriter, "publish_json", publish_json)
    monkeypatch.setattr(ArtifactWriter, "pending_view", pending_view)
    monkeypatch.setattr(ArtifactWriter, "complete", complete)
    monkeypatch.setattr(ppo_module, "save_ppo_model", save)
    monkeypatch.setattr(ppo_module, "validate_ppo_artifact", validate)
    monkeypatch.setattr(ppo_module, "load_ppo_actor", load_actor)
    monkeypatch.setattr(
        ppo_module,
        "load_artifact",
        forbidden_post_completion_load,
        raising=False,
    )

    artifact = train_ppo_artifact(
        profile=profile,
        seed=0,
        device=DeviceName.CPU,
        output=output,
    )

    assert artifact.manifest.status is ArtifactStatus.COMPLETE
    assert artifact is completed_views[0]
    assert events.count("actor-exposed:cpu") == 1
    assert sum(event.startswith("evaluate:") for event in events) == expected_rollouts
    assert events.index("begin") < events.index("train:cpu")
    assert events.index("train:cpu") < events.index("publish:training-config.json")
    assert events.index("publish:training-config.json") < events.index(
        "publish:training-summary.json"
    )
    assert events.index("publish:training-summary.json") < events.index("save")
    core_pending = next(
        index
        for index, event in enumerate(events)
        if event.startswith("pending:") and event.count(",") == 2
    )
    assert events.index("save") < core_pending < events.index("load-request:cpu")
    assert events.index("validate:3") < events.index("actor-exposed:cpu")
    assert events.index("actor-exposed:cpu") < next(
        index for index, event in enumerate(events) if event.startswith("evaluate:")
    )
    assert max(
        index for index, event in enumerate(events) if event.startswith("evaluate:")
    ) < events.index(f"publish:{expected_evaluations[0]}")
    full_size = len(required_payload_names(TrainerKind.PPO, profile))
    assert events.index(f"validate:{full_size}") < events.index("complete")
    assert events[-1] == "complete"

    assert tuple(
        record.relative_path for record in artifact.manifest.files
    ) == required_payload_names(TrainerKind.PPO, profile)
    assert artifact.manifest.criterion_eligible is (profile is ProfileName.CHECKPOINT)
    assert artifact.manifest.criterion_met is None
    assert artifact.manifest.criterion_status is (
        CriterionStatus.ELIGIBLE_FOR_AGGREGATE
        if profile is ProfileName.CHECKPOINT
        else CriterionStatus.INELIGIBLE
    )
    config = TrainingConfigDocument.from_document(artifact.document("training-config.json"))
    assert config.bc_training_digest_sha256 is None
    assert config.bc_validation_digest_sha256 is None
    summary = TrainingSummaryDocument.from_document(artifact.document("training-summary.json"))
    assert isinstance(summary.summary, PPOTrainingSummary)
    assert (
        summary.summary.requested_environment_steps
        == summary.summary.completed_environment_steps
        == PROFILE_CONFIGS[profile].ppo.total_timesteps
    )


@pytest.mark.parametrize(
    "boundary",
    [
        "training-env",
        "train",
        "publish-config",
        "publish-summary",
        "save",
        "pending-core",
        "load",
        "evaluate",
        "publish-evaluation",
        "pending-full",
        "validate-full",
        "complete",
    ],
)
@pytest.mark.filterwarnings("ignore:CUDA initialization:UserWarning")
def test_every_post_bootstrap_failure_keeps_the_artifact_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    """Catch any failure path that prematurely publishes a completion claim."""

    output = tmp_path / boundary
    events: list[str] = []
    _install_fast_ppo_workflow_fakes(monkeypatch, events)

    def injected(*args, **kwargs):
        del args, kwargs
        raise RuntimeError(f"injected {boundary}")

    if boundary == "training-env":
        monkeypatch.setattr(ppo_module, "make_ppo_vec_env", injected)
    elif boundary == "train":
        monkeypatch.setattr(ppo_module, "train_ppo", injected)
    elif boundary in {"publish-config", "publish-summary", "publish-evaluation"}:
        real_publish = ArtifactWriter.publish_json

        def fail_publish(writer: ArtifactWriter, filename: str, document: object):
            matches = {
                "publish-config": filename == "training-config.json",
                "publish-summary": filename == "training-summary.json",
                "publish-evaluation": filename.startswith("evaluation-"),
            }[boundary]
            if matches:
                return injected()
            return real_publish(writer, filename, document)  # type: ignore[arg-type]

        monkeypatch.setattr(ArtifactWriter, "publish_json", fail_publish)
    elif boundary == "save":
        monkeypatch.setattr(ppo_module, "save_ppo_model", injected)
    elif boundary in {"pending-core", "pending-full"}:
        real_pending = ArtifactWriter.pending_view

        def fail_pending(
            writer: ArtifactWriter,
            names: Sequence[str],
        ) -> PendingArtifactView:
            is_full = len(tuple(names)) > len(ppo_module._PPO_CORE_PAYLOADS)
            if is_full == (boundary == "pending-full"):
                return injected()  # type: ignore[return-value]
            return real_pending(writer, names)

        monkeypatch.setattr(ArtifactWriter, "pending_view", fail_pending)
    elif boundary == "load":
        monkeypatch.setattr(ppo_module, "load_ppo_actor", injected)
    elif boundary == "evaluate":
        monkeypatch.setattr(ppo_module, "evaluate_learned_actor", injected)
    elif boundary == "validate-full":
        real_validate = ppo_module.validate_ppo_artifact

        def fail_validate(artifact) -> None:
            if isinstance(artifact, PendingArtifactView) and len(artifact.files) > len(
                ppo_module._PPO_CORE_PAYLOADS
            ):
                injected()
            real_validate(artifact)

        monkeypatch.setattr(ppo_module, "validate_ppo_artifact", fail_validate)
    else:
        monkeypatch.setattr(ArtifactWriter, "complete", injected)

    with pytest.raises(RuntimeError, match=f"injected {boundary}"):
        train_ppo_artifact(
            profile=ProfileName.SMOKE,
            seed=0,
            device=DeviceName.CPU,
            output=output,
        )

    manifest = ArtifactManifest.from_document(read_json_document(output / "manifest.json"))
    assert manifest.status is ArtifactStatus.INCOMPLETE
    assert manifest.completed_at_utc is None
    assert manifest.files == ()
    assert manifest.criterion_eligible is False
    assert manifest.criterion_met is None
    with pytest.raises(ValueError, match="incomplete"):
        load_artifact(output)


@pytest.mark.parametrize(
    ("profile", "seed", "device", "dirty_source", "eligible"),
    [
        (ProfileName.SMOKE, 0, DeviceName.CPU, False, False),
        (ProfileName.CHECKPOINT, 5, DeviceName.CPU, False, False),
        (ProfileName.CHECKPOINT, 0, DeviceName.CUDA, False, False),
        (ProfileName.CHECKPOINT, 0, DeviceName.CPU, True, False),
        (ProfileName.CHECKPOINT, 4, DeviceName.CPU, False, True),
    ],
)
@pytest.mark.filterwarnings("ignore:CUDA initialization:UserWarning")
def test_profile_device_source_and_seed_eligibility_never_store_the_aggregate_criterion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: ProfileName,
    seed: int,
    device: DeviceName,
    dirty_source: bool,
    eligible: bool,
) -> None:
    """Catch exploratory artifacts claiming or persisting the five-seed PPO result."""

    events: list[str] = []
    _install_fast_ppo_workflow_fakes(monkeypatch, events)
    if dirty_source:
        monkeypatch.setattr(ppo_module, "capture_source_status", lambda path: _source_status())
    if device is DeviceName.CUDA:
        monkeypatch.setattr(ppo_module.torch.cuda, "is_available", lambda: True)
    actor_devices: list[DeviceName] = []
    real_load_actor = ppo_module.load_ppo_actor

    def load_actor(artifact, *, device: DeviceName = DeviceName.CPU) -> PPOActor:
        actor_devices.append(device)
        return real_load_actor(artifact, device=device)

    monkeypatch.setattr(ppo_module, "load_ppo_actor", load_actor)
    artifact = train_ppo_artifact(
        profile=profile,
        seed=seed,
        device=device,
        output=tmp_path / f"{profile.value}-{seed}-{device.value}-{dirty_source}",
    )

    assert artifact.manifest.status is ArtifactStatus.COMPLETE
    assert artifact.manifest.runtime.device is device
    assert artifact.manifest.evaluation_device is DeviceName.CPU
    assert actor_devices == [DeviceName.CPU]
    assert artifact.manifest.criterion_eligible is eligible
    assert artifact.manifest.criterion_met is None
    assert artifact.manifest.criterion_status is (
        CriterionStatus.ELIGIBLE_FOR_AGGREGATE if eligible else CriterionStatus.INELIGIBLE
    )
    for record in artifact.manifest.files:
        if record.relative_path.startswith("evaluation-"):
            document = artifact.document(record.relative_path)
            assert "criterion" not in document
            assert document["next_action_accuracy"] is None


def test_train_ppo_artifact_rejects_inputs_before_source_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch expensive/provenance work beginning before closed input validation."""

    captured = False

    def capture(path: Path) -> SourceStatus:
        nonlocal captured
        del path
        captured = True
        return _clean_source_status()

    monkeypatch.setattr(ppo_module, "capture_source_status", capture)
    monkeypatch.setattr(ppo_module.torch.cuda, "is_available", lambda: False)
    existing = tmp_path / "existing"
    existing.mkdir()
    invalid_calls = (
        {"profile": "smoke", "seed": 0, "device": DeviceName.CPU, "output": tmp_path / "a"},
        {
            "profile": ProfileName.SMOKE,
            "seed": True,
            "device": DeviceName.CPU,
            "output": tmp_path / "b",
        },
        {
            "profile": ProfileName.SMOKE,
            "seed": -1,
            "device": DeviceName.CPU,
            "output": tmp_path / "c",
        },
        {"profile": ProfileName.SMOKE, "seed": 0, "device": "cpu", "output": tmp_path / "d"},
        {"profile": ProfileName.SMOKE, "seed": 0, "device": DeviceName.CPU, "output": "e"},
        {
            "profile": ProfileName.SMOKE,
            "seed": 0,
            "device": DeviceName.CPU,
            "output": existing,
        },
        {
            "profile": ProfileName.SMOKE,
            "seed": 0,
            "device": DeviceName.CUDA,
            "output": tmp_path / "cuda",
        },
    )
    for arguments in invalid_calls:
        with pytest.raises((ValueError, FileExistsError)):
            train_ppo_artifact(**arguments)  # type: ignore[arg-type]
    assert captured is False


@pytest.mark.filterwarnings("ignore:CUDA initialization:UserWarning")
def test_real_smoke_artifact_trains_saves_reloads_evaluates_and_repeats_actions(
    tmp_path: Path,
) -> None:
    """Catch integration drift hidden by workflow fakes across the real smoke profile."""

    artifact = train_ppo_artifact(
        profile=ProfileName.SMOKE,
        seed=0,
        device=DeviceName.CPU,
        output=tmp_path / "real-smoke",
    )

    assert artifact.manifest.status is ArtifactStatus.COMPLETE
    assert artifact.manifest.runtime.device is DeviceName.CPU
    assert artifact.manifest.evaluation_device is DeviceName.CPU
    assert artifact.manifest.criterion_eligible is False
    assert artifact.manifest.criterion_met is None
    assert isinstance(artifact.manifest.training_counts, PPOTrainingCounts)
    assert (
        artifact.manifest.training_counts.requested_environment_steps
        == artifact.manifest.training_counts.completed_environment_steps
        == PROFILE_CONFIGS[ProfileName.SMOKE].ppo.total_timesteps
    )
    assert artifact.file("model.zip").is_file()
    evaluation = EvaluationFile.from_document(artifact.document("evaluation-smoke.json"))
    assert tuple((row.subset, row.probe) for row in evaluation.rows) == (
        ("combined", None),
        ("combined", ZERO_SPECTRUM_PROBE),
        ("combined", SHUFFLED_SPECTRUM_PROBE),
    )
    assert evaluation.next_action_accuracy is None

    first_actor = load_ppo_actor(artifact)
    second_actor = load_ppo_actor(artifact)
    env = SinePitchEnv()
    try:
        observation, _ = env.reset(options={"target_note_index": 12, "source_pitch_cents": 6_137})
    finally:
        env.close()
    assert first_actor.act(observation) is second_actor.act(observation)
