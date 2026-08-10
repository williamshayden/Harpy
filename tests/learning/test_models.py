"""Contract tests for immutable learned-policy models."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import numpy as np
import pytest

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
    PPOProfile,
    PPOTrainingSummary,
    ProfileName,
    TrainerKind,
)


def test_episode_spec_owns_only_exact_reset_truth() -> None:
    episode = EpisodeSpec(target_note_index=7, source_pitch_cents=5_432)

    assert episode.pair == (7, 5_432)
    first = episode.reset_options()
    second = episode.reset_options()
    assert first == {"target_note_index": 7, "source_pitch_cents": 5_432}
    assert first is not second


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_note_index", True),
        ("target_note_index", 25),
        ("source_pitch_cents", False),
        ("source_pitch_cents", 4_799),
    ],
)
def test_episode_spec_rejects_invalid_integer_truth(field: str, value: object) -> None:
    values = {"target_note_index": 0, "source_pitch_cents": 4_800, field: value}

    with pytest.raises(ValueError, match=field):
        EpisodeSpec(**values)


def test_episode_spec_normalizes_index_scalars_to_owned_python_integers() -> None:
    episode = EpisodeSpec(target_note_index=np.int64(7), source_pitch_cents=np.int64(5_432))

    assert type(episode.target_note_index) is int
    assert type(episode.source_pitch_cents) is int


def test_profiles_pin_rollout_aligned_values() -> None:
    smoke = PROFILE_CONFIGS[ProfileName.SMOKE]
    checkpoint = PROFILE_CONFIGS[ProfileName.CHECKPOINT]

    assert (smoke.bc.train_episodes, smoke.bc.validation_episodes) == (128, 64)
    assert (checkpoint.bc.train_episodes, checkpoint.bc.validation_episodes) == (4_096, 512)
    assert (smoke.ppo.total_timesteps, smoke.ppo.n_steps, smoke.ppo.batch_size) == (
        2_048,
        256,
        64,
    )
    assert (
        checkpoint.ppo.total_timesteps,
        checkpoint.ppo.n_steps,
        checkpoint.ppo.batch_size,
    ) == (256_000, 1_024, 256)
    assert checkpoint.ppo.total_timesteps % checkpoint.ppo.n_steps == 0
    assert checkpoint.ppo.gamma == 1.0


def test_profiles_pin_all_checked_in_optimizer_and_evaluation_values() -> None:
    smoke = PROFILE_CONFIGS[ProfileName.SMOKE]
    checkpoint = PROFILE_CONFIGS[ProfileName.CHECKPOINT]

    assert (
        smoke.bc.max_epochs,
        smoke.bc.early_stopping_patience,
        smoke.bc.batch_size,
        smoke.bc.optimizer,
        smoke.bc.learning_rate,
        smoke.bc.weight_decay,
    ) == (2, None, 128, "AdamW", 3e-4, 1e-4)
    assert (
        checkpoint.bc.max_epochs,
        checkpoint.bc.early_stopping_patience,
        checkpoint.bc.batch_size,
        checkpoint.bc.optimizer,
        checkpoint.bc.learning_rate,
        checkpoint.bc.weight_decay,
    ) == (50, 5, 256, "AdamW", 3e-4, 1e-4)
    for profile in (smoke, checkpoint):
        assert (
            profile.ppo.n_epochs,
            profile.ppo.learning_rate,
            profile.ppo.gae_lambda,
            profile.ppo.clip_range,
            profile.ppo.ent_coef,
            profile.ppo.vf_coef,
        ) == (10, 3e-4, 0.95, 0.2, 0.01, 0.5)
    assert smoke.evaluation_suites == (EvaluationSuiteId.SMOKE,)
    assert checkpoint.evaluation_suites == (
        EvaluationSuiteId.IID,
        EvaluationSuiteId.REGISTER_OOD,
    )


def test_closed_enums_and_schema_identifiers_are_exact() -> None:
    assert [(member.name, member.value) for member in ProfileName] == [
        ("SMOKE", "smoke"),
        ("CHECKPOINT", "checkpoint"),
    ]
    assert [(member.name, member.value) for member in TrainerKind] == [
        ("BC", "bc"),
        ("PPO", "ppo"),
    ]
    assert [(member.name, member.value) for member in DeviceName] == [
        ("CPU", "cpu"),
        ("CUDA", "cuda"),
    ]
    assert [(member.name, member.value) for member in EvaluationSuiteId] == [
        ("SMOKE", "harpy-sine-policy-eval-smoke-v1"),
        ("IID", "harpy-sine-policy-eval-iid-v1"),
        ("REGISTER_OOD", "harpy-sine-policy-eval-register-ood-v1"),
    ]
    assert (
        ENVIRONMENT_ID,
        ENVIRONMENT_CONTRACT_ID,
        SPECTRUM_GRID_ID,
        PREPROCESSING_SCHEMA_ID,
        ARCHITECTURE_SCHEMA_ID,
    ) == (
        "Harpy/SinePitch-v0",
        "harpy-sine-pitch-v0-contract-v1",
        "harpy-sine-spectrum-grid-v1",
        "harpy-sine-policy-observation-v1",
        "harpy-sine-policy-conv-v1",
    )


def test_models_are_frozen_slots_and_profile_mapping_is_immutable() -> None:
    episode = EpisodeSpec(target_note_index=7, source_pitch_cents=5_432)

    with pytest.raises(FrozenInstanceError):
        episode.target_note_index = 8  # type: ignore[misc]
    assert not hasattr(episode, "__dict__")
    assert isinstance(PROFILE_CONFIGS, MappingProxyType)
    with pytest.raises(TypeError):
        PROFILE_CONFIGS[ProfileName.SMOKE] = PROFILE_CONFIGS[ProfileName.SMOKE]  # type: ignore[index]


@pytest.mark.parametrize("seed", [True, -1])
def test_seed_bearing_models_reject_boolean_and_negative_seeds(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        EpisodeSuite(
            schema_version=1,
            suite_id=EvaluationSuiteId.SMOKE,
            suite_seed=seed,
            episodes=(EpisodeSpec(0, 4_800),),
            digest_sha256="0" * 64,
        )


def test_bcepoch_metrics_and_training_summary_validate_finite_ordered_data() -> None:
    first = BCEpochMetrics(
        epoch=1,
        training_loss=1.0,
        validation_loss=0.8,
        validation_accuracy=0.5,
    )
    second = BCEpochMetrics(
        epoch=2,
        training_loss=0.5,
        validation_loss=0.4,
        validation_accuracy=0.75,
    )
    summary = BCTrainingSummary(
        history=(first, second),
        selected_epoch=2,
        training_examples=12,
        validation_examples=4,
        training_wall_time_seconds=1.5,
    )

    assert summary.history == (first, second)
    assert summary.selected_epoch == 2
    with pytest.raises(ValueError, match="history"):
        BCTrainingSummary(
            history=(second, first),
            selected_epoch=2,
            training_examples=12,
            validation_examples=4,
            training_wall_time_seconds=1.5,
        )
    with pytest.raises(ValueError, match="selected_epoch"):
        BCTrainingSummary(
            history=(first, second),
            selected_epoch=3,
            training_examples=12,
            validation_examples=4,
            training_wall_time_seconds=1.5,
        )


@pytest.mark.parametrize(
    ("factory", "field", "value"),
    [
        (BCEpochMetrics, "training_loss", float("nan")),
        (BCEpochMetrics, "validation_loss", -0.1),
        (BCEpochMetrics, "validation_accuracy", 1.1),
        (BCTrainingSummary, "training_wall_time_seconds", float("inf")),
        (PPOTrainingSummary, "training_wall_time_seconds", -0.1),
    ],
)
def test_summary_models_reject_invalid_finite_metrics(
    factory: type[object], field: str, value: object
) -> None:
    if factory is BCEpochMetrics:
        values: dict[str, object] = {
            "epoch": 1,
            "training_loss": 1.0,
            "validation_loss": 0.5,
            "validation_accuracy": 0.5,
        }
    elif factory is BCTrainingSummary:
        values = {
            "history": (BCEpochMetrics(1, 1.0, 0.5, 0.5),),
            "selected_epoch": 1,
            "training_examples": 1,
            "validation_examples": 1,
            "training_wall_time_seconds": 1.0,
        }
    else:
        values = {
            "requested_environment_steps": 1,
            "completed_environment_steps": 1,
            "training_wall_time_seconds": 1.0,
        }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        factory(**values)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    ("requested_environment_steps", "completed_environment_steps"),
    [(0, 0), (1, 0), (1, 2), (True, 1)],
)
def test_ppo_summary_requires_exact_positive_integer_step_counts(
    requested_environment_steps: object, completed_environment_steps: object
) -> None:
    with pytest.raises(ValueError):
        PPOTrainingSummary(
            requested_environment_steps=requested_environment_steps,
            completed_environment_steps=completed_environment_steps,
            training_wall_time_seconds=0.0,
        )


@pytest.mark.parametrize(
    ("profile_type", "values"),
    [
        (
            BCProfile,
            {
                "train_episodes": 1,
                "validation_episodes": 1,
                "max_epochs": 1,
                "early_stopping_patience": 0,
                "batch_size": 1,
                "optimizer": "SGD",
                "learning_rate": 1.0,
                "weight_decay": 0.0,
            },
        ),
        (
            PPOProfile,
            {
                "total_timesteps": 1,
                "n_steps": 2,
                "batch_size": 1,
                "n_epochs": 1,
                "learning_rate": 1.0,
                "gamma": 1.0,
                "gae_lambda": 1.0,
                "clip_range": 1.0,
                "ent_coef": 0.0,
                "vf_coef": 0.0,
            },
        ),
    ],
)
def test_profiles_reject_values_that_break_their_training_contract(
    profile_type: type[object], values: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        profile_type(**values)  # type: ignore[call-arg]
