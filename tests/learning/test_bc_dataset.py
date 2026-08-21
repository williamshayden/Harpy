"""Compact oracle-trajectory dataset contracts for behavior cloning."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import gymnasium
import numpy as np
import pytest
import torch

from harpy.envs.models import MAX_STEPS, ControlState, PitchAction
from harpy.envs.planning import minimum_action_plan
from harpy.envs.spectrum import LOG_SPECTRUM_SIZE
from harpy.learning.bc import (
    BCExample,
    OracleTrajectoryDataset,
    build_oracle_examples,
    training_class_weights,
)
from harpy.learning.cache import SpectrumEvidenceCache
from harpy.learning.envs import make_cached_sine_pitch_env
from harpy.learning.errors import LearningContractError
from harpy.learning.models import EpisodeSpec


class _TrackingEnv(gymnasium.Wrapper):
    def __init__(
        self,
        env: gymnasium.Env,
        *,
        corrupt_reset: bool = False,
        corrupt_submit: bool = False,
    ) -> None:
        super().__init__(env)
        self.closed = False
        self._corrupt_reset = corrupt_reset
        self._corrupt_submit = corrupt_submit

    def reset(self, **kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        observation, info = self.env.reset(**kwargs)
        if not self._corrupt_reset:
            return observation, info
        corrupted = dict(observation)
        corrupted["steps_remaining"] = np.int64(MAX_STEPS - 1)
        return corrupted, info

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        if not self._corrupt_submit or PitchAction(action) is not PitchAction.SUBMIT:
            return observation, reward, terminated, truncated, info
        corrupted = dict(info)
        corrupted["submitted_success"] = False
        return observation, reward, terminated, truncated, corrupted

    def close(self) -> None:
        self.closed = True
        super().close()


class _FailingResetEnv(_TrackingEnv):
    def reset(self, **kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        del kwargs
        raise RuntimeError("injected reset failure")


class _EvidenceProvider:
    def __init__(self, spectrum: object | None = None) -> None:
        self.calls: list[int] = []
        self.spectrum = (
            np.linspace(0.0, 1.0, LOG_SPECTRUM_SIZE, dtype=np.float32)
            if spectrum is None
            else spectrum
        )

    def spectrum_for_cents(self, effective_pitch_cents: int) -> object:
        self.calls.append(effective_pitch_cents)
        return self.spectrum


def _example(
    action: PitchAction = PitchAction.SUBMIT,
    *,
    source_pitch_cents: int = 5_151,
    controls: ControlState | None = None,
    steps_remaining: int = 17,
) -> BCExample:
    return BCExample(
        episode=EpisodeSpec(target_note_index=12, source_pitch_cents=source_pitch_cents),
        controls=(ControlState(octaves=1, semitones=-2, cents=3) if controls is None else controls),
        steps_remaining=steps_remaining,
        action=action,
    )


def test_examples_are_real_minimum_plan_trajectories() -> None:
    episode = EpisodeSpec(target_note_index=12, source_pitch_cents=5_151)
    cache = SpectrumEvidenceCache()

    examples = build_oracle_examples(
        (episode,),
        env_factory=lambda: make_cached_sine_pitch_env(cache),
    )

    assert examples[-1].action is PitchAction.SUBMIT
    assert tuple(example.action for example in examples) == minimum_action_plan(
        5_151 - 6_000,
        ControlState(),
        tolerance_cents=5,
    )
    assert all(not hasattr(example, "spectrum") for example in examples)
    assert all(not hasattr(example, "error_cents") for example in examples)


def test_examples_pin_pre_action_state_order_budget_and_terminal_boundary() -> None:
    episodes = (
        EpisodeSpec(target_note_index=0, source_pitch_cents=4_806),
        EpisodeSpec(target_note_index=24, source_pitch_cents=7_194),
    )
    envs: list[_TrackingEnv] = []

    def factory() -> gymnasium.Env:
        env = _TrackingEnv(make_cached_sine_pitch_env(SpectrumEvidenceCache()))
        envs.append(env)
        return env

    examples = build_oracle_examples(episodes, env_factory=factory)

    assert len(envs) == 1
    assert envs[0].closed
    assert tuple(example.episode for example in examples) == (
        episodes[0],
        episodes[0],
        episodes[1],
        episodes[1],
    )
    assert tuple(example.controls for example in examples) == (
        ControlState(),
        ControlState(cents=-1),
        ControlState(),
        ControlState(cents=1),
    )
    assert tuple(example.steps_remaining for example in examples) == (
        MAX_STEPS,
        MAX_STEPS - 1,
        MAX_STEPS,
        MAX_STEPS - 1,
    )
    assert tuple(example.action for example in examples) == (
        PitchAction.CENT_DOWN,
        PitchAction.SUBMIT,
        PitchAction.CENT_UP,
        PitchAction.SUBMIT,
    )


def test_examples_are_frozen_slotted_records() -> None:
    example = _example()

    with pytest.raises(FrozenInstanceError):
        example.action = PitchAction.CENT_UP  # type: ignore[misc]
    assert not hasattr(example, "__dict__")


def test_example_builder_closes_environment_after_failure_and_rejects_mismatch() -> None:
    episode = EpisodeSpec(target_note_index=0, source_pitch_cents=4_806)
    failing = _FailingResetEnv(make_cached_sine_pitch_env(SpectrumEvidenceCache()))

    with pytest.raises(RuntimeError, match="injected reset failure"):
        build_oracle_examples((episode,), env_factory=lambda: failing)
    assert failing.closed

    corrupt = _TrackingEnv(
        make_cached_sine_pitch_env(SpectrumEvidenceCache()),
        corrupt_reset=True,
    )
    with pytest.raises(RuntimeError, match="trajectory mismatch"):
        build_oracle_examples((episode,), env_factory=lambda: corrupt)
    assert corrupt.closed

    failed_submit = _TrackingEnv(
        make_cached_sine_pitch_env(SpectrumEvidenceCache()),
        corrupt_submit=True,
    )
    with pytest.raises(RuntimeError, match="trajectory mismatch"):
        build_oracle_examples((episode,), env_factory=lambda: failed_submit)
    assert failed_submit.closed


def test_dataset_is_lazy_reconstructs_actor_visible_input_and_returns_owned_arrays() -> None:
    example = _example()
    provider = _EvidenceProvider()

    dataset = OracleTrajectoryDataset((example,), provider)  # type: ignore[arg-type]

    assert provider.calls == []
    observation, label = dataset[0]
    assert provider.calls == [6_154]
    assert type(label) is np.int64
    assert label == np.int64(PitchAction.SUBMIT)
    np.testing.assert_array_equal(
        observation["state"],
        np.array([0.0, 0.5, -2 / 12, 0.03, 17 / MAX_STEPS], dtype=np.float32),
    )
    np.testing.assert_array_equal(observation["spectrum"], provider.spectrum)

    observation["spectrum"].fill(0.0)
    observation["state"].fill(0.0)
    second, _ = dataset[0]
    assert provider.calls == [6_154, 6_154]
    assert np.count_nonzero(second["spectrum"]) > 0
    assert np.count_nonzero(second["state"]) > 0
    assert not np.shares_memory(observation["spectrum"], second["spectrum"])
    assert not np.shares_memory(observation["state"], second["state"])


@pytest.mark.parametrize(
    "spectrum",
    [
        np.zeros(LOG_SPECTRUM_SIZE, dtype=np.float64),
        np.zeros(LOG_SPECTRUM_SIZE - 1, dtype=np.float32),
        np.full(LOG_SPECTRUM_SIZE, np.nan, dtype=np.float32),
    ],
)
def test_dataset_rejects_malformed_evidence(spectrum: np.ndarray[Any, Any]) -> None:
    dataset = OracleTrajectoryDataset(
        (_example(),),
        _EvidenceProvider(spectrum),  # type: ignore[arg-type]
    )

    with pytest.raises(LearningContractError, match="spectrum"):
        dataset[0]


def test_dataset_rejects_empty_examples_invalid_records_and_bool_indices() -> None:
    provider = _EvidenceProvider()

    with pytest.raises(ValueError, match="examples"):
        OracleTrajectoryDataset((), provider)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="BCExample"):
        OracleTrajectoryDataset((_example(), object()), provider)  # type: ignore[arg-type]

    dataset = OracleTrajectoryDataset((_example(),), provider)  # type: ignore[arg-type]
    with pytest.raises((IndexError, ValueError), match="index"):
        dataset[True]
    with pytest.raises(IndexError):
        dataset[1]


def test_training_class_weights_use_all_seven_training_counts_only() -> None:
    actions = tuple(
        action
        for action, count in zip(PitchAction, (1, 2, 3, 4, 5, 6, 7), strict=True)
        for _ in range(count)
    )
    weights = training_class_weights(actions)

    assert weights.dtype is torch.float32
    assert weights.device == torch.device("cpu")
    torch.testing.assert_close(
        weights,
        torch.tensor(
            [4.0, 2.0, 4 / 3, 1.0, 0.8, 2 / 3, 4 / 7],
            dtype=torch.float32,
        ),
        rtol=1e-6,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "actions",
    [
        (),
        tuple(PitchAction)[:-1],
        (*tuple(PitchAction), 0),
    ],
)
def test_training_class_weights_reject_empty_missing_or_non_action_labels(
    actions: tuple[object, ...],
) -> None:
    with pytest.raises(ValueError, match=r"actions|class"):
        training_class_weights(actions)  # type: ignore[arg-type]
