"""Narrow learned-actor inference boundary contracts."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from harpy.envs.models import PitchAction
from harpy.envs.sine_pitch import SinePitchEnv
from harpy.learning.actors import Actor, BCActor, PPOActor
from harpy.learning.errors import LearningContractError, LearningExecutionError
from harpy.learning.network import HarpySineFeaturesExtractor
from harpy.learning.observations import PolicyObservationWrapper


def valid_raw_observation() -> dict[str, object]:
    return {
        "spectrum": np.linspace(0, 1, 1_961, dtype=np.float32),
        "target_note": np.int64(24),
        "controls": np.array([-2, 12, -100], dtype=np.int16),
        "steps_remaining": np.int64(32),
    }


class ConstantLogitBC(torch.nn.Module):
    """A real minimal network whose intentionally tied logits expose argmax ordering."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)
        self.last_observation: dict[str, torch.Tensor] | None = None

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        self.last_observation = observations
        if self.logits.ndim != 1:
            return self.logits
        return self.logits.expand(observations["state"].shape[0], -1)


class FixedPredictionPPO:
    """A minimal prediction boundary that returns a controlled SB3-shaped action."""

    def __init__(self, action: object) -> None:
        self.action = action
        self.observations: list[dict[str, np.ndarray]] = []
        self.deterministic_values: list[bool] = []

    def predict(
        self, observation: dict[str, np.ndarray], *, deterministic: bool
    ) -> tuple[object, None]:
        self.observations.append(observation)
        self.deterministic_values.append(deterministic)
        return self.action, None


def test_actor_protocol_and_learned_actor_signatures_accept_observation_only() -> None:
    assert tuple(inspect.signature(Actor.act).parameters) == ("self", "observation")
    assert tuple(inspect.signature(BCActor.act).parameters) == ("self", "observation")
    assert tuple(inspect.signature(PPOActor.act).parameters) == ("self", "observation")


def test_bc_actor_uses_first_argmax_on_a_tie() -> None:
    model = ConstantLogitBC(torch.zeros(7))
    actor = BCActor(model, device=torch.device("cpu"))

    assert isinstance(actor, Actor)
    assert actor.act(valid_raw_observation()) is PitchAction.OCTAVE_DOWN
    assert model.last_observation is not None
    assert tuple(model.last_observation) == ("spectrum", "state")
    assert model.last_observation["spectrum"].shape == (1, 1_961)
    assert model.last_observation["state"].shape == (1, 5)
    assert model.last_observation["spectrum"].dtype is torch.float32
    assert model.logits.device.type == "cpu"
    assert not model.training


def test_bc_actor_validates_raw_observations_before_model_inference() -> None:
    raw = valid_raw_observation()
    raw["source_pitch_cents"] = 6_000
    actor = BCActor(ConstantLogitBC(torch.zeros(7)))

    with pytest.raises(LearningContractError, match="exactly"):
        actor.act(raw)


@pytest.mark.parametrize(
    "logits",
    [
        torch.zeros(6),
        torch.zeros(8),
        torch.zeros((1, 7, 1)),
        torch.full((7,), float("nan")),
    ],
)
def test_bc_actor_rejects_malformed_model_logits(logits: torch.Tensor) -> None:
    actor = BCActor(ConstantLogitBC(logits))

    with pytest.raises(LearningExecutionError, match="logits"):
        actor.act(valid_raw_observation())


def test_ppo_actor_passes_the_transformed_observation_and_uses_deterministic_prediction() -> None:
    model = FixedPredictionPPO(np.array(6, dtype=np.int64))
    actor = PPOActor(model)

    assert isinstance(actor, Actor)
    assert actor.act(valid_raw_observation()) is PitchAction.OCTAVE_UP
    assert model.deterministic_values == [True]
    assert tuple(model.observations[0]) == ("spectrum", "state")
    assert model.observations[0]["spectrum"].shape == (1_961,)
    assert model.observations[0]["state"].shape == (5,)
    assert model.observations[0]["spectrum"].dtype == np.float32


@pytest.mark.parametrize(
    "action",
    [
        np.array([0], dtype=np.int64),
        np.array([[0]], dtype=np.int64),
        np.array(0.0, dtype=np.float32),
        np.array(True),
        np.array(-1, dtype=np.int64),
        np.array(7, dtype=np.int64),
        np.int64(0),
        0,
        False,
    ],
)
def test_ppo_actor_rejects_non_scalar_or_invalid_prediction_actions(action: object) -> None:
    actor = PPOActor(FixedPredictionPPO(action))

    with pytest.raises(LearningExecutionError, match="action"):
        actor.act(valid_raw_observation())


def test_ppo_actor_accepts_an_actual_sb3_unvectorized_discrete_prediction() -> None:
    environment = PolicyObservationWrapper(SinePitchEnv())
    model = PPO(
        "MultiInputPolicy",
        environment,
        device="cpu",
        n_steps=2,
        batch_size=2,
        policy_kwargs={
            "features_extractor_class": HarpySineFeaturesExtractor,
            "net_arch": [],
        },
    )
    actor = PPOActor(model)

    action = actor.act(valid_raw_observation())

    assert isinstance(action, PitchAction)
    assert model.device.type == "cpu"
