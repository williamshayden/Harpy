"""Stateless learned-policy inference adapters over actor-safe observations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np

from harpy.envs.models import PitchAction
from harpy.learning.dependencies import require_training_dependencies
from harpy.learning.errors import LearningContractError, LearningExecutionError
from harpy.learning.observations import preprocess_observation


@runtime_checkable
class Actor(Protocol):
    """A stateless policy that selects one pitch action from one raw observation."""

    def act(self, observation: Mapping[str, object]) -> PitchAction:
        """Return the next action using actor-safe current observation data only."""


class BCActor:
    """Adapt a behavior-cloning PyTorch classifier to the narrow actor protocol."""

    def __init__(self, model: object, *, device: object | None = None) -> None:
        training_stack = require_training_dependencies()
        self._torch = training_stack.torch
        if not isinstance(model, self._torch.nn.Module):
            raise LearningContractError("BC model must be a torch.nn.Module")
        self._device = self._torch.device("cpu" if device is None else device)
        self._model = model.to(self._device)
        self._model.eval()

    def act(self, observation: Mapping[str, object]) -> PitchAction:
        """Select the first maximum of seven BC logits for one raw observation."""
        policy_observation = preprocess_observation(observation)
        tensors = {
            "spectrum": self._torch.from_numpy(policy_observation["spectrum"])
            .unsqueeze(0)
            .to(self._device),
            "state": self._torch.from_numpy(policy_observation["state"])
            .unsqueeze(0)
            .to(self._device),
        }
        with self._torch.inference_mode():
            logits = self._model(tensors)
        if not isinstance(logits, self._torch.Tensor) or logits.shape != (1, len(PitchAction)):
            raise LearningExecutionError("BC model must return logits with shape (1, 7)")
        if not self._torch.isfinite(logits).all():
            raise LearningExecutionError("BC model logits must contain only finite values")
        return PitchAction(int(self._torch.argmax(logits, dim=1).item()))


class PPOActor:
    """Adapt an SB3 PPO model to the narrow actor protocol."""

    def __init__(self, model: object) -> None:
        if not callable(getattr(model, "predict", None)):
            raise LearningContractError("PPO model must provide a callable predict method")
        self._model = model

    def act(self, observation: Mapping[str, object]) -> PitchAction:
        """Request a deterministic SB3 action and validate its unvectorized result."""
        policy_observation = preprocess_observation(observation)
        prediction = self._model.predict(policy_observation, deterministic=True)
        if not isinstance(prediction, tuple) or len(prediction) != 2:
            raise LearningExecutionError("PPO predict must return an action and state tuple")
        return _ppo_action(prediction[0])


def _ppo_action(value: object) -> PitchAction:
    if not isinstance(value, np.ndarray) or value.ndim != 0:
        raise LearningExecutionError("PPO action must be a zero-dimensional integer NumPy array")
    if not np.issubdtype(value.dtype, np.integer) or np.issubdtype(value.dtype, np.bool_):
        raise LearningExecutionError("PPO action must be an integer NumPy array")
    action = value.item()
    if isinstance(action, bool) or not isinstance(action, int):
        raise LearningExecutionError("PPO action must be an integer")
    if not 0 <= action < len(PitchAction):
        raise LearningExecutionError("PPO action must be within 0..6")
    return PitchAction(action)


__all__ = ["Actor", "BCActor", "PPOActor"]
