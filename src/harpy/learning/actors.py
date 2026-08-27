"""Stateless learned-policy inference adapters over actor-safe observations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np

from harpy.envs.models import ControlState, PitchAction
from harpy.learning.action_masks import legal_action_mask
from harpy.learning.dependencies import require_training_dependencies
from harpy.learning.errors import LearningContractError, LearningExecutionError
from harpy.learning.observations import preprocess_observation

BC_ACTOR_SEMANTICS_ID = "harpy-sine-policy-bc-v1"
MASKED_BC_ACTOR_SEMANTICS_ID = "harpy-sine-policy-bc-public-bound-mask-v1"
PPO_ACTOR_SEMANTICS_ID = "harpy-sine-policy-ppo-v1"
_RAW_OBSERVATION_KEYS = frozenset({"spectrum", "target_note", "controls", "steps_remaining"})


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
        logits = self._validated_logits(observation)
        return PitchAction(int(self._torch.argmax(logits, dim=1).item()))

    def _validated_logits(self, observation: Mapping[str, object]) -> object:
        """Validate raw input and return one finite row of persisted-model logits."""
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
        return logits


class MaskedBCActor(BCActor):
    """Diagnostic BC adapter that removes only publicly bound-blocked actions."""

    def act(self, observation: Mapping[str, object]) -> PitchAction:
        """Mask illegal finite logits, then select their stable first maximum."""
        snapshot = _owned_raw_observation_snapshot(observation)
        logits = self._validated_logits(snapshot)
        if not self._torch.is_floating_point(logits):
            raise LearningExecutionError("BC model logits must use a floating-point dtype")
        raw_controls = snapshot["controls"]
        if not isinstance(raw_controls, np.ndarray):  # validated above; narrows typing
            raise LearningContractError("controls must be an int16 ndarray")
        controls = ControlState(*(int(value) for value in raw_controls))
        legal = legal_action_mask(controls)
        if not legal[PitchAction.SUBMIT]:
            raise LearningExecutionError("Submit must remain legal in the BC action mask")
        mask = self._torch.tensor(
            legal,
            dtype=self._torch.bool,
            device=logits.device,
        ).unsqueeze(0)
        masked_logits = logits.masked_fill(~mask, -self._torch.inf)
        return PitchAction(int(self._torch.argmax(masked_logits, dim=1).item()))


def _owned_raw_observation_snapshot(
    observation: Mapping[str, object],
) -> dict[str, object]:
    """Read raw fields once and own mutable arrays for masked-actor consistency."""
    if not isinstance(observation, Mapping):
        raise LearningContractError("observation must be a mapping")
    if set(observation) != _RAW_OBSERVATION_KEYS:
        raise LearningContractError(
            "observation must contain exactly spectrum, target_note, controls, and steps_remaining"
        )
    snapshot = {
        "spectrum": observation["spectrum"],
        "target_note": observation["target_note"],
        "controls": observation["controls"],
        "steps_remaining": observation["steps_remaining"],
    }
    for field in ("spectrum", "controls"):
        value = snapshot[field]
        if isinstance(value, np.ndarray):
            snapshot[field] = np.array(value, copy=True, order="C")
    return snapshot


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


__all__ = [
    "BC_ACTOR_SEMANTICS_ID",
    "MASKED_BC_ACTOR_SEMANTICS_ID",
    "PPO_ACTOR_SEMANTICS_ID",
    "Actor",
    "BCActor",
    "MaskedBCActor",
    "PPOActor",
]
