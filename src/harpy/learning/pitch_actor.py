"""Stateless estimator-plus-planner actor for reliable sine tuning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from harpy.envs.models import (
    SOURCE_MAX_CENTS,
    SOURCE_MIN_CENTS,
    TARGET_MIN_COORDINATE,
    ControlState,
    PitchAction,
)
from harpy.envs.planning import minimum_action_plan
from harpy.learning.action_masks import legal_action_mask
from harpy.learning.dependencies import require_pitch_dependencies
from harpy.learning.errors import LearningContractError, LearningExecutionError
from harpy.learning.observations import preprocess_observation

_PITCH_GRID_MIN_CENTS = 1_100
_PITCH_GRID_MAX_CENTS = 10_900
_PITCH_GRID_STEP_CENTS = 5
_PITCH_CLASS_COUNT = 1_961
_BASE_ERROR_MIN_CENTS = -2_400
_BASE_ERROR_MAX_CENTS = 2_400
_RAW_OBSERVATION_KEYS = (
    "spectrum",
    "target_note",
    "controls",
    "steps_remaining",
)


class PitchDecoding(StrEnum):
    """Explicit estimator decoding; the released actor uses global argmax."""

    GLOBAL_ARGMAX = "global-argmax"
    FEASIBLE_ARGMAX = "feasible-argmax"

    @property
    def semantics_id(self) -> str:
        """Distinguish the released actor from the explicit decoding intervention."""
        if self is PitchDecoding.GLOBAL_ARGMAX:
            return "harpy-sine-pitch-estimator-planner-v1"
        return "harpy-sine-pitch-feasible-estimator-planner-v1"


@dataclass(frozen=True, slots=True)
class PitchDecision:
    """One typed estimator diagnosis paired with its chosen public action."""

    action: PitchAction
    estimated_candidate_cents: int

    def __post_init__(self) -> None:
        if not isinstance(self.action, PitchAction):
            raise ValueError("action must be a PitchAction")
        estimate = self.estimated_candidate_cents
        if isinstance(estimate, bool) or not isinstance(estimate, int):
            raise ValueError("estimated_candidate_cents must be an integer")
        if not _PITCH_GRID_MIN_CENTS <= estimate <= _PITCH_GRID_MAX_CENTS:
            raise ValueError(
                "estimated_candidate_cents must be within "
                f"{_PITCH_GRID_MIN_CENTS}..{_PITCH_GRID_MAX_CENTS}"
            )
        if (estimate - _PITCH_GRID_MIN_CENTS) % _PITCH_GRID_STEP_CENTS:
            raise ValueError("estimated_candidate_cents must lie on the five-cent grid")


class PitchPlannerActor:
    """Re-estimate and replan from every complete actor-visible observation."""

    def __init__(
        self,
        model: object,
        *,
        device: object | None = None,
        decoding: PitchDecoding = PitchDecoding.GLOBAL_ARGMAX,
    ) -> None:
        if not isinstance(decoding, PitchDecoding):
            raise LearningContractError("decoding must be a PitchDecoding")
        training_stack = require_pitch_dependencies()
        self._torch = training_stack.torch
        # Importing the Torch-defined network module remains lazy until actor construction.
        from harpy.learning.pitch_network import validate_pitch_estimator_model

        validate_pitch_estimator_model(model)
        requested_device = self._torch.device("cpu" if device is None else device)
        self._model = model.to(requested_device)
        self._device = next(self._model.parameters()).device
        self._model.eval()
        self._decoding = decoding

    @property
    def decoding(self) -> PitchDecoding:
        """Identify the explicitly selected estimator decoding behavior."""
        return self._decoding

    @property
    def semantics_id(self) -> str:
        """Identify the estimator-plus-planner behavior for comparison reports."""
        return self._decoding.semantics_id

    def act(self, observation: Mapping[str, object]) -> PitchAction:
        """Return only the chosen action through the existing narrow actor protocol."""
        return self.decide(observation).action

    def decide(self, observation: Mapping[str, object]) -> PitchDecision:
        """Estimate current pitch and execute the first fresh bounded plan action."""
        snapshot = _owned_observation_snapshot(observation)
        policy_observation = preprocess_observation(snapshot)
        spectrum = (
            self._torch.from_numpy(policy_observation["spectrum"]).unsqueeze(0).to(self._device)
        )
        with self._torch.inference_mode():
            logits = self._model(spectrum)
        raw_controls = snapshot["controls"]
        if not isinstance(raw_controls, np.ndarray):  # proved by preprocessing; narrows typing
            raise LearningContractError("controls must be an int16 ndarray")
        controls = ControlState(*(int(value) for value in raw_controls))
        estimated_candidate_cents = self._estimate_from_logits(logits, controls=controls)
        target_note = int(snapshot["target_note"])
        target_cents = TARGET_MIN_COORDINATE * 100 + 100 * target_note
        estimated_base_error = estimated_candidate_cents - controls.offset_cents - target_cents
        estimated_base_error = min(
            _BASE_ERROR_MAX_CENTS,
            max(_BASE_ERROR_MIN_CENTS, estimated_base_error),
        )
        plan = minimum_action_plan(
            estimated_base_error,
            controls,
            tolerance_cents=0,
        )
        if not plan or not isinstance(plan[0], PitchAction):
            raise LearningExecutionError("pitch planner must return a nonempty PitchAction plan")
        action = plan[0]
        mask = legal_action_mask(controls)
        if not mask[action]:
            raise LearningExecutionError("pitch planner returned an action that is not legal")
        return PitchDecision(
            action=action,
            estimated_candidate_cents=estimated_candidate_cents,
        )

    def _estimate_from_logits(self, logits: object, *, controls: ControlState) -> int:
        if not isinstance(logits, self._torch.Tensor):
            raise LearningExecutionError("pitch estimator logits must be a torch tensor")
        if logits.shape != (1, _PITCH_CLASS_COUNT):
            raise LearningExecutionError(
                f"pitch estimator logits must have shape (1, {_PITCH_CLASS_COUNT})"
            )
        if logits.dtype != self._torch.float32:
            raise LearningExecutionError("pitch estimator logits must have dtype float32")
        if logits.device != self._device:
            raise LearningExecutionError("pitch estimator logits must share the actor device")
        if not self._torch.isfinite(logits).all():
            raise LearningExecutionError("pitch estimator logits must contain only finite values")
        first_class = 0
        if self._decoding is PitchDecoding.FEASIBLE_ARGMAX:
            # Integer-cent sources round to five-cent classes with up to two cents
            # of quantization error. Include the classes nearest both public bounds,
            # even when a class center lies just outside the unrounded interval.
            offset = controls.offset_cents
            half_step = _PITCH_GRID_STEP_CENTS // 2
            first_class = (
                SOURCE_MIN_CENTS + offset - _PITCH_GRID_MIN_CENTS + half_step
            ) // _PITCH_GRID_STEP_CENTS
            last_class = (
                SOURCE_MAX_CENTS + offset - _PITCH_GRID_MIN_CENTS + half_step
            ) // _PITCH_GRID_STEP_CENTS
            logits = logits[:, first_class : last_class + 1]
        first_maximum = first_class + int(self._torch.argmax(logits, dim=1).item())
        return _PITCH_GRID_MIN_CENTS + _PITCH_GRID_STEP_CENTS * first_maximum


def _owned_observation_snapshot(observation: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(observation, Mapping):
        raise LearningContractError("observation must be a mapping")
    if set(observation) != set(_RAW_OBSERVATION_KEYS):
        raise LearningContractError(
            "observation must contain exactly spectrum, target_note, controls, and steps_remaining"
        )
    snapshot = {key: observation[key] for key in _RAW_OBSERVATION_KEYS}
    for key in ("spectrum", "controls"):
        value = snapshot[key]
        if isinstance(value, np.ndarray):
            snapshot[key] = np.array(value, copy=True, order="C")
    return snapshot


__all__ = ["PitchDecision", "PitchDecoding", "PitchPlannerActor"]
