"""Strict conversion from frozen raw observations to learned-policy inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium
import numpy as np

from harpy.envs.models import (
    CENT_MAX,
    CENT_MIN,
    MAX_STEPS,
    OCTAVE_MAX,
    OCTAVE_MIN,
    SEMITONE_MAX,
    SEMITONE_MIN,
    TARGET_NOTE_COUNT,
)
from harpy.envs.spectrum import LOG_SPECTRUM_SIZE
from harpy.learning.errors import LearningContractError
from harpy.learning.models import PREPROCESSING_SCHEMA_ID

type PolicyObservation = dict[str, np.ndarray[Any, np.dtype[np.float32]]]

_RAW_KEYS = frozenset({"spectrum", "target_note", "controls", "steps_remaining"})
_CONTROL_LOW = np.array([OCTAVE_MIN, SEMITONE_MIN, CENT_MIN], dtype=np.int16)
_CONTROL_HIGH = np.array([OCTAVE_MAX, SEMITONE_MAX, CENT_MAX], dtype=np.int16)

POLICY_OBSERVATION_SPACE = gymnasium.spaces.Dict(
    {
        "spectrum": gymnasium.spaces.Box(0.0, 1.0, (LOG_SPECTRUM_SIZE,), np.float32),
        "state": gymnasium.spaces.Box(
            low=np.array([-1, -1, -1, -1, 0], dtype=np.float32),
            high=np.ones(5, dtype=np.float32),
            dtype=np.float32,
        ),
    }
)


def preprocess_observation(observation: Mapping[str, object]) -> PolicyObservation:
    """Validate one frozen raw observation and return owned policy features."""
    if not isinstance(observation, Mapping):
        raise LearningContractError("observation must be a mapping")
    if set(observation) != _RAW_KEYS:
        raise LearningContractError(
            "observation must contain exactly spectrum, target_note, controls, and steps_remaining"
        )

    spectrum = _spectrum(observation["spectrum"])
    target_note = _int64_scalar(observation["target_note"], "target_note")
    controls = _controls(observation["controls"])
    steps_remaining = _int64_scalar(observation["steps_remaining"], "steps_remaining")

    if not 0 <= target_note < TARGET_NOTE_COUNT:
        raise LearningContractError(f"target_note must be within 0..{TARGET_NOTE_COUNT - 1}")
    if not 0 <= steps_remaining <= MAX_STEPS:
        raise LearningContractError(f"steps_remaining must be within 0..{MAX_STEPS}")

    state = np.array(
        [
            (target_note - 12) / 12,
            controls[0] / 2,
            controls[1] / 12,
            controls[2] / 100,
            steps_remaining / MAX_STEPS,
        ],
        dtype=np.float32,
        order="C",
    )
    return {"spectrum": spectrum, "state": state}


class PolicyObservationWrapper(gymnasium.ObservationWrapper):
    """Expose the policy observation contract for a frozen sine-pitch environment."""

    observation_space = POLICY_OBSERVATION_SPACE

    def observation(self, observation: Mapping[str, object]) -> PolicyObservation:
        """Transform one raw environment observation through the shared validator."""
        return preprocess_observation(observation)


def _spectrum(value: object) -> np.ndarray[Any, np.dtype[np.float32]]:
    if not isinstance(value, np.ndarray):
        raise LearningContractError("spectrum must be a float32 ndarray")
    if value.dtype != np.dtype(np.float32):
        raise LearningContractError("spectrum must have dtype float32")
    if value.ndim != 1 or value.shape != (LOG_SPECTRUM_SIZE,):
        raise LearningContractError(f"spectrum must have shape ({LOG_SPECTRUM_SIZE},)")
    if not np.isfinite(value).all():
        raise LearningContractError("spectrum must contain only finite values")
    if np.any(value < 0.0) or np.any(value > 1.0):
        raise LearningContractError("spectrum must be within 0..1")
    return np.array(value, dtype=np.float32, copy=True, order="C")


def _int64_scalar(value: object, field: str) -> int:
    if not isinstance(value, np.int64):
        raise LearningContractError(f"{field} must be a numpy int64 scalar")
    return int(value)


def _controls(value: object) -> np.ndarray[Any, np.dtype[np.int16]]:
    if not isinstance(value, np.ndarray):
        raise LearningContractError("controls must be an int16 ndarray")
    if value.dtype != np.dtype(np.int16):
        raise LearningContractError("controls must have dtype int16")
    if value.ndim != 1 or value.shape != (3,):
        raise LearningContractError("controls must have shape (3,)")
    if np.any(value < _CONTROL_LOW) or np.any(value > _CONTROL_HIGH):
        raise LearningContractError("controls must be within frozen control bounds")
    return value


__all__ = [
    "POLICY_OBSERVATION_SPACE",
    "PREPROCESSING_SCHEMA_ID",
    "PolicyObservation",
    "PolicyObservationWrapper",
    "preprocess_observation",
]
