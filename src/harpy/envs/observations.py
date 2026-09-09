"""Owned, exact public observation validation for user-supplied experiment actors."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from harpy.envs.models import ControlState, ObservationMode
from harpy.envs.spectrum import GYM_ANALYSIS_CONFIG, LOG_SPECTRUM_SIZE


def owned_observation(
    observation: Mapping[str, object], observation_mode: ObservationMode
) -> dict[str, np.ndarray | np.int64]:
    """Validate without coercion, then copy every mutable actor-visible value."""
    if not isinstance(observation_mode, ObservationMode):
        raise ValueError("observation_mode must be an ObservationMode")
    if not isinstance(observation, Mapping):
        raise ValueError("observation must be a mapping")
    keys = {"target_note", "controls", "steps_remaining"}
    evidence = {
        ObservationMode.SPECTRUM: ("spectrum", (LOG_SPECTRUM_SIZE,), 0.0, 1.0),
        ObservationMode.WAVEFORM: ("waveform", (GYM_ANALYSIS_CONFIG.fft_frames,), None, None),
        ObservationMode.ORACLE: ("current_pitch_coordinate", (1,), 11.0, 109.0),
    }.get(observation_mode)
    if evidence is not None:
        keys.add(evidence[0])
    if set(observation) != keys:
        raise ValueError(f"observation keys must be exactly {sorted(keys)}")
    for key, maximum in (("target_note", 24), ("steps_remaining", 64)):
        value = observation[key]
        if not isinstance(value, np.int64) or not 0 <= value <= maximum:
            raise ValueError(f"{key} must be a scalar int64 within 0..{maximum}")
    controls = observation["controls"]
    if not isinstance(controls, np.ndarray) or controls.dtype != np.int16 or controls.shape != (3,):
        raise ValueError("controls must be an int16 array with shape (3,)")
    ControlState(*(int(value) for value in controls))
    if evidence is not None:
        key, shape, minimum, maximum = evidence
        value = observation[key]
        if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != shape:
            raise ValueError(f"{key} must be a float32 array with shape {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} must contain only finite values")
        if minimum is not None and (np.any(value < minimum) or np.any(value > maximum)):
            raise ValueError(f"{key} must be within {minimum}..{maximum}")
    return {
        key: np.array(value, copy=True, order="C") if isinstance(value, np.ndarray) else value
        for key, value in observation.items()
    }
