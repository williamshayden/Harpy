"""Fixed log-frequency evidence for the headless sine pitch environment."""

from __future__ import annotations

import numpy as np

from harpy.analysis import AnalysisConfig, analyze
from harpy.tuning import Tuning

_SAMPLE_RATE_HZ = 48_000
GYM_ANALYSIS_CONFIG = AnalysisConfig(
    fft_frames=262_144,
    spectrum_min_hz=_SAMPLE_RATE_HZ / 262_144,
    spectrum_max_hz=24_000.0,
    spectrum_floor_dbfs=-120.0,
)

_LOG_COORDINATE_CENTS = np.arange(1_100, 10_901, 5, dtype=np.float64)
LOG_SPECTRUM_SIZE = 1_961
_TUNING = Tuning()
LOG_FREQUENCY_GRID_HZ = np.array(
    [_TUNING.frequency_hz_for_midi_coordinate(cents / 100.0) for cents in _LOG_COORDINATE_CENTS],
    dtype=np.float64,
    copy=True,
    order="C",
)
LOG_FREQUENCY_GRID_HZ.setflags(write=False)


def encode_log_spectrum(samples: np.ndarray) -> np.ndarray:
    """Return normalized fixed-grid spectrum evidence for recent mono samples."""

    if not np.all(np.isfinite(samples)):
        raise ValueError("samples must contain only finite values")
    observation = analyze(samples, _SAMPLE_RATE_HZ, GYM_ANALYSIS_CONFIG)
    if not observation.has_signal:
        return np.zeros(LOG_SPECTRUM_SIZE, dtype=np.float32)

    interpolated_dbfs = np.interp(
        LOG_FREQUENCY_GRID_HZ,
        observation.spectrum_frequency_hz,
        observation.spectrum_level_dbfs,
    )
    normalized = (np.clip(interpolated_dbfs, -120.0, 0.0) + 120.0) / 120.0
    return np.array(normalized, dtype=np.float32, copy=True, order="C")
