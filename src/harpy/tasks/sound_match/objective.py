"""Whole-clip diagnostics with the frozen initial spectral objective.

This objective measures signal differences, not calibrated perceptual success.
Magnitude spectra use periodic Hann windows, no centering or alignment, every
frame start below FRAME_COUNT, right zero-padding, and no doubled interior bins.
All analysis uses float64. Level error is signed candidate-minus-target dB, with
an RMS floor of 1e-7 (-140 dBFS); envelope error is linear-amplitude RMS-bin RMSE.
"""

from __future__ import annotations

import math

import numpy as np

from harpy.tasks.sound_match.renderer import FRAME_COUNT

OBJECTIVE_ID = "harpy-multires-spectral-v1"
FFT_SIZES = (512, 2048, 8192)
SPECTRAL_EPSILON = 1e-7
LEVEL_FLOOR_RMS = 1e-7
ENVELOPE_BIN_FRAMES = 480


def _audio(value: np.ndarray, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float32
        or value.shape != (FRAME_COUNT,)
        or not np.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a finite float32 array with shape ({FRAME_COUNT},)")
    return value.astype(np.float64)


def _magnitudes(audio: np.ndarray, fft_size: int) -> np.ndarray:
    hop = fft_size // 4
    frame_count = (FRAME_COUNT + hop - 1) // hop
    padded_length = (frame_count - 1) * hop + fft_size
    padded = np.pad(audio, (0, padded_length - FRAME_COUNT))
    frames = np.lib.stride_tricks.sliding_window_view(padded, fft_size)[::hop]
    window = 0.5 - 0.5 * np.cos(math.tau * np.arange(fft_size, dtype=np.float64) / fft_size)
    return np.abs(np.fft.rfft(frames * window, axis=-1)) / np.sum(window, dtype=np.float64)


def _rms(audio: np.ndarray, *, axis: int | None = None) -> np.ndarray:
    return np.sqrt(np.mean(np.square(audio), axis=axis, dtype=np.float64))


def score(target: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    """Return finite JSON-safe scores without changing either input or retaining audio.

    Silence and out-of-range but finite calibration fixtures are accepted. The
    renderer enforces candidate amplitude bounds; this measurement does not clip.
    Working memory is bounded by the fixed clip and largest analysis window.
    """
    target = _audio(target, "target")
    candidate = _audio(candidate, "candidate")
    spectral = []
    for fft_size in FFT_SIZES:
        target_magnitude = _magnitudes(target, fft_size)
        candidate_magnitude = _magnitudes(candidate, fft_size)
        difference = target_magnitude - candidate_magnitude
        difference_norm = math.sqrt(float(np.sum(np.square(difference), dtype=np.float64)))
        target_norm = math.sqrt(float(np.sum(np.square(target_magnitude), dtype=np.float64)))
        convergence = difference_norm / max(target_norm, SPECTRAL_EPSILON)
        np.maximum(target_magnitude, SPECTRAL_EPSILON, out=target_magnitude)
        np.maximum(candidate_magnitude, SPECTRAL_EPSILON, out=candidate_magnitude)
        np.log(target_magnitude, out=target_magnitude)
        np.log(candidate_magnitude, out=candidate_magnitude)
        log_distance = float(np.mean(np.abs(target_magnitude - candidate_magnitude)))
        spectral.append(
            {"fft_size": fft_size, "convergence": convergence, "log_distance": log_distance}
        )
    target_envelope = _rms(target.reshape(-1, ENVELOPE_BIN_FRAMES), axis=1)
    candidate_envelope = _rms(candidate.reshape(-1, ENVELOPE_BIN_FRAMES), axis=1)
    envelope_rmse = float(_rms(target_envelope - candidate_envelope))
    level_error_db = 20.0 * (
        math.log10(max(float(_rms(candidate)), LEVEL_FLOOR_RMS))
        - math.log10(max(float(_rms(target)), LEVEL_FLOOR_RMS))
    )
    return {
        "distance": sum(item["convergence"] + item["log_distance"] for item in spectral) / 3,
        "spectral": spectral,
        "envelope_rmse": envelope_rmse,
        "level_error_db": level_error_db,
    }
