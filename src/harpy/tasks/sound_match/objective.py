"""Whole-clip diagnostics with the frozen initial spectral objective.

This objective measures signal differences, not calibrated perceptual success.
Magnitude spectra use periodic Hann windows, no centering or alignment, every
frame start below FRAME_COUNT, right zero-padding, and no doubled interior bins.
All analysis uses float64. Level error is signed candidate-minus-target dB, with
an RMS floor of 1e-7 (-140 dBFS); envelope error is linear-amplitude RMS-bin RMSE.
Process-local content caches retain at most two immutable waveform analyses
(20 MiB of array payload) and 128 scores. They retain no input audio, never waive
candidate evaluations, and return independently owned score documents.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

import numpy as np

from harpy.tasks.sound_match.renderer import FRAME_COUNT

OBJECTIVE_ID = "harpy-multires-spectral-v1"
FFT_SIZES = (512, 2048, 8192)
SPECTRAL_EPSILON = 1e-7
LEVEL_FLOOR_RMS = 1e-7
ENVELOPE_BIN_FRAMES = 480

_MAX_ANALYSIS_ENTRIES = 2
_MAX_ANALYSIS_BYTES = 20 * 1024 * 1024
_MAX_SCORE_ENTRIES = 128
_CACHE_LOCK = RLock()


@dataclass(frozen=True, slots=True)
class _WaveAnalysis:
    magnitudes: tuple[np.ndarray, ...]
    log_magnitudes: tuple[np.ndarray, ...]
    norms: tuple[float, ...]
    envelope: np.ndarray
    level_rms: float
    array_bytes: int


_ANALYSES: OrderedDict[bytes, _WaveAnalysis] = OrderedDict()
_SCORES: OrderedDict[tuple[bytes, bytes], dict[str, object]] = OrderedDict()
_ANALYSIS_BYTES = 0
_CACHE_COUNTS = {"analysis_hits": 0, "analysis_misses": 0, "score_hits": 0, "score_misses": 0}


def _clear_score_cache() -> None:
    """Reset bounded process-local reuse and counters; private test/profile hook."""
    global _ANALYSIS_BYTES
    with _CACHE_LOCK:
        _ANALYSES.clear()
        _SCORES.clear()
        _ANALYSIS_BYTES = 0
        for name in _CACHE_COUNTS:
            _CACHE_COUNTS[name] = 0


def _score_cache_info() -> dict[str, int]:
    """Return owned diagnostics, without exposing cached arrays or score objects."""
    with _CACHE_LOCK:
        return {
            "analysis_entries": len(_ANALYSES),
            "analysis_array_bytes": _ANALYSIS_BYTES,
            "max_analysis_entries": _MAX_ANALYSIS_ENTRIES,
            "max_analysis_array_bytes": _MAX_ANALYSIS_BYTES,
            "score_entries": len(_SCORES),
            "max_score_entries": _MAX_SCORE_ENTRIES,
            **_CACHE_COUNTS,
        }


def _audio(value: np.ndarray, name: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.float32
        or value.shape != (FRAME_COUNT,)
    ):
        raise ValueError(f"{name} must be a finite float32 array with shape ({FRAME_COUNT},)")
    snapshot = value.astype(np.float64)
    if not np.isfinite(snapshot).all():
        raise ValueError(f"{name} must be a finite float32 array with shape ({FRAME_COUNT},)")
    return snapshot


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


def _analysis(key: bytes, audio: np.ndarray) -> _WaveAnalysis:
    """Analyze an owned snapshot. Caller holds the cache lock throughout scoring."""
    global _ANALYSIS_BYTES
    if key in _ANALYSES:
        _CACHE_COUNTS["analysis_hits"] += 1
        _ANALYSES.move_to_end(key)
        return _ANALYSES[key]
    _CACHE_COUNTS["analysis_misses"] += 1
    # Evict before allocating the next analysis, avoiding three cached analyses
    # being resident during ordinary two-entry operation.
    while _ANALYSES and len(_ANALYSES) >= _MAX_ANALYSIS_ENTRIES:
        _, removed = _ANALYSES.popitem(last=False)
        _ANALYSIS_BYTES -= removed.array_bytes
        del removed
    magnitudes, logarithms, norms = [], [], []
    for fft_size in FFT_SIZES:
        magnitude = _magnitudes(audio, fft_size)
        norms.append(math.sqrt(float(np.sum(np.square(magnitude), dtype=np.float64))))
        logarithm = magnitude.copy()
        np.maximum(logarithm, SPECTRAL_EPSILON, out=logarithm)
        np.log(logarithm, out=logarithm)
        magnitude.flags.writeable = logarithm.flags.writeable = False
        magnitudes.append(magnitude)
        logarithms.append(logarithm)
    envelope = _rms(audio.reshape(-1, ENVELOPE_BIN_FRAMES), axis=1)
    envelope.flags.writeable = False
    analysis = _WaveAnalysis(
        tuple(magnitudes),
        tuple(logarithms),
        tuple(norms),
        envelope,
        float(_rms(audio)),
        sum(value.nbytes for value in (*magnitudes, *logarithms, envelope)),
    )
    if _MAX_ANALYSIS_ENTRIES > 0 and analysis.array_bytes <= _MAX_ANALYSIS_BYTES:
        while _ANALYSES and _ANALYSIS_BYTES + analysis.array_bytes > _MAX_ANALYSIS_BYTES:
            _, removed = _ANALYSES.popitem(last=False)
            _ANALYSIS_BYTES -= removed.array_bytes
            del removed
        _ANALYSES[key] = analysis
        _ANALYSIS_BYTES += analysis.array_bytes
    return analysis


def _score_analyses(target: _WaveAnalysis, candidate: _WaveAnalysis) -> dict[str, object]:
    spectral = []
    for index, fft_size in enumerate(FFT_SIZES):
        target_magnitude = target.magnitudes[index]
        candidate_magnitude = candidate.magnitudes[index]
        difference = target_magnitude - candidate_magnitude
        difference_norm = math.sqrt(float(np.sum(np.square(difference), dtype=np.float64)))
        convergence = difference_norm / max(target.norms[index], SPECTRAL_EPSILON)
        log_distance = float(
            np.mean(np.abs(target.log_magnitudes[index] - candidate.log_magnitudes[index]))
        )
        spectral.append(
            {"fft_size": fft_size, "convergence": convergence, "log_distance": log_distance}
        )
    envelope_rmse = float(_rms(target.envelope - candidate.envelope))
    level_error_db = 20.0 * (
        math.log10(max(candidate.level_rms, LEVEL_FLOOR_RMS))
        - math.log10(max(target.level_rms, LEVEL_FLOOR_RMS))
    )
    return {
        "distance": sum(item["convergence"] + item["log_distance"] for item in spectral) / 3,
        "spectral": spectral,
        "envelope_rmse": envelope_rmse,
        "level_error_db": level_error_db,
    }


def score(target: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    """Return finite owned scores without changing inputs or retaining their audio.

    Validate and snapshot both inputs on every call, including cache hits. Content
    hashes describe these owned snapshots, so later caller mutations cannot change
    cached analyses. Silence and finite out-of-range calibration audio are accepted;
    the renderer enforces candidate amplitude bounds and measurement does not clip.
    Cached computation never changes logical candidate charges or DSP counters.
    """
    target = _audio(target, "target")
    candidate = _audio(candidate, "candidate")
    keys = (
        hashlib.sha256(target.tobytes()).digest(),
        hashlib.sha256(candidate.tobytes()).digest(),
    )
    with _CACHE_LOCK:
        if keys in _SCORES:
            _CACHE_COUNTS["score_hits"] += 1
            _SCORES.move_to_end(keys)
            return copy.deepcopy(_SCORES[keys])
        _CACHE_COUNTS["score_misses"] += 1
        result = _score_analyses(_analysis(keys[0], target), _analysis(keys[1], candidate))
        if _MAX_SCORE_ENTRIES > 0:
            while len(_SCORES) >= _MAX_SCORE_ENTRIES:
                _SCORES.popitem(last=False)
            _SCORES[keys] = result
        return copy.deepcopy(result)
