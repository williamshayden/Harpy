"""Bounded text summaries derived only from public sound observations.

Spectral peaks are measured frequency bins, not evaluator pitches or source labels.
This module never synthesizes an audio candidate.
"""

from __future__ import annotations

import math

import numpy as np

from harpy.tasks.sound_match.models import Observation
from harpy.tasks.sound_match.objective import OBJECTIVE_ID, score
from harpy.tasks.sound_match.patch import PARAMETER_BOUNDS

FEATURE_ID = "harpy-llm-whole-clip-peaks-envelope/1"
PEAK_COUNT = 12
ENVELOPE_BINS = 64
AMPLITUDE_FLOOR = 1e-12


def feature_configuration() -> dict:
    return {
        "id": FEATURE_ID,
        "spectral_window": "whole-clip-periodic-hann",
        "spectral_peaks": PEAK_COUNT,
        "peak_selection": "positive-frequency-local-maxima-descending-magnitude-ties-lower-bin",
        "peak_frequency": "FFT-bin-frequency-Hz; no interpolation or pitch-label inference",
        "relative_amplitude": "20log10(peak/strongest-peak), floor -120 dB",
        "envelope_bins": ENVELOPE_BINS,
        "envelope": "RMS of equally sized consecutive bins spanning the whole clip",
        "level": "whole-clip RMS dBFS, amplitude floor 1e-12",
        "rounding": "six decimal places",
        "objective_id": OBJECTIVE_ID,
        "candidate_synthesis": "none",
    }


def _rounded(value: float) -> float:
    return round(float(value), 6)


def waveform_features(audio: np.ndarray, sample_rate_hz: int) -> dict:
    """Summarize an owned/public waveform without modifying or retaining it."""
    if (
        not isinstance(audio, np.ndarray)
        or audio.dtype != np.float32
        or audio.ndim != 1
        or len(audio) < ENVELOPE_BINS
        or len(audio) % ENVELOPE_BINS
        or not np.isfinite(audio).all()
    ):
        raise ValueError("features require a finite float32 mono clip divisible into envelope bins")
    if type(sample_rate_hz) is not int or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be a positive integer")
    samples = audio.astype(np.float64)
    window = 0.5 - 0.5 * np.cos(math.tau * np.arange(len(samples)) / len(samples))
    magnitude = np.abs(np.fft.rfft(samples * window))
    maxima = (
        np.flatnonzero((magnitude[1:-1] >= magnitude[:-2]) & (magnitude[1:-1] > magnitude[2:])) + 1
    )
    maxima = maxima[magnitude[maxima] > AMPLITUDE_FLOOR]
    ordered = maxima[np.lexsort((maxima, -magnitude[maxima]))][:PEAK_COUNT]
    strongest = float(magnitude[ordered[0]]) if len(ordered) else 0.0
    peaks = [
        {
            "frequency_hz": _rounded(index * sample_rate_hz / len(samples)),
            "relative_amplitude_db": _rounded(
                max(-120.0, 20 * math.log10(float(magnitude[index]) / strongest))
            ),
        }
        for index in ordered
    ]
    rms = math.sqrt(float(np.mean(samples**2)))
    envelope = np.sqrt(np.mean(samples.reshape(ENVELOPE_BINS, -1) ** 2, axis=1))
    return {
        "spectral_peaks": peaks,
        "rms_envelope": [_rounded(value) for value in envelope],
        "rms_dbfs": _rounded(20 * math.log10(max(rms, AMPLITUDE_FLOOR))),
        "peak_amplitude": _rounded(np.max(np.abs(samples))),
    }


def public_observation(observation: Observation) -> dict:
    """Whitelist public state: no episode identity, target labels, paths, or hashes."""
    objective = score(observation.target_audio, observation.candidate_audio)
    return {
        "feature_id": FEATURE_ID,
        "audio": {
            "sample_rate_hz": observation.sample_rate_hz,
            "frame_count": observation.frame_count,
            "note_off_frame": observation.note_off_frame,
            "envelope_bin_seconds": _rounded(
                observation.frame_count / observation.sample_rate_hz / ENVELOPE_BINS
            ),
        },
        "target_features": waveform_features(observation.target_audio, observation.sample_rate_hz),
        "candidate_features": waveform_features(
            observation.candidate_audio, observation.sample_rate_hz
        ),
        "current_patch": observation.patch.to_dict(),
        "editable_fields": list(observation.editable_fields),
        "allowed_values": {
            "bounds": {
                name: list(observation.allowed_values["bounds"][name]) for name in PARAMETER_BOUNDS
            },
            "waveforms": list(observation.allowed_values["waveforms"]),
            "voice_count": len(observation.patch.notes_cents),
        },
        "budgets": {
            "candidate_evaluations_remaining": observation.render_budget_remaining,
            "decisions_remaining": observation.decision_budget_remaining,
        },
        "last_feedback": {
            "status": observation.last_action_status.get("status"),
            "reason": observation.last_action_status.get("reason"),
        },
        "candidate_objective": {
            "id": OBJECTIVE_ID,
            "distance": _rounded(objective["distance"]),
            "envelope_rmse": _rounded(objective["envelope_rmse"]),
            "level_error_db": _rounded(objective["level_error_db"]),
        },
    }
