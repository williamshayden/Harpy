"""Research-only spectrum alternatives; the qualified Harpy encoder is unchanged.

All actors receive the same waveform track and delegate actions to Harpy's
committed controller. The FFT reference deliberately retains five-cent decoding
so finer action precision cannot explain the primary representation comparison.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from harpy.analysis import analyze
from harpy.envs.models import SOURCE_MAX_CENTS, SOURCE_MIN_CENTS, ControlState, ObservationMode
from harpy.envs.observations import owned_observation
from harpy.envs.spectrum import (
    GYM_ANALYSIS_CONFIG,
    LOG_FREQUENCY_GRID_HZ,
    LOG_SPECTRUM_SIZE,
    encode_log_spectrum,
)
from harpy.experiments import ActorSpec, spectrum_peak_actor_spec

SAMPLE_RATE_HZ = 48_000
CELL_MAX_ID = "harpy-research-log-cell-max-dbfs-v1"
FFT_REFERENCE_ID = "harpy-research-hann-quadratic-fft-v1"
GRID_CENTS = np.arange(1_100, 10_901, 5, dtype=np.float64)
CELL_EDGES_HZ = np.append(
    LOG_FREQUENCY_GRID_HZ / 2.0 ** (2.5 / 1_200.0),
    LOG_FREQUENCY_GRID_HZ[-1] * 2.0 ** (2.5 / 1_200.0),
)
GRID_CENTS.setflags(write=False)
CELL_EDGES_HZ.setflags(write=False)


def _max_over_cells(frequencies: np.ndarray, levels: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Exact cell maxima of the piecewise-linear dB curve, including empty cells."""
    boundary_levels = np.interp(edges, frequencies, levels)
    maxima = np.maximum(boundary_levels[:-1], boundary_levels[1:])
    membership = np.searchsorted(edges, frequencies, side="right") - 1
    covered = (membership >= 0) & (membership < maxima.size)
    np.maximum.at(maxima, membership[covered], levels[covered])
    return maxima


def encode_cell_max(samples: np.ndarray) -> np.ndarray:
    """Keep each five-cent cell's maximum without changing FFT or calibration."""
    observation = analyze(samples, SAMPLE_RATE_HZ, GYM_ANALYSIS_CONFIG)
    if not observation.has_signal:
        return np.zeros(LOG_SPECTRUM_SIZE, dtype=np.float32)
    levels = _max_over_cells(
        observation.spectrum_frequency_hz,
        observation.spectrum_level_dbfs,
        CELL_EDGES_HZ,
    )
    return np.array((np.clip(levels, -120.0, 0.0) + 120.0) / 120.0, dtype=np.float32)


def _feasible_grid(controls: np.ndarray) -> tuple[int, int]:
    offset = ControlState(*(int(value) for value in controls)).offset_cents
    first = (SOURCE_MIN_CENTS + offset - 1_100 + 2) // 5
    stop = (SOURCE_MAX_CENTS + offset - 1_100 + 2) // 5 + 1
    return first, stop


def encode_fft_reference(samples: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Find a bounded sub-bin FFT peak, then explicitly quantize to five cents.

    Search the same feasible cells' frequency coverage as the grid actors.
    A concave three-bin dB fit is limited to half a bin; degenerate fits retain
    the winning bin. The returned one-hot vector is a decision adapter, not a
    spectrum or a calibrated audio representation.
    """
    observation = analyze(samples, SAMPLE_RATE_HZ, GYM_ANALYSIS_CONFIG)
    scores = np.zeros(LOG_SPECTRUM_SIZE, dtype=np.float32)
    if not observation.has_signal:
        return scores
    first, stop = _feasible_grid(controls)
    frequencies = observation.spectrum_frequency_hz
    levels = observation.spectrum_level_dbfs
    eligible = np.flatnonzero(
        (frequencies >= CELL_EDGES_HZ[first]) & (frequencies <= CELL_EDGES_HZ[stop])
    )
    winning = int(eligible[int(np.argmax(levels[eligible]))])
    offset = 0.0
    if 0 < winning < levels.size - 1:
        left, center, right = levels[winning - 1 : winning + 2]
        curvature = left - 2.0 * center + right
        if curvature < 0.0:
            offset = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    frequency = frequencies[winning] + offset * SAMPLE_RATE_HZ / GYM_ANALYSIS_CONFIG.fft_frames
    coordinate = 6_900.0 + 1_200.0 * np.log2(frequency / 440.0)
    # Same nearest-five-cent coordinates and lower-index tie rule as argmax.
    index = first + int(np.argmin(np.abs(GRID_CENTS[first:stop] - coordinate)))
    scores[index] = 1.0
    return scores


class WaveformActor:
    """An episode-local encoding adapter around the unchanged committed planner."""

    def __init__(self, encoder: str) -> None:
        if encoder not in {"legacy-point", "cell-max", "quadratic-fft"}:
            raise ValueError("unknown research encoder")
        self._encoder = encoder
        self._controller = spectrum_peak_actor_spec().factory()
        self._scores = None

    def decide(self, observation: Mapping[str, object]):
        observation = owned_observation(observation, ObservationMode.WAVEFORM)
        if self._scores is None:
            samples = observation["waveform"]
            if self._encoder == "legacy-point":
                self._scores = encode_log_spectrum(samples)
            elif self._encoder == "cell-max":
                self._scores = encode_cell_max(samples)
            else:
                self._scores = encode_fft_reference(samples, observation["controls"])
        view = {key: value for key, value in observation.items() if key != "waveform"}
        view["spectrum"] = self._scores
        return self._controller.decide(view)


def actor_specs(*, artifact_paths: tuple[Path, ...] = ()) -> tuple[ActorSpec, ...]:
    baseline = spectrum_peak_actor_spec()
    encoders = {
        "legacy-point": "harpy-legacy-log-point-dbfs-v1",
        "cell-max": CELL_MAX_ID,
        "quadratic-fft": FFT_REFERENCE_ID,
    }
    return tuple(
        ActorSpec(
            name=name,
            factory=lambda name=name: WaveformActor(name),
            observation_mode=ObservationMode.WAVEFORM,
            estimator=identity,
            decoder=(
                "harpy-feasible-fft-peak-nearest-five-cent-v1"
                if name == "quadratic-fft"
                else baseline.decoder
            ),
            controller=baseline.controller,
            artifact_paths=(Path(__file__).resolve(), *artifact_paths),
            artifact_provenance={
                "status": "research",
                "encoder": identity,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "capture_frames": GYM_ANALYSIS_CONFIG.fft_frames,
                "grid_size": LOG_SPECTRUM_SIZE,
                "grid_step_cents": 5,
                "weights": "none",
                "qualified_v1_encoder_changed": False,
                "reference_decoding": "FFT reference also quantized to five cents",
            },
        )
        for name, identity in encoders.items()
    )
