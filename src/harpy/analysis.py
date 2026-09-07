"""Pure waveform and calibrated spectrum observations for captured audio."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from functools import cache

import numpy as np


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Configuration for waveform and spectrum analysis."""

    waveform_window_seconds: float = 0.050
    fft_frames: int = 16_384
    spectrum_min_hz: float = 20.0
    spectrum_max_hz: float = 20_000.0
    spectrum_floor_dbfs: float = -120.0
    window: str = "hann"


@dataclass(frozen=True, slots=True)
class AudioObservation:
    """An immutable snapshot of waveform and spectrum data."""

    has_signal: bool
    waveform_samples: np.ndarray
    waveform_time_ms: np.ndarray
    spectrum_frequency_hz: np.ndarray
    spectrum_level_dbfs: np.ndarray
    peak_amplitude_fs: float | None
    peak_frequency_hz: float | None
    peak_level_dbfs: float | None

    def __post_init__(self) -> None:
        array_dtypes = {
            "waveform_samples": np.float32,
            "waveform_time_ms": np.float64,
            "spectrum_frequency_hz": np.float64,
            "spectrum_level_dbfs": np.float64,
        }
        for name, dtype in array_dtypes.items():
            supplied = np.asarray(getattr(self, name), dtype=dtype).reshape(-1)
            owned = np.array(supplied, dtype=dtype, copy=True, order="C")
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)

        if self.waveform_samples.size != self.waveform_time_ms.size:
            raise ValueError("waveform samples and time arrays must have equal length")
        if self.spectrum_frequency_hz.size != self.spectrum_level_dbfs.size:
            raise ValueError("spectrum frequency and level arrays must have equal length")


def _waveform_frames(config: AnalysisConfig, sample_rate_hz: float) -> int:
    try:
        duration = float(config.waveform_window_seconds)
    except (TypeError, ValueError) as error:
        raise ValueError("waveform_window_seconds must produce 1..fft_frames") from error
    if not math.isfinite(duration):
        raise ValueError("waveform_window_seconds must produce 1..fft_frames")
    frame_count = duration * sample_rate_hz
    if not math.isfinite(frame_count):
        raise ValueError("waveform_window_seconds must produce 1..fft_frames")
    return math.floor(frame_count + 0.5)


def validate_analysis_config(config: AnalysisConfig, sample_rate_hz: int) -> None:
    """Reject analysis settings that cannot form a calibrated observation."""

    try:
        numeric_sample_rate = float(sample_rate_hz)
    except (TypeError, ValueError) as error:
        raise ValueError("sample_rate_hz must be finite and positive") from error
    if not math.isfinite(numeric_sample_rate) or numeric_sample_rate <= 0.0:
        raise ValueError("sample_rate_hz must be finite and positive")

    try:
        fft_frames = operator.index(config.fft_frames)
    except TypeError as error:
        raise ValueError("fft_frames must be an even integer of at least 4") from error
    if isinstance(config.fft_frames, bool) or fft_frames < 4 or fft_frames % 2:
        raise ValueError("fft_frames must be an even integer of at least 4")

    waveform_frames = _waveform_frames(config, numeric_sample_rate)
    if not 1 <= waveform_frames <= fft_frames:
        raise ValueError("waveform window must produce 1..fft_frames")

    try:
        minimum_hz = float(config.spectrum_min_hz)
        maximum_hz = float(config.spectrum_max_hz)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "spectrum bounds must be finite, ordered, positive, and within Nyquist"
        ) from error
    if (
        not math.isfinite(minimum_hz)
        or not math.isfinite(maximum_hz)
        or minimum_hz <= 0.0
        or minimum_hz >= maximum_hz
        or maximum_hz > numeric_sample_rate / 2.0
    ):
        raise ValueError("spectrum bounds must be finite, ordered, positive, and within Nyquist")

    try:
        floor_dbfs = float(config.spectrum_floor_dbfs)
    except (TypeError, ValueError) as error:
        raise ValueError("spectrum_floor_dbfs must be finite and nonpositive") from error
    if not math.isfinite(floor_dbfs) or floor_dbfs > 0.0:
        raise ValueError("spectrum_floor_dbfs must be finite and nonpositive")

    if config.window != "hann":
        raise ValueError("window must be 'hann'")


def _empty_observation() -> AudioObservation:
    return AudioObservation(
        has_signal=False,
        waveform_samples=np.empty(0),
        waveform_time_ms=np.empty(0),
        spectrum_frequency_hz=np.empty(0),
        spectrum_level_dbfs=np.empty(0),
        peak_amplitude_fs=None,
        peak_frequency_hz=None,
        peak_level_dbfs=None,
    )


@cache
def _hann_window(fft_frames: int) -> np.ndarray:
    window = np.hanning(fft_frames)
    window.setflags(write=False)
    return window


def _quadratic_peak(
    amplitudes: np.ndarray,
    winning_bin: int,
    bin_width_hz: float,
) -> tuple[float, float]:
    winning_level = 20.0 * math.log10(float(amplitudes[winning_bin]))
    peak_frequency_hz = winning_bin * bin_width_hz
    if winning_bin == 0 or winning_bin == amplitudes.size - 1:
        return peak_frequency_hz, winning_level

    neighbors = amplitudes[winning_bin - 1 : winning_bin + 2]
    if np.any(neighbors <= 0.0):
        return peak_frequency_hz, winning_level
    lower_level, center_level, upper_level = 20.0 * np.log10(neighbors)
    denominator = lower_level - 2.0 * center_level + upper_level
    if not np.isfinite(denominator) or denominator == 0.0:
        return peak_frequency_hz, winning_level

    bin_offset = 0.5 * (lower_level - upper_level) / denominator
    peak_frequency_hz = (winning_bin + bin_offset) * bin_width_hz
    peak_level_dbfs = center_level - 0.25 * (lower_level - upper_level) * bin_offset
    return float(peak_frequency_hz), float(peak_level_dbfs)


def analyze(
    samples: np.ndarray,
    sample_rate_hz: int,
    config: AnalysisConfig = AnalysisConfig(),  # noqa: B008 - immutable public API default
) -> AudioObservation:
    """Observe the most recent complete analysis window in ``samples``."""

    validate_analysis_config(config, sample_rate_hz)
    sample_array = np.asarray(samples)
    if sample_array.ndim != 1:
        raise ValueError("samples must be a one-dimensional array")
    if sample_array.size < config.fft_frames:
        raise ValueError(f"analysis requires at least {config.fft_frames} samples")

    waveform_frames = _waveform_frames(config, float(sample_rate_hz))
    capture = sample_array[-config.fft_frames :]
    if not np.all(np.isfinite(capture)):
        raise ValueError("analysis capture must contain only finite samples")
    trailing_waveform = np.asarray(capture[-waveform_frames:], dtype=np.float64)
    trailing_peak_amplitude = float(np.max(np.abs(trailing_waveform)))
    if trailing_peak_amplitude <= 1e-6:
        return _empty_observation()

    latest_start = config.fft_frames - waveform_frames
    eligible_crossings = np.flatnonzero(
        (capture[1 : latest_start + 1] > 0.0) & (capture[:latest_start] <= 0.0)
    )
    waveform_start = int(eligible_crossings[-1] + 1) if eligible_crossings.size else latest_start
    waveform = capture[waveform_start : waveform_start + waveform_frames]
    peak_amplitude = float(np.max(np.abs(np.asarray(waveform, dtype=np.float64))))
    waveform_time_ms = np.arange(waveform_frames, dtype=np.float64) * 1_000.0 / sample_rate_hz

    window = _hann_window(config.fft_frames)
    windowed_capture = np.asarray(capture, dtype=np.float64) * window
    amplitudes = 2.0 * np.abs(np.fft.rfft(windowed_capture)) / np.sum(window)
    amplitudes[-1] *= 0.5
    frequencies_hz = np.fft.rfftfreq(config.fft_frames, d=1.0 / sample_rate_hz)
    in_range = (
        (frequencies_hz > 0.0)
        & (frequencies_hz >= config.spectrum_min_hz)
        & (frequencies_hz <= config.spectrum_max_hz)
    )
    spectrum_frequency_hz = frequencies_hz[in_range]
    spectrum_amplitudes = amplitudes[in_range]
    with np.errstate(divide="ignore"):
        raw_spectrum_dbfs = 20.0 * np.log10(spectrum_amplitudes)
    spectrum_level_dbfs = np.maximum(raw_spectrum_dbfs, config.spectrum_floor_dbfs)

    peak_frequency_hz: float | None = None
    peak_level_dbfs: float | None = None
    floor_amplitude = 10.0 ** (config.spectrum_floor_dbfs / 20.0)
    if spectrum_amplitudes.size and np.max(spectrum_amplitudes) > floor_amplitude:
        winning_local_bin = int(np.argmax(spectrum_amplitudes))
        in_range_bins = np.flatnonzero(in_range)
        winning_bin = int(in_range_bins[winning_local_bin])
        peak_frequency_hz, peak_level_dbfs = _quadratic_peak(
            amplitudes,
            winning_bin,
            sample_rate_hz / config.fft_frames,
        )

    return AudioObservation(
        has_signal=True,
        waveform_samples=waveform,
        waveform_time_ms=waveform_time_ms,
        spectrum_frequency_hz=spectrum_frequency_hz,
        spectrum_level_dbfs=spectrum_level_dbfs,
        peak_amplitude_fs=peak_amplitude,
        peak_frequency_hz=peak_frequency_hz,
        peak_level_dbfs=peak_level_dbfs,
    )
