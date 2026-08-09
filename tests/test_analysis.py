from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import harpy.analysis as analysis_module
from harpy.analysis import AnalysisConfig, AudioObservation, analyze, validate_analysis_config


def test_analysis_rejects_short_fft_input() -> None:
    with pytest.raises(ValueError, match="16384"):
        analyze(np.zeros(16_383, dtype=np.float32), 48_000)


@pytest.mark.parametrize("level", [0.0, 1e-6])
def test_silent_tail_returns_owned_empty_observation(level: float) -> None:
    samples = np.zeros(16_384, dtype=np.float32)
    samples[-2_400:] = level

    result = analyze(samples, 48_000)

    assert not result.has_signal
    assert result.peak_amplitude_fs is None
    assert result.peak_frequency_hz is None
    assert result.peak_level_dbfs is None
    assert all(
        array.ndim == 1 and array.size == 0 and not array.flags.writeable and array.flags.owndata
        for array in (
            result.waveform_samples,
            result.waveform_time_ms,
            result.spectrum_frequency_hz,
            result.spectrum_level_dbfs,
        )
    )


@pytest.mark.parametrize("fft_frames", [0, -2, 2, 3])
def test_validation_rejects_fft_sizes_below_four_or_odd(fft_frames: int) -> None:
    with pytest.raises(ValueError, match="even integer of at least 4"):
        validate_analysis_config(AnalysisConfig(fft_frames=fft_frames), 48_000)


def test_four_frame_hann_analysis_has_finite_results() -> None:
    config = AnalysisConfig(waveform_window_seconds=4 / 48_000, fft_frames=4)

    result = analyze(np.array([-0.25, 0.25, -0.25, 0.25], dtype=np.float32), 48_000, config)

    assert result.has_signal
    assert np.all(np.isfinite(result.spectrum_level_dbfs))


@pytest.mark.parametrize("duration", [0.0, 0.49 / 48_000, 16_384.5 / 48_000])
def test_validation_rejects_waveform_frame_counts_outside_fft(duration: float) -> None:
    with pytest.raises(ValueError, match="waveform"):
        validate_analysis_config(AnalysisConfig(waveform_window_seconds=duration), 48_000)


def test_validation_rejects_waveform_duration_whose_frame_product_overflows() -> None:
    config = AnalysisConfig(waveform_window_seconds=1e308)

    with pytest.raises(ValueError, match="waveform_window_seconds"):
        validate_analysis_config(config, 48_000)


@pytest.mark.parametrize(
    ("minimum_hz", "maximum_hz"),
    [
        (0.0, 20_000.0),
        (-1.0, 20_000.0),
        (20_000.0, 20_000.0),
        (20_001.0, 20_000.0),
        (math.nan, 20_000.0),
        (20.0, math.inf),
        (20.0, 24_000.1),
    ],
)
def test_validation_rejects_invalid_spectrum_bounds(minimum_hz: float, maximum_hz: float) -> None:
    config = AnalysisConfig(spectrum_min_hz=minimum_hz, spectrum_max_hz=maximum_hz)
    with pytest.raises(ValueError, match="spectrum"):
        validate_analysis_config(config, 48_000)


@pytest.mark.parametrize("sample_rate_hz", [0, -48_000, math.nan, math.inf])
def test_validation_rejects_nonpositive_or_nonfinite_sample_rate(sample_rate_hz: float) -> None:
    with pytest.raises(ValueError, match="sample_rate_hz"):
        validate_analysis_config(AnalysisConfig(), sample_rate_hz)  # type: ignore[arg-type]


@pytest.mark.parametrize("floor_dbfs", [0.01, math.nan, math.inf, -math.inf])
def test_validation_rejects_positive_or_nonfinite_floor(floor_dbfs: float) -> None:
    with pytest.raises(ValueError, match="spectrum_floor_dbfs"):
        validate_analysis_config(AnalysisConfig(spectrum_floor_dbfs=floor_dbfs), 48_000)


def test_validation_rejects_window_other_than_hann() -> None:
    with pytest.raises(ValueError, match="window"):
        validate_analysis_config(AnalysisConfig(window="blackman"), 48_000)


def test_analyze_calls_shared_validator_before_inspecting_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = ValueError("shared validator sentinel")
    calls: list[tuple[AnalysisConfig, int]] = []

    def reject(config: AnalysisConfig, sample_rate_hz: int) -> None:
        calls.append((config, sample_rate_hz))
        raise sentinel

    config = AnalysisConfig()
    monkeypatch.setattr(analysis_module, "validate_analysis_config", reject)

    with pytest.raises(ValueError, match="shared validator sentinel") as caught:
        analyze(np.zeros(1, dtype=np.float32), 48_000, config)

    assert caught.value is sentinel
    assert calls == [(config, 48_000)]


def test_audio_observation_is_deeply_immutable_and_owns_flat_arrays() -> None:
    source = np.arange(6, dtype=np.float64).reshape(2, 3)
    observation = AudioObservation(
        has_signal=True,
        waveform_samples=source,
        waveform_time_ms=source,
        spectrum_frequency_hz=source[:, ::-1],
        spectrum_level_dbfs=source[:, ::-1],
        peak_amplitude_fs=1.0,
        peak_frequency_hz=440.0,
        peak_level_dbfs=0.0,
    )
    source.fill(-1.0)

    with pytest.raises(FrozenInstanceError):
        observation.has_signal = False  # type: ignore[misc]
    assert all(
        array.ndim == 1
        and array.flags.c_contiguous
        and array.flags.owndata
        and not array.flags.writeable
        and not np.shares_memory(array, source)
        for array in (
            observation.waveform_samples,
            observation.waveform_time_ms,
            observation.spectrum_frequency_hz,
            observation.spectrum_level_dbfs,
        )
    )
    assert observation.waveform_samples.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_direct_audio_observation_normalizes_each_public_array_dtype() -> None:
    observation = AudioObservation(
        has_signal=True,
        waveform_samples=np.array([1, -1], dtype=np.int16),
        waveform_time_ms=np.array([0, 1], dtype=np.float32),
        spectrum_frequency_hz=np.array([220], dtype=np.float32),
        spectrum_level_dbfs=np.array([-12], dtype=np.float32),
        peak_amplitude_fs=1.0,
        peak_frequency_hz=220.0,
        peak_level_dbfs=-12.0,
    )

    assert observation.waveform_samples.dtype == np.float32
    assert observation.waveform_time_ms.dtype == np.float64
    assert observation.spectrum_frequency_hz.dtype == np.float64
    assert observation.spectrum_level_dbfs.dtype == np.float64


@pytest.mark.parametrize(
    ("waveform_size", "time_size", "frequency_size", "level_size"),
    [(2, 1, 0, 0), (0, 0, 2, 1)],
)
def test_audio_observation_rejects_unpaired_array_lengths(
    waveform_size: int, time_size: int, frequency_size: int, level_size: int
) -> None:
    with pytest.raises(ValueError, match="length"):
        AudioObservation(
            has_signal=True,
            waveform_samples=np.zeros(waveform_size),
            waveform_time_ms=np.zeros(time_size),
            spectrum_frequency_hz=np.zeros(frequency_size),
            spectrum_level_dbfs=np.zeros(level_size),
            peak_amplitude_fs=1.0,
            peak_frequency_hz=440.0,
            peak_level_dbfs=-6.0,
        )


def test_waveform_uses_half_up_frame_count_and_exact_relative_timestamps() -> None:
    config = AnalysisConfig(waveform_window_seconds=2_399.5 / 48_000)
    samples = np.linspace(0.1, 1.0, config.fft_frames, dtype=np.float32)

    result = analyze(samples, 48_000, config)

    expected_samples = samples[-2_400:]
    expected_time_ms = np.arange(2_400, dtype=np.float64) * 1_000.0 / 48_000.0
    np.testing.assert_array_equal(result.waveform_samples, expected_samples)
    np.testing.assert_array_equal(result.waveform_time_ms, expected_time_ms)
    assert result.peak_amplitude_fs == 1.0


def test_waveform_starts_at_most_recent_eligible_rising_zero_crossing() -> None:
    samples = np.full(16_384, -0.5, dtype=np.float32)
    eligible_crossing = 13_000
    samples[eligible_crossing:15_000] = 0.5
    samples[15_000:15_500] = -0.25
    samples[15_500:] = 0.75  # Newer crossing cannot fit a complete waveform window.

    result = analyze(samples, 48_000)

    np.testing.assert_array_equal(
        result.waveform_samples, samples[eligible_crossing : eligible_crossing + 2_400]
    )


def test_waveform_without_eligible_crossing_uses_trailing_window() -> None:
    samples = np.linspace(0.1, 0.9, 16_384, dtype=np.float32)

    result = analyze(samples, 48_000)

    np.testing.assert_array_equal(result.waveform_samples, samples[-2_400:])


def test_published_peak_amplitude_comes_from_trigger_aligned_waveform() -> None:
    samples = np.full(16_384, -0.2, dtype=np.float32)
    samples[13_000:] = 0.2
    samples[13_010] = 0.8

    result = analyze(samples, 48_000)

    assert np.max(np.abs(result.waveform_samples)) == pytest.approx(0.8)
    assert result.peak_amplitude_fs == pytest.approx(0.8)


def test_signal_validity_comes_from_untriggered_tail() -> None:
    samples = np.full(16_384, 0.5, dtype=np.float32)
    samples[-2_400:] = 1e-6

    result = analyze(samples, 48_000)

    assert not result.has_signal
    assert result.waveform_samples.size == 0


def test_bin_centered_sine_has_hann_coherent_gain_corrected_dbfs_peak() -> None:
    frames = np.arange(16_384, dtype=np.float64)
    frequency_hz = 300 * 48_000 / 16_384
    samples = (0.25 * np.sin(math.tau * frequency_hz * frames / 48_000 + 0.37)).astype(np.float32)

    result = analyze(samples, 48_000)

    assert result.peak_frequency_hz == pytest.approx(frequency_hz, abs=0.01)
    assert result.peak_level_dbfs == pytest.approx(-12.041199826559248, abs=0.02)
    assert np.max(result.spectrum_level_dbfs) == pytest.approx(-12.041199826559248, abs=0.02)


def test_spectrum_contains_strictly_increasing_positive_log_range_bins() -> None:
    frames = np.arange(16_384, dtype=np.float64)
    samples = (0.25 * np.sin(math.tau * 440.0 * frames / 48_000)).astype(np.float32)

    result = analyze(samples, 48_000)

    assert result.spectrum_frequency_hz.size > 0
    assert np.all(np.diff(result.spectrum_frequency_hz) > 0.0)
    assert np.all(result.spectrum_frequency_hz > 0.0)
    assert result.spectrum_frequency_hz[0] >= 20.0
    assert result.spectrum_frequency_hz[-1] <= 20_000.0
    assert np.all(result.spectrum_level_dbfs >= -120.0)


def test_nyquist_bin_is_not_doubled() -> None:
    frames = np.arange(16_384)
    samples = (0.25 * np.where(frames % 2 == 0, 1.0, -1.0)).astype(np.float32)
    config = AnalysisConfig(spectrum_min_hz=23_000.0, spectrum_max_hz=24_000.0)

    result = analyze(samples, 48_000, config)

    nyquist_bin = np.flatnonzero(result.spectrum_frequency_hz == 24_000.0)
    assert nyquist_bin.size == 1
    assert result.spectrum_level_dbfs[nyquist_bin[0]] == pytest.approx(
        -12.041199826559248, abs=0.02
    )


def test_analysis_uses_only_the_most_recent_exact_fft_capture() -> None:
    frames = np.arange(16_384, dtype=np.float64)
    old_capture = 0.9 * np.sin(math.tau * 300.0 * frames / 48_000)
    recent_capture = 0.25 * np.sin(math.tau * 900.0 * frames / 48_000)
    samples = np.concatenate((old_capture, recent_capture)).astype(np.float32)

    result = analyze(samples, 48_000)

    assert result.peak_frequency_hz == pytest.approx(900.0, abs=0.1)


def test_below_floor_in_range_spectrum_has_signal_without_spectral_peak() -> None:
    frames = np.arange(16_384, dtype=np.float64)
    frequency_hz = 500 * 48_000 / 16_384
    samples = (0.005 * np.sin(math.tau * frequency_hz * frames / 48_000)).astype(np.float32)
    config = AnalysisConfig(spectrum_floor_dbfs=-40.0)

    result = analyze(samples, 48_000, config)

    assert result.has_signal
    assert result.peak_amplitude_fs is not None
    assert result.peak_frequency_hz is None
    assert result.peak_level_dbfs is None
    np.testing.assert_array_equal(
        result.spectrum_level_dbfs,
        np.full(result.spectrum_level_dbfs.size, -40.0),
    )


def test_frequency_range_between_fft_bins_has_no_spectral_peak() -> None:
    samples = np.full(16_384, 0.25, dtype=np.float32)
    config = AnalysisConfig(spectrum_min_hz=20.1, spectrum_max_hz=20.2)

    result = analyze(samples, 48_000, config)

    assert result.has_signal
    assert result.spectrum_frequency_hz.size == 0
    assert result.spectrum_level_dbfs.size == 0
    assert result.peak_frequency_hz is None
    assert result.peak_level_dbfs is None


def test_live_and_empty_analysis_results_publish_declared_array_dtypes() -> None:
    frames = np.arange(16_384, dtype=np.float64)
    live_samples = (0.25 * np.sin(math.tau * 440.0 * frames / 48_000)).astype(np.float64)

    live = analyze(live_samples, 48_000)
    empty = analyze(np.zeros(16_384, dtype=np.float32), 48_000)

    for observation in (live, empty):
        assert observation.waveform_samples.dtype == np.float32
        assert observation.waveform_time_ms.dtype == np.float64
        assert observation.spectrum_frequency_hz.dtype == np.float64
        assert observation.spectrum_level_dbfs.dtype == np.float64


@pytest.mark.parametrize("frequency_hz", np.linspace(130.8127826502993, 523.2511306011972, 1_001))
@pytest.mark.parametrize("phase", np.linspace(0.0, math.tau, 9, endpoint=False))
def test_clean_sine_peak_is_within_tenth_hz(frequency_hz: float, phase: float) -> None:
    frames = np.arange(16_384, dtype=np.float64)
    samples = (0.25 * np.sin(math.tau * frequency_hz * frames / 48_000 + phase)).astype(np.float32)

    result = analyze(samples, 48_000)

    assert result.peak_frequency_hz is not None
    assert abs(result.peak_frequency_hz - frequency_hz) <= 0.1
