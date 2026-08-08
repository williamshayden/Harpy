import numpy as np
import pytest

from harpy.gui.visualizer import SampleRingBuffer, spectrum_dbfs, waveform_time_ms


def test_ring_buffer_keeps_only_newest_samples() -> None:
    buffer = SampleRingBuffer(capacity_frames=5)
    buffer.append(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    buffer.append(np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))
    np.testing.assert_array_equal(buffer.snapshot(5), [3.0, 4.0, 5.0, 6.0, 7.0])


def test_snapshot_left_pads_missing_history_with_zero() -> None:
    buffer = SampleRingBuffer(capacity_frames=8)
    buffer.append(np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(buffer.snapshot(4), [0.0, 0.0, 1.0, 2.0])


def test_snapshot_is_a_copy_and_clear_restores_silence() -> None:
    buffer = SampleRingBuffer(capacity_frames=4)
    buffer.append(np.ones(4, dtype=np.float32))
    snapshot = buffer.snapshot(4)
    snapshot[:] = 9.0
    np.testing.assert_array_equal(buffer.snapshot(4), np.ones(4))
    buffer.clear()
    np.testing.assert_array_equal(buffer.snapshot(4), np.zeros(4))


def test_bin_centered_sine_reports_correct_dbfs() -> None:
    sample_rate = 48_000
    frame_count = 4_096
    frequency_hz = 375.0
    amplitude = 0.25
    time = np.arange(frame_count, dtype=np.float64) / sample_rate
    samples = amplitude * np.sin(2.0 * np.pi * frequency_hz * time)
    frequencies, levels = spectrum_dbfs(samples, sample_rate, fft_frames=frame_count)
    peak_index = int(np.argmax(levels))
    assert frequencies[peak_index] == pytest.approx(frequency_hz)
    assert levels[peak_index] == pytest.approx(20.0 * np.log10(amplitude), abs=0.1)
    assert np.min(levels) >= -120.0


def test_waveform_axis_is_in_milliseconds() -> None:
    np.testing.assert_allclose(
        waveform_time_ms(3, sample_rate_hz=1_000),
        [-2.0, -1.0, 0.0],
    )
