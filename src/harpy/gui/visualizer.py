from __future__ import annotations

import threading

import numpy as np


class SampleRingBuffer:
    def __init__(self, capacity_frames: int) -> None:
        if not isinstance(capacity_frames, int) or capacity_frames <= 0:
            raise ValueError("capacity_frames must be a positive integer")
        self._capacity = capacity_frames
        self._data = np.zeros(capacity_frames, dtype=np.float32)
        self._write_index = 0
        self._size = 0
        self._lock = threading.Lock()

    def append(self, samples: np.ndarray) -> bool:
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return True
        if not self._lock.acquire(blocking=False):
            return False
        try:
            values = values[-self._capacity :]
            first_count = min(values.size, self._capacity - self._write_index)
            self._data[self._write_index : self._write_index + first_count] = values[:first_count]
            second_count = values.size - first_count
            if second_count:
                self._data[:second_count] = values[first_count:]
            self._write_index = (self._write_index + values.size) % self._capacity
            self._size = min(self._capacity, self._size + values.size)
            return True
        finally:
            self._lock.release()

    def clear(self) -> None:
        with self._lock:
            self._data.fill(0.0)
            self._write_index = 0
            self._size = 0

    def snapshot(self, frame_count: int) -> np.ndarray:
        if not isinstance(frame_count, int) or not 0 <= frame_count <= self._capacity:
            raise ValueError("frame_count must be between zero and capacity_frames")
        with self._lock:
            available = min(frame_count, self._size)
            output = np.zeros(frame_count, dtype=np.float32)
            if available == 0:
                return output
            start = (self._write_index - available) % self._capacity
            first_count = min(available, self._capacity - start)
            destination = frame_count - available
            output[destination : destination + first_count] = self._data[
                start : start + first_count
            ]
            if first_count < available:
                output[destination + first_count :] = self._data[: available - first_count]
            return output


def waveform_time_ms(frame_count: int, sample_rate_hz: int) -> np.ndarray:
    if frame_count < 0 or sample_rate_hz <= 0:
        raise ValueError("frame_count must be non-negative and sample_rate_hz positive")
    if frame_count == 0:
        return np.empty(0, dtype=np.float64)
    return (np.arange(frame_count, dtype=np.float64) - (frame_count - 1)) * (
        1_000.0 / sample_rate_hz
    )


def spectrum_dbfs(
    samples: np.ndarray,
    sample_rate_hz: int,
    *,
    fft_frames: int = 4_096,
    floor_dbfs: float = -120.0,
) -> tuple[np.ndarray, np.ndarray]:
    if fft_frames <= 0 or fft_frames % 2:
        raise ValueError("fft_frames must be a positive even integer")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    frame = np.zeros(fft_frames, dtype=np.float64)
    copied = min(values.size, fft_frames)
    if copied:
        frame[-copied:] = values[-copied:]
    window = np.hanning(fft_frames)
    magnitudes = np.abs(np.fft.rfft(frame * window)) / np.sum(window)
    if magnitudes.size > 2:
        magnitudes[1:-1] *= 2.0
    floor_amplitude = 10.0 ** (floor_dbfs / 20.0)
    levels = 20.0 * np.log10(np.maximum(magnitudes, floor_amplitude))
    frequencies = np.fft.rfftfreq(fft_frames, d=1.0 / sample_rate_hz)
    return frequencies, levels
