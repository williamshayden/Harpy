"""Generation-safe sample history and live capture presentation state."""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock

import numpy as np

from harpy.analysis import (
    AnalysisConfig,
    AudioObservation,
    analyze,
    validate_analysis_config,
)


@dataclass(frozen=True, slots=True)
class HistorySnapshot:
    """An immutable, independently owned view of one history generation."""

    generation: int
    samples: np.ndarray

    def __post_init__(self) -> None:
        owned = np.array(np.asarray(self.samples).reshape(-1), copy=True, order="C")
        owned.setflags(write=False)
        object.__setattr__(self, "samples", owned)


class SampleHistory:
    """A bounded mono sample history accepting only its current generation."""

    def __init__(self, capacity_frames: int) -> None:
        try:
            capacity = operator.index(capacity_frames)
        except TypeError as error:
            raise ValueError("capacity_frames must be a positive integer") from error
        if isinstance(capacity_frames, bool) or capacity <= 0:
            raise ValueError("capacity_frames must be a positive integer")

        self._capacity_frames = capacity
        self._generation = 0
        self._samples = np.empty(0, dtype=np.float32)
        self._lock = Lock()

    @property
    def generation(self) -> int:
        """Return the currently accepted generation identifier."""

        with self._lock:
            return self._generation

    @property
    def available_frames(self) -> int:
        """Return the number of actually available frames without padding."""

        with self._lock:
            return int(self._samples.size)

    def begin_generation(self) -> int:
        """Allocate a generation and atomically clear all prior samples."""

        with self._lock:
            self._generation += 1
            self._samples = np.empty(0, dtype=np.float32)
            return self._generation

    def append(self, samples: np.ndarray, generation: int) -> bool:
        """Append mono samples when ``generation`` is current, otherwise reject them."""

        supplied = np.asarray(samples)
        if supplied.ndim != 1:
            raise ValueError("samples must be a one-dimensional array")
        owned = np.array(supplied, copy=True, order="C")

        with self._lock:
            if generation != self._generation:
                return False
            if owned.size == 0:
                return True
            if owned.size >= self._capacity_frames:
                self._samples = np.array(
                    owned[-self._capacity_frames :],
                    copy=True,
                    order="C",
                )
            else:
                combined = np.concatenate((self._samples, owned))
                self._samples = combined[-self._capacity_frames :]
            return True

    def snapshot_recent(self, frame_count: int) -> HistorySnapshot:
        """Copy up to ``frame_count`` of the newest actually available frames."""

        try:
            requested = operator.index(frame_count)
        except TypeError as error:
            raise ValueError("frame_count must be a nonnegative integer") from error
        if isinstance(frame_count, bool) or requested < 0:
            raise ValueError("frame_count must be a nonnegative integer")

        with self._lock:
            recent = self._samples[-requested:] if requested else self._samples[:0]
            return HistorySnapshot(generation=self._generation, samples=recent)


class CaptureState(StrEnum):
    """Presentation states for live audio capture."""

    EMPTY = "empty"
    MEASURING = "measuring"
    LIVE = "live"
    CAPTURED = "captured"


@dataclass(frozen=True, slots=True)
class CaptureView:
    """Public state and observation for the current capture generation."""

    state: CaptureState
    observation: AudioObservation | None
    generation: int


@dataclass(frozen=True, slots=True)
class _CapturedObservation:
    generation: int
    observation: AudioObservation


_ANALYZING_STATES = frozenset((CaptureState.MEASURING, CaptureState.LIVE))


class CaptureCoordinator:
    """Publish capture state while rejecting superseded in-flight analysis."""

    def __init__(
        self,
        history: SampleHistory,
        sample_rate_hz: int,
        analysis_config: AnalysisConfig,
        analyzer: Callable[..., AudioObservation] = analyze,
    ) -> None:
        validate_analysis_config(analysis_config, sample_rate_hz)
        self._history = history
        self._sample_rate_hz = sample_rate_hz
        self._analysis_config = analysis_config
        self._analyzer = analyzer
        self._lock = Lock()
        self._snapshot_lock = Lock()
        self._state = CaptureState.EMPTY
        self._captured: _CapturedObservation | None = None
        self._generation = history.generation
        self._next_refresh_sequence = 0
        self._published_refresh_sequence = 0

    def begin(self) -> int:
        """Start measuring a newly allocated note/retrigger generation."""

        return self._begin_state(CaptureState.MEASURING)

    def clear(self, voice_active: bool) -> int:
        """Clear capture and measure again only while the voice remains active."""

        state = CaptureState.MEASURING if voice_active else CaptureState.EMPTY
        return self._begin_state(state)

    def reset(self) -> int:
        """Clear capture and enter Empty for reset, patch, or device failure."""

        return self._begin_state(CaptureState.EMPTY)

    def refresh(self) -> CaptureView:
        """Analyze a complete current snapshot and publish it if still current."""

        with self._lock:
            if self._state not in _ANALYZING_STATES:
                return self._view_locked()
            generation = self._generation

        with self._snapshot_lock:
            snapshot = self._history.snapshot_recent(self._analysis_config.fft_frames)
            self._next_refresh_sequence += 1
            refresh_sequence = self._next_refresh_sequence
        if (
            snapshot.generation != generation
            or snapshot.samples.size < self._analysis_config.fft_frames
        ):
            return self._current_view()

        observation = self._analyzer(
            snapshot.samples,
            self._sample_rate_hz,
            self._analysis_config,
        )
        candidate = _CapturedObservation(
            generation=snapshot.generation,
            observation=observation,
        )

        with self._lock:
            if (
                self._generation != candidate.generation
                or self._history.generation != candidate.generation
                or refresh_sequence <= self._published_refresh_sequence
            ):
                return self._view_locked()

            self._published_refresh_sequence = refresh_sequence
            if observation.has_signal:
                self._captured = candidate
                self._state = CaptureState.LIVE
            elif (
                self._state in (CaptureState.LIVE, CaptureState.CAPTURED)
                and self._captured is not None
            ):
                self._state = CaptureState.CAPTURED
            else:
                self._captured = None
                self._state = CaptureState.EMPTY
            return self._view_locked()

    def _begin_state(self, state: CaptureState) -> int:
        with self._lock:
            generation = self._history.begin_generation()
            self._generation = generation
            self._state = state
            self._captured = None
            return generation

    def _current_view(self) -> CaptureView:
        with self._lock:
            return self._view_locked()

    def _view_locked(self) -> CaptureView:
        observation = self._captured.observation if self._captured is not None else None
        return CaptureView(
            state=self._state,
            observation=observation,
            generation=self._generation,
        )
