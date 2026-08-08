from __future__ import annotations

from collections.abc import Callable, Iterator
from threading import Event, Thread

import numpy as np

from harpy.analysis import AnalysisConfig, AudioObservation
from harpy.capture import CaptureCoordinator, CaptureState, SampleHistory

SAMPLE_RATE_HZ = 48_000
ANALYSIS_CONFIG = AnalysisConfig(
    waveform_window_seconds=4 / SAMPLE_RATE_HZ,
    fft_frames=4,
)


def _observation(marker: float, *, has_signal: bool = True) -> AudioObservation:
    values = np.array([marker], dtype=np.float32) if has_signal else np.empty(0)
    return AudioObservation(
        has_signal=has_signal,
        waveform_samples=values,
        waveform_time_ms=np.zeros(values.size),
        spectrum_frequency_hz=values,
        spectrum_level_dbfs=np.zeros(values.size),
        peak_amplitude_fs=marker if has_signal else None,
        peak_frequency_hz=marker if has_signal else None,
        peak_level_dbfs=-6.0 if has_signal else None,
    )


def _sequence_analyzer(
    observations: Iterator[AudioObservation],
) -> Callable[..., AudioObservation]:
    def analyzer(*_args: object) -> AudioObservation:
        return next(observations)

    return analyzer


def test_history_snapshot_returns_only_available_frames_without_padding() -> None:
    history = SampleHistory(capacity_frames=8)
    generation = history.begin_generation()
    history.append(np.array([1.0, 2.0, 3.0], dtype=np.float32), generation)

    snapshot = history.snapshot_recent(6)

    assert snapshot.generation == generation
    np.testing.assert_array_equal(snapshot.samples, [1.0, 2.0, 3.0])
    assert history.available_frames == 3


def test_history_capacity_keeps_newest_frames_in_owned_read_only_snapshots() -> None:
    history = SampleHistory(capacity_frames=4)
    generation = history.begin_generation()
    source = np.arange(6, dtype=np.float32)

    assert history.append(source, generation)
    snapshot = history.snapshot_recent(4)
    source.fill(-1.0)

    np.testing.assert_array_equal(snapshot.samples, [2.0, 3.0, 4.0, 5.0])
    assert snapshot.samples.flags.owndata
    assert not snapshot.samples.flags.writeable
    assert not np.shares_memory(snapshot.samples, source)


def test_superseded_append_is_rejected_without_changing_current_history() -> None:
    history = SampleHistory(capacity_frames=4)
    superseded = history.begin_generation()
    current = history.begin_generation()
    assert history.append(np.array([7.0], dtype=np.float32), current)

    assert not history.append(np.array([99.0], dtype=np.float32), superseded)

    snapshot = history.snapshot_recent(4)
    assert snapshot.generation == current
    np.testing.assert_array_equal(snapshot.samples, [7.0])


def test_launch_is_empty_and_reset_enters_empty_with_one_new_generation() -> None:
    history = SampleHistory(capacity_frames=8)
    coordinator = CaptureCoordinator(history, SAMPLE_RATE_HZ, ANALYSIS_CONFIG)

    launched = coordinator.refresh()
    before_reset = history.generation
    reset_generation = coordinator.reset()
    reset = coordinator.refresh()

    assert launched.state is CaptureState.EMPTY
    assert launched.observation is None
    assert reset_generation == before_reset + 1 == history.generation
    assert reset.state is CaptureState.EMPTY
    assert reset.observation is None
    assert reset.generation == reset_generation


def test_begin_enters_measuring_and_insufficient_history_stays_measuring() -> None:
    history = SampleHistory(capacity_frames=8)
    coordinator = CaptureCoordinator(
        history,
        SAMPLE_RATE_HZ,
        ANALYSIS_CONFIG,
        analyzer=lambda *_args: _observation(1.0),
    )

    before_begin = history.generation
    generation = coordinator.begin()
    assert history.append(np.ones(3, dtype=np.float32), generation)
    view = coordinator.refresh()

    assert generation == before_begin + 1 == history.generation
    assert view.state is CaptureState.MEASURING
    assert view.observation is None
    assert view.generation == generation


def test_first_and_later_valid_observations_enter_and_refresh_live() -> None:
    first = _observation(1.0)
    second = _observation(2.0)
    history = SampleHistory(capacity_frames=8)
    coordinator = CaptureCoordinator(
        history,
        SAMPLE_RATE_HZ,
        ANALYSIS_CONFIG,
        analyzer=_sequence_analyzer(iter((first, second))),
    )
    generation = coordinator.begin()
    history.append(np.arange(4, dtype=np.float32), generation)

    first_view = coordinator.refresh()
    history.append(np.array([4.0], dtype=np.float32), generation)
    second_view = coordinator.refresh()

    assert first_view.state is CaptureState.LIVE
    assert first_view.observation is first
    assert second_view.state is CaptureState.LIVE
    assert second_view.observation is second


def test_silence_before_any_live_observation_returns_to_empty() -> None:
    history = SampleHistory(capacity_frames=8)
    coordinator = CaptureCoordinator(
        history,
        SAMPLE_RATE_HZ,
        ANALYSIS_CONFIG,
        analyzer=lambda *_args: _observation(0.0, has_signal=False),
    )
    generation = coordinator.begin()
    history.append(np.zeros(4, dtype=np.float32), generation)

    view = coordinator.refresh()

    assert view.state is CaptureState.EMPTY
    assert view.observation is None


def test_silence_after_live_retains_last_valid_object_as_captured() -> None:
    valid = _observation(1.0)
    silence = _observation(0.0, has_signal=False)
    history = SampleHistory(capacity_frames=8)
    coordinator = CaptureCoordinator(
        history,
        SAMPLE_RATE_HZ,
        ANALYSIS_CONFIG,
        analyzer=_sequence_analyzer(iter((valid, silence))),
    )
    generation = coordinator.begin()
    history.append(np.ones(4, dtype=np.float32), generation)
    assert coordinator.refresh().state is CaptureState.LIVE
    history.append(np.zeros(4, dtype=np.float32), generation)

    captured = coordinator.refresh()

    assert captured.state is CaptureState.CAPTURED
    assert captured.observation is valid


def test_clear_idle_and_active_each_allocate_exactly_one_generation() -> None:
    history = SampleHistory(capacity_frames=8)
    coordinator = CaptureCoordinator(history, SAMPLE_RATE_HZ, ANALYSIS_CONFIG)

    before_idle = history.generation
    idle_generation = coordinator.clear(voice_active=False)
    idle = coordinator.refresh()
    before_active = history.generation
    active_generation = coordinator.clear(voice_active=True)
    active = coordinator.refresh()

    assert idle_generation == before_idle + 1
    assert idle.state is CaptureState.EMPTY
    assert idle.observation is None
    assert active_generation == before_active + 1 == history.generation
    assert active.state is CaptureState.MEASURING
    assert active.observation is None


def test_every_semantic_capture_event_allocates_exactly_one_generation() -> None:
    history = SampleHistory(capacity_frames=8)
    coordinator = CaptureCoordinator(history, SAMPLE_RATE_HZ, ANALYSIS_CONFIG)

    events = (
        coordinator.begin,
        lambda: coordinator.clear(voice_active=False),
        lambda: coordinator.clear(voice_active=True),
        coordinator.reset,
    )
    for event in events:
        before = history.generation
        returned = event()
        assert returned == before + 1 == history.generation


def test_latest_of_two_queued_semantic_events_is_the_only_accepted_generation() -> None:
    history = SampleHistory(capacity_frames=8)
    coordinator = CaptureCoordinator(history, SAMPLE_RATE_HZ, ANALYSIS_CONFIG)

    first_generation = coordinator.begin()
    second_generation = coordinator.clear(voice_active=True)

    assert not history.append(np.array([1.0], dtype=np.float32), first_generation)
    assert history.append(np.array([2.0], dtype=np.float32), second_generation)
    np.testing.assert_array_equal(history.snapshot_recent(8).samples, [2.0])


def test_reset_discards_a_live_capture_and_enters_empty() -> None:
    valid = _observation(1.0)
    history = SampleHistory(capacity_frames=8)
    coordinator = CaptureCoordinator(
        history,
        SAMPLE_RATE_HZ,
        ANALYSIS_CONFIG,
        analyzer=lambda *_args: valid,
    )
    generation = coordinator.begin()
    history.append(np.ones(4, dtype=np.float32), generation)
    assert coordinator.refresh().observation is valid

    before_reset = history.generation
    reset_generation = coordinator.reset()
    reset = coordinator.refresh()

    assert reset_generation == before_reset + 1 == history.generation
    assert reset.state is CaptureState.EMPTY
    assert reset.observation is None


def test_refresh_rejects_analysis_cleared_while_it_was_in_flight() -> None:
    valid = _observation(1.0)
    history = SampleHistory(capacity_frames=8)
    coordinator: CaptureCoordinator

    def clearing_analyzer(*_args: object) -> AudioObservation:
        coordinator.clear(voice_active=False)
        return valid

    coordinator = CaptureCoordinator(
        history,
        SAMPLE_RATE_HZ,
        ANALYSIS_CONFIG,
        analyzer=clearing_analyzer,
    )
    generation = coordinator.begin()
    history.append(np.ones(4, dtype=np.float32), generation)

    view = coordinator.refresh()

    assert view.generation == generation + 1 == history.generation
    assert view.state is CaptureState.EMPTY
    assert view.observation is None


def test_older_same_generation_analysis_cannot_overwrite_newer_live_refresh() -> None:
    initial = _observation(1.0)
    silence = _observation(0.0, has_signal=False)
    newer = _observation(2.0)
    history = SampleHistory(capacity_frames=8)
    older_started = Event()
    release_older = Event()

    def analyzer(samples: np.ndarray, *_args: object) -> AudioObservation:
        marker = float(samples[-1])
        if marker == 0.0:
            older_started.set()
            assert release_older.wait(timeout=1.0), "older analysis was not released"
            return silence
        if marker == 1.0:
            return initial
        assert marker == 2.0
        return newer

    coordinator = CaptureCoordinator(
        history,
        SAMPLE_RATE_HZ,
        ANALYSIS_CONFIG,
        analyzer=analyzer,
    )
    generation = coordinator.begin()
    history.append(np.ones(4, dtype=np.float32), generation)
    assert coordinator.refresh().observation is initial

    history.append(np.zeros(4, dtype=np.float32), generation)
    older_views: list[object] = []
    older_thread = Thread(target=lambda: older_views.append(coordinator.refresh()), daemon=True)
    older_thread.start()
    assert older_started.wait(timeout=1.0), "older analysis did not start"

    try:
        history.append(np.full(4, 2.0, dtype=np.float32), generation)
        newer_view = coordinator.refresh()
    finally:
        release_older.set()
        older_thread.join(timeout=1.0)

    assert not older_thread.is_alive()
    assert len(older_views) == 1
    assert newer_view.state is CaptureState.LIVE
    assert newer_view.observation is newer
    final = coordinator.refresh()
    assert final.state is CaptureState.LIVE
    assert final.observation is newer


def test_overlapping_silence_refreshes_keep_exact_retained_capture() -> None:
    valid = _observation(1.0)
    silence = _observation(0.0, has_signal=False)
    history = SampleHistory(capacity_frames=8)
    first_started = Event()
    second_started = Event()
    release_first = Event()
    release_second = Event()

    def analyzer(samples: np.ndarray, *_args: object) -> AudioObservation:
        marker = float(samples[-1])
        if marker == 1.0:
            return valid
        if marker == 0.0:
            first_started.set()
            assert release_first.wait(timeout=1.0), "first silence was not released"
            return silence
        assert marker == -1.0
        second_started.set()
        assert release_second.wait(timeout=1.0), "second silence was not released"
        return silence

    coordinator = CaptureCoordinator(
        history,
        SAMPLE_RATE_HZ,
        ANALYSIS_CONFIG,
        analyzer=analyzer,
    )
    generation = coordinator.begin()
    history.append(np.ones(4, dtype=np.float32), generation)
    assert coordinator.refresh().observation is valid

    history.append(np.zeros(4, dtype=np.float32), generation)
    first_views: list[object] = []
    first_thread = Thread(target=lambda: first_views.append(coordinator.refresh()), daemon=True)
    first_thread.start()
    assert first_started.wait(timeout=1.0), "first silence did not start"

    history.append(np.full(4, -1.0, dtype=np.float32), generation)
    second_views: list[object] = []
    second_thread = Thread(target=lambda: second_views.append(coordinator.refresh()), daemon=True)
    second_thread.start()
    try:
        assert second_started.wait(timeout=1.0), "second silence did not start"
        release_first.set()
        first_thread.join(timeout=1.0)
        assert len(first_views) == 1
        after_first = coordinator.refresh()
        assert after_first.state is CaptureState.CAPTURED
        assert after_first.observation is valid

        release_second.set()
        second_thread.join(timeout=1.0)
    finally:
        release_first.set()
        release_second.set()
        first_thread.join(timeout=1.0)
        second_thread.join(timeout=1.0)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert len(second_views) == 1
    final = coordinator.refresh()
    assert final.state is CaptureState.CAPTURED
    assert final.observation is valid


def test_refresh_runs_analysis_after_releasing_the_history_lock() -> None:
    valid = _observation(1.0)
    history = SampleHistory(capacity_frames=8)
    append_finished = Event()
    append_results: list[bool] = []
    generation = 0

    def analyzer(*_args: object) -> AudioObservation:
        def append_from_audio_thread() -> None:
            append_results.append(history.append(np.array([5.0], dtype=np.float32), generation))
            append_finished.set()

        Thread(target=append_from_audio_thread, daemon=True).start()
        assert append_finished.wait(timeout=1.0), "analysis ran while the history lock was held"
        return valid

    coordinator = CaptureCoordinator(
        history,
        SAMPLE_RATE_HZ,
        ANALYSIS_CONFIG,
        analyzer=analyzer,
    )
    generation = coordinator.begin()
    history.append(np.ones(4, dtype=np.float32), generation)

    view = coordinator.refresh()

    assert append_results == [True]
    assert view.state is CaptureState.LIVE
    assert view.observation is valid
