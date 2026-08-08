from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from threading import Event, Thread

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QObject, Signal
from PySide6.QtMultimedia import QAudio, QAudioFormat

import harpy.gui.qt_audio as qt_audio
from harpy.analysis import AnalysisConfig
from harpy.capture import CaptureCoordinator, SampleHistory
from harpy.gui.qt_audio import (
    QtAudioBackend,
    SynthAudioSource,
    audio_format_candidates,
    choose_audio_format,
    encode_mono_samples,
)
from harpy.gui.workbench_controller import WorkbenchController
from harpy.gui.workbench_spec import WorkbenchSpec
from harpy.playback import AudioCommand, AudioCommandKind
from harpy.synth.models import EnvelopeConfig, RenderConfig, SynthPatch
from harpy.tuning import Tuning

SAMPLE_RATE_HZ = 48_000


def short_patch(
    *,
    output_gain_dbfs: float = -6.0,
    release_frames: int = 1,
) -> SynthPatch:
    one_frame = 1 / SAMPLE_RATE_HZ
    return SynthPatch(
        envelope=EnvelopeConfig(
            attack_seconds=one_frame,
            decay_seconds=one_frame,
            sustain_db=-6.0,
            release_seconds=release_frames / SAMPLE_RATE_HZ,
        ),
        output_gain_dbfs=output_gain_dbfs,
    )


def source_setup(
    *,
    patch: SynthPatch | None = None,
    generation: int = 0,
) -> tuple[SynthAudioSource, SampleHistory, RenderConfig, QAudioFormat]:
    render = RenderConfig(sample_rate_hz=SAMPLE_RATE_HZ, block_frames=4)
    history = SampleHistory(capacity_frames=64)
    for _ in range(generation):
        history.begin_generation()
    audio_format = audio_format_candidates(render.sample_rate_hz)[1]
    source = SynthAudioSource(
        render,
        patch or short_patch(),
        audio_format,
        history,
        generation,
    )
    return source, history, render, audio_format


def read_block(
    source: SynthAudioSource,
    render: RenderConfig,
    audio_format: QAudioFormat,
) -> np.ndarray:
    byte_count = render.block_frames * audio_format.bytesPerFrame()
    return np.frombuffer(source.readData(byte_count), dtype=np.float32)


class FormatDevice:
    def __init__(self, accepted: set[tuple[int, QAudioFormat.SampleFormat]]) -> None:
        self.accepted = accepted
        self.probed: list[tuple[int, QAudioFormat.SampleFormat]] = []

    def isFormatSupported(self, audio_format: QAudioFormat) -> bool:
        value = (audio_format.channelCount(), audio_format.sampleFormat())
        self.probed.append(value)
        return value in self.accepted


def test_format_candidates_have_approved_priority() -> None:
    values = [
        (item.channelCount(), item.sampleFormat()) for item in audio_format_candidates(48_000)
    ]
    assert values == [
        (2, QAudioFormat.SampleFormat.Float),
        (1, QAudioFormat.SampleFormat.Float),
        (2, QAudioFormat.SampleFormat.Int16),
        (1, QAudioFormat.SampleFormat.Int16),
    ]


def test_choose_audio_format_stops_at_first_supported_candidate() -> None:
    device = FormatDevice({(2, QAudioFormat.SampleFormat.Int16)})

    selected = choose_audio_format(device, 48_000)

    assert selected is not None
    assert selected.channelCount() == 2
    assert selected.sampleFormat() is QAudioFormat.SampleFormat.Int16
    assert device.probed == [
        (2, QAudioFormat.SampleFormat.Float),
        (1, QAudioFormat.SampleFormat.Float),
        (2, QAudioFormat.SampleFormat.Int16),
    ]


def test_float_stereo_encoding_duplicates_mono() -> None:
    audio_format = audio_format_candidates(48_000)[0]

    encoded = encode_mono_samples(np.array([0.25, -0.5], dtype=np.float32), audio_format)

    np.testing.assert_array_equal(
        np.frombuffer(encoded, dtype=np.float32),
        [0.25, 0.25, -0.5, -0.5],
    )


def test_int16_encoding_clips_rounds_and_duplicates_only_for_stereo() -> None:
    stereo_format = audio_format_candidates(48_000)[2]
    mono_format = audio_format_candidates(48_000)[3]
    samples = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32)

    stereo = np.frombuffer(encode_mono_samples(samples, stereo_format), dtype=np.int16)
    mono = np.frombuffer(encode_mono_samples(samples, mono_format), dtype=np.int16)

    np.testing.assert_array_equal(
        stereo.reshape(-1, 2),
        [[-32767, -32767], [-16384, -16384], [16384, 16384], [32767, 32767]],
    )
    np.testing.assert_array_equal(mono, [-32767, -16384, 16384, 32767])


def test_source_is_sequential_and_advertises_pull_bytes() -> None:
    source, _, _, _ = source_setup()

    assert source.isSequential()
    assert source.bytesAvailable() > 0


def test_note_on_adopts_generation_and_appends_the_rendered_mono_samples() -> None:
    source, history, render, audio_format = source_setup(generation=1)
    source.submit(
        AudioCommand(
            AudioCommandKind.NOTE_ON,
            frequency_hz=220.0,
            generation=1,
        )
    )

    samples = read_block(source, render, audio_format)

    assert np.any(samples != 0.0)
    snapshot = history.snapshot_recent(render.block_frames)
    assert snapshot.generation == 1
    np.testing.assert_array_equal(snapshot.samples, samples)


def test_retune_changes_frequency_without_resetting_phase_or_envelope() -> None:
    retuned, _, render, audio_format = source_setup()
    control, _, _, _ = source_setup()
    note_on = AudioCommand(AudioCommandKind.NOTE_ON, frequency_hz=220.0, generation=0)
    retuned.submit(note_on)
    control.submit(note_on)
    np.testing.assert_array_equal(
        read_block(retuned, render, audio_format),
        read_block(control, render, audio_format),
    )
    retuned.submit(AudioCommand(AudioCommandKind.RETUNE, frequency_hz=330.0))

    after_retune = read_block(retuned, render, audio_format)
    without_retune = read_block(control, render, audio_format)

    assert after_retune[0] == pytest.approx(without_retune[0], abs=1e-7)
    assert not np.array_equal(after_retune[1:], without_retune[1:])


def test_retune_after_idle_is_a_harmless_noop() -> None:
    idle_source, _, idle_render, idle_format = source_setup()

    idle_source.submit(AudioCommand(AudioCommandKind.RETUNE, frequency_hz=330.0))

    np.testing.assert_array_equal(
        read_block(idle_source, idle_render, idle_format),
        np.zeros(idle_render.block_frames, dtype=np.float32),
    )


def test_note_off_releases_and_emits_natural_idle_once_for_current_generation() -> None:
    source, _, render, audio_format = source_setup(
        patch=short_patch(release_frames=4),
        generation=1,
    )
    idle_generations: list[int] = []
    source.voice_idle.connect(idle_generations.append)
    source.submit(AudioCommand(AudioCommandKind.NOTE_ON, frequency_hz=220.0, generation=1))
    read_block(source, render, audio_format)
    source.submit(AudioCommand(AudioCommandKind.NOTE_OFF))

    release = read_block(source, render, audio_format)
    read_block(source, render, audio_format)

    assert np.any(release != 0.0)
    assert idle_generations == [1]


def test_clear_capture_changes_only_the_local_append_generation() -> None:
    source, history, render, audio_format = source_setup()
    source.submit(AudioCommand(AudioCommandKind.NOTE_ON, frequency_hz=220.0, generation=0))
    read_block(source, render, audio_format)
    before = history.snapshot_recent(64)
    source.submit(AudioCommand(AudioCommandKind.CLEAR_CAPTURE, generation=1))

    still_live = read_block(source, render, audio_format)

    assert np.any(still_live != 0.0)
    unchanged = history.snapshot_recent(64)
    assert unchanged.generation == 0
    np.testing.assert_array_equal(unchanged.samples, before.samples)
    assert history.begin_generation() == 1
    read_block(source, render, audio_format)
    assert history.available_frames == render.block_frames


def test_reset_preempts_partial_pcm_and_does_not_allocate_or_clear_history() -> None:
    source, history, render, audio_format = source_setup()
    source.submit(AudioCommand(AudioCommandKind.NOTE_ON, frequency_hz=220.0, generation=0))
    half_block_bytes = render.block_frames * audio_format.bytesPerFrame() // 2
    assert np.any(np.frombuffer(source.readData(half_block_bytes), dtype=np.float32) != 0.0)
    assert history.begin_generation() == 1
    source.submit(AudioCommand(AudioCommandKind.RESET, generation=1))

    after_reset = np.frombuffer(source.readData(half_block_bytes), dtype=np.float32)

    np.testing.assert_array_equal(after_reset, np.zeros(after_reset.size, dtype=np.float32))
    assert history.generation == 1
    assert history.available_frames == render.block_frames


def test_replace_patch_preempts_partial_pcm_without_mixing_old_patch() -> None:
    old_patch = short_patch(output_gain_dbfs=0.0)
    replacement = short_patch(output_gain_dbfs=-24.0)
    source, history, render, audio_format = source_setup(patch=old_patch)
    source.submit(AudioCommand(AudioCommandKind.NOTE_ON, frequency_hz=220.0, generation=0))
    half_block_bytes = render.block_frames * audio_format.bytesPerFrame() // 2
    assert np.any(np.frombuffer(source.readData(half_block_bytes), dtype=np.float32) != 0.0)
    assert history.begin_generation() == 1
    source.submit(
        AudioCommand(
            AudioCommandKind.REPLACE_PATCH,
            patch=replacement,
            generation=1,
        )
    )
    source.submit(AudioCommand(AudioCommandKind.NOTE_ON, frequency_hz=220.0, generation=1))

    actual = read_block(source, render, audio_format)
    expected, _, expected_render, expected_format = source_setup(
        patch=replacement,
        generation=1,
    )
    expected.submit(AudioCommand(AudioCommandKind.NOTE_ON, frequency_hz=220.0, generation=1))

    np.testing.assert_array_equal(actual, read_block(expected, expected_render, expected_format))


@pytest.mark.parametrize("kind", [AudioCommandKind.RESET, AudioCommandKind.REPLACE_PATCH])
@pytest.mark.parametrize("partial_staging", [False, True], ids=["new-block", "partial-staging"])
def test_reset_and_replace_submit_linearize_against_inflight_pcm_return(
    monkeypatch,
    kind: AudioCommandKind,
    partial_staging: bool,
) -> None:
    source, history, render, audio_format = source_setup()
    source.submit(AudioCommand(AudioCommandKind.NOTE_ON, frequency_hz=220.0, generation=0))
    block_bytes = render.block_frames * audio_format.bytesPerFrame()
    source.readData(block_bytes // 2 if partial_staging else block_bytes)
    generation = history.begin_generation()
    if kind is AudioCommandKind.REPLACE_PATCH:
        command = AudioCommand(
            kind, patch=short_patch(output_gain_dbfs=-18.0), generation=generation
        )
    else:
        command = AudioCommand(kind, generation=generation)
    drain_completed = Event()
    allow_output_commit = Event()
    submit_started = Event()
    submit_returned = Event()
    output: list[np.ndarray] = []
    completion_order: list[str] = []
    original_drain = source._drain_commands
    drain_call_count = 0
    target_drain_call = 1 if partial_staging else 2

    def pause_after_drain() -> None:
        nonlocal drain_call_count
        original_drain()
        drain_call_count += 1
        if drain_call_count == target_drain_call:
            drain_completed.set()
            assert allow_output_commit.wait(timeout=2.0)

    monkeypatch.setattr(source, "_drain_commands", pause_after_drain)

    def read_pcm() -> None:
        output.append(np.frombuffer(source.readData(block_bytes // 2), dtype=np.float32))
        completion_order.append("read")

    def submit_reset_or_replace() -> None:
        submit_started.set()
        source.submit(command)
        completion_order.append("submit")
        submit_returned.set()

    reader = Thread(target=read_pcm)
    reader.start()
    assert drain_completed.wait(timeout=2.0)
    submitter = Thread(target=submit_reset_or_replace)
    submitter.start()
    assert submit_started.wait(timeout=2.0)
    submit_returned.wait(timeout=0.5)
    allow_output_commit.set()
    reader.join(timeout=2.0)
    submitter.join(timeout=2.0)

    assert not reader.is_alive()
    assert not submitter.is_alive()
    assert len(output) == 1
    assert set(completion_order) == {"read", "submit"}
    if completion_order.index("submit") < completion_order.index("read"):
        np.testing.assert_array_equal(output[0], np.zeros(output[0].size, dtype=np.float32))
    after_submit = np.frombuffer(source.readData(block_bytes // 2), dtype=np.float32)
    np.testing.assert_array_equal(
        after_submit,
        np.zeros(after_submit.size, dtype=np.float32),
    )


@pytest.mark.parametrize("kind", [AudioCommandKind.RESET, AudioCommandKind.REPLACE_PATCH])
def test_reset_and_replace_are_exactly_silent_without_false_idle(kind: AudioCommandKind) -> None:
    source, history, render, audio_format = source_setup()
    idle_generations: list[int] = []
    source.voice_idle.connect(idle_generations.append)
    read_block(source, render, audio_format)
    source.submit(AudioCommand(AudioCommandKind.NOTE_ON, frequency_hz=220.0, generation=0))
    read_block(source, render, audio_format)
    assert history.begin_generation() == 1
    if kind is AudioCommandKind.REPLACE_PATCH:
        command = AudioCommand(kind, patch=short_patch(output_gain_dbfs=-18.0), generation=1)
    else:
        command = AudioCommand(kind, generation=1)
    source.submit(command)

    samples = read_block(source, render, audio_format)

    np.testing.assert_array_equal(samples, np.zeros(render.block_frames, dtype=np.float32))
    assert idle_generations == []


def test_newer_queued_generation_tags_render_and_rejects_delayed_old_append() -> None:
    source, history, render, audio_format = source_setup()
    first_generation = history.begin_generation()
    second_generation = history.begin_generation()
    source.submit(
        AudioCommand(
            AudioCommandKind.NOTE_ON,
            frequency_hz=220.0,
            generation=first_generation,
        )
    )
    source.submit(AudioCommand(AudioCommandKind.CLEAR_CAPTURE, generation=second_generation))

    rendered = read_block(source, render, audio_format)
    before_delayed_append = history.snapshot_recent(render.block_frames)

    assert np.any(rendered != 0.0)
    assert before_delayed_append.generation == second_generation
    np.testing.assert_array_equal(before_delayed_append.samples, rendered)
    assert not history.append(np.ones(2, dtype=np.float32), first_generation)
    np.testing.assert_array_equal(
        history.snapshot_recent(render.block_frames).samples,
        before_delayed_append.samples,
    )


def test_clear_after_immediate_note_off_emits_only_the_final_generation_idle() -> None:
    source, history, render, audio_format = source_setup()
    idle_generations: list[int] = []
    source.voice_idle.connect(idle_generations.append)
    note_generation = history.begin_generation()
    clear_generation = history.begin_generation()
    source.submit(
        AudioCommand(
            AudioCommandKind.NOTE_ON,
            frequency_hz=220.0,
            generation=note_generation,
        )
    )
    source.submit(AudioCommand(AudioCommandKind.NOTE_OFF))
    source.submit(AudioCommand(AudioCommandKind.CLEAR_CAPTURE, generation=clear_generation))

    read_block(source, render, audio_format)

    assert idle_generations == [clear_generation]


def test_clear_waiting_on_release_completion_gets_new_generation_idle(
    monkeypatch,
    qapp,
) -> None:
    source, history, render, audio_format = source_setup(patch=short_patch(release_frames=4))
    idle_generations: list[int] = []
    source.voice_idle.connect(idle_generations.append)
    source.submit(AudioCommand(AudioCommandKind.NOTE_ON, frequency_hz=220.0, generation=0))
    read_block(source, render, audio_format)
    source.submit(AudioCommand(AudioCommandKind.NOTE_OFF))
    release_rendered = Event()
    allow_release_return = Event()
    clear_started = Event()
    original_render = source._engine.render

    def pause_after_release_render(frame_count: int) -> np.ndarray:
        samples = original_render(frame_count)
        release_rendered.set()
        assert allow_release_return.wait(timeout=2.0)
        return samples

    monkeypatch.setattr(source._engine, "render", pause_after_release_render)
    reader = Thread(target=read_block, args=(source, render, audio_format))
    reader.start()
    assert release_rendered.wait(timeout=2.0)
    clear_generation = history.begin_generation()

    def submit_clear() -> None:
        clear_started.set()
        source.submit(AudioCommand(AudioCommandKind.CLEAR_CAPTURE, generation=clear_generation))

    submitter = Thread(target=submit_clear)
    submitter.start()
    assert clear_started.wait(timeout=2.0)
    allow_release_return.set()
    reader.join(timeout=2.0)
    submitter.join(timeout=2.0)
    read_block(source, render, audio_format)
    qapp.processEvents()

    assert not reader.is_alive()
    assert not submitter.is_alive()
    assert idle_generations.count(clear_generation) == 1


class FakeSignal:
    def __init__(self) -> None:
        self.callbacks: list[Callable[[], None]] = []

    def connect(self, callback: Callable[[], None]) -> None:
        self.callbacks.append(callback)

    def disconnect(self, callback: Callable[[], None]) -> None:
        self.callbacks.remove(callback)

    def emit(self) -> None:
        for callback in tuple(self.callbacks):
            callback()


class DefaultDevice(FormatDevice):
    def isNull(self) -> bool:
        return False

    def description(self) -> str:
        return "Fake speakers"


class NullDevice(DefaultDevice):
    def __init__(self) -> None:
        super().__init__(set())

    def isNull(self) -> bool:
        return True


class FakeMediaDevices:
    def __init__(self, device: DefaultDevice) -> None:
        self.device = device
        self.audioOutputsChanged = FakeSignal()

    def defaultAudioOutput(self) -> DefaultDevice:
        return self.device


class FakeSink(QObject):
    stateChanged = Signal(object)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        start_error: object | None = None,
    ) -> None:
        super().__init__(parent)
        self.source: SynthAudioSource | None = None
        self.reset_count = 0
        self.current_error: object = QAudio.Error.NoError
        self.start_error = start_error

    def start(self, source: SynthAudioSource) -> None:
        self.source = source
        if self.start_error is not None:
            self.current_error = self.start_error
            self.stateChanged.emit(QAudio.State.StoppedState)

    def reset(self) -> None:
        self.reset_count += 1

    def error(self) -> object:
        return self.current_error


class ForeignAudioError(Enum):
    NoError = 0
    OpenError = 1


class ForeignAudioState(Enum):
    StoppedState = 2


def backend_setup(
    *,
    device: DefaultDevice | None = None,
    patch: SynthPatch | None = None,
) -> tuple[
    QtAudioBackend,
    FakeMediaDevices,
    list[FakeSink],
    SampleHistory,
    RenderConfig,
]:
    render = RenderConfig(sample_rate_hz=SAMPLE_RATE_HZ, block_frames=4)
    history = SampleHistory(capacity_frames=64)
    media_devices = FakeMediaDevices(
        device or DefaultDevice({(1, QAudioFormat.SampleFormat.Float)})
    )
    sinks: list[FakeSink] = []

    def sink_factory(_device: object, _format: object, owner: QObject) -> FakeSink:
        sink = FakeSink(owner)
        sinks.append(sink)
        return sink

    backend = QtAudioBackend(
        render,
        patch or short_patch(),
        history,
        media_devices=media_devices,
        sink_factory=sink_factory,
    )
    return backend, media_devices, sinks, history, render


def test_healthy_start_emits_availability_without_device_copy(qapp) -> None:
    backend, _, sinks, _, _ = backend_setup()
    availability: list[bool] = []
    failures: list[str] = []
    backend.availability_changed.connect(availability.append)
    backend.audio_failure.connect(failures.append)

    backend.start()

    assert availability == [True]
    assert failures == []
    assert len(sinks) == 1
    assert sinks[0].source is not None


def test_default_media_devices_instance_supplies_default_output(qapp, monkeypatch) -> None:
    device = DefaultDevice({(1, QAudioFormat.SampleFormat.Float)})
    media_devices = FakeMediaDevices(device)
    sinks: list[FakeSink] = []
    monkeypatch.setattr(qt_audio, "QMediaDevices", lambda _owner: media_devices)
    backend = QtAudioBackend(
        RenderConfig(block_frames=4),
        short_patch(),
        SampleHistory(capacity_frames=64),
        sink_factory=lambda *_args: sinks.append(FakeSink()) or sinks[-1],
    )
    availability: list[bool] = []
    backend.availability_changed.connect(availability.append)

    backend.start()

    assert availability == [True]
    assert media_devices.device is device
    assert len(sinks) == 1


@pytest.mark.parametrize(
    ("device", "message"),
    [
        (NullDevice(), "No default audio output"),
        (
            DefaultDevice(set()),
            "Default output has no compatible 48 kHz format",
        ),
    ],
)
def test_startup_device_failure_emits_unavailable_failure_and_force_stop(
    qapp,
    device: DefaultDevice,
    message: str,
) -> None:
    backend, _, _, _, _ = backend_setup(device=device)
    availability: list[bool] = []
    failures: list[str] = []
    forced_stops: list[bool] = []
    backend.availability_changed.connect(availability.append)
    backend.audio_failure.connect(failures.append)
    backend.force_stop_requested.connect(lambda: forced_stops.append(True))

    backend.start()
    backend.start()

    assert availability == [False]
    assert failures == [message]
    assert forced_stops == [True]


def test_repeated_sink_failure_is_reported_once_and_force_stops(qapp) -> None:
    backend, _, sinks, _, _ = backend_setup()
    availability: list[bool] = []
    forced_stops: list[bool] = []
    failures: list[str] = []
    backend.availability_changed.connect(availability.append)
    backend.force_stop_requested.connect(lambda: forced_stops.append(True))
    backend.audio_failure.connect(failures.append)
    backend.start()
    sink = sinks[0]
    sink.current_error = QAudio.Error.OpenError

    sink.stateChanged.emit(QAudio.State.StoppedState)
    sink.stateChanged.emit(QAudio.State.StoppedState)

    assert availability == [True, False]
    assert forced_stops == [True]
    assert len(failures) == 1
    assert "OpenError" in failures[0]
    assert len(sinks) == 1


def test_backend_compares_audio_codes_across_qt_binding_enum_types(qapp) -> None:
    backend, _, sinks, _, _ = backend_setup()
    availability: list[bool] = []
    failures: list[str] = []
    backend.availability_changed.connect(availability.append)
    backend.audio_failure.connect(failures.append)
    backend.start()
    sink = sinks[0]
    sink.current_error = ForeignAudioError.OpenError

    sink.stateChanged.emit(ForeignAudioState.StoppedState)

    assert availability == [True, False]
    assert failures == ["Audio output failed: OpenError"]


def test_output_hotplug_rebuilds_once_and_deletes_disconnected_qobjects(qapp) -> None:
    backend, media_devices, sinks, _, _ = backend_setup()
    forced_stops: list[bool] = []
    failures: list[str] = []
    availability: list[bool] = []
    backend.force_stop_requested.connect(lambda: forced_stops.append(True))
    backend.audio_failure.connect(failures.append)
    backend.availability_changed.connect(availability.append)
    backend.start()
    old_sink = sinks[0]
    old_source = old_sink.source
    assert old_source is not None
    destroyed: list[str] = []
    old_sink.destroyed.connect(lambda: destroyed.append("sink"))
    old_source.destroyed.connect(lambda: destroyed.append("source"))

    media_devices.audioOutputsChanged.emit()
    old_sink.current_error = QAudio.Error.OpenError
    old_sink.stateChanged.emit(QAudio.State.StoppedState)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    assert availability == [True, True]
    assert forced_stops == [True]
    assert failures == []
    assert len(sinks) == 2
    assert old_sink.reset_count == 1
    assert set(destroyed) == {"sink", "source"}
    assert len(backend.children()) == 2


def test_hotplug_to_null_output_reports_and_force_stops_once(qapp) -> None:
    backend, media_devices, sinks, _, _ = backend_setup()
    availability: list[bool] = []
    failures: list[str] = []
    forced_stops: list[bool] = []
    backend.availability_changed.connect(availability.append)
    backend.audio_failure.connect(failures.append)
    backend.force_stop_requested.connect(lambda: forced_stops.append(True))
    backend.start()
    media_devices.device = NullDevice()

    media_devices.audioOutputsChanged.emit()
    media_devices.audioOutputsChanged.emit()

    assert availability == [True, False]
    assert failures == ["No default audio output"]
    assert forced_stops == [True]
    assert len(sinks) == 1
    assert sinks[0].reset_count == 1


def test_hotplug_sink_start_failure_reuses_stop_then_runtime_failure_stops_once(qapp) -> None:
    render = RenderConfig(sample_rate_hz=SAMPLE_RATE_HZ, block_frames=4)
    history = SampleHistory(capacity_frames=64)
    media_devices = FakeMediaDevices(DefaultDevice({(1, QAudioFormat.SampleFormat.Float)}))
    start_errors = iter((None, QAudio.Error.OpenError, None))
    sinks: list[FakeSink] = []

    def sink_factory(_device: object, _format: object, owner: QObject) -> FakeSink:
        sink = FakeSink(owner, start_error=next(start_errors))
        sinks.append(sink)
        return sink

    backend = QtAudioBackend(
        render,
        short_patch(),
        history,
        media_devices=media_devices,
        sink_factory=sink_factory,
    )
    availability: list[bool] = []
    forced_stops: list[bool] = []
    failures: list[str] = []
    backend.availability_changed.connect(availability.append)
    backend.force_stop_requested.connect(lambda: forced_stops.append(True))
    backend.audio_failure.connect(failures.append)
    backend.start()

    media_devices.audioOutputsChanged.emit()

    assert availability == [True, False]
    assert forced_stops == [True]
    assert failures == ["Audio output failed: OpenError"]
    assert len(sinks) == 2

    media_devices.audioOutputsChanged.emit()
    assert availability == [True, False, True]
    recovered_sink = sinks[2]
    recovered_sink.current_error = QAudio.Error.OpenError

    recovered_sink.stateChanged.emit(QAudio.State.StoppedState)

    assert availability == [True, False, True, False]
    assert forced_stops == [True, True]
    assert failures == [
        "Audio output failed: OpenError",
        "Audio output failed: OpenError",
    ]


def test_successful_recovery_clears_failure_without_healthy_status_copy(qapp) -> None:
    backend, media_devices, sinks, _, _ = backend_setup(device=NullDevice())
    availability: list[bool] = []
    failures: list[str] = []
    backend.availability_changed.connect(availability.append)
    backend.audio_failure.connect(failures.append)
    backend.start()
    media_devices.device = DefaultDevice({(1, QAudioFormat.SampleFormat.Float)})

    media_devices.audioOutputsChanged.emit()

    assert availability == [False, True]
    assert failures == ["No default audio output"]
    assert len(sinks) == 1


def test_control_commands_cache_while_unavailable_and_rebuild_idle(qapp) -> None:
    replacement = short_patch(output_gain_dbfs=-24.0)
    backend, media_devices, sinks, history, render = backend_setup(device=NullDevice())
    backend.start()
    replacement_generation = history.begin_generation()
    backend.submit(
        AudioCommand(
            AudioCommandKind.REPLACE_PATCH,
            patch=replacement,
            generation=replacement_generation,
        )
    )
    reset_generation = history.begin_generation()
    backend.submit(AudioCommand(AudioCommandKind.RESET, generation=reset_generation))
    backend.submit(AudioCommand(AudioCommandKind.NOTE_OFF))
    backend.submit(AudioCommand(AudioCommandKind.RETUNE, frequency_hz=330.0))
    with pytest.raises(RuntimeError, match="unavailable"):
        backend.submit(
            AudioCommand(
                AudioCommandKind.NOTE_ON,
                frequency_hz=220.0,
                generation=reset_generation,
            )
        )

    media_devices.device = DefaultDevice({(1, QAudioFormat.SampleFormat.Float)})
    media_devices.audioOutputsChanged.emit()
    source = sinks[0].source
    assert source is not None
    audio_format = audio_format_candidates(render.sample_rate_hz)[1]

    idle = read_block(source, render, audio_format)

    np.testing.assert_array_equal(idle, np.zeros(render.block_frames, dtype=np.float32))
    assert history.available_frames == render.block_frames
    source.submit(
        AudioCommand(
            AudioCommandKind.NOTE_ON,
            frequency_hz=220.0,
            generation=reset_generation,
        )
    )
    live = read_block(source, render, audio_format)
    assert 0.0 < np.max(np.abs(live)) <= replacement.output_gain


def test_voice_idle_is_queued_with_its_old_generation_across_retrigger(qapp) -> None:
    patch = short_patch(release_frames=4)
    backend, _, sinks, history, render = backend_setup(patch=patch)
    coordinator = CaptureCoordinator(
        history,
        render.sample_rate_hz,
        AnalysisConfig(
            waveform_window_seconds=render.block_frames / render.sample_rate_hz,
            fft_frames=render.block_frames,
        ),
    )
    controller = WorkbenchController(
        WorkbenchSpec.from_tuning(Tuning()),
        render,
        patch,
        coordinator,
        backend.submit,
    )
    received: list[int] = []
    backend.voice_idle.connect(received.append)
    backend.voice_idle.connect(controller.mark_voice_idle)
    backend.start()
    source = sinks[0].source
    assert source is not None
    audio_format = audio_format_candidates(render.sample_rate_hz)[1]
    first_generation = controller.press_play().capture.generation
    read_block(source, render, audio_format)
    controller.release_play()
    read_block(source, render, audio_format)
    second_generation = controller.press_play().capture.generation

    assert received == []
    assert second_generation > first_generation
    assert controller.state.voice_may_be_active
    qapp.processEvents()
    assert received == [first_generation]
    assert controller.state.voice_may_be_active


def test_repeated_shutdown_disconnects_hotplug_and_disposes_once(qapp) -> None:
    backend, media_devices, sinks, _, _ = backend_setup()
    backend.start()

    backend.shutdown()
    backend.shutdown()
    media_devices.audioOutputsChanged.emit()

    assert len(sinks) == 1
    assert sinks[0].reset_count == 1
