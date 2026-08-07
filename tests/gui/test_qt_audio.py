from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QObject, Signal
from PySide6.QtMultimedia import QAudio, QAudioFormat

from harpy.config import DEFAULT_CONFIG
from harpy.gui.controller import AudioCommand, AudioCommandKind
from harpy.gui.qt_audio import (
    QtAudioEngine,
    SynthAudioDevice,
    audio_format_candidates,
    choose_audio_format,
    encode_mono_samples,
)
from harpy.gui.visualizer import SampleRingBuffer
from harpy.pitch import Pitch


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
    decoded = np.frombuffer(encoded, dtype=np.float32)
    np.testing.assert_array_equal(decoded, [0.25, 0.25, -0.5, -0.5])


def test_int16_encoding_clips_rounds_and_duplicates() -> None:
    audio_format = audio_format_candidates(48_000)[2]
    encoded = encode_mono_samples(
        np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float32),
        audio_format,
    )
    decoded = np.frombuffer(encoded, dtype=np.int16).reshape(-1, 2)
    np.testing.assert_array_equal(
        decoded,
        [[-32767, -32767], [-16384, -16384], [16384, 16384], [32767, 32767]],
    )


def test_audio_device_applies_note_command_and_streams_one_stereo_block() -> None:
    history = SampleRingBuffer(capacity_frames=48_000)
    audio_format = audio_format_candidates(48_000)[0]
    source = SynthAudioDevice(DEFAULT_CONFIG, audio_format, history)
    source.submit(AudioCommand(AudioCommandKind.NOTE_ON, Pitch.from_midi(60)))
    byte_count = DEFAULT_CONFIG.render.block_frames * audio_format.bytesPerFrame()
    encoded = source.readData(byte_count)
    decoded = np.frombuffer(encoded, dtype=np.float32).reshape(-1, 2)
    assert encoded and len(encoded) == byte_count
    np.testing.assert_array_equal(decoded[:, 0], decoded[:, 1])
    assert np.any(decoded != 0.0)
    assert np.any(history.snapshot(256) != 0.0)


def test_reset_command_makes_next_staging_block_exact_silence() -> None:
    history = SampleRingBuffer(capacity_frames=48_000)
    audio_format = audio_format_candidates(48_000)[0]
    source = SynthAudioDevice(DEFAULT_CONFIG, audio_format, history)
    byte_count = DEFAULT_CONFIG.render.block_frames * audio_format.bytesPerFrame()
    source.submit(AudioCommand(AudioCommandKind.NOTE_ON, Pitch.from_midi(60)))
    source.readData(byte_count)
    source.submit(AudioCommand(AudioCommandKind.RESET))
    decoded = np.frombuffer(source.readData(byte_count), dtype=np.float32)
    np.testing.assert_array_equal(decoded, np.zeros(decoded.size, dtype=np.float32))


class FakeSignal:
    def __init__(self) -> None:
        self.callbacks: list[Callable[[], None]] = []

    def connect(self, callback: Callable[[], None]) -> None:
        self.callbacks.append(callback)

    def emit(self) -> None:
        for callback in self.callbacks:
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

    def __init__(self) -> None:
        super().__init__()
        self.source: object | None = None
        self.reset_count = 0
        self.current_error = QAudio.Error.NoError

    def start(self, source: object) -> None:
        self.source = source

    def reset(self) -> None:
        self.reset_count += 1

    def error(self) -> QAudio.Error:
        return self.current_error


def test_engine_starts_default_device_and_rebuilds_once_on_hotplug(qapp) -> None:
    device = DefaultDevice({(2, QAudioFormat.SampleFormat.Float)})
    media_devices = FakeMediaDevices(device)
    sinks: list[FakeSink] = []

    def sink_factory(*_args: object) -> FakeSink:
        sink = FakeSink()
        sinks.append(sink)
        return sink

    history = SampleRingBuffer(capacity_frames=48_000)
    engine = QtAudioEngine(
        DEFAULT_CONFIG,
        history,
        media_devices=media_devices,
        sink_factory=sink_factory,
    )
    forced_stops: list[bool] = []
    engine.force_stop_requested.connect(lambda: forced_stops.append(True))
    engine.start()
    assert len(sinks) == 1
    assert sinks[0].source is not None
    media_devices.audioOutputsChanged.emit()
    assert forced_stops == [True]
    assert sinks[0].reset_count == 1
    assert len(sinks) == 2


def test_engine_disables_play_when_default_format_is_incompatible(qapp) -> None:
    media_devices = FakeMediaDevices(DefaultDevice(set()))
    history = SampleRingBuffer(capacity_frames=48_000)
    engine = QtAudioEngine(DEFAULT_CONFIG, history, media_devices=media_devices)
    statuses: list[tuple[str, bool]] = []
    engine.status_changed.connect(lambda message, playable: statuses.append((message, playable)))
    engine.start()
    assert statuses == [("Default output has no compatible 48 kHz format", False)]


def test_engine_disables_play_when_no_default_output_exists(qapp) -> None:
    media_devices = FakeMediaDevices(NullDevice())
    history = SampleRingBuffer(capacity_frames=48_000)
    engine = QtAudioEngine(DEFAULT_CONFIG, history, media_devices=media_devices)
    statuses: list[tuple[str, bool]] = []
    engine.status_changed.connect(lambda message, playable: statuses.append((message, playable)))
    engine.start()
    assert statuses == [("No default audio output", False)]


def test_sink_error_forces_stop_and_reports_failure(qapp) -> None:
    device = DefaultDevice({(2, QAudioFormat.SampleFormat.Float)})
    media_devices = FakeMediaDevices(device)
    sink = FakeSink()
    engine = QtAudioEngine(
        DEFAULT_CONFIG,
        SampleRingBuffer(capacity_frames=48_000),
        media_devices=media_devices,
        sink_factory=lambda *_args: sink,
    )
    forced_stops: list[bool] = []
    statuses: list[tuple[str, bool]] = []
    engine.force_stop_requested.connect(lambda: forced_stops.append(True))
    engine.status_changed.connect(lambda message, playable: statuses.append((message, playable)))
    engine.start()
    sink.current_error = QAudio.Error.OpenError
    sink.stateChanged.emit(QAudio.State.StoppedState)
    assert forced_stops == [True]
    assert statuses[-1] == ("Audio output failed: OpenError", False)
