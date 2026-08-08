from __future__ import annotations

import queue
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QIODevice, QObject, Signal
from PySide6.QtMultimedia import (
    QAudio,
    QAudioDevice,
    QAudioFormat,
    QAudioSink,
    QMediaDevices,
)

from harpy.config import AppConfig
from harpy.gui.controller import AudioCommand, AudioCommandKind
from harpy.gui.visualizer import SampleRingBuffer
from harpy.synth.reference import SineVoice


def _format_sample_rate(sample_rate_hz: int) -> str:
    if sample_rate_hz < 1_000:
        return f"{sample_rate_hz} Hz"
    kilohertz, remainder_hz = divmod(sample_rate_hz, 1_000)
    if remainder_hz == 0:
        return f"{kilohertz} kHz"
    decimal = f"{remainder_hz:03d}".rstrip("0")
    return f"{kilohertz}.{decimal} kHz"


def audio_format_candidates(sample_rate_hz: int) -> tuple[QAudioFormat, ...]:
    formats: list[QAudioFormat] = []
    for channels, sample_format in (
        (2, QAudioFormat.SampleFormat.Float),
        (1, QAudioFormat.SampleFormat.Float),
        (2, QAudioFormat.SampleFormat.Int16),
        (1, QAudioFormat.SampleFormat.Int16),
    ):
        audio_format = QAudioFormat()
        audio_format.setSampleRate(sample_rate_hz)
        audio_format.setChannelCount(channels)
        audio_format.setSampleFormat(sample_format)
        formats.append(audio_format)
    return tuple(formats)


def choose_audio_format(
    device: QAudioDevice,
    sample_rate_hz: int,
) -> QAudioFormat | None:
    return next(
        (
            audio_format
            for audio_format in audio_format_candidates(sample_rate_hz)
            if device.isFormatSupported(audio_format)
        ),
        None,
    )


def encode_mono_samples(samples: np.ndarray, audio_format: QAudioFormat) -> bytes:
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    channels = audio_format.channelCount()
    interleaved = np.repeat(mono[:, None], channels, axis=1).reshape(-1)
    if audio_format.sampleFormat() is QAudioFormat.SampleFormat.Float:
        return interleaved.astype(np.float32, copy=False).tobytes()
    if audio_format.sampleFormat() is QAudioFormat.SampleFormat.Int16:
        pcm = np.rint(np.clip(interleaved, -1.0, 1.0) * 32767.0).astype(np.int16)
        return pcm.tobytes()
    raise ValueError("unsupported device sample format")


class SynthAudioDevice(QIODevice):
    def __init__(
        self,
        config: AppConfig,
        audio_format: QAudioFormat,
        sample_history: SampleRingBuffer,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._audio_format = audio_format
        self._sample_history = sample_history
        self._voice = SineVoice(config.tuning, config.render, config.patch)
        self._commands: queue.SimpleQueue[AudioCommand] = queue.SimpleQueue()
        self._staging = bytearray()
        self.open(QIODevice.OpenModeFlag.ReadOnly)

    def submit(self, command: AudioCommand) -> None:
        self._commands.put(command)

    def readData(self, maxlen: int) -> bytes:
        if maxlen <= 0:
            return b""
        while True:
            self._drain_commands()
            if len(self._staging) >= maxlen:
                break
            mono = self._voice.render_block(self._config.render.block_frames)
            self._sample_history.append(mono)
            self._staging.extend(encode_mono_samples(mono, self._audio_format))
        output = bytes(self._staging[:maxlen])
        del self._staging[:maxlen]
        return output

    def writeData(self, data: bytes) -> int:
        return -1

    def isSequential(self) -> bool:
        return True

    def reset_after_sink_stop(self) -> None:
        while True:
            try:
                self._commands.get_nowait()
            except queue.Empty:
                break
        self._voice.reset()
        self._staging.clear()
        self._sample_history.clear()

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            if command.kind is AudioCommandKind.NOTE_ON:
                if command.pitch is None:
                    raise RuntimeError("validated NOTE_ON command lost its pitch")
                self._voice.note_on(command.pitch)
            elif command.kind is AudioCommandKind.NOTE_OFF:
                self._voice.note_off()
            elif command.kind is AudioCommandKind.RESET:
                self._voice.reset()
                self._staging.clear()
                self._sample_history.clear()


SinkFactory = Callable[[QAudioDevice, QAudioFormat, QObject], QAudioSink]


class QtAudioEngine(QObject):
    status_changed = Signal(str, bool)
    force_stop_requested = Signal()

    def __init__(
        self,
        config: AppConfig,
        sample_history: SampleRingBuffer,
        *,
        media_devices: QMediaDevices | None = None,
        sink_factory: SinkFactory | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._sample_history = sample_history
        self._media_devices = media_devices or QMediaDevices(self)
        self._sink_factory = sink_factory or (
            lambda device, audio_format, owner: QAudioSink(device, audio_format, owner)
        )
        self._sink: QAudioSink | None = None
        self._source: SynthAudioDevice | None = None
        self._disposing = False
        self._shutdown = False
        self._media_devices.audioOutputsChanged.connect(self._on_audio_outputs_changed)

    def start(self) -> None:
        if self._shutdown:
            return
        self._initialize_default_output()

    def submit(self, command: AudioCommand) -> None:
        if self._source is None:
            raise RuntimeError("audio output is unavailable")
        self._source.submit(command)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._media_devices.audioOutputsChanged.disconnect(self._on_audio_outputs_changed)
        self._dispose_sink()

    def _initialize_default_output(self) -> None:
        device = self._media_devices.defaultAudioOutput()
        if device.isNull():
            self.status_changed.emit("No default audio output", False)
            return
        sample_rate_hz = self._config.render.sample_rate_hz
        sample_rate_text = _format_sample_rate(sample_rate_hz)
        audio_format = choose_audio_format(device, sample_rate_hz)
        if audio_format is None:
            self.status_changed.emit(
                f"Default output has no compatible {sample_rate_text} format",
                False,
            )
            return
        self._source = SynthAudioDevice(
            self._config,
            audio_format,
            self._sample_history,
            self,
        )
        self._sink = self._sink_factory(device, audio_format, self)
        self._sink.stateChanged.connect(self._on_sink_state_changed)
        sink = self._sink
        sink.start(self._source)
        if sink.error().value != QAudio.Error.NoError.value:
            self._on_sink_state_changed(QAudio.State.StoppedState)
            return
        if self._sink is not sink:
            return
        format_name = (
            "Float" if audio_format.sampleFormat() is QAudioFormat.SampleFormat.Float else "Int16"
        )
        self.status_changed.emit(
            f"{device.description()} · {sample_rate_text} · "
            f"{audio_format.channelCount()} ch · {format_name}",
            True,
        )

    def _on_audio_outputs_changed(self) -> None:
        if self._shutdown:
            return
        self.force_stop_requested.emit()
        self._dispose_sink()
        self._initialize_default_output()

    def _on_sink_state_changed(self, state: QAudio.State) -> None:
        if self._disposing or self._sink is None:
            return
        error = self._sink.error()
        if (
            state.value == QAudio.State.StoppedState.value
            and error.value != QAudio.Error.NoError.value
        ):
            error_name = error.name
            self.force_stop_requested.emit()
            self._dispose_sink()
            self.status_changed.emit(f"Audio output failed: {error_name}", False)

    def _dispose_sink(self) -> None:
        self._disposing = True
        sink = self._sink
        source = self._source
        self._sink = None
        self._source = None
        try:
            if sink is not None:
                sink.stateChanged.disconnect(self._on_sink_state_changed)
                sink.reset()
            if source is not None:
                source.reset_after_sink_stop()
                source.close()
        finally:
            if sink is not None:
                sink.deleteLater()
            if source is not None:
                source.deleteLater()
            self._disposing = False
