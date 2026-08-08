from __future__ import annotations

import queue
from collections.abc import Callable
from contextlib import suppress
from threading import Lock

import numpy as np
from PySide6.QtCore import QIODevice, QObject, Qt, Signal, Slot
from PySide6.QtMultimedia import (
    QAudio,
    QAudioDevice,
    QAudioFormat,
    QAudioSink,
    QMediaDevices,
)

from harpy.capture import SampleHistory
from harpy.playback import AudioCommand, AudioCommandKind
from harpy.synth.engine import SynthEngine
from harpy.synth.models import RenderConfig, SynthPatch, validate_renderable_patch

_FLOAT_SAMPLE_FORMAT = QAudioFormat.SampleFormat["float".title()]


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
        (2, _FLOAT_SAMPLE_FORMAT),
        (1, _FLOAT_SAMPLE_FORMAT),
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
    if audio_format.sampleFormat() is _FLOAT_SAMPLE_FORMAT:
        return interleaved.astype(np.float32, copy=False).tobytes()
    if audio_format.sampleFormat() is QAudioFormat.SampleFormat.Int16:
        pcm = np.rint(np.clip(interleaved, -1.0, 1.0) * 32767.0).astype(np.int16)
        return pcm.tobytes()
    raise ValueError("unsupported device sample format")


class SynthAudioSource(QIODevice):
    voice_idle = Signal(int)

    def __init__(
        self,
        render: RenderConfig,
        patch: SynthPatch,
        audio_format: QAudioFormat,
        history: SampleHistory,
        capture_generation: int,
    ) -> None:
        super().__init__()
        if not isinstance(history, SampleHistory):
            raise ValueError("history must be a SampleHistory")
        if (
            isinstance(capture_generation, bool)
            or not isinstance(capture_generation, int)
            or capture_generation < 0
        ):
            raise ValueError("capture_generation must be a nonnegative integer")
        self._render = render
        self._audio_format = audio_format
        self._history = history
        self._capture_generation = capture_generation
        self._engine = SynthEngine(render, patch)
        self._commands: queue.SimpleQueue[AudioCommand] = queue.SimpleQueue()
        self._staging = bytearray()
        self._io_lock = Lock()
        self._natural_idle_pending = False
        self.open(QIODevice.OpenModeFlag.ReadOnly)

    def submit(self, command: AudioCommand) -> None:
        if not isinstance(command, AudioCommand):
            raise ValueError("command must be an AudioCommand")
        with self._io_lock:
            self._commands.put(command)

    def readData(self, maxlen: int) -> bytes:
        if maxlen <= 0:
            return b""
        with self._io_lock:
            while True:
                self._drain_commands()
                if len(self._staging) >= maxlen:
                    break
                was_idle = self._engine.is_idle
                mono = self._engine.render(self._render.block_frames)
                self._history.append(mono, self._capture_generation)
                self._staging.extend(encode_mono_samples(mono, self._audio_format))
                if not was_idle and self._engine.is_idle:
                    self._natural_idle_pending = True
                self._emit_natural_idle_if_needed()
            output = bytes(self._staging[:maxlen])
            del self._staging[:maxlen]
            return output

    def writeData(self, data: bytes) -> int:
        return -1

    def isSequential(self) -> bool:
        return True

    def bytesAvailable(self) -> int:
        block_bytes = self._render.block_frames * self._audio_format.bytesPerFrame()
        return super().bytesAvailable() + block_bytes

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                break
            if command.kind is AudioCommandKind.NOTE_ON:
                if command.frequency_hz is None or command.generation is None:
                    raise RuntimeError("validated NOTE_ON command lost its payload")
                self._capture_generation = command.generation
                self._engine.note_on(command.frequency_hz)
                self._natural_idle_pending = False
            elif command.kind is AudioCommandKind.RETUNE:
                if command.frequency_hz is None:
                    raise RuntimeError("validated RETUNE command lost its frequency")
                if not self._engine.is_idle:
                    self._engine.retune(command.frequency_hz)
            elif command.kind is AudioCommandKind.NOTE_OFF:
                was_idle = self._engine.is_idle
                self._engine.note_off()
                if not was_idle and self._engine.is_idle:
                    self._natural_idle_pending = True
            elif command.kind is AudioCommandKind.CLEAR_CAPTURE:
                if command.generation is None:
                    raise RuntimeError("validated CLEAR_CAPTURE command lost its generation")
                self._capture_generation = command.generation
            elif command.kind is AudioCommandKind.REPLACE_PATCH:
                if command.patch is None or command.generation is None:
                    raise RuntimeError("validated REPLACE_PATCH command lost its payload")
                self._staging.clear()
                self._engine.replace_patch(command.patch)
                self._capture_generation = command.generation
                self._natural_idle_pending = False
            elif command.kind is AudioCommandKind.RESET:
                if command.generation is None:
                    raise RuntimeError("validated RESET command lost its generation")
                self._staging.clear()
                self._engine.reset()
                self._capture_generation = command.generation
                self._natural_idle_pending = False
        self._emit_natural_idle_if_needed()

    def _emit_natural_idle_if_needed(self) -> None:
        if self._natural_idle_pending and self._engine.is_idle:
            self._natural_idle_pending = False
            self.voice_idle.emit(self._capture_generation)


class QtAudioBackend(QObject):
    availability_changed = Signal(bool)
    audio_failure = Signal(str)
    force_stop_requested = Signal()
    voice_idle = Signal(int)

    def __init__(
        self,
        render: RenderConfig,
        patch: SynthPatch,
        history: SampleHistory,
        *,
        media_devices: QObject | None = None,
        sink_factory: Callable[..., QAudioSink] = QAudioSink,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(render, RenderConfig):
            raise ValueError("render must be a RenderConfig")
        if not isinstance(patch, SynthPatch):
            raise ValueError("patch must be a SynthPatch")
        if not isinstance(history, SampleHistory):
            raise ValueError("history must be a SampleHistory")
        validate_renderable_patch(patch, render)
        self._render = render
        self._patch = patch
        self._history = history
        self._capture_generation = history.generation
        self._media_devices = media_devices if media_devices is not None else QMediaDevices(self)
        self._sink_factory = sink_factory
        self._sink: QAudioSink | None = None
        self._source: SynthAudioSource | None = None
        self._starting_sink: QAudioSink | None = None
        self._start_failure_requests_force_stop = True
        self._idle_generation_aliases: dict[int, int] = {}
        self._active_failure_key: str | None = None
        self._disposing = False
        self._shutdown = False
        self._media_devices.audioOutputsChanged.connect(self._on_audio_outputs_changed)

    def start(self) -> None:
        if self._shutdown or self._sink is not None:
            return
        self._initialize_default_output(request_force_stop_on_failure=True)

    def submit(self, command: AudioCommand) -> None:
        if not isinstance(command, AudioCommand):
            raise ValueError("command must be an AudioCommand")
        source = self._source
        if command.kind is AudioCommandKind.NOTE_ON and source is None:
            raise RuntimeError("audio output is unavailable")
        if command.kind is AudioCommandKind.REPLACE_PATCH:
            if command.patch is None or command.generation is None:
                raise RuntimeError("validated REPLACE_PATCH command lost its payload")
            validate_renderable_patch(command.patch, self._render)
            self._idle_generation_aliases.clear()
            self._patch = command.patch
            self._capture_generation = command.generation
        elif command.kind is AudioCommandKind.NOTE_ON:
            if command.generation is None:
                raise RuntimeError("validated NOTE_ON command lost its generation")
            self._idle_generation_aliases.clear()
            self._capture_generation = command.generation
        elif command.kind is AudioCommandKind.CLEAR_CAPTURE:
            if command.generation is None:
                raise RuntimeError("validated CLEAR_CAPTURE command lost its generation")
            if self._capture_generation != command.generation:
                self._idle_generation_aliases[self._capture_generation] = command.generation
            self._capture_generation = command.generation
        elif command.kind is AudioCommandKind.RESET:
            if command.generation is None:
                raise RuntimeError("validated RESET command lost its generation")
            self._idle_generation_aliases.clear()
            self._capture_generation = command.generation
        if source is not None:
            source.submit(command)

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._media_devices.audioOutputsChanged.disconnect(self._on_audio_outputs_changed)
        self._dispose_sink()

    def _initialize_default_output(self, *, request_force_stop_on_failure: bool) -> None:
        try:
            device = self._media_devices.defaultAudioOutput()
        except Exception as error:
            self._report_failure(
                f"Audio output failed: {error}",
                request_force_stop=request_force_stop_on_failure,
            )
            return
        if device.isNull():
            self._report_failure(
                "No default audio output",
                request_force_stop=request_force_stop_on_failure,
            )
            return
        sample_rate_hz = self._render.sample_rate_hz
        audio_format = choose_audio_format(device, sample_rate_hz)
        if audio_format is None:
            self._report_failure(
                f"Default output has no compatible {_format_sample_rate(sample_rate_hz)} format",
                request_force_stop=request_force_stop_on_failure,
            )
            return

        try:
            source = SynthAudioSource(
                self._render,
                self._patch,
                audio_format,
                self._history,
                self._capture_generation,
            )
            source.setParent(self)
            self._source = source
            sink = self._sink_factory(device, audio_format, self)
            self._sink = sink
            source.voice_idle.connect(
                self._forward_voice_idle,
                Qt.ConnectionType.QueuedConnection,
            )
            sink.stateChanged.connect(self._on_sink_state_changed)
            self._starting_sink = sink
            self._start_failure_requests_force_stop = request_force_stop_on_failure
            sink.start(source)
        except Exception as error:
            self._dispose_sink()
            self._report_failure(
                f"Audio output failed: {error}",
                request_force_stop=request_force_stop_on_failure,
            )
            return

        if self._sink is not sink:
            return
        error = sink.error()
        if _enum_value(error) != _enum_value(QAudio.Error.NoError):
            self._handle_sink_failure(error)
            return
        self._starting_sink = None
        self._active_failure_key = None
        self.availability_changed.emit(True)

    def _on_audio_outputs_changed(self) -> None:
        if self._shutdown:
            return
        had_source = self._source is not None
        if had_source:
            self.force_stop_requested.emit()
        self._dispose_sink()
        self._initialize_default_output(request_force_stop_on_failure=not had_source)

    def _on_sink_state_changed(self, state: QAudio.State) -> None:
        if self._disposing or self._sink is None:
            return
        error = self._sink.error()
        if _enum_value(state) == _enum_value(QAudio.State.StoppedState) and _enum_value(
            error
        ) != _enum_value(QAudio.Error.NoError):
            self._handle_sink_failure(error)

    def _handle_sink_failure(self, error: object) -> None:
        error_name = getattr(error, "name", str(error))
        request_force_stop = (
            self._start_failure_requests_force_stop
            if self._starting_sink is self._sink and self._sink is not None
            else True
        )
        self._starting_sink = None
        self._dispose_sink()
        self._report_failure(
            f"Audio output failed: {error_name}",
            request_force_stop=request_force_stop,
        )

    def _report_failure(self, message: str, *, request_force_stop: bool) -> None:
        if self._active_failure_key == message:
            return
        self._active_failure_key = message
        self.availability_changed.emit(False)
        self.audio_failure.emit(message)
        if request_force_stop:
            self.force_stop_requested.emit()

    @Slot(int)
    def _forward_voice_idle(self, generation: int) -> None:
        forwarded_generation = generation
        visited: set[int] = set()
        while (
            forwarded_generation in self._idle_generation_aliases
            and forwarded_generation not in visited
        ):
            visited.add(forwarded_generation)
            forwarded_generation = self._idle_generation_aliases[forwarded_generation]
        for aliased_generation in visited:
            self._idle_generation_aliases.pop(aliased_generation, None)
        self.voice_idle.emit(forwarded_generation)

    def _dispose_sink(self) -> None:
        self._disposing = True
        sink = self._sink
        source = self._source
        self._sink = None
        self._source = None
        self._starting_sink = None
        try:
            if sink is not None:
                with suppress(RuntimeError, TypeError):
                    sink.stateChanged.disconnect(self._on_sink_state_changed)
                sink.reset()
            if source is not None:
                with suppress(RuntimeError, TypeError):
                    source.voice_idle.disconnect(self._forward_voice_idle)
                source.close()
        finally:
            if sink is not None:
                sink.deleteLater()
            if source is not None:
                source.deleteLater()
            self._disposing = False


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)
