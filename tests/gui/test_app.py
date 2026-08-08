from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from harpy.capture import SampleHistory
from harpy.config import DEFAULT_CONFIG
from harpy.gui.app import build_runtime
from harpy.gui.qt_audio import QtAudioBackend
from harpy.playback import AudioCommand, AudioCommandKind
from harpy.synth.models import RenderConfig, SynthPatch


class ComposedBackend(QObject):
    availability_changed = Signal(bool)
    audio_failure = Signal(str)
    force_stop_requested = Signal()
    voice_idle = Signal(int)

    def __init__(self, render: RenderConfig, patch: SynthPatch, history: SampleHistory) -> None:
        super().__init__()
        self.render = render
        self.patch = patch
        self.history = history
        self.commands: list[AudioCommand] = []
        self.events: list[str] = []

    def submit(self, command: AudioCommand) -> None:
        self.commands.append(command)
        self.events.append(f"submit:{command.kind.name}")

    def start(self) -> None:
        self.events.append("start")
        self.availability_changed.emit(True)

    def shutdown(self) -> None:
        self.events.append("shutdown")


def test_build_runtime_composes_final_graph_with_fft_capacity(qapp) -> None:
    created: list[ComposedBackend] = []

    def factory(render, patch, history):
        backend = ComposedBackend(render, patch, history)
        created.append(backend)
        return backend

    runtime = build_runtime(qapp, DEFAULT_CONFIG, audio_factory=factory)

    assert runtime.config is DEFAULT_CONFIG
    assert runtime.audio is created[0]
    assert runtime.history is runtime.audio.history
    assert runtime.history.snapshot_recent(DEFAULT_CONFIG.analysis.fft_frames).samples.size == 0
    assert runtime.window._refresh_timer.interval() >= 34
    assert isinstance(runtime.audio, ComposedBackend)


def test_backend_signals_are_connected_to_single_window_paths_before_start(qapp) -> None:
    runtime = build_runtime(qapp, DEFAULT_CONFIG, audio_factory=ComposedBackend)
    audio = runtime.audio

    audio.start()
    runtime.window.play_button.pressed.emit()
    generation = runtime.controller.state.capture.generation
    audio.voice_idle.emit(generation)
    audio.force_stop_requested.emit()
    audio.availability_changed.emit(False)
    audio.audio_failure.emit("No default audio output")

    assert audio.events[0] == "start"
    assert [command.kind for command in audio.commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.RESET,
    ]
    assert not runtime.controller.state.voice_may_be_active
    assert not runtime.controller.state.audio_available
    assert not runtime.window.play_button.isEnabled()


def test_about_to_quit_uses_reset_shutdown_teardown_order(qapp) -> None:
    runtime = build_runtime(qapp, DEFAULT_CONFIG, audio_factory=ComposedBackend)
    runtime.window.play_button.pressed.emit()

    qapp.aboutToQuit.emit()
    runtime.window.close()

    assert runtime.audio.events == ["submit:NOTE_ON", "submit:RESET", "shutdown"]
    assert not runtime.window._refresh_timer.isActive()


def test_default_audio_factory_is_the_final_backend() -> None:
    from inspect import signature

    parameter = signature(build_runtime).parameters["audio_factory"]
    assert parameter.default is QtAudioBackend
