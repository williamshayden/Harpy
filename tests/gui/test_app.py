from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtTest import QTest

from harpy.capture import SampleHistory
from harpy.config import DEFAULT_CONFIG
from harpy.gui.app import build_runtime
from harpy.gui.envelope_entry import EnvelopeValueEntry
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


class CountingDialogs:
    def __init__(self) -> None:
        self.save_count = 0

    def choose_open_path(self, _parent) -> Path | None:
        return None

    def choose_save_path(self, _parent) -> Path | None:
        self.save_count += 1
        return None


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
    assert runtime.window.envelope_editor._render is DEFAULT_CONFIG.render
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


def test_about_to_quit_applies_pending_patch_once_before_backend_shutdown(qapp, qtbot) -> None:
    dialogs = CountingDialogs()
    runtime = build_runtime(
        qapp,
        DEFAULT_CONFIG,
        audio_factory=ComposedBackend,
        patch_dialogs=dialogs,
    )
    qtbot.addWidget(runtime.window)
    runtime.window.play_button.pressed.emit()
    candidate = replace(
        DEFAULT_CONFIG.patch,
        envelope=replace(DEFAULT_CONFIG.patch.envelope, attack_seconds=0.025),
    )
    runtime.window.envelope_editor.patch_commit_requested.emit(candidate)
    attack = runtime.window.envelope_editor.findChild(EnvelopeValueEntry, "attackEntry")
    assert attack is not None
    attack.selectAll()
    QTest.keyClicks(attack, "not complete")
    QTest.keyClick(attack, Qt.Key.Key_Return)

    qapp.aboutToQuit.emit()
    runtime.window.close()

    assert runtime.audio.events == ["submit:NOTE_ON", "submit:REPLACE_PATCH", "shutdown"]
    assert runtime.audio.commands[-1].patch == candidate
    assert attack.text() == "25 ms"
    assert attack.property("validationState") is None
    assert dialogs.save_count == 0
    assert not runtime.window._refresh_timer.isActive()


def test_default_audio_factory_is_the_final_backend(qapp) -> None:
    runtime = build_runtime(qapp)

    assert isinstance(runtime.audio, QtAudioBackend)

    runtime.window.close()
