from PySide6.QtCore import QEvent, QObject, Qt, Signal

from harpy.config import DEFAULT_CONFIG
from harpy.gui.controller import AudioCommand, AudioCommandKind, LabController
from harpy.gui.visualizer import SampleRingBuffer
from harpy.gui.window import HarpyWindow


class FakeAudioEngine(QObject):
    status_changed = Signal(str, bool)
    force_stop_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.commands: list[AudioCommand] = []
        self.lifecycle_events: list[str] = []
        self.shutdown_count = 0

    def submit(self, command: AudioCommand) -> None:
        self.commands.append(command)
        self.lifecycle_events.append(f"submit:{command.kind.name}")

    def shutdown(self) -> None:
        self.shutdown_count += 1
        self.lifecycle_events.append("shutdown")


def make_window(qtbot) -> tuple[HarpyWindow, FakeAudioEngine]:
    audio = FakeAudioEngine()
    controller = LabController(DEFAULT_CONFIG, audio.submit)
    history = SampleRingBuffer(capacity_frames=48_000)
    window = HarpyWindow(DEFAULT_CONFIG, controller, audio, history)
    qtbot.addWidget(window)
    audio.status_changed.emit("Fake speakers · 48 kHz", True)
    return window, audio


def test_window_starts_at_middle_c_with_fixed_patch_copy(qtbot) -> None:
    window, _ = make_window(qtbot)
    assert window.pitch_slider.value() == 60
    assert window.pitch_readout.text() == "C3 · MIDI 60 · 261.626 Hz"
    assert window.tuning_readout.text() == "Concert A reference (MIDI 69) · 440.0 Hz"
    assert "Sine" in window.patch_label.text()
    assert "\N{MINUS SIGN}12 dBFS" in window.patch_label.text()
    assert "1 ms" in window.patch_label.text()
    assert "600 ms" in window.patch_label.text()
    assert "\N{MINUS SIGN}6 dB" in window.patch_label.text()


def test_hold_button_locks_pitch_and_sends_note_on_off(qtbot) -> None:
    window, audio = make_window(qtbot)
    qtbot.mousePress(window.play_button, Qt.MouseButton.LeftButton)
    assert not window.pitch_slider.isEnabled()
    assert [command.kind for command in audio.commands] == [AudioCommandKind.NOTE_ON]
    qtbot.mouseRelease(window.play_button, Qt.MouseButton.LeftButton)
    assert window.pitch_slider.isEnabled()
    assert [command.kind for command in audio.commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.NOTE_OFF,
    ]


def test_slider_updates_readout_when_idle(qtbot) -> None:
    window, _ = make_window(qtbot)
    window.pitch_slider.setValue(61)
    assert window.pitch_readout.text().startswith("C♯3 · MIDI 61")


def test_deactivation_forces_stop_and_raises_button(qtbot) -> None:
    window, audio = make_window(qtbot)
    qtbot.mousePress(window.play_button, Qt.MouseButton.LeftButton)
    event = QEvent(QEvent.Type.WindowDeactivate)
    window.event(event)
    assert [command.kind for command in audio.commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.RESET,
    ]
    assert not window.play_button.isDown()
    assert window.pitch_slider.isEnabled()


def test_audio_failure_disables_play_and_forces_stop(qtbot) -> None:
    window, audio = make_window(qtbot)
    qtbot.mousePress(window.play_button, Qt.MouseButton.LeftButton)
    audio.force_stop_requested.emit()
    audio.status_changed.emit("Audio output failed: OpenError", False)
    assert not window.play_button.isEnabled()
    assert window.status_label.text() == "Audio output failed: OpenError"
    assert audio.commands[-1].kind is AudioCommandKind.RESET


def test_close_forces_stop_and_shuts_down_audio(qtbot) -> None:
    window, audio = make_window(qtbot)
    qtbot.mousePress(window.play_button, Qt.MouseButton.LeftButton)
    assert window._plot_timer.isActive()
    window.close()
    assert audio.commands[-1].kind is AudioCommandKind.RESET
    assert audio.lifecycle_events[-2:] == ["submit:RESET", "shutdown"]
    assert audio.shutdown_count == 1
    assert not window._plot_timer.isActive()


def test_plot_failure_stops_visual_timer_without_touching_audio(qtbot, monkeypatch) -> None:
    window, audio = make_window(qtbot)

    def fail_spectrum(*_args: object, **_kwargs: object) -> tuple[object, object]:
        raise RuntimeError("plot calculation failed")

    monkeypatch.setattr("harpy.gui.window.spectrum_dbfs", fail_spectrum)
    window._update_plots()
    assert not window._plot_timer.isActive()
    assert audio.commands == []
