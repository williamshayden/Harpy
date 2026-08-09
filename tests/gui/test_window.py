from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QSize, Qt
from PySide6.QtGui import QAction, QKeyEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollBar,
    QWidget,
)

from harpy.analysis import AnalysisConfig, AudioObservation
from harpy.capture import CaptureCoordinator, CaptureState, SampleHistory
from harpy.gui.envelope_editor import EnvelopeEditor
from harpy.gui.envelope_entry import EnvelopeValueEntry
from harpy.gui.envelope_graph import EnvelopeGraph
from harpy.gui.envelope_stage_control import EnvelopeStageControl
from harpy.gui.frequency_entry import FrequencyEntry
from harpy.gui.frequency_knob import FrequencyKnob
from harpy.gui.signal_views import SpectrumView, WaveformView
from harpy.gui.window import HarpyWindow
from harpy.gui.workbench_controller import PatchApplyState, WorkbenchController
from harpy.gui.workbench_spec import WorkbenchSpec
from harpy.playback import AudioCommand, AudioCommandKind
from harpy.synth.models import EnvelopeConfig, RenderConfig, SynthPatch
from harpy.synth.patch_json import dumps_patch, load_patch
from harpy.tuning import Tuning

SAMPLE_RATE = 48_000
ANALYSIS = AnalysisConfig(waveform_window_seconds=4 / SAMPLE_RATE, fft_frames=4)


@dataclass
class Dialogs:
    open_path: Path | None = None
    save_path: Path | None = None
    deactivate_during_open: bool = False
    deactivate_during_save: bool = False
    open_count: int = 0
    save_count: int = 0

    def choose_open_path(self, parent) -> Path | None:
        self.open_count += 1
        if self.deactivate_during_open:
            parent.event(QEvent(QEvent.Type.WindowDeactivate))
        return self.open_path

    def choose_save_path(self, parent) -> Path | None:
        self.save_count += 1
        if self.deactivate_during_save:
            parent.event(QEvent(QEvent.Type.WindowDeactivate))
        return self.save_path


def observed_signal() -> AudioObservation:
    return AudioObservation(
        has_signal=True,
        waveform_samples=np.array([0.0, 0.2, -0.1, 0.0]),
        waveform_time_ms=np.array([0.0, 1.0, 2.0, 3.0]),
        spectrum_frequency_hz=np.array([20.0, 261.6, 20_000.0]),
        spectrum_level_dbfs=np.array([-80.0, -12.0, -100.0]),
        peak_amplitude_fs=0.2,
        peak_frequency_hz=261.6,
        peak_level_dbfs=-12.0,
    )


def observed_silence() -> AudioObservation:
    return AudioObservation(
        has_signal=False,
        waveform_samples=np.empty(0),
        waveform_time_ms=np.empty(0),
        spectrum_frequency_hz=np.empty(0),
        spectrum_level_dbfs=np.empty(0),
        peak_amplitude_fs=None,
        peak_frequency_hz=None,
        peak_level_dbfs=None,
    )


def make_window(qtbot, *, patch: SynthPatch | None = None, analyzer=None, send_command=None):
    tuning = Tuning()
    spec = WorkbenchSpec.from_tuning(tuning)
    render = RenderConfig(sample_rate_hz=SAMPLE_RATE)
    history = SampleHistory(capacity_frames=8)
    kwargs = {} if analyzer is None else {"analyzer": analyzer}
    capture = CaptureCoordinator(history, SAMPLE_RATE, ANALYSIS, **kwargs)
    commands: list[AudioCommand] = []
    sender = commands.append if send_command is None else send_command
    controller = WorkbenchController(
        spec,
        render,
        SynthPatch() if patch is None else patch,
        capture,
        sender,
    )
    dialogs = Dialogs()
    window = HarpyWindow(controller, tuning, spec, render, dialogs)
    qtbot.addWidget(window)
    return window, controller, commands, history, dialogs


def editor_child(
    window: HarpyWindow,
    child_type: type[QWidget],
    name: str,
):
    editor = window.findChild(EnvelopeEditor, "envelopeEditor")
    assert editor is not None
    child = editor.findChild(child_type, name)
    assert child is not None
    return child


def replace_entry_text(qtbot, entry: QLineEdit, text: str) -> None:
    entry.setFocus()
    entry.selectAll()
    qtbot.keyClicks(entry, text)


def open_exact_stage_editor(
    qtbot,
    window: HarpyWindow,
    control_name: str,
) -> EnvelopeValueEntry:
    window.show()
    QApplication.processEvents()
    control = editor_child(window, EnvelopeStageControl, control_name)
    control.setFocus(Qt.FocusReason.OtherFocusReason)
    qtbot.keyPress(control, Qt.Key.Key_F2)
    entry = window.findChild(EnvelopeValueEntry, "envelopeInlineEditor")
    assert entry is not None and entry.isVisible() and entry.hasFocus()
    return entry


def commit_exact_stage(
    qtbot,
    window: HarpyWindow,
    control_name: str,
    text: str,
) -> None:
    entry = open_exact_stage_editor(qtbot, window, control_name)
    replace_entry_text(qtbot, entry, text)
    qtbot.keyClick(entry, Qt.Key.Key_Return)


def make_invalid_editor_draft(qtbot, window: HarpyWindow):
    attack = open_exact_stage_editor(qtbot, window, "attackValueControl")
    replace_entry_text(qtbot, attack, "invalid")
    qtbot.keyClick(attack, Qt.Key.Key_Return)
    graph = editor_child(window, EnvelopeGraph, "envelopeGraph")
    graph.field_previewed.emit("release_curve", -0.75)
    return attack, graph


def test_final_widget_contract_and_copy_contains_no_legacy_or_device_status(qtbot) -> None:
    window, _, _, _, _ = make_window(qtbot)

    assert window.windowTitle() == "Harpy"
    expected = {
        "frequencyControlGroup": QFrame,
        "frequencyKnob": FrequencyKnob,
        "frequencyEntry": FrequencyEntry,
        "derivedPitchLabel": QLabel,
        "playButton": QPushButton,
        "measurementStateLabel": QLabel,
        "clearButton": QPushButton,
        "waveformPlot": WaveformView,
        "spectrumPlot": SpectrumView,
        "envelopeEditor": EnvelopeEditor,
        "envelopeGraph": EnvelopeGraph,
        "attackValueControl": EnvelopeStageControl,
        "decayValueControl": EnvelopeStageControl,
        "sustainValueControl": EnvelopeStageControl,
        "releaseValueControl": EnvelopeStageControl,
        "attackCurveHandle": QWidget,
        "decayCurveHandle": QWidget,
        "releaseCurveHandle": QWidget,
        "curveValueReadout": QLabel,
        "patchStatusLabel": QLabel,
        "resetEnvelopeButton": QPushButton,
        "oscillatorFact": QLabel,
        "outputFact": QLabel,
        "loadPatchButton": QPushButton,
        "savePatchButton": QPushButton,
        "audioErrorBanner": QFrame,
    }
    for object_name, widget_type in expected.items():
        widget = window.findChild(widget_type, object_name)
        assert widget is not None, object_name
    reset_action = window.findChild(QAction, "resetEnvelopeAction")
    assert reset_action is not None
    assert reset_action.shortcut() == QKeySequence("Ctrl+R")
    assert window.findChild(QAction, "reset" + "CurvesAction") is None
    assert window.findChild(QWidget, "patchFacts") is None
    assert window.findChildren(QLabel, "factName") == []
    assert window.findChildren(QLabel, "factValue") == []
    all_copy = " ".join(label.text() for label in window.findChildren(QLabel))
    for forbidden in (
        "Harpy Sine Lab",
        "Native Sine Lab",
        "Synth Engine",
        "MIDI ",
        "backend",
        "device",
        "channel",
        "sample format",
        "Output ready",
        "healthy",
    ):
        assert forbidden.casefold() not in all_copy.casefold()


def test_frequency_editor_has_explicit_hertz_accessibility_copy(qtbot) -> None:
    window, _, _, _, _ = make_window(qtbot)

    assert window.frequency_entry.accessibleName() == "Frequency in hertz"
    assert "frequency" in window.frequency_entry.accessibleDescription().casefold()
    assert "hertz" in window.frequency_entry.accessibleDescription().casefold()


def test_knob_and_entry_share_one_frequency_without_feedback_loops(qtbot) -> None:
    window, controller, commands, _, _ = make_window(qtbot)
    knob = window.findChild(FrequencyKnob, "frequencyKnob")
    entry = window.findChild(FrequencyEntry, "frequencyEntry")
    label = window.findChild(QLabel, "derivedPitchLabel")

    knob.set_frequency_hz(330.0, emit=True)
    assert controller.state.selected_frequency_hz == 330.0
    assert entry.text() == "330.000"
    assert commands == []

    entry.setText("261.625565")
    qtbot.keyClick(entry, Qt.Key.Key_Return)
    assert controller.state.selected_frequency_hz == pytest.approx(261.625565)
    assert knob.frequency_hz == pytest.approx(261.625565)
    assert label.text() == "C3 +0.0¢"
    assert commands == []


def test_refresh_timer_preserves_a_focused_human_edit_until_explicit_knob_commit(qtbot) -> None:
    window, controller, commands, _, _ = make_window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    entry = window.findChild(FrequencyEntry, "frequencyEntry")
    knob = window.findChild(FrequencyKnob, "frequencyKnob")
    entry.setFocus()
    entry.selectAll()
    qtbot.keyClicks(entry, "333.123")

    qtbot.wait(window._refresh_timer.interval() * 2 + 10)

    assert entry.text() == "333.123"
    assert entry.isModified()
    assert controller.state.selected_frequency_hz == pytest.approx(261.6255653005986)
    assert commands == []

    knob.set_frequency_hz(330.0, emit=True)
    assert entry.text() == "330.000"
    assert not entry.isModified()
    assert controller.state.selected_frequency_hz == 330.0


def test_invalid_entry_style_and_banner_survive_refresh_then_escape_restores_model(qtbot) -> None:
    window, controller, commands, _, _ = make_window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    entry = window.findChild(FrequencyEntry, "frequencyEntry")
    banner = window.findChild(QFrame, "audioErrorBanner")
    entry.setFocus()
    entry.selectAll()
    qtbot.keyClicks(entry, "999")
    qtbot.keyClick(entry, Qt.Key.Key_Return)

    assert entry.text() == "999"
    assert entry.property("validationState") == "error"
    assert "frequency" in banner.findChild(QLabel).text().casefold()
    qtbot.wait(window._refresh_timer.interval() * 2 + 10)
    assert entry.text() == "999"
    assert entry.property("validationState") == "error"
    assert banner.isVisibleTo(window)
    assert controller.state.selected_frequency_hz == pytest.approx(261.6255653005986)
    assert commands == []

    qtbot.keyClick(entry, Qt.Key.Key_Escape)
    assert entry.text() == "261.626"
    assert entry.property("validationState") is None
    assert banner.isHidden()


def test_valid_entry_commit_clears_prior_validation_error(qtbot) -> None:
    window, controller, _, _, _ = make_window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    entry = window.findChild(FrequencyEntry, "frequencyEntry")
    banner = window.findChild(QFrame, "audioErrorBanner")
    entry.setFocus()
    entry.selectAll()
    qtbot.keyClicks(entry, "999")
    qtbot.keyClick(entry, Qt.Key.Key_Return)
    entry.selectAll()
    qtbot.keyClicks(entry, "330")

    qtbot.keyClick(entry, Qt.Key.Key_Return)

    assert controller.state.selected_frequency_hz == 330.0
    assert entry.text() == "330.000"
    assert entry.property("validationState") is None
    assert banner.isHidden()


@pytest.mark.parametrize(
    "target_name",
    [
        "frequencyKnob",
        "playButton",
        "clearButton",
        "resetEnvelopeButton",
        "loadPatchButton",
        "savePatchButton",
    ],
)
def test_space_hold_routes_from_each_focused_non_editor_child(qtbot, target_name: str) -> None:
    window, _, commands, _, dialogs = make_window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    target = window.findChild(QWidget, target_name)
    target.setFocus()

    qtbot.keyPress(target, Qt.Key.Key_Space)
    qtbot.keyRelease(target, Qt.Key.Key_Space)

    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.NOTE_OFF,
    ]
    assert dialogs.open_count == 0
    assert dialogs.save_count == 0


def test_space_is_owned_by_focused_frequency_editor(qtbot) -> None:
    window, _, commands, _, _ = make_window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    entry = window.findChild(FrequencyEntry, "frequencyEntry")
    entry.setFocus()
    entry.setText("330")
    entry.setCursorPosition(3)

    qtbot.keyPress(entry, Qt.Key.Key_Space)
    qtbot.keyRelease(entry, Qt.Key.Key_Space)

    assert entry.text() == "330 "
    assert commands == []


@pytest.mark.parametrize(
    "control_name",
    ["attackValueControl", "sustainValueControl"],
)
def test_space_is_text_input_in_transient_envelope_entry(qtbot, control_name: str) -> None:
    # Routing an editor Space through transport would both corrupt text entry and sound a note.
    window, _, commands, _, _ = make_window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    entry = open_exact_stage_editor(qtbot, window, control_name)
    entry.setText("1")
    entry.setCursorPosition(1)
    entry.setFocus()

    qtbot.keyPress(entry, Qt.Key.Key_Space)
    qtbot.keyRelease(entry, Qt.Key.Key_Space)

    assert entry.text() == "1 "
    assert commands == []


def test_space_autorepeat_is_consumed_without_retrigger_or_early_release(qtbot) -> None:
    window, _, commands, _, _ = make_window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    target = window.findChild(QPushButton, "clearButton")
    target.setFocus()
    qtbot.keyPress(target, Qt.Key.Key_Space)
    repeat_press = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Space,
        Qt.KeyboardModifier.NoModifier,
        " ",
        True,
        2,
    )
    repeat_release = QKeyEvent(
        QEvent.Type.KeyRelease,
        Qt.Key.Key_Space,
        Qt.KeyboardModifier.NoModifier,
        " ",
        True,
        2,
    )

    QApplication.sendEvent(target, repeat_press)
    QApplication.sendEvent(target, repeat_release)
    assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]
    qtbot.keyRelease(target, Qt.Key.Key_Space)
    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.NOTE_OFF,
    ]


def test_window_space_filter_does_not_leak_to_another_top_level(qtbot) -> None:
    window, _, commands, _, _ = make_window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    other = QPushButton("Other")
    qtbot.addWidget(other)
    other.show()
    qtbot.waitExposed(other)
    clicks: list[bool] = []
    other.clicked.connect(lambda: clicks.append(True))
    other.setFocus()

    qtbot.keyClick(other, Qt.Key.Key_Space)

    assert clicks == [True]
    assert commands == []


def test_space_release_from_owned_modal_top_level_always_releases_held_gate(qtbot) -> None:
    window, controller, commands, _, _ = make_window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    source = window.findChild(QPushButton, "clearButton")
    qtbot.keyPress(source, Qt.Key.Key_Space)
    assert controller.state.gate_held

    dialog = QDialog(window)
    dialog.setWindowModality(Qt.WindowModality.WindowModal)
    release_target = QPushButton("Dialog target", dialog)
    qtbot.addWidget(dialog)
    window._dialog_chooser_active = True
    try:
        dialog.show()
        qtbot.waitExposed(dialog)
        QApplication.sendEvent(
            release_target,
            QKeyEvent(
                QEvent.Type.KeyRelease,
                Qt.Key.Key_Space,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
    finally:
        window._dialog_chooser_active = False
        dialog.close()

    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.NOTE_OFF,
    ]
    assert not controller.state.gate_held
    assert not window._space_held


def test_secondary_actions_have_non_space_keyboard_activation(qtbot) -> None:
    window, controller, commands, _, dialogs = make_window(qtbot)
    window.show()
    qtbot.waitExposed(window)
    window.activateWindow()
    window.frequency_knob.setFocus()
    QApplication.processEvents()
    assert window.isActiveWindow()
    controller.press_play()

    qtbot.keyClick(window.frequency_knob, Qt.Key.Key_K, Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClick(window.frequency_knob, Qt.Key.Key_O, Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClick(
        window.frequency_knob,
        Qt.Key.Key_S,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    )

    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.CLEAR_CAPTURE,
    ]
    assert dialogs.open_count == 1
    assert dialogs.save_count == 1
    shortcuts = window.findChildren(QShortcut)
    assert len(shortcuts) == 3
    assert window.findChild(EnvelopeEditor, "envelopeEditor").findChildren(QShortcut) == []
    assert (
        sum(
            shortcut.key().matches(QKeySequence.StandardKey.Open)
            is QKeySequence.SequenceMatch.ExactMatch
            for shortcut in shortcuts
        )
        == 1
    )
    assert (
        sum(
            shortcut.key().matches(QKeySequence.StandardKey.SaveAs)
            is QKeySequence.SequenceMatch.ExactMatch
            for shortcut in shortcuts
        )
        == 1
    )


def test_live_frequency_edit_retunes_without_retrigger(qtbot) -> None:
    window, _, commands, _, _ = make_window(qtbot)
    button = window.findChild(QPushButton, "playButton")
    entry = window.findChild(FrequencyEntry, "frequencyEntry")

    qtbot.mousePress(button, Qt.MouseButton.LeftButton)
    entry.setText("330")
    qtbot.keyClick(entry, Qt.Key.Key_Return)

    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.RETUNE,
    ]


def test_natural_idle_stops_retune_and_clear_enters_empty(qtbot) -> None:
    window, controller, commands, _, _ = make_window(qtbot)
    button = window.findChild(QPushButton, "playButton")
    entry = window.findChild(FrequencyEntry, "frequencyEntry")
    generation = controller.state.capture.generation + 1

    qtbot.mousePress(button, Qt.MouseButton.LeftButton)
    qtbot.mouseRelease(button, Qt.MouseButton.LeftButton)
    window.handle_voice_idle(generation)
    entry.setText("330")
    qtbot.keyClick(entry, Qt.Key.Key_Return)
    qtbot.mouseClick(window.findChild(QPushButton, "clearButton"), Qt.MouseButton.LeftButton)

    assert AudioCommandKind.RETUNE not in [command.kind for command in commands]
    assert controller.state.capture.state is CaptureState.EMPTY


def test_measurement_states_and_captured_observation_survive_until_clear(qtbot) -> None:
    observation = observed_signal()
    analyses = iter((observation, observed_silence()))
    window, controller, _, history, _ = make_window(qtbot, analyzer=lambda *_args: next(analyses))
    button = window.findChild(QPushButton, "playButton")
    label = window.findChild(QLabel, "measurementStateLabel")
    waveform = window.findChild(WaveformView, "waveformPlot")
    spectrum = window.findChild(SpectrumView, "spectrumPlot")

    assert label.text() == ""
    assert waveform.curve.xData is None
    assert spectrum.curve.xData is None
    qtbot.mousePress(button, Qt.MouseButton.LeftButton)
    assert label.text() == "Measuring…"
    generation = controller.state.capture.generation
    history.append(np.ones(4, dtype=np.float32), generation)
    window.refresh_capture()
    assert label.text() == "Live"
    qtbot.mouseRelease(button, Qt.MouseButton.LeftButton)
    window.handle_voice_idle(generation)
    window.refresh_capture()
    assert label.text() == "Captured"
    assert waveform.curve.xData is not None
    assert spectrum.curve.xData is not None
    qtbot.mouseClick(window.findChild(QPushButton, "clearButton"), Qt.MouseButton.LeftButton)
    assert label.text() == ""
    assert waveform.curve.xData is None
    assert spectrum.curve.xData is None


def test_idle_attack_commit_applies_once_clears_capture_and_preserves_frequency(qtbot) -> None:
    # Bypassing the controller would either leave stale plots or lose selected frequency state.
    analyses = iter((observed_signal(), observed_silence()))
    window, controller, commands, history, _ = make_window(
        qtbot,
        analyzer=lambda *_args: next(analyses),
    )
    window.frequency_entry.setText("330")
    qtbot.keyClick(window.frequency_entry, Qt.Key.Key_Return)
    window._press_play()
    held_generation = controller.state.capture.generation
    history.append(np.ones(4, dtype=np.float32), held_generation)
    window.refresh_capture()
    window._release_play()
    window.handle_voice_idle(held_generation)
    window.refresh_capture()
    assert controller.state.capture.state is CaptureState.CAPTURED
    assert window.waveform_view.curve.xData is not None
    commands.clear()

    commit_exact_stage(qtbot, window, "attackValueControl", "25 ms")

    assert controller.state.patch.envelope.attack_seconds == 0.025
    assert controller.state.patch_apply_state is PatchApplyState.APPLIED
    assert controller.state.selected_frequency_hz == 330.0
    assert [command.kind for command in commands] == [AudioCommandKind.REPLACE_PATCH]
    assert commands[0].patch == controller.state.patch
    assert commands[0].generation == controller.state.capture.generation
    assert controller.state.capture.state is CaptureState.EMPTY
    assert window.waveform_view.curve.xData is None
    assert window.spectrum_view.curve.xData is None
    assert editor_child(window, QLabel, "patchStatusLabel").text() == "Active"


def test_held_patch_commit_releases_then_applies_only_at_matching_idle(qtbot) -> None:
    # Disabling Play while its gate is held would strand NOTE_ON; accepting stale idle cuts audio.
    observation = observed_signal()
    window, controller, commands, history, _ = make_window(
        qtbot,
        analyzer=lambda *_args: observation,
    )
    window._press_play()
    held_generation = controller.state.capture.generation
    history.append(np.ones(4, dtype=np.float32), held_generation)
    window.refresh_capture()
    before_waveform = np.array(window.waveform_view.curve.yData, copy=True)

    commit_exact_stage(qtbot, window, "attackValueControl", "25 ms")

    authored_patch = controller.state.patch
    assert controller.state.patch_apply_state is PatchApplyState.PENDING
    assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]
    assert window.play_button.isEnabled()
    assert window.play_button.isDown()
    assert editor_child(window, QLabel, "patchStatusLabel").text() == "Pending"
    np.testing.assert_array_equal(window.waveform_view.curve.yData, before_waveform)

    window._release_play()
    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.NOTE_OFF,
    ]
    assert not controller.state.gate_held
    assert controller.state.voice_may_be_active
    assert not window.play_button.isEnabled()

    window._press_play()
    window.handle_voice_idle(held_generation - 1)
    assert controller.state.patch_apply_state is PatchApplyState.PENDING
    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.NOTE_OFF,
    ]

    window.handle_voice_idle(held_generation)
    assert controller.state.patch_apply_state is PatchApplyState.APPLIED
    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.NOTE_OFF,
        AudioCommandKind.REPLACE_PATCH,
    ]
    assert commands[-1].patch == authored_patch
    assert commands[-1].generation == controller.state.capture.generation
    assert window.play_button.isEnabled()
    assert editor_child(window, QLabel, "patchStatusLabel").text() == "Active"
    assert window.waveform_view.curve.xData is None


def test_graph_preview_storm_commits_one_latest_patch_after_release(qtbot) -> None:
    # Sending every preview would flood the audio command boundary with stale replacements.
    window, controller, commands, _, _ = make_window(qtbot)
    graph = editor_child(window, EnvelopeGraph, "envelopeGraph")
    window._press_play()
    generation = controller.state.capture.generation

    graph.field_previewed.emit("attack_curve", 0.10)
    graph.field_previewed.emit("attack_curve", 0.20)
    graph.field_previewed.emit("attack_curve", 0.30)
    graph.field_commit_requested.emit("attack_curve", 0.30)

    assert controller.state.patch.envelope.attack_curve == 0.30
    assert controller.state.patch_apply_state is PatchApplyState.PENDING
    assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]
    window._release_play()
    window.handle_voice_idle(generation)
    replacements = [
        command for command in commands if command.kind is AudioCommandKind.REPLACE_PATCH
    ]
    assert len(replacements) == 1
    assert replacements[0].patch == controller.state.patch
    assert replacements[0].generation == controller.state.capture.generation


def test_clear_stays_available_pending_and_only_aliased_idle_applies_patch(qtbot) -> None:
    # Binding pending application to the pre-Clear token would ignore backend aliasing.
    window, controller, commands, _, _ = make_window(qtbot)
    window._press_play()
    held_generation = controller.state.capture.generation
    commit_exact_stage(qtbot, window, "attackValueControl", "25 ms")
    window._release_play()
    assert window.clear_button.isEnabled()

    qtbot.mouseClick(window.clear_button, Qt.MouseButton.LeftButton)

    clear_generation = controller.state.capture.generation
    assert clear_generation > held_generation
    assert controller.state.patch_apply_state is PatchApplyState.PENDING
    assert controller.state.capture.state is CaptureState.MEASURING
    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.NOTE_OFF,
        AudioCommandKind.CLEAR_CAPTURE,
    ]
    assert commands[-1].generation == clear_generation

    window.handle_voice_idle(held_generation)
    assert controller.state.patch_apply_state is PatchApplyState.PENDING
    window.handle_voice_idle(clear_generation)
    assert controller.state.patch_apply_state is PatchApplyState.APPLIED
    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.NOTE_OFF,
        AudioCommandKind.CLEAR_CAPTURE,
        AudioCommandKind.REPLACE_PATCH,
    ]


def test_capture_refresh_preserves_invalid_editor_text_and_graph_draft(qtbot) -> None:
    # Treating a 34 ms capture refresh as a patch acknowledgment erases in-progress authoring.
    window, controller, commands, _, _ = make_window(qtbot)
    attack = open_exact_stage_editor(qtbot, window, "attackValueControl")
    graph = editor_child(window, EnvelopeGraph, "envelopeGraph")
    replace_entry_text(qtbot, attack, "not complete")
    qtbot.keyClick(attack, Qt.Key.Key_Return)
    graph.field_previewed.emit("release_curve", -0.75)

    window.refresh_capture()

    assert attack.text() == "not complete"
    assert attack.property("validationState") == "error"
    assert graph.envelope.release_curve == -0.75
    assert controller.state.patch == SynthPatch()
    assert commands == []


def test_device_error_outranks_editor_error_and_recovery_clears_only_audio(qtbot) -> None:
    # Letting a later editor signal overwrite device failure hides the actionable root cause.
    window, _, _, _, _ = make_window(qtbot)
    attack = open_exact_stage_editor(qtbot, window, "attackValueControl")
    window.handle_audio_failure("device failed")
    replace_entry_text(qtbot, attack, "0.001 ms")
    qtbot.keyClick(attack, Qt.Key.Key_Return)

    assert attack.property("validationState") == "error"
    assert window._error_label.text() == "device failed"
    qtbot.keyClick(attack, Qt.Key.Key_Escape)
    assert window._error_label.text() == "device failed"

    window.handle_audio_availability(True)

    assert window._error_label.text() == ""
    assert window.audio_error_banner.isHidden()


def test_error_categories_keep_priority_and_success_clears_only_its_owner(
    qtbot,
    tmp_path,
) -> None:
    # A refresh or unrelated success must not clear a still-current higher-priority failure.
    window, controller, commands, _, dialogs = make_window(qtbot)
    dialogs.save_path = tmp_path / "missing" / "saved.json"
    window.findChild(QPushButton, "savePatchButton").click()
    file_error = window._error_label.text()
    assert file_error

    attack = open_exact_stage_editor(qtbot, window, "attackValueControl")
    replace_entry_text(qtbot, attack, "0.001 ms")
    qtbot.keyClick(attack, Qt.Key.Key_Return)
    editor_error = editor_child(window, QLabel, "envelopeFieldError").text()
    frequency_errors: list[str] = []
    window.frequency_entry.validation_failed.connect(frequency_errors.append)
    window.frequency_entry.setText("999")
    qtbot.keyClick(window.frequency_entry, Qt.Key.Key_Return)
    frequency_error = frequency_errors[-1]
    window.refresh_capture()
    assert window._error_label.text() == file_error

    window.handle_audio_failure("device failed")
    assert window._error_label.text() == "device failed"
    window.handle_audio_availability(True)
    assert window._error_label.text() == file_error

    dialogs.save_path = tmp_path / "saved.json"
    window.findChild(QPushButton, "savePatchButton").click()

    assert window._error_label.text() == editor_error
    assert controller.state.patch == SynthPatch()
    assert commands == []
    editor_child(window, QPushButton, "resetEnvelopeButton").click()
    assert window._error_label.text() == frequency_error
    window.frequency_entry.setText("330")
    qtbot.keyClick(window.frequency_entry, Qt.Key.Key_Return)
    assert window._error_label.text() == ""


def test_modal_load_cancel_is_a_true_noop(qtbot) -> None:
    window, controller, commands, _, dialogs = make_window(qtbot)
    window._press_play()
    commit_exact_stage(qtbot, window, "attackValueControl", "25")
    decay = open_exact_stage_editor(qtbot, window, "decayValueControl")
    replace_entry_text(qtbot, decay, "invalid")
    qtbot.keyClick(decay, Qt.Key.Key_Return)
    graph = editor_child(window, EnvelopeGraph, "envelopeGraph")
    graph.field_previewed.emit("release_curve", -0.75)
    before = controller.state
    dialogs.deactivate_during_open = True

    window.findChild(QPushButton, "loadPatchButton").click()

    assert controller.state == before
    assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]
    assert decay.text() == "invalid"
    assert graph.envelope.release_curve == -0.75
    assert dialogs.open_count == 1


def test_modal_save_cancel_preserves_pending_gate_capture_and_editor_draft(qtbot) -> None:
    # Treating cancellation as success would clear errors or force-stop an active gesture.
    window, controller, commands, _, dialogs = make_window(qtbot)
    window._press_play()
    commit_exact_stage(qtbot, window, "attackValueControl", "25 ms")
    decay = open_exact_stage_editor(qtbot, window, "decayValueControl")
    replace_entry_text(qtbot, decay, "not complete")
    qtbot.keyClick(decay, Qt.Key.Key_Return)
    before = controller.state
    dialogs.deactivate_during_save = True

    window.findChild(QPushButton, "savePatchButton").click()

    assert controller.state == before
    assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]
    assert decay.text() == "not complete"
    assert dialogs.save_count == 1


def test_modal_invalid_load_is_atomic(qtbot, tmp_path) -> None:
    window, controller, commands, history, dialogs = make_window(qtbot)
    generation = controller.press_play().capture.generation
    history.append(np.ones(4, dtype=np.float32), generation)
    window.refresh_capture()
    attack, graph = make_invalid_editor_draft(qtbot, window)
    before = controller.state
    bad = tmp_path / "bad.json"
    document = json.loads(dumps_patch(SynthPatch()))
    document["envelope"]["attack_seconds"] = "fast"
    bad.write_text(json.dumps(document), encoding="utf-8")
    dialogs.open_path = bad
    dialogs.deactivate_during_open = True
    window.findChild(QPushButton, "loadPatchButton").click()

    assert controller.state == before
    assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]
    assert attack.text() == "invalid"
    assert graph.envelope.release_curve == -0.75
    banner = window.findChild(QFrame, "audioErrorBanner")
    assert "envelope.attack_seconds" in banner.findChild(QLabel).text()


def test_modal_read_failure_preserves_playback_and_capture(qtbot, tmp_path) -> None:
    window, controller, commands, history, dialogs = make_window(qtbot)
    generation = controller.press_play().capture.generation
    history.append(np.ones(4, dtype=np.float32), generation)
    window.refresh_capture()
    attack, graph = make_invalid_editor_draft(qtbot, window)
    before = controller.state
    dialogs.open_path = tmp_path / "missing.json"
    dialogs.deactivate_during_open = True

    window.findChild(QPushButton, "loadPatchButton").click()

    assert controller.state == before
    assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]
    assert attack.text() == "invalid"
    assert graph.envelope.release_curve == -0.75
    assert window.findChild(QFrame, "audioErrorBanner").isVisibleTo(window)


def test_valid_load_preserves_frequency_force_stops_and_replaces_patch(qtbot, tmp_path) -> None:
    window, controller, commands, _, dialogs = make_window(qtbot)
    replacement = SynthPatch(
        envelope=EnvelopeConfig(attack_seconds=0.025, sustain_db=-9.0),
        output_gain_dbfs=-18.0,
    )
    path = tmp_path / "replacement.json"
    path.write_text(dumps_patch(replacement), encoding="utf-8")
    dialogs.open_path = path
    dialogs.deactivate_during_open = True
    controller.set_frequency(330.0)
    controller.press_play()

    window.findChild(QPushButton, "loadPatchButton").click()

    assert controller.state.selected_frequency_hz == 330.0
    assert controller.state.patch == replacement
    assert controller.state.capture.state is CaptureState.EMPTY
    assert not controller.state.voice_may_be_active
    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.REPLACE_PATCH,
    ]
    assert (
        editor_child(window, EnvelopeStageControl, "attackValueControl").display_text == "A 25 ms"
    )
    assert editor_child(window, QLabel, "outputFact").text() == "-18 dBFS"
    assert editor_child(window, EnvelopeGraph, "envelopeGraph").envelope == replacement.envelope


def test_render_invalid_loaded_patch_is_reported_without_mutation(qtbot, tmp_path) -> None:
    window, controller, commands, history, dialogs = make_window(qtbot)
    generation = controller.press_play().capture.generation
    history.append(np.ones(4, dtype=np.float32), generation)
    window.refresh_capture()
    attack, graph = make_invalid_editor_draft(qtbot, window)
    before = controller.state
    path = tmp_path / "sub-frame.json"
    path.write_text(
        dumps_patch(SynthPatch(envelope=EnvelopeConfig(attack_seconds=1e-12))),
        encoding="utf-8",
    )
    dialogs.open_path = path

    try:
        window._load_patch()
    except ValueError as error:
        pytest.fail(f"render validation escaped the load boundary: {error}")

    assert controller.state == before
    assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]
    assert attack.text() == "invalid"
    assert graph.envelope.release_curve == -0.75
    message = window.findChild(QFrame, "audioErrorBanner").findChild(QLabel).text()
    assert "attack" in message
    assert "frame" in message


def test_valid_load_while_unavailable_updates_patch_and_command_cache(qtbot, tmp_path) -> None:
    cached: list[AudioCommand] = []
    window, controller, _, _, dialogs = make_window(qtbot, send_command=cached.append)
    replacement = SynthPatch(output_gain_dbfs=-24.0)
    path = tmp_path / "replacement.json"
    path.write_text(dumps_patch(replacement), encoding="utf-8")
    dialogs.open_path = path
    window.handle_audio_availability(False)
    window.handle_audio_failure("No default audio output")

    window.findChild(QPushButton, "loadPatchButton").click()

    assert controller.state.patch == replacement
    assert not controller.state.audio_available
    assert cached[-1].kind is AudioCommandKind.REPLACE_PATCH
    assert cached[-1].patch == replacement


def test_save_as_writes_only_canonical_patch_and_failure_does_not_mutate(qtbot, tmp_path) -> None:
    window, controller, commands, _, dialogs = make_window(qtbot)
    controller.press_play()
    attack, graph = make_invalid_editor_draft(qtbot, window)
    before = controller.state
    target = tmp_path / "saved.json"
    dialogs.save_path = target
    dialogs.deactivate_during_save = True
    window.findChild(QPushButton, "savePatchButton").click()

    assert load_patch(target) == controller.state.patch
    assert target.read_text(encoding="utf-8") == dumps_patch(controller.state.patch)
    assert controller.state == before
    assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]
    assert attack.text() == "invalid"
    assert graph.envelope.release_curve == -0.75
    dialogs.save_path = tmp_path / "missing" / "saved.json"
    window.findChild(QPushButton, "savePatchButton").click()
    assert controller.state == before
    assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]
    assert attack.text() == "invalid"
    assert graph.envelope.release_curve == -0.75
    assert window.findChild(QFrame, "audioErrorBanner").isVisibleTo(window)


def test_pending_save_as_writes_latest_authored_patch_as_canonical_v2(qtbot, tmp_path) -> None:
    # Saving the applied audio patch would silently lose the user's newest pending authoring.
    window, controller, commands, _, dialogs = make_window(qtbot)
    window._press_play()
    commit_exact_stage(qtbot, window, "attackValueControl", "25 ms")
    assert controller.state.patch_apply_state is PatchApplyState.PENDING
    target = tmp_path / "pending.json"
    dialogs.save_path = target

    window.findChild(QPushButton, "savePatchButton").click()

    text = target.read_text(encoding="utf-8")
    assert json.loads(text)["schema_version"] == 2
    assert text == dumps_patch(controller.state.patch)
    assert load_patch(target) == controller.state.patch
    assert controller.state.patch_apply_state is PatchApplyState.PENDING
    assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]


def test_v1_load_force_stops_space_discards_drafts_and_later_saves_v2(
    qtbot,
    tmp_path,
) -> None:
    # Merging a pre-Load draft or leaving Space held would corrupt the loaded patch lifecycle.
    curved = SynthPatch(
        envelope=replace(
            SynthPatch().envelope,
            attack_curve=0.25,
            decay_curve=-0.50,
            release_curve=0.75,
        )
    )
    window, controller, commands, _, dialogs = make_window(qtbot, patch=curved)
    window.frequency_entry.setText("330")
    qtbot.keyClick(window.frequency_entry, Qt.Key.Key_Return)
    source = window.clear_button
    qtbot.keyPress(source, Qt.Key.Key_Space)
    commit_exact_stage(qtbot, window, "attackValueControl", "25")
    decay = open_exact_stage_editor(qtbot, window, "decayValueControl")
    replace_entry_text(qtbot, decay, "invalid")
    qtbot.keyClick(decay, Qt.Key.Key_Return)
    graph = editor_child(window, EnvelopeGraph, "envelopeGraph")
    graph.field_previewed.emit("release_curve", -0.75)
    frequency_errors: list[str] = []
    window.frequency_entry.validation_failed.connect(frequency_errors.append)
    window.frequency_entry.setText("999")
    qtbot.keyClick(window.frequency_entry, Qt.Key.Key_Return)
    assert controller.state.patch_apply_state is PatchApplyState.PENDING
    assert controller.state.gate_held

    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{", encoding="utf-8")
    dialogs.open_path = bad_path
    window.findChild(QPushButton, "loadPatchButton").click()
    assert controller.state.patch_apply_state is PatchApplyState.PENDING
    assert controller.state.gate_held
    assert decay.text() == "invalid"
    assert graph.envelope.release_curve == -0.75

    v1_document = {
        "schema_version": 1,
        "oscillator": {"type": "sine"},
        "envelope": {
            "attack_seconds": 0.125,
            "decay_seconds": 0.375,
            "sustain_db": -12.0,
            "release_seconds": 0.625,
            "curve": "linear_amplitude",
        },
        "output_gain_dbfs": -18.0,
    }
    loaded_path = tmp_path / "v1.json"
    loaded_path.write_text(json.dumps(v1_document), encoding="utf-8")
    dialogs.open_path = loaded_path
    dialogs.deactivate_during_open = True
    window.findChild(QPushButton, "loadPatchButton").click()
    qtbot.keyRelease(source, Qt.Key.Key_Space)

    loaded = load_patch(loaded_path)
    assert controller.state.patch == loaded
    assert controller.state.patch_apply_state is PatchApplyState.APPLIED
    assert controller.state.selected_frequency_hz == 330.0
    assert not controller.state.gate_held
    assert not controller.state.voice_may_be_active
    assert not window._space_held
    assert not window.play_button.isDown()
    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.REPLACE_PATCH,
    ]
    assert commands[-1].patch == loaded
    assert commands[-1].generation == controller.state.capture.generation
    assert window.findChild(EnvelopeValueEntry, "envelopeInlineEditor") is None
    assert editor_child(window, EnvelopeStageControl, "decayValueControl").display_text == (
        "D 375 ms"
    )
    assert graph.envelope == loaded.envelope
    for stage in ("attack", "decay", "release"):
        handle = graph.findChild(QWidget, f"{stage}CurveHandle")
        assert handle is not None
        assert handle.current_curve == 0.0
    assert editor_child(window, QLabel, "envelopeFieldError").text() == ""
    assert editor_child(window, QLabel, "patchStatusLabel").text() == "Active"
    assert window._error_label.text() == frequency_errors[-1]

    saved_path = tmp_path / "resaved.json"
    dialogs.save_path = saved_path
    window.findChild(QPushButton, "savePatchButton").click()
    assert json.loads(saved_path.read_text(encoding="utf-8"))["schema_version"] == 2
    assert load_patch(saved_path) == loaded
    assert window._error_label.text() == frequency_errors[-1]


def test_v2_load_synchronizes_all_editor_values_without_feedback(qtbot, tmp_path) -> None:
    # Programmatic Load synchronization must not loop back into a second patch replacement.
    window, controller, commands, _, dialogs = make_window(qtbot)
    editor = window.findChild(EnvelopeEditor, "envelopeEditor")
    graph = editor_child(window, EnvelopeGraph, "envelopeGraph")
    attack = open_exact_stage_editor(qtbot, window, "attackValueControl")
    replace_entry_text(qtbot, attack, "not complete")
    qtbot.keyClick(attack, Qt.Key.Key_Return)
    graph.field_previewed.emit("release_curve", -0.75)
    editor_errors: list[str] = []
    editor_commits: list[SynthPatch] = []
    editor.validation_failed.connect(editor_errors.append)
    editor.patch_commit_requested.connect(editor_commits.append)
    loaded = SynthPatch(
        envelope=EnvelopeConfig(
            attack_seconds=0.125,
            decay_seconds=0.375,
            sustain_db=-15.0,
            release_seconds=0.625,
            attack_curve=0.20,
            decay_curve=-0.40,
            release_curve=0.60,
        ),
        output_gain_dbfs=-24.0,
    )
    path = tmp_path / "v2.json"
    path.write_text(dumps_patch(loaded), encoding="utf-8")
    dialogs.open_path = path

    window.findChild(QPushButton, "loadPatchButton").click()

    assert controller.state.patch == loaded
    assert [command.kind for command in commands] == [AudioCommandKind.REPLACE_PATCH]
    assert editor_commits == []
    assert editor_errors == []
    assert editor_child(window, EnvelopeStageControl, "attackValueControl").display_text == (
        "A 125 ms"
    )
    assert editor_child(window, EnvelopeStageControl, "decayValueControl").display_text == (
        "D 375 ms"
    )
    assert editor_child(window, EnvelopeStageControl, "sustainValueControl").display_text == (
        "S -15 dB"
    )
    assert editor_child(window, EnvelopeStageControl, "releaseValueControl").display_text == (
        "R 625 ms"
    )
    assert graph.envelope == loaded.envelope
    assert graph.findChild(QWidget, "attackCurveHandle").current_curve == 0.20
    assert graph.findChild(QWidget, "decayCurveHandle").current_curve == -0.40
    assert graph.findChild(QWidget, "releaseCurveHandle").current_curve == 0.60


def test_audio_failure_is_single_concise_banner_and_recovery_has_no_success_copy(qtbot) -> None:
    window, controller, _, _, _ = make_window(qtbot)
    button = window.findChild(QPushButton, "playButton")
    banner = window.findChild(QFrame, "audioErrorBanner")

    window.handle_audio_availability(False)
    assert not button.isEnabled()
    window.handle_audio_failure("Audio output failed: OpenError")
    window.handle_audio_failure("Audio output failed: OpenError")

    assert not controller.state.audio_available
    assert not button.isEnabled()
    assert banner.findChild(QLabel).text() == "Audio output failed: OpenError"
    assert len(window.findChildren(QFrame, "audioErrorBanner")) == 1
    window.handle_force_stop()
    assert banner.findChild(QLabel).text() == "Audio output failed: OpenError"
    assert not button.isEnabled()

    window.handle_audio_availability(True)
    assert controller.state.audio_available
    assert button.isEnabled()
    assert banner.isHidden()
    assert "ready" not in " ".join(label.text() for label in window.findChildren(QLabel)).lower()


def test_deactivate_and_shutdown_converge_on_idempotent_force_stop(qtbot) -> None:
    window, _, commands, _, _ = make_window(qtbot)
    button = window.findChild(QPushButton, "playButton")
    shutdowns: list[bool] = []
    window.shutdown_requested.connect(lambda: shutdowns.append(True))
    qtbot.mousePress(button, Qt.MouseButton.LeftButton)

    window.event(QEvent(QEvent.Type.WindowDeactivate))
    window.prepare_shutdown()
    window.prepare_shutdown()

    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.RESET,
    ]
    assert not button.isDown()
    assert shutdowns == [True]
    assert not window._refresh_timer.isActive()


@pytest.mark.parametrize("interaction", ["value_preview", "invalid_exact_entry"])
def test_deactivation_cancels_envelope_interaction_before_force_stop(
    qtbot,
    interaction: str,
) -> None:
    events: list[str] = []

    def record_command(command: AudioCommand) -> None:
        events.append(f"submit:{command.kind.name}")

    window, controller, _, _, _ = make_window(qtbot, send_command=record_command)
    graph = editor_child(window, EnvelopeGraph, "envelopeGraph")
    graph.field_reverted.connect(lambda _field: events.append("field_reverted"))
    window.show()
    qtbot.waitExposed(window)
    window._press_play()
    events.clear()

    if interaction == "value_preview":
        attack = editor_child(window, EnvelopeStageControl, "attackValueControl")
        qtbot.mousePress(attack, Qt.MouseButton.LeftButton, pos=attack.rect().center())
        qtbot.mouseMove(attack, QPoint(attack.rect().center().x(), 0))
        assert attack.is_interacting
    else:
        entry = open_exact_stage_editor(qtbot, window, "attackValueControl")
        replace_entry_text(qtbot, entry, "invalid")
        qtbot.keyClick(entry, Qt.Key.Key_Return)
        assert entry.isVisible()

    window.event(QEvent(QEvent.Type.WindowDeactivate))

    assert events == ["field_reverted", "submit:RESET"]
    assert window.findChild(EnvelopeValueEntry, "envelopeInlineEditor") is None
    assert not controller.state.voice_may_be_active
    assert not window.play_button.isDown()

    window.event(QEvent(QEvent.Type.WindowDeactivate))

    assert events == ["field_reverted", "submit:RESET"]


def test_pending_shutdown_applies_once_discards_invalid_draft_and_never_saves(qtbot) -> None:
    # A RESET followed by replacement, or an implicit Save As, violates shutdown ownership.
    window, controller, commands, _, dialogs = make_window(qtbot)
    shutdowns: list[bool] = []
    window.shutdown_requested.connect(lambda: shutdowns.append(True))
    window._press_play()
    commit_exact_stage(qtbot, window, "attackValueControl", "25 ms")
    decay = open_exact_stage_editor(qtbot, window, "decayValueControl")
    replace_entry_text(qtbot, decay, "not complete")
    qtbot.keyClick(decay, Qt.Key.Key_Return)
    assert controller.state.patch_apply_state is PatchApplyState.PENDING

    window.prepare_shutdown()
    window.prepare_shutdown()

    assert controller.state.patch_apply_state is PatchApplyState.APPLIED
    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.REPLACE_PATCH,
    ]
    assert commands[-1].patch == controller.state.patch
    assert commands[-1].generation == controller.state.capture.generation
    assert window.findChild(EnvelopeValueEntry, "envelopeInlineEditor") is None
    assert editor_child(window, EnvelopeStageControl, "decayValueControl").display_text == (
        "D 600 ms"
    )
    assert editor_child(window, QLabel, "envelopeFieldError").text() == ""
    assert dialogs.save_count == 0
    assert shutdowns == [True]
    assert not window._refresh_timer.isActive()


def test_reset_envelope_has_a_non_space_keyboard_route(qtbot) -> None:
    # If Reset is Space-only, the global transport filter makes the action inaccessible.
    curved = SynthPatch(
        envelope=replace(
            SynthPatch().envelope,
            attack_seconds=0.025,
            decay_seconds=0.250,
            sustain_db=-18.0,
            release_seconds=1.25,
            attack_curve=0.25,
            decay_curve=-0.50,
            release_curve=0.75,
        )
    )
    window, controller, commands, _, _ = make_window(qtbot, patch=curved)
    window.show()
    qtbot.waitExposed(window)
    window.activateWindow()
    QApplication.processEvents()
    assert window.isActiveWindow()
    controller.set_frequency(330.0)
    window._press_play()
    generation = controller.state.capture.generation
    reset = editor_child(window, QPushButton, "resetEnvelopeButton")
    reset.setFocus()

    qtbot.keyClick(reset, Qt.Key.Key_R, Qt.KeyboardModifier.ControlModifier)

    assert controller.state.patch.envelope == EnvelopeConfig()
    assert controller.state.selected_frequency_hz == 330.0
    assert controller.state.patch_apply_state is PatchApplyState.PENDING
    assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]

    window._release_play()
    window.handle_voice_idle(generation)

    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.NOTE_OFF,
        AudioCommandKind.REPLACE_PATCH,
    ]
    assert commands[-1].patch == controller.state.patch


def test_reset_replaces_an_existing_pending_candidate_before_one_idle_apply(qtbot) -> None:
    # Treating Reset as the first pending edit misses stale-candidate or duplicate audio applies.
    starting_patch = SynthPatch(
        envelope=EnvelopeConfig(
            attack_seconds=0.125,
            decay_seconds=0.400,
            sustain_db=-9.0,
            release_seconds=0.850,
            attack_curve=0.2,
            decay_curve=-0.3,
            release_curve=0.4,
        ),
        output_gain_dbfs=-24.0,
    )
    window, controller, commands, _, _ = make_window(qtbot, patch=starting_patch)
    controller.set_frequency(330.0)
    window._press_play()
    generation = controller.state.capture.generation
    window._release_play()

    commit_exact_stage(qtbot, window, "attackValueControl", "250 ms")

    first_pending = replace(
        starting_patch,
        envelope=replace(starting_patch.envelope, attack_seconds=0.250),
    )
    assert controller.state.patch == first_pending
    assert controller.state.patch_apply_state is PatchApplyState.PENDING

    editor_child(window, QPushButton, "resetEnvelopeButton").click()

    expected = replace(starting_patch, envelope=EnvelopeConfig())
    assert controller.state.patch == expected
    assert controller.state.patch != first_pending
    assert controller.state.patch.output_gain_dbfs == -24.0
    assert controller.state.selected_frequency_hz == 330.0
    assert controller.state.patch_apply_state is PatchApplyState.PENDING
    assert [command.kind for command in commands] == [
        AudioCommandKind.NOTE_ON,
        AudioCommandKind.NOTE_OFF,
    ]

    window.handle_voice_idle(generation)

    replacements = [
        command for command in commands if command.kind is AudioCommandKind.REPLACE_PATCH
    ]
    assert len(replacements) == 1
    assert replacements[0].patch == expected
    assert replacements[0].patch.envelope == EnvelopeConfig()
    assert controller.state.patch_apply_state is PatchApplyState.APPLIED

    window._press_play()

    assert commands[-2] is replacements[0]
    assert commands[-1].kind is AudioCommandKind.NOTE_ON
    assert commands[-1].frequency_hz == 330.0
    assert controller.state.patch.envelope == EnvelopeConfig()


@pytest.mark.parametrize("size", [QSize(1_280, 720), QSize(1_024, 640)])
def test_envelope_workbench_layout_contract(qtbot, size: QSize) -> None:
    # Allowing the inspector or error banner to squeeze the actual plot canvas breaks analysis.
    window, _, _, _, _ = make_window(qtbot)
    assert window.minimumSize() == QSize(1024, 640)
    assert window.size() == QSize(1280, 720)
    window.resize(size)
    window.show()
    qtbot.waitExposed(window)

    def assert_geometry() -> None:
        QApplication.processEvents()
        assert window.size() == size
        assert window.transport.height() <= 132
        editor = window.findChild(EnvelopeEditor, "envelopeEditor")
        assert editor is not None and editor.width() == 288
        assert window.frequency_knob.size() == QSize(88, 88)
        plot_container = window.waveform_view.parentWidget()
        assert plot_container is not None
        assert editor.height() == plot_container.height()
        assert editor.mapTo(window.centralWidget(), editor.rect().topLeft()).y() == (
            plot_container.mapTo(window.centralWidget(), plot_container.rect().topLeft()).y()
        )
        waveform_width = window.waveform_view.width()
        spectrum_width = window.spectrum_view.width()
        assert abs(waveform_width / spectrum_width - 4 / 6) <= 0.03
        assert window.waveform_view.plot_item.vb.height() >= 320
        assert window.spectrum_view.plot_item.vb.height() >= 320
        for scrollbar in window.findChildren(QScrollBar):
            if scrollbar.orientation() is Qt.Orientation.Horizontal:
                assert not scrollbar.isVisibleTo(window)

        for child in editor.findChildren(QWidget):
            if child.isWindow() or child.rect().isEmpty() or not child.isVisibleTo(editor):
                continue
            top_left = child.mapTo(editor, child.rect().topLeft())
            bottom_right = child.mapTo(editor, child.rect().bottomRight())
            assert editor.rect().contains(top_left), (child.objectName(), top_left)
            assert editor.rect().contains(bottom_right), (child.objectName(), bottom_right)

        frequency_group = window.findChild(QFrame, "frequencyControlGroup")
        assert frequency_group is not None
        for name in (
            "frequencyLabel",
            "frequencyKnob",
            "frequencyEntry",
            "frequencySuffix",
            "derivedPitchLabel",
        ):
            child = frequency_group.findChild(QWidget, name)
            assert child is not None, name
            top_left = child.mapTo(frequency_group, child.rect().topLeft())
            bottom_right = child.mapTo(frequency_group, child.rect().bottomRight())
            assert frequency_group.rect().contains(top_left), (name, top_left)
            assert frequency_group.rect().contains(bottom_right), (name, bottom_right)

    assert_geometry()
    window._press_play()
    commit_exact_stage(qtbot, window, "attackValueControl", "25 ms")
    assert window._controller.state.patch_apply_state is PatchApplyState.PENDING
    assert_geometry()
    window.handle_audio_failure("Audio output failed: OpenError")
    assert window.audio_error_banner.isVisibleTo(window)
    assert_geometry()
    make_invalid_editor_draft(qtbot, window)
    assert editor_child(window, QLabel, "envelopeFieldError").isVisibleTo(window)
    assert_geometry()
