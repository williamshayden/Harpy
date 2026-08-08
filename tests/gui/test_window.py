from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QRect, QSize, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from harpy.analysis import AnalysisConfig, AudioObservation
from harpy.capture import CaptureCoordinator, CaptureState, SampleHistory
from harpy.gui.frequency_entry import FrequencyEntry
from harpy.gui.frequency_knob import FrequencyKnob
from harpy.gui.signal_views import SpectrumView, WaveformView
from harpy.gui.window import HarpyWindow
from harpy.gui.workbench_controller import WorkbenchController
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
    window = HarpyWindow(controller, tuning, spec, dialogs)
    qtbot.addWidget(window)
    return window, controller, commands, history, dialogs


def patch_fact_pairs(window: HarpyWindow) -> list[tuple[str, str]]:
    facts = window.findChild(QFrame, "patchFacts")
    layout = facts.layout()
    assert isinstance(layout, QGridLayout)
    pairs: list[tuple[str, str]] = []
    for column in range(7):
        heading_item = layout.itemAtPosition(0, column)
        value_item = layout.itemAtPosition(1, column)
        assert heading_item is not None
        assert value_item is not None
        heading = heading_item.widget()
        value = value_item.widget()
        assert isinstance(heading, QLabel)
        assert isinstance(value, QLabel)
        pairs.append((heading.text(), value.text()))
    return pairs


def test_final_widget_contract_and_copy_contains_no_legacy_or_device_status(qtbot) -> None:
    window, _, _, _, _ = make_window(qtbot)

    assert window.windowTitle() == "Harpy"
    expected = {
        "frequencyKnob": FrequencyKnob,
        "frequencyEntry": FrequencyEntry,
        "derivedPitchLabel": QLabel,
        "playButton": QPushButton,
        "measurementStateLabel": QLabel,
        "clearButton": QPushButton,
        "waveformPlot": WaveformView,
        "spectrumPlot": SpectrumView,
        "patchFacts": QFrame,
        "loadPatchButton": QPushButton,
        "savePatchButton": QPushButton,
        "audioErrorBanner": QFrame,
    }
    for object_name, widget_type in expected.items():
        widget = window.findChild(widget_type, object_name)
        assert widget is not None, object_name
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
    ["frequencyKnob", "playButton", "clearButton", "loadPatchButton", "savePatchButton"],
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


def test_patch_facts_include_curve_and_keep_output_as_seventh_value(qtbot) -> None:
    window, _, _, _, _ = make_window(qtbot)
    assert patch_fact_pairs(window) == [
        ("Oscillator", "Sine"),
        ("Attack", "1 ms"),
        ("Decay", "600 ms"),
        ("Sustain", "\N{MINUS SIGN}6 dB"),
        ("Release", "600 ms"),
        ("Curve", "Linear"),
        ("Output", "\N{MINUS SIGN}12 dBFS"),
    ]

    curved_patch = replace(
        SynthPatch(),
        envelope=replace(SynthPatch().envelope, attack_curve=0.25),
    )
    curved_window, _, _, _, _ = make_window(qtbot, patch=curved_patch)
    assert patch_fact_pairs(curved_window) == [
        ("Oscillator", "Sine"),
        ("Attack", "1 ms"),
        ("Decay", "600 ms"),
        ("Sustain", "\N{MINUS SIGN}6 dB"),
        ("Release", "600 ms"),
        ("Curve", "Curved"),
        ("Output", "\N{MINUS SIGN}12 dBFS"),
    ]


def test_modal_load_cancel_is_a_true_noop(qtbot) -> None:
    window, controller, commands, _, dialogs = make_window(qtbot)
    controller.press_play()
    before = controller.state
    dialogs.deactivate_during_open = True

    window.findChild(QPushButton, "loadPatchButton").click()

    assert controller.state == before
    assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]
    assert dialogs.open_count == 1


def test_modal_invalid_load_is_atomic(qtbot, tmp_path) -> None:
    window, controller, commands, history, dialogs = make_window(qtbot)
    generation = controller.press_play().capture.generation
    history.append(np.ones(4, dtype=np.float32), generation)
    window.refresh_capture()
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
    banner = window.findChild(QFrame, "audioErrorBanner")
    assert "envelope.attack_seconds" in banner.findChild(QLabel).text()


def test_modal_read_failure_preserves_playback_and_capture(qtbot, tmp_path) -> None:
    window, controller, commands, history, dialogs = make_window(qtbot)
    generation = controller.press_play().capture.generation
    history.append(np.ones(4, dtype=np.float32), generation)
    window.refresh_capture()
    before = controller.state
    dialogs.open_path = tmp_path / "missing.json"
    dialogs.deactivate_during_open = True

    window.findChild(QPushButton, "loadPatchButton").click()

    assert controller.state == before
    assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]
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
    facts = " ".join(
        label.text() for label in window.findChild(QFrame, "patchFacts").findChildren(QLabel)
    )
    assert "25 ms" in facts
    assert "\N{MINUS SIGN}18 dBFS" in facts


def test_render_invalid_loaded_patch_is_reported_without_mutation(qtbot, tmp_path) -> None:
    window, controller, commands, history, dialogs = make_window(qtbot)
    generation = controller.press_play().capture.generation
    history.append(np.ones(4, dtype=np.float32), generation)
    window.refresh_capture()
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
    before = controller.state
    target = tmp_path / "saved.json"
    dialogs.save_path = target
    dialogs.deactivate_during_save = True
    window.findChild(QPushButton, "savePatchButton").click()

    assert load_patch(target) == controller.state.patch
    assert target.read_text(encoding="utf-8") == dumps_patch(controller.state.patch)
    assert controller.state == before
    assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]
    dialogs.save_path = tmp_path / "missing" / "saved.json"
    window.findChild(QPushButton, "savePatchButton").click()
    assert controller.state == before
    assert [command.kind for command in commands] == [AudioCommandKind.NOTE_ON]
    assert window.findChild(QFrame, "audioErrorBanner").isVisibleTo(window)


def test_audio_failure_is_single_concise_banner_and_recovery_has_no_success_copy(qtbot) -> None:
    window, controller, _, _, _ = make_window(qtbot)
    button = window.findChild(QPushButton, "playButton")
    banner = window.findChild(QFrame, "audioErrorBanner")

    window.handle_audio_availability(False)
    assert button.isEnabled()
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


def test_layout_contract_at_minimum_and_initial_sizes(qtbot, tmp_path) -> None:
    window, _, _, _, _ = make_window(qtbot)
    assert window.minimumSize() == QSize(1024, 640)
    assert window.size() == QSize(1280, 720)
    names = (
        "frequencyKnob",
        "frequencyEntry",
        "derivedPitchLabel",
        "playButton",
        "measurementStateLabel",
        "clearButton",
        "waveformPlot",
        "spectrumPlot",
        "patchFacts",
        "loadPatchButton",
        "savePatchButton",
    )
    geometry_states = (None, "Audio output failed: OpenError")
    for size, error in (
        (QSize(1280, 720), None),
        *[(QSize(1024, 640), state) for state in geometry_states],
    ):
        if error is None:
            window.handle_audio_availability(True)
        else:
            window.handle_audio_failure(error)
        window.resize(size)
        window.show()
        qtbot.waitExposed(window)
        qtbot.wait(1)
        assert window.size() == size
        client = QRect(window.centralWidget().rect())
        assert window.transport.height() <= 144
        assert window.findChild(QFrame, "patchFacts").height() <= 96
        waveform = window.findChild(WaveformView, "waveformPlot")
        spectrum = window.findChild(SpectrumView, "spectrumPlot")
        assert waveform.height() >= 320
        assert spectrum.height() >= 320
        assert waveform.plot_item.vb.sceneBoundingRect().height() >= 320
        assert spectrum.plot_item.vb.sceneBoundingRect().height() >= 320
        assert spectrum.width() > waveform.width()
        for name in names:
            widget = window.findChild(QWidget, name)
            top_left = widget.mapTo(window.centralWidget(), widget.rect().topLeft())
            geometry = QRect(top_left, widget.size())
            assert not geometry.isEmpty(), name
            assert client.contains(geometry), (name, geometry, client)
        if error is not None:
            banner = window.findChild(QFrame, "audioErrorBanner")
            banner_top_left = banner.mapTo(window.centralWidget(), banner.rect().topLeft())
            assert banner.isVisibleTo(window)
            assert client.contains(QRect(banner_top_left, banner.size()))

    window.resize(1280, 720)
    image = window.grab().toImage()
    if image.isNull() or image.size() != QSize(1280, 720):
        image.save(str(tmp_path / "workbench-layout-failure.png"))
    assert not image.isNull()
    assert image.size() == QSize(1280, 720)
