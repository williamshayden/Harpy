from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QCloseEvent, QKeyEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from harpy.capture import CaptureState
from harpy.gui.envelope_editor import EnvelopeEditor
from harpy.gui.frequency_entry import FrequencyEntry
from harpy.gui.frequency_knob import FrequencyKnob
from harpy.gui.patch_dialogs import PatchDialogPort
from harpy.gui.signal_views import SpectrumView, WaveformView
from harpy.gui.workbench_controller import (
    PatchApplyState,
    WorkbenchController,
    WorkbenchState,
)
from harpy.gui.workbench_spec import WorkbenchSpec
from harpy.synth.models import RenderConfig, SynthPatch
from harpy.synth.patch_json import load_patch, save_patch
from harpy.tuning import Tuning


class HarpyWindow(QMainWindow):
    """Native workbench view backed by one semantic controller."""

    shutdown_requested = Signal()

    def __init__(
        self,
        controller: WorkbenchController,
        tuning: Tuning,
        spec: WorkbenchSpec,
        render: RenderConfig,
        patch_dialogs: PatchDialogPort,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._tuning = tuning
        self._patch_dialogs = patch_dialogs
        self._shutdown_prepared = False
        self._frequency_error: str | None = None
        self._editor_error: str | None = None
        self._file_error: str | None = None
        self._space_held = False
        self._dialog_chooser_active = False
        self._space_filter_installed = False

        self.setWindowTitle("Harpy")
        self.setMinimumSize(1_024, 640)
        self.resize(1_280, 720)

        self.frequency_knob = FrequencyKnob(
            spec.minimum_frequency_hz,
            spec.center_frequency_hz,
            spec.maximum_frequency_hz,
        )
        self.frequency_knob.setObjectName("frequencyKnob")
        self.frequency_knob.setFixedSize(88, 88)

        self.frequency_entry = FrequencyEntry(
            spec.minimum_frequency_hz,
            spec.maximum_frequency_hz,
        )
        self.frequency_entry.setObjectName("frequencyEntry")
        self.frequency_entry.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.frequency_entry.setMinimumWidth(154)
        self.frequency_entry.setMaximumWidth(190)
        self.frequency_entry.setAccessibleName("Frequency in hertz")
        self.frequency_entry.setAccessibleDescription("Enter the playback frequency in hertz.")
        hz_suffix = QLabel("Hz")
        hz_suffix.setObjectName("frequencySuffix")

        self.derived_pitch_label = QLabel()
        self.derived_pitch_label.setObjectName("derivedPitchLabel")
        self.derived_pitch_label.setMinimumWidth(116)

        frequency_label = QLabel("Frequency")
        frequency_label.setObjectName("frequencyLabel")
        self.frequency_control_group = QFrame()
        self.frequency_control_group.setObjectName("frequencyControlGroup")
        frequency_group_layout = QVBoxLayout(self.frequency_control_group)
        frequency_group_layout.setContentsMargins(10, 3, 10, 6)
        frequency_group_layout.setSpacing(1)
        frequency_group_layout.addWidget(frequency_label)
        frequency_row = QHBoxLayout()
        frequency_row.setContentsMargins(0, 0, 0, 0)
        frequency_row.setSpacing(7)
        frequency_row.addWidget(self.frequency_knob)
        entry_row = QHBoxLayout()
        entry_row.setContentsMargins(0, 0, 0, 0)
        entry_row.setSpacing(6)
        entry_row.addWidget(self.frequency_entry)
        entry_row.addWidget(hz_suffix)
        frequency_row.addLayout(entry_row)
        frequency_row.addWidget(self.derived_pitch_label)
        frequency_group_layout.addLayout(frequency_row)

        self.play_button = QPushButton("Play")
        self.play_button.setObjectName("playButton")
        self.play_button.setAccessibleDescription("Press and hold to play")
        self.play_button.setMinimumSize(112, 44)
        self.play_button.setMaximumHeight(48)

        self.transport = QFrame()
        self.transport.setObjectName("transportStrip")
        self.transport.setMaximumHeight(132)
        transport_layout = QHBoxLayout(self.transport)
        transport_layout.setContentsMargins(14, 6, 14, 6)
        transport_layout.setSpacing(14)
        transport_layout.addWidget(self.frequency_control_group)
        transport_layout.addStretch(1)
        transport_layout.addWidget(self.play_button)

        self.measurement_state_label = QLabel()
        self.measurement_state_label.setObjectName("measurementStateLabel")
        self.clear_button = QPushButton("&Clear")
        self.clear_button.setObjectName("clearButton")
        self.clear_button.setToolTip("Clear measurement (Ctrl+K)")
        self.clear_button.setMaximumWidth(88)
        measurement_header = QHBoxLayout()
        measurement_header.setContentsMargins(2, 0, 2, 0)
        measurement_header.addWidget(self.measurement_state_label)

        self.waveform_view = WaveformView()
        self.waveform_view.setObjectName("waveformPlot")
        self.waveform_view.setMinimumHeight(320)
        self.waveform_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.waveform_view.plot_item.setTitle("Waveform")
        self.spectrum_view = SpectrumView()
        self.spectrum_view.setObjectName("spectrumPlot")
        self.spectrum_view.setMinimumHeight(320)
        self.spectrum_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.spectrum_view.plot_item.setTitle("Spectrum")
        plot_container = QWidget()
        plot_container.setObjectName("plotContainer")
        plot_layout = QHBoxLayout(plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(12)
        plot_layout.addWidget(self.waveform_view, 4)
        plot_layout.addWidget(self.spectrum_view, 6)
        self.envelope_editor = EnvelopeEditor(render, controller.state.patch)
        self.envelope_editor.setObjectName("envelopeEditor")
        self.envelope_editor.setFixedWidth(288)
        self._reset_envelope_button = self.envelope_editor.findChild(
            QPushButton,
            "resetEnvelopeButton",
        )
        if self._reset_envelope_button is None:
            raise RuntimeError("EnvelopeEditor is missing resetEnvelopeButton")
        self._reset_envelope_button.setToolTip("Reset envelope to defaults (Ctrl+R)")
        self._reset_envelope_action = QAction(self)
        self._reset_envelope_action.setObjectName("resetEnvelopeAction")
        self._reset_envelope_action.setShortcut(QKeySequence("Ctrl+R"))
        self._reset_envelope_action.triggered.connect(self._reset_envelope_button.click)
        self.addAction(self._reset_envelope_action)
        main_row = QHBoxLayout()
        main_row.setContentsMargins(0, 0, 0, 0)
        main_row.setSpacing(12)
        main_row.addWidget(plot_container, 1)
        main_row.addWidget(self.envelope_editor)

        self.audio_error_banner = QFrame()
        self.audio_error_banner.setObjectName("audioErrorBanner")
        error_layout = QHBoxLayout(self.audio_error_banner)
        error_layout.setContentsMargins(12, 6, 12, 6)
        self._error_label = QLabel()
        self._error_label.setWordWrap(True)
        error_layout.addWidget(self._error_label)
        self.audio_error_banner.hide()
        measurement_header.addWidget(self.audio_error_banner, 1)
        measurement_header.addWidget(self.clear_button)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 8, 14, 8)
        root.setSpacing(7)
        root.addWidget(self.transport)
        root.addLayout(measurement_header)
        root.addLayout(main_row, 1)
        self.setCentralWidget(central)

        self.frequency_knob.frequency_changed.connect(self._set_frequency)
        self.frequency_entry.frequency_committed.connect(self._set_frequency)
        self.frequency_entry.validation_failed.connect(self._show_frequency_error)
        self.play_button.pressed.connect(self._press_play)
        self.play_button.released.connect(self._release_play)
        self.clear_button.clicked.connect(self._clear_measurement)
        self.envelope_editor.patch_commit_requested.connect(self._commit_authored_patch)
        self.envelope_editor.validation_failed.connect(self._show_editor_error)
        self.envelope_editor.validation_cleared.connect(self._clear_editor_error)
        self.envelope_editor.load_requested.connect(self._load_patch)
        self.envelope_editor.save_requested.connect(self._save_patch)
        self._clear_shortcut = QShortcut(QKeySequence("Ctrl+K"), self)
        self._load_shortcut = QShortcut(QKeySequence.StandardKey.Open, self)
        self._save_shortcut = QShortcut(QKeySequence.StandardKey.SaveAs, self)
        self._clear_shortcut.activated.connect(self._clear_measurement)
        self._load_shortcut.activated.connect(self._load_patch)
        self._save_shortcut.activated.connect(self._save_patch)

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
            self._space_filter_installed = True

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(34)
        self._refresh_timer.timeout.connect(self.refresh_capture)
        self._refresh_timer.start()

        self._apply_style()
        self._apply_state(controller.state, sync_frequency=True)

    @Slot(bool)
    def handle_audio_availability(self, available: bool) -> None:
        self._apply_state(self._controller.set_audio_availability(available))

    @Slot(str)
    def handle_audio_failure(self, message: str) -> None:
        self._apply_state(self._controller.set_audio_availability(False, message))

    @Slot()
    def handle_force_stop(self) -> None:
        self.play_button.setDown(False)
        self._space_held = False
        self._apply_state(self._controller.force_stop())

    @Slot(int)
    def handle_voice_idle(self, generation: int) -> None:
        self._apply_state(self._controller.mark_voice_idle(generation))

    @Slot()
    def refresh_capture(self) -> None:
        self._apply_state(self._controller.refresh_capture())

    @Slot()
    def prepare_shutdown(self) -> None:
        if self._shutdown_prepared:
            return
        self._shutdown_prepared = True
        self._refresh_timer.stop()
        self.envelope_editor.discard_draft()
        self._editor_error = None
        self.handle_force_stop()
        self._remove_space_event_filter()
        self.shutdown_requested.emit()

    def event(self, event: QEvent) -> bool:
        if event.type() is QEvent.Type.WindowDeactivate and not self._dialog_chooser_active:
            self.handle_force_stop()
        return super().event(event)

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        if (
            isinstance(event, QKeyEvent)
            and event.type() is QEvent.Type.KeyRelease
            and event.key() == Qt.Key.Key_Space
            and not event.isAutoRepeat()
            and self._space_held
        ):
            self._space_held = False
            self.play_button.setDown(False)
            self._release_play()
            return True
        if not isinstance(watched, QWidget) or watched.window() is not self:
            return super().eventFilter(watched, event)
        if not isinstance(event, QKeyEvent):
            return super().eventFilter(watched, event)
        if event.type() is QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
            if watched is self.frequency_entry:
                self._frequency_error = None
                self._apply_state(self._controller.state, sync_frequency=True)
                return True
            return super().eventFilter(watched, event)
        if event.key() != Qt.Key.Key_Space or isinstance(watched, QLineEdit):
            return super().eventFilter(watched, event)
        if event.type() is QEvent.Type.KeyPress:
            if not event.isAutoRepeat() and not self._space_held and self.play_button.isEnabled():
                self._space_held = True
                self.play_button.setDown(True)
                self._press_play()
            return True
        if event.type() is QEvent.Type.KeyRelease:
            if not event.isAutoRepeat() and self._space_held:
                self._space_held = False
                self.play_button.setDown(False)
                self._release_play()
            return True
        return super().eventFilter(watched, event)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.prepare_shutdown()
        event.accept()

    @Slot(float)
    def _set_frequency(self, frequency_hz: float) -> None:
        self._frequency_error = None
        self._apply_state(
            self._controller.set_frequency(frequency_hz),
            sync_frequency=True,
        )

    @Slot(object)
    def _commit_authored_patch(self, patch: SynthPatch) -> None:
        self._apply_state(self._controller.commit_authored_patch(patch))

    @Slot()
    def _press_play(self) -> None:
        self._apply_state(self._controller.press_play())

    @Slot()
    def _release_play(self) -> None:
        self._apply_state(self._controller.release_play())

    @Slot()
    def _clear_measurement(self) -> None:
        self._apply_state(self._controller.clear_measurement())

    @Slot()
    def _load_patch(self) -> None:
        path = self._choose_open_path()
        if path is None:
            return
        try:
            patch = load_patch(path)
        except (OSError, ValueError) as error:
            self._show_file_error(str(error))
            return
        try:
            state = self._controller.replace_patch(patch)
        except ValueError as error:
            self._show_file_error(str(error))
            return
        self.play_button.setDown(False)
        self._space_held = False
        self._file_error = None
        self._editor_error = None
        self._apply_state(state, discard_editor_draft=True)

    @Slot()
    def _save_patch(self) -> None:
        path = self._choose_save_path()
        if path is None:
            return
        try:
            save_patch(path, self._controller.state.patch)
        except (OSError, ValueError) as error:
            self._show_file_error(str(error))
            return
        self._file_error = None
        self._apply_state(self._controller.state)

    @Slot(str)
    def _show_file_error(self, message: str) -> None:
        self._file_error = message
        self._apply_state(self._controller.state)

    @Slot(str)
    def _show_frequency_error(self, message: str) -> None:
        self._frequency_error = message
        self._apply_state(self._controller.state)

    @Slot(str)
    def _show_editor_error(self, message: str) -> None:
        self._editor_error = message
        self._apply_state(self._controller.state)

    @Slot()
    def _clear_editor_error(self) -> None:
        self._editor_error = None
        self._apply_state(self._controller.state)

    def _apply_state(
        self,
        state: WorkbenchState,
        *,
        sync_frequency: bool = False,
        discard_editor_draft: bool = False,
    ) -> None:
        frequency = state.selected_frequency_hz
        self.frequency_knob.set_frequency_hz(frequency)
        if sync_frequency or not self.frequency_entry.isModified():
            self.frequency_entry.set_frequency_hz(frequency)
        reading = self._tuning.describe_frequency(frequency)
        cents = 0.0 if abs(reading.cents) < 0.05 else reading.cents
        self.derived_pitch_label.setText(f"{reading.name} {cents:+.1f}¢")
        can_release_held_gate = state.gate_held
        can_start_new_voice = (
            state.audio_available and state.patch_apply_state is PatchApplyState.APPLIED
        )
        self.play_button.setEnabled(can_release_held_gate or can_start_new_voice)
        self.play_button.setDown(state.gate_held or self._space_held)

        capture_labels = {
            CaptureState.EMPTY: "",
            CaptureState.MEASURING: "Measuring…",
            CaptureState.LIVE: "Live",
            CaptureState.CAPTURED: "Captured",
        }
        self.measurement_state_label.setText(capture_labels[state.capture.state])
        self.waveform_view.set_observation(state.capture.observation)
        self.spectrum_view.set_observation(state.capture.observation)
        self.envelope_editor.set_patch_state(
            state.patch,
            state.patch_apply_state,
            discard_draft=discard_editor_draft,
        )

        error = state.audio_error or self._file_error or self._editor_error or self._frequency_error
        self._error_label.setText(error or "")
        self.audio_error_banner.setVisible(error is not None)

    def _choose_open_path(self):
        self._dialog_chooser_active = True
        try:
            return self._patch_dialogs.choose_open_path(self)
        finally:
            self._dialog_chooser_active = False

    def _choose_save_path(self):
        self._dialog_chooser_active = True
        try:
            return self._patch_dialogs.choose_save_path(self)
        finally:
            self._dialog_chooser_active = False

    def _remove_space_event_filter(self) -> None:
        if not self._space_filter_installed:
            return
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        self._space_filter_installed = False

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #11151b;
                color: #e8edf4;
                font-size: 12px;
            }
            QLabel { background: transparent; border: none; }
            QFrame#transportStrip, QFrame#frequencyControlGroup {
                background: #181e27;
                border: 1px solid #2a3441;
                border-radius: 8px;
            }
            QLabel#frequencyLabel {
                color: #95a2b2;
                font-size: 12px;
                font-weight: 600;
            }
            QLineEdit#frequencyEntry {
                background: #0d1117;
                border: 1px solid #465364;
                border-radius: 5px;
                color: #f4f7fb;
                font-family: monospace;
                font-size: 22px;
                padding: 6px 8px;
            }
            QLineEdit#frequencyEntry:focus { border: 2px solid #65d8ff; }
            QLineEdit#frequencyEntry[validationState="error"] { border: 2px solid #ff6b72; }
            QLabel#derivedPitchLabel {
                color: #65d8ff;
                font-family: monospace;
                font-size: 20px;
                font-weight: 600;
            }
            QLabel#measurementStateLabel { color: #bd8cff; font-weight: 600; }
            QPushButton {
                background: #252e3a;
                border: 1px solid #3a4655;
                border-radius: 6px;
                color: #eef3f8;
                min-height: 32px;
                padding: 4px 12px;
            }
            QPushButton:hover, QPushButton:focus { border-color: #65d8ff; }
            QPushButton#playButton { background: #65d8ff; color: #081218; font-weight: 700; }
            QPushButton#playButton:pressed { background: #bd8cff; }
            QPushButton#playButton:disabled { background: #313945; color: #778290; }
            QFrame#audioErrorBanner {
                background: #321b21;
                border: 1px solid #ff6b72;
                border-radius: 6px;
            }
            """
        )
