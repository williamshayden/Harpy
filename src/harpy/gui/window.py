from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from harpy.config import AppConfig
from harpy.gui.controller import ControllerState, LabController
from harpy.gui.qt_audio import QtAudioEngine
from harpy.gui.visualizer import SampleRingBuffer, spectrum_dbfs, waveform_time_ms
from harpy.note_names import format_tuning_readout


class HarpyWindow(QMainWindow):
    def __init__(
        self,
        config: AppConfig,
        controller: LabController,
        audio_engine: QtAudioEngine,
        sample_history: SampleRingBuffer,
    ) -> None:
        super().__init__()
        self._config = config
        self._controller = controller
        self._audio_engine = audio_engine
        self._sample_history = sample_history
        self.setWindowTitle("Harpy · Sine Lab")
        self.resize(1_280, 720)

        self.status_label = QLabel("Audio output not initialized")
        self.pitch_readout = QLabel()
        self.tuning_readout = QLabel(format_tuning_readout(config.tuning))
        self.patch_label = QLabel(
            "Sine · Peak \N{MINUS SIGN}12 dBFS · A 1 ms · D 600 ms · "
            "S \N{MINUS SIGN}6 dB · R 600 ms · Linear amplitude"
        )
        self.pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_slider.setRange(
            config.keyboard.minimum_note.number,
            config.keyboard.maximum_note.number,
        )
        self.pitch_slider.setSingleStep(1)
        self.pitch_slider.setPageStep(1)
        self.play_button = QPushButton("Hold to Play")
        self.play_button.setObjectName("playButton")
        self.play_button.setMinimumHeight(76)
        self.waveform_plot = pg.PlotWidget(title="Recent waveform")
        self.spectrum_plot = pg.PlotWidget(title="Spectrum")
        self._waveform_curve = self.waveform_plot.plot(pen=pg.mkPen("#68d6ff", width=2))
        self._spectrum_curve = self.spectrum_plot.plot(pen=pg.mkPen("#b388ff", width=2))

        header = QHBoxLayout()
        title = QLabel("Harpy · Sine Lab")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.status_label)

        pitch_panel = QFrame()
        pitch_layout = QVBoxLayout(pitch_panel)
        pitch_layout.addWidget(self.pitch_readout)
        pitch_layout.addWidget(self.pitch_slider)
        pitch_layout.addWidget(self.tuning_readout)
        pitch_layout.addWidget(self.patch_label)
        pitch_layout.addWidget(self.play_button)

        plots = QHBoxLayout()
        plots.addWidget(self.waveform_plot, 1)
        plots.addWidget(self.spectrum_plot, 1)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addLayout(header)
        layout.addWidget(pitch_panel)
        layout.addLayout(plots, 1)
        self.setCentralWidget(root)

        self.pitch_slider.valueChanged.connect(self._on_note_changed)
        self.play_button.pressed.connect(self._on_play_pressed)
        self.play_button.released.connect(self._on_play_released)
        audio_engine.status_changed.connect(self._on_audio_status)
        audio_engine.force_stop_requested.connect(self._force_stop)

        self._plot_timer = QTimer(self)
        self._plot_timer.setInterval(33)
        self._plot_timer.timeout.connect(self._update_plots)
        self._plot_timer.start()
        self._audio_playable = False
        self._apply_state(controller.state)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #111318; color: #eef2f7; }
            QFrame { background: #181c23; border: 1px solid #2a313d; border-radius: 12px; }
            QLabel#title { font-size: 24px; font-weight: 700; }
            QLabel { font-size: 15px; }
            QPushButton#playButton {
                background: #68d6ff; color: #071017; border: 0; border-radius: 12px;
                font-size: 20px; font-weight: 700; padding: 16px;
            }
            QPushButton#playButton:pressed { background: #b388ff; }
            QPushButton#playButton:disabled { background: #343b46; color: #7f8998; }
            QSlider::groove:horizontal { height: 8px; background: #2a313d; border-radius: 4px; }
            QSlider::handle:horizontal {
                width: 22px; margin: -8px 0; background: #68d6ff; border-radius: 11px;
            }
            """
        )

    def _apply_state(self, state: ControllerState) -> None:
        self.pitch_slider.blockSignals(True)
        self.pitch_slider.setValue(state.note.number)
        self.pitch_slider.blockSignals(False)
        self.pitch_readout.setText(state.readout)
        self.pitch_slider.setEnabled(state.selector_enabled)
        self.play_button.setEnabled(self._audio_playable)

    def _on_note_changed(self, number: int) -> None:
        self._apply_state(self._controller.set_note(number))

    def _on_play_pressed(self) -> None:
        self._apply_state(self._controller.press_play())

    def _on_play_released(self) -> None:
        self._apply_state(self._controller.release_play())

    def _force_stop(self) -> None:
        self.play_button.setDown(False)
        self._sample_history.clear()
        self._apply_state(self._controller.force_stop())

    def _on_audio_status(self, message: str, playable: bool) -> None:
        self._audio_playable = playable
        self.status_label.setText(message)
        self._apply_state(self._controller.state)

    def _update_plots(self) -> None:
        try:
            waveform = self._sample_history.snapshot(2_048)
            time_ms = waveform_time_ms(waveform.size, self._config.render.sample_rate_hz)
            self._waveform_curve.setData(time_ms, waveform)
            frequencies, levels = spectrum_dbfs(
                self._sample_history.snapshot(4_096),
                self._config.render.sample_rate_hz,
            )
            self._spectrum_curve.setData(frequencies, levels)
            self.waveform_plot.setLabel("bottom", "Time", units="ms")
            self.spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
            self.spectrum_plot.setLabel("left", "Level", units="dBFS")
            self.spectrum_plot.setYRange(-120.0, 0.0)
            self.spectrum_plot.setXRange(0.0, 2_000.0)
        except (FloatingPointError, RuntimeError, ValueError) as error:
            self._plot_timer.stop()
            message = f"Visualization disabled: {error}"
            self.waveform_plot.setTitle(message)
            self.spectrum_plot.setTitle(message)

    def event(self, event: QEvent) -> bool:
        if event.type() is QEvent.Type.WindowDeactivate:
            self._force_stop()
        return super().event(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._plot_timer.stop()
        self._force_stop()
        self._audio_engine.shutdown()
        event.accept()
