"""Fixed-scale, observation-only waveform and spectrum displays."""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from harpy.analysis import AudioObservation


class _LockedViewBox(pg.ViewBox):
    """A ViewBox that cannot alter the deliberate scientific view ranges."""

    wheel_zoom_disabled = True

    def wheelEvent(self, event, axis=None):  # type: ignore[no-untyped-def]
        event.ignore()

    def mouseClickEvent(self, event):  # type: ignore[no-untyped-def]
        event.ignore()

    def mouseDragEvent(self, event, axis=None):  # type: ignore[no-untyped-def]
        event.ignore()

    def raiseContextMenu(self, event):  # type: ignore[no-untyped-def]
        event.ignore()


class _ScientificView(QWidget):
    _EMPTY_MESSAGE = "Hold Play to inspect the signal."

    def __init__(self, *, pen: str, marker: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        view_box = _LockedViewBox(enableMenu=False)
        self.plot = pg.PlotWidget(viewBox=view_box)
        self.plot_item = self.plot.getPlotItem()
        self.plot_item.setMenuEnabled(False)
        self.plot_item.setMouseEnabled(x=False, y=False)
        self.plot_item.vb.setMenuEnabled(False)
        self.plot_item.vb.setMouseEnabled(x=False, y=False)
        self.plot.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.curve = self.plot_item.plot(pen=pg.mkPen(pen, width=2))
        self.marker = self.plot_item.plot(
            pen=pg.mkPen(marker, width=1.5),
            symbolPen=pg.mkPen(marker, width=1.5),
            symbolBrush=pg.mkBrush(marker),
            size=9,
            symbol="o",
        )
        self.readout = QLabel(self._EMPTY_MESSAGE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.readout)
        self._set_empty()

    def _set_empty(self) -> None:
        self.curve.setData(np.empty(0), np.empty(0))
        self.marker.setData(np.empty(0), np.empty(0))
        self.readout.setText(self._EMPTY_MESSAGE)


class WaveformView(_ScientificView):
    """A fixed 50 ms, full-scale waveform display."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(pen="#68d6ff", marker="#68d6ff", parent=parent)
        self.plot_item.setLabel("bottom", "Time (ms)")
        self.plot_item.setLabel("left", "Amplitude (FS)")
        self.plot_item.setXRange(0.0, 50.0, padding=0.0)
        self.plot_item.setYRange(-1.0, 1.0, padding=0.0)
        self.plot_item.vb.setLimits(xMin=0.0, xMax=50.0, yMin=-1.0, yMax=1.0)
        self.plot_item.vb.enableAutoRange(enable=False)

    def set_observation(self, observation: AudioObservation | None) -> None:
        if observation is None or not observation.has_signal:
            self._set_empty()
            return
        self.curve.setData(observation.waveform_time_ms, observation.waveform_samples)
        self.marker.setData(np.empty(0), np.empty(0))
        if observation.peak_amplitude_fs is None:
            self.readout.setText(self._EMPTY_MESSAGE)
        else:
            self.readout.setText(f"Peak {observation.peak_amplitude_fs:.3f} FS")


class SpectrumView(_ScientificView):
    """A fixed log-frequency spectrum driven solely by measured observations."""

    _TICKS = (
        (20, "20"),
        (50, "50"),
        (100, "100"),
        (200, "200"),
        (500, "500"),
        (1_000, "1k"),
        (2_000, "2k"),
        (5_000, "5k"),
        (10_000, "10k"),
        (20_000, "20k"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(pen="#b388ff", marker="#ffca6a", parent=parent)
        self.plot_item.setLogMode(x=True, y=False)
        self.plot_item.setLabel("bottom", "Frequency (Hz)")
        self.plot_item.setLabel("left", "Level (dBFS)")
        self.plot_item.getAxis("bottom").setTicks(
            [[(math.log10(frequency), label) for frequency, label in self._TICKS]]
        )
        self.plot_item.setXRange(math.log10(20.0), math.log10(20_000.0), padding=0.0)
        self.plot_item.setYRange(-120.0, 0.0, padding=0.0)
        self.plot_item.vb.setLimits(
            xMin=math.log10(20.0),
            xMax=math.log10(20_000.0),
            yMin=-120.0,
            yMax=0.0,
        )
        self.plot_item.vb.enableAutoRange(enable=False)

    def set_observation(self, observation: AudioObservation | None) -> None:
        if observation is None or not observation.has_signal:
            self._set_empty()
            return
        frequencies = observation.spectrum_frequency_hz
        levels = observation.spectrum_level_dbfs
        positive = np.isfinite(frequencies) & (frequencies > 0.0)
        self.curve.setData(frequencies[positive], levels[positive])
        if (
            observation.peak_frequency_hz is None
            or observation.peak_level_dbfs is None
            or observation.peak_frequency_hz <= 0.0
        ):
            self.marker.setData(np.empty(0), np.empty(0))
            self.readout.setText(self._EMPTY_MESSAGE)
            return
        self.marker.setData([observation.peak_frequency_hz], [observation.peak_level_dbfs])
        self.readout.setText(
            f"Peak {observation.peak_frequency_hz:.1f} Hz · {observation.peak_level_dbfs:.1f} dBFS"
        )
