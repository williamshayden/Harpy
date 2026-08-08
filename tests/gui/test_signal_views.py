import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QContextMenuEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMenu

from harpy.analysis import AudioObservation
from harpy.gui.signal_views import SpectrumView, WaveformView


def observation() -> AudioObservation:
    return AudioObservation(
        has_signal=True,
        waveform_samples=np.array([0.0, 0.251, -0.1]),
        waveform_time_ms=np.array([0.0, 10.0, 20.0]),
        spectrum_frequency_hz=np.array([0.0, 20.0, 440.0, 20_000.0]),
        spectrum_level_dbfs=np.array([-120.0, -50.0, -12.3, -70.0]),
        peak_amplitude_fs=0.251,
        peak_frequency_hz=439.8,
        peak_level_dbfs=-12.3,
    )


def test_waveform_view_has_fixed_scientific_axes_and_observed_peak(qtbot) -> None:
    # Autoscaling or deriving peak from a configured tone would misrepresent the observation.
    view = WaveformView()
    qtbot.addWidget(view)
    view.set_observation(observation())

    assert view.plot_item.vb.viewRange()[0] == pytest.approx([0.0, 50.0])
    assert view.plot_item.vb.viewRange()[1] == pytest.approx([-1.0, 1.0])
    assert view.plot_item.getAxis("bottom").labelText == "Time (ms)"
    assert view.plot_item.getAxis("left").labelText == "Amplitude (FS)"
    assert view.readout.text() == "Peak 0.251 FS"
    assert view.curve.xData.tolist() == [0.0, 10.0, 20.0]
    np.testing.assert_array_equal(
        view.curve.yData,
        np.array([0.0, 0.251, -0.1], dtype=np.float32),
    )


def test_spectrum_view_uses_log_coordinates_fixed_limits_and_emphasized_ticks(qtbot) -> None:
    # Passing Hz directly to a log ViewBox would create a huge, incorrect display range.
    view = SpectrumView()
    qtbot.addWidget(view)

    assert view.plot_item.vb.viewRange()[0] == pytest.approx([np.log10(20.0), np.log10(20_000.0)])
    assert view.plot_item.vb.viewRange()[1] == pytest.approx([-120.0, 0.0])
    assert view.plot_item.getAxis("bottom").labelText == "Frequency (Hz)"
    assert view.plot_item.getAxis("left").labelText == "Level (dBFS)"
    ticks = view.plot_item.getAxis("bottom")._tickLevels[0]
    assert [label for _, label in ticks] == [
        "20",
        "50",
        "100",
        "200",
        "500",
        "1k",
        "2k",
        "5k",
        "10k",
        "20k",
    ]
    assert [position for position, _ in ticks] == pytest.approx(
        [np.log10(value) for value in (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)]
    )
    assert view.plot_item.getAxis("bottom").grid >= 64
    assert view.plot_item.getAxis("left").grid >= 64


def test_real_plot_navigation_and_context_events_leave_ranges_locked_and_no_menu(qtbot) -> None:
    # An interactive ViewBox would permit changing the approved fixed scientific scale.
    for view in (WaveformView(), SpectrumView()):
        qtbot.addWidget(view)
        view.resize(600, 400)
        view.show()
        qtbot.waitExposed(view)
        viewport = view.plot.viewport()
        center = viewport.rect().center()
        before = np.asarray(view.plot_item.vb.viewRange(), dtype=np.float64)

        QApplication.sendEvent(
            viewport,
            QWheelEvent(
                QPointF(center),
                QPointF(viewport.mapToGlobal(center)),
                QPoint(0, 0),
                QPoint(0, 120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.ScrollUpdate,
                False,
            ),
        )
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=center)
        QTest.mouseMove(viewport, center + QPoint(50, 30), delay=1)
        QTest.mouseRelease(
            viewport,
            Qt.MouseButton.LeftButton,
            pos=center + QPoint(50, 30),
        )
        QApplication.sendEvent(
            viewport,
            QContextMenuEvent(
                QContextMenuEvent.Reason.Mouse,
                center,
                viewport.mapToGlobal(center),
            ),
        )
        QApplication.processEvents()

        np.testing.assert_allclose(view.plot_item.vb.viewRange(), before)
        assert not any(
            menu.isVisible() for menu in QApplication.topLevelWidgets() if isinstance(menu, QMenu)
        )
        assert view.plot_item.menuEnabled() is False
        assert view.plot_item.vb.menuEnabled() is False
        assert view.plot_item.vb.state["mouseEnabled"] == [False, False]
        assert view.plot_item.vb.wheel_zoom_disabled


def test_none_or_empty_observation_clears_without_fabricated_trace(qtbot) -> None:
    # Retaining prior samples or inserting a zero/floor trace would falsely imply a signal.
    empty = AudioObservation(
        has_signal=False,
        waveform_samples=np.empty(0),
        waveform_time_ms=np.empty(0),
        spectrum_frequency_hz=np.empty(0),
        spectrum_level_dbfs=np.empty(0),
        peak_amplitude_fs=None,
        peak_frequency_hz=None,
        peak_level_dbfs=None,
    )
    for view in (WaveformView(), SpectrumView()):
        qtbot.addWidget(view)
        view.set_observation(observation())
        view.set_observation(None)
        assert view.curve.xData is None
        assert view.curve.yData is None
        assert view.marker.xData is None
        assert view.readout.text() == "Hold Play to inspect the signal."
        view.set_observation(empty)
        assert view.curve.xData is None
        assert view.curve.yData is None


def test_spectrum_uses_positive_observed_bins_and_measured_peak_marker(qtbot) -> None:
    # Including zero in the log curve or placing the marker at a selected pitch is wrong.
    view = SpectrumView()
    qtbot.addWidget(view)
    view.set_observation(observation())

    assert view.curve.xData.tolist() == [20.0, 440.0, 20_000.0]
    assert view.marker.xData.tolist() == [439.8]
    assert view.marker.yData.tolist() == [-12.3]
    assert view.readout.text() == "Peak 439.8 Hz · -12.3 dBFS"


def test_signal_without_in_range_peak_retains_curve_and_reports_truthfully(qtbot) -> None:
    view = SpectrumView()
    qtbot.addWidget(view)
    no_peak = AudioObservation(
        has_signal=True,
        waveform_samples=np.array([0.2], dtype=np.float32),
        waveform_time_ms=np.array([0.0]),
        spectrum_frequency_hz=np.array([20.0, 440.0, 20_000.0]),
        spectrum_level_dbfs=np.array([-120.0, -120.0, -120.0]),
        peak_amplitude_fs=0.2,
        peak_frequency_hz=None,
        peak_level_dbfs=None,
    )

    view.set_observation(no_peak)

    assert view.curve.xData.tolist() == [20.0, 440.0, 20_000.0]
    assert view.marker.xData is None
    assert view.readout.text() == "No in-range spectral peak"


def test_spectrum_peak_marker_is_rendered_at_its_log_frequency_position(qtbot) -> None:
    # A raw-Hz ScatterPlotItem on the log ViewBox renders outside the fixed spectrum range.
    view = SpectrumView()
    qtbot.addWidget(view)
    view.set_observation(observation())

    rendered_x, rendered_y = view.marker.getData()
    assert rendered_x.tolist() == pytest.approx([np.log10(439.8)])
    assert rendered_y.tolist() == pytest.approx([-12.3])
