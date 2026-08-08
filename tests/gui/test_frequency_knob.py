import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QAccessible, QColor, QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from harpy.gui.frequency_knob import FrequencyKnob, frequency_to_unit


def make_knob(qtbot) -> FrequencyKnob:
    knob = FrequencyKnob(100.0, 200.0, 400.0)
    knob.resize(120, 120)
    qtbot.addWidget(knob)
    knob.show()
    return knob


def test_frequency_to_unit_maps_runtime_minimum_center_and_maximum_in_log_space() -> None:
    # A linear mapping would put 200 Hz at one third instead of the midpoint.
    assert frequency_to_unit(100.0, 100.0, 400.0) == 0.0
    assert frequency_to_unit(200.0, 100.0, 400.0) == 0.5
    assert frequency_to_unit(400.0, 100.0, 400.0) == 1.0


def test_relative_vertical_drag_changes_from_current_value_without_absolute_jump(qtbot) -> None:
    # Replacing relative delta tracking with an absolute dial angle would jump on press.
    knob = make_knob(qtbot)
    knob.set_frequency_hz(200.0)

    QTest.mousePress(knob, Qt.MouseButton.LeftButton, pos=QPoint(8, 100))
    assert knob.frequency_hz == 200.0
    QTest.mouseMove(knob, QPoint(8, 88), delay=1)
    QTest.mouseRelease(knob, Qt.MouseButton.LeftButton, pos=QPoint(8, 88))

    assert knob.frequency_hz == pytest.approx(200.0 * 2 ** (144.0 / 1200.0))


def test_default_drag_traverses_two_octaves_in_200_upward_pixels(qtbot) -> None:
    # A wrong cents-per-pixel conversion would miss the runtime upper endpoint.
    knob = make_knob(qtbot)
    knob.set_frequency_hz(100.0)

    QTest.mousePress(knob, Qt.MouseButton.LeftButton, pos=QPoint(60, 210))
    QTest.mouseMove(knob, QPoint(60, 10), delay=1)
    QTest.mouseRelease(knob, Qt.MouseButton.LeftButton, pos=QPoint(60, 10))

    assert knob.frequency_hz == pytest.approx(400.0)


def test_shift_drag_uses_one_tenth_sensitivity(qtbot) -> None:
    # Ignoring Shift would make a 10-pixel drag move 120 cents rather than 12.
    knob = make_knob(qtbot)
    knob.set_frequency_hz(200.0)

    QTest.mousePress(
        knob,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ShiftModifier,
        QPoint(60, 80),
    )
    QApplication.sendEvent(
        knob,
        QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(60, 70),
            QPointF(60, 70),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ShiftModifier,
        ),
    )
    QTest.mouseRelease(
        knob,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ShiftModifier,
        QPoint(60, 70),
    )

    assert knob.frequency_hz == pytest.approx(200.0 * 2 ** (12.0 / 1200.0))


def test_drag_samples_shift_modifier_from_each_mouse_move(qtbot) -> None:
    knob = make_knob(qtbot)
    knob.set_frequency_hz(200.0)
    QTest.mousePress(
        knob,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ShiftModifier,
        QPoint(60, 80),
    )

    QApplication.sendEvent(
        knob,
        QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(60, 70),
            QPointF(60, 70),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ),
    )
    QTest.mouseRelease(knob, Qt.MouseButton.LeftButton, pos=QPoint(60, 70))

    assert knob.frequency_hz == pytest.approx(200.0 * 2 ** (120.0 / 1200.0))


def test_keyboard_and_wheel_apply_cent_steps_and_clamp_without_wrapping(qtbot) -> None:
    # Reversing a key or allowing modular wraparound would violate the bounded pitch control.
    knob = make_knob(qtbot)
    knob.setFocus()
    QTest.keyClick(knob, Qt.Key.Key_Up)
    assert knob.frequency_hz == pytest.approx(200.0 * 2 ** (1.0 / 1200.0))
    QTest.keyClick(knob, Qt.Key.Key_Left)
    assert knob.frequency_hz == pytest.approx(200.0)
    QTest.keyClick(knob, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
    assert knob.frequency_hz == pytest.approx(200.0 * 2 ** (0.1 / 1200.0))
    QTest.keyClick(knob, Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
    assert knob.frequency_hz == pytest.approx(200.0)

    knob.set_frequency_hz(400.0)
    QTest.keyClick(knob, Qt.Key.Key_Up)
    assert knob.frequency_hz == 400.0
    knob.set_frequency_hz(100.0)
    QTest.keyClick(knob, Qt.Key.Key_Down)
    assert knob.frequency_hz == 100.0

    QApplication.sendEvent(
        knob,
        QWheelEvent(
            QPointF(60, 60),
            QPointF(60, 60),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        ),
    )
    assert knob.frequency_hz == pytest.approx(100.0 * 2 ** (1.0 / 1200.0))


def test_double_click_restores_runtime_center_and_programmatic_updates_do_not_emit(qtbot) -> None:
    # Emitting from programmatic updates or restoring a fixed A440 C3 would desynchronize state.
    knob = make_knob(qtbot)
    emitted: list[float] = []
    knob.frequency_changed.connect(emitted.append)

    knob.set_frequency_hz(250.0)
    assert emitted == []
    knob.set_frequency_hz(250.0, emit=True)
    assert emitted == [250.0]

    QTest.mouseDClick(knob, Qt.MouseButton.LeftButton, pos=QPoint(60, 60))
    assert knob.frequency_hz == 200.0
    assert emitted[-1] == 200.0


def test_knob_has_focus_and_accessible_name(qtbot) -> None:
    # Losing keyboard focus support would make the keyboard interaction unreachable.
    knob = make_knob(qtbot)
    assert knob.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert knob.accessibleName() == "Frequency"


def test_keyboard_focus_renders_a_high_contrast_ring_without_changing_value(qtbot) -> None:
    # Omitting a custom focus cue makes this keyboard-operated painted widget look idle.
    knob = make_knob(qtbot)
    focus_sink = QPushButton("Focus sink")
    qtbot.addWidget(focus_sink)
    focus_sink.show()
    focus_sink.activateWindow()
    focus_sink.setFocus()
    QApplication.processEvents()
    assert not knob.hasFocus()
    unfocused = knob.grab().toImage().copy()
    before = knob.frequency_hz

    knob.activateWindow()
    knob.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    assert knob.hasFocus()
    focused = knob.grab().toImage().copy()

    focus_color = QColor("#65d8ff")
    focused_pixels = sum(
        focused.pixelColor(x, y) == focus_color
        for y in range(focused.height())
        for x in range(focused.width())
    )
    unfocused_pixels = sum(
        unfocused.pixelColor(x, y) == focus_color
        for y in range(unfocused.height())
        for x in range(unfocused.width())
    )
    differing_pixels = sum(
        focused.pixel(x, y) != unfocused.pixel(x, y)
        for y in range(focused.height())
        for x in range(focused.width())
    )
    assert differing_pixels >= 100
    assert focused_pixels >= 40
    assert unfocused_pixels == 0
    assert knob.frequency_hz == before
    assert knob.size().width() == 120
    assert knob.size().height() == 120


def test_knob_exposes_real_adjustable_accessibility_value_and_actions(qtbot) -> None:
    # A generic Client interface has no value semantics for assistive technology to adjust.
    knob = make_knob(qtbot)
    interface = QAccessible.queryAccessibleInterface(knob)

    assert interface.role() == QAccessible.Role.Dial
    value = interface.valueInterface()
    assert value is not None
    assert value.currentValue() == 200.0
    assert value.minimumValue() == 100.0
    assert value.maximumValue() == 400.0

    actions = interface.actionInterface()
    assert actions is not None
    actions.doAction(actions.increaseAction())
    assert knob.frequency_hz == pytest.approx(200.0 * 2 ** (1.0 / 1200.0))
    actions.doAction(actions.decreaseAction())
    assert knob.frequency_hz == pytest.approx(200.0)
