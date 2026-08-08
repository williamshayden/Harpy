"""A relative, logarithmic frequency control for the workbench."""

from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QAccessible,
    QAccessibleValueChangeEvent,
    QAccessibleValueInterface,
    QColor,
    QEnterEvent,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QRadialGradient,
    QWheelEvent,
)
from PySide6.QtWidgets import QAccessibleWidget, QWidget


def frequency_to_unit(frequency_hz: float, minimum_hz: float, maximum_hz: float) -> float:
    """Return a frequency's clamped position in the logarithmic control range."""

    if minimum_hz <= 0.0 or maximum_hz <= minimum_hz:
        raise ValueError("frequency bounds must be positive and ordered")
    unit = math.log2(frequency_hz / minimum_hz) / math.log2(maximum_hz / minimum_hz)
    return min(1.0, max(0.0, unit))


def unit_to_angle_degrees(unit: float) -> float:
    """Return the conventional non-wrapping dial angle for a unit position."""

    return 225.0 - 270.0 * min(1.0, max(0.0, unit))


class FrequencyKnob(QWidget):
    """A custom-painted non-wrapping control with relative vertical dragging."""

    frequency_changed = Signal(float)

    def __init__(
        self,
        minimum_hz: float,
        center_hz: float,
        maximum_hz: float,
        *,
        cents_per_pixel: float = 12.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not (
            math.isfinite(minimum_hz)
            and math.isfinite(center_hz)
            and math.isfinite(maximum_hz)
            and math.isfinite(cents_per_pixel)
            and minimum_hz > 0.0
            and minimum_hz <= center_hz <= maximum_hz
            and maximum_hz > minimum_hz
            and cents_per_pixel > 0.0
        ):
            raise ValueError("frequency bounds and drag sensitivity must be finite and ordered")
        self._minimum_hz = minimum_hz
        self._center_hz = center_hz
        self._maximum_hz = maximum_hz
        self._cents_per_pixel = cents_per_pixel
        self._frequency_hz = center_hz
        self._drag_origin_y: float | None = None
        self._drag_origin_hz: float | None = None
        self._hovered = False
        self._dragging = False
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Frequency")
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setToolTip(
            "Vertical drag to tune · Shift for fine mode · Wheel or arrow keys to adjust · "
            "Double-click to reset center"
        )
        self.setMinimumSize(72, 72)

    @property
    def frequency_hz(self) -> float:
        return self._frequency_hz

    def set_frequency_hz(self, frequency_hz: float, *, emit: bool = False) -> None:
        """Set the stored hertz value, constraining it to this runtime's range."""

        if not math.isfinite(frequency_hz):
            raise ValueError("frequency must be finite")
        value = min(self._maximum_hz, max(self._minimum_hz, frequency_hz))
        changed = value != self._frequency_hz
        self._frequency_hz = value
        if changed:
            self.update()
            QAccessible.updateAccessibility(QAccessibleValueChangeEvent(self, value))
        if emit:
            self.frequency_changed.emit(value)

    def _adjust_cents(self, cents: float) -> None:
        self.set_frequency_hz(self._frequency_hz * 2 ** (cents / 1200.0), emit=True)

    def enterEvent(self, event: QEnterEvent) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        if not self._dragging:
            self._hovered = False
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() is Qt.MouseButton.LeftButton:
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self._drag_origin_y = event.position().y()
            self._drag_origin_hz = self._frequency_hz
            self._hovered = True
            self._dragging = True
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._hovered:
            self._hovered = True
            self.update()
        if self._drag_origin_y is not None and self._drag_origin_hz is not None:
            shift_held = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            cents_per_pixel = self._cents_per_pixel * (0.1 if shift_held else 1.0)
            cents = (self._drag_origin_y - event.position().y()) * cents_per_pixel
            self.set_frequency_hz(self._drag_origin_hz * 2 ** (cents / 1200.0), emit=True)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() is Qt.MouseButton.LeftButton:
            self._drag_origin_y = None
            self._drag_origin_hz = None
            self._dragging = False
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() is Qt.MouseButton.LeftButton:
            self.set_frequency_hz(self._center_hz, emit=True)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        cents = 0.1 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Right):
            self._adjust_cents(cents)
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Left):
            self._adjust_cents(-cents)
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta:
            self._adjust_cents(1.0 if delta > 0 else -1.0)
            event.accept()
            return
        super().wheelEvent(event)

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#11151b"))

        side = float(min(self.width(), self.height()))
        center = QPointF(self.width() / 2.0, self.height() * 0.52)
        cap_radius = max(14.0, side * 0.235)
        track_radius = max(21.0, side * 0.34)
        track_rect = QRectF(
            center.x() - track_radius,
            center.y() - track_radius,
            track_radius * 2.0,
            track_radius * 2.0,
        )

        track_color = QColor("#2a3441")
        if self._hovered:
            track_color = QColor("#65d8ff")
        if self._dragging:
            track_color = QColor("#bd8cff")
        painter.setPen(QPen(track_color, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(track_rect, 225 * 16, -270 * 16)

        for unit in (0.0, 0.5, 1.0):
            angle = math.radians(unit_to_angle_degrees(unit))
            inner = QPointF(
                center.x() + (track_radius - 4.0) * math.cos(angle),
                center.y() - (track_radius - 4.0) * math.sin(angle),
            )
            outer = QPointF(
                center.x() + (track_radius + (5.0 if unit == 0.5 else 3.0)) * math.cos(angle),
                center.y() - (track_radius + (5.0 if unit == 0.5 else 3.0)) * math.sin(angle),
            )
            notch_color = QColor("#65d8ff") if unit == 0.5 else QColor("#9da9ba")
            notch_width = 2.5 if unit == 0.5 else 1.5
            painter.setPen(
                QPen(
                    notch_color,
                    notch_width,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                )
            )
            painter.drawLine(inner, outer)

        if self.hasFocus():
            halo_radius = cap_radius + 4.0
            halo_rect = QRectF(
                center.x() - halo_radius,
                center.y() - halo_radius,
                halo_radius * 2.0,
                halo_radius * 2.0,
            )
            painter.setPen(QPen(QColor("#65d8ff"), 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(halo_rect)

        cap_rect = QRectF(
            center.x() - cap_radius,
            center.y() - cap_radius,
            cap_radius * 2.0,
            cap_radius * 2.0,
        )
        face = QRadialGradient(
            QPointF(center.x() - cap_radius * 0.3, center.y() - cap_radius * 0.35),
            cap_radius * 1.35,
        )
        face.setColorAt(0.0, QColor("#2a3441"))
        face.setColorAt(0.6, QColor("#181e27"))
        face.setColorAt(1.0, QColor("#11151b"))
        painter.setPen(QPen(QColor("#2a3441"), 1.5))
        painter.setBrush(face)
        painter.drawEllipse(cap_rect)

        unit = frequency_to_unit(self._frequency_hz, self._minimum_hz, self._maximum_hz)
        angle = math.radians(unit_to_angle_degrees(unit))
        pointer = QPointF(
            center.x() + (cap_radius - 4.0) * math.cos(angle),
            center.y() - (cap_radius - 4.0) * math.sin(angle),
        )
        pointer_color = QColor("#bd8cff") if self._dragging else QColor("#65d8ff")
        painter.setPen(QPen(pointer_color, 3.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(center, pointer)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(pointer_color)
        painter.drawEllipse(center, 2.5, 2.5)

        label_font = painter.font()
        label_font.setPixelSize(max(8, int(side * 0.09)))
        label_font.setBold(True)
        painter.setFont(label_font)
        painter.setPen(QColor("#9da9ba"))
        label_height = max(10, int(side * 0.13))
        painter.drawText(
            QRectF(0.0, 0.0, self.width(), label_height),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "C3",
        )
        painter.drawText(
            QRectF(3.0, self.height() - label_height, self.width() * 0.35, label_height),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "C2",
        )
        painter.drawText(
            QRectF(
                self.width() * 0.65 - 3.0,
                self.height() - label_height,
                self.width() * 0.35,
                label_height,
            ),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            "C4",
        )


class _FrequencyKnobAccessible(QAccessibleWidget, QAccessibleValueInterface):
    """Qt accessibility bridge for the custom, float-valued frequency control."""

    def __init__(self, knob: FrequencyKnob) -> None:
        QAccessibleWidget.__init__(self, knob, QAccessible.Role.Dial)

    def valueInterface(self) -> QAccessibleValueInterface:
        return self

    def currentValue(self) -> float:
        knob = self._knob()
        return knob.frequency_hz if knob is not None else 0.0

    def setCurrentValue(self, value: float) -> None:
        knob = self._knob()
        if knob is None:
            return
        try:
            knob.set_frequency_hz(float(value), emit=True)
        except (TypeError, ValueError):
            return

    def minimumValue(self) -> float:
        knob = self._knob()
        return knob._minimum_hz if knob is not None else 0.0

    def maximumValue(self) -> float:
        knob = self._knob()
        return knob._maximum_hz if knob is not None else 0.0

    def minimumStepSize(self) -> float:
        knob = self._knob()
        if knob is None:
            return 0.0
        return knob._minimum_hz * (2 ** (1.0 / 1200.0) - 1.0)

    def actionNames(self) -> list[str]:
        return [self.increaseAction(), self.decreaseAction(), self.setFocusAction()]

    def doAction(self, action_name: str) -> None:
        knob = self._knob()
        if knob is None:
            return
        if action_name == self.increaseAction():
            knob._adjust_cents(1.0)
        elif action_name == self.decreaseAction():
            knob._adjust_cents(-1.0)
        elif action_name == self.setFocusAction():
            knob.setFocus(Qt.FocusReason.OtherFocusReason)

    def _knob(self) -> FrequencyKnob | None:
        widget = self.widget()
        return widget if isinstance(widget, FrequencyKnob) else None


def _frequency_knob_accessible_factory(
    _class_name: str, object_: object
) -> _FrequencyKnobAccessible | None:
    if isinstance(object_, FrequencyKnob):
        return _FrequencyKnobAccessible(object_)
    return None


QAccessible.installFactory(_frequency_knob_accessible_factory)
