"""A relative, logarithmic frequency control for the workbench."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPaintEvent, QPen, QWheelEvent
from PySide6.QtWidgets import QWidget


def frequency_to_unit(frequency_hz: float, minimum_hz: float, maximum_hz: float) -> float:
    """Return a frequency's clamped position in the logarithmic control range."""

    if minimum_hz <= 0.0 or maximum_hz <= minimum_hz:
        raise ValueError("frequency bounds must be positive and ordered")
    unit = math.log2(frequency_hz / minimum_hz) / math.log2(maximum_hz / minimum_hz)
    return min(1.0, max(0.0, unit))


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
        self._drag_shift = False
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Frequency")
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
        if emit:
            self.frequency_changed.emit(value)

    def _adjust_cents(self, cents: float) -> None:
        self.set_frequency_hz(self._frequency_hz * 2 ** (cents / 1200.0), emit=True)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() is Qt.MouseButton.LeftButton:
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self._drag_origin_y = event.position().y()
            self._drag_origin_hz = self._frequency_hz
            self._drag_shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin_y is not None and self._drag_origin_hz is not None:
            cents_per_pixel = self._cents_per_pixel * (0.1 if self._drag_shift else 1.0)
            cents = (self._drag_origin_y - event.position().y()) * cents_per_pixel
            self.set_frequency_hz(self._drag_origin_hz * 2 ** (cents / 1200.0), emit=True)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() is Qt.MouseButton.LeftButton:
            self._drag_origin_y = None
            self._drag_origin_hz = None
            self._drag_shift = False
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
        rect = self.rect().adjusted(12, 12, -12, -12)
        painter.setPen(QPen(QColor("#3a4352"), 5.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, 225 * 16, 270 * 16)

        center = QPointF(rect.center())
        radius = min(rect.width(), rect.height()) / 2.0
        for unit in (0.0, 0.5, 1.0):
            angle = math.radians(225.0 + 270.0 * unit)
            inner = QPointF(
                center.x() + (radius - 8.0) * math.cos(angle),
                center.y() - (radius - 8.0) * math.sin(angle),
            )
            outer = QPointF(
                center.x() + (radius + 2.0) * math.cos(angle),
                center.y() - (radius + 2.0) * math.sin(angle),
            )
            painter.setPen(
                QPen(QColor("#9da9ba"), 1.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            )
            painter.drawLine(inner, outer)

        unit = frequency_to_unit(self._frequency_hz, self._minimum_hz, self._maximum_hz)
        angle = math.radians(225.0 + 270.0 * unit)
        indicator = QPointF(
            center.x() + (radius - 7.0) * math.cos(angle),
            center.y() - (radius - 7.0) * math.sin(angle),
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#68d6ff"))
        painter.drawEllipse(indicator, 5.0, 5.0)
