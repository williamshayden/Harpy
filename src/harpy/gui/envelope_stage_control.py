"""Graph-native direct controls for immutable envelope stage values."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from enum import StrEnum

from PySide6.QtCore import QEvent, QPointF, QSize, Qt, Signal
from PySide6.QtGui import (
    QAccessible,
    QAccessibleValueChangeEvent,
    QAccessibleValueInterface,
    QColor,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
)
from PySide6.QtWidgets import QAccessibleWidget, QApplication, QWidget

from harpy.gui.envelope_entry import (
    EnvelopeFieldKind,
    EnvelopeValueEntry,
    format_envelope_decibels,
    format_envelope_duration,
)
from harpy.synth.models import RenderConfig


class EnvelopeValueStage(StrEnum):
    """The four model-owned, directly editable envelope values."""

    ATTACK = "attack"
    DECAY = "decay"
    SUSTAIN = "sustain"
    RELEASE = "release"

    @property
    def field_name(self) -> str:
        return _FIELD_NAMES[self]

    @property
    def short_label(self) -> str:
        return _SHORT_LABELS[self]


_FIELD_NAMES = {
    EnvelopeValueStage.ATTACK: "attack_seconds",
    EnvelopeValueStage.DECAY: "decay_seconds",
    EnvelopeValueStage.SUSTAIN: "sustain_db",
    EnvelopeValueStage.RELEASE: "release_seconds",
}
_SHORT_LABELS = {
    EnvelopeValueStage.ATTACK: "A",
    EnvelopeValueStage.DECAY: "D",
    EnvelopeValueStage.SUSTAIN: "S",
    EnvelopeValueStage.RELEASE: "R",
}

_ACCESSIBLE_NAMES = {
    EnvelopeValueStage.ATTACK: "Attack duration",
    EnvelopeValueStage.DECAY: "Decay duration",
    EnvelopeValueStage.SUSTAIN: "Sustain level",
    EnvelopeValueStage.RELEASE: "Release duration",
}


class EnvelopeStageControl(QWidget):
    """Paint and propose one graph-owned envelope stage value."""

    value_previewed = Signal(str, float)
    value_commit_requested = Signal(str, float)
    value_reverted = Signal(str)
    validation_failed = Signal(str, str)
    editing_changed = Signal(str, bool)

    def __init__(
        self,
        stage: EnvelopeValueStage,
        render: RenderConfig,
        current_value: Callable[[], float],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(stage, EnvelopeValueStage):
            raise TypeError("stage must be an EnvelopeValueStage")
        if not isinstance(render, RenderConfig):
            raise TypeError("render must be a RenderConfig")
        if not callable(current_value):
            raise TypeError("current_value must be callable")
        self._stage = stage
        self._render = render
        self._current_value = current_value
        self._hovered = False
        self._press_global: QPointF | None = None
        self._last_global_y: float | None = None
        self._origin_value: float | None = None
        self._accumulated_units: float | None = None
        self._dragging = False
        self._had_preview = False
        self._preview_was_rejected = False
        self._exact_entry: EnvelopeValueEntry | None = None
        self._closing_exact_editor = False
        self.setObjectName(f"{stage.value}ValueControl")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setMouseTracking(True)
        self.setMinimumSize(24, 24)
        self.setAccessibleName(_ACCESSIBLE_NAMES[stage])
        self.setAccessibleDescription(self._editor_description())

    @property
    def current_value(self) -> float:
        value = self._current_value()
        if isinstance(value, bool):
            raise ValueError("owner value must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("owner value must be a finite number")
        return numeric

    @property
    def display_text(self) -> str:
        value = self.current_value
        text = (
            format_envelope_decibels(value)
            if self._stage is EnvelopeValueStage.SUSTAIN
            else format_envelope_duration(value)
        )
        return f"{self._stage.short_label} {text}"

    @property
    def is_interacting(self) -> bool:
        return self._press_global is not None

    def minimumSizeHint(self) -> QSize:
        return QSize(24, 24)

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        if self._dragging:
            background = QColor("#34304a")
        elif self.hasFocus() or self._hovered:
            background = QColor("#202a36")
        else:
            background = QColor("#171c24")
        painter.fillRect(self.rect(), background)
        painter.setPen(QColor("#d9e5f2"))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.display_text)

    def enterEvent(self, event: QEvent) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        if not self._dragging:
            self._hovered = False
            self.update()
        super().leaveEvent(event)

    def owner_value_changed(self, previous: float, current: float) -> None:
        """Repaint after the immutable owner has accepted a proposal."""

        if self.current_value != current:
            raise ValueError("owner must update current_value before notifying the control")
        self.update()
        if previous != current:
            QAccessible.updateAccessibility(QAccessibleValueChangeEvent(self, current))

    def owner_preview_accepted(self) -> None:
        """Allow the active scrub to commit after synchronous owner validation."""

        if self.is_interacting:
            self._preview_was_rejected = False

    def owner_preview_rejected(self) -> None:
        """Suppress release commit after synchronous owner rejection."""

        if self.is_interacting:
            self._preview_was_rejected = True

    def open_exact_editor(self) -> None:
        """Open a temporary exact-value editor over this graph-native control."""

        if self._exact_entry is not None:
            self._exact_entry.setFocus(Qt.FocusReason.OtherFocusReason)
            return
        parent = self.parentWidget() or self
        kind = (
            EnvelopeFieldKind.DECIBELS
            if self._stage is EnvelopeValueStage.SUSTAIN
            else EnvelopeFieldKind.DURATION
        )
        entry = EnvelopeValueEntry(self._stage.value.title(), kind, parent)
        entry.setObjectName("envelopeInlineEditor")
        entry.set_editor_accessibility(_ACCESSIBLE_NAMES[self._stage], self._editor_description())
        entry.set_exact_value(self.current_value)
        entry.value_commit_requested.connect(self._request_exact_commit)
        entry.validation_failed.connect(self._entry_validation_failed)
        entry.draft_reverted.connect(self._cancel_exact_editor)
        entry.focus_cancel_requested.connect(self._focus_cancel_exact_editor)
        position = self.mapTo(parent, self.rect().topLeft())
        width = max(96, self.width())
        width = min(width, parent.width())
        height = max(24, self.height())
        x = min(max(0, position.x()), max(0, parent.width() - width))
        y = min(max(0, position.y()), max(0, parent.height() - height))
        entry.setGeometry(x, y, width, height)
        self._exact_entry = entry
        self.editing_changed.emit(self._stage.field_name, True)
        entry.show()
        entry.setFocus(Qt.FocusReason.OtherFocusReason)
        entry.selectAll()

    def accept_exact_value(self, value: float) -> None:
        """Close the exact editor after its immutable owner accepts *value*."""

        if self.current_value != value:
            raise ValueError("owner must accept the exact value before notifying the control")
        self._close_exact_editor(return_focus=True)
        self.update()

    def reject_exact_value(self, message: str) -> None:
        """Keep the exact draft open and expose an owner rejection."""

        if self._exact_entry is not None:
            self._exact_entry.mark_commit_rejected(message)
        else:
            self.mark_error(message)

    def mark_error(self, message: str) -> None:
        """Expose a model validation error without changing owner truth."""

        self.validation_failed.emit(self._stage.field_name, message)

    def clear_error(self) -> None:
        if self._exact_entry is not None:
            self._exact_entry.clear_error()

    def _request_exact_commit(self, value: float) -> None:
        self.value_commit_requested.emit(self._stage.field_name, value)

    def _entry_validation_failed(self, message: str) -> None:
        self.validation_failed.emit(self._stage.field_name, message)

    def _cancel_exact_editor(self) -> None:
        self._close_exact_editor(return_focus=True)

    def _focus_cancel_exact_editor(self) -> None:
        if not self._closing_exact_editor:
            self._close_exact_editor(return_focus=False)

    def _close_exact_editor(self, *, return_focus: bool) -> None:
        entry = self._exact_entry
        if entry is None:
            return
        self._closing_exact_editor = True
        self._exact_entry = None
        entry.hide()
        entry.setParent(None)
        entry.deleteLater()
        if return_focus:
            self.setFocus(Qt.FocusReason.OtherFocusReason)
        self.editing_changed.emit(self._stage.field_name, False)
        self._closing_exact_editor = False

    def _editor_description(self) -> str:
        unit = (
            "decibels" if self._stage is EnvelopeValueStage.SUSTAIN else "milliseconds or seconds"
        )
        return f"Enter {unit}. Vertical drag adjusts; Enter or F2 edits exactly."

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() is Qt.MouseButton.LeftButton:
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            origin = self.current_value
            self._press_global = event.globalPosition()
            self._last_global_y = event.globalPosition().y()
            self._origin_value = origin
            self._accumulated_units = 0.0
            self._dragging = False
            self._had_preview = False
            self._preview_was_rejected = False
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            self._press_global is None
            or self._last_global_y is None
            or self._origin_value is None
            or self._accumulated_units is None
        ):
            super().mouseMoveEvent(event)
            return
        started_drag = False
        global_position = event.globalPosition()
        upward_pixels = self._last_global_y - global_position.y()
        scale = 0.1 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
        self._last_global_y = global_position.y()
        if self._stage is EnvelopeValueStage.SUSTAIN:
            self._accumulated_units += upward_pixels * 0.1 * scale
            value = min(self._origin_value + self._accumulated_units, 0.0)
        else:
            self._accumulated_units += upward_pixels / 100.0 * scale
            value = max(
                _scaled_duration(self._origin_value, self._accumulated_units),
                1.0 / self._render.sample_rate_hz,
            )
        if not self._dragging:
            distance = math.hypot(
                global_position.x() - self._press_global.x(),
                global_position.y() - self._press_global.y(),
            )
            if distance < QApplication.startDragDistance():
                event.accept()
                return
            self._dragging = True
            started_drag = True
        if value != self.current_value:
            self._had_preview = True
            self.value_previewed.emit(self._stage.field_name, value)
            self.update()
        if started_drag:
            self.grabMouse()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() is Qt.MouseButton.LeftButton and self._press_global is not None:
            dragging = self._dragging
            had_preview = self._had_preview
            preview_was_rejected = self._preview_was_rejected
            value = self.current_value
            self._clear_interaction()
            if dragging:
                self.releaseMouse()
            if had_preview and not preview_was_rejected:
                self.value_commit_requested.emit(self._stage.field_name, value)
            self.update()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape and self.is_interacting:
            self.cancel_interaction()
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_F2):
            self.open_exact_editor()
            event.accept()
            return
        increase = event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Right)
        decrease = event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Left)
        if increase or decrease:
            self._request_key_adjustment(increase, event.modifiers())
            event.accept()
            return
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.cancel_interaction(emit_revert=False)
            self.open_exact_editor()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.UngrabMouse and self._dragging:
            self.cancel_interaction()
            return True
        return super().event(event)

    def cancel_interaction(self, emit_revert: bool = True, return_focus: bool = False) -> None:
        """End an armed drag, reverting the owner preview if one was sent."""

        was_dragging = self._dragging
        had_preview = self._had_preview
        self._clear_interaction()
        if was_dragging:
            self.releaseMouse()
        if emit_revert and had_preview:
            self.value_reverted.emit(self._stage.field_name)
        if return_focus:
            self.setFocus(Qt.FocusReason.OtherFocusReason)
        self.update()

    def _clear_interaction(self) -> None:
        self._press_global = None
        self._last_global_y = None
        self._origin_value = None
        self._accumulated_units = None
        self._dragging = False
        self._had_preview = False
        self._preview_was_rejected = False

    def _request_key_adjustment(self, increase: bool, modifiers: Qt.KeyboardModifiers) -> None:
        value = self.current_value
        fine = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        if self._stage is EnvelopeValueStage.SUSTAIN:
            step = 0.01 if fine else 0.1
            proposed = min(value + (step if increase else -step), 0.0)
        else:
            factor = 1.001 if fine else 1.01
            proposed = value * factor if increase else value / factor
            proposed = max(proposed, 1.0 / self._render.sample_rate_hz)
        if proposed != value:
            self.value_commit_requested.emit(self._stage.field_name, proposed)


def _scaled_duration(origin: float, units: float) -> float:
    try:
        value = origin * math.exp2(units)
    except OverflowError:
        return sys.float_info.max
    if math.isinf(value):
        return sys.float_info.max
    return 0.0 if value == 0.0 else value


class _EnvelopeStageControlAccessible(QAccessibleWidget, QAccessibleValueInterface):
    """Accessibility bridge that always delegates values back to the graph owner."""

    def __init__(self, control: EnvelopeStageControl) -> None:
        QAccessibleWidget.__init__(self, control, QAccessible.Role.SpinBox)

    def valueInterface(self) -> QAccessibleValueInterface:
        return self

    def text(self, text_type: QAccessible.Text) -> str:
        control = self._control()
        if control is None:
            return ""
        if text_type == QAccessible.Text.Name:
            return _ACCESSIBLE_NAMES[control._stage]
        if text_type == QAccessible.Text.Value:
            value = control.current_value
            return (
                format_envelope_decibels(value)
                if control._stage is EnvelopeValueStage.SUSTAIN
                else format_envelope_duration(value)
            )
        if text_type == QAccessible.Text.Description:
            return control._editor_description()
        return super().text(text_type)

    def currentValue(self) -> float:
        control = self._control()
        return control.current_value if control is not None else 0.0

    def setCurrentValue(self, value: float) -> None:
        control = self._control()
        if control is None:
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return
        if not math.isfinite(numeric):
            return
        if control._stage is EnvelopeValueStage.SUSTAIN:
            proposed = min(numeric, 0.0)
        else:
            proposed = max(numeric, 1.0 / control._render.sample_rate_hz)
        if proposed != control.current_value:
            control.value_commit_requested.emit(control._stage.field_name, proposed)

    def minimumValue(self) -> float:
        control = self._control()
        if control is None or control._stage is EnvelopeValueStage.SUSTAIN:
            return -math.inf
        return 1.0 / control._render.sample_rate_hz

    def maximumValue(self) -> float:
        control = self._control()
        if control is None:
            return 0.0
        return 0.0 if control._stage is EnvelopeValueStage.SUSTAIN else math.inf

    def minimumStepSize(self) -> float:
        control = self._control()
        if control is None:
            return 0.0
        if control._stage is EnvelopeValueStage.SUSTAIN:
            return 0.01
        return control.current_value * 0.001

    def actionNames(self) -> list[str]:
        return [self.increaseAction(), self.decreaseAction(), self.setFocusAction()]

    def doAction(self, action_name: str) -> None:
        control = self._control()
        if control is None:
            return
        if action_name == self.increaseAction():
            control._request_key_adjustment(True, Qt.KeyboardModifier.NoModifier)
        elif action_name == self.decreaseAction():
            control._request_key_adjustment(False, Qt.KeyboardModifier.NoModifier)
        elif action_name == self.setFocusAction():
            control.setFocus(Qt.FocusReason.OtherFocusReason)

    def _control(self) -> EnvelopeStageControl | None:
        widget = self.widget()
        return widget if isinstance(widget, EnvelopeStageControl) else None


def _envelope_stage_control_accessible_factory(
    _class_name: str,
    object_: object,
) -> _EnvelopeStageControlAccessible | None:
    if isinstance(object_, EnvelopeStageControl):
        return _EnvelopeStageControlAccessible(object_)
    return None


QAccessible.installFactory(_envelope_stage_control_accessible_factory)
