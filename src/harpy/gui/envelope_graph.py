"""Custom-painted, transformed-time ADSR envelope graph."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QAccessible,
    QAccessibleAnnouncementEvent,
    QAccessibleValueChangeEvent,
    QAccessibleValueInterface,
    QColor,
    QFocusEvent,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QResizeEvent,
)
from PySide6.QtWidgets import QAccessibleWidget, QApplication, QLabel, QWidget

from harpy.gui.envelope_entry import format_envelope_curvature
from harpy.gui.envelope_stage_control import EnvelopeStageControl, EnvelopeValueStage
from harpy.synth.curves import (
    curvature_from_control_level,
    quadratic_control_level,
    sample_envelope_preview,
)
from harpy.synth.models import EnvelopeConfig, RenderConfig, seconds_to_frames


class CurveStage(StrEnum):
    ATTACK = "attack"
    DECAY = "decay"
    RELEASE = "release"


@dataclass(frozen=True, slots=True)
class EnvelopeGraphGeometry:
    contents: QRectF
    stage_rects: tuple[tuple[str, QRectF], ...]
    stage_control_rects: tuple[tuple[str, QRectF], ...]
    handle_centers: tuple[tuple[CurveStage, QPointF], ...]


def envelope_stage_fractions(
    envelope: EnvelopeConfig,
    render: RenderConfig,
) -> dict[str, float]:
    """Project physical ADSR durations into readable display-width fractions."""

    weights = {
        stage: math.log1p(
            seconds_to_frames(getattr(envelope, f"{stage}_seconds"), render.sample_rate_hz)
        )
        for stage in ("attack", "decay", "release")
    }
    total_weight = sum(weights.values())
    attack = 0.14 + 0.42 * weights["attack"] / total_weight
    decay = 0.14 + 0.42 * weights["decay"] / total_weight
    sustain = 0.16
    release = 1.0 - attack - decay - sustain
    return {
        "attack": attack,
        "decay": decay,
        "sustain": sustain,
        "release": release,
    }


class EnvelopeGraph(QWidget):
    """Readable ADSR display whose curve proposals remain owner-controlled."""

    field_previewed = Signal(str, float)
    field_commit_requested = Signal(str, float)
    field_reverted = Signal(str)
    field_validation_failed = Signal(str, str)
    field_editing_changed = Signal(str, bool)

    def __init__(self, render: RenderConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if not isinstance(render, RenderConfig):
            raise ValueError("render must be a RenderConfig")
        self._render = render
        self._envelope = EnvelopeConfig()
        self._selected_stage = CurveStage.ATTACK
        self._geometry = self._calculate_geometry()
        self._handles: dict[CurveStage, _CurveHandle] = {}
        self._stage_controls: dict[EnvelopeValueStage, EnvelopeStageControl] = {}
        self._editing_control: EnvelopeStageControl | None = None
        self._accepted_exact_field: str | None = None
        self._suppress_field_revert = False
        self._curve_readout = QLabel(self)
        self._curve_readout.setObjectName("curveValueReadout")
        self._curve_readout.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._curve_readout.hide()
        for stage in EnvelopeValueStage:
            control = EnvelopeStageControl(
                stage,
                self._render,
                lambda stage=stage: getattr(self._envelope, stage.field_name),
                self,
            )
            control.setToolTip(f"Adjust {stage.value} value")
            control.value_previewed.connect(self.field_previewed)
            control.value_commit_requested.connect(self.field_commit_requested)
            control.value_reverted.connect(self.field_reverted)
            control.validation_failed.connect(self.field_validation_failed)
            control.editing_changed.connect(self._stage_editing_changed)
            self._stage_controls[stage] = control
        for stage in CurveStage:
            handle = _CurveHandle(stage, lambda stage=stage: self._curve_value(stage), self)
            handle.previewed.connect(
                lambda stage_name, value: self.field_previewed.emit(
                    self._curve_field(stage_name), value
                )
            )
            handle.commit_requested.connect(
                lambda stage_name, value: self.field_commit_requested.emit(
                    self._curve_field(stage_name), value
                )
            )
            handle.reverted.connect(
                lambda stage_name: self.field_reverted.emit(self._curve_field(stage_name))
            )
            handle.selected.connect(self._handle_selected)
            handle.context_changed.connect(self._update_curve_readout)
            self._handles[stage] = handle
        self.setMinimumSize(240, 150)
        self._layout_graph()

    def set_envelope(self, envelope: EnvelopeConfig) -> None:
        if not isinstance(envelope, EnvelopeConfig):
            raise ValueError("envelope must be an EnvelopeConfig")
        previous = self._envelope
        self._envelope = envelope
        self._layout_graph()
        self.update()
        for stage, control in self._stage_controls.items():
            old_value = getattr(previous, stage.field_name)
            new_value = getattr(envelope, stage.field_name)
            if old_value != new_value:
                control.owner_value_changed(old_value, new_value)
        for stage, handle in self._handles.items():
            handle.update()
            old_value = getattr(previous, f"{stage.value}_curve")
            new_value = getattr(envelope, f"{stage.value}_curve")
            if old_value != new_value:
                QAccessible.updateAccessibility(QAccessibleValueChangeEvent(handle, new_value))
            handle.context_changed.emit(stage.value)

    def select_stage(self, stage: CurveStage) -> None:
        selected = CurveStage(stage)
        if selected is self._selected_stage:
            return
        self._selected_stage = selected
        self.update()
        for handle in self._handles.values():
            handle.update()

    @property
    def selected_stage(self) -> CurveStage:
        return self._selected_stage

    @property
    def envelope(self) -> EnvelopeConfig:
        return self._envelope

    def preview_levels(self, stage: CurveStage) -> np.ndarray:
        preview = sample_envelope_preview(self._envelope, self._render)
        levels = {
            CurveStage.ATTACK: preview.attack_level,
            CurveStage.DECAY: preview.decay_level,
            CurveStage.RELEASE: preview.release_level,
        }[CurveStage(stage)]
        result = np.array(levels, dtype=np.float64, copy=True, order="C")
        result.flags.writeable = False
        return result

    def display_geometry(self) -> EnvelopeGraphGeometry:
        geometry = self._geometry
        return EnvelopeGraphGeometry(
            contents=QRectF(geometry.contents),
            stage_rects=tuple((name, QRectF(rect)) for name, rect in geometry.stage_rects),
            stage_control_rects=tuple(
                (name, QRectF(rect)) for name, rect in geometry.stage_control_rects
            ),
            handle_centers=tuple(
                (stage, QPointF(center)) for stage, center in geometry.handle_centers
            ),
        )

    def resizeEvent(self, event: QResizeEvent) -> None:
        self._layout_graph()
        super().resizeEvent(event)

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#10151c"))

        geometry = self._geometry
        painter.setPen(QPen(QColor("#2a3542"), 1.0))
        painter.setBrush(QColor("#141b24"))
        painter.drawRoundedRect(geometry.contents, 5.0, 5.0)

        stage_rects = dict(geometry.stage_rects)
        selected_rect = stage_rects[self._selected_stage.value]
        painter.fillRect(selected_rect, QColor(72, 53, 102, 70))

        boundary_pen = QPen(QColor("#344252"), 1.0, Qt.PenStyle.DashLine)
        painter.setPen(boundary_pen)
        for _, rect in geometry.stage_rects[:-1]:
            painter.drawLine(
                QPointF(rect.right(), rect.top()),
                QPointF(rect.right(), rect.bottom()),
            )

        preview = sample_envelope_preview(self._envelope, self._render)
        levels = {
            CurveStage.ATTACK: preview.attack_level,
            CurveStage.DECAY: preview.decay_level,
            CurveStage.RELEASE: preview.release_level,
        }
        positions = preview.position
        handle_centers = dict(geometry.handle_centers)
        guide_pen = QPen(QColor("#56677c"), 1.0, Qt.PenStyle.DashLine)
        curve_pen = QPen(
            QColor("#69dcff"),
            2.2,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        for stage in CurveStage:
            rect = stage_rects[stage.value]
            start, end = self._stage_levels(stage)
            p0 = QPointF(rect.left(), rect.bottom() - start * rect.height())
            p1 = handle_centers[stage]
            p2 = QPointF(rect.right(), rect.bottom() - end * rect.height())
            painter.setPen(guide_pen)
            painter.drawLine(p0, p1)
            painter.drawLine(p1, p2)

            path = QPainterPath()
            for index, (position, level) in enumerate(zip(positions, levels[stage], strict=True)):
                point = QPointF(
                    rect.left() + float(position) * rect.width(),
                    rect.bottom() - float(level) * rect.height(),
                )
                if index == 0:
                    path.moveTo(point)
                else:
                    path.lineTo(point)
            painter.setPen(curve_pen)
            painter.drawPath(path)

        sustain_rect = stage_rects["sustain"]
        sustain_y = sustain_rect.bottom() - preview.sustain_level * sustain_rect.height()
        painter.drawLine(
            QPointF(sustain_rect.left(), sustain_y),
            QPointF(sustain_rect.right(), sustain_y),
        )

    def _curve_value(self, stage: CurveStage) -> float:
        return getattr(self._envelope, f"{stage.value}_curve")

    def _stage_levels(self, stage: CurveStage) -> tuple[float, float]:
        if stage is CurveStage.ATTACK:
            return 0.0, 1.0
        if stage is CurveStage.DECAY:
            return 1.0, self._envelope.sustain_amplitude
        return self._envelope.sustain_amplitude, 0.0

    def _stage_rect(self, stage: CurveStage) -> QRectF:
        return dict(self._geometry.stage_rects)[stage.value]

    def _stage_mouse_adjustable(self, stage: CurveStage) -> bool:
        start, end = self._stage_levels(stage)
        return abs(end - start) * self._stage_rect(stage).height() >= 1.0

    def _curve_from_vertical_delta(
        self,
        stage: CurveStage,
        origin_curve: float,
        delta_y: float,
    ) -> float | None:
        if not self._stage_mouse_adjustable(stage):
            return None
        start, end = self._stage_levels(stage)
        stage_rect = self._stage_rect(stage)
        control = quadratic_control_level(start, end, origin_curve)
        control_y = stage_rect.bottom() - control * stage_rect.height()
        low_y, high_y = sorted(
            (
                stage_rect.bottom() - start * stage_rect.height(),
                stage_rect.bottom() - end * stage_rect.height(),
            )
        )
        dragged_y = min(max(control_y + delta_y, low_y), high_y)
        dragged_level = (stage_rect.bottom() - dragged_y) / stage_rect.height()
        return min(
            max(curvature_from_control_level(start, end, dragged_level), -1.0),
            1.0,
        )

    def _handle_selected(self, stage_name: str) -> None:
        stage = CurveStage(stage_name)
        self.select_stage(stage)

    def _update_curve_readout(self, _stage_name: str) -> None:
        stage = next(
            (stage for stage in CurveStage if self._handles[stage].is_contextual),
            None,
        )
        if stage is None:
            self._curve_readout.hide()
            return
        self._curve_readout.setText(f"Curve {format_envelope_curvature(self._curve_value(stage))}")
        self._curve_readout.adjustSize()
        handle = self._handles[stage]
        position = handle.geometry().bottomLeft()
        self._curve_readout.move(
            round(min(position.x(), self.width() - self._curve_readout.width())),
            round(min(position.y() + 4, self.height() - self._curve_readout.height())),
        )
        self._curve_readout.show()

    def accept_exact_edit(self, field_name: str, value: float) -> None:
        control = self._value_control(field_name)
        if control is None or control is not self._editing_control:
            return
        self._accepted_exact_field = field_name
        control.accept_exact_value(value)

    def reject_exact_edit(self, field_name: str, message: str) -> None:
        control = self._value_control(field_name)
        if control is None or control is not self._editing_control:
            return
        control.reject_exact_value(message)

    def is_exact_editing(self, field_name: str) -> bool:
        """Return whether *field_name* owns the currently open exact editor."""

        control = self._value_control(field_name)
        return control is not None and control is self._editing_control

    def mark_field_error(self, field_name: str, message: str) -> None:
        control = self._value_control(field_name)
        widget = control if control is not None else self._curve_handle(field_name)
        if widget is None:
            return
        widget.setProperty("validationState", "error")
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()
        QAccessible.updateAccessibility(QAccessibleAnnouncementEvent(widget, message))

    def clear_field_error(self, field_name: str) -> None:
        control = self._value_control(field_name)
        widget = control if control is not None else self._curve_handle(field_name)
        if widget is None:
            return
        widget.setProperty("validationState", None)
        widget.style().unpolish(widget)
        widget.style().polish(widget)
        widget.update()

    def cancel_interactions(self, emit_revert: bool = True) -> None:
        if self._editing_control is not None:
            self._suppress_field_revert = not emit_revert
            try:
                self._editing_control._close_exact_editor(return_focus=False)
            finally:
                self._suppress_field_revert = False
        for control in self._stage_controls.values():
            control.cancel_interaction(emit_revert=emit_revert)
        for handle in self._handles.values():
            handle.cancel_interaction(emit_revert=emit_revert)

    def _value_control(self, field_name: str) -> EnvelopeStageControl | None:
        for stage, control in self._stage_controls.items():
            if stage.field_name == field_name:
                return control
        return None

    def _curve_handle(self, field_name: str) -> _CurveHandle | None:
        for stage, handle in self._handles.items():
            if field_name == f"{stage.value}_curve":
                return handle
        return None

    def _curve_field(self, stage_name: str) -> str:
        return f"{CurveStage(stage_name).value}_curve"

    def _stage_editing_changed(self, field_name: str, is_editing: bool) -> None:
        control = self._value_control(field_name)
        if is_editing and control is not None:
            if self._editing_control is not None and self._editing_control is not control:
                self._suppress_field_revert = True
                try:
                    self._editing_control._close_exact_editor(return_focus=False)
                finally:
                    self._suppress_field_revert = False
            self._editing_control = control
        elif control is self._editing_control:
            self._editing_control = None
        if not is_editing:
            if self._accepted_exact_field == field_name:
                self._accepted_exact_field = None
            elif not self._suppress_field_revert:
                self.field_reverted.emit(field_name)
        self.field_editing_changed.emit(field_name, is_editing)

    def _layout_graph(self) -> None:
        self._geometry = self._calculate_geometry()
        centers = dict(self._geometry.handle_centers)
        control_rects = dict(self._geometry.stage_control_rects)
        for stage, control in self._stage_controls.items():
            rect = control_rects[stage.value]
            left = math.ceil(rect.left())
            top = math.ceil(rect.top())
            right = math.floor(rect.right())
            bottom = math.floor(rect.bottom())
            control.setGeometry(left, top, max(24, right - left), max(24, bottom - top))
        for stage, handle in self._handles.items():
            center = centers[stage]
            handle.resize(24, 24)
            handle.move(round(center.x()) - 12, round(center.y()) - 12)
            handle.raise_()
        for first, second in zip(
            (
                self._stage_controls[EnvelopeValueStage.ATTACK],
                self._handles[CurveStage.ATTACK],
                self._stage_controls[EnvelopeValueStage.DECAY],
                self._handles[CurveStage.DECAY],
                self._stage_controls[EnvelopeValueStage.SUSTAIN],
                self._stage_controls[EnvelopeValueStage.RELEASE],
                self._handles[CurveStage.RELEASE],
            ),
            (
                self._handles[CurveStage.ATTACK],
                self._stage_controls[EnvelopeValueStage.DECAY],
                self._handles[CurveStage.DECAY],
                self._stage_controls[EnvelopeValueStage.SUSTAIN],
                self._stage_controls[EnvelopeValueStage.RELEASE],
                self._handles[CurveStage.RELEASE],
            ),
            strict=False,
        ):
            QWidget.setTabOrder(first, second)

    def _calculate_geometry(self) -> EnvelopeGraphGeometry:
        contents = QRectF(
            12.0,
            10.0,
            max(1.0, self.width() - 24.0),
            max(1.0, self.height() - 20.0),
        )
        plot_top = contents.top() + 20.0
        plot_bottom = max(plot_top + 1.0, contents.bottom() - 32.0)
        plot_height = plot_bottom - plot_top
        fractions = envelope_stage_fractions(self._envelope, self._render)
        stage_rects: list[tuple[str, QRectF]] = []
        x = contents.left()
        for index, name in enumerate(("attack", "decay", "sustain", "release")):
            right = contents.right() if index == 3 else x + contents.width() * fractions[name]
            stage_rects.append((name, QRectF(x, plot_top, right - x, plot_height)))
            x = right
        stage_control_rects = tuple(
            (name, QRectF(rect.left(), plot_bottom + 4.0, rect.width(), 24.0))
            for name, rect in stage_rects
        )
        rect_by_name = dict(stage_rects)
        handle_centers = []
        for stage in CurveStage:
            rect = rect_by_name[stage.value]
            start, end = self._stage_levels(stage)
            control = quadratic_control_level(start, end, self._curve_value(stage))
            handle_centers.append(
                (
                    stage,
                    QPointF(rect.center().x(), rect.bottom() - control * rect.height()),
                )
            )
        return EnvelopeGraphGeometry(
            contents=contents,
            stage_rects=tuple(stage_rects),
            stage_control_rects=stage_control_rects,
            handle_centers=tuple(handle_centers),
        )


class _CurveHandle(QWidget):
    previewed = Signal(str, float)
    commit_requested = Signal(str, float)
    reverted = Signal(str)
    selected = Signal(str)
    context_changed = Signal(str)

    def __init__(
        self,
        stage: CurveStage,
        current_curve: Callable[[], float],
        parent: QWidget,
    ) -> None:
        super().__init__(parent)
        self._stage = stage
        self._current_curve = current_curve
        self._drag_origin_global_y: float | None = None
        self._drag_origin_global_x: float | None = None
        self._drag_origin_curve: float | None = None
        self._drag_had_preview = False
        self._dragging = False
        self._error_message: str | None = None
        self._selecting_from_press = False
        self.setObjectName(f"{stage.value}CurveHandle")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setAccessibleName(f"{stage.value.title()} curve")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    @property
    def current_curve(self) -> float:
        return self._current_curve()

    @property
    def is_contextual(self) -> bool:
        return self.underMouse() or self.hasFocus() or self._dragging

    def focusInEvent(self, event: QFocusEvent) -> None:
        if not self._selecting_from_press:
            self.selected.emit(self._stage.value)
        self.update()
        self.context_changed.emit(self._stage.value)
        super().focusInEvent(event)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        self.update()
        self.context_changed.emit(self._stage.value)
        super().focusOutEvent(event)

    def enterEvent(self, event: QEvent) -> None:
        self.context_changed.emit(self._stage.value)
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        if not self._dragging:
            self.context_changed.emit(self._stage.value)
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() is Qt.MouseButton.LeftButton:
            self._selecting_from_press = True
            try:
                self.setFocus(Qt.FocusReason.MouseFocusReason)
            finally:
                self._selecting_from_press = False
            self.selected.emit(self._stage.value)
            graph = self.parentWidget()
            if isinstance(graph, EnvelopeGraph) and graph._stage_mouse_adjustable(self._stage):
                self._drag_origin_global_y = event.globalPosition().y()
                self._drag_origin_global_x = event.globalPosition().x()
                self._drag_origin_curve = self.current_curve
                self._drag_had_preview = False
                self._dragging = False
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin_curve is not None and not self._dragging:
            if (
                math.hypot(
                    event.globalPosition().x() - self._drag_origin_global_x,
                    event.globalPosition().y() - self._drag_origin_global_y,
                )
                < QApplication.startDragDistance()
            ):
                event.accept()
                return
            self._dragging = True
            self.grabMouse()
            self.context_changed.emit(self._stage.value)
        value = self._drag_value(event.globalPosition().y())
        if value is not None and value != self.current_curve:
            self._drag_had_preview = True
            self.previewed.emit(self._stage.value, value)
        if self._drag_origin_curve is not None:
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() is Qt.MouseButton.LeftButton and self._drag_origin_curve is not None:
            value = self._drag_value(event.globalPosition().y())
            had_preview = self._drag_had_preview
            was_dragging = self._dragging
            self._drag_origin_global_y = None
            self._drag_origin_global_x = None
            self._drag_origin_curve = None
            self._drag_had_preview = False
            self._dragging = False
            if was_dragging:
                self.releaseMouse()
            if was_dragging and value is not None and had_preview:
                self.commit_requested.emit(self._stage.value, value)
            self.context_changed.emit(self._stage.value)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() is Qt.MouseButton.LeftButton:
            self.cancel_interaction(emit_revert=False)
            self._request_curve(0.0)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Right):
            step = 0.001 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 0.01
            self._request_curve(self.current_curve + step)
            event.accept()
            return
        if key in (Qt.Key.Key_Down, Qt.Key.Key_Left):
            step = 0.001 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 0.01
            self._request_curve(self.current_curve - step)
            event.accept()
            return
        if key == Qt.Key.Key_Home:
            self._request_curve(0.0)
            event.accept()
            return
        if key == Qt.Key.Key_Escape:
            self.cancel_interaction()
            event.accept()
            return
        super().keyPressEvent(event)

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.UngrabMouse and self._dragging:
            self.cancel_interaction()
            return True
        return super().event(event)

    def paintEvent(self, _event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = QPointF(self.rect().center())
        graph = self.parentWidget()
        selected = isinstance(graph, EnvelopeGraph) and graph.selected_stage is self._stage
        if self.hasFocus():
            painter.setPen(QPen(QColor("#f1f7ff"), 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(center, 8.0, 8.0)
        painter.setPen(QPen(QColor("#10151c"), 1.5))
        painter.setBrush(QColor("#bd8cff") if selected else QColor("#69dcff"))
        painter.drawEllipse(center, 5.5, 5.5)

    def mark_error(self, message: str) -> None:
        self._error_message = message
        self.setProperty("validationState", "error")
        self.update()

    def clear_error(self) -> None:
        self._error_message = None
        self.setProperty("validationState", None)
        self.update()

    def _request_curve(self, value: float) -> None:
        proposed = min(max(float(value), -1.0), 1.0)
        if proposed != self.current_curve:
            self.commit_requested.emit(self._stage.value, proposed)

    def cancel_interaction(self, emit_revert: bool = True) -> None:
        was_dragging = self._dragging
        had_preview = self._drag_had_preview
        self._drag_origin_global_y = None
        self._drag_origin_global_x = None
        self._drag_origin_curve = None
        self._drag_had_preview = False
        self._dragging = False
        if was_dragging:
            self.releaseMouse()
        if emit_revert and had_preview:
            self.reverted.emit(self._stage.value)
        self.context_changed.emit(self._stage.value)

    def _drag_value(self, pointer_global_y: float) -> float | None:
        if self._drag_origin_global_y is None or self._drag_origin_curve is None:
            return None
        graph = self.parentWidget()
        if not isinstance(graph, EnvelopeGraph):
            return None
        return graph._curve_from_vertical_delta(
            self._stage,
            self._drag_origin_curve,
            pointer_global_y - self._drag_origin_global_y,
        )


class _CurveHandleAccessible(QAccessibleWidget, QAccessibleValueInterface):
    """Accessibility bridge that queries the graph-owned immutable curvature."""

    def __init__(self, handle: _CurveHandle) -> None:
        QAccessibleWidget.__init__(self, handle, QAccessible.Role.Slider)

    def valueInterface(self) -> QAccessibleValueInterface:
        return self

    def currentValue(self) -> float:
        handle = self._handle()
        return handle.current_curve if handle is not None else 0.0

    def setCurrentValue(self, value: float) -> None:
        handle = self._handle()
        if handle is None:
            return
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return
        if not math.isfinite(numeric):
            return
        handle._request_curve(numeric)

    def minimumValue(self) -> float:
        return -1.0

    def maximumValue(self) -> float:
        return 1.0

    def minimumStepSize(self) -> float:
        return 0.001

    def actionNames(self) -> list[str]:
        return [self.increaseAction(), self.decreaseAction(), self.setFocusAction()]

    def doAction(self, action_name: str) -> None:
        handle = self._handle()
        if handle is None:
            return
        if action_name == self.increaseAction():
            handle._request_curve(handle.current_curve + 0.01)
        elif action_name == self.decreaseAction():
            handle._request_curve(handle.current_curve - 0.01)
        elif action_name == self.setFocusAction():
            handle.setFocus(Qt.FocusReason.OtherFocusReason)

    def _handle(self) -> _CurveHandle | None:
        widget = self.widget()
        return widget if isinstance(widget, _CurveHandle) else None


def _curve_handle_accessible_factory(
    _class_name: str,
    object_: object,
) -> _CurveHandleAccessible | None:
    if isinstance(object_, _CurveHandle):
        return _CurveHandleAccessible(object_)
    return None


QAccessible.installFactory(_curve_handle_accessible_factory)
