"""Exact two-phase text entry for authored envelope values."""

from __future__ import annotations

import math
import re
from enum import StrEnum

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAccessible, QAccessibleAnnouncementEvent, QFocusEvent, QKeyEvent
from PySide6.QtWidgets import QLineEdit, QWidget

_PLAIN_DECIMAL = r"[+-]?(?:\d+(?:\.\d{0,9})?|\.\d{1,9})"
_DURATION = re.compile(rf"(?P<number>{_PLAIN_DECIMAL})(?:\s*(?P<unit>ms|s))?\Z")
_DECIBELS = re.compile(rf"(?P<number>{_PLAIN_DECIMAL})(?:\s*dB)?\Z")
_CURVATURE = re.compile(rf"(?P<number>{_PLAIN_DECIMAL})\Z")


class EnvelopeFieldKind(StrEnum):
    """Closed parsing and display modes for envelope value fields."""

    DURATION = "duration"
    DECIBELS = "decibels"
    CURVATURE = "curvature"


def _plain_decimal(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def format_envelope_duration(seconds: float) -> str:
    """Format an envelope duration with the canonical millisecond/second switch."""

    if seconds < 1.0:
        return f"{_plain_decimal(seconds * 1_000.0)} ms"
    return f"{_plain_decimal(seconds)} s"


def format_envelope_decibels(value: float) -> str:
    """Format a decibel value with canonical signed-zero handling."""

    numeric = 0.0 if value == 0.0 else value
    return f"{_plain_decimal(numeric)} dB"


def format_envelope_curvature(value: float) -> str:
    """Format a legacy curvature readout without appending a unit."""

    numeric = 0.0 if value == 0.0 else value
    return _plain_decimal(numeric)


class EnvelopeValueEntry(QLineEdit):
    """Propose parsed values while retaining only parent-accepted exact values."""

    value_commit_requested = Signal(float)
    validation_failed = Signal(str)
    draft_reverted = Signal()
    focus_cancel_requested = Signal()

    def __init__(
        self,
        field_name: str,
        kind: EnvelopeFieldKind,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not isinstance(kind, EnvelopeFieldKind):
            raise TypeError("kind must be an EnvelopeFieldKind")
        self._field_name = field_name
        self._kind = kind
        self._last_rendered_text = ""
        self._last_exact_value: float | None = None
        self._last_duration_unit: str | None = None
        self._base_accessible_description: str | None = None
        self.returnPressed.connect(self._propose_commit)

    @property
    def kind(self) -> EnvelopeFieldKind:
        """Return this entry's immutable parsing and display mode."""

        return self._kind

    def set_exact_value(self, value: float) -> None:
        """Render and cache an exact model value without proposing a commit."""

        self._remember_and_render(self._validated_value(value))
        self._clear_error()

    def accept_proposed_value(self, value: float) -> None:
        """Cache and canonicalize a proposal accepted by the owning editor."""

        self._remember_and_render(self._validated_value(value))
        self._clear_error()

    def mark_commit_rejected(self, message: str) -> None:
        """Mark a parent-rejected draft without changing its text or accepted cache."""

        self._set_error(message)

    def set_editor_accessibility(self, name: str, description: str) -> None:
        """Set the stable screen-reader context for this transient editor."""

        self.setAccessibleName(name)
        self._base_accessible_description = description
        self.setAccessibleDescription(description)

    def restore_last_valid(self) -> None:
        """Restore the last parent-accepted canonical rendering without emitting."""

        self.setText(self._last_rendered_text)
        self._clear_error()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.restore_last_valid()
            self.draft_reverted.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        super().focusOutEvent(event)
        self.focus_cancel_requested.emit()

    def _propose_commit(self) -> None:
        text = self.text()
        try:
            if text == self._last_rendered_text and self._last_exact_value is not None:
                value = self._validated_value(self._last_exact_value)
            else:
                value = self._validated_value(self._parse(text))
        except ValueError as error:
            self._set_error(str(error))
            return
        self._clear_error()
        self.value_commit_requested.emit(value)

    def _parse(self, text: str) -> float:
        stripped = text.strip()
        if self._kind is EnvelopeFieldKind.DURATION:
            match = _DURATION.fullmatch(stripped)
            if match is None:
                raise self._invalid("enter a plain duration with optional ms or s")
            unit = match.group("unit") or self._last_duration_unit
            if unit is None:
                raise self._invalid("include ms or s for the first duration")
            value = float(match.group("number"))
            return value / 1_000.0 if unit == "ms" else value
        if self._kind is EnvelopeFieldKind.DECIBELS:
            match = _DECIBELS.fullmatch(stripped)
            if match is None:
                raise self._invalid("enter a plain decimal with optional dB")
            return float(match.group("number"))
        match = _CURVATURE.fullmatch(stripped)
        if match is None:
            raise self._invalid("enter a plain decimal from -1 to 1")
        return float(match.group("number"))

    def _validated_value(self, value: float) -> float:
        if isinstance(value, bool):
            raise self._invalid("enter a finite number")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise self._invalid("enter a finite number") from error
        if not math.isfinite(numeric):
            raise self._invalid("enter a finite number")
        if self._kind is EnvelopeFieldKind.DURATION and numeric <= 0.0:
            raise self._invalid("enter a duration greater than zero")
        if self._kind is EnvelopeFieldKind.DECIBELS and numeric > 0.0:
            raise self._invalid("enter a decibel value at or below 0 dB")
        if self._kind is EnvelopeFieldKind.CURVATURE and not -1.0 <= numeric <= 1.0:
            raise self._invalid("enter a curvature from -1 to 1")
        return numeric

    def _remember_and_render(self, value: float) -> None:
        rendered = self._render(value)
        self._last_exact_value = value
        self._last_rendered_text = rendered
        if self._kind is EnvelopeFieldKind.DURATION:
            self._last_duration_unit = "ms" if value < 1.0 else "s"
        self.setText(rendered)

    def _render(self, value: float) -> str:
        if self._kind is EnvelopeFieldKind.DURATION:
            return format_envelope_duration(value)
        if self._kind is EnvelopeFieldKind.DECIBELS:
            return format_envelope_decibels(value)
        return format_envelope_curvature(value)

    def _invalid(self, instruction: str) -> ValueError:
        return ValueError(f"{self._field_name}: {instruction}.")

    def _set_error(self, message: str) -> None:
        self.setProperty("validationState", "error")
        self._repolish()
        base_description = self._base_accessible_description or self.accessibleDescription()
        self.setAccessibleDescription(f"{base_description} {message}".strip())
        QAccessible.updateAccessibility(QAccessibleAnnouncementEvent(self, message))
        self.validation_failed.emit(message)

    def _clear_error(self) -> None:
        self.setProperty("validationState", None)
        self._repolish()
        if self._base_accessible_description is not None:
            self.setAccessibleDescription(self._base_accessible_description)

    def _repolish(self) -> None:
        self.style().unpolish(self)
        self.style().polish(self)
