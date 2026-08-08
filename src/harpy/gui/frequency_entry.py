"""Exact-frequency text entry with a stable rendered-value cache."""

from __future__ import annotations

import math
import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QLineEdit, QWidget

_DECIMAL = re.compile(r"[+-]?(?:\d+(?:\.\d{0,6})?|\.\d{1,6})\Z")


class FrequencyEntry(QLineEdit):
    """A bounded Hz entry that preserves exact values behind three-decimal display."""

    frequency_committed = Signal(float)
    validation_failed = Signal(str)

    def __init__(
        self,
        minimum_hz: float,
        maximum_hz: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not (
            math.isfinite(minimum_hz)
            and math.isfinite(maximum_hz)
            and 0.0 < minimum_hz <= maximum_hz
        ):
            raise ValueError("frequency bounds must be finite, positive, and ordered")
        self._minimum_hz = minimum_hz
        self._maximum_hz = maximum_hz
        self._current_rendered_text = ""
        self._current_exact_frequency_hz: float | None = None
        self.returnPressed.connect(self._commit)

    def set_frequency_hz(self, frequency_hz: float) -> None:
        """Render a model-supplied in-range value without emitting a user commit."""

        value = self._validated_frequency(frequency_hz)
        self._remember_and_render(value)
        self._clear_error()

    def restore_last_valid(self) -> None:
        """Discard an incomplete or invalid edit and recover the last displayed value."""

        self.setText(self._current_rendered_text)
        self._clear_error()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.restore_last_valid()
            event.accept()
            return
        super().keyPressEvent(event)

    def _commit(self) -> None:
        text = self.text()
        try:
            if text == self._current_rendered_text and self._current_exact_frequency_hz is not None:
                value = self._validated_frequency(self._current_exact_frequency_hz)
            else:
                value = self._validated_frequency(self._parse(text))
        except ValueError as error:
            self._set_error(str(error))
            return
        self._remember_and_render(value)
        self._clear_error()
        self.frequency_committed.emit(value)

    def _parse(self, text: str) -> float:
        if _DECIMAL.fullmatch(text) is None:
            raise ValueError("Enter a plain decimal frequency with up to six fractional places.")
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("Enter a finite frequency.")
        return value

    def _validated_frequency(self, frequency_hz: float) -> float:
        try:
            value = float(frequency_hz)
        except (TypeError, ValueError) as error:
            raise ValueError("Enter a finite frequency.") from error
        if not math.isfinite(value):
            raise ValueError("Enter a finite frequency.")
        if not self._minimum_hz <= value <= self._maximum_hz:
            raise ValueError(
                f"Enter a frequency from {self._minimum_hz:.3f} to {self._maximum_hz:.3f} Hz."
            )
        return value

    def _remember_and_render(self, value: float) -> None:
        rendered = f"{value:.3f}"
        self._current_rendered_text = rendered
        self._current_exact_frequency_hz = value
        self.setText(rendered)

    def _set_error(self, message: str) -> None:
        self.setProperty("validationState", "error")
        self.style().unpolish(self)
        self.style().polish(self)
        self.validation_failed.emit(message)

    def _clear_error(self) -> None:
        self.setProperty("validationState", None)
        self.style().unpolish(self)
        self.style().polish(self)
