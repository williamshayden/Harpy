"""The native, stateless file-picker boundary for patch commands."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from PySide6.QtWidgets import QFileDialog, QWidget


class PatchDialogPort(Protocol):
    """Choose patch paths without exposing Qt file-dialog details to callers."""

    def choose_open_path(self, parent: QWidget) -> Path | None: ...

    def choose_save_path(self, parent: QWidget) -> Path | None: ...


class NativePatchDialogs:
    """Stateless native JSON pickers for the workbench shell."""

    _JSON_FILTER = "JSON Files (*.json)"

    def choose_open_path(self, parent: QWidget) -> Path | None:
        selected, _ = QFileDialog.getOpenFileName(parent, "Open Patch", "", self._JSON_FILTER)
        return Path(selected) if selected else None

    def choose_save_path(self, parent: QWidget) -> Path | None:
        selected, _ = QFileDialog.getSaveFileName(parent, "Save Patch", "", self._JSON_FILTER)
        return Path(selected) if selected else None
