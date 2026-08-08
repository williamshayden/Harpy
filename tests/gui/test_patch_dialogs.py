from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QWidget

from harpy.gui.patch_dialogs import NativePatchDialogs, PatchDialogPort


def test_native_dialogs_conform_to_the_patch_dialog_port_and_convert_selected_paths(
    qtbot, monkeypatch
) -> None:
    # Returning the Qt string directly would leak toolkit types through the dialog boundary.
    parent = QWidget()
    qtbot.addWidget(parent)
    calls: list[tuple[QWidget, str, str, str]] = []

    def choose_open(*args: object) -> tuple[str, str]:
        calls.append(args)  # type: ignore[arg-type]
        return ("/tmp/example.harpy.json", "JSON Files (*.json)")

    def choose_save(*args: object) -> tuple[str, str]:
        calls.append(args)  # type: ignore[arg-type]
        return ("/tmp/new-patch.json", "JSON Files (*.json)")

    monkeypatch.setattr(QFileDialog, "getOpenFileName", choose_open)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", choose_save)
    dialogs: PatchDialogPort = NativePatchDialogs()

    assert dialogs.choose_open_path(parent) == Path("/tmp/example.harpy.json")
    assert dialogs.choose_save_path(parent) == Path("/tmp/new-patch.json")
    assert [call[0] for call in calls] == [parent, parent]
    assert all("*.json" in call[3] for call in calls)


def test_native_dialogs_return_none_on_cancel(qtbot, monkeypatch) -> None:
    # Treating a cancelled picker as a path would trigger an unintended file operation.
    parent = QWidget()
    qtbot.addWidget(parent)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_args: ("", ""))
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: ("", ""))
    dialogs = NativePatchDialogs()

    assert dialogs.choose_open_path(parent) is None
    assert dialogs.choose_save_path(parent) is None
