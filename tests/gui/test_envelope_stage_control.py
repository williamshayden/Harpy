import math

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QAccessible, QKeyEvent, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QVBoxLayout, QWidget

from harpy.gui.envelope_stage_control import EnvelopeStageControl, EnvelopeValueStage
from harpy.synth.models import RenderConfig


def make_control(qtbot, stage, value, *, sample_rate=48_000):
    owner = {stage.field_name: value}
    control = EnvelopeStageControl(
        stage,
        RenderConfig(sample_rate_hz=sample_rate),
        lambda: owner[stage.field_name],
    )

    def accept(field: str, proposed: float) -> None:
        previous = owner[field]
        owner[field] = proposed
        control.owner_value_changed(previous, proposed)

    control.value_previewed.connect(accept)
    control.value_commit_requested.connect(accept)
    control.resize(84, 28)
    qtbot.addWidget(control)
    control.show()
    QApplication.processEvents()
    return control, owner


@pytest.mark.parametrize(
    ("stage", "value", "object_name", "text"),
    [
        (EnvelopeValueStage.ATTACK, 0.001, "attackValueControl", "A 1 ms"),
        (EnvelopeValueStage.DECAY, 0.600, "decayValueControl", "D 600 ms"),
        (EnvelopeValueStage.SUSTAIN, -6.0, "sustainValueControl", "S -6 dB"),
        (EnvelopeValueStage.RELEASE, 0.600, "releaseValueControl", "R 600 ms"),
    ],
)
def test_control_is_graph_native_and_queries_owner_truth(
    qtbot, stage, value, object_name, text
) -> None:
    # Caching durable values or using a line edit by default would desynchronize graph truth.
    control, owner = make_control(qtbot, stage, value)
    assert control.objectName() == object_name
    assert control.display_text == text
    owner[stage.field_name] = value * 2 if stage is not EnvelopeValueStage.SUSTAIN else -9.0
    control.update()
    assert control.current_value == owner[stage.field_name]
    assert control.findChildren(QLineEdit) == []
    assert control.minimumSizeHint().width() >= 24
    assert control.minimumSizeHint().height() >= 24


def send_mouse(control, event_type, local_y, global_y, button, buttons, modifiers=Qt.NoModifier):
    QApplication.sendEvent(
        control,
        QMouseEvent(
            event_type,
            QPointF(42, local_y),
            QPointF(42, local_y),
            QPointF(142, global_y),
            button,
            buttons,
            modifiers,
        ),
    )


def drag(control, *, origin_y=200, target_y=100, modifiers=Qt.NoModifier) -> None:
    send_mouse(
        control,
        QEvent.Type.MouseButtonPress,
        14,
        origin_y,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    send_mouse(
        control,
        QEvent.Type.MouseMove,
        14,
        target_y,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        modifiers,
    )
    send_mouse(
        control,
        QEvent.Type.MouseButtonRelease,
        14,
        target_y,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        modifiers,
    )


def test_subthreshold_pointer_motion_does_not_propose_or_revert(qtbot) -> None:
    # Treating every click as a drag would create accidental immutable edits.
    control, _ = make_control(qtbot, EnvelopeValueStage.ATTACK, 0.6)
    previews: list[float] = []
    commits: list[float] = []
    reverts: list[str] = []
    control.value_previewed.connect(lambda _field, value: previews.append(value))
    control.value_commit_requested.connect(lambda _field, value: commits.append(value))
    control.value_reverted.connect(reverts.append)
    threshold = QApplication.startDragDistance()
    drag(control, target_y=200 - max(1, threshold - 1))
    assert previews == []
    assert commits == []
    assert reverts == []


@pytest.mark.parametrize(
    ("stage", "origin", "target_y", "expected"),
    [
        (EnvelopeValueStage.ATTACK, 0.6, 100, 1.2),
        (EnvelopeValueStage.ATTACK, 0.6, 300, 0.3),
        (EnvelopeValueStage.SUSTAIN, -6.0, 180, -4.0),
        (EnvelopeValueStage.SUSTAIN, -1.0, 100, 0.0),
    ],
)
def test_vertical_scrub_uses_stage_specific_units(qtbot, stage, origin, target_y, expected) -> None:
    # A linear duration scrub or unbounded sustain would violate the direct-control contract.
    control, owner = make_control(qtbot, stage, origin)
    previews: list[float] = []
    commits: list[float] = []
    control.value_previewed.connect(lambda _field, value: previews.append(value))
    control.value_commit_requested.connect(lambda _field, value: commits.append(value))
    drag(control, target_y=target_y)
    assert previews
    assert commits == [previews[-1]]
    assert owner[stage.field_name] == pytest.approx(expected, rel=0, abs=1e-12)


def test_shift_scrub_accumulates_piecewise_without_jumping(qtbot) -> None:
    # Recalculating from press with current modifiers causes visible modifier jumps.
    control, owner = make_control(qtbot, EnvelopeValueStage.ATTACK, 0.6)
    send_mouse(control, QEvent.Type.MouseButtonPress, 14, 300, Qt.LeftButton, Qt.LeftButton)
    send_mouse(control, QEvent.Type.MouseMove, 14, 200, Qt.NoButton, Qt.LeftButton)
    send_mouse(
        control, QEvent.Type.MouseMove, 14, 200, Qt.NoButton, Qt.LeftButton, Qt.ShiftModifier
    )
    send_mouse(
        control, QEvent.Type.MouseMove, 14, 100, Qt.NoButton, Qt.LeftButton, Qt.ShiftModifier
    )
    send_mouse(control, QEvent.Type.MouseMove, 14, 100, Qt.NoButton, Qt.LeftButton)
    send_mouse(control, QEvent.Type.MouseMove, 14, 0, Qt.NoButton, Qt.LeftButton)
    send_mouse(control, QEvent.Type.MouseButtonRelease, 14, 0, Qt.LeftButton, Qt.NoButton)
    assert owner[EnvelopeValueStage.ATTACK.field_name] == pytest.approx(0.6 * 2**2.1)


def test_horizontal_motion_does_not_change_value(qtbot) -> None:
    # Using local Euclidean motion would let irrelevant horizontal motion edit a value.
    control, owner = make_control(qtbot, EnvelopeValueStage.ATTACK, 0.6)
    send_mouse(control, QEvent.Type.MouseButtonPress, 14, 200, Qt.LeftButton, Qt.LeftButton)
    QApplication.sendEvent(
        control,
        QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(83, 14),
            QPointF(83, 14),
            QPointF(9_999, 200),
            Qt.NoButton,
            Qt.LeftButton,
            Qt.NoModifier,
        ),
    )
    send_mouse(control, QEvent.Type.MouseButtonRelease, 14, 200, Qt.LeftButton, Qt.NoButton)
    assert owner[EnvelopeValueStage.ATTACK.field_name] == 0.6


def test_escape_ungrab_and_explicit_cancel_revert_once_after_preview(qtbot, monkeypatch) -> None:
    # Missing cancellation cleanup can leave a graph preview stranded after a lost grab.
    for cancellation in ("escape", "ungrab", "explicit"):
        control, _ = make_control(qtbot, EnvelopeValueStage.ATTACK, 0.6)
        reverts: list[str] = []
        commits: list[float] = []
        control.value_reverted.connect(reverts.append)
        control.value_commit_requested.connect(
            lambda _field, value, commits=commits: commits.append(value)
        )
        monkeypatch.setattr(control, "grabMouse", lambda: None)
        send_mouse(control, QEvent.Type.MouseButtonPress, 14, 200, Qt.LeftButton, Qt.LeftButton)
        send_mouse(control, QEvent.Type.MouseMove, 14, 100, Qt.NoButton, Qt.LeftButton)
        assert control.is_interacting
        if cancellation == "escape":
            QApplication.sendEvent(
                control,
                QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.NoModifier),
            )
        elif cancellation == "ungrab":
            QApplication.sendEvent(control, QEvent(QEvent.Type.UngrabMouse))
        else:
            control.cancel_interaction()
        assert reverts == [EnvelopeValueStage.ATTACK.field_name], cancellation
        assert commits == []


def test_duration_scrub_never_emits_nonfinite_value(qtbot) -> None:
    # An overflowing exponential proposal must stay finite so the owner can reject it.
    control, _ = make_control(qtbot, EnvelopeValueStage.ATTACK, 1.0)
    previews: list[float] = []
    control.value_previewed.connect(lambda _field, value: previews.append(value))
    send_mouse(control, QEvent.Type.MouseButtonPress, 14, 1e308, Qt.LeftButton, Qt.LeftButton)
    send_mouse(control, QEvent.Type.MouseMove, 14, -1e308, Qt.NoButton, Qt.LeftButton)
    assert previews and math.isfinite(previews[-1])


@pytest.mark.parametrize(
    ("stage", "value", "key", "modifier", "expected"),
    [
        (EnvelopeValueStage.ATTACK, 0.6, Qt.Key.Key_Up, Qt.NoModifier, 0.606),
        (EnvelopeValueStage.ATTACK, 0.6, Qt.Key.Key_Down, Qt.ShiftModifier, 0.6 / 1.001),
        (EnvelopeValueStage.SUSTAIN, -6.0, Qt.Key.Key_Right, Qt.NoModifier, -5.0),
        (EnvelopeValueStage.SUSTAIN, -6.0, Qt.Key.Key_Left, Qt.ShiftModifier, -6.1),
    ],
)
def test_arrow_keys_commit_normal_and_fine_stage_steps(
    qtbot, stage, value, key, modifier, expected
) -> None:
    # Additive duration keys would make opposite commands irreversibly drift.
    control, owner = make_control(qtbot, stage, value)
    QTest.keyClick(control, key, modifier)
    assert owner[stage.field_name] == pytest.approx(expected, rel=0, abs=1e-12)


def make_hosted_control(qtbot, stage=EnvelopeValueStage.ATTACK, value=0.001):
    host = QWidget()
    host.resize(240, 120)
    layout = QVBoxLayout(host)
    control = EnvelopeStageControl(
        stage,
        RenderConfig(),
        lambda: owner[stage.field_name],
        host,
    )
    owner = {stage.field_name: value}
    layout.addWidget(control)
    qtbot.addWidget(host)
    host.show()
    QApplication.processEvents()
    return host, control, owner


@pytest.mark.parametrize("trigger", [Qt.Key.Key_Return, Qt.Key.Key_F2])
def test_keyboard_exact_editor_is_transient_and_described(qtbot, trigger) -> None:
    # Creating a permanent line edit would violate the graph-native resting control.
    host, control, _ = make_hosted_control(qtbot)
    QTest.keyClick(control, trigger)
    entry = host.findChild(QLineEdit, "envelopeInlineEditor")
    assert entry is not None
    assert entry.accessibleName() == "Attack duration"
    assert "milliseconds" in entry.accessibleDescription().lower()
    assert "vertical drag" in entry.accessibleDescription().lower()
    assert "enter" in entry.accessibleDescription().lower()


def test_double_click_opens_exact_editor_without_scrub_proposal(qtbot) -> None:
    # Letting the double-click's first click commit would cause an accidental graph edit.
    host, control, _ = make_hosted_control(qtbot)
    previews: list[float] = []
    commits: list[float] = []
    control.value_previewed.connect(lambda _field, value: previews.append(value))
    control.value_commit_requested.connect(lambda _field, value: commits.append(value))
    QTest.mouseDClick(control, Qt.LeftButton)
    assert host.findChild(QLineEdit, "envelopeInlineEditor") is not None
    assert previews == []
    assert commits == []


def test_exact_entry_rejection_and_escape_follow_owner_lifecycle(qtbot, monkeypatch) -> None:
    # Closing an invalid entry or retaining it after Escape loses the exact-entry boundary.
    host, control, _ = make_hosted_control(qtbot)
    control.open_exact_editor()
    entry = host.findChild(QLineEdit, "envelopeInlineEditor")
    assert entry is not None
    announcements: list[object] = []
    monkeypatch.setattr(QAccessible, "updateAccessibility", announcements.append)
    errors: list[tuple[str, str]] = []
    control.validation_failed.connect(lambda field, message: errors.append((field, message)))
    entry.selectAll()
    QTest.keyClicks(entry, "nope")
    QTest.keyClick(entry, Qt.Key.Key_Return)
    assert entry.text() == "nope"
    assert entry.property("validationState") == "error"
    assert errors == [("attack_seconds", "Attack: enter a plain duration with optional ms or s.")]
    assert len(announcements) == 1
    QTest.keyClick(entry, Qt.Key.Key_Escape)
    QApplication.processEvents()
    assert host.findChild(QLineEdit, "envelopeInlineEditor") is None
    assert control.hasFocus()


def test_owner_acceptance_closes_exact_editor_and_emits_commit(qtbot) -> None:
    # A stage control must not cache acceptance before the immutable owner responds.
    host, control, owner = make_hosted_control(qtbot)
    commits: list[float] = []
    control.value_commit_requested.connect(lambda _field, value: commits.append(value))
    control.open_exact_editor()
    entry = host.findChild(QLineEdit, "envelopeInlineEditor")
    assert entry is not None
    entry.selectAll()
    QTest.keyClicks(entry, "2 ms")
    QTest.keyClick(entry, Qt.Key.Key_Return)
    assert commits == [0.002]
    assert host.findChild(QLineEdit, "envelopeInlineEditor") is entry
    owner[EnvelopeValueStage.ATTACK.field_name] = 0.002
    control.accept_exact_value(0.002)
    QApplication.processEvents()
    assert host.findChild(QLineEdit, "envelopeInlineEditor") is None
    assert control.hasFocus()


def test_owner_rejection_announces_exact_editor_error_and_escape_restores_context(
    qtbot, monkeypatch
) -> None:
    # Hiding an owner rejection from assistive technology leaves a valid draft unexplained.
    host, control, _ = make_hosted_control(qtbot)
    control.open_exact_editor()
    entry = host.findChild(QLineEdit, "envelopeInlineEditor")
    assert entry is not None
    base_description = entry.accessibleDescription()
    events: list[object] = []
    monkeypatch.setattr(QAccessible, "updateAccessibility", events.append)

    control.reject_exact_value("Attack duration is outside the renderable frame.")

    assert entry.property("validationState") == "error"
    assert entry.accessibleDescription() == (
        f"{base_description} Attack duration is outside the renderable frame."
    )
    assert len(events) == 1
    QTest.keyClick(entry, Qt.Key.Key_Escape)
    assert entry.accessibleDescription() == base_description
    assert len(events) == 1


def test_stage_accessibility_exposes_spinbox_owner_truth_and_actions(qtbot, monkeypatch) -> None:
    # Omitting the value interface would make the custom control inaccessible as a direct editor.
    control, owner = make_control(qtbot, EnvelopeValueStage.ATTACK, 0.6)
    events: list[object] = []
    monkeypatch.setattr(QAccessible, "updateAccessibility", events.append)
    interface = QAccessible.queryAccessibleInterface(control)
    assert interface.role() == QAccessible.Role.SpinBox
    assert interface.text(QAccessible.Text.Name) == "Attack duration"
    assert interface.text(QAccessible.Text.Value) == "600 ms"
    value_interface = interface.valueInterface()
    assert value_interface.currentValue() == 0.6
    assert value_interface.minimumValue() == pytest.approx(1 / 48_000)
    assert math.isinf(value_interface.maximumValue())
    assert value_interface.minimumStepSize() == pytest.approx(0.6 * 0.001)
    value_interface.doAction(value_interface.increaseAction())
    assert owner[EnvelopeValueStage.ATTACK.field_name] == pytest.approx(0.606)
    assert len(events) == 1
