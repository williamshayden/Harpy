from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QAccessible, QImage, QKeyEvent, QMouseEvent, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from harpy.gui.envelope_entry import EnvelopeValueEntry
from harpy.gui.envelope_graph import CurveStage, EnvelopeGraph, envelope_stage_fractions
from harpy.gui.envelope_stage_control import EnvelopeStageControl
from harpy.synth.curves import evaluate_quadratic_segment, sample_envelope_preview
from harpy.synth.models import EnvelopeConfig, RenderConfig


def make_graph(qtbot, envelope: EnvelopeConfig | None = None) -> EnvelopeGraph:
    graph = EnvelopeGraph(RenderConfig())
    graph.resize(360, 220)
    qtbot.addWidget(graph)
    graph.set_envelope(envelope or EnvelopeConfig())
    graph.show()
    QApplication.processEvents()
    return graph


def accept_graph_proposals(graph: EnvelopeGraph) -> None:
    def accept_graph_curve(stage: str, value: float) -> None:
        graph.set_envelope(replace(graph.envelope, **{f"{stage}_curve": value}))

    graph.curve_previewed.connect(accept_graph_curve)
    graph.curve_commit_requested.connect(accept_graph_curve)


def accept_value_field_proposals(graph: EnvelopeGraph) -> None:
    def accept(field_name: str, value: float) -> None:
        graph.set_envelope(replace(graph.envelope, **{field_name: value}))

    graph.field_previewed.connect(accept)
    graph.field_commit_requested.connect(accept)


def send_handle_drag(
    handle: QWidget,
    y_offsets: list[int],
    x_offsets: list[int] | None = None,
) -> None:
    origin = handle.rect().center()
    press_global = handle.mapToGlobal(origin)
    horizontal = x_offsets if x_offsets is not None else [0] * len(y_offsets)
    assert len(horizontal) == len(y_offsets)
    QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=origin)
    for x_offset, y_offset in zip(horizontal, y_offsets, strict=True):
        global_position = press_global + QPoint(x_offset, y_offset)
        position = QPointF(handle.mapFromGlobal(global_position))
        event = QMouseEvent(
            QEvent.Type.MouseMove,
            position,
            position,
            QPointF(global_position),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(handle, event)
    release_global = press_global + QPoint(horizontal[-1], y_offsets[-1])
    QTest.mouseRelease(
        handle,
        Qt.MouseButton.LeftButton,
        pos=handle.mapFromGlobal(release_global),
    )


def render_widget(widget: QWidget) -> QImage:
    image = QImage(widget.size(), QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    try:
        widget.render(painter, QPoint())
    finally:
        painter.end()
    return image


def image_bytes(image: QImage) -> bytes:
    return bytes(image.constBits()[: image.sizeInBytes()])


def send_native_double_click_sequence(handle: QWidget) -> None:
    position = handle.rect().center()
    QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=position)
    QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=position)
    QTest.mouseDClick(handle, Qt.MouseButton.LeftButton, pos=position)
    QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=position)


def test_stage_fractions_keep_fixed_sustain_and_log_distribute_durations() -> None:
    # A linear time axis would make the 1 ms attack unreadable beside long stages.
    render = RenderConfig(sample_rate_hz=48_000)
    envelope = EnvelopeConfig(
        attack_seconds=0.001,
        decay_seconds=0.600,
        release_seconds=1.200,
    )

    fractions = envelope_stage_fractions(envelope, render)

    assert fractions["sustain"] == 0.16
    assert sum(fractions.values()) == pytest.approx(1.0, rel=0.0, abs=1e-15)
    assert fractions["attack"] >= 0.14
    assert fractions["decay"] >= 0.14
    assert fractions["release"] >= 0.14
    assert fractions["attack"] < fractions["decay"] < fractions["release"]


def test_stage_fractions_follow_exact_log_weight_ratio() -> None:
    # Using raw frame counts instead of log1p would turn the 1:2:3 weights into 1:3:7.
    render = RenderConfig(sample_rate_hz=1)
    envelope = EnvelopeConfig(
        attack_seconds=1.0,
        decay_seconds=3.0,
        release_seconds=7.0,
    )

    fractions = envelope_stage_fractions(envelope, render)

    assert fractions == pytest.approx(
        {"attack": 0.21, "decay": 0.28, "sustain": 0.16, "release": 0.35},
        rel=0.0,
        abs=1e-15,
    )


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("attackValueControl", "A 125 ms"),
        ("decayValueControl", "D 400 ms"),
        ("sustainValueControl", "S -9 dB"),
        ("releaseValueControl", "R 850 ms"),
    ],
)
def test_graph_uses_real_inline_controls_and_actual_sustain(qtbot, name: str, text: str) -> None:
    # Painted duration labels would leave graph values inaccessible and sustain misleading.
    graph = make_graph(
        qtbot,
        EnvelopeConfig(
            attack_seconds=0.125,
            decay_seconds=0.400,
            sustain_db=-9.0,
            release_seconds=0.850,
        ),
    )

    control = graph.findChild(EnvelopeStageControl, name)

    assert control is not None
    assert control.display_text == text
    assert "hold" not in control.display_text.lower()
    assert control.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert control.cursor().shape() == Qt.CursorShape.SizeVerCursor
    assert control.toolTip()
    geometry = graph.display_geometry()
    rect = dict(geometry.stage_control_rects)[name.removesuffix("ValueControl")]
    assert control.geometry().height() >= 24
    assert rect.contains(QRectF(control.geometry()))
    assert graph.findChildren(EnvelopeValueEntry) == []
    readout = graph.findChild(QLabel, "curveValueReadout")
    assert readout is not None
    assert readout.isHidden()
    assert readout.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


def test_vertical_drag_previews_each_move_and_commits_once_on_release(qtbot) -> None:
    # Mutating graph state directly or committing each move would bypass editor ownership.
    graph = make_graph(qtbot)
    accept_graph_proposals(graph)
    previews: list[tuple[str, float]] = []
    commits: list[tuple[str, float]] = []
    graph.curve_previewed.connect(lambda stage, value: previews.append((stage, value)))
    graph.curve_commit_requested.connect(lambda stage, value: commits.append((stage, value)))
    handle = graph.findChild(QWidget, "attackCurveHandle")
    assert handle is not None

    send_handle_drag(handle, y_offsets=[-6, -12, -18], x_offsets=[-14, 0, 19])

    assert len(previews) == 3
    assert all(stage == "attack" and -1.0 <= value <= 1.0 for stage, value in previews)
    assert commits == [previews[-1]]


def test_owner_loopback_relayout_keeps_release_at_last_global_pointer_position(qtbot) -> None:
    # Child-local deltas feed handle relayout back into the next move and corrupt release value.
    graph = make_graph(qtbot)
    accept_graph_proposals(graph)
    handle = graph.findChild(QWidget, "attackCurveHandle")
    assert handle is not None
    previews: list[tuple[str, float]] = []
    commits: list[tuple[str, float]] = []
    graph.curve_previewed.connect(lambda stage, value: previews.append((stage, value)))
    graph.curve_commit_requested.connect(lambda stage, value: commits.append((stage, value)))

    send_handle_drag(handle, y_offsets=[-6, -12, -18])

    assert commits == [previews[-1]]
    assert len(previews) == 2


def test_horizontal_motion_cannot_change_drag_curvature(qtbot) -> None:
    # Deriving curvature from display x would make transformed stage duration alter the value.
    envelope = EnvelopeConfig(attack_curve=-0.25)
    graph = make_graph(qtbot, envelope)
    accept_graph_proposals(graph)
    handle = graph.findChild(QWidget, "attackCurveHandle")
    assert handle is not None
    previews: list[tuple[str, float]] = []
    commits: list[tuple[str, float]] = []
    graph.curve_previewed.connect(lambda stage, value: previews.append((stage, value)))
    graph.curve_commit_requested.connect(lambda stage, value: commits.append((stage, value)))

    send_handle_drag(handle, y_offsets=[-4, -9, -15], x_offsets=[-120, 0, 160])
    first_previews = list(previews)
    first_commit = list(commits)

    graph.set_envelope(envelope)
    previews.clear()
    commits.clear()
    send_handle_drag(handle, y_offsets=[-4, -9, -15], x_offsets=[800, -700, 999])

    assert previews == first_previews
    assert commits == first_commit


def test_tab_focus_visits_value_controls_and_curve_handles_in_stage_order(qtbot) -> None:
    # Missing an adjacent order pair would skip an editable graph-native control.
    graph = make_graph(qtbot)
    names = (
        "attackValueControl",
        "attackCurveHandle",
        "decayValueControl",
        "decayCurveHandle",
        "sustainValueControl",
        "releaseValueControl",
        "releaseCurveHandle",
    )
    controls = [graph.findChild(QWidget, name) for name in names]
    assert all(control is not None for control in controls)
    attack, *expected = controls
    attack.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    assert attack.hasFocus()

    for control in expected:
        QTest.keyClick(QApplication.focusWidget(), Qt.Key.Key_Tab)
        assert control.hasFocus()


def test_value_controls_propose_model_field_intents_without_replacing_curve_stage_intents(
    qtbot,
) -> None:
    # Routing a direct value through the curve channel would make its field ambiguous to the owner.
    graph = make_graph(qtbot)
    accept_graph_proposals(graph)
    accept_value_field_proposals(graph)
    curve_commits: list[tuple[str, float]] = []
    field_commits: list[tuple[str, float]] = []
    graph.curve_commit_requested.connect(lambda stage, value: curve_commits.append((stage, value)))
    graph.field_commit_requested.connect(lambda field, value: field_commits.append((field, value)))
    attack_curve = graph.findChild(QWidget, "attackCurveHandle")
    sustain = graph.findChild(EnvelopeStageControl, "sustainValueControl")
    release = graph.findChild(EnvelopeStageControl, "releaseValueControl")
    assert attack_curve is not None
    assert sustain is not None
    assert release is not None

    qtbot.keyPress(attack_curve, Qt.Key.Key_Right)
    qtbot.keyPress(sustain, Qt.Key.Key_Right)
    qtbot.keyPress(release, Qt.Key.Key_Right)

    assert curve_commits == [("attack", 0.01)]
    assert field_commits[0] == ("sustain_db", -5.9)
    assert field_commits[1] == ("release_seconds", pytest.approx(0.606))


def test_ownerless_value_proposal_preserves_graph_and_accessibility_truth(qtbot) -> None:
    # Caching a direct-stage proposal would create a second value truth without owner acceptance.
    envelope = EnvelopeConfig(sustain_db=-9.0)
    graph = make_graph(qtbot, envelope)
    control = graph.findChild(EnvelopeStageControl, "sustainValueControl")
    assert control is not None
    commits: list[tuple[str, float]] = []
    graph.field_commit_requested.connect(lambda field, value: commits.append((field, value)))

    qtbot.keyPress(control, Qt.Key.Key_Right)

    assert commits == [("sustain_db", -8.9)]
    assert graph.envelope is envelope
    interface = QAccessible.queryAccessibleInterface(control)
    assert interface is not None
    assert interface.valueInterface().currentValue() == -9.0


def test_curve_subthreshold_pointer_motion_is_a_noop(qtbot) -> None:
    # Treating an armed click as a drag would commit a curve while selecting it.
    graph = make_graph(qtbot)
    handle = graph.findChild(QWidget, "attackCurveHandle")
    assert handle is not None
    previews: list[tuple[str, float]] = []
    commits: list[tuple[str, float]] = []
    reverts: list[str] = []
    graph.curve_previewed.connect(lambda stage, value: previews.append((stage, value)))
    graph.curve_commit_requested.connect(lambda stage, value: commits.append((stage, value)))
    graph.curve_reverted.connect(reverts.append)
    threshold = QApplication.startDragDistance()

    send_handle_drag(handle, y_offsets=[-max(1, threshold - 1)])

    assert previews == []
    assert commits == []
    assert reverts == []


def test_curve_readout_uses_full_model_precision_only_while_handle_is_contextual(qtbot) -> None:
    # Reusing rounded entry text would hide the curve value the handle actually owns.
    graph = make_graph(qtbot, EnvelopeConfig(attack_curve=0.1234567))
    handle = graph.findChild(QWidget, "attackCurveHandle")
    readout = graph.findChild(QLabel, "curveValueReadout")
    assert handle is not None
    assert readout is not None

    handle.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()

    assert readout.isVisible()
    assert readout.text() == "Curve 0.123457"
    graph.setFocus(Qt.FocusReason.OtherFocusReason)
    QApplication.processEvents()
    assert readout.isHidden()


def test_curve_readout_survives_owner_loopback_while_its_handle_is_focused(qtbot) -> None:
    # Reconciling each unrelated handle independently must not hide active attack context.
    graph = make_graph(qtbot, EnvelopeConfig(attack_curve=0.25))
    accept_graph_proposals(graph)
    handle = graph.findChild(QWidget, "attackCurveHandle")
    readout = graph.findChild(QLabel, "curveValueReadout")
    assert handle is not None
    assert readout is not None

    handle.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    QTest.keyClick(handle, Qt.Key.Key_Right)

    assert graph.envelope.attack_curve == 0.26
    assert readout.isVisible()
    assert readout.text() == "Curve 0.26"


def test_stale_exact_accept_after_cancellation_cannot_suppress_a_later_revert(qtbot) -> None:
    # Remembering a canceled owner response would make a later draft close as if accepted.
    graph = make_graph(qtbot)
    attack = graph.findChild(EnvelopeStageControl, "attackValueControl")
    assert attack is not None
    reverts: list[str] = []
    graph.field_reverted.connect(reverts.append)

    attack.open_exact_editor()
    graph.cancel_interactions()
    reverts.clear()
    graph.accept_exact_edit("attack_seconds", graph.envelope.attack_seconds)
    attack.open_exact_editor()
    graph.setFocus(Qt.FocusReason.OtherFocusReason)
    QApplication.processEvents()

    assert reverts == ["attack_seconds"]


def test_stale_or_mismatched_exact_rejection_is_a_noop(qtbot) -> None:
    # Routing a response to an inactive field would surface an error unrelated to the open draft.
    graph = make_graph(qtbot)
    attack = graph.findChild(EnvelopeStageControl, "attackValueControl")
    assert attack is not None
    failures: list[tuple[str, str]] = []
    graph.field_validation_failed.connect(lambda field, message: failures.append((field, message)))

    attack.open_exact_editor()
    graph.reject_exact_edit("decay_seconds", "late owner response")
    graph.cancel_interactions()
    graph.reject_exact_edit("attack_seconds", "late owner response")

    assert failures == []


def test_curve_readout_survives_owner_loopback_during_drag_without_focus(qtbot) -> None:
    # Drag context must keep the readout visible even after focus has moved elsewhere.
    graph = make_graph(qtbot, EnvelopeConfig(attack_curve=0.25))
    accept_graph_proposals(graph)
    handle = graph.findChild(QWidget, "attackCurveHandle")
    readout = graph.findChild(QLabel, "curveValueReadout")
    assert handle is not None
    assert readout is not None
    origin = handle.rect().center()
    press_global = handle.mapToGlobal(origin)

    QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=origin)
    graph.setFocus(Qt.FocusReason.OtherFocusReason)
    global_position = press_global + QPoint(0, -20)
    QApplication.sendEvent(
        handle,
        QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(handle.mapFromGlobal(global_position)),
            QPointF(handle.mapFromGlobal(global_position)),
            QPointF(global_position),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        ),
    )

    assert readout.isVisible()
    assert readout.text() != "Curve 0.25"
    graph.cancel_interactions()


def test_curve_release_after_out_of_bounds_drag_commits_once_without_later_revert(qtbot) -> None:
    # A late capture-loss event must not undo a completed out-of-bounds release.
    graph = make_graph(qtbot)
    accept_graph_proposals(graph)
    handle = graph.findChild(QWidget, "attackCurveHandle")
    assert handle is not None
    commits: list[tuple[str, float]] = []
    reverts: list[str] = []
    graph.curve_commit_requested.connect(lambda stage, value: commits.append((stage, value)))
    graph.curve_reverted.connect(reverts.append)

    send_handle_drag(handle, y_offsets=[-20, -10_000])
    QApplication.sendEvent(handle, QEvent(QEvent.Type.UngrabMouse))

    assert len(commits) == 1
    assert reverts == []


def test_curve_ungrab_and_graph_cancel_revert_each_preview_once_without_commit(qtbot) -> None:
    # Cancellation must clear capture before release so the owner receives one
    # revert, never a commit.
    for cancellation in ("ungrab", "graph"):
        graph = make_graph(qtbot)
        handle = graph.findChild(QWidget, "attackCurveHandle")
        assert handle is not None
        reverts: list[str] = []
        commits: list[tuple[str, float]] = []
        graph.curve_reverted.connect(reverts.append)
        graph.curve_commit_requested.connect(
            lambda stage, value, commits=commits: commits.append((stage, value))
        )
        origin = handle.rect().center()
        press_global = handle.mapToGlobal(origin)
        QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=origin)
        global_position = press_global + QPoint(0, -20)
        QApplication.sendEvent(
            handle,
            QMouseEvent(
                QEvent.Type.MouseMove,
                QPointF(handle.mapFromGlobal(global_position)),
                QPointF(handle.mapFromGlobal(global_position)),
                QPointF(global_position),
                Qt.MouseButton.NoButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
        if cancellation == "ungrab":
            QApplication.sendEvent(handle, QEvent(QEvent.Type.UngrabMouse))
        else:
            graph.cancel_interactions()

        assert reverts == ["attack"], cancellation
        assert commits == [], cancellation


def test_second_editor_cancels_first_and_same_envelope_refresh_preserves_invalid_text(
    qtbot,
) -> None:
    # A model refresh must neither retain two overlays nor erase text the model did not accept.
    graph = make_graph(qtbot)
    attack = graph.findChild(EnvelopeStageControl, "attackValueControl")
    decay = graph.findChild(EnvelopeStageControl, "decayValueControl")
    assert attack is not None
    assert decay is not None

    attack.open_exact_editor()
    first = graph.findChild(EnvelopeValueEntry, "envelopeInlineEditor")
    assert first is not None
    first.setText("not a duration")
    graph.set_envelope(graph.envelope)
    assert first.text() == "not a duration"
    decay.open_exact_editor()
    editors = graph.findChildren(EnvelopeValueEntry, "envelopeInlineEditor")

    assert len(editors) == 1
    assert editors[0].hasFocus()


def test_focus_loss_cancellation_preserves_the_new_control_target(qtbot) -> None:
    # Returning focus to the closed editor's owner would steal the user's intended target.
    graph = make_graph(qtbot)
    attack = graph.findChild(EnvelopeStageControl, "attackValueControl")
    decay = graph.findChild(EnvelopeStageControl, "decayValueControl")
    assert attack is not None
    assert decay is not None

    attack.open_exact_editor()
    decay.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()

    assert decay.hasFocus()


def test_graph_forwards_exact_value_editor_lifecycle_and_owner_responses(qtbot) -> None:
    # Closing before owner acceptance would discard a rejected exact draft and its focus.
    graph = make_graph(qtbot)
    attack = graph.findChild(EnvelopeStageControl, "attackValueControl")
    decay = graph.findChild(EnvelopeStageControl, "decayValueControl")
    assert attack is not None
    assert decay is not None
    editing: list[tuple[str, bool]] = []
    reverts: list[str] = []
    commits: list[tuple[str, float]] = []
    graph.field_editing_changed.connect(lambda field, active: editing.append((field, active)))
    graph.field_reverted.connect(reverts.append)
    graph.field_commit_requested.connect(lambda field, value: commits.append((field, value)))

    attack.open_exact_editor()
    first = graph.findChild(EnvelopeValueEntry, "envelopeInlineEditor")
    assert first is not None
    first.setText("125 ms")
    QTest.keyClick(first, Qt.Key.Key_Return)
    assert commits == [("attack_seconds", 0.125)]
    assert first.isVisible()

    graph.reject_exact_edit("attack_seconds", "too long")
    assert first.isVisible()
    assert first.hasFocus()
    graph.set_envelope(replace(graph.envelope, attack_seconds=0.125))
    graph.accept_exact_edit("attack_seconds", 0.125)
    assert graph.findChild(EnvelopeValueEntry, "envelopeInlineEditor") is None
    assert attack.hasFocus()

    decay.open_exact_editor()
    second = graph.findChild(EnvelopeValueEntry, "envelopeInlineEditor")
    assert second is not None
    second.setFocus()
    graph.setFocus(Qt.FocusReason.OtherFocusReason)
    QApplication.processEvents()
    assert editing == [
        ("attack_seconds", True),
        ("attack_seconds", False),
        ("decay_seconds", True),
        ("decay_seconds", False),
    ]
    assert reverts == ["decay_seconds"]


@pytest.mark.parametrize("stage", list(CurveStage))
def test_handle_accessibility_reports_owner_value_and_adjusts_by_patch_request(
    qtbot, stage: CurveStage
) -> None:
    # A generic Client or pixel value would hide the bounded curvature model from AT.
    curves = {"attack": 0.2, "decay": -0.3, "release": 0.4}
    graph = make_graph(
        qtbot,
        EnvelopeConfig(**{f"{name}_curve": value for name, value in curves.items()}),
    )
    accept_graph_proposals(graph)
    handle = graph.findChild(QWidget, f"{stage.value}CurveHandle")
    assert handle is not None
    commits: list[tuple[str, float]] = []
    graph.curve_commit_requested.connect(lambda name, value: commits.append((name, value)))

    interface = QAccessible.queryAccessibleInterface(handle)
    assert interface is not None
    assert interface.role() == QAccessible.Role.Slider
    assert stage.value in interface.text(QAccessible.Text.Name).lower()
    value_interface = interface.valueInterface()
    assert value_interface is not None
    assert value_interface.currentValue() == curves[stage.value]
    assert value_interface.minimumValue() == -1.0
    assert value_interface.maximumValue() == 1.0
    assert value_interface.minimumStepSize() == 0.001

    actions = interface.actionInterface()
    assert actions is not None
    actions.doAction(actions.increaseAction())
    actions.doAction(actions.decreaseAction())
    value_interface.setCurrentValue(0.375)

    assert commits == [
        (stage.value, pytest.approx(curves[stage.value] + 0.01)),
        (stage.value, pytest.approx(curves[stage.value])),
        (stage.value, 0.375),
    ]
    assert getattr(graph.envelope, f"{stage.value}_curve") == 0.375


@pytest.mark.parametrize("stage", list(CurveStage))
def test_keyboard_arrows_shift_home_and_autorepeat_obey_patch_steps(
    qtbot, stage: CurveStage
) -> None:
    # Wrong direction, step size, or autorepeat would create unintended patch history.
    graph = make_graph(qtbot)
    accept_graph_proposals(graph)
    handle = graph.findChild(QWidget, f"{stage.value}CurveHandle")
    assert handle is not None
    commits: list[tuple[str, float]] = []
    graph.curve_commit_requested.connect(lambda name, value: commits.append((name, value)))
    handle.setFocus()

    qtbot.keyPress(handle, Qt.Key.Key_Right)
    qtbot.keyPress(handle, Qt.Key.Key_Up)
    qtbot.keyPress(handle, Qt.Key.Key_Left)
    qtbot.keyPress(handle, Qt.Key.Key_Down)
    qtbot.keyPress(
        handle,
        Qt.Key.Key_Right,
        modifier=Qt.KeyboardModifier.ShiftModifier,
    )
    qtbot.keyPress(handle, Qt.Key.Key_Home)
    commit_count = len(commits)
    QApplication.sendEvent(
        handle,
        QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_Right,
            Qt.KeyboardModifier.NoModifier,
            "",
            True,
            1,
        ),
    )

    assert commits == [
        (stage.value, 0.01),
        (stage.value, 0.02),
        (stage.value, 0.01),
        (stage.value, 0.0),
        (stage.value, 0.001),
        (stage.value, 0.0),
        (stage.value, 0.01),
    ]
    assert len(commits) == commit_count + 1


def test_keyboard_sequence_uses_synchronously_replaced_envelope_truth(qtbot) -> None:
    # Caching a handle-local curve would make the fine adjustment restart from zero.
    graph = make_graph(qtbot)
    accept_graph_proposals(graph)
    handle = graph.findChild(QWidget, "attackCurveHandle")
    assert handle is not None
    commits: list[tuple[str, float]] = []
    graph.curve_commit_requested.connect(lambda stage, value: commits.append((stage, value)))

    handle.setFocus()
    qtbot.keyPress(handle, Qt.Key.Key_Right)
    qtbot.keyPress(
        handle,
        Qt.Key.Key_Up,
        modifier=Qt.KeyboardModifier.ShiftModifier,
    )
    qtbot.keyPress(handle, Qt.Key.Key_Home)

    assert commits[-3:] == [
        ("attack", 0.01),
        ("attack", 0.011),
        ("attack", 0.0),
    ]


@pytest.mark.parametrize("stage", list(CurveStage))
def test_escape_reverts_without_commit_and_double_click_commits_only_stage_zero(
    qtbot, stage: CurveStage
) -> None:
    # Treating Escape as reset or resetting every curve would corrupt unrelated draft state.
    graph = make_graph(
        qtbot,
        EnvelopeConfig(attack_curve=0.2, decay_curve=-0.3, release_curve=0.4),
    )
    handle = graph.findChild(QWidget, f"{stage.value}CurveHandle")
    assert handle is not None
    reverts: list[str] = []
    commits: list[tuple[str, float]] = []
    graph.curve_reverted.connect(reverts.append)
    graph.curve_commit_requested.connect(lambda name, value: commits.append((name, value)))
    handle.setFocus()

    qtbot.keyPress(handle, Qt.Key.Key_Escape)
    assert reverts == []
    assert commits == []

    QTest.mouseDClick(handle, Qt.MouseButton.LeftButton, pos=handle.rect().center())
    assert reverts == []
    assert commits == [(stage.value, 0.0)]


def test_native_double_click_sequence_emits_only_one_zero_reset_commit(qtbot) -> None:
    # Committing the first click's unchanged value creates an extra patch before the reset.
    graph = make_graph(qtbot, EnvelopeConfig(attack_curve=0.4))
    accept_graph_proposals(graph)
    handle = graph.findChild(QWidget, "attackCurveHandle")
    assert handle is not None
    commits: list[tuple[str, float]] = []
    graph.curve_commit_requested.connect(lambda stage, value: commits.append((stage, value)))

    send_native_double_click_sequence(handle)

    assert commits == [("attack", 0.0)]
    assert graph.envelope.attack_curve == 0.0


@pytest.mark.parametrize("stage", list(CurveStage))
def test_focus_and_mouse_press_select_the_owned_stage(qtbot, stage: CurveStage) -> None:
    # Failing to identify the active stage would desynchronize graph and editor selection.
    graph = make_graph(qtbot)
    handle = graph.findChild(QWidget, f"{stage.value}CurveHandle")
    assert handle is not None
    other_stage = CurveStage.DECAY if stage is CurveStage.ATTACK else CurveStage.ATTACK
    other_handle = graph.findChild(QWidget, f"{other_stage.value}CurveHandle")
    assert other_handle is not None
    other_handle.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    selected: list[str] = []
    graph.stage_selected.connect(selected.append)

    handle.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    assert selected[-1] == stage.value

    selected.clear()
    QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=handle.rect().center())
    assert selected[-1] == stage.value
    QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=handle.rect().center())


@pytest.mark.parametrize(
    ("stage", "envelope", "endpoint"),
    [
        (CurveStage.DECAY, EnvelopeConfig(sustain_db=0.0, decay_curve=0.4), "top"),
        (
            CurveStage.RELEASE,
            EnvelopeConfig(sustain_db=-10_000.0, release_curve=-0.4),
            "bottom",
        ),
    ],
)
def test_zero_span_handles_are_stable_mouse_inert_and_nonmouse_adjustable(
    qtbot, stage: CurveStage, envelope: EnvelopeConfig, endpoint: str
) -> None:
    # Dividing by a collapsed amplitude span would crash or emit meaningless mouse values.
    graph = make_graph(qtbot, envelope)
    accept_graph_proposals(graph)
    handle = graph.findChild(QWidget, f"{stage.value}CurveHandle")
    assert handle is not None
    geometry = graph.display_geometry()
    stage_rect = dict(geometry.stage_rects)[stage.value]
    center = dict(geometry.handle_centers)[stage]
    expected_y = stage_rect.top() if endpoint == "top" else stage_rect.bottom()
    assert center.y() == expected_y

    graph.set_envelope(replace(graph.envelope, **{f"{stage.value}_curve": -0.9}))
    assert dict(graph.display_geometry().handle_centers)[stage] == center
    previews: list[tuple[str, float]] = []
    commits: list[tuple[str, float]] = []
    graph.curve_previewed.connect(lambda name, value: previews.append((name, value)))
    graph.curve_commit_requested.connect(lambda name, value: commits.append((name, value)))
    send_handle_drag(handle, y_offsets=[-8, -16, 12])
    assert previews == []
    assert commits == []

    qtbot.keyPress(handle, Qt.Key.Key_Right)
    interface = QAccessible.queryAccessibleInterface(handle)
    assert interface is not None
    actions = interface.actionInterface()
    assert actions is not None
    actions.doAction(actions.increaseAction())
    assert commits == [(stage.value, -0.89), (stage.value, -0.88)]


def test_preview_levels_match_synth_curve_equations_and_are_owned_read_only(qtbot) -> None:
    # A painter-only approximation or shared writable sample would diverge from synth truth.
    render = RenderConfig(sample_rate_hz=48_000)
    envelope = EnvelopeConfig(
        sustain_db=-9.0,
        attack_curve=-0.75,
        decay_curve=0.35,
        release_curve=0.8,
    )
    graph = EnvelopeGraph(render)
    graph.resize(360, 220)
    qtbot.addWidget(graph)
    graph.set_envelope(envelope)
    preview = sample_envelope_preview(envelope, render)
    cases = [
        (CurveStage.ATTACK, 0.0, 1.0, envelope.attack_curve, preview.attack_level),
        (
            CurveStage.DECAY,
            1.0,
            envelope.sustain_amplitude,
            envelope.decay_curve,
            preview.decay_level,
        ),
        (
            CurveStage.RELEASE,
            envelope.sustain_amplitude,
            0.0,
            envelope.release_curve,
            preview.release_level,
        ),
    ]

    for stage, start, end, curve, sampled in cases:
        levels = graph.preview_levels(stage)
        expected = evaluate_quadratic_segment(start, end, curve, preview.position)
        np.testing.assert_allclose(levels, expected, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(levels, sampled, rtol=0.0, atol=1e-12)
        assert not levels.flags.writeable
        assert graph.preview_levels(stage) is not levels
        with pytest.raises(ValueError):
            levels[0] = 99.0


@pytest.mark.parametrize("width", [240, 288, 360, 512])
def test_display_geometry_stays_inside_owned_contents_at_supported_widths(
    qtbot, width: int
) -> None:
    # Independent layout calculations would create gaps or let labels and handles clip.
    graph = make_graph(qtbot)
    graph.resize(width, 220)
    QApplication.processEvents()
    geometry = graph.display_geometry()

    assert [name for name, _ in geometry.stage_rects] == [
        "attack",
        "decay",
        "sustain",
        "release",
    ]
    assert geometry.stage_rects[0][1].left() == geometry.contents.left()
    assert geometry.stage_rects[-1][1].right() == geometry.contents.right()
    assert sum(rect.width() for _, rect in geometry.stage_rects) == pytest.approx(
        geometry.contents.width(), rel=0.0, abs=1e-12
    )
    assert all(geometry.contents.contains(rect) for _, rect in geometry.stage_rects)
    assert all(geometry.contents.contains(rect) for _, rect in geometry.stage_control_rects)
    assert all(geometry.contents.contains(center) for _, center in geometry.handle_centers)

    geometry.contents.setLeft(99.0)
    geometry.stage_rects[0][1].setLeft(88.0)
    geometry.handle_centers[0][1].setX(77.0)
    fresh = graph.display_geometry()
    assert fresh.contents.left() == 12.0
    assert fresh.stage_rects[0][1].left() == 12.0
    assert fresh.handle_centers[0][1].x() != 77.0


def test_focus_selection_and_linear_curve_render_distinct_without_paint_mutation(qtbot) -> None:
    # Missing state cues or paint-time model edits would make authoring ambiguous.
    envelope = EnvelopeConfig(attack_curve=0.75, decay_curve=-0.6, release_curve=0.5)
    graph = make_graph(qtbot, envelope)
    for stage in CurveStage:
        handle = graph.findChild(QWidget, f"{stage.value}CurveHandle")
        assert handle is not None
        handle.clearFocus()
    QApplication.processEvents()
    before = graph.envelope
    idle_image = render_widget(graph)

    attack = graph.findChild(QWidget, "attackCurveHandle")
    assert attack is not None
    attack.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()
    focused_image = render_widget(graph)
    assert graph.envelope == before

    graph.select_stage(CurveStage.DECAY)
    selected_image = render_widget(graph)
    assert graph.envelope == before

    linear_graph = make_graph(qtbot, EnvelopeConfig())
    for stage in CurveStage:
        linear_handle = linear_graph.findChild(QWidget, f"{stage.value}CurveHandle")
        assert linear_handle is not None
        linear_handle.clearFocus()
    QApplication.processEvents()
    linear_before = linear_graph.envelope
    linear_image = render_widget(linear_graph)
    assert linear_graph.envelope == linear_before

    assert image_bytes(idle_image) != image_bytes(focused_image)
    assert image_bytes(focused_image) != image_bytes(selected_image)
    assert image_bytes(idle_image) != image_bytes(linear_image)


def test_ownerless_proposals_never_create_a_second_curve_truth(qtbot) -> None:
    # A handle-local durable value would change after a proposal even without owner acceptance.
    envelope = EnvelopeConfig(attack_curve=0.25)
    graph = make_graph(qtbot, envelope)
    handle = graph.findChild(QWidget, "attackCurveHandle")
    assert handle is not None
    commits: list[tuple[str, float]] = []
    graph.curve_commit_requested.connect(lambda stage, value: commits.append((stage, value)))

    qtbot.keyPress(handle, Qt.Key.Key_Right)

    assert commits == [("attack", 0.26)]
    assert graph.envelope is envelope
    interface = QAccessible.queryAccessibleInterface(handle)
    assert interface is not None
    assert interface.valueInterface().currentValue() == 0.25
