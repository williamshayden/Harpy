from dataclasses import replace

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import (
    QAccessible,
    QAccessibleAnnouncementEvent,
    QAccessibleStateChangeEvent,
    QMouseEvent,
    QShortcut,
)
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

from harpy.gui.envelope_editor import EnvelopeEditor
from harpy.gui.envelope_entry import EnvelopeFieldKind, EnvelopeValueEntry
from harpy.gui.envelope_graph import EnvelopeGraph
from harpy.gui.envelope_stage_control import EnvelopeStageControl
from harpy.gui.workbench_controller import PatchApplyState
from harpy.synth.models import EnvelopeConfig, RenderConfig, SynthPatch


@pytest.fixture
def render() -> RenderConfig:
    return RenderConfig(sample_rate_hz=1_000)


@pytest.fixture
def starting_patch() -> SynthPatch:
    return SynthPatch(
        envelope=EnvelopeConfig(
            attack_seconds=0.125,
            decay_seconds=0.400,
            sustain_db=-9.0,
            release_seconds=0.850,
            attack_curve=0.2,
            decay_curve=-0.3,
            release_curve=0.4,
        ),
        output_gain_dbfs=-18.0,
    )


@pytest.fixture
def editor(qtbot, render: RenderConfig, starting_patch: SynthPatch) -> EnvelopeEditor:
    result = EnvelopeEditor(render, starting_patch)
    result.resize(288, 640)
    qtbot.addWidget(result)
    result.show()
    QApplication.processEvents()
    return result


def child(parent: QWidget, child_type: type[QWidget], name: str):
    result = parent.findChild(child_type, name)
    assert result is not None
    return result


def open_exact_editor(
    qtbot,
    editor: EnvelopeEditor,
    control_name: str,
) -> EnvelopeValueEntry:
    editor.activateWindow()
    QApplication.processEvents()
    control = child(editor, EnvelopeStageControl, control_name)
    control.setFocus(Qt.FocusReason.OtherFocusReason)
    qtbot.keyPress(control, Qt.Key.Key_F2)
    entry = editor.findChild(EnvelopeValueEntry, "envelopeInlineEditor")
    assert entry is not None and entry.isVisible() and entry.hasFocus()
    return entry


def commit_exact(
    qtbot,
    editor: EnvelopeEditor,
    control_name: str,
    text: str,
) -> EnvelopeValueEntry:
    entry = open_exact_editor(qtbot, editor, control_name)
    entry.selectAll()
    qtbot.keyClicks(entry, text)
    qtbot.keyPress(entry, Qt.Key.Key_Return)
    return entry


def send_stage_mouse(
    control: EnvelopeStageControl,
    event_type: QEvent.Type,
    global_y: float,
    button: Qt.MouseButton,
    buttons: Qt.MouseButton,
) -> None:
    QApplication.sendEvent(
        control,
        QMouseEvent(
            event_type,
            QPointF(control.rect().center()),
            QPointF(control.rect().center()),
            QPointF(100.0, global_y),
            button,
            buttons,
            Qt.KeyboardModifier.NoModifier,
        ),
    )


def test_construction_exposes_only_the_graph_native_editor_surface(
    editor: EnvelopeEditor,
) -> None:
    # Reintroducing a permanent text field would duplicate graph-native ownership.
    assert editor.objectName() == "envelopeEditor"
    object_types = {
        "envelopeGraph": EnvelopeGraph,
        "attackValueControl": EnvelopeStageControl,
        "decayValueControl": EnvelopeStageControl,
        "sustainValueControl": EnvelopeStageControl,
        "releaseValueControl": EnvelopeStageControl,
        "attackCurveHandle": QWidget,
        "decayCurveHandle": QWidget,
        "releaseCurveHandle": QWidget,
        "curveValueReadout": QLabel,
        "patchStatusLabel": QLabel,
        "resetEnvelopeButton": QPushButton,
        "envelopeFieldError": QLabel,
        "oscillatorFact": QLabel,
        "outputFact": QLabel,
        "loadPatchButton": QPushButton,
        "savePatchButton": QPushButton,
    }
    for name, object_type in object_types.items():
        assert editor.findChild(object_type, name) is not None, name
    removed_names = [f"{stage}Entry" for stage in ("attack", "decay", "sustain", "release")]
    removed_names.extend(("curve" + "Entry", "reset" + "CurvesButton"))
    for name in removed_names:
        assert editor.findChild(QWidget, name) is None
    assert editor.findChildren(EnvelopeValueEntry) == []
    assert set(EnvelopeFieldKind) == {
        EnvelopeFieldKind.DURATION,
        EnvelopeFieldKind.DECIBELS,
    }
    assert child(editor, QPushButton, "resetEnvelopeButton").text() == "Reset"
    assert child(editor, QLabel, "oscillatorFact").text() == "Sine"
    assert child(editor, QLabel, "outputFact").text() == "-18 dBFS"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("attack_seconds", 0.250),
        ("decay_seconds", 0.750),
        ("sustain_db", -12.0),
        ("release_seconds", 1.25),
        ("attack_curve", 0.375),
        ("decay_curve", -0.625),
        ("release_curve", 0.125),
    ],
)
def test_unified_preview_and_commit_change_only_the_named_field(
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
    field_name: str,
    value: float,
) -> None:
    # A field/stage translation bug would mutate a sibling or emit during preview.
    emitted: list[SynthPatch] = []
    editor.patch_commit_requested.connect(emitted.append)
    graph = child(editor, EnvelopeGraph, "envelopeGraph")

    graph.field_previewed.emit(field_name, value)

    expected_envelope = replace(starting_patch.envelope, **{field_name: value})
    assert graph.envelope == expected_envelope
    assert emitted == []

    graph.field_commit_requested.emit(field_name, value)

    assert emitted == [replace(starting_patch, envelope=expected_envelope)]
    for sibling in EnvelopeConfig.__dataclass_fields__:
        if sibling != field_name:
            assert getattr(emitted[0].envelope, sibling) == getattr(
                starting_patch.envelope, sibling
            )


def test_duration_exact_commit_and_pointer_commit_produce_equal_immutable_patches(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    # Exact entry and pointer release must share one owner validation/patch path.
    exact: list[SynthPatch] = []
    editor.patch_commit_requested.connect(exact.append)
    commit_exact(qtbot, editor, "attackValueControl", "250 ms")
    expected = replace(
        starting_patch,
        envelope=replace(starting_patch.envelope, attack_seconds=0.250),
    )
    assert exact == [expected]

    editor.set_patch_state(starting_patch, PatchApplyState.APPLIED, discard_draft=True)
    pointer: list[SynthPatch] = []
    editor.patch_commit_requested.disconnect(exact.append)
    editor.patch_commit_requested.connect(pointer.append)
    graph = child(editor, EnvelopeGraph, "envelopeGraph")
    graph.field_previewed.emit("attack_seconds", 0.250)
    graph.field_commit_requested.emit("attack_seconds", 0.250)
    assert pointer == [expected]
    assert pointer[0] == exact[0]
    assert pointer[0] is not exact[0]


def test_rejected_overflow_scrub_release_is_not_recast_as_authored_commit(
    qtbot,
    monkeypatch,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    # Losing the owner's preview rejection would turn release into an authored-value no-op commit.
    emitted: list[SynthPatch] = []
    errors: list[str] = []
    cleared: list[None] = []
    editor.patch_commit_requested.connect(emitted.append)
    editor.validation_failed.connect(errors.append)
    editor.validation_cleared.connect(lambda: cleared.append(None))
    control = child(editor, EnvelopeStageControl, "attackValueControl")
    error_label = child(editor, QLabel, "envelopeFieldError")
    monkeypatch.setattr(control, "grabMouse", lambda: None)
    monkeypatch.setattr(control, "releaseMouse", lambda: None)

    send_stage_mouse(
        control,
        QEvent.Type.MouseButtonPress,
        1e308,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
    )
    send_stage_mouse(
        control,
        QEvent.Type.MouseMove,
        -1e308,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
    )
    send_stage_mouse(
        control,
        QEvent.Type.MouseButtonRelease,
        -1e308,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
    )

    assert len(errors) == 1
    assert "finite frame count" in errors[0]
    assert emitted == []
    assert cleared == []
    assert error_label.text() == errors[0]
    assert control.property("validationState") == "error"
    assert child(editor, EnvelopeGraph, "envelopeGraph").envelope == starting_patch.envelope

    qtbot.keyPress(control, Qt.Key.Key_Up)

    assert len(emitted) == 1
    assert emitted[0].envelope.attack_seconds == pytest.approx(0.12625)
    assert cleared == [None]
    assert error_label.text() == ""
    assert control.property("validationState") is None


def test_sustain_exact_commit_uses_the_real_transient_editor(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    emitted: list[SynthPatch] = []
    editor.patch_commit_requested.connect(emitted.append)

    entry = commit_exact(qtbot, editor, "sustainValueControl", "-12 dB")

    assert emitted == [
        replace(
            starting_patch,
            envelope=replace(starting_patch.envelope, sustain_db=-12.0),
        )
    ]
    assert not entry.isVisible()
    assert child(editor, EnvelopeGraph, "envelopeGraph").envelope.sustain_db == -12.0


@pytest.mark.parametrize(
    ("field_name", "preview"),
    [
        ("attack_seconds", 0.250),
        ("sustain_db", -12.0),
        ("release_curve", -0.75),
    ],
)
def test_revert_restores_only_the_latest_authored_value(
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
    field_name: str,
    preview: float,
) -> None:
    graph = child(editor, EnvelopeGraph, "envelopeGraph")
    graph.field_previewed.emit("decay_curve", 0.75)
    graph.field_previewed.emit(field_name, preview)

    graph.field_reverted.emit(field_name)

    assert getattr(graph.envelope, field_name) == getattr(starting_patch.envelope, field_name)
    if field_name != "decay_curve":
        assert graph.envelope.decay_curve == 0.75


def test_owner_acknowledgment_canonicalizes_graph_without_feedback_commit(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    emitted: list[SynthPatch] = []

    def acknowledge(candidate: SynthPatch) -> None:
        emitted.append(candidate)
        editor.set_patch_state(candidate, PatchApplyState.APPLIED)

    editor.patch_commit_requested.connect(acknowledge)
    commit_exact(qtbot, editor, "attackValueControl", "0.250 s")
    candidate = replace(
        starting_patch,
        envelope=replace(starting_patch.envelope, attack_seconds=0.250),
    )
    graph = child(editor, EnvelopeGraph, "envelopeGraph")
    assert emitted == [candidate]
    assert graph.envelope == candidate.envelope
    assert child(editor, EnvelopeStageControl, "attackValueControl").display_text == "A 250 ms"
    assert child(editor, QLabel, "patchStatusLabel").text() == "Active"


def test_same_patch_refresh_preserves_invalid_transient_until_explicit_discard(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    graph = child(editor, EnvelopeGraph, "envelopeGraph")
    entry = open_exact_editor(qtbot, editor, "attackValueControl")
    entry.selectAll()
    qtbot.keyClicks(entry, "not complete")
    qtbot.keyPress(entry, Qt.Key.Key_Return)
    graph.field_previewed.emit("release_curve", -0.8)

    editor.set_patch_state(starting_patch, PatchApplyState.APPLIED)

    assert entry.text() == "not complete"
    assert entry.property("validationState") == "error"
    assert graph.envelope.release_curve == -0.8

    editor.set_patch_state(
        starting_patch,
        PatchApplyState.APPLIED,
        discard_draft=True,
    )
    QApplication.processEvents()
    assert editor.findChild(EnvelopeValueEntry, "envelopeInlineEditor") is None
    assert graph.envelope == starting_patch.envelope
    assert child(editor, QLabel, "envelopeFieldError").text() == ""


def test_nested_unacknowledged_commit_restores_outer_acknowledgment_scope(
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    emitted: list[SynthPatch] = []
    graph = child(editor, EnvelopeGraph, "envelopeGraph")

    def reenter_once(candidate: SynthPatch) -> None:
        emitted.append(candidate)
        if len(emitted) == 1:
            graph.field_commit_requested.emit("decay_seconds", 0.750)
            editor.set_patch_state(candidate, PatchApplyState.APPLIED)

    editor.patch_commit_requested.connect(reenter_once)
    graph.field_commit_requested.emit("attack_seconds", starting_patch.envelope.attack_seconds)
    nested = replace(
        starting_patch,
        envelope=replace(starting_patch.envelope, decay_seconds=0.750),
    )
    assert emitted == [starting_patch, nested]
    assert graph.envelope == starting_patch.envelope
    assert child(editor, QLabel, "patchStatusLabel").text() == "Active"


def test_status_precedence_and_state_change_accessibility_events(
    monkeypatch,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    # Pending must outrank local draft; unchanged refreshes must not spam assistive tech.
    events: list[object] = []
    monkeypatch.setattr(QAccessible, "updateAccessibility", events.append)
    graph = child(editor, EnvelopeGraph, "envelopeGraph")
    status = child(editor, QLabel, "patchStatusLabel")

    graph.field_previewed.emit("decay_curve", 0.75)
    editor.set_patch_state(starting_patch, PatchApplyState.PENDING)
    editor.set_patch_state(starting_patch, PatchApplyState.PENDING)
    editor.set_patch_state(starting_patch, PatchApplyState.APPLIED)
    graph.field_reverted.emit("decay_curve")
    graph.field_reverted.emit("decay_curve")

    state_events = [event for event in events if isinstance(event, QAccessibleStateChangeEvent)]
    assert status.text() == "Active"
    assert status.property("statusState") == "active"
    assert [event.object() for event in state_events] == [status, status, status, status]


def test_pending_invalid_transient_preserves_text_error_focus_and_single_announcement(
    qtbot,
    monkeypatch,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    events: list[object] = []
    monkeypatch.setattr(QAccessible, "updateAccessibility", events.append)
    entry = open_exact_editor(qtbot, editor, "attackValueControl")
    base_description = entry.accessibleDescription()
    entry.selectAll()
    qtbot.keyClicks(entry, "0.1 ms")
    qtbot.keyPress(entry, Qt.Key.Key_Return)
    editor.set_patch_state(starting_patch, PatchApplyState.PENDING)

    announcements = [event for event in events if isinstance(event, QAccessibleAnnouncementEvent)]
    error = child(editor, QLabel, "envelopeFieldError").text()
    assert len(announcements) == 1
    assert announcements[0].object() is entry
    assert announcements[0].message() == error
    assert entry.hasFocus()
    assert entry.text() == "0.1 ms"
    assert entry.property("validationState") == "error"
    assert entry.accessibleDescription() == f"{base_description} {error}"
    assert child(editor, QLabel, "patchStatusLabel").text() == "Pending"


@pytest.mark.parametrize("signal_name", ["field_previewed", "field_commit_requested"])
def test_nonentry_composed_rejection_marks_owner_and_announces_once(
    monkeypatch,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
    signal_name: str,
) -> None:
    events: list[object] = []
    monkeypatch.setattr(QAccessible, "updateAccessibility", events.append)
    errors: list[str] = []
    editor.validation_failed.connect(errors.append)
    graph = child(editor, EnvelopeGraph, "envelopeGraph")

    getattr(graph, signal_name).emit("attack_seconds", 0.0001)

    control = child(editor, EnvelopeStageControl, "attackValueControl")
    announcements = [event for event in events if isinstance(event, QAccessibleAnnouncementEvent)]
    assert len(errors) == 1 and "at least one frame" in errors[0]
    assert child(editor, QLabel, "envelopeFieldError").text() == errors[0]
    assert control.property("validationState") == "error"
    assert len(announcements) == 1
    assert announcements[0].object() is control
    assert announcements[0].message() == errors[0]
    assert graph.envelope == starting_patch.envelope


def test_valid_preview_clears_the_owned_local_error_once(
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    # Clearing only the local label would leave the window's editor-error owner stale.
    graph = child(editor, EnvelopeGraph, "envelopeGraph")
    cleared: list[None] = []
    editor.validation_cleared.connect(lambda: cleared.append(None))
    graph.field_commit_requested.emit("attack_seconds", 0.0001)

    graph.field_previewed.emit("attack_seconds", 0.250)

    assert cleared == [None]
    assert child(editor, QLabel, "envelopeFieldError").text() == ""
    assert graph.envelope == replace(starting_patch.envelope, attack_seconds=0.250)


def test_reset_emits_one_envelope_only_default_candidate(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    emitted: list[SynthPatch] = []
    editor.patch_commit_requested.connect(emitted.append)

    qtbot.mouseClick(
        child(editor, QPushButton, "resetEnvelopeButton"),
        Qt.MouseButton.LeftButton,
    )

    assert emitted == [replace(starting_patch, envelope=EnvelopeConfig())]
    assert emitted[0].oscillator == starting_patch.oscillator
    assert emitted[0].output_gain_dbfs == starting_patch.output_gain_dbfs
    assert not hasattr(editor, "selected_frequency_hz")


def test_reset_clean_defaults_emits_nothing_and_discards_invalid_transient(qtbot) -> None:
    patch = SynthPatch()
    editor = EnvelopeEditor(RenderConfig(sample_rate_hz=1_000), patch)
    qtbot.addWidget(editor)
    editor.show()
    emitted: list[SynthPatch] = []
    cleared: list[None] = []
    editor.patch_commit_requested.connect(emitted.append)
    editor.validation_cleared.connect(lambda: cleared.append(None))
    entry = open_exact_editor(qtbot, editor, "attackValueControl")
    entry.selectAll()
    qtbot.keyClicks(entry, "invalid")
    qtbot.keyPress(entry, Qt.Key.Key_Return)

    qtbot.mouseClick(child(editor, QPushButton, "resetEnvelopeButton"), Qt.MouseButton.LeftButton)
    QApplication.processEvents()

    assert emitted == []
    assert cleared == [None]
    assert editor.findChild(EnvelopeValueEntry, "envelopeInlineEditor") is None
    assert child(editor, QLabel, "envelopeFieldError").text() == ""
    assert child(editor, QLabel, "patchStatusLabel").text() == "Active"


def test_reset_cancels_value_and_curve_previews_before_constructing_defaults(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    emitted: list[SynthPatch] = []
    editor.patch_commit_requested.connect(emitted.append)
    graph = child(editor, EnvelopeGraph, "envelopeGraph")
    graph.field_previewed.emit("attack_seconds", 0.250)
    graph.field_previewed.emit("release_curve", -0.75)

    qtbot.mouseClick(child(editor, QPushButton, "resetEnvelopeButton"), Qt.MouseButton.LeftButton)

    assert emitted == [replace(starting_patch, envelope=EnvelopeConfig())]
    assert graph.envelope == EnvelopeConfig()


def test_pending_reset_emits_latest_authored_candidate_once_and_no_audio_command(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    emitted: list[SynthPatch] = []
    editor.patch_commit_requested.connect(emitted.append)
    editor.set_patch_state(starting_patch, PatchApplyState.PENDING)

    qtbot.mouseClick(child(editor, QPushButton, "resetEnvelopeButton"), Qt.MouseButton.LeftButton)

    assert emitted == [replace(starting_patch, envelope=EnvelopeConfig())]
    assert not hasattr(editor, "_send_command")
    assert child(editor, QLabel, "patchStatusLabel").text() == "Pending"


def test_reset_rejection_marks_button_and_leaves_authored_and_draft_truth(qtbot) -> None:
    render = RenderConfig(sample_rate_hz=100)
    patch = SynthPatch(
        envelope=EnvelopeConfig(
            attack_seconds=0.02,
            decay_seconds=0.02,
            release_seconds=0.02,
        )
    )
    editor = EnvelopeEditor(render, patch)
    qtbot.addWidget(editor)
    editor.show()
    emitted: list[SynthPatch] = []
    errors: list[str] = []
    editor.patch_commit_requested.connect(emitted.append)
    editor.validation_failed.connect(errors.append)
    reset = child(editor, QPushButton, "resetEnvelopeButton")

    qtbot.mouseClick(reset, Qt.MouseButton.LeftButton)

    assert emitted == []
    assert errors == ["Envelope: attack segment must contain at least one frame."]
    assert child(editor, QLabel, "envelopeFieldError").text() == errors[0]
    assert reset.property("validationState") == "error"
    assert child(editor, EnvelopeGraph, "envelopeGraph").envelope == patch.envelope


def test_programmatic_sync_discard_and_file_buttons_emit_no_feedback(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    observed: list[tuple[str, object | None]] = []
    editor.patch_commit_requested.connect(lambda patch: observed.append(("patch", patch)))
    editor.validation_failed.connect(lambda error: observed.append(("failed", error)))
    editor.validation_cleared.connect(lambda: observed.append(("cleared", None)))
    editor.load_requested.connect(lambda: observed.append(("load", None)))
    editor.save_requested.connect(lambda: observed.append(("save", None)))
    external = replace(
        starting_patch,
        envelope=replace(starting_patch.envelope, sustain_db=-15.0),
    )
    editor.set_patch_state(external, PatchApplyState.PENDING)
    editor.set_patch_state(external, PatchApplyState.APPLIED, discard_draft=True)
    editor.discard_draft()
    assert observed == []
    assert child(editor, EnvelopeGraph, "envelopeGraph").envelope == external.envelope

    qtbot.mouseClick(child(editor, QPushButton, "loadPatchButton"), Qt.MouseButton.LeftButton)
    qtbot.mouseClick(child(editor, QPushButton, "savePatchButton"), Qt.MouseButton.LeftButton)
    assert observed == [("load", None), ("save", None)]
    assert editor.findChildren(QShortcut) == []
