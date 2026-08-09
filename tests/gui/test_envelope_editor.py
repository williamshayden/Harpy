from dataclasses import replace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QShortcut
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

from harpy.gui.envelope_editor import EnvelopeEditor
from harpy.gui.envelope_entry import EnvelopeFieldKind, EnvelopeValueEntry
from harpy.gui.envelope_graph import CurveStage, EnvelopeGraph
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
    result.resize(320, 640)
    qtbot.addWidget(result)
    result.show()
    QApplication.processEvents()
    return result


def child(editor: EnvelopeEditor, child_type: type[QWidget], name: str):
    result = editor.findChild(child_type, name)
    assert result is not None
    return result


def replace_entry_text(qtbot, entry: EnvelopeValueEntry, text: str) -> None:
    entry.selectAll()
    qtbot.keyClicks(entry, text)


def test_construction_exposes_the_complete_accessible_object_contract(
    editor: EnvelopeEditor,
) -> None:
    # Omitting or renaming a control would break window composition and accessibility lookup.
    assert editor.objectName() == "envelopeEditor"
    object_types = {
        "envelopeGraph": EnvelopeGraph,
        "attackEntry": EnvelopeValueEntry,
        "decayEntry": EnvelopeValueEntry,
        "sustainEntry": EnvelopeValueEntry,
        "releaseEntry": EnvelopeValueEntry,
        "curveEntry": EnvelopeValueEntry,
        "patchStatusLabel": QLabel,
        "envelopeFieldError": QLabel,
        "resetCurvesButton": QPushButton,
        "oscillatorFact": QLabel,
        "outputFact": QLabel,
        "loadPatchButton": QPushButton,
        "savePatchButton": QPushButton,
    }
    for name, object_type in object_types.items():
        assert editor.findChild(object_type, name) is not None

    entries = {
        "attackEntry": (
            EnvelopeFieldKind.DURATION,
            "Attack duration",
            "Duration in milliseconds or seconds; at least one frame.",
        ),
        "decayEntry": (
            EnvelopeFieldKind.DURATION,
            "Decay duration",
            "Duration in milliseconds or seconds; at least one frame.",
        ),
        "sustainEntry": (
            EnvelopeFieldKind.DECIBELS,
            "Sustain level",
            "Level in decibels at or below 0 dB.",
        ),
        "releaseEntry": (
            EnvelopeFieldKind.DURATION,
            "Release duration",
            "Duration in milliseconds or seconds; at least one frame.",
        ),
        "curveEntry": (
            EnvelopeFieldKind.CURVATURE,
            "Selected curve",
            "Curvature from -1 to 1; 0 is linear.",
        ),
    }
    for name, (kind, accessible_name, accessible_description) in entries.items():
        entry = child(editor, EnvelopeValueEntry, name)
        assert entry.kind is kind
        assert entry.accessibleName() == accessible_name
        assert entry.accessibleDescription() == accessible_description

    assert child(editor, QLabel, "oscillatorFact").text() == "Sine"
    assert child(editor, QLabel, "outputFact").text() == "-18 dBFS"


@pytest.mark.parametrize(
    ("entry_name", "field_name", "text", "value"),
    [
        ("attackEntry", "attack_seconds", "250 ms", 0.250),
        ("decayEntry", "decay_seconds", "750 ms", 0.750),
        ("sustainEntry", "sustain_db", "-12 dB", -12.0),
        ("releaseEntry", "release_seconds", "1.25 s", 1.25),
    ],
)
def test_numeric_return_emits_one_complete_patch_changing_only_that_envelope_field(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
    entry_name: str,
    field_name: str,
    text: str,
    value: float,
) -> None:
    # Building from defaults or a partial envelope would erase unrelated authored settings.
    emitted: list[SynthPatch] = []
    editor.patch_commit_requested.connect(emitted.append)
    entry = child(editor, EnvelopeValueEntry, entry_name)

    replace_entry_text(qtbot, entry, text)
    qtbot.keyPress(entry, Qt.Key.Key_Return)

    assert emitted == [
        replace(
            starting_patch,
            envelope=replace(starting_patch.envelope, **{field_name: value}),
        )
    ]
    assert emitted[0].oscillator == starting_patch.oscillator
    assert emitted[0].output_gain_dbfs == starting_patch.output_gain_dbfs


def test_sub_frame_duration_stays_invalid_and_escape_restores_authored_attack(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    # Accepting a parseable sub-frame value would poison the entry cache and emitted patch.
    emitted: list[SynthPatch] = []
    errors: list[str] = []
    cleared: list[None] = []
    editor.patch_commit_requested.connect(emitted.append)
    editor.validation_failed.connect(errors.append)
    editor.validation_cleared.connect(lambda: cleared.append(None))
    attack = child(editor, EnvelopeValueEntry, "attackEntry")
    graph = child(editor, EnvelopeGraph, "envelopeGraph")
    error_label = child(editor, QLabel, "envelopeFieldError")
    status = child(editor, QLabel, "patchStatusLabel")

    replace_entry_text(qtbot, attack, "0.1 ms")
    qtbot.keyPress(attack, Qt.Key.Key_Return)

    assert emitted == []
    assert len(errors) == 1
    assert "Attack" in errors[0]
    assert "at least one frame" in errors[0]
    assert error_label.text() == errors[0]
    assert attack.text() == "0.1 ms"
    assert attack.property("validationState") == "error"
    assert graph.envelope == starting_patch.envelope
    assert status.text() == "Editing"

    qtbot.keyPress(attack, Qt.Key.Key_Escape)

    assert attack.text() == "125 ms"
    assert attack.property("validationState") is None
    assert error_label.text() == ""
    assert status.text() == "Active"
    assert emitted == []
    assert cleared == [None]


def test_graph_previews_stay_local_and_release_emits_only_the_latest_complete_patch(
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    # Emitting on preview or committing a stale preview would create extra patch history.
    emitted: list[SynthPatch] = []
    editor.patch_commit_requested.connect(emitted.append)
    graph = child(editor, EnvelopeGraph, "envelopeGraph")
    curve_entry = child(editor, EnvelopeValueEntry, "curveEntry")

    graph.curve_previewed.emit("attack", 0.10)
    graph.curve_previewed.emit("attack", 0.20)
    graph.curve_previewed.emit("attack", 0.30)

    assert emitted == []
    assert graph.envelope.attack_curve == 0.30
    assert curve_entry.text() == "0.3"

    graph.curve_commit_requested.emit("attack", 0.30)

    assert emitted == [
        replace(
            starting_patch,
            envelope=replace(starting_patch.envelope, attack_curve=0.30),
        )
    ]


@pytest.mark.parametrize(("stage", "value"), [("decay", 0.375), ("release", -0.625)])
def test_graph_curve_commit_maps_only_to_its_owned_stage(
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
    stage: str,
    value: float,
) -> None:
    # Mapping every handle through the selected stage would update the wrong curve.
    emitted: list[SynthPatch] = []
    editor.patch_commit_requested.connect(emitted.append)
    graph = child(editor, EnvelopeGraph, "envelopeGraph")

    graph.curve_previewed.emit(stage, value)
    graph.curve_commit_requested.emit(stage, value)

    assert emitted == [
        replace(
            starting_patch,
            envelope=replace(starting_patch.envelope, **{f"{stage}_curve": value}),
        )
    ]


def test_handle_double_click_and_reset_all_each_emit_one_atomic_curve_patch(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    # Resetting through individual fields would emit partial or multiple candidates.
    emitted: list[SynthPatch] = []
    editor.patch_commit_requested.connect(emitted.append)
    graph = child(editor, EnvelopeGraph, "envelopeGraph")
    attack_handle = child(graph, QWidget, "attackCurveHandle")

    QTest.mouseDClick(
        attack_handle,
        Qt.MouseButton.LeftButton,
        pos=attack_handle.rect().center(),
    )

    assert emitted == [
        replace(
            starting_patch,
            envelope=replace(starting_patch.envelope, attack_curve=0.0),
        )
    ]

    emitted.clear()
    reset = child(editor, QPushButton, "resetCurvesButton")
    qtbot.mouseClick(reset, Qt.MouseButton.LeftButton)

    assert emitted == [
        replace(
            starting_patch,
            envelope=replace(
                starting_patch.envelope,
                attack_curve=0.0,
                decay_curve=0.0,
                release_curve=0.0,
            ),
        )
    ]


def test_selected_curve_entry_commits_only_selected_stage_without_feedback(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    # Ignoring graph selection or feeding set_envelope back into commits would change twice.
    emitted: list[SynthPatch] = []
    editor.patch_commit_requested.connect(emitted.append)
    graph = child(editor, EnvelopeGraph, "envelopeGraph")
    curve_entry = child(editor, EnvelopeValueEntry, "curveEntry")

    graph.stage_selected.emit("decay")
    assert graph.selected_stage is CurveStage.DECAY
    assert curve_entry.text() == "-0.3"
    replace_entry_text(qtbot, curve_entry, "0.375")
    qtbot.keyPress(curve_entry, Qt.Key.Key_Return)

    assert emitted == [
        replace(
            starting_patch,
            envelope=replace(starting_patch.envelope, decay_curve=0.375),
        )
    ]
    assert graph.envelope.decay_curve == 0.375

    graph.select_stage(CurveStage.ATTACK)
    graph.select_stage(CurveStage.RELEASE)
    assert len(emitted) == 1


@pytest.mark.parametrize("stage", ["attack", "release"])
def test_invalid_selected_curve_text_stays_local_until_escape_restores_authored_stage(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
    stage: str,
) -> None:
    # Caching an invalid field or graph preview would make Escape restore non-authored state.
    emitted: list[SynthPatch] = []
    editor.patch_commit_requested.connect(emitted.append)
    graph = child(editor, EnvelopeGraph, "envelopeGraph")
    curve_entry = child(editor, EnvelopeValueEntry, "curveEntry")
    graph.stage_selected.emit(stage)

    replace_entry_text(qtbot, curve_entry, "not a curve")
    qtbot.keyPress(curve_entry, Qt.Key.Key_Return)

    assert curve_entry.text() == "not a curve"
    assert emitted == []

    qtbot.keyPress(curve_entry, Qt.Key.Key_Escape)

    authored = getattr(starting_patch.envelope, f"{stage}_curve")
    assert curve_entry.text() == f"{authored:g}"
    assert getattr(graph.envelope, f"{stage}_curve") == authored
    assert child(editor, QLabel, "patchStatusLabel").text() == "Active"
    assert emitted == []


def test_real_handle_selection_clears_prior_curve_text_error_and_tracks_new_stage(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    # Reading selection from the graph after its event mutates it loses the prior field owner.
    emitted: list[SynthPatch] = []
    cleared: list[None] = []
    editor.patch_commit_requested.connect(emitted.append)
    editor.validation_cleared.connect(lambda: cleared.append(None))
    graph = child(editor, EnvelopeGraph, "envelopeGraph")
    curve_entry = child(editor, EnvelopeValueEntry, "curveEntry")
    error_label = child(editor, QLabel, "envelopeFieldError")
    status = child(editor, QLabel, "patchStatusLabel")
    decay_handle = child(graph, QWidget, "decayCurveHandle")

    replace_entry_text(qtbot, curve_entry, "not a curve")
    qtbot.keyPress(curve_entry, Qt.Key.Key_Return)
    assert error_label.text()
    assert status.text() == "Editing"

    decay_handle.setFocus(Qt.FocusReason.TabFocusReason)
    QApplication.processEvents()

    assert decay_handle.hasFocus()
    assert graph.selected_stage is CurveStage.DECAY
    assert curve_entry.text() == "-0.3"
    assert curve_entry.property("validationState") is None
    assert error_label.text() == ""
    assert status.text() == "Active"
    assert cleared == [None]

    curve_entry.setFocus(Qt.FocusReason.TabFocusReason)
    qtbot.keyPress(curve_entry, Qt.Key.Key_Escape)

    assert graph.selected_stage is CurveStage.DECAY
    assert graph.envelope == starting_patch.envelope
    assert curve_entry.text() == "-0.3"
    assert error_label.text() == ""
    assert status.text() == "Active"
    assert emitted == []


def test_same_patch_refresh_preserves_uncommitted_and_invalid_field_text(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    # Treating every 34 ms state refresh as an acknowledgment would erase active typing.
    attack = child(editor, EnvelopeValueEntry, "attackEntry")
    status = child(editor, QLabel, "patchStatusLabel")

    replace_entry_text(qtbot, attack, "250")
    editor.set_patch_state(starting_patch, PatchApplyState.APPLIED)
    assert attack.text() == "250"
    assert status.text() == "Editing"

    qtbot.keyPress(attack, Qt.Key.Key_Escape)
    replace_entry_text(qtbot, attack, "not complete")
    qtbot.keyPress(attack, Qt.Key.Key_Return)
    editor.set_patch_state(starting_patch, PatchApplyState.APPLIED)

    assert attack.text() == "not complete"
    assert attack.property("validationState") == "error"
    assert status.text() == "Editing"


def test_synchronous_acknowledgment_canonicalizes_every_field_and_graph_handle(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    # Updating only the source control would leave sibling fields and handles on stale draft state.
    emitted: list[SynthPatch] = []

    def acknowledge(candidate: SynthPatch) -> None:
        emitted.append(candidate)
        editor.set_patch_state(candidate, PatchApplyState.APPLIED)

    editor.patch_commit_requested.connect(acknowledge)
    attack = child(editor, EnvelopeValueEntry, "attackEntry")
    replace_entry_text(qtbot, attack, "0.250 s")
    qtbot.keyPress(attack, Qt.Key.Key_Return)

    candidate = replace(
        starting_patch,
        envelope=replace(starting_patch.envelope, attack_seconds=0.250),
    )
    assert emitted == [candidate]
    assert attack.text() == "250 ms"
    assert child(editor, EnvelopeValueEntry, "decayEntry").text() == "400 ms"
    assert child(editor, EnvelopeValueEntry, "sustainEntry").text() == "-9 dB"
    assert child(editor, EnvelopeValueEntry, "releaseEntry").text() == "850 ms"
    assert child(editor, EnvelopeValueEntry, "curveEntry").text() == "0.2"
    assert child(editor, EnvelopeGraph, "envelopeGraph").envelope == candidate.envelope
    assert child(editor, QLabel, "patchStatusLabel").text() == "Active"


def test_same_value_synchronous_acknowledgment_is_not_mistaken_for_timer_refresh(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    # Equality with prior authored state must still acknowledge an intentional no-op commit.
    emitted: list[SynthPatch] = []

    def acknowledge(candidate: SynthPatch) -> None:
        emitted.append(candidate)
        editor.set_patch_state(candidate, PatchApplyState.APPLIED)

    editor.patch_commit_requested.connect(acknowledge)
    attack = child(editor, EnvelopeValueEntry, "attackEntry")
    replace_entry_text(qtbot, attack, "125.0 ms")
    qtbot.keyPress(attack, Qt.Key.Key_Return)

    assert emitted == [starting_patch]
    assert attack.text() == "125 ms"
    assert child(editor, QLabel, "patchStatusLabel").text() == "Active"


def test_unacknowledged_same_value_commit_cannot_consume_a_later_routine_refresh(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    # Keeping an unconsumed marker after signal dispatch erases newer text and graph drafts.
    emitted: list[SynthPatch] = []
    editor.patch_commit_requested.connect(emitted.append)
    attack = child(editor, EnvelopeValueEntry, "attackEntry")
    graph = child(editor, EnvelopeGraph, "envelopeGraph")
    error_label = child(editor, QLabel, "envelopeFieldError")
    status = child(editor, QLabel, "patchStatusLabel")

    replace_entry_text(qtbot, attack, "125.0 ms")
    qtbot.keyPress(attack, Qt.Key.Key_Return)
    assert emitted == [starting_patch]

    replace_entry_text(qtbot, attack, "not complete")
    qtbot.keyPress(attack, Qt.Key.Key_Return)
    graph.curve_previewed.emit("release", -0.8)
    editor.set_patch_state(starting_patch, PatchApplyState.APPLIED)

    assert attack.text() == "not complete"
    assert attack.property("validationState") == "error"
    assert "Attack" in error_label.text()
    assert graph.envelope.release_curve == -0.8
    assert status.text() == "Editing"
    assert emitted == [starting_patch]


def test_nested_unacknowledged_commit_restores_outer_synchronous_ack_scope(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    # Clearing a nested marker instead of restoring its parent loses the outer acknowledgment.
    emitted: list[SynthPatch] = []
    attack = child(editor, EnvelopeValueEntry, "attackEntry")
    decay = child(editor, EnvelopeValueEntry, "decayEntry")

    def reenter_once(candidate: SynthPatch) -> None:
        emitted.append(candidate)
        if len(emitted) != 1:
            return
        replace_entry_text(qtbot, decay, "750 ms")
        qtbot.keyPress(decay, Qt.Key.Key_Return)
        editor.set_patch_state(candidate, PatchApplyState.APPLIED)

    editor.patch_commit_requested.connect(reenter_once)
    replace_entry_text(qtbot, attack, "125.0 ms")
    qtbot.keyPress(attack, Qt.Key.Key_Return)

    nested = replace(
        starting_patch,
        envelope=replace(starting_patch.envelope, decay_seconds=0.750),
    )
    assert emitted == [starting_patch, nested]
    assert child(editor, EnvelopeGraph, "envelopeGraph").envelope == starting_patch.envelope
    assert attack.text() == "125 ms"
    assert decay.text() == "400 ms"
    assert child(editor, QLabel, "patchStatusLabel").text() == "Active"


def test_pending_status_outranks_newer_local_draft_and_invalid_text(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    # Displaying Editing ahead of Pending would hide controller/audio application latency.
    attack = child(editor, EnvelopeValueEntry, "attackEntry")
    status = child(editor, QLabel, "patchStatusLabel")
    replace_entry_text(qtbot, attack, "not complete")
    qtbot.keyPress(attack, Qt.Key.Key_Return)

    editor.set_patch_state(starting_patch, PatchApplyState.PENDING)

    assert attack.text() == "not complete"
    assert status.text() == "Pending"

    qtbot.keyPress(attack, Qt.Key.Key_Escape)
    editor.set_patch_state(starting_patch, PatchApplyState.APPLIED)
    child(editor, EnvelopeGraph, "envelopeGraph").curve_previewed.emit("decay", 0.75)
    editor.set_patch_state(starting_patch, PatchApplyState.PENDING)
    assert status.text() == "Pending"


def test_graph_draft_is_editing_only_while_applied_and_revert_returns_active(
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    # Failing to clear the reverted stage marker would leave a clean editor stuck on Editing.
    graph = child(editor, EnvelopeGraph, "envelopeGraph")
    status = child(editor, QLabel, "patchStatusLabel")
    emitted: list[SynthPatch] = []
    editor.patch_commit_requested.connect(emitted.append)

    graph.curve_previewed.emit("decay", 0.75)
    assert status.text() == "Editing"
    editor.set_patch_state(starting_patch, PatchApplyState.PENDING)
    assert status.text() == "Pending"
    editor.set_patch_state(starting_patch, PatchApplyState.APPLIED)
    assert status.text() == "Editing"

    graph.curve_reverted.emit("decay")

    assert graph.envelope == starting_patch.envelope
    assert status.text() == "Active"
    assert emitted == []


def test_discarding_draft_replaces_invalid_text_graph_preview_and_local_error(
    qtbot,
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    # Keeping any local state during Load would merge the old draft into the loaded patch.
    attack = child(editor, EnvelopeValueEntry, "attackEntry")
    graph = child(editor, EnvelopeGraph, "envelopeGraph")
    replace_entry_text(qtbot, attack, "not complete")
    qtbot.keyPress(attack, Qt.Key.Key_Return)
    graph.curve_previewed.emit("release", -0.8)
    loaded_patch = replace(
        starting_patch,
        envelope=replace(
            starting_patch.envelope,
            attack_seconds=0.250,
            release_curve=-0.125,
        ),
        output_gain_dbfs=-24.0,
    )

    editor.set_patch_state(
        loaded_patch,
        PatchApplyState.APPLIED,
        discard_draft=True,
    )

    assert attack.text() == "250 ms"
    assert attack.property("validationState") is None
    assert graph.envelope == loaded_patch.envelope
    assert child(editor, EnvelopeValueEntry, "curveEntry").text() == "0.2"
    assert child(editor, QLabel, "outputFact").text() == "-24 dBFS"
    assert child(editor, QLabel, "envelopeFieldError").text() == ""
    assert child(editor, QLabel, "patchStatusLabel").text() == "Active"


def test_programmatic_synchronization_and_discard_emit_no_editor_intent_or_validation(
    editor: EnvelopeEditor,
    starting_patch: SynthPatch,
) -> None:
    # Programmatic state application must not loop back into controller or file actions.
    observed: list[tuple[str, object | None]] = []
    editor.patch_commit_requested.connect(lambda patch: observed.append(("patch", patch)))
    editor.validation_failed.connect(lambda error: observed.append(("failed", error)))
    editor.validation_cleared.connect(lambda: observed.append(("cleared", None)))
    editor.load_requested.connect(lambda: observed.append(("load", None)))
    editor.save_requested.connect(lambda: observed.append(("save", None)))
    external_patch = replace(
        starting_patch,
        envelope=replace(starting_patch.envelope, sustain_db=-15.0),
    )

    editor.set_patch_state(external_patch, PatchApplyState.PENDING)
    editor.set_patch_state(external_patch, PatchApplyState.APPLIED, discard_draft=True)
    editor.discard_draft()

    assert observed == []
    assert child(editor, EnvelopeValueEntry, "sustainEntry").text() == "-15 dB"
    assert child(editor, EnvelopeGraph, "envelopeGraph").envelope == external_patch.envelope
    assert child(editor, QLabel, "patchStatusLabel").text() == "Active"


def test_file_buttons_emit_one_intent_each_and_editor_owns_no_shortcuts(
    qtbot,
    editor: EnvelopeEditor,
) -> None:
    # Opening dialogs/files or creating shortcuts here would steal window ownership.
    intents: list[str] = []
    patches: list[SynthPatch] = []
    editor.load_requested.connect(lambda: intents.append("load"))
    editor.save_requested.connect(lambda: intents.append("save"))
    editor.patch_commit_requested.connect(patches.append)

    qtbot.mouseClick(child(editor, QPushButton, "loadPatchButton"), Qt.MouseButton.LeftButton)
    qtbot.mouseClick(child(editor, QPushButton, "savePatchButton"), Qt.MouseButton.LeftButton)

    assert intents == ["load", "save"]
    assert patches == []
    assert editor.findChildren(QShortcut) == []
