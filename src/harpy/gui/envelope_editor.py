"""Local-draft envelope inspector."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from harpy.gui.envelope_entry import EnvelopeFieldKind, EnvelopeValueEntry
from harpy.gui.envelope_graph import CurveStage, EnvelopeGraph
from harpy.gui.workbench_controller import PatchApplyState
from harpy.synth.models import (
    RenderConfig,
    SynthPatch,
    validate_renderable_patch,
)

_NUMERIC_FIELDS = {
    "attack_seconds": ("Attack", "attackEntry", EnvelopeFieldKind.DURATION),
    "decay_seconds": ("Decay", "decayEntry", EnvelopeFieldKind.DURATION),
    "sustain_db": ("Sustain", "sustainEntry", EnvelopeFieldKind.DECIBELS),
    "release_seconds": ("Release", "releaseEntry", EnvelopeFieldKind.DURATION),
}


class EnvelopeEditor(QFrame):
    """Compose envelope authoring controls around one local draft."""

    patch_commit_requested = Signal(object)
    validation_failed = Signal(str)
    validation_cleared = Signal()
    load_requested = Signal()
    save_requested = Signal()

    def __init__(
        self,
        render: RenderConfig,
        patch: SynthPatch,
        parent: QWidget | None = None,
    ) -> None:
        if not isinstance(render, RenderConfig):
            raise ValueError("render must be a RenderConfig")
        if not isinstance(patch, SynthPatch):
            raise ValueError("patch must be a SynthPatch")
        validate_renderable_patch(patch, render)
        super().__init__(parent)

        self._render = render
        self._authored_patch = patch
        self._draft_envelope = patch.envelope
        self._apply_state = PatchApplyState.APPLIED
        self._text_dirty_fields: set[str] = set()
        self._awaiting_ack_patch: SynthPatch | None = None
        self._error_field: str | None = None
        self._error_widget: QWidget | None = None

        self.setObjectName("envelopeEditor")
        self.setFixedWidth(320)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        heading = QLabel("Envelope")
        heading.setObjectName("envelopeHeading")
        self._patch_status = QLabel()
        self._patch_status.setObjectName("patchStatusLabel")
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.addWidget(heading)
        status_row.addStretch(1)
        status_row.addWidget(self._patch_status)

        self._graph = EnvelopeGraph(render)
        self._graph.setObjectName("envelopeGraph")
        self._graph.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        field_grid = QGridLayout()
        field_grid.setContentsMargins(0, 0, 0, 0)
        field_grid.setHorizontalSpacing(8)
        field_grid.setVerticalSpacing(5)
        self._entries: dict[str, EnvelopeValueEntry] = {}
        for row, (field_name, (label_text, object_name, kind)) in enumerate(
            _NUMERIC_FIELDS.items()
        ):
            label = QLabel(label_text)
            entry = EnvelopeValueEntry(label_text, kind)
            entry.setObjectName(object_name)
            entry.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._configure_accessibility(entry, field_name)
            field_grid.addWidget(label, row, 0)
            field_grid.addWidget(entry, row, 1)
            self._entries[field_name] = entry

        curve_label = QLabel("Curve")
        self._curve_entry = EnvelopeValueEntry("Curve", EnvelopeFieldKind.CURVATURE)
        self._curve_entry.setObjectName("curveEntry")
        self._curve_entry.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._curve_entry.setAccessibleName("Selected curve")
        self._curve_entry.setAccessibleDescription("Curvature from -1 to 1; 0 is linear.")
        curve_row = QHBoxLayout()
        curve_row.setContentsMargins(0, 0, 0, 0)
        curve_row.setSpacing(8)
        curve_row.addWidget(curve_label)
        curve_row.addWidget(self._curve_entry, 1)

        self._reset_curves = QPushButton("Reset curves to linear")
        self._reset_curves.setObjectName("resetCurvesButton")

        self._field_error = QLabel()
        self._field_error.setObjectName("envelopeFieldError")
        self._field_error.setWordWrap(True)
        self._field_error.hide()

        facts = QGridLayout()
        facts.setContentsMargins(0, 0, 0, 0)
        facts.setHorizontalSpacing(12)
        facts.setVerticalSpacing(2)
        oscillator_name = QLabel("Oscillator")
        output_name = QLabel("Output")
        oscillator_name.setObjectName("envelopeFactName")
        output_name.setObjectName("envelopeFactName")
        self._oscillator_fact = QLabel()
        self._oscillator_fact.setObjectName("oscillatorFact")
        self._output_fact = QLabel()
        self._output_fact.setObjectName("outputFact")
        facts.addWidget(oscillator_name, 0, 0)
        facts.addWidget(output_name, 0, 1)
        facts.addWidget(self._oscillator_fact, 1, 0)
        facts.addWidget(self._output_fact, 1, 1)

        self._load_button = QPushButton("Load")
        self._load_button.setObjectName("loadPatchButton")
        self._save_button = QPushButton("Save As…")
        self._save_button.setObjectName("savePatchButton")
        file_row = QHBoxLayout()
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.setSpacing(8)
        file_row.addWidget(self._load_button)
        file_row.addWidget(self._save_button)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 9, 10, 9)
        root.setSpacing(7)
        root.addLayout(status_row)
        root.addWidget(self._graph, 1)
        root.addLayout(field_grid)
        root.addLayout(curve_row)
        root.addWidget(self._reset_curves)
        root.addWidget(self._field_error)
        root.addLayout(facts)
        root.addLayout(file_row)

        for field_name, entry in self._entries.items():
            entry.textEdited.connect(
                lambda _text, field_name=field_name: self._mark_text_dirty(field_name)
            )
            entry.value_commit_requested.connect(
                lambda value, field_name=field_name, entry=entry: self._commit_entry_value(
                    field_name,
                    value,
                    entry,
                )
            )
            entry.validation_failed.connect(
                lambda message, field_name=field_name, entry=entry: self._entry_failed(
                    field_name,
                    entry,
                    message,
                )
            )
            entry.draft_reverted.connect(
                lambda field_name=field_name: self._revert_field(field_name)
            )
        self._curve_entry.textEdited.connect(self._mark_curve_text_dirty)
        self._curve_entry.value_commit_requested.connect(self._commit_selected_curve)
        self._curve_entry.validation_failed.connect(self._selected_curve_failed)
        self._curve_entry.draft_reverted.connect(self._revert_selected_curve)
        self._graph.curve_previewed.connect(self._preview_curve)
        self._graph.curve_commit_requested.connect(self._commit_graph_curve)
        self._graph.curve_reverted.connect(self._revert_curve)
        self._graph.stage_selected.connect(self._select_curve_stage)
        self._reset_curves.clicked.connect(self._reset_all_curves)
        self._load_button.clicked.connect(lambda _checked=False: self.load_requested.emit())
        self._save_button.clicked.connect(lambda _checked=False: self.save_requested.emit())

        self._apply_style()
        self._accept_patch(patch)
        self._update_status()

    def set_patch_state(
        self,
        patch: SynthPatch,
        apply_state: PatchApplyState,
        *,
        discard_draft: bool = False,
    ) -> None:
        """Apply controller-authored state without erasing routine local editing."""

        if not isinstance(patch, SynthPatch):
            raise ValueError("patch must be a SynthPatch")
        if not isinstance(apply_state, PatchApplyState):
            raise ValueError("apply_state must be a PatchApplyState")
        validate_renderable_patch(patch, self._render)

        self._apply_state = apply_state
        is_acknowledgment = (
            self._awaiting_ack_patch is not None and patch == self._awaiting_ack_patch
        )
        if discard_draft or is_acknowledgment or patch != self._authored_patch:
            self._awaiting_ack_patch = None
            self._accept_patch(patch)
        self._update_status()

    def discard_draft(self) -> None:
        """Restore the current controller-authored patch without emitting intents."""

        self._awaiting_ack_patch = None
        self._accept_patch(self._authored_patch)
        self._update_status()

    def _configure_accessibility(
        self,
        entry: EnvelopeValueEntry,
        field_name: str,
    ) -> None:
        label = _NUMERIC_FIELDS[field_name][0]
        if field_name.endswith("_seconds"):
            entry.setAccessibleName(f"{label} duration")
            entry.setAccessibleDescription(
                "Duration in milliseconds or seconds; at least one frame."
            )
        else:
            entry.setAccessibleName("Sustain level")
            entry.setAccessibleDescription("Level in decibels at or below 0 dB.")

    def _accept_patch(self, patch: SynthPatch) -> None:
        self._authored_patch = patch
        self._draft_envelope = patch.envelope
        self._text_dirty_fields.clear()
        self._clear_all_error_state()
        for field_name, entry in self._entries.items():
            entry.set_exact_value(getattr(patch.envelope, field_name))
        self._graph.set_envelope(patch.envelope)
        self._render_selected_curve(canonical=True)
        self._oscillator_fact.setText(patch.oscillator.type.value.title())
        self._output_fact.setText(f"{patch.output_gain_dbfs:g} dBFS")

    def _mark_text_dirty(self, field_name: str) -> None:
        self._text_dirty_fields.add(field_name)
        self._update_status()

    def _mark_curve_text_dirty(self, _text: str) -> None:
        self._text_dirty_fields.add(self._selected_curve_field())
        self._update_status()

    def _commit_entry_value(
        self,
        field_name: str,
        value: float,
        source_entry: EnvelopeValueEntry,
    ) -> None:
        self._commit_candidate({field_name: value}, field_name, source_entry)

    def _commit_selected_curve(self, value: float) -> None:
        field_name = self._selected_curve_field()
        self._commit_candidate({field_name: value}, field_name, self._curve_entry)

    def _preview_curve(self, stage_name: str, value: float) -> None:
        field_name = self._curve_field(stage_name)
        try:
            draft = replace(self._draft_envelope, **{field_name: value})
        except ValueError as error:
            self._show_field_error(
                field_name,
                self._field_message(field_name, error),
                self._curve_handle(stage_name),
            )
            return
        self._draft_envelope = draft
        self._graph.set_envelope(draft)
        if self._selected_curve_field() == field_name:
            self._render_selected_curve()
        self._update_status()

    def _commit_graph_curve(self, stage_name: str, value: float) -> None:
        field_name = self._curve_field(stage_name)
        self._commit_candidate(
            {field_name: value},
            field_name,
            None,
            error_widget=self._curve_handle(stage_name),
        )

    def _reset_all_curves(self, _checked: bool = False) -> None:
        self._commit_candidate(
            {"attack_curve": 0.0, "decay_curve": 0.0, "release_curve": 0.0},
            self._selected_curve_field(),
            None,
            error_widget=self._curve_handle(self._graph.selected_stage.value),
        )

    def _commit_candidate(
        self,
        changes: dict[str, float],
        field_name: str,
        source_entry: EnvelopeValueEntry | None,
        *,
        error_widget: QWidget | None = None,
    ) -> None:
        try:
            candidate_envelope = replace(self._draft_envelope, **changes)
            candidate_patch = replace(self._authored_patch, envelope=candidate_envelope)
            validate_renderable_patch(candidate_patch, self._render)
        except ValueError as error:
            message = self._field_message(field_name, error)
            if source_entry is not None:
                source_entry.mark_commit_rejected(message)
            else:
                self._show_field_error(field_name, message, error_widget)
            return

        self._draft_envelope = candidate_envelope
        self._graph.set_envelope(candidate_envelope)
        if source_entry is not None:
            source_entry.accept_proposed_value(changes[field_name])
        elif any(name.endswith("_curve") for name in changes):
            self._render_selected_curve()
        self._text_dirty_fields.difference_update(changes)
        self._clear_field_error(field_name)
        self.validation_cleared.emit()
        self._awaiting_ack_patch = candidate_patch
        self._update_status()
        self.patch_commit_requested.emit(candidate_patch)

    def _entry_failed(
        self,
        field_name: str,
        entry: EnvelopeValueEntry,
        message: str,
    ) -> None:
        self._show_field_error(field_name, message, entry)

    def _selected_curve_failed(self, message: str) -> None:
        self._show_field_error(
            self._selected_curve_field(),
            message,
            self._curve_entry,
        )

    def _show_field_error(
        self,
        field_name: str,
        message: str,
        widget: QWidget | None,
    ) -> None:
        if self._error_widget is not None and self._error_widget is not widget:
            self._set_validation_state(self._error_widget, None)
        self._error_field = field_name
        self._error_widget = widget
        if widget is not None:
            self._set_validation_state(widget, "error")
        self._field_error.setText(message)
        self._field_error.show()
        self._update_status()
        self.validation_failed.emit(message)

    def _clear_field_error(self, field_name: str) -> bool:
        if self._error_field != field_name:
            return False
        if self._error_widget is not None:
            self._set_validation_state(self._error_widget, None)
        self._error_field = None
        self._error_widget = None
        self._field_error.clear()
        self._field_error.hide()
        return True

    def _clear_all_error_state(self) -> None:
        if self._error_widget is not None:
            self._set_validation_state(self._error_widget, None)
        self._error_field = None
        self._error_widget = None
        self._field_error.clear()
        self._field_error.hide()

    def _revert_field(self, field_name: str) -> None:
        self._awaiting_ack_patch = None
        self._text_dirty_fields.discard(field_name)
        self._draft_envelope = replace(
            self._draft_envelope,
            **{field_name: getattr(self._authored_patch.envelope, field_name)},
        )
        self._entries[field_name].set_exact_value(
            getattr(self._authored_patch.envelope, field_name)
        )
        self._graph.set_envelope(self._draft_envelope)
        cleared = self._clear_field_error(field_name)
        self._update_status()
        if cleared:
            self.validation_cleared.emit()

    def _revert_selected_curve(self) -> None:
        self._revert_curve(self._graph.selected_stage.value)

    def _revert_curve(self, stage_name: str) -> None:
        field_name = self._curve_field(stage_name)
        self._awaiting_ack_patch = None
        self._text_dirty_fields.discard(field_name)
        self._draft_envelope = replace(
            self._draft_envelope,
            **{field_name: getattr(self._authored_patch.envelope, field_name)},
        )
        self._graph.set_envelope(self._draft_envelope)
        if self._selected_curve_field() == field_name:
            self._render_selected_curve(canonical=True)
        cleared = self._clear_field_error(field_name)
        self._update_status()
        if cleared:
            self.validation_cleared.emit()

    def _select_curve_stage(self, stage_name: str) -> None:
        previous_field = self._selected_curve_field()
        stage = CurveStage(stage_name)
        self._graph.select_stage(stage)
        selected_field = self._selected_curve_field()
        if selected_field != previous_field:
            self._text_dirty_fields.discard(previous_field)
            self._clear_field_error(previous_field)
        self._render_selected_curve()
        self._update_status()

    def _render_selected_curve(self, *, canonical: bool = False) -> None:
        field_name = self._selected_curve_field()
        authored_value = getattr(self._authored_patch.envelope, field_name)
        draft_value = getattr(self._draft_envelope, field_name)
        self._curve_entry.set_exact_value(authored_value)
        if not canonical and draft_value != authored_value:
            self._curve_entry.setText(_format_curvature(draft_value))

    def _selected_curve_field(self) -> str:
        return f"{self._graph.selected_stage.value}_curve"

    def _curve_field(self, stage_name: str) -> str:
        return f"{CurveStage(stage_name).value}_curve"

    def _curve_handle(self, stage_name: str) -> QWidget | None:
        return self._graph.findChild(QWidget, f"{CurveStage(stage_name).value}CurveHandle")

    def _field_message(self, field_name: str, error: ValueError) -> str:
        labels = {
            "attack_seconds": "Attack",
            "decay_seconds": "Decay",
            "sustain_db": "Sustain",
            "release_seconds": "Release",
            "attack_curve": "Attack curve",
            "decay_curve": "Decay curve",
            "release_curve": "Release curve",
        }
        detail = str(error).rstrip(".")
        return f"{labels[field_name]}: {detail}."

    def _update_status(self) -> None:
        if self._apply_state is PatchApplyState.PENDING:
            text = "Pending"
            state = "pending"
        elif self._has_local_draft():
            text = "Editing"
            state = "editing"
        else:
            text = "Active"
            state = "active"
        self._patch_status.setText(text)
        self._set_dynamic_property(self._patch_status, "statusState", state)

    def _has_local_draft(self) -> bool:
        return bool(
            self._text_dirty_fields
            or self._error_field is not None
            or self._awaiting_ack_patch is not None
            or self._draft_envelope != self._authored_patch.envelope
        )

    def _set_validation_state(self, widget: QWidget, state: str | None) -> None:
        self._set_dynamic_property(widget, "validationState", state)
        widget.update()

    def _set_dynamic_property(
        self,
        widget: QWidget,
        name: str,
        value: str | None,
    ) -> None:
        widget.setProperty(name, value)
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QFrame#envelopeEditor {
                background: #181e27;
                border: 1px solid #2a3441;
                border-radius: 8px;
                color: #e8edf4;
            }
            QFrame#envelopeEditor QLabel {
                background: transparent;
                border: none;
                color: #c9d2dd;
                font-size: 12px;
            }
            QLabel#envelopeHeading { color: #f1f4f8; font-size: 14px; font-weight: 700; }
            QLabel#patchStatusLabel[statusState="pending"] { color: #bd8cff; }
            QLabel#patchStatusLabel[statusState="editing"] { color: #69dcff; }
            QLabel#patchStatusLabel[statusState="active"] { color: #aab7c8; }
            QLabel#envelopeFieldError { color: #ff6b72; }
            QLabel#envelopeFactName { color: #95a2b2; font-size: 11px; }
            QLabel#oscillatorFact, QLabel#outputFact {
                color: #f1f4f8;
                font-family: monospace;
            }
            QLineEdit {
                background: #0d1117;
                border: 1px solid #465364;
                border-radius: 4px;
                color: #f4f7fb;
                font-family: monospace;
                padding: 4px 6px;
            }
            QLineEdit:focus { border: 2px solid #65d8ff; }
            QLineEdit[validationState="error"] { border: 2px solid #ff6b72; }
            QPushButton {
                background: #252e3a;
                border: 1px solid #3a4655;
                border-radius: 5px;
                color: #eef3f8;
                min-height: 28px;
                padding: 3px 9px;
            }
            QPushButton:hover, QPushButton:focus { border-color: #65d8ff; }
            """
        )


def _format_curvature(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"
