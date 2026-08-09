"""Local-draft graph-native envelope inspector."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Signal
from PySide6.QtGui import QAccessible, QAccessibleStateChangeEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from harpy.gui.envelope_graph import EnvelopeGraph
from harpy.gui.workbench_controller import PatchApplyState
from harpy.synth.models import (
    EnvelopeConfig,
    RenderConfig,
    SynthPatch,
    validate_renderable_patch,
)

_FIELD_LABELS = {
    "attack_seconds": "Attack",
    "decay_seconds": "Decay",
    "sustain_db": "Sustain",
    "release_seconds": "Release",
    "attack_curve": "Attack curve",
    "decay_curve": "Decay curve",
    "release_curve": "Release curve",
}


class EnvelopeEditor(QFrame):
    """Compose envelope authoring around one graph-owned local draft."""

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
        self._last_status_state: str | None = None

        self.setObjectName("envelopeEditor")
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

        self._reset_envelope_button = QPushButton("Reset")
        self._reset_envelope_button.setObjectName("resetEnvelopeButton")

        self._field_error = QLabel()
        self._field_error.setObjectName("envelopeFieldError")
        self._field_error.setWordWrap(True)
        self._field_error.hide()

        oscillator_name = QLabel("Oscillator")
        oscillator_name.setObjectName("envelopeFactName")
        output_name = QLabel("Output")
        output_name.setObjectName("envelopeFactName")
        self._oscillator_fact = QLabel()
        self._oscillator_fact.setObjectName("oscillatorFact")
        self._output_fact = QLabel()
        self._output_fact.setObjectName("outputFact")
        oscillator_fact = QVBoxLayout()
        oscillator_fact.setContentsMargins(0, 0, 0, 0)
        oscillator_fact.setSpacing(2)
        oscillator_fact.addWidget(oscillator_name)
        oscillator_fact.addWidget(self._oscillator_fact)
        output_fact = QVBoxLayout()
        output_fact.setContentsMargins(0, 0, 0, 0)
        output_fact.setSpacing(2)
        output_fact.addWidget(output_name)
        output_fact.addWidget(self._output_fact)
        facts = QHBoxLayout()
        facts.setContentsMargins(0, 0, 0, 0)
        facts.setSpacing(12)
        facts.addLayout(oscillator_fact, 1)
        facts.addLayout(output_fact, 1)

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
        root.addWidget(self._reset_envelope_button)
        root.addWidget(self._field_error)
        root.addLayout(facts)
        root.addLayout(file_row)

        self._load_button.clicked.connect(lambda _checked=False: self.load_requested.emit())
        self._save_button.clicked.connect(lambda _checked=False: self.save_requested.emit())

        self._apply_style()
        self._accept_patch(patch)
        self._update_status()

        self._graph.field_previewed.connect(self._preview_field)
        self._graph.field_commit_requested.connect(self._commit_field)
        self._graph.field_reverted.connect(self._revert_field)
        self._graph.field_editing_changed.connect(self._set_field_editing)
        self._graph.field_validation_failed.connect(self._field_parse_failed)
        self._reset_envelope_button.clicked.connect(self._reset_envelope)

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

    def cancel_interactions(self) -> None:
        """Cancel graph gestures and exact entry without changing authored truth."""

        self._graph.cancel_interactions()

    def _accept_patch(self, patch: SynthPatch) -> None:
        self._graph.cancel_interactions(emit_revert=False)
        self._authored_patch = patch
        self._draft_envelope = patch.envelope
        self._text_dirty_fields.clear()
        self._clear_all_error_state()
        self._graph.set_envelope(patch.envelope)
        self._oscillator_fact.setText(patch.oscillator.type.value.title())
        self._output_fact.setText(f"{patch.output_gain_dbfs:g} dBFS")

    def _preview_field(self, field_name: str, value: float) -> None:
        try:
            candidate_envelope = replace(
                self._draft_envelope,
                **{field_name: value},
            )
            candidate_patch = replace(self._authored_patch, envelope=candidate_envelope)
            validate_renderable_patch(candidate_patch, self._render)
        except ValueError as error:
            self._reject_field(field_name, error, allow_exact=False)
            return

        self._draft_envelope = candidate_envelope
        self._graph.set_envelope(candidate_envelope)
        self._graph.accept_field_preview(field_name)
        if self._clear_field_error(field_name):
            self.validation_cleared.emit()
        self._update_status()

    def _commit_field(self, field_name: str, value: float) -> None:
        try:
            candidate_envelope = replace(
                self._draft_envelope,
                **{field_name: value},
            )
            candidate_patch = replace(self._authored_patch, envelope=candidate_envelope)
            validate_renderable_patch(candidate_patch, self._render)
        except ValueError as error:
            self._reject_field(field_name, error, allow_exact=True)
            return

        self._draft_envelope = candidate_envelope
        self._graph.set_envelope(candidate_envelope)
        self._graph.accept_exact_edit(field_name, value)
        self._text_dirty_fields.discard(field_name)
        if self._clear_field_error(field_name):
            self.validation_cleared.emit()
        self._emit_validated_candidate(candidate_patch)

    def _reject_field(
        self,
        field_name: str,
        error: ValueError,
        *,
        allow_exact: bool,
    ) -> None:
        message = self._field_message(field_name, error)
        if not allow_exact:
            self._graph.reject_field_preview(field_name)
        if allow_exact and self._graph.is_exact_editing(field_name):
            self._graph.reject_exact_edit(field_name, message)
            return
        self._graph.mark_field_error(field_name, message)
        self._show_field_error(field_name, message, self._field_widget(field_name))

    def _field_parse_failed(self, field_name: str, message: str) -> None:
        self._show_field_error(field_name, message, self._field_widget(field_name))

    def _revert_field(self, field_name: str) -> None:
        self._text_dirty_fields.discard(field_name)
        self._draft_envelope = replace(
            self._draft_envelope,
            **{field_name: getattr(self._authored_patch.envelope, field_name)},
        )
        self._graph.set_envelope(self._draft_envelope)
        cleared = self._clear_field_error(field_name)
        self._update_status()
        if cleared:
            self.validation_cleared.emit()

    def _set_field_editing(self, field_name: str, dirty: bool) -> None:
        if dirty:
            self._text_dirty_fields.add(field_name)
        else:
            self._text_dirty_fields.discard(field_name)
        self._update_status()

    def _reset_envelope(self, _checked: bool = False) -> None:
        self.cancel_interactions()
        default_envelope = EnvelopeConfig()
        candidate_patch = replace(self._authored_patch, envelope=default_envelope)
        try:
            validate_renderable_patch(candidate_patch, self._render)
        except ValueError as error:
            self._show_reset_error(f"Envelope: {str(error).rstrip('.')}.")
            return
        had_error = self._error_field is not None
        self._clear_all_error_state()
        if had_error:
            self.validation_cleared.emit()
        self._text_dirty_fields.clear()
        self._draft_envelope = default_envelope
        self._graph.set_envelope(default_envelope)
        if default_envelope == self._authored_patch.envelope:
            self._update_status()
            return
        self._emit_validated_candidate(candidate_patch)

    def _emit_validated_candidate(self, candidate_patch: SynthPatch) -> None:
        previous_ack_patch = self._awaiting_ack_patch
        self._awaiting_ack_patch = candidate_patch
        self._update_status()
        try:
            self.patch_commit_requested.emit(candidate_patch)
        finally:
            if self._awaiting_ack_patch is candidate_patch:
                self._awaiting_ack_patch = previous_ack_patch
                self._update_status()

    def _field_widget(self, field_name: str) -> QWidget | None:
        object_names = {
            "attack_seconds": "attackValueControl",
            "decay_seconds": "decayValueControl",
            "sustain_db": "sustainValueControl",
            "release_seconds": "releaseValueControl",
            "attack_curve": "attackCurveHandle",
            "decay_curve": "decayCurveHandle",
            "release_curve": "releaseCurveHandle",
        }
        return self._graph.findChild(QWidget, object_names[field_name])

    def _field_message(self, field_name: str, error: ValueError) -> str:
        return f"{_FIELD_LABELS[field_name]}: {str(error).rstrip('.')}."

    def _show_field_error(
        self,
        field_name: str,
        message: str,
        widget: QWidget | None,
    ) -> None:
        if self._error_field is not None and self._error_field != field_name:
            self._graph.clear_field_error(self._error_field)
        self._error_field = field_name
        self._error_widget = widget
        self._field_error.setText(message)
        self._field_error.show()
        self._update_status()
        self.validation_failed.emit(message)

    def _show_reset_error(self, message: str) -> None:
        if self._error_field is not None:
            self._graph.clear_field_error(self._error_field)
        self._error_field = "envelope"
        self._error_widget = self._reset_envelope_button
        self._set_validation_state(self._reset_envelope_button, "error")
        self._field_error.setText(message)
        self._field_error.show()
        self._update_status()
        self.validation_failed.emit(message)

    def _clear_field_error(self, field_name: str) -> bool:
        self._graph.clear_field_error(field_name)
        if self._error_field != field_name:
            return False
        if self._error_widget is self._reset_envelope_button:
            self._set_validation_state(self._reset_envelope_button, None)
        self._error_field = None
        self._error_widget = None
        self._field_error.clear()
        self._field_error.hide()
        return True

    def _clear_all_error_state(self) -> None:
        if self._error_field is not None:
            self._graph.clear_field_error(self._error_field)
        if self._error_widget is self._reset_envelope_button:
            self._set_validation_state(self._reset_envelope_button, None)
        self._error_field = None
        self._error_widget = None
        self._field_error.clear()
        self._field_error.hide()

    def _update_status(self) -> None:
        if self._apply_state is PatchApplyState.PENDING:
            text, state = "Pending", "pending"
        elif self._has_local_draft():
            text, state = "Editing", "editing"
        else:
            text, state = "Active", "active"
        if state == self._last_status_state:
            return
        self._last_status_state = state
        self._patch_status.setText(text)
        self._set_dynamic_property(self._patch_status, "statusState", state)
        QAccessible.updateAccessibility(
            QAccessibleStateChangeEvent(self._patch_status, QAccessible.State())
        )

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
            EnvelopeStageControl[validationState="error"],
            QWidget[validationState="error"] { border: 2px solid #ff6b72; }
            QPushButton {
                background: #252e3a;
                border: 1px solid #3a4655;
                border-radius: 5px;
                color: #eef3f8;
                min-height: 28px;
                padding: 3px 9px;
            }
            QPushButton:hover, QPushButton:focus { border-color: #65d8ff; }
            QPushButton[validationState="error"] { border: 2px solid #ff6b72; }
            """
        )
