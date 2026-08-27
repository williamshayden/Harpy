"""Strict canonical JSON codecs and path preflight for diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from harpy.envs.models import ControlState, PitchAction, TerminalReason
from harpy.learning.artifacts import canonical_json_bytes, decode_json_bytes
from harpy.learning.diagnostics import (
    DecisionDiagnostic,
    DiagnosticBundle,
    DiagnosticReport,
    DiagnosticReportInventory,
    EpisodeDiagnostic,
    EstimateDiagnostic,
    PerActionMetrics,
    SubmitErrorBand,
    SurvivalEntry,
)
from harpy.learning.models import JSONValue


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a string-keyed object")
    return value


def _exact_fields(mapping: Mapping[str, object], fields: set[str], field: str) -> None:
    if set(mapping) != fields:
        raise ValueError(f"{field} fields must be exactly {sorted(fields)}")


def _list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _integer(
    value: object,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return value


def _float(value: object, field: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be a finite number")
    return normalized


def _optional_float(value: object, field: str) -> float | None:
    return None if value is None else _float(value, field)


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a bool")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _action(value: object, field: str) -> PitchAction:
    normalized = _integer(value, field, minimum=0, maximum=len(PitchAction) - 1)
    return PitchAction(normalized)


def _optional_action(value: object, field: str) -> PitchAction | None:
    return None if value is None else _action(value, field)


def _optional_integer(value: object, field: str) -> int | None:
    return None if value is None else _integer(value, field, minimum=0)


def _terminal_reason(value: object) -> TerminalReason:
    if not isinstance(value, str):
        raise ValueError("terminal_reason must be a valid TerminalReason")
    try:
        return TerminalReason(value)
    except ValueError as error:
        raise ValueError("terminal_reason must be a valid TerminalReason") from error


def _controls_to_document(controls: ControlState) -> dict[str, JSONValue]:
    return {
        "octaves": controls.octaves,
        "semitones": controls.semitones,
        "cents": controls.cents,
    }


def _controls_from_document(value: object) -> ControlState:
    mapping = _mapping(value, "controls")
    _exact_fields(mapping, {"octaves", "semitones", "cents"}, "controls")
    return ControlState(
        octaves=_integer(mapping["octaves"], "octaves"),
        semitones=_integer(mapping["semitones"], "semitones"),
        cents=_integer(mapping["cents"], "cents"),
    )


def _estimate_to_document(estimate: EstimateDiagnostic) -> dict[str, JSONValue]:
    return {
        "estimated_candidate_cents": estimate.estimated_candidate_cents,
        "signed_error_cents": estimate.signed_error_cents,
        "absolute_error_cents": estimate.absolute_error_cents,
    }


def _estimate_from_document(value: object) -> EstimateDiagnostic:
    mapping = _mapping(value, "estimate")
    _exact_fields(
        mapping,
        {"estimated_candidate_cents", "signed_error_cents", "absolute_error_cents"},
        "estimate",
    )
    return EstimateDiagnostic(
        estimated_candidate_cents=_integer(
            mapping["estimated_candidate_cents"], "estimated_candidate_cents"
        ),
        signed_error_cents=_integer(mapping["signed_error_cents"], "signed_error_cents"),
        absolute_error_cents=_integer(
            mapping["absolute_error_cents"], "absolute_error_cents", minimum=0
        ),
    )


def _decision_to_document(decision: DecisionDiagnostic) -> dict[str, JSONValue]:
    return {
        "step": decision.step,
        "controls": _controls_to_document(decision.controls),
        "steps_remaining": decision.steps_remaining,
        "true_candidate_cents": decision.true_candidate_cents,
        "true_absolute_error_cents": decision.true_absolute_error_cents,
        "teacher_action": int(decision.teacher_action),
        "predicted_action": int(decision.predicted_action),
        "canonical_match": decision.canonical_match,
        "in_shortest_action_set": decision.in_shortest_action_set,
        "legal": decision.legal,
        "bound_blocked": decision.bound_blocked,
        "repeated_state_visit": decision.repeated_state_visit,
        "loop_transition": decision.loop_transition,
        "shortest_plan_length": decision.shortest_plan_length,
        "recoverable": decision.recoverable,
        "estimate": None if decision.estimate is None else _estimate_to_document(decision.estimate),
    }


def _decision_from_document(value: object) -> DecisionDiagnostic:
    mapping = _mapping(value, "decision")
    fields = {
        "step",
        "controls",
        "steps_remaining",
        "true_candidate_cents",
        "true_absolute_error_cents",
        "teacher_action",
        "predicted_action",
        "canonical_match",
        "in_shortest_action_set",
        "legal",
        "bound_blocked",
        "repeated_state_visit",
        "loop_transition",
        "shortest_plan_length",
        "recoverable",
        "estimate",
    }
    _exact_fields(mapping, fields, "decision")
    return DecisionDiagnostic(
        step=_integer(mapping["step"], "step", minimum=1),
        controls=_controls_from_document(mapping["controls"]),
        steps_remaining=_integer(mapping["steps_remaining"], "steps_remaining", minimum=1),
        true_candidate_cents=_integer(mapping["true_candidate_cents"], "true_candidate_cents"),
        true_absolute_error_cents=_integer(
            mapping["true_absolute_error_cents"], "true_absolute_error_cents", minimum=0
        ),
        teacher_action=_action(mapping["teacher_action"], "teacher_action"),
        predicted_action=_action(mapping["predicted_action"], "predicted_action"),
        canonical_match=_boolean(mapping["canonical_match"], "canonical_match"),
        in_shortest_action_set=_boolean(
            mapping["in_shortest_action_set"], "in_shortest_action_set"
        ),
        legal=_boolean(mapping["legal"], "legal"),
        bound_blocked=_boolean(mapping["bound_blocked"], "bound_blocked"),
        repeated_state_visit=_boolean(mapping["repeated_state_visit"], "repeated_state_visit"),
        loop_transition=_boolean(mapping["loop_transition"], "loop_transition"),
        shortest_plan_length=_integer(
            mapping["shortest_plan_length"], "shortest_plan_length", minimum=1
        ),
        recoverable=_boolean(mapping["recoverable"], "recoverable"),
        estimate=(
            None if mapping["estimate"] is None else _estimate_from_document(mapping["estimate"])
        ),
    )


def _episode_to_document(episode: EpisodeDiagnostic) -> dict[str, JSONValue]:
    return {
        "episode_index": episode.episode_index,
        "source_pitch_cents": episode.source_pitch_cents,
        "target_note_index": episode.target_note_index,
        "target_pitch_cents": episode.target_pitch_cents,
        "decisions": [_decision_to_document(item) for item in episode.decisions],
        "first_canonical_mismatch_step": episode.first_canonical_mismatch_step,
        "first_canonical_mismatch_action": (
            None
            if episode.first_canonical_mismatch_action is None
            else int(episode.first_canonical_mismatch_action)
        ),
        "first_shortest_divergence_step": episode.first_shortest_divergence_step,
        "first_shortest_divergence_action": (
            None
            if episode.first_shortest_divergence_action is None
            else int(episode.first_shortest_divergence_action)
        ),
        "first_loop_step": episode.first_loop_step,
        "first_unrecoverable_step": episode.first_unrecoverable_step,
        "legal_action_count": episode.legal_action_count,
        "bound_blocked_action_count": episode.bound_blocked_action_count,
        "repeated_state_visit_count": episode.repeated_state_visit_count,
        "loop_transition_count": episode.loop_transition_count,
        "terminal_reason": episode.terminal_reason.value,
        "final_absolute_error_cents": episode.final_absolute_error_cents,
        "action_count": episode.action_count,
        "excess_actions": episode.excess_actions,
    }


def _episode_from_document(value: object) -> EpisodeDiagnostic:
    mapping = _mapping(value, "episode diagnostic")
    fields = {
        "episode_index",
        "source_pitch_cents",
        "target_note_index",
        "target_pitch_cents",
        "decisions",
        "first_canonical_mismatch_step",
        "first_canonical_mismatch_action",
        "first_shortest_divergence_step",
        "first_shortest_divergence_action",
        "first_loop_step",
        "first_unrecoverable_step",
        "legal_action_count",
        "bound_blocked_action_count",
        "repeated_state_visit_count",
        "loop_transition_count",
        "terminal_reason",
        "final_absolute_error_cents",
        "action_count",
        "excess_actions",
    }
    _exact_fields(mapping, fields, "episode diagnostic")
    return EpisodeDiagnostic(
        episode_index=_integer(mapping["episode_index"], "episode_index", minimum=0),
        source_pitch_cents=_integer(mapping["source_pitch_cents"], "source_pitch_cents"),
        target_note_index=_integer(mapping["target_note_index"], "target_note_index", minimum=0),
        target_pitch_cents=_integer(mapping["target_pitch_cents"], "target_pitch_cents"),
        decisions=tuple(
            _decision_from_document(item) for item in _list(mapping["decisions"], "decisions")
        ),
        first_canonical_mismatch_step=_optional_integer(
            mapping["first_canonical_mismatch_step"], "first_canonical_mismatch_step"
        ),
        first_canonical_mismatch_action=_optional_action(
            mapping["first_canonical_mismatch_action"], "first_canonical_mismatch_action"
        ),
        first_shortest_divergence_step=_optional_integer(
            mapping["first_shortest_divergence_step"], "first_shortest_divergence_step"
        ),
        first_shortest_divergence_action=_optional_action(
            mapping["first_shortest_divergence_action"], "first_shortest_divergence_action"
        ),
        first_loop_step=_optional_integer(mapping["first_loop_step"], "first_loop_step"),
        first_unrecoverable_step=_optional_integer(
            mapping["first_unrecoverable_step"], "first_unrecoverable_step"
        ),
        legal_action_count=_integer(mapping["legal_action_count"], "legal_action_count", minimum=0),
        bound_blocked_action_count=_integer(
            mapping["bound_blocked_action_count"], "bound_blocked_action_count", minimum=0
        ),
        repeated_state_visit_count=_integer(
            mapping["repeated_state_visit_count"], "repeated_state_visit_count", minimum=0
        ),
        loop_transition_count=_integer(
            mapping["loop_transition_count"], "loop_transition_count", minimum=0
        ),
        terminal_reason=_terminal_reason(mapping["terminal_reason"]),
        final_absolute_error_cents=_integer(
            mapping["final_absolute_error_cents"], "final_absolute_error_cents", minimum=0
        ),
        action_count=_integer(mapping["action_count"], "action_count", minimum=1),
        excess_actions=_optional_integer(mapping["excess_actions"], "excess_actions"),
    )


def _metrics_to_document(metrics: PerActionMetrics) -> dict[str, JSONValue]:
    return {
        "action": int(metrics.action),
        "precision_numerator": metrics.precision_numerator,
        "precision_denominator": metrics.precision_denominator,
        "precision": metrics.precision,
        "recall_numerator": metrics.recall_numerator,
        "recall_denominator": metrics.recall_denominator,
        "recall": metrics.recall,
        "support": metrics.support,
    }


def _metrics_from_document(value: object) -> PerActionMetrics:
    mapping = _mapping(value, "per-action metrics")
    fields = {
        "action",
        "precision_numerator",
        "precision_denominator",
        "precision",
        "recall_numerator",
        "recall_denominator",
        "recall",
        "support",
    }
    _exact_fields(mapping, fields, "per-action metrics")
    return PerActionMetrics(
        action=_action(mapping["action"], "action"),
        precision_numerator=_integer(
            mapping["precision_numerator"], "precision_numerator", minimum=0
        ),
        precision_denominator=_integer(
            mapping["precision_denominator"], "precision_denominator", minimum=0
        ),
        precision=_optional_float(mapping["precision"], "precision"),
        recall_numerator=_integer(mapping["recall_numerator"], "recall_numerator", minimum=0),
        recall_denominator=_integer(mapping["recall_denominator"], "recall_denominator", minimum=0),
        recall=_optional_float(mapping["recall"], "recall"),
        support=_integer(mapping["support"], "support", minimum=0),
    )


def _survival_to_document(entry: SurvivalEntry) -> dict[str, JSONValue]:
    return {
        "step": entry.step,
        "at_risk": entry.at_risk,
        "survivors": entry.survivors,
        "rate": entry.rate,
    }


def _survival_from_document(value: object) -> SurvivalEntry:
    mapping = _mapping(value, "survival entry")
    _exact_fields(mapping, {"step", "at_risk", "survivors", "rate"}, "survival entry")
    return SurvivalEntry(
        step=_integer(mapping["step"], "step", minimum=1),
        at_risk=_integer(mapping["at_risk"], "at_risk", minimum=1),
        survivors=_integer(mapping["survivors"], "survivors", minimum=0),
        rate=_float(mapping["rate"], "rate"),
    )


def _band_to_document(band: SubmitErrorBand) -> dict[str, JSONValue]:
    return {
        "label": band.label,
        "minimum_cents": band.minimum_cents,
        "maximum_cents": band.maximum_cents,
        "count": band.count,
    }


def _band_from_document(value: object) -> SubmitErrorBand:
    mapping = _mapping(value, "Submit error band")
    _exact_fields(
        mapping,
        {"label", "minimum_cents", "maximum_cents", "count"},
        "Submit error band",
    )
    return SubmitErrorBand(
        label=_string(mapping["label"], "label"),
        minimum_cents=_integer(mapping["minimum_cents"], "minimum_cents", minimum=0),
        maximum_cents=_optional_integer(mapping["maximum_cents"], "maximum_cents"),
        count=_integer(mapping["count"], "count", minimum=0),
    )


def diagnostic_report_to_document(report: DiagnosticReport) -> dict[str, JSONValue]:
    """Convert a validated report into its exact JSON vocabulary."""

    if not isinstance(report, DiagnosticReport):
        raise ValueError("report must be a DiagnosticReport")
    return {
        "schema_id": report.schema_id,
        "seed": report.seed,
        "artifact_manifest_sha256": report.artifact_manifest_sha256,
        "actor_semantics": report.actor_semantics,
        "bound_mask": report.bound_mask,
        "suite_id": report.suite_id,
        "suite_digest_sha256": report.suite_digest_sha256,
        "episodes": [_episode_to_document(item) for item in report.episodes],
        "confusion_matrix": [list(row) for row in report.confusion_matrix],
        "per_action_metrics": [_metrics_to_document(item) for item in report.per_action_metrics],
        "shortest_action_correct": report.shortest_action_correct,
        "shortest_action_decisions": report.shortest_action_decisions,
        "shortest_action_accuracy": report.shortest_action_accuracy,
        "survival_curve": [_survival_to_document(item) for item in report.survival_curve],
        "submit_error_bands": [_band_to_document(item) for item in report.submit_error_bands],
    }


def diagnostic_report_from_document(value: object) -> DiagnosticReport:
    """Construct and re-derive a report from one strict decoded object."""

    mapping = _mapping(value, "diagnostic report")
    fields = {
        "schema_id",
        "seed",
        "artifact_manifest_sha256",
        "actor_semantics",
        "bound_mask",
        "suite_id",
        "suite_digest_sha256",
        "episodes",
        "confusion_matrix",
        "per_action_metrics",
        "shortest_action_correct",
        "shortest_action_decisions",
        "shortest_action_accuracy",
        "survival_curve",
        "submit_error_bands",
    }
    _exact_fields(mapping, fields, "diagnostic report")
    matrix_values = _list(mapping["confusion_matrix"], "confusion_matrix")
    return DiagnosticReport(
        schema_id=_string(mapping["schema_id"], "schema_id"),
        seed=_integer(mapping["seed"], "seed", minimum=0),
        artifact_manifest_sha256=_string(
            mapping["artifact_manifest_sha256"], "artifact_manifest_sha256"
        ),
        actor_semantics=_string(mapping["actor_semantics"], "actor_semantics"),
        bound_mask=_boolean(mapping["bound_mask"], "bound_mask"),
        suite_id=_string(mapping["suite_id"], "suite_id"),
        suite_digest_sha256=_string(mapping["suite_digest_sha256"], "suite_digest_sha256"),
        episodes=tuple(
            _episode_from_document(item) for item in _list(mapping["episodes"], "episodes")
        ),
        confusion_matrix=tuple(
            tuple(_integer(item, "confusion count", minimum=0) for item in _list(row, "row"))
            for row in matrix_values
        ),
        per_action_metrics=tuple(
            _metrics_from_document(item)
            for item in _list(mapping["per_action_metrics"], "per_action_metrics")
        ),
        shortest_action_correct=_integer(
            mapping["shortest_action_correct"], "shortest_action_correct", minimum=0
        ),
        shortest_action_decisions=_integer(
            mapping["shortest_action_decisions"], "shortest_action_decisions", minimum=1
        ),
        shortest_action_accuracy=_float(
            mapping["shortest_action_accuracy"], "shortest_action_accuracy"
        ),
        survival_curve=tuple(
            _survival_from_document(item)
            for item in _list(mapping["survival_curve"], "survival_curve")
        ),
        submit_error_bands=tuple(
            _band_from_document(item)
            for item in _list(mapping["submit_error_bands"], "submit_error_bands")
        ),
    )


def diagnostic_report_bytes(report: DiagnosticReport) -> bytes:
    """Encode one report as canonical finite JSON plus one trailing newline."""

    return canonical_json_bytes(diagnostic_report_to_document(report))


def diagnostic_report_from_bytes(content: bytes) -> DiagnosticReport:
    """Decode duplicate-free finite canonical report bytes and re-derive all metrics."""

    document = decode_json_bytes(content)
    report = diagnostic_report_from_document(document)
    if content != diagnostic_report_bytes(report):
        raise ValueError("diagnostic report bytes must use canonical JSON encoding")
    return report


def _inventory_to_document(item: DiagnosticReportInventory) -> dict[str, JSONValue]:
    return {
        "seed": item.seed,
        "artifact_manifest_sha256": item.artifact_manifest_sha256,
        "report_sha256": item.report_sha256,
    }


def _inventory_from_document(value: object) -> DiagnosticReportInventory:
    mapping = _mapping(value, "diagnostic inventory entry")
    _exact_fields(
        mapping,
        {"seed", "artifact_manifest_sha256", "report_sha256"},
        "diagnostic inventory entry",
    )
    return DiagnosticReportInventory(
        seed=_integer(mapping["seed"], "seed", minimum=0),
        artifact_manifest_sha256=_string(
            mapping["artifact_manifest_sha256"], "artifact_manifest_sha256"
        ),
        report_sha256=_string(mapping["report_sha256"], "report_sha256"),
    )


def diagnostic_bundle_to_document(bundle: DiagnosticBundle) -> dict[str, JSONValue]:
    """Convert a validated bundle into its exact embedded-report vocabulary."""

    if not isinstance(bundle, DiagnosticBundle):
        raise ValueError("bundle must be a DiagnosticBundle")
    return {
        "schema_id": bundle.schema_id,
        "suite_id": bundle.suite_id,
        "suite_digest_sha256": bundle.suite_digest_sha256,
        "device": bundle.device,
        "bound_mask": bundle.bound_mask,
        "reports": [diagnostic_report_to_document(report) for report in bundle.reports],
        "inventory": [_inventory_to_document(item) for item in bundle.inventory],
    }


def diagnostic_bundle_from_document(value: object) -> DiagnosticBundle:
    """Construct and hash-close a bundle from one strict decoded object."""

    mapping = _mapping(value, "diagnostic bundle")
    fields = {
        "schema_id",
        "suite_id",
        "suite_digest_sha256",
        "device",
        "bound_mask",
        "reports",
        "inventory",
    }
    _exact_fields(mapping, fields, "diagnostic bundle")
    return DiagnosticBundle(
        schema_id=_string(mapping["schema_id"], "schema_id"),
        suite_id=_string(mapping["suite_id"], "suite_id"),
        suite_digest_sha256=_string(mapping["suite_digest_sha256"], "suite_digest_sha256"),
        device=_string(mapping["device"], "device"),
        bound_mask=_boolean(mapping["bound_mask"], "bound_mask"),
        reports=tuple(
            diagnostic_report_from_document(item) for item in _list(mapping["reports"], "reports")
        ),
        inventory=tuple(
            _inventory_from_document(item) for item in _list(mapping["inventory"], "inventory")
        ),
    )


def diagnostic_bundle_bytes(bundle: DiagnosticBundle) -> bytes:
    """Encode one bundle as canonical finite JSON plus one trailing newline."""

    return canonical_json_bytes(diagnostic_bundle_to_document(bundle))


def diagnostic_bundle_from_bytes(content: bytes) -> DiagnosticBundle:
    """Decode canonical bundle bytes and verify every embedded report inventory hash."""

    document = decode_json_bytes(content)
    bundle = diagnostic_bundle_from_document(document)
    if content != diagnostic_bundle_bytes(bundle):
        raise ValueError("diagnostic bundle bytes must use canonical JSON encoding")
    return bundle


def validate_diagnostic_output_path(
    output_path: Path,
    *,
    artifact_directories: Sequence[Path],
) -> Path:
    """Resolve a create-only output and reject aliases inside any input artifact."""

    if not isinstance(output_path, Path):
        raise ValueError("output_path must be a Path")
    if isinstance(artifact_directories, (str, bytes)):
        raise ValueError("artifact_directories must contain Path values")
    directories = tuple(artifact_directories)
    if not directories or not all(isinstance(path, Path) for path in directories):
        raise ValueError("artifact_directories must contain Path values")
    lexical_output = output_path.absolute()
    resolved_output = output_path.resolve()
    if output_path.exists() or output_path.is_symlink() or resolved_output.exists():
        raise ValueError("diagnostic output path must not already exist or be a symlink")
    for artifact in directories:
        lexical_artifact = artifact.absolute()
        resolved_artifact = artifact.resolve()
        if (
            lexical_output == lexical_artifact
            or lexical_artifact in lexical_output.parents
            or resolved_output == resolved_artifact
            or resolved_artifact in resolved_output.parents
        ):
            raise ValueError(
                "diagnostic output path must be outside every input artifact directory"
            )
    return resolved_output


__all__ = [
    "diagnostic_bundle_bytes",
    "diagnostic_bundle_from_bytes",
    "diagnostic_bundle_from_document",
    "diagnostic_bundle_to_document",
    "diagnostic_report_bytes",
    "diagnostic_report_from_bytes",
    "diagnostic_report_from_document",
    "diagnostic_report_to_document",
    "validate_diagnostic_output_path",
]
