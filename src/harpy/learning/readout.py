"""Human readouts of strictly validated saved evidence, without loading models."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence

from harpy.learning.artifacts import decode_json_bytes


def summarize_report_bytes(content: bytes, *, format: str = "text") -> str:
    """Validate a report through its owning codec before formatting any evidence.

    The readout is a view of the input, never a new scientific result. Models and
    artifact directories are neither opened nor needed to inspect a saved report.
    """

    if format not in {"text", "markdown"}:
        raise ValueError("format must be text or markdown")
    document = decode_json_bytes(content)
    if "schema_version" in document:
        from harpy.learning.workflows import evaluation_report_from_bytes

        report = evaluation_report_from_bytes(content)
        document = report.to_document()
    elif str(document.get("schema_id", "")).startswith("harpy-pitch-exploration-") or (
        document.get("schema_id") == "harpy-artifact-run-v1"
    ):
        from harpy.learning.experiment_results import readout_from_bytes

        document = readout_from_bytes(content).to_document()
    else:
        from harpy.learning.diagnostic_codecs import (
            diagnostic_bundle_from_bytes,
            diagnostic_bundle_to_document,
            diagnostic_report_from_bytes,
            diagnostic_report_to_document,
        )

        if "reports" in document:
            document = diagnostic_bundle_to_document(diagnostic_bundle_from_bytes(content))
        else:
            document = diagnostic_report_to_document(diagnostic_report_from_bytes(content))

    lines = [_heading("Harpy research readout", format), ""]
    identity = document.get("schema_id", f"evaluation schema v{document.get('schema_version')}")
    lines.append(f"Report: {_cell(identity, format)}")
    if "profile" in document:
        lines.append(f"Profile: {document['profile']}")
    if "evaluation_device" in document:
        lines.append(f"Evaluation device: {document['evaluation_device']}")
    if "training_device" in document:
        lines.append(f"Training device: {document['training_device']}")
    if "cohort_digest_sha256" in document:
        lines.append(f"Cohort: {document['cohort_digest_sha256']}")

    criteria = []
    for key, label in (("criterion", "Pitch"), ("bc_criterion", "BC"), ("ppo_criterion", "PPO")):
        criterion = document.get(key)
        if criterion is not None:
            if document.get("schema_id") == "harpy-artifact-run-v1":
                label = "Demonstration"
            criteria.append((label, criterion))
    for label, criterion in criteria:
        lines.extend(["", f"{label} criterion: {criterion['status']}"])
        lines.append(f"Scientifically eligible: {'yes' if criterion['eligible'] else 'no'}")
        if not criterion["eligible"]:
            lines.append("This report does not establish that a scientific criterion was met.")
        if criterion.get("reason"):
            lines.append(f"Reason: {_cell(criterion['reason'], format)}")
        failed = criterion.get("failed_gates", [])
        if failed:
            lines.append("Failed gates: " + "; ".join(_cell(gate, format) for gate in failed))
        for key in (
            "heldout_next_action_accuracy",
            "iid_submitted_success_rate",
            "median_iid_submitted_success_rate",
            "seeds_strictly_beating_random",
        ):
            value = criterion.get(key)
            if value is not None:
                lines.append(f"{key.replace('_', ' ')}: {_number(value)}")

    provenance = document.get("provenance")
    if isinstance(provenance, Mapping):
        lines.extend(["", "Artifact provenance:"])
        for key in ("trainer", "training_seed", "profile", "manifest_sha256", "model_sha256"):
            if key in provenance:
                lines.append(f"{key.replace('_', ' ')}: {_cell(provenance[key], format)}")

    evidence = document.get("evidence", document)
    rows = evidence.get("terminal_rows", evidence.get("rows", []))
    if rows:
        lines.extend(["", _heading("Policy outcomes and matched baselines", format, level=2), ""])
        lines.extend(_outcome_table(rows, format))
        lines.extend(
            [
                "",
                "Success means explicit submission within 5 cents. "
                "MAE is final absolute error in cents.",
                "Observation tracks and spectrum probes are separate conditions; "
                "do not pool these rows.",
            ]
        )

    diagnostic = document.get("diagnostics", document)
    reports = diagnostic.get("reports", [])
    if not reports and "actor_semantics" in diagnostic:
        reports = [diagnostic]
    if reports:
        lines.extend(["", _heading("Diagnostic hotspots", format, level=2), ""])
        lines.append("Diagnostics describe behavior; they do not grant scientific eligibility.")
        lines.extend(_diagnostic_tables(reports, format))
    if "trace" in document:
        trace = document["trace"]
        lines.extend(
            [
                "",
                _heading("Demonstration outcome", format, level=2),
                "",
                f"Episode seed: {trace['seed']}",
                f"Target: {_cell(trace['target_note'], format)}",
                f"Reason: {trace['terminal_reason']}",
                f"Submitted success: {'yes' if trace['submitted_success'] else 'no'}",
                f"Final absolute error: {trace['final_absolute_error_cents']} cents",
                f"Actions: {trace['action_count']}",
            ]
        )
    return "\n".join(lines) + "\n"


def _outcome_table(rows: Sequence[Mapping], format: str) -> list[str]:
    values = []
    for row in rows:
        metrics = row["metrics"]
        values.append(
            [
                row["actor_id"],
                _number(row.get("seed")),
                row["observation_mode"],
                _suite_label(row["suite_id"]),
                (row.get("probe") or "none").removesuffix("_spectrum"),
                str(metrics["episodes"]),
                _percent(metrics["submitted_success_rate"]),
                _number(metrics["mean_absolute_final_error_cents"]),
                _number(metrics["mean_actions"]),
                _percent(metrics["truncation_rate"]),
            ]
        )
    return _table(
        [
            "Actor",
            "Seed",
            "Track",
            "Suite",
            "Probe",
            "N",
            "Success",
            "MAE (cents)",
            "Actions",
            "Truncated",
        ],
        values,
        format,
    )


def _diagnostic_tables(reports: Sequence[Mapping], format: str) -> list[str]:
    rows = []
    hotspots = []
    for report in reports:
        episodes = report["episodes"]
        success = sum(item["terminal_reason"] == "submitted_success" for item in episodes)
        loops = sum(item["first_loop_step"] is not None for item in episodes)
        blocked = sum(item["bound_blocked_action_count"] for item in episodes)
        exhausted = sum(item["terminal_reason"] == "budget_exhausted" for item in episodes)
        rows.append(
            [
                str(report["seed"]),
                report["actor_semantics"],
                report["suite_id"],
                str(len(episodes)),
                _percent(success / len(episodes)),
                str(exhausted),
                str(loops),
                str(blocked),
                _percent(report["shortest_action_accuracy"]),
            ]
        )
        failures = sorted(
            (item for item in episodes if item["terminal_reason"] != "submitted_success"),
            key=lambda item: (-item["final_absolute_error_cents"], item["episode_index"]),
        )[:3]
        for item in failures:
            hotspots.append(
                [
                    str(report["seed"]),
                    str(item["episode_index"]),
                    item["terminal_reason"],
                    str(item["final_absolute_error_cents"]),
                    str(item["action_count"]),
                    _number(item["first_loop_step"]),
                ]
            )
    lines = _table(
        [
            "Seed",
            "Actor",
            "Suite",
            "N",
            "Success",
            "Exhausted",
            "Loop episodes",
            "Bound blocks",
            "Shortest-action accuracy",
        ],
        rows,
        format,
    )
    if hotspots:
        lines.extend(["", "Largest final errors (up to three failed episodes per seed):", ""])
        lines.extend(
            _table(
                ["Seed", "Episode", "Reason", "Error (cents)", "Actions", "First loop step"],
                hotspots,
                format,
            )
        )
    else:
        lines.extend(["", "No failed episodes in the diagnostic report."])
    return lines


def _heading(text: str, format: str, *, level: int = 1) -> str:
    return f"{'#' * level} {text}" if format == "markdown" else text


def _suite_label(value: str) -> str:
    return value.removeprefix("harpy-sine-policy-eval-").removeprefix("harpy-sine-pitch-e-")


def _number(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _cell(value: object, format: str) -> str:
    text = " ".join(str(value).split())
    text = "".join(character for character in text if character.isprintable())
    if format == "markdown":
        return html.escape(text).replace("\\", "\\\\").replace("|", "\\|").replace("`", "\\`")
    return text


def _table(headers: list[str], rows: list[list[str]], format: str) -> list[str]:
    cells = [[_cell(value, format) for value in row] for row in [headers, *rows]]
    if format == "markdown":
        return [
            "| " + " | ".join(cells[0]) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in cells[1:]),
        ]
    widths = [max(len(row[index]) for row in cells) for index in range(len(headers))]
    return [
        "  ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)).rstrip()
        for row in cells
    ]
