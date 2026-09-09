"""Derived CSV and Markdown views of the complete frozen spectrum study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

from harpy.envs.robustness import ROBUSTNESS_CONDITIONS
from harpy.experiments.models import _json_bytes, _json_loads, _plain
from harpy.experiments.results import _initial_error, _p99, _perception_metrics, load_result
from harpy.learning.artifacts import write_new_bytes
from research.spectrum_preservation.membership import episodes_from_manifest

_ACTORS = ("legacy-point", "cell-max", "quadratic-fft")
_PARTITIONS = ("iid", "lower", "upper")
_COUNTS = (
    "episodes",
    "submitted_successes",
    "submitted_within_1_cent",
    "invalid_actions",
    "truncations",
    "actions",
    "inferences",
    "initial_estimates",
    "initial_within_1_cent",
    "initial_within_5_cents",
)


def _paired(candidate, legacy, *, clean: bool) -> dict:
    """Pair episode identities rather than assuming input order or using net changes."""
    before = {record.episode_id: record for record in legacy}
    after = {record.episode_id: record for record in candidate}
    if len(before) != len(legacy) or len(after) != len(candidate) or before.keys() != after.keys():
        raise ValueError("actor comparisons require identical unique episode membership")
    pairs = [(before[key], after[key]) for key in before]
    values = {
        "paired_episodes": len(pairs),
        "rescued_vs_legacy": sum(
            not a.terminal.submitted_success and b.terminal.submitted_success for a, b in pairs
        ),
        "regressed_vs_legacy": sum(
            a.terminal.submitted_success and not b.terminal.submitted_success for a, b in pairs
        ),
    }
    for name, predicate in (
        ("initial", lambda record: _initial_error(record) <= 1),
        (
            "submitted",
            lambda record: record.terminal.submitted_success and record.terminal.within_1_cent,
        ),
    ):
        values[f"clean_{name}_within_1_cent_gains"] = (
            sum(not predicate(a) and predicate(b) for a, b in pairs) if clean else None
        )
        values[f"clean_{name}_within_1_cent_losses"] = (
            sum(predicate(a) and not predicate(b) for a, b in pairs) if clean else None
        )
    return values


def _view(row: dict) -> dict:
    initial = row["initial_perception"]
    return {
        "condition": row["condition"],
        "partition": row["partition"],
        "actor": row["actor"],
        **{
            key: row[key]
            for key in (
                "episodes",
                "submitted_successes",
                "invalid_actions",
                "truncations",
                "actions",
                "inferences",
            )
        },
        "submitted_within_1_cent": round(row["submitted_within_1_cent_rate"] * row["episodes"]),
        "initial_estimates": initial["estimates"],
        "initial_within_1_cent": round(initial["within_1_cents_rate"] * initial["estimates"]),
        "initial_within_5_cents": round(initial["within_5_cents_rate"] * initial["estimates"]),
        "initial_within_1_cent_rate": initial["within_1_cents_rate"],
        "initial_within_5_cents_rate": initial["within_5_cents_rate"],
        "initial_p99_error_cents": initial["p99_absolute_error_cents"],
        "initial_max_error_cents": initial["max_absolute_error_cents"],
        "final_p99_error_cents": row["final_p99_absolute_error_cents"],
        "final_max_error_cents": row["final_max_absolute_error_cents"],
        "submitted_success_rate": row["submitted_success_rate"],
        "mean_actions": row["mean_actions"],
        "mean_inferences": row["mean_inferences"],
    }


def comparison_rows(result) -> tuple[dict, ...]:
    """Preserve register rows; derive condition totals from counts, never pooled conditions."""
    membership = _json_loads(Path(__file__).with_name("membership.json").read_bytes())
    if result.protocol is None or _json_bytes(result.protocol) != _json_bytes(membership):
        raise ValueError("result must bind the exact frozen spectrum-study membership")
    episodes = episodes_from_manifest(membership)
    if result.episodes != episodes or len(episodes) != 600:
        raise ValueError("result must contain all 600 frozen study episodes")
    if tuple(actor["name"] for actor in result.actors) != _ACTORS or any(
        actor["observation_mode"] != "waveform" for actor in result.actors
    ):
        raise ValueError(
            "result must contain legacy-point, cell-max, and quadratic-fft waveform actors"
        )
    if (
        _plain(result.conditions)
        != [condition.to_document() for condition in ROBUSTNESS_CONDITIONS]
        or len(result.records) != 14_400
    ):
        raise ValueError("readout requires all eight conditions and 14400 complete actor-episodes")
    if any(
        len(record.estimates) != 1 or record.estimates[0]["step"] != 1 for record in result.records
    ):
        raise ValueError("the study requires one recorded initial estimate per actor-episode")
    if any(
        actor["artifact_files"] != result.actors[0]["artifact_files"]
        or actor["controller"] != result.actors[0]["controller"]
        for actor in result.actors
    ):
        raise ValueError("study actors must bind the same files and controller")
    groups = defaultdict(list)
    partitions = {episode.id: episode.partition for episode in episodes}
    for record in result.records:
        for partition in (partitions[record.episode_id], "all"):
            groups[(record.condition_id, partition, record.actor_name)].append(record)
    rows = [_view(row) for row in result.rows]
    for condition in ROBUSTNESS_CONDITIONS:
        for actor in _ACTORS:
            parts = [
                row for row in rows if row["condition"] == condition.id and row["actor"] == actor
            ]
            total = {key: sum(row[key] for row in parts) for key in _COUNTS}
            records = tuple(groups[(condition.id, "all", actor)])
            initial = _perception_metrics(records, initial_only=True)
            errors = [record.terminal.final_absolute_error_cents for record in records]
            rows.append(
                {
                    "condition": condition.id,
                    "partition": "all",
                    "actor": actor,
                    **total,
                    "initial_within_1_cent_rate": total["initial_within_1_cent"]
                    / total["initial_estimates"],
                    "initial_within_5_cents_rate": total["initial_within_5_cents"]
                    / total["initial_estimates"],
                    "initial_p99_error_cents": initial["p99_absolute_error_cents"],
                    "initial_max_error_cents": initial["max_absolute_error_cents"],
                    "final_p99_error_cents": _p99(errors),
                    "final_max_error_cents": max(errors),
                    "submitted_success_rate": total["submitted_successes"] / total["episodes"],
                    "mean_actions": total["actions"] / total["episodes"],
                    "mean_inferences": total["inferences"] / total["episodes"],
                }
            )
    empty = dict.fromkeys(_paired((), (), clean=False))
    for row in rows:
        key = (row["condition"], row["partition"])
        row.update(
            empty
            if row["actor"] == "legacy-point"
            else _paired(
                groups[(*key, row["actor"])],
                groups[(*key, "legacy-point")],
                clean=row["condition"] == "clean",
            )
        )
    order = {condition.id: index for index, condition in enumerate(ROBUSTNESS_CONDITIONS)}
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                order[row["condition"]],
                ("all", *_PARTITIONS).index(row["partition"]),
                _ACTORS.index(row["actor"]),
            ),
        )
    )


def _table(rows) -> str:
    lines = [
        "| Register | Actor | N | Initial ≤1c / ≤5c | Initial p99 / max (c) | Success | "
        "Invalid / truncations | Mean actions / inferences | Rescued / regressed vs legacy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        paired = (
            "—"
            if row["actor"] == "legacy-point"
            else f"{row['rescued_vs_legacy']} / {row['regressed_vs_legacy']}"
        )
        lines.append(
            f"| {row['partition']} | {row['actor']} | {row['episodes']} | "
            f"{row['initial_within_1_cent']} / {row['initial_within_5_cents']} | "
            f"{row['initial_p99_error_cents']:.2f} / {row['initial_max_error_cents']:.2f} | "
            f"{row['submitted_successes']} | {row['invalid_actions']} / {row['truncations']} | "
            f"{row['mean_actions']:.2f} / {row['mean_inferences']:.2f} | {paired} |"
        )
    return "\n".join(lines)


def write_readout(result_path: Path, output: Path) -> None:
    """Strict-load the full result, then publish new derived views without overwrites."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    content = result_path.read_bytes()
    result_hash = hashlib.sha256(content).hexdigest()
    result = load_result(result_path)
    rows = comparison_rows(result)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    summary = [
        "# Spectrum preservation study measurements",
        "",
        "Counts use matched episodes. Conditions are never pooled; `all` combines registers "
        "within one condition. Full numerical values, terminal p99/max errors, and all paired "
        "counts are in [paired-actors.csv](paired-actors.csv).",
        "",
    ]
    for condition in ROBUSTNESS_CONDITIONS:
        summary.extend(
            (
                f"## {condition.id}",
                "",
                _table([row for row in rows if row["condition"] == condition.id]),
                "",
            )
        )
    summary.extend(
        (
            "## Clean one-cent changes versus legacy",
            "",
            "Initial perception and successful terminal tuning are separate outcomes. "
            "Gains and losses are reported separately.",
            "",
            "| Register | Actor | Initial gains | Initial losses | "
            "Submitted gains | Submitted losses |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        )
    )
    for row in rows:
        if row["condition"] == "clean" and row["actor"] != "legacy-point":
            summary.append(
                f"| {row['partition']} | {row['actor']} | "
                f"{row['clean_initial_within_1_cent_gains']} | "
                f"{row['clean_initial_within_1_cent_losses']} | "
                f"{row['clean_submitted_within_1_cent_gains']} | "
                f"{row['clean_submitted_within_1_cent_losses']} |"
            )
    identity = {
        "result_path": str(result_path.resolve()),
        "result_sha256": result_hash,
        "result_size_bytes": len(content),
        "protocol_id": result.protocol["id"],
        "protocol_digest_sha256": result.protocol["digest_sha256"],
        "selection_seed": result.protocol["selection"]["seed"],
        "readout_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "actor_identities": [
            {
                key: _plain(actor[key])
                for key in (
                    "name",
                    "observation_mode",
                    "estimator",
                    "decoder",
                    "controller",
                    "device",
                )
            }
            for actor in result.actors
        ],
        "study_files": _plain(result.actors[0]["artifact_files"]),
        "provenance": _plain(result.provenance),
    }
    readme = (
        "# Spectrum study derived views\n\n[Summary](summary.md) · "
        "[Paired CSV](paired-actors.csv)\n\n"
        "These are descriptive finite-cohort measurements, not a qualification decision. "
        "Membership is fresh combinations in the known rendering family, excluding the "
        "recorded inventory. It becomes consumed evidence after this run. No statistical "
        "significance or learned-model claim is inferred.\n\n"
        "Success means a submitted terminal within five cents. Rescued/regressed counts "
        "compare the same episode under the same condition against legacy-point. Initial "
        "perception uses the recorded first estimate. P99 uses the existing ordinary-result "
        "linear-interpolation convention. Overall rates use summed numerators and "
        "denominators; p99 values are recomputed from pooled records within each condition, "
        "never averaged across registers.\n\n## Exact input and execution identity\n\n```json\n"
        + json.dumps(identity, indent=2, sort_keys=True, allow_nan=False)
        + "\n```\n"
    )
    if result_path.read_bytes() != content:
        raise ValueError("result changed during readout generation")
    output.mkdir(parents=True, exist_ok=False)
    for name, text in (
        ("paired-actors.csv", stream.getvalue()),
        ("summary.md", "\n".join(summary) + "\n"),
        ("README.md", readme),
    ):
        write_new_bytes(output / name, text.encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new derived-view directory")
    args = parser.parse_args(argv)
    write_readout(args.result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
