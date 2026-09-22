"""Export portable GNU time measurements while retaining hashes of raw local logs."""

from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

from harpy._serialization import (
    canonical_json_bytes,
    decode_json_bytes,
    sha256_file,
    write_new_bytes,
)

SPLITS = ("validation", "iid-test", "compositional-test")


def scheduling_binding(study: Path, plan_sha256: str) -> tuple[int, dict]:
    helper = Path(__file__).with_name("scheduling.py")
    spec = importlib.util.spec_from_file_location("_harpy_scheduling_reader", helper)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    path = study / "scheduling-receipt.json"
    document = decode_json_bytes(path.read_bytes())
    maximum = reader.validate_receipt(document, plan_sha256)
    if document["helper_source_sha256"] != sha256_file(helper):
        raise ValueError("scheduling helper differs from retained receipt")
    inputs = [document["observations_input"]]
    inputs.extend(row["completion_report"] for row in document["evaluations"])
    for item in inputs:
        name = item["name"]
        if not name or Path(name).name != name or any(char in name for char in ("\\", ":")):
            raise ValueError("scheduling input names must be portable basenames")
        retained = study / name
        if retained.stat().st_size != item["size_bytes"] or sha256_file(retained) != item["sha256"]:
            raise ValueError("scheduling input differs from retained local evidence")
    observations = decode_json_bytes((study / document["observations_input"]["name"]).read_bytes())
    if canonical_json_bytes(observations) != canonical_json_bytes(document["observations"]):
        raise ValueError("scheduling observations differ from retained local evidence")
    for row in document["evaluations"]:
        retained = study / row["completion_report"]["name"]
        written_ns = reader._timestamp(row["completion_report"]["written_utc"])
        if (
            written_ns != retained.stat().st_mtime_ns
            or measurements(retained.read_text())["wall_seconds"] != row["wall_seconds"]
        ):
            raise ValueError("scheduling completion report differs from retained measurements")
    return maximum, {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def measurements(text: str) -> dict:
    result = {}
    for line in text.splitlines():
        key, separator, value = line.strip().partition(": ")
        if not separator or key == "Command being timed":
            continue
        result[key] = value
    if result.get("Exit status") != "0":
        raise ValueError("final study resource receipt requires a successful completed process")
    wall = result["Elapsed (wall clock) time (h:mm:ss or m:ss)"]
    elapsed = 0.0
    for part in wall.split(":"):
        elapsed = elapsed * 60 + float(part)
    return {
        "wall_seconds": elapsed,
        "user_seconds": float(result["User time (seconds)"]),
        "system_seconds": float(result["System time (seconds)"]),
        "peak_rss_mib": int(result["Maximum resident set size (kbytes)"]) / 1024,
        "gnu_time_measurements": result,
    }


def build_receipts(study: Path, splits: tuple[str, ...] = SPLITS) -> dict:
    """Recompute portable measurements from successful, frozen local executions."""
    if splits not in (SPLITS, ("validation",)):
        raise ValueError("resource receipts require the complete study or validation only")
    plan = decode_json_bytes((study / "plan.json").read_bytes())
    plan_sha256 = sha256_file(study / "plan.json")
    maximum, scheduling = scheduling_binding(study, plan_sha256) if splits == SPLITS else (1, None)
    rows = []
    for split in splits:
        time_report = study / f"{split}-time.txt"
        progress_log = study / f"{split}.log"
        measured = measurements(time_report.read_text())
        final_line = next(line for line in reversed(progress_log.read_bytes().splitlines()) if line)
        final = decode_json_bytes(final_line)
        expected = plan["splits"][split]["targets"] * len(plan["actors"])
        if (
            final.get("complete") is not True
            or final.get("split") != split
            or type(final.get("completed")) is not int
            or final["completed"] != expected
            or isinstance(final.get("wall_seconds"), bool)
            or not isinstance(final.get("wall_seconds"), (int, float))
            or not math.isfinite(final["wall_seconds"])
            or not 0 <= final["wall_seconds"] <= measured["wall_seconds"]
        ):
            raise ValueError("resource progress log must complete its frozen split")
        manifest_path = study / split / "manifest.json"
        manifest = decode_json_bytes(manifest_path.read_bytes())
        completion = decode_json_bytes((study / split / "complete.json").read_bytes())
        if (
            manifest["configuration"]
            != {
                "study_plan_sha256": plan_sha256,
                "split": split,
                "dataset_manifest_sha256": plan["splits"][split]["sha256"],
            }
            or manifest["identity"]["source"] != plan["source"]
            or manifest["identity"]["runtime"] != plan["runtime"]
            or completion["manifest_sha256"] != sha256_file(manifest_path)
            or type(completion["completed"]) is not int
            or type(completion["expected"]) is not int
            or completion["completed"] != expected
            or completion["expected"] != expected
        ):
            raise ValueError("resource bundle identity differs from frozen study")
        rows.append(
            {
                "split": split,
                **measured,
                "target_hash_preflight_render_calls": plan["splits"][split]["targets"],
                "render_count_scope": (
                    "Preflight calls are outside bundle DSP counters but inside driver walltime; "
                    "bundle counts are not whole-process render totals"
                ),
                "raw_time_report": {
                    "name": time_report.name,
                    "sha256": sha256_file(time_report),
                    "size_bytes": time_report.stat().st_size,
                },
                "raw_progress_log": {
                    "name": progress_log.name,
                    "sha256": sha256_file(progress_log),
                    "size_bytes": progress_log.stat().st_size,
                },
            }
        )
    document = {
        "schema_id": "harpy.sound-match.portable-resource-receipts/1",
        "study_plan_sha256": plan_sha256,
        "runtime_package_source_sha256": plan["source"]["sha256"],
        "runtime": plan["runtime"],
        "processes_at_once": maximum,
        "torch_threads": 1,
        "scope": "complete-study" if splits == SPLITS else "validation-only",
        "transformation": (
            "GNU time command line omitted; metric labels/values retained; "
            "raw local time reports and progress logs hash-bound but not included"
        ),
        "rows": rows,
    }
    if scheduling is not None:
        document["scheduling_receipt"] = scheduling
        document["process_count_scope"] = (
            "evaluation workers only; excludes host apps and report readers"
        )
    return document


def collect(study: Path, output: Path, splits: tuple[str, ...] = SPLITS) -> dict:
    document = build_receipts(study, splits)
    write_new_bytes(output, canonical_json_bytes(document))
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    result = collect(args.study, args.output, ("validation",) if args.validation_only else SPLITS)
    print(canonical_json_bytes(result).decode())


if __name__ == "__main__":
    main()
