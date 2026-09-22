"""Describe observed evaluation-worker overlap without reconstructing unknown starts."""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
from datetime import UTC, datetime
from pathlib import Path

from harpy._serialization import (
    canonical_json_bytes,
    decode_json_bytes,
    sha256_file,
    write_new_bytes,
)

SPLITS = ("validation", "iid-test", "compositional-test")
SCHEMA = "harpy.sound-match.scheduling-receipt/1"
START_SOURCE = "Observed ps lstart in UTC, displayed to whole seconds"
REPORT_SOURCE = "GNU time report filesystem mtime; report write, not exact worker exit"
SCOPE = {
    "evaluation_splits": list(SPLITS),
    "invocations_per_split": 1,
    "excludes": "Other host applications, monitoring and report readers",
}
TIMING_CAVEAT = (
    "Shared-host wall times are not isolated benchmark timings. GNU elapsed times remain "
    "separate from observed starts and report-write timestamps; validation start is unknown. "
    "Timestamp digits describe recorded clock values, not exact event-time precision."
)
OVERLAP_SEMANTICS = (
    "Observed span is the timestamp span between simultaneous-worker snapshots, not an exact "
    "worker-lifetime overlap. The upper bound intersects whole-second observed-start to "
    "completion-report-write windows. Neither quantity measures simultaneous CPU execution."
)
_STAMP = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?Z")
_SHA = re.compile(r"[0-9a-f]{64}")
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _timestamp(value: object) -> int:
    if not isinstance(value, str) or not (match := _STAMP.fullmatch(value)):
        raise ValueError("scheduling timestamps must be UTC Z values")
    parsed = datetime.strptime(match[1], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)
    delta = parsed - _EPOCH
    return (delta.days * 86400 + delta.seconds) * 10**9 + int((match[2] or "").ljust(9, "0"))


def _utc_ns(value: int) -> str:
    seconds, fraction = divmod(value, 10**9)
    return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%S") + f".{fraction:09d}Z"


def _positive(value: object, *, integer: bool = False) -> bool:
    return (
        type(value) in ((int,) if integer else (int, float)) and math.isfinite(value) and value > 0
    )


def _file_reference(reference: dict, name: str | None = None) -> None:
    if (
        not isinstance(reference, dict)
        or not isinstance(reference.get("name"), str)
        or not reference["name"]
        or Path(reference["name"]).name != reference["name"]
        or "\\" in reference["name"]
        or reference["name"] in (".", "..")
        or (name is not None and reference["name"] != name)
        or not isinstance(reference.get("sha256"), str)
        or not _SHA.fullmatch(reference["sha256"])
        or not _positive(reference.get("size_bytes"), integer=True)
    ):
        raise ValueError("invalid scheduling retained-file reference")


def _derive(document: dict, expected_plan_sha256: str) -> tuple[int, dict]:
    if (
        not isinstance(expected_plan_sha256, str)
        or not _SHA.fullmatch(expected_plan_sha256)
        or document.get("schema_id") != SCHEMA
        or document.get("study_plan_sha256") != expected_plan_sha256
        or document.get("scope") != SCOPE
        or type(document["scope"]["invocations_per_split"]) is not int
        or document.get("timing_caveat") != TIMING_CAVEAT
        or not isinstance(document.get("helper_source_sha256"), str)
        or not _SHA.fullmatch(document["helper_source_sha256"])
    ):
        raise ValueError("scheduling receipt identity or scope differs")
    _file_reference(document["observations_input"])
    rows = document["evaluations"]
    if not isinstance(rows, list) or [row.get("split") for row in rows] != list(SPLITS):
        raise ValueError("scheduling requires exactly the three ordered evaluation splits")
    intervals = {}
    for row in rows:
        split = row["split"]
        report = row["completion_report"]
        _file_reference(report, f"{split}-time.txt")
        end = _timestamp(report["written_utc"])
        if report.get("timestamp_source") != REPORT_SOURCE or not _positive(row["wall_seconds"]):
            raise ValueError("invalid scheduling timing evidence")
        if split == "validation":
            if row.get("observed_start_utc") is not None or row.get("start_source") is not None:
                raise ValueError("validation start must remain unknown")
            start = None
        else:
            start = _timestamp(row["observed_start_utc"])
            if (
                row.get("start_source") != START_SOURCE
                or "." in row["observed_start_utc"]
                or start >= end
            ):
                raise ValueError("invalid observed evaluation interval")
        intervals[split] = (start, end)
    if not (
        intervals["validation"][1] < intervals["iid-test"][0] < intervals["compositional-test"][0]
    ):
        raise ValueError("validation must finish before the ordered final worker starts")

    observations = document["observations"]
    if (
        observations.get("schema_id") != "harpy.sound-match.scheduling-observations/1"
        or observations.get("study_plan_sha256") != expected_plan_sha256
    ):
        raise ValueError("scheduling observations differ from study plan")
    authorization = observations["authorization"]
    threshold = authorization["threshold_free_physical_kib"]
    if not _positive(threshold, integer=True) or any(
        not isinstance(authorization.get(key), str) or not authorization[key]
        for key in ("policy", "reason")
    ):
        raise ValueError("invalid scheduling authorization evidence")
    snapshots = observations["observations"]
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("scheduling requires live observations")
    live = []
    maximum_workers = 1
    previous = None
    headroom = False
    for snapshot in snapshots:
        stamp = _timestamp(snapshot["utc"])
        if previous is not None and stamp <= previous:
            raise ValueError("scheduling observations must be strictly ordered")
        previous = stamp
        if "windows_free_physical_kib" in snapshot and not _positive(
            snapshot["windows_free_physical_kib"], integer=True
        ):
            raise ValueError("invalid observed host memory")
        if snapshot.get("phase") == "before-compositional-launch":
            if (
                not 0 < intervals["compositional-test"][0] - stamp <= 60 * 10**9
                or snapshot.get("windows_free_physical_kib", 0) < threshold
            ):
                raise ValueError("missing fresh authorized launch headroom")
            headroom = True
        if "workers" not in snapshot:
            continue
        workers = snapshot["workers"]
        if not isinstance(workers, list) or not workers:
            raise ValueError("invalid live worker observation")
        seen = set()
        for worker in workers:
            split = worker["split"]
            if (
                split not in SPLITS[1:]
                or split in seen
                or type(worker.get("threads")) is not int
                or worker["threads"] != 1
                or not _positive(worker.get("rss_kib"), integer=True)
                or not intervals[split][0] <= stamp <= intervals[split][1]
            ):
                raise ValueError("live worker observation lies outside its frozen invocation")
            seen.add(split)
        maximum_workers = max(maximum_workers, len(seen))
        if seen == set(SPLITS[1:]):
            live.append(snapshot["utc"])
    if not headroom or not live:
        raise ValueError("overlapping schedule needs launch headroom and simultaneous observations")
    overlap = {
        "simultaneous_live_observations_utc": live,
        "observed_span_seconds": (_timestamp(live[-1]) - _timestamp(live[0])) / 10**9,
        "upper_bound_seconds": (
            min(intervals[split][1] for split in SPLITS[1:])
            - max(intervals[split][0] for split in SPLITS[1:])
        )
        / 10**9,
        "semantics": OVERLAP_SEMANTICS,
    }
    return maximum_workers, overlap


def validate_receipt(document: dict, expected_plan_sha256: str) -> int:
    """Recompute evaluation-worker concurrency from portable evidence, without file I/O."""
    try:
        maximum, overlap = _derive(document, expected_plan_sha256)
        if (
            type(document.get("maximum_evaluation_workers_at_once")) is not int
            or document["maximum_evaluation_workers_at_once"] != maximum
            or document.get("overlap") != overlap
            or any(
                type(document["overlap"][key]) not in (int, float)
                or not math.isfinite(document["overlap"][key])
                for key in ("observed_span_seconds", "upper_bound_seconds")
            )
        ):
            raise ValueError("declared scheduling concurrency or overlap differs from observations")
        return maximum
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("malformed scheduling receipt") from error


def build_receipt(study: Path, observations: Path, starts: dict[str, str]) -> dict:
    """Read completed GNU reports and explicit starts; never infer validation's start."""
    if set(starts) != set(SPLITS[1:]):
        raise ValueError("supply only the two observed final evaluation starts")
    spec = importlib.util.spec_from_file_location(
        "scheduling_resource_receipts", Path(__file__).with_name("resource_receipts.py")
    )
    resources = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resources)
    rows = []
    for split in SPLITS:
        path = study / f"{split}-time.txt"
        rows.append(
            {
                "split": split,
                "observed_start_utc": starts.get(split),
                "start_source": START_SOURCE if split in starts else None,
                "wall_seconds": resources.measurements(path.read_text())["wall_seconds"],
                "completion_report": {
                    "name": path.name,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "written_utc": _utc_ns(path.stat().st_mtime_ns),
                    "timestamp_source": REPORT_SOURCE,
                },
            }
        )
    document = {
        "schema_id": SCHEMA,
        "study_plan_sha256": sha256_file(study / "plan.json"),
        "helper_source_sha256": sha256_file(Path(__file__)),
        "scope": {**SCOPE, "evaluation_splits": list(SPLITS)},
        "timing_caveat": TIMING_CAVEAT,
        "observations_input": {
            "name": observations.name,
            "sha256": sha256_file(observations),
            "size_bytes": observations.stat().st_size,
        },
        "observations": decode_json_bytes(observations.read_bytes()),
        "evaluations": rows,
    }
    maximum, overlap = _derive(document, document["study_plan_sha256"])
    document["maximum_evaluation_workers_at_once"] = maximum
    document["overlap"] = overlap
    validate_receipt(document, document["study_plan_sha256"])
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--iid-start", required=True)
    parser.add_argument("--compositional-start", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = build_receipt(
        args.study,
        args.observations,
        {"iid-test": args.iid_start, "compositional-test": args.compositional_start},
    )
    write_new_bytes(args.output, canonical_json_bytes(document))
    print(canonical_json_bytes(document).decode())


if __name__ == "__main__":
    main()
