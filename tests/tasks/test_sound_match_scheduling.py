"""Scheduling claims distinguish observed overlap from reconstructed process lifetimes."""

import copy
import importlib.util
import os
from pathlib import Path

import pytest

from harpy._serialization import canonical_json_bytes, sha256_file

_ROOT = Path(__file__).parents[2] / "research/sound_matching"
_SPEC = importlib.util.spec_from_file_location("scheduling", _ROOT / "scheduling.py")
scheduling = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scheduling)


@pytest.fixture
def completed(tmp_path):
    (tmp_path / "plan.json").write_bytes(canonical_json_bytes({"fixture": "schedule"}))
    starts = {"iid-test": "2026-09-22T20:00:00Z", "compositional-test": "2026-09-22T20:10:00Z"}
    for split, end in zip(
        scheduling.SPLITS,
        (
            "2026-09-22T19:55:00.105138673Z",
            "2026-09-22T21:00:00.500000001Z",
            "2026-09-22T20:50:00.700000009Z",
        ),
        strict=True,
    ):
        path = tmp_path / f"{split}-time.txt"
        path.write_text(
            'Command being timed: "/private/fixture"\n'
            "User time (seconds): 0.8\nSystem time (seconds): 0.1\n"
            "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:01.20\n"
            "Maximum resident set size (kbytes): 2048\nExit status: 0\n"
        )
        stamp = scheduling._timestamp(end)
        os.utime(path, ns=(stamp, stamp))
    workers = [{"split": split, "threads": 1, "rss_kib": 314000} for split in scheduling.SPLITS[1:]]
    observations = {
        "schema_id": "harpy.sound-match.scheduling-observations/1",
        "study_plan_sha256": sha256_file(tmp_path / "plan.json"),
        "authorization": {
            "policy": "Fresh headroom of 1.5 GiB; one thread per worker",
            "threshold_free_physical_kib": 1572864,
            "reason": "Additional memory headroom",
        },
        "observations": [
            {
                "utc": "2026-09-22T20:09:50.0896204Z",
                "windows_free_physical_kib": 2034560,
                "phase": "before-compositional-launch",
            },
            {"utc": "2026-09-22T20:10:39Z", "workers": copy.deepcopy(workers)},
            {
                "utc": "2026-09-22T20:12:40.2987166Z",
                "windows_free_physical_kib": 1602140,
                "workers": copy.deepcopy(workers),
            },
        ],
    }
    observation_path = tmp_path / "scheduling-observations.json"
    observation_path.write_bytes(canonical_json_bytes(observations))
    return tmp_path, observation_path, starts


def test_receipt_preserves_unknown_start_and_observation_precision(completed):
    document = scheduling.build_receipt(*completed)
    assert scheduling.validate_receipt(document, document["study_plan_sha256"]) == 2
    validation = document["evaluations"][0]
    assert validation["observed_start_utc"] is None
    assert validation["start_source"] is None
    assert validation["completion_report"]["written_utc"] == "2026-09-22T19:55:00.105138673Z"
    assert validation["wall_seconds"] == 1.2
    assert document["overlap"]["observed_span_seconds"] == 121.2987166
    assert document["overlap"]["upper_bound_seconds"] == 2400.700000009
    assert "/private/fixture" not in str(document)


@pytest.mark.parametrize(
    "mutation",
    (
        "invent_validation_start",
        "plan_mismatch",
        "observation_plan_mismatch",
        "nonutc",
        "bad_calendar",
        "reversed_interval",
        "validation_overlaps",
        "worker_before_start",
        "worker_after_completion",
        "duplicate_worker",
        "unknown_worker",
        "threads",
        "negative_rss",
        "missing_overlap",
        "stale_headroom",
        "insufficient_headroom",
        "out_of_order",
        "declared_maximum",
        "declared_overlap",
        "boolean_timing",
        "nonfinite_timing",
        "missing_field",
    ),
)
def test_invalid_scheduling_evidence_is_rejected(completed, mutation):
    document = scheduling.build_receipt(*completed)
    rows = document["evaluations"]
    snapshots = document["observations"]["observations"]
    if mutation == "invent_validation_start":
        rows[0]["observed_start_utc"] = "2026-09-22T19:00:00Z"
    elif mutation == "plan_mismatch":
        document["study_plan_sha256"] = "f" * 64
    elif mutation == "observation_plan_mismatch":
        document["observations"]["study_plan_sha256"] = "f" * 64
    elif mutation == "nonutc":
        rows[1]["observed_start_utc"] = "2026-09-22T20:00:00+00:00"
    elif mutation == "bad_calendar":
        rows[1]["observed_start_utc"] = "2026-09-32T20:00:00Z"
    elif mutation == "reversed_interval":
        rows[1]["observed_start_utc"] = "2026-09-22T22:00:00Z"
    elif mutation == "validation_overlaps":
        rows[0]["completion_report"]["written_utc"] = "2026-09-22T20:15:00Z"
    elif mutation in ("worker_before_start", "worker_after_completion"):
        snapshots[-1]["utc"] = (
            "2026-09-22T19:59:00Z" if mutation == "worker_before_start" else "2026-09-22T22:00:00Z"
        )
    elif mutation == "duplicate_worker":
        snapshots[1]["workers"][1]["split"] = "iid-test"
    elif mutation == "unknown_worker":
        snapshots[1]["workers"][1]["split"] = "host-app"
    elif mutation == "threads":
        snapshots[1]["workers"][1]["threads"] = 2
    elif mutation == "negative_rss":
        snapshots[1]["workers"][1]["rss_kib"] = -1
    elif mutation == "missing_overlap":
        del snapshots[1:]
    elif mutation == "stale_headroom":
        snapshots[0]["utc"] = "2026-09-22T20:00:00Z"
    elif mutation == "insufficient_headroom":
        snapshots[0]["windows_free_physical_kib"] = 1500000
    elif mutation == "out_of_order":
        snapshots.reverse()
    elif mutation == "declared_maximum":
        document["maximum_evaluation_workers_at_once"] = 1
    elif mutation == "declared_overlap":
        document["overlap"]["observed_span_seconds"] += 100
    elif mutation == "boolean_timing":
        rows[1]["wall_seconds"] = True
    elif mutation == "nonfinite_timing":
        rows[1]["wall_seconds"] = float("nan")
    else:
        del document["observations"]
    expected = sha256_file(completed[0] / "plan.json")
    with pytest.raises(ValueError):
        scheduling.validate_receipt(document, expected)


def test_builder_requires_completed_reports_and_only_observed_final_starts(completed):
    study, observations, starts = completed
    with pytest.raises(ValueError, match="only the two observed"):
        scheduling.build_receipt(study, observations, {**starts, "validation": starts["iid-test"]})
    (study / "iid-test-time.txt").write_text("")
    with pytest.raises(ValueError, match="successful completed process"):
        scheduling.build_receipt(study, observations, starts)


def test_validation_does_not_need_retained_files(completed):
    document = scheduling.build_receipt(*completed)
    for path in completed[0].iterdir():
        path.unlink()
    assert scheduling.validate_receipt(document, document["study_plan_sha256"]) == 2


def test_scope_cannot_be_mutated_through_returned_document(completed):
    document = scheduling.build_receipt(*completed)
    document["scope"]["invocations_per_split"] = True
    with pytest.raises(ValueError):
        scheduling.validate_receipt(document, document["study_plan_sha256"])
    assert scheduling.SCOPE["invocations_per_split"] == 1
