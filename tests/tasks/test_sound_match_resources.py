"""Portable resource exports remain bound to their frozen completed executions."""

import importlib.util
import os
from pathlib import Path

import pytest

from harpy._serialization import canonical_json_bytes, decode_json_bytes, sha256_file

_ROOT = Path(__file__).parents[2] / "research/sound_matching"
_SPEC = importlib.util.spec_from_file_location("resource_receipts", _ROOT / "resource_receipts.py")
resources = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(resources)
_SPEC = importlib.util.spec_from_file_location("archive_study", _ROOT / "archive_study.py")
archive = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(archive)
_SPEC = importlib.util.spec_from_file_location("scheduling", _ROOT / "scheduling.py")
scheduling = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(scheduling)


@pytest.fixture
def completed_study(tmp_path):
    plan = {
        "source": {"sha256": "source", "files": []},
        "runtime": {"python": "fixture"},
        "actors": ["fixed"],
        "splits": {split: {"targets": 1, "sha256": split} for split in resources.SPLITS},
    }
    (tmp_path / "plan.json").write_bytes(canonical_json_bytes(plan))
    plan_sha256 = sha256_file(tmp_path / "plan.json")
    for split in resources.SPLITS:
        bundle = tmp_path / split
        bundle.mkdir()
        manifest = {
            "configuration": {
                "study_plan_sha256": plan_sha256,
                "split": split,
                "dataset_manifest_sha256": split,
            },
            "identity": {"source": plan["source"], "runtime": plan["runtime"]},
        }
        (bundle / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        (bundle / "complete.json").write_bytes(
            canonical_json_bytes(
                {
                    "manifest_sha256": sha256_file(bundle / "manifest.json"),
                    "completed": 1,
                    "expected": 1,
                }
            )
        )
        (tmp_path / f"{split}.log").write_bytes(
            canonical_json_bytes(
                {"complete": True, "completed": 1, "split": split, "wall_seconds": 1.0}
            )
        )
        (tmp_path / f"{split}-time.txt").write_text(
            'Command being timed: "/private/local/python study.py"\n'
            "User time (seconds): 0.8\nSystem time (seconds): 0.1\n"
            "Elapsed (wall clock) time (h:mm:ss or m:ss): 0:01.20\n"
            "Maximum resident set size (kbytes): 2048\nExit status: 0\n"
        )
    for split, end in zip(
        resources.SPLITS,
        ("2026-09-22T19:55:00.105138673Z", "2026-09-22T20:20:00Z", "2026-09-22T20:30:00Z"),
        strict=True,
    ):
        stamp = scheduling._timestamp(end)
        os.utime(tmp_path / f"{split}-time.txt", ns=(stamp, stamp))
    observations = tmp_path / "scheduling-observations.json"
    observations.write_bytes(
        canonical_json_bytes(
            {
                "schema_id": "harpy.sound-match.scheduling-observations/1",
                "study_plan_sha256": plan_sha256,
                "authorization": {
                    "policy": "Fresh headroom; one thread per evaluation worker",
                    "threshold_free_physical_kib": 1572864,
                    "reason": "Additional memory headroom",
                },
                "observations": [
                    {
                        "utc": "2026-09-22T20:09:50Z",
                        "windows_free_physical_kib": 2034560,
                        "phase": "before-compositional-launch",
                    },
                    {
                        "utc": "2026-09-22T20:11:00Z",
                        "workers": [
                            {"split": split, "threads": 1, "rss_kib": 314000}
                            for split in resources.SPLITS[1:]
                        ],
                    },
                ],
            }
        )
    )
    schedule = scheduling.build_receipt(
        tmp_path,
        observations,
        {"iid-test": "2026-09-22T20:00:00Z", "compositional-test": "2026-09-22T20:10:00Z"},
    )
    (tmp_path / "scheduling-receipt.json").write_bytes(canonical_json_bytes(schedule))
    return tmp_path


def test_archive_recomputes_all_resource_values_and_raw_evidence_hashes(completed_study):
    output = completed_study / "resource-receipts.json"
    original = resources.collect(completed_study, output)
    assert original["processes_at_once"] == 2
    assert "/private/local" not in output.read_text()
    assert archive.verify_resource_receipts(completed_study) == _ROOT / "resource_receipts.py"
    original["rows"][1]["peak_rss_mib"] = 999
    output.write_bytes(canonical_json_bytes(original))
    with pytest.raises(ValueError, match="resource receipts differ"):
        archive.verify_resource_receipts(completed_study)


def test_resource_binding_rejects_changed_raw_scheduling_input(completed_study):
    path = completed_study / "scheduling-observations.json"
    observations = decode_json_bytes(path.read_bytes())
    observations["observations"][0]["windows_free_physical_kib"] += 1
    path.write_bytes(canonical_json_bytes(observations))
    with pytest.raises(ValueError, match="scheduling input differs"):
        resources.build_receipts(completed_study)


@pytest.mark.parametrize("mutation", ("foreign_split", "incomplete", "foreign_source"))
def test_export_requires_matching_complete_split_identity(completed_study, mutation):
    if mutation == "foreign_source":
        path = completed_study / "iid-test" / "manifest.json"
        document = decode_json_bytes(path.read_bytes())
        document["identity"]["source"]["sha256"] = "other"
    else:
        path = completed_study / "iid-test.log"
        document = decode_json_bytes(path.read_bytes())
        document["split" if mutation == "foreign_split" else "complete"] = (
            "validation" if mutation == "foreign_split" else False
        )
    path.write_bytes(canonical_json_bytes(document))
    with pytest.raises(ValueError, match=r"resource (progress log|bundle identity)"):
        resources.build_receipts(completed_study)
