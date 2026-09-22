"""Prepare explicitly validation-only listening examples while final tests wait."""

from __future__ import annotations

import argparse
import importlib.util
from collections import Counter, defaultdict
from pathlib import Path

from harpy._serialization import (
    canonical_json_bytes,
    decode_json_bytes,
    sha256_file,
    write_new_bytes,
)
from harpy.tasks.sound_match.datasets import load_manifest
from harpy.tasks.sound_match.durable import iter_records
from harpy.tasks.sound_match.models import _plain
from harpy.tasks.sound_match.results import EpisodeRecord


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot_file(evaluator: Path, snapshot: dict, name: str) -> Path:
    rows = [row for row in snapshot["files"] if row["path"] == name]
    if len(rows) != 1:
        raise ValueError("frozen evaluator inventory must identify the listening inputs once")
    path, row = evaluator / name, rows[0]
    if path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
        raise ValueError("listening input differs from frozen evaluator inventory")
    return path


def _preflight(study: Path, selector: Path, evaluator: Path):
    """Bind selection to the frozen reader and a fully recomputed validation report."""
    plan_path = study / "plan.json"
    plan = decode_json_bytes(plan_path.read_bytes())
    plan_hash = sha256_file(plan_path)
    snapshot = decode_json_bytes((evaluator / "snapshot-manifest.json").read_bytes())
    if snapshot["schema_id"] != "harpy.source-snapshot/1":
        raise ValueError("listening requires a frozen evaluator source snapshot")
    driver = _snapshot_file(evaluator, snapshot, "research/sound_matching/study.py")
    frozen_selector = _snapshot_file(evaluator, snapshot, "research/sound_matching/listening.py")
    if sha256_file(driver) != plan["driver_sha256"]:
        raise ValueError("listening report reader differs from frozen study driver")
    if sha256_file(selector) != sha256_file(frozen_selector):
        raise ValueError("listening selector differs from frozen evaluator snapshot")
    reader = _load_module("_frozen_validation_reader", driver)
    membership = load_manifest("validation")
    if (
        plan["schema_id"] != reader.STUDY_ID
        or plan["actors"] != list(reader.ACTORS)
        or plan["splits"]["validation"] != {"sha256": membership.manifest_sha256, "targets": 192}
        or len(membership.episodes) != 192
    ):
        raise ValueError("validation listening requires the full frozen 192-target, 11-actor study")
    result = reader.split_report(study / "validation", plan, plan_hash, "validation")
    if (
        not result["report"]["complete"]
        or result["report"]["completed"] != 192 * 11
        or len(result["actors"]) != 11
        or any(actor["episodes"] != 192 for actor in result["actors"])
    ):
        raise ValueError("validation listening requires the complete frozen validation split")
    expected = {
        "schema_id": "harpy.sound-match.matched-study-report/1",
        "study_plan_sha256": plan_hash,
        "splits": [result],
        "reference_quality": reader.reference_quality(result["actors"]),
        "final_tests_complete": False,
        "perceptual_success_cutoff": None,
        "listening_review": "not-performed",
    }
    retained = decode_json_bytes((study / "validation-report.json").read_bytes())
    if canonical_json_bytes(retained) != canonical_json_bytes(expected):
        raise ValueError("validation report differs from strict frozen study recomputation")
    bindings = {
        plan_path: plan_hash,
        study / "validation/manifest.json": result["bundle_manifest_sha256"],
        study / "validation/complete.json": result["completion_sha256"],
        selector: sha256_file(frozen_selector),
    }
    return _load_module("_frozen_listening_selector", selector), membership, bindings


def prepare(study: Path, selector: Path, output: Path, *, evaluator: Path) -> dict:
    module, membership, bindings = _preflight(study, selector, evaluator)
    entries = {entry["episode"]["episode_id"]: entry for entry in membership.entries}
    groups, denominators, failures = defaultdict(list), Counter(), []
    for payload in iter_records(study / "validation"):
        episode_id, actor = payload["key"]
        entry = entries[episode_id]
        key = actor, entry["cohort"]
        denominators[key] += 1
        if (
            payload["target"] is not None
            and payload["target"]["audio_sha256"] != entry["target_audio_sha256"]
        ):
            raise ValueError("validation listening target differs from frozen membership")
        files = {
            item["role"]: {**item, "path": "validation/" + item["path"]}
            for item in payload["artifacts"]
        }
        base = {
            "split": "validation",
            "actor": actor,
            "cohort": entry["cohort"],
            "cell": entry["cell"],
            "episode_id": episode_id,
            "target_file": files.get("target_audio"),
        }
        if payload["status"] != "submitted":
            failures.append({**base, "status": payload["status"], "setup_error": payload["error"]})
            continue
        candidate = EpisodeRecord.from_dict(payload["record"]).submitted_candidate
        groups[key].append(
            {
                **base,
                "distance": candidate.distance,
                "submitted_file": files["submitted_audio"],
                "submitted_patch": candidate.patch.to_dict(),
                "objective": _plain(candidate.objective),
                "pitch_error_cents": list(candidate.pitch_error_cents),
            }
        )
    samples = []
    for key, rows in sorted(groups.items()):
        for label, chosen in zip(
            ("lower-median-submitted-distance", "largest-submitted-distance"),
            module.select(rows),
            strict=True,
        ):
            samples.append(
                {
                    **chosen,
                    "selection": label,
                    "submitted_in_cohort": len(rows),
                    "episodes_in_cohort": denominators[key],
                }
            )
    result = {
        "schema_id": "harpy.sound-match.validation-listening-index/1",
        "study_plan_sha256": sha256_file(study / "plan.json"),
        "selector_sha256": sha256_file(selector),
        "selection_rule": module.SELECTION_RULE,
        "partial_study": True,
        "splits": ["validation"],
        "final_test_audio_included": False,
        "paths_relative_to": "study-directory",
        "listening_review": "not-performed",
        "perceptual_success_cutoff": None,
        "samples": samples,
        "terminal_failures": failures,
    }
    if any(sha256_file(path) != digest for path, digest in bindings.items()):
        raise ValueError("frozen validation inputs changed during listening selection")
    write_new_bytes(output, canonical_json_bytes(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "selector", "output", "evaluator"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.study, args.selector, args.output, evaluator=args.evaluator)
    print(f"Indexed {len(result['samples'])} validation-only listening pairs")


if __name__ == "__main__":
    main()
