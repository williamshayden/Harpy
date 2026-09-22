"""Recheck retained calibration evidence without evaluating or tuning any actor."""

from __future__ import annotations

import argparse
import hashlib
import time
from collections import Counter
from pathlib import Path

from harpy._serialization import (
    canonical_json_bytes,
    decode_json_bytes,
    sha256_file,
    write_new_bytes,
)
from harpy.tasks.sound_match.calibration import calibration_pairs, near_equivalence_report
from harpy.tasks.sound_match.datasets import SPLIT_COUNTS, load_manifest, validate_disjoint
from harpy.tasks.sound_match.models import _plain
from harpy.tasks.sound_match.objective import score
from harpy.tasks.sound_match.renderer import render
from harpy.tasks.sound_match.results import _wav_bytes, audio_hash, load_result


def verify(pilot: Path, fixtures: Path, snapshot: Path, output: Path) -> dict:
    started = time.perf_counter()
    report = decode_json_bytes((pilot / "pilot.json").read_bytes())
    source = report["source"]
    for row in source["files"]:
        path = snapshot / "harpy" / row["path"]
        if path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise ValueError(f"retained pilot source differs: {row['path']}")
    result = load_result(pilot / "bundle")
    if _plain(result.source) != source:
        raise ValueError("pilot report source differs from bundle")
    manifests = tuple(load_manifest(split) for split in SPLIT_COUNTS)
    validate_disjoint(manifests)
    calibration = manifests[0]
    if (
        result.episodes != calibration.episodes
        or report["manifest_sha256"] != calibration.manifest_sha256
    ):
        raise ValueError("pilot membership differs from frozen calibration")
    if len(result.records) != 288 or Counter(row.status for row in result.records) != {
        "submitted": 288
    }:
        raise ValueError("retained pilot terminal counts differ")
    if canonical_json_bytes(near_equivalence_report(calibration)) != canonical_json_bytes(
        report["near_equivalence"]
    ):
        raise ValueError("calibration near-equivalence report differs")
    verified = {}
    for manifest in manifests:
        for entry, episode in zip(manifest.entries, manifest.episodes, strict=True):
            if audio_hash(render(episode.target_patch)) != entry["target_audio_sha256"]:
                raise ValueError(f"frozen target rendering differs: {episode.episode_id}")
        verified[manifest.split] = {
            "targets": len(manifest.entries),
            "source_families": len({entry["source_family_id"] for entry in manifest.entries}),
            "manifest_sha256": manifest.manifest_sha256,
        }
    fixture_report = decode_json_bytes((fixtures / "calibration.json").read_bytes())
    pairs = dict((name, (target, candidate)) for name, target, candidate in calibration_pairs())
    for row in fixture_report["rows"]:
        target, candidate = pairs.pop(row["name"])
        if canonical_json_bytes(score(target, candidate)) != canonical_json_bytes(row["objective"]):
            raise ValueError(f"fixture objective differs: {row['name']}")
        for role, audio in (("target", target), ("candidate", candidate)):
            content = _wav_bytes(audio)
            info = row[f"{role}_file"]
            if (fixtures / info["path"]).read_bytes() != content:
                raise ValueError(f"fixture WAV differs: {info['path']}")
            if (
                audio_hash(audio) != row[f"{role}_audio_sha256"]
                or hashlib.sha256(content).hexdigest() != info["sha256"]
            ):
                raise ValueError("fixture audio hash differs")
    if pairs:
        raise ValueError("fixture report is incomplete")
    document = {
        "schema_id": "harpy.sound-match.m2-reverification/1",
        "pilot_source_sha256": source["sha256"],
        "pilot_source_files_verified": len(source["files"]),
        "pilot_manifest_sha256": sha256_file(pilot / "bundle" / "manifest.json"),
        "pilot_report_sha256": sha256_file(pilot / "pilot.json"),
        "strictly_reloaded_actor_episodes": len(result.records),
        "terminal_counts": dict(Counter(row.status for row in result.records)),
        "memberships": verified,
        "all_target_audio_hashes_recomputed": True,
        "cross_split_group_structure_audio_episode_collisions": 0,
        "fixture_pairs_recomputed": len(fixture_report["rows"]),
        "near_equivalence_recomputed": True,
        "final_test_actor_evaluation": "not-performed",
        "listening_review": "not-performed",
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_new_bytes(output, canonical_json_bytes(document))
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pilot", "fixtures", "snapshot", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(
        canonical_json_bytes(verify(args.pilot, args.fixtures, args.snapshot, args.output)).decode()
    )


if __name__ == "__main__":
    main()
