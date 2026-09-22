"""Extract portable M2 measurement, membership and listening-index evidence."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from harpy._serialization import (
    canonical_json_bytes,
    decode_json_bytes,
    sha256_file,
    write_new_bytes,
)
from harpy.tasks.sound_match.datasets import (
    DATA_DIRECTORY,
    GENERATOR_ID,
    SELECTION_CRITERION,
    SPLIT_COUNTS,
    load_manifest,
    manifest_summary,
    validate_disjoint,
)
from harpy.tasks.sound_match.results import load_result


def report(pilot: Path, fixtures: Path, output: Path) -> None:
    manifests = tuple(load_manifest(split) for split in SPLIT_COUNTS)
    validate_disjoint(manifests)
    pilot_report = decode_json_bytes((pilot / "pilot.json").read_bytes())
    calibration = decode_json_bytes((fixtures / "calibration.json").read_bytes())
    result = load_result(pilot / "bundle")
    calibration_manifest = next(value for value in manifests if value.split == "calibration")
    if result.episodes != calibration_manifest.episodes:
        raise ValueError("pilot membership differs from frozen calibration")
    if pilot_report["manifest_sha256"] != calibration_manifest.manifest_sha256:
        raise ValueError("pilot report manifest differs")
    fixture_hashes = {
        row[f"{role}_audio_sha256"]
        for row in calibration["rows"]
        for role in ("target", "candidate")
    }
    for manifest in manifests:
        if any(entry["target_audio_sha256"] in fixture_hashes for entry in manifest.entries):
            raise ValueError("handcrafted calibration fixture overlaps a generated target")
    output.mkdir(parents=True, exist_ok=False)
    dataset_audit = {
        "schema_id": "harpy.sound-match.dataset-audit/1",
        "generator_id": GENERATOR_ID,
        "catalog_sha256": sha256_file(DATA_DIRECTORY / "catalog.json"),
        "memberships": [manifest_summary(manifest) for manifest in manifests],
        "cross_split_exact_collisions": 0,
        "handcrafted_fixture_exact_overlap": 0,
        "selection_criterion": SELECTION_CRITERION,
        "final_test_actor_evaluation": "not-performed-at-M2-freeze",
    }
    for filename, document in (
        ("datasets.json", dataset_audit),
        ("calibration.json", calibration),
        ("pilot.json", pilot_report),
        ("pilot-cell-metrics.json", result.metrics),
    ):
        write_new_bytes(output / filename, canonical_json_bytes(document))
    metadata = {entry["episode"]["episode_id"]: entry for entry in calibration_manifest.entries}
    episode_index = {episode.episode_id: index for index, episode in enumerate(result.episodes)}
    grouped = defaultdict(list)
    for index, record in enumerate(result.records):
        candidate = record.submitted_candidate
        if candidate is not None:
            grouped[(metadata[record.episode_id]["cohort"], record.actor_name)].append(
                (candidate.distance, record.episode_id, index, record, candidate)
            )
    manifest_document = decode_json_bytes((pilot / "bundle" / "manifest.json").read_bytes())
    inventory = {entry["path"]: entry for entry in manifest_document["files"]}
    samples = []
    for (cohort, actor), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (row[0], row[1]))
        for label, row in (
            ("median-submitted-distance", rows[(len(rows) - 1) // 2]),
            ("largest-submitted-distance", rows[-1]),
        ):
            _, _, index, record, candidate = row
            target_name = f"audio/target-{episode_index[record.episode_id]:04d}.wav"
            submitted_name = f"audio/submitted-{index:04d}.wav"
            samples.append(
                {
                    "selection": label,
                    "cohort": cohort,
                    "actor": actor,
                    "episode_id": record.episode_id,
                    "cell": metadata[record.episode_id]["cell"],
                    "target_file": inventory[target_name],
                    "submitted_file": inventory[submitted_name],
                    "submitted_patch": candidate.patch.to_dict(),
                    "objective": dict(candidate.objective),
                    "pitch_error_cents": list(candidate.pitch_error_cents),
                }
            )
    # _plain handles the nested immutable objective component maps.
    from harpy.tasks.sound_match.models import _plain

    write_new_bytes(
        output / "listening-index.json",
        canonical_json_bytes(
            _plain(
                {
                    "schema_id": "harpy.sound-match.listening-index/1",
                    "paths_relative_to": "pilot-result-bundle",
                    "listening_review": "not-performed",
                    "selection_rule": (
                        "lower median and largest submitted distance per actor/cohort; "
                        "ties by opaque ID"
                    ),
                    "samples": samples,
                }
            )
        ),
    )
    print(f"Wrote M2 evidence: {output}; {len(samples)} indexed listening pairs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.pilot, args.fixtures, args.output)


if __name__ == "__main__":
    main()
