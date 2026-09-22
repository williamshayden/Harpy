"""Select declared listening examples without inventing perceptual success labels."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from harpy._serialization import canonical_json_bytes, sha256_file, write_new_bytes
from harpy.tasks.sound_match.datasets import load_manifest
from harpy.tasks.sound_match.durable import iter_records
from harpy.tasks.sound_match.models import _plain
from harpy.tasks.sound_match.results import EpisodeRecord

SELECTION_RULE = (
    "lower median and largest submitted distance per split/actor/cohort; "
    "sort by distance then opaque episode ID; retain all terminal errors and truncations"
)


def select(rows: list[dict]) -> tuple[dict, dict]:
    ordered = sorted(rows, key=lambda row: (row["distance"], row["episode_id"]))
    return ordered[(len(ordered) - 1) // 2], ordered[-1]


def build_index(study: Path) -> dict:
    groups = defaultdict(list)
    denominators = Counter()
    failures = []
    for split in ("validation", "iid-test", "compositional-test"):
        membership = load_manifest(split)
        entries = {entry["episode"]["episode_id"]: entry for entry in membership.entries}
        for payload in iter_records(study / split):
            episode_id, actor = payload["key"]
            entry = entries[episode_id]
            key = split, actor, entry["cohort"]
            denominators[key] += 1
            if (
                payload["target"] is not None
                and payload["target"]["audio_sha256"] != entry["target_audio_sha256"]
            ):
                raise ValueError("listening target differs from frozen membership")
            files = {
                item["role"]: {**item, "path": f"{split}/{item['path']}"}
                for item in payload["artifacts"]
            }
            record = (
                None if payload["record"] is None else EpisodeRecord.from_dict(payload["record"])
            )
            base = {
                "split": split,
                "actor": actor,
                "cohort": entry["cohort"],
                "cell": entry["cell"],
                "episode_id": episode_id,
                "target_file": files.get("target_audio"),
            }
            if payload["status"] != "submitted":
                last = None if record is None else record.last_candidate
                failures.append(
                    {
                        **base,
                        "status": payload["status"],
                        "setup_error": payload["error"],
                        "last_candidate_patch": None if last is None else last.patch.to_dict(),
                        "last_candidate_audio_sha256": None if last is None else last.audio_sha256,
                    }
                )
                continue
            candidate = record.submitted_candidate
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
            select(rows),
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
    document = {
        "schema_id": "harpy.sound-match.study-listening-index/1",
        "study_plan_sha256": sha256_file(study / "plan.json"),
        "selection_rule": SELECTION_RULE,
        "paths_relative_to": "study-directory",
        "listening_review": "not-performed",
        "perceptual_success_cutoff": None,
        "samples": samples,
        "terminal_failures": failures,
    }
    return document


def write_index(study: Path, output: Path) -> dict:
    document = build_index(study)
    write_new_bytes(output, canonical_json_bytes(document))
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = write_index(args.study, args.output)
    print(
        f"Indexed {len(result['samples'])} listening pairs and "
        f"{len(result['terminal_failures'])} terminal failures"
    )


if __name__ == "__main__":
    main()
