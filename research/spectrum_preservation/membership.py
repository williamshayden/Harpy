"""Freeze fresh source/target combinations before a spectrum-preservation study.

This utility selects membership only. It does not render, evaluate, or train.
Run from the checkout with ``python -m research.spectrum_preservation.membership
OUTPUT.json --seed 20260908`` when the study protocol is ready to freeze.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from harpy.envs.models import TARGET_NOTE_COUNT
from harpy.experiments.models import EpisodeSpec
from harpy.experiments.protocols import (
    CONFIRMATION_DIGEST_SHA256,
    CONFIRMATION_PROTOCOL_ID,
    EXCLUSIONS_DIGEST_SHA256,
    benchmark_episodes,
    confirmation_episodes,
    excluded_historical_pairs,
)
from harpy.learning.artifacts import canonical_json_bytes, write_new_bytes
from harpy.learning.pitch_data import pitch_coordinate_split

MEMBERSHIP_ID = "harpy-spectrum-preservation-membership-v1"
DEFAULT_SELECTION_SEED = 20260908
_PAIRS_PER_TARGET = 8
_SELECTION_TOKEN = MEMBERSHIP_ID + ":{seed}:{partition}:{source}:{target}"
_NUISANCE_TOKEN = "harpy-spectrum-preservation-nuisance-v1:{seed}:{source}:{target}"


def _digest(document: object) -> str:
    # Match the ordinary experiment protocol digest convention.
    return hashlib.sha256(canonical_json_bytes(document).rstrip(b"\n")).hexdigest()


def make_manifest(seed: int = DEFAULT_SELECTION_SEED) -> dict:
    """Select 200 IID, 200 lower, and 200 upper pairs without evaluation."""
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("selection seed must be an integer in 0..2**64-1")
    historical = excluded_historical_pairs()  # validates the packaged inventory digest
    benchmark = {(item.target_note_index, item.source_pitch_cents) for item in benchmark_episodes()}
    if len(benchmark) != 650 or not benchmark <= historical:
        raise ValueError("the historical exclusion inventory must cover all 650 benchmark pairs")
    confirmation = {
        (item.target_note_index, item.source_pitch_cents) for item in confirmation_episodes()
    }
    if len(confirmation) != 1000:
        raise ValueError("confirmation exclusions must contain all 1000 frozen pairs")
    excluded = historical | confirmation
    split = pitch_coordinate_split()
    pools = {
        "iid": split.iid_holdout_coordinates,
        "lower": split.ood_lower_coordinates,
        "upper": split.ood_upper_coordinates,
    }
    episodes = []
    for partition, coordinates in pools.items():
        for target in range(TARGET_NOTE_COUNT):
            available = [source for source in coordinates if (target, source) not in excluded]
            if len(available) < _PAIRS_PER_TARGET:
                raise ValueError(f"insufficient fresh pairs for {partition} target {target}")
            ranked = sorted(
                available,
                key=lambda source: (
                    hashlib.sha256(
                        _SELECTION_TOKEN.format(
                            seed=seed, partition=partition, source=source, target=target
                        ).encode("ascii")
                    ).digest(),
                    source,
                ),
            )
            for index, source in enumerate(ranked[:_PAIRS_PER_TARGET]):
                nuisance = hashlib.sha256(
                    _NUISANCE_TOKEN.format(seed=seed, source=source, target=target).encode("ascii")
                ).digest()
                episodes.append(
                    EpisodeSpec(
                        id=f"spectrum-preservation-{seed}-{partition}-{target:02d}-{index:02d}",
                        source_pitch_cents=source,
                        target_note_index=target,
                        nuisance_seed=int.from_bytes(nuisance[:8], "big"),
                        partition=partition,
                    )
                )
    document = {
        "id": MEMBERSHIP_ID,
        "scope": (
            "fresh combinations in known rendering family, excluding recorded inventory; "
            "not a new audio domain or unseen rendering process"
        ),
        "selection": {
            "algorithm": (
                "ascending SHA256 token bytes, then source coordinate; "
                "without replacement within each target and partition"
            ),
            "seed": seed,
            "token": _SELECTION_TOKEN,
            "targets": list(range(TARGET_NOTE_COUNT)),
            "pairs_per_target_per_partition": _PAIRS_PER_TARGET,
            "additional_difficulty_filter": None,
        },
        "nuisance": {
            "algorithm": "first 8 SHA256 token bytes interpreted as an unsigned big-endian integer",
            "token": _NUISANCE_TOKEN,
            "reuse": "one fixed nuisance seed per episode, shared across actors and conditions",
        },
        "source_split_sha256": split.digest_sha256,
        "exclusions": {
            "historical": {
                "id": "harpy-evaluated-pair-exclusions-v1",
                "resource": "harpy.experiments/data/historical-exclusions-v1.json",
                "digest_sha256": EXCLUSIONS_DIGEST_SHA256,
                "pair_count": len(historical),
            },
            "confirmation": {
                "id": CONFIRMATION_PROTOCOL_ID,
                "resource": "harpy.experiments/data/confirmation-v1.json",
                "digest_sha256": CONFIRMATION_DIGEST_SHA256,
                "pair_count": len(confirmation),
            },
            "union_pair_count": len(excluded),
            "benchmark_pairs_covered_by_historical": len(benchmark),
        },
        "partition_counts": {name: 200 for name in pools},
        "episodes": [episode.to_document() for episode in episodes],
    }
    document["digest_sha256"] = _digest(document)
    return document


def episodes_from_manifest(document: object) -> tuple[EpisodeSpec, ...]:
    """Verify the digest, exact seeded membership, and exclusion identities."""
    if not isinstance(document, dict) or not isinstance(document.get("selection"), dict):
        raise ValueError("membership manifest must contain a selection object")
    expected = make_manifest(document["selection"].get("seed"))
    if canonical_json_bytes(document) != canonical_json_bytes(expected):
        raise ValueError("membership manifest differs from its declared deterministic selection")
    return tuple(EpisodeSpec.from_document(item) for item in expected["episodes"])


def write_manifest(document: dict, path: Path) -> None:
    """Validate and atomically create a membership file; never replace an existing path."""
    episodes_from_manifest(document)
    if not isinstance(path, Path):
        raise ValueError("manifest path must be a Path")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_new_bytes(path, canonical_json_bytes(document))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New membership JSON path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SELECTION_SEED)
    args = parser.parse_args(argv)
    try:
        document = make_manifest(args.seed)
        write_manifest(document, args.output)
    except (ValueError, OSError) as error:
        parser.exit(2, f"membership: {error}\n")
    print(f"Frozen 600 episode pairs: {args.output} ({document['digest_sha256']})")
    return 0


__all__ = [
    "DEFAULT_SELECTION_SEED",
    "MEMBERSHIP_ID",
    "episodes_from_manifest",
    "make_manifest",
    "write_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())
