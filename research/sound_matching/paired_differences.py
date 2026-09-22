"""Derive matched submitted-distance differences from the completed frozen study.

This postprocessor reads strict bundles without models, synthesis or new actor
evaluation. Negative differences mean the named actor has lower distance than its
comparator. Terminal failures stay in denominators and are never assigned scores.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from harpy._journal import source_identity
from harpy._serialization import (
    canonical_json_bytes,
    decode_json_bytes,
    sha256_file,
    write_new_bytes,
)
from harpy.tasks.sound_match.datasets import (
    CELL_NAMES,
    SELECTION_CRITERION,
    SPLIT_COUNTS,
    load_manifest,
)
from harpy.tasks.sound_match.durable import iter_records
from harpy.tasks.sound_match.models import _plain
from harpy.tasks.sound_match.results import EpisodeRecord

SCHEMA_ID = "harpy.sound-match.paired-differences/1"
STUDY_SCHEMA_ID = "harpy.sound-match.frozen-matched-study/1"
SPLITS = ("validation", "iid-test", "compositional-test")
ACTORS = (
    "fixed",
    "pattern-search",
    *(f"random-search-{seed}" for seed in range(3)),
    *(f"{kind}-{seed}" for seed in range(3) for kind in ("inverse", "inverse-refine")),
)
STATUSES = ("submitted", "error", "truncated")


def _equal(actual, expected, label):
    if canonical_json_bytes({"value": _plain(actual)}) != canonical_json_bytes(
        {"value": _plain(expected)}
    ):
        raise ValueError(f"{label} differs from the frozen study")


def comparisons() -> tuple[tuple[str, str], ...]:
    """Fixed comparisons, with every learned seed retained independently."""
    pairs = [(name, "fixed") for name in ACTORS if name != "fixed"]
    for seed in range(3):
        pairs.append((f"inverse-refine-{seed}", f"inverse-{seed}"))
        for kind in ("inverse", "inverse-refine"):
            pairs.extend(
                (f"{kind}-{seed}", baseline)
                for baseline in ("pattern-search", f"random-search-{seed}")
            )
    return tuple(pairs)


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def paired_group(episode_ids: tuple[str, ...], records: dict, actor: str, comparator: str) -> dict:
    """Pair only exact episode IDs, retaining all terminal-status combinations."""
    if len(set(episode_ids)) != len(episode_ids) or not episode_ids:
        raise ValueError("paired membership must be nonempty and unique")
    statuses = Counter()
    differences, actor_costs, comparator_costs = [], [], []
    for episode_id in episode_ids:
        try:
            left, right = records[(episode_id, actor)], records[(episode_id, comparator)]
        except KeyError as error:
            raise ValueError("a comparison is missing an expected actor/episode") from error
        if left["status"] not in STATUSES or right["status"] not in STATUSES:
            raise ValueError("paired row has an unknown terminal status")
        statuses[(left["status"], right["status"])] += 1
        if left["status"] == right["status"] == "submitted":
            if left["distance"] is None or right["distance"] is None:
                raise ValueError("submitted pair lacks submitted distance")
            differences.append(left["distance"] - right["distance"])
            actor_costs.append(left["evaluations"])
            comparator_costs.append(right["evaluations"])
    return {
        "expected_episodes": len(episode_ids),
        "both_submitted": len(differences),
        "excluded_from_distance_difference": len(episode_ids) - len(differences),
        "status_pairs": {
            f"{left}|{right}": statuses[(left, right)] for left in STATUSES for right in STATUSES
        },
        "submitted_distance_difference": _distribution(differences),
        "actor_evaluations_on_submitted_pairs": _distribution(actor_costs),
        "comparator_evaluations_on_submitted_pairs": _distribution(comparator_costs),
    }


def paired_comparisons(entries: tuple, records: dict, pairs=None) -> list[dict]:
    cells = {
        cell: tuple(entry["episode"]["episode_id"] for entry in entries if entry["cell"] == cell)
        for cell in CELL_NAMES
    }
    cohorts = {
        cohort: tuple(
            entry["episode"]["episode_id"] for entry in entries if entry["cohort"] == cohort
        )
        for cohort in sorted({entry["cohort"] for entry in entries})
    }
    rows = []
    for actor, comparator in comparisons() if pairs is None else pairs:
        by_cell = {
            cell: paired_group(ids, records, actor, comparator) for cell, ids in cells.items()
        }
        by_cohort = {
            cohort: paired_group(ids, records, actor, comparator) for cohort, ids in cohorts.items()
        }
        complete_pairs = all(
            row["expected_episodes"] == row["both_submitted"] for row in by_cell.values()
        )
        rows.append(
            {
                "actor": actor,
                "comparator": comparator,
                "all_expected_pairs_submitted": complete_pairs,
                "equal_eight_cell_mean_difference": (
                    sum(
                        by_cell[cell]["submitted_distance_difference"]["mean"]
                        for cell in CELL_NAMES
                    )
                    / len(CELL_NAMES)
                    if complete_pairs
                    else None
                ),
                "cells": by_cell,
                "cohorts": by_cohort,
            }
        )
    return rows


def _bundle_identity(document: dict, plan: dict, plan_sha256: str, split: str, membership) -> None:
    identity = document["identity"]
    _equal(
        document["configuration"],
        {
            "study_plan_sha256": plan_sha256,
            "split": split,
            "dataset_manifest_sha256": plan["splits"][split]["sha256"],
        },
        "bundle configuration",
    )
    _equal(identity["source"], plan["source"], "bundle evaluator source")
    runtime_keys = ("python", "platform", "numpy", "torch")
    _equal(
        identity["runtime"],
        {name: plan["runtime"][name] for name in runtime_keys},
        "bundle runtime",
    )
    _equal(
        identity["cache"],
        {"policy": "lru-waveforms-v1", "max_audio_bytes": plan["cache_max_audio_bytes"]},
        "bundle cache",
    )
    _equal([row["name"] for row in identity["actors"]], plan["actors"], "ordered actors")
    _equal(
        identity["episodes"],
        [episode.to_dict() for episode in membership.episodes],
        "exact dataset membership",
    )
    for actor in identity["actors"]:
        expected = plan["actor_protocols"][actor["name"]]
        _equal({name: actor["metadata"].get(name) for name in expected}, expected, "actor protocol")
        if actor["name"].startswith("inverse"):
            seed = int(actor["name"].rsplit("-", 1)[1])
            model = plan["models"][seed]
            _equal(actor["artifact_files"], model["files"], "model payload identities")
            _equal(actor["metadata"]["manifest_sha256"], model["manifest_sha256"], "model manifest")
            _equal(actor["metadata"]["seed"], seed, "model seed")
            _equal(
                actor["metadata"]["inference_torch_version"],
                plan["runtime"]["torch"],
                "inference Torch version",
            )


def read_split(bundle: Path, plan: dict, plan_sha256: str, split: str, membership) -> tuple:
    """Strictly scan one completed bundle once, retaining only compact paired data."""
    if not (bundle / "complete.json").is_file():
        raise ValueError("paired report requires completed input bundles")
    _equal(membership.manifest_sha256, plan["splits"][split]["sha256"], "dataset manifest")
    _equal(len(membership.entries), plan["splits"][split]["targets"], "dataset size")
    manifest_bytes = (bundle / "manifest.json").read_bytes()
    manifest = decode_json_bytes(manifest_bytes)
    _bundle_identity(manifest, plan, plan_sha256, split, membership)
    completion_bytes = (bundle / "complete.json").read_bytes()
    entries = {entry["episode"]["episode_id"]: entry for entry in membership.entries}
    records, failures = {}, []
    for payload in iter_records(bundle):
        episode_id, actor = payload["key"]
        key = episode_id, actor
        if episode_id not in entries or actor not in plan["actors"] or key in records:
            raise ValueError("paired record is duplicate or outside frozen membership")
        entry = entries[episode_id]
        if payload["target"] is not None:
            _equal(payload["target"]["audio_sha256"], entry["target_audio_sha256"], "target audio")
        record = None if payload["record"] is None else EpisodeRecord.from_dict(payload["record"])
        metrics = None if record is None else record.metrics
        records[key] = {
            "status": payload["status"],
            "distance": None if metrics is None else metrics["submitted_distance"],
            "evaluations": None if metrics is None else metrics["evaluations"],
        }
        if payload["status"] != "submitted":
            failures.append(
                {
                    "episode_id": episode_id,
                    "actor": actor,
                    "cohort": entry["cohort"],
                    "cell": entry["cell"],
                    "status": payload["status"],
                    "setup_error": payload["error"],
                    "reason": None if record is None else record.reason,
                    "error_type": None if record is None else record.error_type,
                    "last_distance_diagnostic": None
                    if metrics is None
                    else metrics["last_distance"],
                }
            )
    if set(records) != {(episode_id, actor) for episode_id in entries for actor in plan["actors"]}:
        raise ValueError("paired report requires every frozen actor/episode record")
    if (bundle / "manifest.json").read_bytes() != manifest_bytes or (
        bundle / "complete.json"
    ).read_bytes() != completion_bytes:
        raise ValueError("bundle identity changed during paired readout")
    binding = {
        "split": split,
        "dataset_manifest_sha256": membership.manifest_sha256,
        "bundle_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "bundle_completion_sha256": hashlib.sha256(completion_bytes).hexdigest(),
        "expected_records": len(entries) * len(plan["actors"]),
        "retained_records": len(records),
    }
    return binding, records, failures


def build_report(study: Path) -> dict:
    plan_bytes = (study / "plan.json").read_bytes()
    plan = decode_json_bytes(plan_bytes)
    _equal(plan["schema_id"], STUDY_SCHEMA_ID, "study schema")
    _equal(plan["actors"], list(ACTORS), "complete three-seed actor membership")
    _equal(plan["selection_criterion"], SELECTION_CRITERION, "selection criterion")
    _equal(plan["budgets"], {"candidate_evaluations": 32, "decisions": 64}, "matched budgets")
    _equal([model["seed"] for model in plan["models"]], [0, 1, 2], "model seeds")
    if any(not (study / split / "complete.json").is_file() for split in SPLITS):
        raise ValueError("paired report requires all three completed study splits")
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    reports = []
    for split in SPLITS:
        membership = load_manifest(split)
        _equal(len(membership.entries), SPLIT_COUNTS[split], "complete frozen split")
        binding, records, failures = read_split(study / split, plan, plan_sha256, split, membership)
        reports.append(
            {
                **binding,
                "comparisons": paired_comparisons(membership.entries, records),
                "terminal_failures": failures,
            }
        )
    if (study / "plan.json").read_bytes() != plan_bytes:
        raise ValueError("study plan changed during paired readout")
    return {
        "schema_id": SCHEMA_ID,
        "study_plan_sha256": plan_sha256,
        "postprocessor_source_sha256": sha256_file(Path(__file__)),
        "reader_source": source_identity(),
        "reader_runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "sign_convention": "actor minus comparator; negative means lower submitted distance",
        "denominator_policy": (
            "all expected episodes remain in status-pair counts; numerical differences use only "
            "pairs with two explicit submissions; no imputation for errors or truncation; "
            "equal-eight-cell mean requires every expected pair to submit"
        ),
        "distribution_policy": (
            "NumPy arithmetic mean, median, percentile(90) with linear interpolation"
        ),
        "comparisons_are": "descriptive paired differences; no significance or perceptual claims",
        "splits": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.study)
    write_new_bytes(args.output, canonical_json_bytes(report))
    if "torch" in sys.modules or "stable_baselines3" in sys.modules:
        raise AssertionError("paired readout must not import optional model packages")
    print(f"Wrote {len(SPLITS)} splits and {len(comparisons())} comparisons per split", flush=True)


if __name__ == "__main__":
    main()
