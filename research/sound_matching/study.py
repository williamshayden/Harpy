"""Freeze and execute matched v2 validation/final studies from a source snapshot.

Freeze only after M0-M4 checks pass. Keep the evaluator source snapshot and every
model artifact immutable across all splits; final results do not select models.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from harpy._journal import artifact_inventory, source_identity
from harpy._serialization import (
    canonical_json_bytes,
    decode_json_bytes,
    sha256_file,
    write_new_bytes,
)
from harpy.tasks.sound_match.actors import builtin_actor_spec
from harpy.tasks.sound_match.datasets import (
    CELL_NAMES,
    DATA_DIRECTORY,
    MASKS,
    SELECTION_CRITERION,
    SPLIT_COUNTS,
    load_manifest,
    validate_disjoint,
)
from harpy.tasks.sound_match.durable import evaluate_bundle, iter_records, read_bundle
from harpy.tasks.sound_match.inverse_artifacts import load_inverse_artifact
from harpy.tasks.sound_match.models import _plain
from harpy.tasks.sound_match.patch import PATCH_FIELDS
from harpy.tasks.sound_match.renderer import render
from harpy.tasks.sound_match.results import audio_hash

STUDY_ID = "harpy.sound-match.frozen-matched-study/1"
SPLITS = ("validation", "iid-test", "compositional-test")
ACTORS = (
    "fixed",
    "pattern-search",
    *(f"random-search-{seed}" for seed in range(3)),
    *(f"{kind}-{seed}" for seed in range(3) for kind in ("inverse", "inverse-refine")),
)
CACHE_BYTES = 16 * 1024 * 1024


def _equal(left, right, label):
    if canonical_json_bytes({"value": _plain(left)}) != canonical_json_bytes(
        {"value": _plain(right)}
    ):
        raise ValueError(f"{label} differs from frozen study")


def _models(root: Path) -> list[dict]:
    rows = []
    for seed in range(3):
        artifact = load_inverse_artifact(root / f"seed{seed}")
        if artifact.manifest["seed"] != seed or artifact.manifest["profile"] != "reference":
            raise ValueError("study requires every reference-profile seed 0, 1 and 2")
        rows.append(
            {
                "seed": seed,
                "manifest_sha256": artifact.manifest_sha256,
                "files": artifact_inventory(artifact.artifact_paths),
            }
        )
    return rows


def identity(models: Path) -> dict:
    manifests = tuple(load_manifest(split) for split in SPLIT_COUNTS)
    validate_disjoint(manifests)
    if any(len(item.entries) != SPLIT_COUNTS[item.split] for item in manifests):
        raise ValueError("study requires complete frozen memberships")
    return {
        "schema_id": STUDY_ID,
        "source": source_identity(),
        "driver_sha256": sha256_file(Path(__file__)),
        "dataset_catalog_sha256": sha256_file(DATA_DIRECTORY / "catalog.json"),
        "splits": {
            item.split: {"sha256": item.manifest_sha256, "targets": len(item.entries)}
            for item in manifests
        },
        "models": _models(models),
        "actors": list(ACTORS),
        "actor_protocols": actor_protocols(),
        "selection_criterion": SELECTION_CRITERION,
        "seed_selection": "all-three-retained; bundled-seed-zero-declared-before-evaluation",
        "fixed_and_pattern_repetitions": 1,
        "random_and_training_seeds": [0, 1, 2],
        "budgets": {"candidate_evaluations": 32, "decisions": 64},
        "cache_max_audio_bytes": CACHE_BYTES,
        "inference_device": "cpu",
        "torch_threads": 1,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": importlib.metadata.version("torch"),
        },
        "final_tests": "report-only-after-M0-M4-and-this-freeze; no-retraining-or-selection",
        "perceptual_success_cutoff": None,
        "listening_review": "not-performed",
    }


def actor_protocols() -> dict:
    protocols = {}
    for name in ACTORS:
        if name.startswith("inverse"):
            seed = int(name.rsplit("-", 1)[1])
            refine = name.startswith("inverse-refine")
            protocols[name] = {
                "kind": "supervised-inverse",
                "algorithm": "inverse-refine" if refine else "inverse",
                "seed": seed,
                "profile": "reference",
                "inference_device": "cpu",
                "refinement": "ordinary-pattern-search-v1" if refine else "none",
            }
        else:
            random = name.startswith("random-search")
            protocols[name] = _plain(
                builtin_actor_spec(
                    "random-search" if random else name,
                    int(name.rsplit("-", 1)[1]) if random else 0,
                ).metadata
            )
    return protocols


def freeze(models: Path, output: Path) -> None:
    """Persist source/model/membership/driver identities before any final test."""
    write_new_bytes(output, canonical_json_bytes(identity(models)))
    print(f"Frozen {output.name}: {sha256_file(output)}", flush=True)


def _actors(models: Path):
    # Imported only on execution, never for metadata freeze or report reading.
    import torch

    from harpy.tasks.sound_match.inverse import inverse_actor_spec

    torch.set_num_threads(1)
    actors = [builtin_actor_spec("fixed"), builtin_actor_spec("pattern-search")]
    actors.extend(
        replace(builtin_actor_spec("random-search", seed), name=f"random-search-{seed}")
        for seed in range(3)
    )
    for seed in range(3):
        for refine in (False, True):
            name = f"{'inverse-refine' if refine else 'inverse'}-{seed}"
            actors.append(inverse_actor_spec(models / f"seed{seed}", refine=refine, name=name))
    if tuple(actor.name for actor in actors) != ACTORS:
        raise AssertionError("actor order differs")
    return tuple(actors)


def verify_rendered_targets(manifest) -> None:
    """Check declared float32 identities before any actor receives a target."""
    for entry, episode in zip(manifest.entries, manifest.episodes, strict=True):
        _equal(
            audio_hash(render(episode.target_patch)),
            entry["target_audio_sha256"],
            f"rendered target audio for {episode.episode_id}",
        )


def verify_recorded_targets(bundle: Path, manifest) -> None:
    """Bind every retained target to its frozen evaluator membership."""
    expected = {
        entry["episode"]["episode_id"]: entry["target_audio_sha256"] for entry in manifest.entries
    }
    for payload in iter_records(bundle):
        episode_id = payload["key"][0]
        if episode_id not in expected:
            raise ValueError("recorded target is outside frozen membership")
        # A target-setup error deliberately has no target audio and remains a
        # terminal error; every produced target must bind the exact declared hash.
        if payload["target"] is not None:
            _equal(
                payload["target"]["audio_sha256"],
                expected[episode_id],
                f"recorded target audio for {episode_id}",
            )


def run(plan: Path, models: Path, split: str, output: Path, resume: bool) -> None:
    if split not in SPLITS:
        raise ValueError("study only evaluates validation and the two frozen final splits")
    frozen = decode_json_bytes(plan.read_bytes())
    _equal(identity(models), frozen, "source/model/membership/driver identity")
    started = time.perf_counter()
    membership = load_manifest(split)
    verify_rendered_targets(membership)

    def progress(event):
        if event["completed"] % 11 == 0 or event["completed"] == event["expected"]:
            print(
                canonical_json_bytes(
                    {**event, "wall_seconds": time.perf_counter() - started}
                ).decode(),
                flush=True,
            )

    actors = _actors(models)
    import torch

    _equal(str(torch.__version__), frozen["runtime"]["torch"], "loaded inference Torch version")
    result = evaluate_bundle(
        membership.episodes,
        actors,
        output,
        resume=resume,
        progress=progress,
        cache_max_audio_bytes=CACHE_BYTES,
        configuration={
            "study_plan_sha256": sha256_file(plan),
            "split": split,
            "dataset_manifest_sha256": frozen["splits"][split]["sha256"],
        },
    )
    verify_recorded_targets(output, membership)
    _equal(identity(models), frozen, "source/model/membership/driver identity after evaluation")
    print(
        canonical_json_bytes(
            {
                "split": split,
                "complete": result.complete,
                "completed": result.completed,
                "wall_seconds": time.perf_counter() - started,
            }
        ).decode(),
        flush=True,
    )


def split_report(bundle: Path, plan: dict, plan_sha256: str, split: str) -> dict:
    report = read_bundle(bundle)
    membership = load_manifest(split)
    verify_recorded_targets(bundle, membership)
    manifest = decode_json_bytes((bundle / "manifest.json").read_bytes())
    _equal(
        manifest["configuration"],
        {
            "study_plan_sha256": plan_sha256,
            "split": split,
            "dataset_manifest_sha256": plan["splits"][split]["sha256"],
        },
        "bundle configuration",
    )
    _equal(manifest["identity"]["source"], plan["source"], "bundle evaluator source")
    _equal(
        manifest["identity"]["runtime"],
        {key: plan["runtime"][key] for key in ("python", "platform", "numpy", "torch")},
        "bundle numerical runtime",
    )
    _equal(
        manifest["identity"]["cache"],
        {"policy": "lru-waveforms-v1", "max_audio_bytes": plan["cache_max_audio_bytes"]},
        "bundle cache policy",
    )
    _equal(
        [actor["name"] for actor in manifest["identity"]["actors"]], plan["actors"], "bundle actors"
    )
    for actor in manifest["identity"]["actors"]:
        expected = plan["actor_protocols"][actor["name"]]
        _equal(
            {key: actor["metadata"].get(key) for key in expected}, expected, "bundle actor protocol"
        )
        if actor["name"].startswith("inverse"):
            seed = int(actor["name"].rsplit("-", 1)[1])
            _equal(
                actor["metadata"]["inference_torch_version"],
                plan["runtime"]["torch"],
                "bundle inverse inference Torch version",
            )
            model = plan["models"][seed]
            _equal(actor["artifact_files"], model["files"], "bundle model files")
            _equal(
                actor["metadata"]["manifest_sha256"],
                model["manifest_sha256"],
                "bundle model manifest",
            )
            _equal(actor["metadata"]["seed"], seed, "bundle model seed")
    _equal(
        manifest["identity"]["episodes"],
        [episode.to_dict() for episode in membership.episodes],
        "bundle membership",
    )
    actors = actor_report(report.metrics["groups"])
    return {
        "split": split,
        "bundle_manifest_sha256": sha256_file(bundle / "manifest.json"),
        "completion_sha256": sha256_file(bundle / "complete.json"),
        "report": report.to_document(),
        "actors": actors,
    }


def actor_report(groups: list[dict]) -> list[dict]:
    """Equal cell weighting; a missing submission makes the macro ineligible."""
    grouped = defaultdict(dict)
    masks = {PATCH_FIELDS: "all", **{fields: name for name, fields in MASKS.items()}}
    for row in groups:
        cell = f"{row['voice_count']}:{masks[tuple(row['editable_fields'])]}"
        if cell in grouped[row["actor"]]:
            raise ValueError("duplicate actor/cell")
        grouped[row["actor"]][cell] = row
    actors = []
    for name in ACTORS:
        cells = grouped[name]
        eligible = set(cells) == set(CELL_NAMES) and all(
            row["episodes"] == row["submitted"] == row["submitted_distance"]["count"]
            for row in cells.values()
        )
        macro = (
            sum(cells[cell]["submitted_distance"]["mean"] for cell in CELL_NAMES) / 8
            if eligible
            else None
        )
        actors.append(
            {
                "actor": name,
                "eligible": eligible,
                "macro_submitted_distance": macro,
                "episodes": sum(row["episodes"] for row in cells.values()),
                "submitted": sum(row["submitted"] for row in cells.values()),
                "errors": sum(row["error"] for row in cells.values()),
                "truncated": sum(row["truncated"] for row in cells.values()),
                "cell_means": {cell: cells[cell]["submitted_distance"]["mean"] for cell in cells},
            }
        )
    return actors


def reference_quality(validation_actors: list[dict]) -> dict:
    validation = {row["actor"]: row for row in validation_actors}
    fixed = validation["fixed"]["macro_submitted_distance"]
    gates = []
    for seed in range(3):
        inverse = validation[f"inverse-{seed}"]
        value = inverse["macro_submitted_distance"]
        gates.append(
            {
                "seed": seed,
                "inverse_macro_distance": value,
                "fixed_macro_distance": fixed,
                "passed": inverse["eligible"] and fixed is not None and value < fixed,
            }
        )
    return {
        "criterion": SELECTION_CRITERION,
        "seeds": gates,
        "all_three_passed": all(row["passed"] for row in gates),
    }


def summarize(plan: Path, bundles: Path, output: Path, validation_only: bool = False) -> dict:
    frozen = decode_json_bytes(plan.read_bytes())
    splits = ("validation",) if validation_only else SPLITS
    results = [split_report(bundles / split, frozen, sha256_file(plan), split) for split in splits]
    document = {
        "schema_id": "harpy.sound-match.matched-study-report/1",
        "study_plan_sha256": sha256_file(plan),
        "splits": results,
        "reference_quality": reference_quality(results[0]["actors"]),
        "final_tests_complete": not validation_only,
        "perceptual_success_cutoff": None,
        "listening_review": "not-performed",
    }
    write_new_bytes(output, canonical_json_bytes(document))
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    freezing = sub.add_parser("freeze")
    freezing.add_argument("--models", type=Path, required=True)
    freezing.add_argument("--output", type=Path, required=True)
    running = sub.add_parser("run")
    running.add_argument("--plan", type=Path, required=True)
    running.add_argument("--models", type=Path, required=True)
    running.add_argument("--split", choices=SPLITS, required=True)
    running.add_argument("--output", type=Path, required=True)
    running.add_argument("--resume", action="store_true")
    reporting = sub.add_parser("report")
    reporting.add_argument("--plan", type=Path, required=True)
    reporting.add_argument("--bundles", type=Path, required=True)
    reporting.add_argument("--output", type=Path, required=True)
    reporting.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    if args.operation == "freeze":
        freeze(args.models, args.output)
    elif args.operation == "run":
        run(args.plan, args.models, args.split, args.output, args.resume)
    else:
        result = summarize(args.plan, args.bundles, args.output, args.validation_only)
        print(canonical_json_bytes(result["reference_quality"]).decode())
    if args.operation != "run" and "torch" in sys.modules:
        raise AssertionError("metadata and report operations must not import Torch")


if __name__ == "__main__":
    main()
