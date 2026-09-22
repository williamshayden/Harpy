"""Frozen evaluator memberships and bounded, training-only label iteration.

Manifests contain hidden target labels. Pass only their EpisodeSpec objects to the
ordinary runner; actor observations never receive dataset/group metadata.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from harpy._serialization import canonical_json_bytes, decode_json_bytes, write_new_bytes
from harpy.tasks.sound_match.models import TASK_ID, EpisodeSpec, _freeze, _plain
from harpy.tasks.sound_match.objective import OBJECTIVE_ID
from harpy.tasks.sound_match.patch import PATCH_FIELDS, Patch, default_patch
from harpy.tasks.sound_match.renderer import RENDERER_ID, render
from harpy.tasks.sound_match.results import audio_hash

DATASET_SCHEMA_ID = "harpy.sound-match.dataset/1"
GENERATOR_ID = "harpy-grouped-sound-targets-v1"
GENERATION_SEED = 20260916
SPLIT_COUNTS = {
    "calibration": 96,
    "train": 2016,
    "validation": 192,
    "iid-test": 192,
    "compositional-test": 192,
}
MASKS = {
    "pitch": ("notes_cents",),
    "envelope": ("attack_ms", "decay_ms", "sustain_db", "release_ms"),
    "brightness": ("brightness_hz",),
}
TRAIN_CHORDS = {"major": (0, 400, 700), "minor": (0, 300, 700), "suspended": (0, 200, 700)}
HELDOUT_CHORDS = {"diminished": (0, 300, 600), "augmented": (0, 400, 800)}
CELL_NAMES = tuple(f"{voices}:{mask}" for voices in (1, 3) for mask in ("all", *MASKS))
SELECTION_CRITERION = {
    "id": "equal-eight-cell-validation-audio-distance-v1",
    "split": "validation",
    "cells": list(CELL_NAMES),
    "aggregation": "arithmetic mean of eight cell means of submitted distance",
    "missing_or_unsubmitted": "ineligible checkpoint",
    "tie_break": "earliest checkpoint",
    "reference_quality": "all three seeds strictly below fixed actor macro distance",
    "bundled_seed": 0,
    "perceptual_success_cutoff": None,
}
DATA_DIRECTORY = Path(__file__).with_name("data")


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(_plain(value))).hexdigest()


def structural_patch(patch: Patch) -> Patch:
    """Alias rule v1: decay is inactive when sustain equals the attack peak."""
    return replace(patch, decay_ms=100) if patch.sustain_db == 0 else patch


def structural_hash(patch: Patch) -> str:
    return _digest(structural_patch(patch).to_dict())


def chord_class(notes: tuple[int, ...]) -> tuple[int, ...]:
    """Pitch-class orbit under transposition and voice inversion, in cents."""
    pitch_classes = sorted({note % 1200 for note in notes})
    return min(
        tuple(sorted((note - root) % 1200 for note in pitch_classes)) for root in pitch_classes
    )


def heldout_combination(patch: Patch) -> bool:
    # A joint target combination: each constituent remains represented in train.
    return patch.waveform == "square" and patch.attack_ms >= 160


def _hash_string(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"invalid {label}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"invalid {label}") from exc
    return value


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    split: str
    entries: tuple[Mapping, ...]
    generation: Mapping
    manifest_sha256: str

    @property
    def episodes(self) -> tuple[EpisodeSpec, ...]:
        return tuple(EpisodeSpec.from_dict(entry["episode"]) for entry in self.entries)

    def to_dict(self) -> dict:
        return {
            "schema_id": DATASET_SCHEMA_ID,
            "generator_id": GENERATOR_ID,
            "task_id": TASK_ID,
            "renderer_id": RENDERER_ID,
            "objective_id": OBJECTIVE_ID,
            "split": self.split,
            "generation": _plain(self.generation),
            "entries": _plain(self.entries),
        }

    @classmethod
    def from_dict(cls, document: Mapping) -> DatasetManifest:
        fields = {
            "schema_id",
            "generator_id",
            "task_id",
            "renderer_id",
            "objective_id",
            "split",
            "generation",
            "entries",
        }
        if not isinstance(document, Mapping) or set(document) != fields:
            raise ValueError("dataset manifest has missing or unexpected fields")
        for name, expected in {
            "schema_id": DATASET_SCHEMA_ID,
            "generator_id": GENERATOR_ID,
            "task_id": TASK_ID,
            "renderer_id": RENDERER_ID,
            "objective_id": OBJECTIVE_ID,
        }.items():
            if document[name] != expected:
                raise ValueError(f"unsupported dataset {name}")
        if (
            not isinstance(document["split"], str)
            or document["split"] not in SPLIT_COUNTS
            or not isinstance(document["entries"], (list, tuple))
        ):
            raise ValueError("invalid dataset split or entries")
        entries = tuple(document["entries"])
        if len(entries) < 24 or len(entries) % 24:
            raise ValueError("dataset count must be a positive multiple of 24")
        ids, structures, audios = set(), set(), set()
        families: dict[str, list[Mapping]] = {}
        required = {
            "episode",
            "source_family_id",
            "cohort",
            "cell",
            "mask",
            "chord_family",
            "structural_sha256",
            "target_audio_sha256",
        }
        for entry in entries:
            if not isinstance(entry, Mapping) or set(entry) != required:
                raise ValueError("dataset entry has missing or unexpected fields")
            episode = EpisodeSpec.from_dict(entry["episode"])
            family = entry["source_family_id"]
            if not isinstance(family, str) or not family.startswith("g_") or len(family) != 26:
                raise ValueError("invalid source family ID")
            if not episode.episode_id.startswith("e_") or len(episode.episode_id) != 26:
                raise ValueError("episode ID must be opaque")
            if any(
                character not in "0123456789abcdef"
                for character in family[2:] + episode.episode_id[2:]
            ):
                raise ValueError("opaque IDs must contain lowercase hexadecimal digits")
            mask = entry["mask"]
            if not isinstance(mask, str):
                raise ValueError("dataset mask must be a string")
            expected_mask = PATCH_FIELDS if mask == "all" else MASKS.get(mask)
            if expected_mask is None or episode.editable_fields != expected_mask:
                raise ValueError("dataset mask differs from episode")
            cohort = ("single" if episode.voice_count == 1 else "chord") + (
                "-reconstruction" if mask == "all" else "-shaping"
            )
            if entry["cohort"] != cohort or entry["cell"] != f"{episode.voice_count}:{mask}":
                raise ValueError("dataset cohort or cell differs from episode")
            if mask == "all" and episode.initial_patch != default_patch(episode.voice_count):
                raise ValueError("reconstruction initial patch must equal the public default")
            if episode.render_budget != 32 or episode.decision_budget != 64:
                raise ValueError("frozen dataset budgets are 32 evaluations and 64 decisions")
            structure = _hash_string(entry["structural_sha256"], "structural hash")
            audio = _hash_string(entry["target_audio_sha256"], "audio hash")
            if structural_hash(episode.target_patch) != structure:
                raise ValueError("target structural hash differs")
            if episode.episode_id in ids or structure in structures or audio in audios:
                raise ValueError("duplicate dataset target identity or alias")
            ids.add(episode.episode_id)
            structures.add(structure)
            audios.add(audio)
            families.setdefault(family, []).append(entry)
            heldout = document["split"] == "compositional-test"
            if heldout_combination(episode.target_patch) != heldout:
                raise ValueError("target parameter combination violates split holdout")
            if episode.voice_count == 1:
                if entry["chord_family"] is not None:
                    raise ValueError("single note has no chord family")
            else:
                allowed = HELDOUT_CHORDS if heldout else TRAIN_CHORDS
                if (
                    not isinstance(entry["chord_family"], str)
                    or entry["chord_family"] not in allowed
                    or chord_class(episode.target_patch.notes_cents)
                    != chord_class(allowed[entry["chord_family"]])
                ):
                    raise ValueError("target chord family violates split holdout")
        for family_entries in families.values():
            if (
                len(family_entries) != 2
                or sum(entry["mask"] == "all" for entry in family_entries) != 1
            ):
                raise ValueError("source family must pair reconstruction and shaping")
            base = next(entry for entry in family_entries if entry["mask"] == "all")
            shaped = next(entry for entry in family_entries if entry["mask"] != "all")
            if (
                EpisodeSpec.from_dict(base["episode"]).target_patch
                != EpisodeSpec.from_dict(shaped["episode"]).initial_patch
            ):
                raise ValueError("shaping initial patch must be its source family base")
        generation = document["generation"]
        expected_counts = {
            cell: len(entries) // (4 if cell.endswith(":all") else 12) for cell in CELL_NAMES
        }
        actual_counts = dict(Counter(entry["cell"] for entry in entries))
        if actual_counts != expected_counts:
            raise ValueError("dataset cell membership is not balanced")
        generation_fields = {
            "seed",
            "rng",
            "numpy",
            "alias_rule",
            "exact_dedup",
            "duplicate_family_retries",
            "counts",
            "source_family_size",
            "holdout",
        }
        if (
            not isinstance(generation, Mapping)
            or set(generation) != generation_fields
            or type(generation["seed"]) is not int
            or generation["seed"] != GENERATION_SEED
            or generation["rng"] != "numpy-PCG64"
            or not isinstance(generation["numpy"], str)
            or generation["alias_rule"] != "decay-100-at-zero-sustain-v1"
            or generation["exact_dedup"]
            != "global-structural-and-float32-audio-sha256-whole-family-retry"
            or type(generation["duplicate_family_retries"]) is not int
            or generation["duplicate_family_retries"] < 0
            or _digest(generation["counts"]) != _digest(actual_counts)
            or type(generation["source_family_size"]) is not int
            or generation["source_family_size"] != 2
            or generation["holdout"]
            != "square-and-attack-at-least-160ms; diminished-or-augmented-chord-orbits"
        ):
            raise ValueError("unsupported generation policy")
        frozen = cls(document["split"], _freeze(entries), _freeze(generation), _digest(document))
        return frozen


def load_manifest(split: str, *, directory: Path | None = None) -> DatasetManifest:
    """Load one checked-in manifest without synthesizing audio or loading Torch."""
    if not isinstance(split, str) or split not in SPLIT_COUNTS:
        raise ValueError("unknown dataset split")
    directory = DATA_DIRECTORY if directory is None else Path(directory)
    encoded = (directory / f"{split}.json").read_bytes()
    manifest = DatasetManifest.from_dict(decode_json_bytes(encoded))
    if manifest.split != split:
        raise ValueError("manifest split differs from requested split")
    catalog = decode_json_bytes((directory / "catalog.json").read_bytes())
    catalog_fields = {
        "schema_id",
        "generator_id",
        "manifests",
        "counts",
        "selection_criterion",
        "baseline_seeds",
        "evaluation_budget",
        "decision_budget",
        "final_test_use",
    }
    if (
        not isinstance(catalog, Mapping)
        or set(catalog) != catalog_fields
        or catalog["schema_id"] != "harpy.sound-match.dataset-catalog/1"
        or not isinstance(catalog["manifests"], Mapping)
        or not isinstance(catalog["counts"], Mapping)
        or set(catalog["manifests"]) != set(catalog["counts"])
        or any(
            name not in SPLIT_COUNTS or type(count) is not int or count < 24 or count % 24
            for name, count in catalog["counts"].items()
        )
        or catalog["counts"].get(split) != len(manifest.entries)
        or _digest(catalog["selection_criterion"]) != _digest(SELECTION_CRITERION)
        or _digest({"values": catalog["baseline_seeds"]}) != _digest({"values": [0, 1, 2]})
        or type(catalog["evaluation_budget"]) is not int
        or catalog["evaluation_budget"] != 32
        or type(catalog["decision_budget"]) is not int
        or catalog["decision_budget"] != 64
        or catalog["final_test_use"] != "only after M0-M4 completion and actor freeze"
    ):
        raise ValueError("invalid frozen dataset catalog")
    if (
        catalog.get("generator_id") != GENERATOR_ID
        or catalog.get("manifests", {}).get(split) != hashlib.sha256(encoded).hexdigest()
    ):
        raise ValueError("manifest bytes differ from frozen catalog")
    return manifest


def validate_disjoint(manifests: tuple[DatasetManifest, ...]) -> None:
    """Reject group, exact canonical-patch/structural, and rendered-audio leakage."""
    if len({manifest.split for manifest in manifests}) != len(manifests):
        raise ValueError("duplicate split")
    owners: dict[tuple[str, str], str] = {}
    for manifest in manifests:
        for entry in manifest.entries:
            identities = {
                "group": entry["source_family_id"],
                "structure": entry["structural_sha256"],
                "audio": entry["target_audio_sha256"],
                "episode": entry["episode"]["episode_id"],
            }
            for kind, value in identities.items():
                previous = owners.setdefault((kind, value), manifest.split)
                if previous != manifest.split:
                    raise ValueError(f"cross-split {kind} collision")


def manifest_summary(manifest: DatasetManifest) -> dict:
    """Describe distributions and public-start exposures without evaluating actors."""
    episodes = manifest.episodes
    notes = [note for episode in episodes for note in episode.target_patch.notes_cents]
    heldout_chords = {chord_class(value) for value in HELDOUT_CHORDS.values()}
    return {
        "split": manifest.split,
        "manifest_sha256": manifest.manifest_sha256,
        "targets": len(episodes),
        "source_families": len({entry["source_family_id"] for entry in manifest.entries}),
        "cells": dict(Counter(entry["cell"] for entry in manifest.entries)),
        "cohorts": dict(Counter(entry["cohort"] for entry in manifest.entries)),
        "target_pitch_range_cents": [min(notes), max(notes)],
        "target_waveforms": dict(Counter(episode.target_patch.waveform for episode in episodes)),
        "target_chord_families": dict(
            Counter(
                entry["chord_family"]
                for entry in manifest.entries
                if entry["chord_family"] is not None
            )
        ),
        "public_starts_with_heldout_combination": sum(
            heldout_combination(episode.initial_patch) for episode in episodes
        ),
        "public_starts_with_heldout_chord_family": sum(
            episode.voice_count == 3
            and chord_class(episode.initial_patch.notes_cents) in heldout_chords
            for episode in episodes
        ),
        "duplicate_family_retries": manifest.generation["duplicate_family_retries"],
    }


def iter_training_examples(
    *, directory: Path | None = None
) -> Iterator[tuple[EpisodeSpec, np.ndarray]]:
    """Yield training labels/audio one at a time; this API has no test-split switch."""
    manifest = load_manifest("train", directory=directory)
    for entry, episode in zip(manifest.entries, manifest.episodes, strict=True):
        audio = render(episode.target_patch)
        if audio_hash(audio) != entry["target_audio_sha256"]:
            raise ValueError("training audio hash differs from frozen renderer environment")
        yield episode, audio


def _rng(split: str, voice_count: int, group: int, attempt: int) -> np.random.Generator:
    seed = int(
        _digest(
            {
                "seed": GENERATION_SEED,
                "split": split,
                "voices": voice_count,
                "group": group,
                "attempt": attempt,
            }
        )[:32],
        16,
    )
    return np.random.default_rng(seed)


def _notes(
    rng: np.random.Generator, voice_count: int, heldout: bool
) -> tuple[tuple[int, ...], str | None]:
    if voice_count == 1:
        return (int(rng.integers(4800, 7201)),), None
    families = HELDOUT_CHORDS if heldout else TRAIN_CHORDS
    family = tuple(families)[int(rng.integers(len(families)))]
    intervals = list(families[family])
    inversion = int(rng.integers(3))
    intervals = sorted(
        interval + (1200 if index < inversion else 0) for index, interval in enumerate(intervals)
    )
    intervals = [interval - intervals[0] for interval in intervals]
    root = int(rng.integers(4800, 7201 - max(intervals)))
    return tuple(root + interval for interval in intervals), family


def _sample_patch(
    rng: np.random.Generator, voice_count: int, heldout: bool
) -> tuple[Patch, str | None]:
    notes, family = _notes(rng, voice_count, heldout)
    while True:
        patch = Patch(
            notes_cents=notes,
            waveform=("sine", "triangle", "saw", "square")[int(rng.integers(4))],
            brightness_hz=int(np.rint(np.exp(rng.uniform(np.log(200), np.log(12000))))),
            attack_ms=int(rng.integers(5, 251)),
            decay_ms=int(rng.integers(20, 501)),
            sustain_db=int(rng.integers(-24, 1)),
            release_ms=int(rng.integers(20, 751)),
            gain_db=int(rng.integers(-18, 1)),
        )
        if heldout_combination(patch) == heldout:
            return structural_patch(patch), family


def _family(split: str, voice_count: int, group: int, attempt: int) -> list[dict]:
    rng = _rng(split, voice_count, group, attempt)
    heldout = split == "compositional-test"
    base, family = _sample_patch(rng, voice_count, heldout)
    mask = tuple(MASKS)[group % len(MASKS)]
    while True:
        proposal, proposed_family = _sample_patch(rng, voice_count, heldout)
        changed = {name: getattr(proposal, name) for name in MASKS[mask]}
        target = replace(base, **changed)
        if target != base and heldout_combination(target) == heldout:
            break
    family_id = "g_" + _digest({"split": split, "voices": voice_count, "group": group})[:24]
    rows = []
    for kind, patch, initial, fields, chord in (
        ("all", base, default_patch(voice_count), PATCH_FIELDS, family),
        (mask, target, base, MASKS[mask], proposed_family if mask == "pitch" else family),
    ):
        episode_id = "e_" + _digest({"family": family_id, "kind": kind})[:24]
        episode = EpisodeSpec(episode_id, patch, initial, fields)
        rows.append(
            {
                "episode": episode.to_dict(),
                "source_family_id": family_id,
                "cohort": ("single" if voice_count == 1 else "chord")
                + ("-reconstruction" if kind == "all" else "-shaping"),
                "cell": f"{voice_count}:{kind}",
                "mask": kind,
                "chord_family": chord,
                "structural_sha256": structural_hash(patch),
                "target_audio_sha256": audio_hash(render(patch)),
            }
        )
    return rows


def build_manifests(
    directory: Path, *, counts: Mapping[str, int] | None = None
) -> tuple[DatasetManifest, ...]:
    """Deterministically generate hash-bound memberships, streaming each waveform.

    Whole families are retried on *any* exact alias, including within one split.
    Counts overrides are for generator tests; the packaged catalog freezes counts.
    """
    counts = SPLIT_COUNTS if counts is None else dict(counts)
    if not counts or any(
        split not in SPLIT_COUNTS or type(count) is not int or count < 24 or count % 24
        for split, count in counts.items()
    ):
        raise ValueError("split counts must be positive multiples of 24")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    seen_structures, seen_audio = set(), set()
    manifests = []
    hashes = {}
    for split in SPLIT_COUNTS:
        if split not in counts:
            continue
        entries = []
        retries = 0
        for voice_count in (1, 3):
            for group in range(counts[split] // 4):
                for attempt in range(1000):
                    rows = _family(split, voice_count, group, attempt)
                    structures = {row["structural_sha256"] for row in rows}
                    audios = {row["target_audio_sha256"] for row in rows}
                    if (
                        len(structures) == len(audios) == 2
                        and not structures & seen_structures
                        and not audios & seen_audio
                    ):
                        break
                    retries += 1
                else:
                    raise RuntimeError("could not generate disjoint source family")
                entries.extend(rows)
                seen_structures.update(structures)
                seen_audio.update(audios)
        generation = {
            "seed": GENERATION_SEED,
            "rng": "numpy-PCG64",
            "numpy": np.__version__,
            "alias_rule": "decay-100-at-zero-sustain-v1",
            "exact_dedup": "global-structural-and-float32-audio-sha256-whole-family-retry",
            "duplicate_family_retries": retries,
            "counts": dict(Counter(entry["cell"] for entry in entries)),
            "source_family_size": 2,
            "holdout": "square-and-attack-at-least-160ms; diminished-or-augmented-chord-orbits",
        }
        manifest = DatasetManifest.from_dict(
            {
                "schema_id": DATASET_SCHEMA_ID,
                "generator_id": GENERATOR_ID,
                "task_id": TASK_ID,
                "renderer_id": RENDERER_ID,
                "objective_id": OBJECTIVE_ID,
                "split": split,
                "generation": generation,
                "entries": entries,
            }
        )
        content = canonical_json_bytes(manifest.to_dict())
        write_new_bytes(directory / f"{split}.json", content)
        hashes[split] = hashlib.sha256(content).hexdigest()
        manifests.append(manifest)
    validate_disjoint(tuple(manifests))
    write_new_bytes(
        directory / "catalog.json",
        canonical_json_bytes(
            {
                "schema_id": "harpy.sound-match.dataset-catalog/1",
                "generator_id": GENERATOR_ID,
                "manifests": hashes,
                "counts": counts,
                "selection_criterion": SELECTION_CRITERION,
                "baseline_seeds": [0, 1, 2],
                "evaluation_budget": 32,
                "decision_budget": 64,
                "final_test_use": "only after M0-M4 completion and actor freeze",
            }
        ),
    )
    return tuple(manifests)
