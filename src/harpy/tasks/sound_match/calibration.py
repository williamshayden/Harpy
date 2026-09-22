"""CPU calibration fixtures and measured pilot reporting, without success labels."""

from __future__ import annotations

import hashlib
import itertools
import platform
import resource
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np

from harpy._serialization import canonical_json_bytes, write_new_bytes
from harpy.tasks.sound_match.actors import builtin_actor_spec
from harpy.tasks.sound_match.datasets import DatasetManifest
from harpy.tasks.sound_match.objective import OBJECTIVE_ID, score
from harpy.tasks.sound_match.patch import Patch
from harpy.tasks.sound_match.renderer import FRAME_COUNT, RENDERER_ID, render
from harpy.tasks.sound_match.results import _wav_bytes, audio_hash, save_result
from harpy.tasks.sound_match.runner import evaluate

CALIBRATION_ID = "harpy-sound-calibration-v1"
NEAR_EQUIVALENCE_THRESHOLD = 0.03


def calibration_pairs() -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Cover identities, aliases and deliberately out-of-space audio perturbations."""
    patch = Patch(waveform="saw", brightness_hz=5000, sustain_db=-9, gain_db=-4)
    target = render(patch)
    yield "identity", target, render(patch)
    for name, changed in (
        ("detune-5-cents", replace(patch, notes_cents=(6005,))),
        ("detune-100-cents", replace(patch, notes_cents=(6100,))),
        ("brightness-500hz", replace(patch, brightness_hz=500)),
        ("attack-250ms", replace(patch, attack_ms=250)),
        ("release-750ms", replace(patch, release_ms=750)),
        ("level-minus-12db", replace(patch, gain_db=-16)),
    ):
        yield name, target, render(changed)
    yield "silence", target, np.zeros(FRAME_COUNT, dtype=np.float32)
    yield "clipped-at-0.025", target, np.clip(target, -0.025, 0.025)
    yield "phase-polarity-inversion", target, -target
    yield "phase-shift-17-samples", target, np.roll(target, 17)
    chord = replace(patch, notes_cents=(6000, 6400, 6700))
    chord_audio = render(chord)
    yield "chord-permutation", chord_audio, render(replace(chord, notes_cents=(6700, 6000, 6400)))
    missing = (
        render(replace(patch, notes_cents=(6000,))).astype(np.float64)
        + render(replace(patch, notes_cents=(6700,))).astype(np.float64)
    ) / 3
    yield "missing-chord-tone", chord_audio, missing.astype(np.float32)
    alias = replace(patch, sustain_db=0, decay_ms=20)
    yield "inactive-decay-alias", render(alias), render(replace(alias, decay_ms=500))


def write_calibration(output: Path) -> dict:
    """Create finite score evidence plus lossless sample pairs for human listening."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows = []
    for name, target, candidate in calibration_pairs():
        row = {
            "name": name,
            "target_audio_sha256": audio_hash(target),
            "candidate_audio_sha256": audio_hash(candidate),
            "objective": score(target, candidate),
        }
        for role, audio in (("target", target), ("candidate", candidate)):
            content = _wav_bytes(audio)
            filename = f"{name}-{role}.wav"
            write_new_bytes(output / filename, content)
            row[f"{role}_file"] = {
                "path": filename,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        rows.append(row)
    document = {
        "schema_id": CALIBRATION_ID,
        "renderer_id": RENDERER_ID,
        "objective_id": OBJECTIVE_ID,
        "perceptual_cutoff": None,
        "listening_review": "not-performed",
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_new_bytes(output / "calibration.json", canonical_json_bytes(document))
    return document


def _fingerprint(audio: np.ndarray) -> np.ndarray:
    # A screening diagnostic, deliberately distinct from the ranking objective.
    magnitude = np.abs(np.fft.rfft(audio.astype(np.float64))) / len(audio)
    feature = np.array([np.mean(chunk) for chunk in np.array_split(magnitude[:24001], 256)])
    feature = np.log1p(feature / 1e-7)
    norm = np.linalg.norm(feature)
    return feature / max(norm, 1e-12)


def near_equivalence_report(manifest: DatasetManifest) -> dict:
    """Inspect calibration-only near aliases; do not cluster/reassign held-out sets."""
    if manifest.split != "calibration":
        raise ValueError("near-equivalence tuning is restricted to calibration membership")
    features = [_fingerprint(render(episode.target_patch)) for episode in manifest.episodes]
    pairs = []
    minimum = None
    for left, right in itertools.combinations(range(len(features)), 2):
        distance = float(np.linalg.norm(features[left] - features[right]))
        minimum = distance if minimum is None else min(minimum, distance)
        if distance <= NEAR_EQUIVALENCE_THRESHOLD:
            pairs.append(
                {
                    "left": manifest.entries[left]["episode"]["episode_id"],
                    "right": manifest.entries[right]["episode"]["episode_id"],
                    "same_source_family": manifest.entries[left]["source_family_id"]
                    == manifest.entries[right]["source_family_id"],
                    "distance": distance,
                }
            )
    return {
        "feature": "unit-L2-log1p-256-pooled-fullclip-rfft-bins-0-through-24000-v1",
        "threshold": NEAR_EQUIVALENCE_THRESHOLD,
        "threshold_purpose": "investigation-only; not a success or contamination equivalence rule",
        "comparisons": len(features) * (len(features) - 1) // 2,
        "minimum_distance": minimum,
        "flagged_count": len(pairs),
        "pairs": pairs,
        "scope": "calibration-only; approximate cross-split independence is not claimed",
    }


def _distribution(values: list[float]) -> dict:
    return (
        {
            "n": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
        }
        if values
        else {"n": 0, "mean": None, "median": None, "p90": None}
    )


def run_pilot(manifest: DatasetManifest, output: Path) -> dict:
    """Run exactly the 96-target, seed-zero CPU pilot with a bounded waveform cache."""
    if manifest.split != "calibration" or len(manifest.entries) != 96:
        raise ValueError("pilot requires the 96-target calibration manifest")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    metadata = {entry["episode"]["episode_id"]: entry for entry in manifest.entries}
    result = evaluate(
        manifest.episodes,
        tuple(
            builtin_actor_spec(name, seed=0)
            for name in ("fixed", "random-search", "pattern-search")
        ),
        cache_max_audio_bytes=16 * 1024 * 1024,
    )
    evaluation_seconds = time.perf_counter() - started
    save_result(result, output / "bundle")
    groups = defaultdict(list)
    for record in result.records:
        groups[(metadata[record.episode_id]["cohort"], record.actor_name)].append(record)
    report_groups = []
    for (cohort, actor), records in sorted(groups.items()):
        submitted = [
            record.submitted_candidate.distance
            for record in records
            if record.submitted_candidate is not None
        ]
        curves = {
            str(limit): _distribution(
                [
                    min(
                        candidate.distance
                        for candidate in record.candidates
                        if candidate.evaluation <= limit
                    )
                    for record in records
                    if record.candidates
                ]
            )
            for limit in (1, 4, 16, 32)
        }
        report_groups.append(
            {
                "cohort": cohort,
                "actor": actor,
                "episodes": len(records),
                "submitted": sum(record.status == "submitted" for record in records),
                "errors": sum(record.status == "error" for record in records),
                "truncated": sum(record.status == "truncated" for record in records),
                "distance": _distribution(submitted),
                "best_audition_curve": curves,
                "episode_seconds": _distribution([record.elapsed_seconds for record in records]),
                "dsp_renders": sum(record.dsp_renders for record in records),
                "cache_hits": sum(record.cache_hits for record in records),
                "evaluations": sum(record.metrics["evaluations"] for record in records),
            }
        )
    comparison = []
    for cohort in sorted({entry["cohort"] for entry in manifest.entries}):
        actors = {
            actor: {
                record.episode_id: record.submitted_candidate.distance
                for record in records
                if record.submitted_candidate is not None
            }
            for (group, actor), records in groups.items()
            if group == cohort
        }
        for actor in ("random-search", "pattern-search"):
            shared = sorted(actors[actor].keys() & actors["fixed"].keys())
            comparison.append(
                {
                    "cohort": cohort,
                    "actor_minus_fixed": actor,
                    "paired_distance_difference": _distribution(
                        [actors[actor][key] - actors["fixed"][key] for key in shared]
                    ),
                }
            )
    near = near_equivalence_report(manifest)
    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    document = {
        "schema_id": "harpy.sound-match.pilot/1",
        "manifest_sha256": manifest.manifest_sha256,
        "targets": 96,
        "seed": 0,
        "source": _plain_source(result.source),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": "cpu",
        },
        "groups": report_groups,
        "paired_comparisons": comparison,
        "near_equivalence": near,
        "evaluation_seconds": evaluation_seconds,
        "total_seconds_before_report": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "bundle_bytes": size,
        "cache_max_audio_bytes": 16 * 1024 * 1024,
        "target_render_seconds": _distribution(
            [float(target["elapsed_seconds"]) for target in result.targets]
        ),
        "listening_review": "not-performed",
        "perceptual_success_cutoff": None,
    }
    write_new_bytes(output / "pilot.json", canonical_json_bytes(document))
    return document


def _plain_source(source):
    from harpy.tasks.sound_match.models import _plain

    return _plain(source)
