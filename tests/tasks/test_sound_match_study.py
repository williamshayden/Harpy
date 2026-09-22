"""Research readout keeps equal cell weights and all-seed qualification gates."""

import importlib.util
from pathlib import Path

import pytest

from harpy.tasks.sound_match.datasets import CELL_NAMES, MASKS
from harpy.tasks.sound_match.patch import PATCH_FIELDS

_SPEC = importlib.util.spec_from_file_location(
    "sound_matching_study", Path(__file__).parents[2] / "research/sound_matching/study.py"
)
study = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(study)


def groups():
    result = []
    for name in study.ACTORS:
        for cell in CELL_NAMES:
            voices, mask = cell.split(":")
            count = 48 if mask == "all" else 16
            result.append(
                {
                    "actor": name,
                    "voice_count": int(voices),
                    "editable_fields": list(PATCH_FIELDS if mask == "all" else MASKS[mask]),
                    "episodes": count,
                    "submitted": count,
                    "error": 0,
                    "truncated": 0,
                    "submitted_distance": {"count": count, "mean": 8.0 if mask == "all" else 0.0},
                }
            )
    return result


def test_macro_weights_cells_equally_and_rejects_missing_submission():
    rows = groups()
    actors = study.actor_report(rows)
    assert len(actors) == 11
    assert all(row["episodes"] == 192 for row in actors)
    assert all(row["macro_submitted_distance"] == 2.0 for row in actors)
    rows[-1]["submitted"] -= 1
    rows[-1]["submitted_distance"]["count"] -= 1
    actors = study.actor_report(rows)
    assert actors[-1]["eligible"] is False
    assert actors[-1]["macro_submitted_distance"] is None


def test_all_three_seeds_must_strictly_beat_fixed_without_best_seed_selection():
    actors = study.actor_report(groups())
    for row in actors:
        if row["actor"] in ("inverse-0", "inverse-1", "inverse-2"):
            row["macro_submitted_distance"] = 1.0
    assert study.reference_quality(actors)["all_three_passed"] is True
    next(row for row in actors if row["actor"] == "inverse-2")["macro_submitted_distance"] = 2.0
    report = study.reference_quality(actors)
    assert [row["passed"] for row in report["seeds"]] == [True, True, False]
    assert report["all_three_passed"] is False


def test_duplicate_cell_is_not_silently_reweighted():
    rows = groups()
    with pytest.raises(ValueError, match="duplicate actor/cell"):
        study.actor_report([*rows, rows[0]])


def test_frozen_identity_comparison_accepts_arrays_and_scalars_without_coercion():
    for value in (["fixed", "inverse-0"], "abc", 0, {"seed": 0}):
        study._equal(value, value, "same")
    with pytest.raises(ValueError, match="seed differs"):
        study._equal(False, 0, "seed")


def test_real_durable_calibration_groups_feed_scientific_readout(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from harpy._serialization import decode_json_bytes
    from harpy.tasks.sound_match.actors import builtin_actor_spec
    from harpy.tasks.sound_match.datasets import load_manifest
    from harpy.tasks.sound_match.durable import evaluate_bundle

    manifest = load_manifest("calibration")
    chosen = {}
    chosen_entries = {}
    for entry, episode in zip(manifest.entries, manifest.episodes, strict=True):
        chosen.setdefault(entry["cell"], episode)
        chosen_entries.setdefault(entry["cell"], entry)
    bundle = tmp_path / "bundle"
    configuration = {
        "study_plan_sha256": "a" * 64,
        "split": "calibration",
        "dataset_manifest_sha256": "b" * 64,
    }
    result = evaluate_bundle(
        tuple(chosen.values()), (builtin_actor_spec("fixed"),), bundle, configuration=configuration
    )
    report = study.actor_report(result.metrics["groups"])
    assert report[0]["eligible"] is True
    assert report[0]["submitted"] == report[0]["episodes"] == 8
    assert report[0]["macro_submitted_distance"] > 0
    assert report[1]["eligible"] is False
    saved = decode_json_bytes((bundle / "manifest.json").read_bytes())
    plan = {
        "source": saved["identity"]["source"],
        "runtime": saved["identity"]["runtime"],
        "actors": ["fixed"],
        "actor_protocols": {"fixed": study.actor_protocols()["fixed"]},
        "cache_max_audio_bytes": saved["identity"]["cache"]["max_audio_bytes"],
        "splits": {"calibration": {"sha256": "b" * 64}},
    }
    monkeypatch.setattr(
        study,
        "load_manifest",
        lambda _: SimpleNamespace(
            episodes=tuple(chosen.values()), entries=tuple(chosen_entries.values())
        ),
    )
    strict = study.split_report(bundle, plan, "a" * 64, "calibration")
    assert strict["actors"] == report
    plan["runtime"] = {**plan["runtime"], "numpy": "changed"}
    with pytest.raises(ValueError, match="numerical runtime"):
        study.split_report(bundle, plan, "a" * 64, "calibration")
    corrupted_entries = [dict(entry) for entry in chosen_entries.values()]
    corrupted_entries[0]["target_audio_sha256"] = "0" * 64
    corrupted = SimpleNamespace(episodes=tuple(chosen.values()), entries=corrupted_entries)
    with pytest.raises(ValueError, match="rendered target audio"):
        study.verify_rendered_targets(corrupted)
    with pytest.raises(ValueError, match="recorded target audio"):
        study.verify_recorded_targets(bundle, corrupted)
