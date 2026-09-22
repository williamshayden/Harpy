"""Derived differences pair exact membership and preserve terminal denominators."""

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from harpy._serialization import canonical_json_bytes, decode_json_bytes
from harpy.tasks.sound_match.actors import builtin_actor_spec
from harpy.tasks.sound_match.datasets import CELL_NAMES, SELECTION_CRITERION, load_manifest
from harpy.tasks.sound_match.durable import evaluate_bundle
from harpy.tasks.sound_match.models import ActorSpec, Propose, _plain

SCRIPT = Path(__file__).parents[2] / "research/sound_matching/paired_differences.py"
_SPEC = importlib.util.spec_from_file_location("sound_matching_paired", SCRIPT)
paired = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(paired)


def row(distance, *, status="submitted", evaluations=2):
    return {"status": status, "distance": distance, "evaluations": evaluations}


def test_comparisons_cover_every_seed_and_declared_baseline_without_best_seed_choice():
    pairs = paired.comparisons()
    assert len(pairs) == len(set(pairs)) == 25
    assert {(actor, "fixed") for actor in paired.ACTORS if actor != "fixed"} <= set(pairs)
    for seed in range(3):
        assert (f"inverse-refine-{seed}", f"inverse-{seed}") in pairs
        for kind in ("inverse", "inverse-refine"):
            assert (f"{kind}-{seed}", "pattern-search") in pairs
            assert (f"{kind}-{seed}", f"random-search-{seed}") in pairs


def test_differences_pair_exact_ids_and_report_sign_distribution_and_actual_costs():
    records = {
        ("c", "right"): row(6, evaluations=32),
        ("a", "left"): row(8),
        ("b", "right"): row(8, evaluations=32),
        ("c", "left"): row(2),
        ("a", "right"): row(3, evaluations=32),
        ("b", "left"): row(20),
    }
    report = paired.paired_group(("b", "c", "a"), records, "left", "right")
    # The exact paired values are 12, -4, 5, independent of dictionary order.
    assert report["submitted_distance_difference"] == {
        "count": 3,
        "mean": pytest.approx(13 / 3),
        "median": 5.0,
        "p90": pytest.approx(10.6),
    }
    assert report["expected_episodes"] == report["both_submitted"] == 3
    assert report["actor_evaluations_on_submitted_pairs"]["mean"] == 2
    assert report["comparator_evaluations_on_submitted_pairs"]["mean"] == 32
    reverse = paired.paired_group(("a", "b", "c"), records, "right", "left")
    assert reverse["submitted_distance_difference"]["mean"] == pytest.approx(-13 / 3)


def test_failures_stay_in_denominators_without_imputed_or_last_candidate_scores():
    records = {
        ("a", "left"): row(1),
        ("a", "right"): row(2),
        ("b", "left"): row(-100, status="error"),
        ("b", "right"): row(2),
        ("c", "left"): row(1),
        ("c", "right"): row(-100, status="truncated"),
        ("d", "left"): row(None, status="error", evaluations=None),
        ("d", "right"): row(None, status="error", evaluations=None),
    }
    report = paired.paired_group(("a", "b", "c", "d"), records, "left", "right")
    assert report["expected_episodes"] == 4
    assert report["both_submitted"] == 1
    assert report["excluded_from_distance_difference"] == 3
    assert report["submitted_distance_difference"] == {
        "count": 1,
        "mean": -1.0,
        "median": -1.0,
        "p90": -1.0,
    }
    assert report["status_pairs"]["error|submitted"] == 1
    assert report["status_pairs"]["submitted|truncated"] == 1
    assert report["status_pairs"]["error|error"] == 1
    empty = paired.paired_group(("b", "c", "d"), records, "left", "right")
    assert empty["submitted_distance_difference"] == {
        "count": 0,
        "mean": None,
        "median": None,
        "p90": None,
    }


def test_missing_or_duplicate_expected_membership_fails_closed():
    records = {("a", "left"): row(1)}
    with pytest.raises(ValueError, match="missing an expected"):
        paired.paired_group(("a",), records, "left", "right")
    with pytest.raises(ValueError, match="unique"):
        paired.paired_group(("a", "a"), records, "left", "right")


def test_macro_weights_eight_cells_and_becomes_ineligible_on_any_unsubmitted_pair():
    entries, records = [], {}
    for cell in CELL_NAMES:
        voices, mask = cell.split(":")
        cohort = ("single" if voices == "1" else "chord") + (
            "-reconstruction" if mask == "all" else "-shaping"
        )
        for index in range(3 if mask == "all" else 1):
            episode_id = f"{cell}-{index}"
            entries.append({"cell": cell, "cohort": cohort, "episode": {"episode_id": episode_id}})
            records[(episode_id, "left")] = row(8 if mask == "all" else 0)
            records[(episode_id, "right")] = row(0)
    report = paired.paired_comparisons(tuple(entries), records, (("left", "right"),))[0]
    assert report["equal_eight_cell_mean_difference"] == 2.0  # Pooled-row mean would be 4.
    assert report["cohorts"]["single-shaping"]["expected_episodes"] == 3
    assert report["cohorts"]["single-reconstruction"]["expected_episodes"] == 3
    records[(entries[-1]["episode"]["episode_id"], "left")] = row(None, status="error")
    report = paired.paired_comparisons(tuple(entries), records, (("left", "right"),))[0]
    assert report["all_expected_pairs_submitted"] is False
    assert report["equal_eight_cell_mean_difference"] is None


@pytest.fixture(scope="module")
def real_bundle(tmp_path_factory):
    # Small generic-runner fixture uses calibration labels and bounded test budgets.
    # The top-level study reader still requires all full frozen memberships.
    source = load_manifest("calibration")
    selected = {}
    for entry, episode in zip(source.entries, source.episodes, strict=True):
        selected.setdefault(
            entry["cell"], (entry, replace(episode, render_budget=3, decision_budget=4))
        )
    entries = tuple(entry for entry, episode in selected.values())
    episodes = tuple(episode for entry, episode in selected.values())
    actors = (
        builtin_actor_spec("fixed"),
        replace(builtin_actor_spec("random-search"), name="random-search-0"),
    )
    bundle = tmp_path_factory.mktemp("paired") / "bundle"
    configuration = {
        "study_plan_sha256": "a" * 64,
        "split": "calibration",
        "dataset_manifest_sha256": "b" * 64,
    }
    evaluate_bundle(episodes, actors, bundle, configuration=configuration)
    saved = decode_json_bytes((bundle / "manifest.json").read_bytes())
    plan = {
        "source": saved["identity"]["source"],
        "runtime": saved["identity"]["runtime"],
        "actors": [actor.name for actor in actors],
        "actor_protocols": {actor.name: _plain(actor.metadata) for actor in actors},
        "cache_max_audio_bytes": saved["identity"]["cache"]["max_audio_bytes"],
        "splits": {"calibration": {"sha256": "b" * 64, "targets": len(episodes)}},
    }
    membership = SimpleNamespace(episodes=episodes, entries=entries, manifest_sha256="b" * 64)
    return bundle, plan, membership


def test_strict_real_bundle_is_read_once_and_binds_records_hashes_and_denominators(
    real_bundle, monkeypatch
):
    bundle, plan, membership = real_bundle
    calls = []
    original = paired.iter_records

    def tracked(path):
        calls.append(path)
        yield from original(path)

    monkeypatch.setattr(paired, "iter_records", tracked)
    binding, records, failures = paired.read_split(
        bundle, plan, "a" * 64, "calibration", membership
    )
    assert calls == [bundle]
    assert binding["retained_records"] == binding["expected_records"] == 16
    assert len(binding["bundle_manifest_sha256"]) == len(binding["bundle_completion_sha256"]) == 64
    assert failures == []
    result = paired.paired_comparisons(
        membership.entries, records, (("random-search-0", "fixed"),)
    )[0]
    assert result["all_expected_pairs_submitted"] is True
    assert all(
        row["expected_episodes"] == row["both_submitted"] == 1 for row in result["cells"].values()
    )


@pytest.mark.parametrize("changed", ("target", "runtime", "source", "configuration"))
def test_reader_rejects_wrong_target_or_rebound_plan_identity(real_bundle, changed):
    bundle, plan, membership = real_bundle
    modified_plan = json.loads(json.dumps(plan))
    if changed == "target":
        entries = [dict(entry) for entry in membership.entries]
        entries[0]["target_audio_sha256"] = "0" * 64
        membership = SimpleNamespace(
            episodes=membership.episodes, entries=entries, manifest_sha256="b" * 64
        )
    elif changed == "runtime":
        modified_plan["runtime"]["numpy"] = "changed"
    elif changed == "source":
        modified_plan["source"]["sha256"] = "0" * 64
    else:
        modified_plan["splits"]["calibration"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs"):
        paired.read_split(bundle, modified_plan, "a" * 64, "calibration", membership)


class FailingActor:
    def decide(self, observation):
        raise RuntimeError("paired report failure fixture")


class NonSubmittingActor:
    def decide(self, observation):
        return Propose(observation.patch, model_calls=0)


def test_real_errors_and_truncations_survive_derived_readout(tmp_path):
    source = load_manifest("calibration")
    entry = source.entries[0]
    episode = replace(source.episodes[0], decision_budget=1, render_budget=2)
    actors = (
        builtin_actor_spec("fixed"),
        ActorSpec("fails", FailingActor),
        ActorSpec("truncates", NonSubmittingActor),
    )
    bundle = tmp_path / "failures"
    evaluate_bundle(
        (episode,),
        actors,
        bundle,
        configuration={
            "study_plan_sha256": "a" * 64,
            "split": "calibration",
            "dataset_manifest_sha256": "b" * 64,
        },
    )
    saved = decode_json_bytes((bundle / "manifest.json").read_bytes())
    plan = {
        "source": saved["identity"]["source"],
        "runtime": saved["identity"]["runtime"],
        "actors": [actor.name for actor in actors],
        "actor_protocols": {actor.name: _plain(actor.metadata) for actor in actors},
        "cache_max_audio_bytes": saved["identity"]["cache"]["max_audio_bytes"],
        "splits": {"calibration": {"sha256": "b" * 64, "targets": 1}},
    }
    membership = SimpleNamespace(episodes=(episode,), entries=(entry,), manifest_sha256="b" * 64)
    _, records, failures = paired.read_split(bundle, plan, "a" * 64, "calibration", membership)
    assert [(row["actor"], row["status"]) for row in failures] == [
        ("fails", "error"),
        ("truncates", "truncated"),
    ]
    assert failures[0]["error_type"] == "RuntimeError"
    assert all(row["last_distance_diagnostic"] is not None for row in failures)
    for actor in ("fails", "truncates"):
        report = paired.paired_group((episode.episode_id,), records, actor, "fixed")
        assert report["expected_episodes"] == report["excluded_from_distance_difference"] == 1
        assert report["submitted_distance_difference"]["mean"] is None


def test_top_level_report_requires_all_three_completed_splits(tmp_path):
    plan = {
        "schema_id": paired.STUDY_SCHEMA_ID,
        "actors": list(paired.ACTORS),
        "selection_criterion": SELECTION_CRITERION,
        "budgets": {"candidate_evaluations": 32, "decisions": 64},
        "models": [{"seed": seed} for seed in range(3)],
    }
    (tmp_path / "plan.json").write_bytes(canonical_json_bytes(plan))
    with pytest.raises(ValueError, match="all three completed"):
        paired.build_report(tmp_path)


def test_postprocessor_imports_without_model_packages():
    code = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('paired_readout', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'torch' not in sys.modules and 'stable_baselines3' not in sys.modules
assert len(module.comparisons()) == 25
"""
    subprocess.run([sys.executable, "-c", code, str(SCRIPT)], check=True)
