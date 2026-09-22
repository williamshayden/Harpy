"""Exact comparison with preserved v1 code plus bounded, owned cache behavior."""

import gc
import hashlib
import importlib.util
import json
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from harpy._serialization import canonical_json_bytes
from harpy.tasks.sound_match import actors, objective, runner
from harpy.tasks.sound_match.calibration import calibration_pairs
from harpy.tasks.sound_match.datasets import load_manifest
from harpy.tasks.sound_match.models import EpisodeSpec
from harpy.tasks.sound_match.patch import Patch
from harpy.tasks.sound_match.renderer import FRAME_COUNT, render
from harpy.tasks.sound_match.results import audio_hash

FIXTURES = Path(__file__).with_name("fixtures")
REFERENCE = FIXTURES / "sound_match_objective_v1_reference.py"
REFERENCE_SHA256 = "ca8a448dbc8f583bef686fa77c99cefee079ade92e0b718a02a9fda03f3877f7"


@pytest.fixture(scope="module")
def reference():
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == REFERENCE_SHA256
    spec = importlib.util.spec_from_file_location("pre_cache_objective", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def empty_cache():
    objective._clear_score_cache()
    yield
    objective._clear_score_cache()


def exact_score(actual, expected):
    # Canonical JSON preserves the exact round-trippable binary64 values and -0.0.
    assert canonical_json_bytes(actual) == canonical_json_bytes(expected)


def test_every_calibration_fixture_is_bitwise_equal_to_preserved_original(reference):
    assert objective.OBJECTIVE_ID == reference.OBJECTIVE_ID == "harpy-multires-spectral-v1"
    count = 0
    for _name, target, candidate in calibration_pairs():
        expected = reference.score(target, candidate)
        exact_score(objective.score(target, candidate), expected)
        exact_score(objective.score(target.copy(), candidate.copy()), expected)
        count += 1
    assert count == 14
    assert objective._score_cache_info()["score_hits"] >= 14


def test_recorded_pilot_pairs_remain_bitwise_equal_to_original_and_recorded_scores(reference):
    document = json.loads((FIXTURES / "sound_match_objective_pilot_pairs.json").read_text())
    assert document["calibration_manifest_sha256"] == load_manifest("calibration").manifest_sha256
    assert len(document["pairs"]) == 32
    assert len({row["cell"] for row in document["pairs"]}) == 8
    assert {row["candidate_patch"]["waveform"] for row in document["pairs"]} == {
        "sine",
        "saw",
        "square",
        "triangle",
    }
    for row in document["pairs"]:
        target = render(Patch.from_dict(row["target_patch"]))
        candidate = render(Patch.from_dict(row["candidate_patch"]))
        assert audio_hash(target) == row["target_audio_sha256"]
        assert audio_hash(candidate) == row["candidate_audio_sha256"]
        expected = reference.score(target, candidate)
        exact_score(expected, row["objective"])
        exact_score(objective.score(target, candidate), expected)
        exact_score(objective.score(target, candidate), expected)


def test_content_changes_and_returned_nested_mutations_cannot_poison_cache(reference):
    target = render(Patch())
    candidate = target.copy()
    original = objective.score(target, candidate)
    assert original["distance"] == 0
    candidate[4096] += np.float32(0.1)
    expected = reference.score(target, candidate)
    assert expected["distance"] > 0
    actual = objective.score(target, candidate)
    exact_score(actual, expected)
    actual["distance"] = -123
    actual["spectral"][0]["convergence"] = -123
    actual["spectral"].append({"unexpected": True})
    exact_score(objective.score(target, candidate), expected)
    target[6000] -= np.float32(0.2)
    exact_score(objective.score(target, candidate), reference.score(target, candidate))
    # Contiguous copies and noncontiguous arrays use logical contents, not identities.
    exact_score(
        objective.score(target[::-1], candidate[::-1]),
        reference.score(target[::-1], candidate[::-1]),
    )


def test_cache_does_not_retain_input_arrays_and_all_cached_analysis_arrays_are_owned_readonly():
    target = np.zeros(FRAME_COUNT, np.float32)
    candidate = np.full(FRAME_COUNT, 0.25, np.float32)
    references = weakref.ref(target), weakref.ref(candidate)
    objective.score(target, candidate)
    for analysis in objective._ANALYSES.values():
        for array in (*analysis.magnitudes, *analysis.log_magnitudes, analysis.envelope):
            assert array.flags.owndata
            assert not array.flags.writeable
            assert not np.shares_memory(array, target)
            assert not np.shares_memory(array, candidate)
    del target, candidate
    gc.collect()
    assert all(reference() is None for reference in references)


def test_primed_cache_still_validates_every_input():
    valid = np.zeros(FRAME_COUNT, np.float32)
    objective.score(valid, valid)
    nonfinite = valid.copy()
    nonfinite[12] = np.nan
    for invalid in (valid.astype(np.float64), valid.tolist(), valid[:-1], nonfinite):
        with pytest.raises(ValueError, match="target"):
            objective.score(invalid, valid)
        with pytest.raises(ValueError, match="candidate"):
            objective.score(valid, invalid)
    assert objective._score_cache_info()["score_hits"] == 0


def test_analysis_and_score_eviction_stay_within_explicit_caps(reference):
    target = np.zeros(FRAME_COUNT, np.float32)
    first = np.full(FRAME_COUNT, 1 / 1024, np.float32)
    expected = reference.score(target, first)
    for index in range(objective._MAX_SCORE_ENTRIES + 2):
        candidate = np.full(FRAME_COUNT, (index + 1) / 1024, np.float32)
        objective.score(target, candidate)
        info = objective._score_cache_info()
        assert info["analysis_entries"] <= 2
        assert info["analysis_array_bytes"] <= info["max_analysis_array_bytes"] == 20 * 1024**2
        assert info["score_entries"] <= info["max_score_entries"] == 128
        assert info["analysis_array_bytes"] == sum(
            item.array_bytes for item in objective._ANALYSES.values()
        )
    before = objective._score_cache_info()
    exact_score(objective.score(target, first), expected)
    assert objective._score_cache_info()["score_misses"] == before["score_misses"] + 1
    objective._clear_score_cache()
    info = objective._score_cache_info()
    assert info["analysis_entries"] == info["analysis_array_bytes"] == info["score_entries"] == 0


def test_analysis_byte_cap_can_refuse_storage_without_changing_scores(monkeypatch, reference):
    monkeypatch.setattr(objective, "_MAX_ANALYSIS_BYTES", 1)
    target = np.zeros(FRAME_COUNT, np.float32)
    candidate = np.full(FRAME_COUNT, 0.25, np.float32)
    exact_score(objective.score(target, candidate), reference.score(target, candidate))
    assert objective._score_cache_info()["analysis_entries"] == 0
    assert objective._score_cache_info()["analysis_array_bytes"] == 0


def test_concurrent_score_and_clear_calls_are_exact_owned_and_bounded(reference):
    target = np.linspace(-0.5, 0.5, FRAME_COUNT, dtype=np.float32)
    candidates = (target * np.float32(0.5), -target, np.roll(target, 17))
    expected = [canonical_json_bytes(reference.score(target, item)) for item in candidates]

    def worker(index):
        result = objective.score(target, candidates[index % 3])
        assert canonical_json_bytes(result) == expected[index % 3]
        result["spectral"][0]["log_distance"] = -999
        return canonical_json_bytes(objective.score(target, candidates[index % 3]))

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, index) for index in range(24)]
        clearing = executor.submit(objective._clear_score_cache)
        for index, future in enumerate(futures):
            assert future.result() == expected[index % 3]
        clearing.result()
    info = objective._score_cache_info()
    assert info["analysis_entries"] <= 2
    assert info["analysis_array_bytes"] <= 20 * 1024**2
    assert info["score_entries"] <= 128


def test_cache_leaves_complete_actor_traces_and_all_logical_dsp_counters_unchanged(
    monkeypatch, reference
):
    manifest = load_manifest("calibration")
    entry = next(entry for entry in manifest.entries if entry["cell"] == "1:all")
    episode = EpisodeSpec.from_dict(entry["episode"])
    specs = tuple(
        actors.builtin_actor_spec(name, seed=0) for name in ("pattern-search", "random-search")
    )
    with monkeypatch.context() as original:
        original.setattr(runner, "score", reference.score)
        original.setattr(actors, "score", reference.score)
        before = runner.evaluate((episode,), specs, cache_max_audio_bytes=16 * 1024**2)
    objective._clear_score_cache()
    after = runner.evaluate((episode,), specs, cache_max_audio_bytes=16 * 1024**2)
    for old, new in zip(before.records, after.records, strict=True):
        old_row, new_row = old.to_dict(), new.to_dict()
        old_row.pop("elapsed_seconds")
        new_row.pop("elapsed_seconds")
        assert canonical_json_bytes(old_row) == canonical_json_bytes(new_row)
        assert new.metrics["evaluations"] == 32
        assert new.metrics["model_calls"] == 0
    assert objective._score_cache_info()["score_hits"] > 0
