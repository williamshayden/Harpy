"""Bounded orchestration checks; these are not the frozen research study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from harpy.envs.robustness import DEFAULT_CACHE_BYTES, ROBUSTNESS_CONDITIONS
from harpy.experiments import (
    EpisodeSpec,
    evaluate,
    load_result,
    oracle_actor_spec,
    save_result,
    summarize,
)
from harpy.experiments.models import _plain
from harpy.experiments.runner import _artifact_files, _source
from research.spectrum_preservation.study import (
    _actor_documents,
    _actors,
    _assert_unchanged,
    _worker,
    merge_shards,
    run_study,
)


def _protocol(episode: EpisodeSpec, identity: str = "driver-test") -> dict:
    document = {"id": identity, "episodes": [episode.to_document()]}
    content = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return {**document, "digest_sha256": hashlib.sha256(content).hexdigest()}


@pytest.fixture(scope="module")
def shards():
    episode = EpisodeSpec("driver-only", 6000, 12, 22)
    return tuple(
        evaluate(
            (episode,), (oracle_actor_spec(),), conditions=(condition,), protocol=_protocol(episode)
        )
        for condition in ROBUSTNESS_CONDITIONS
    )


def test_merge_reloads_ordinary_results_and_preserves_order_and_metrics(shards, tmp_path):
    result = merge_shards(shards, workers=4, elapsed_seconds=12.5)
    path = tmp_path / "experiment.json"
    save_result(result, path)
    loaded = load_result(path)
    assert len(loaded.records) == 8
    assert tuple(record.condition_id for record in loaded.records) == tuple(
        condition.id for condition in ROBUSTNESS_CONDITIONS
    )
    assert loaded.rows == result.rows
    assert len(loaded.paired_clean) == 7
    assert loaded.provenance["runtime"]["worker_limit"] == 4
    assert loaded.provenance["runtime"]["condition_runs"] == 8
    assert loaded.provenance["runtime"]["per_process_cache_budget_bytes"] == DEFAULT_CACHE_BYTES
    assert "paired_clean.mean_change_final_error_cents" in summarize(loaded, format="csv")
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_result(result, path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("change", ["actors", "source", "episodes", "protocol", "runtime"])
def test_merge_rejects_cross_shard_identity_changes(shards, change):
    part = shards[-1]
    if change == "actors":
        actors = _plain(part.actors)
        actors[0]["decoder"] = "a-different-decoder"
        part = replace(part, actors=tuple(actors))
    elif change == "episodes":
        part = replace(
            part, episodes=(replace(part.episodes[0], partition="different"),), protocol=None
        )
    elif change == "protocol":
        part = replace(part, protocol=_protocol(part.episodes[0], "other-study"))
    else:
        provenance = _plain(part.provenance)
        if change == "source":
            provenance["source"]["package_sha256"] = "0" * 64
        else:
            provenance["runtime"]["numpy"] = "different-version"
        part = replace(part, provenance=provenance)
    with pytest.raises(ValueError, match="shards must share"):
        merge_shards((*shards[:-1], part), workers=4, elapsed_seconds=1.0)


def test_merge_rejects_missing_or_reordered_conditions(shards):
    with pytest.raises(ValueError, match="eight conditions"):
        merge_shards(shards[:-1], workers=4, elapsed_seconds=1.0)
    with pytest.raises(ValueError, match="eight-condition order"):
        merge_shards(tuple(reversed(shards)), workers=4, elapsed_seconds=1.0)


def test_guard_checks_real_script_inventory_without_mocking_source(tmp_path):
    path = tmp_path / "study-input.txt"
    path.write_text("frozen", encoding="utf-8")
    source, files = _source(), _artifact_files((path,))
    _assert_unchanged(source, files, (path,))
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="scripts, manifest, or plan changed"):
        _assert_unchanged(source, files, (path,))


def test_condition_worker_uses_real_actors_runner_and_codec_on_tiny_fixture(tmp_path, monkeypatch):
    from research.spectrum_preservation import membership

    episode = EpisodeSpec("driver-worker-only", 6000, 12, 23)
    manifest = _protocol(episode)
    manifest_path = tmp_path / "tiny-fixture.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    # Only the test's cohort admission is reduced; source/artifact guards and the
    # three real waveform actors, environment, runner, and codecs are unchanged.
    monkeypatch.setattr(membership, "episodes_from_manifest", lambda _: (episode,))
    directory = Path(__file__).resolve().parent
    paths = tuple(directory / name for name in ("study.py", "encoders.py", "membership.py"))
    paths += (manifest_path,)
    source, files = _source(), _artifact_files(paths)
    actors = _actor_documents(_actors(paths, manifest))
    output = tmp_path / "condition"
    output.mkdir()
    assert _worker(0, manifest_path, output, paths, source, files, actors) == 0
    result = load_result(output / "clean.json")
    assert len(result.records) == 3
    assert all(record.terminal.submitted_success for record in result.records)
    assert all(sum(record.inference_counts) == 1 for record in result.records)
    assert _plain(result.actors) == actors
    events = [json.loads(line)["event"] for line in (output / "clean.log").read_text().splitlines()]
    assert events == ["condition_started", "condition_complete"]


@pytest.mark.parametrize("workers", [0, 9, True, 1.5])
def test_invalid_worker_limit_creates_nothing(tmp_path, workers):
    output = tmp_path / "new-study"
    with pytest.raises(ValueError, match="workers"):
        run_study(tmp_path / "absent.json", output, workers=workers)
    assert not output.exists()


def test_existing_output_is_never_reused(tmp_path):
    output = tmp_path / "partial"
    output.mkdir()
    marker = output / "original.txt"
    marker.write_bytes(b"preserve")
    with pytest.raises(FileExistsError, match="new directory"):
        run_study(Path("missing-manifest.json"), output)
    assert marker.read_bytes() == b"preserve"
