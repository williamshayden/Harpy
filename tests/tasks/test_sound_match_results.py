"""Saved bundles preserve replayable evidence, costs, membership, and error states."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from harpy.tasks.sound_match import (
    ActorSpec,
    EpisodeSpec,
    Propose,
    Submit,
    builtin_actor_spec,
    default_patch,
    evaluate,
    load_result,
    run,
    save_result,
    summarize,
    verify_replay,
)
from harpy.tasks.sound_match import results as results_module
from harpy.tasks.sound_match.models import _finite


@pytest.fixture
def result():
    initial = default_patch()
    episode = EpisodeSpec("opaque", replace(initial, notes_cents=(6100,)), render_budget=4)
    return evaluate([episode], [builtin_actor_spec("fixed"), builtin_actor_spec("pattern-search")])


def test_complete_bundle_roundtrip_and_explicit_replay(tmp_path, result, monkeypatch):
    output = save_result(result, tmp_path / "bundle")
    loaded = load_result(output)
    assert loaded.manifest() == result.manifest()
    assert [record.to_dict() for record in loaded.records] == [
        record.to_dict() for record in result.records
    ]
    verify_replay(loaded)
    assert sorted(
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    ) == [
        "audio/submitted-0000.wav",
        "audio/submitted-0001.wav",
        "audio/target-0000.wav",
        "complete.json",
        "manifest.json",
        "patches/submitted-0000.json",
        "patches/submitted-0001.json",
        "records.jsonl",
    ]

    # Summarization is a readout: neither actors nor expensive renderer replay is
    # invoked unless the caller explicitly requests verify_replay.
    def no_render(*args):
        raise AssertionError("readout rendered audio")

    monkeypatch.setattr(results_module, "render", no_render)
    assert load_result(output).metrics == result.metrics
    for format in ("text", "markdown", "csv"):
        text = summarize(output, format=format)
        assert "fixed" in text
        assert "pattern-search" in text


def test_result_inventories_are_portable_and_bind_builtin_identity(result):
    source = result.source
    assert source["kind"] == "package-python-files"
    assert any(row["path"] == "tasks/sound_match/renderer.py" for row in source["files"])
    for actor in result.actors:
        assert actor["information_track"] == "audio-only-metered"
        assert actor["metadata"]["training"] == "none"
        assert actor["metadata"]["model_calls"] == 0
        assert actor["artifact_files"][0]["path"] == "artifact-0000/actors.py"
    for row in [*source["files"], *result.actors[0]["artifact_files"]]:
        assert not row["path"].startswith("/")
        assert "\\" not in row["path"]
        assert ":" not in row["path"]
        assert len(row["sha256"]) == 64


def test_error_and_truncated_records_preserve_denominators_and_missing_scores(tmp_path):
    initial = default_patch()
    episode = EpisodeSpec("opaque", replace(initial, notes_cents=(6100,)), decision_budget=1)

    class Truncated:
        def decide(self, observation):
            return Propose(episode.target_patch)

    def broken_factory():
        raise RuntimeError("not included in artifacts")

    result = evaluate(
        [episode],
        [
            builtin_actor_spec("fixed"),
            ActorSpec("truncated", Truncated),
            ActorSpec("error", broken_factory),
        ],
    )
    output = save_result(result, tmp_path / "mixed")
    loaded = load_result(output)
    groups = {group["actor"]: group for group in loaded.metrics["groups"]}
    assert groups["fixed"]["submitted"] == 1
    assert groups["truncated"]["truncated"] == 1
    assert groups["error"]["error"] == 1
    assert groups["error"]["model_calls"] is None
    for name in ("truncated", "error"):
        assert groups[name]["submitted_distance"] == {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
        }
    assert groups["truncated"]["last_distance"]["mean"] == 0
    assert groups["truncated"]["best_distance"]["mean"] == 0
    assert len(list((output / "patches").glob("*.json"))) == 1
    assert len(list((output / "audio").glob("submitted-*.wav"))) == 1


def test_aggregation_separates_voice_count_and_editable_mask():
    tone, chord = default_patch(), default_patch(3)
    episodes = [
        EpisodeSpec("tone", tone),
        EpisodeSpec("chord", chord, initial_patch=chord),
        EpisodeSpec("masked", tone, editable_fields=("gain_db",)),
    ]
    result = evaluate(episodes, [builtin_actor_spec("fixed")])
    groups = result.metrics["groups"]
    assert len(groups) == 3
    assert [(group["voice_count"], group["episodes"]) for group in groups] == [
        (1, 1),
        (3, 1),
        (1, 1),
    ]
    assert groups[2]["editable_fields"] == ["gain_db"]


def test_save_is_create_only_and_incomplete_export_is_not_loadable(tmp_path, result, monkeypatch):
    output = save_result(result, tmp_path / "complete")
    manifest_before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        save_result(result, output)
    assert (output / "manifest.json").read_bytes() == manifest_before

    def fail(patch, expected):
        raise ValueError("export failed")

    monkeypatch.setattr(results_module, "_export_audio", fail)
    incomplete = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="export failed"):
        save_result(result, incomplete)
    assert incomplete.is_dir()
    assert not (incomplete / "complete.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        load_result(incomplete)


def test_completion_and_membership_are_required(tmp_path, result):
    output = save_result(result, tmp_path / "bundle")
    (output / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        load_result(output)
    (output / "extra.json").unlink()
    manifest = json.loads((output / "manifest.json").read_bytes())
    manifest["elapsed_seconds"] += 1
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest was changed"):
        load_result(output)


def test_verify_replay_rejects_changed_audio_evidence(result, monkeypatch):
    original = results_module.render

    def changed(patch):
        return original(patch) * 0.5

    monkeypatch.setattr(results_module, "render", changed)
    with pytest.raises(ValueError, match="target replay hash differs"):
        verify_replay(result)


def test_custom_actor_provenance_is_unknown_when_not_declared():
    class Custom:
        def decide(self, observation):
            return Submit()

    result = run(EpisodeSpec("opaque", default_patch()), ActorSpec("custom", Custom))
    assert result.actors[0]["metadata"]["status"] == "unknown"
    assert result.records[0].metrics["model_calls"] is None


@pytest.mark.parametrize("value", [10**400, -(10**400), float("inf"), float("nan"), True, "1"])
def test_finite_validation_rejects_unrepresentable_values_as_value_errors(value):
    with pytest.raises(ValueError):
        _finite(value, "elapsed_seconds")
