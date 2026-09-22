"""Durable sound bundles survive interrupted work without inventing evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from harpy._serialization import canonical_json_bytes, read_json_document
from harpy.tasks.sound_match import (
    ActorSpec,
    EpisodeSpec,
    Submit,
    builtin_actor_spec,
    default_patch,
    durable,
)
from harpy.tasks.sound_match.durable import (
    evaluate_bundle,
    iter_records,
    read_bundle,
    read_configuration,
    summarize_bundle,
    verify_replay,
)


@pytest.fixture(autouse=True)
def stable_package_source(monkeypatch):
    # Scientific behavior cases may run alongside unrelated source edits. Actual
    # subprocess qualification and the explicit identity-change case do not rely
    # on a mutable checkout snapshot between individual candidate evaluations.
    snapshot = durable.source_identity()
    monkeypatch.setattr(durable, "source_identity", lambda: snapshot)


def episodes():
    initial = default_patch()
    return [
        EpisodeSpec("first", replace(initial, notes_cents=(6100,)), render_budget=4),
        EpisodeSpec("second", replace(initial, notes_cents=(6200,)), render_budget=4),
    ]


def test_complete_sound_bundle_streams_validated_records(tmp_path):
    output = tmp_path / "bundle"
    events = []
    result = evaluate_bundle(
        episodes(), [builtin_actor_spec("fixed")], output, progress=events.append
    )
    assert result.complete and result.completed == result.expected == 2
    assert result.attempts == 2 and result.interrupted_attempts == 0
    assert len(events) == 2 and events[-1]["completed"] == 2
    records = list(iter_records(output))
    assert [row["key"] for row in records] == [["first", "fixed"], ["second", "fixed"]]
    assert result.metrics["groups"][0]["submitted"] == 2
    assert "COMPLETE" in summarize_bundle(output)
    verify_replay(output)
    assert read_configuration(output)["task_id"] == "sound-match-v1"
    with pytest.raises(FileExistsError):
        evaluate_bundle(episodes(), [builtin_actor_spec("fixed")], output)


def test_committed_prefix_resumes_without_repeating_finished_members(tmp_path):
    output = tmp_path / "partial"

    def stop(event):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        evaluate_bundle(episodes(), [builtin_actor_spec("fixed")], output, progress=stop)
    with pytest.raises(ValueError, match="incomplete"):
        read_bundle(output)
    partial = read_bundle(output, allow_incomplete=True)
    assert partial.completed == 1 and partial.missing == 1 and not partial.complete
    assert "INCOMPLETE" in summarize_bundle(output, allow_incomplete=True)
    first = next(iter_records(output, allow_incomplete=True))
    events = []
    completed = evaluate_bundle(
        episodes(), [builtin_actor_spec("fixed")], output, resume=True, progress=events.append
    )
    assert completed.complete and completed.interrupted_attempts == 0
    assert [event["key"] for event in events] == [["second", "fixed"]]
    assert next(iter_records(output)) == first


def test_interrupted_attempt_remains_unknown_after_resume(tmp_path):
    output = tmp_path / "interrupted"

    class Actor:
        interrupt = True

        def decide(self, observation):
            if self.interrupt:
                raise KeyboardInterrupt
            return Submit(0)

    spec = ActorSpec("controlled", Actor)
    with pytest.raises(KeyboardInterrupt):
        evaluate_bundle(episodes()[:1], [spec], output)
    partial = read_bundle(output, allow_incomplete=True)
    assert partial.completed == 0 and partial.interrupted_attempts == 1
    Actor.interrupt = False
    result = evaluate_bundle(episodes()[:1], [spec], output, resume=True)
    assert result.completed == 1 and result.attempts == 2
    assert result.interrupted_attempts == 1
    assert result.to_document()["interrupted_costs"] == "unknown"


def test_resume_rejects_membership_or_actor_identity_changes(tmp_path):
    output = tmp_path / "bundle"
    evaluate_bundle(episodes()[:1], [builtin_actor_spec("fixed")], output)
    with pytest.raises(ValueError, match="resume identity"):
        evaluate_bundle(episodes(), [builtin_actor_spec("fixed")], output, resume=True)
    with pytest.raises(ValueError, match="resume identity"):
        evaluate_bundle(
            episodes()[:1], [builtin_actor_spec("random-search", seed=1)], output, resume=True
        )


def test_resume_recovers_only_uncommitted_tails(tmp_path):
    output = tmp_path / "partial"

    def stop(event):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        evaluate_bundle(episodes(), [builtin_actor_spec("fixed")], output, progress=stop)
    with (output / "records.jsonl").open("ab") as stream:
        stream.write(b'{"uncommitted":"torn')
    with (output / "commits.jsonl").open("ab") as stream:
        stream.write(b'{"index":2')
    with (output / "attempts.jsonl").open("ab") as stream:
        stream.write(b'{"attempt":2')
    result = evaluate_bundle(episodes(), [builtin_actor_spec("fixed")], output, resume=True)
    assert result.complete and result.completed == 2
    data = (output / "records.jsonl").read_bytes()
    (output / "records.jsonl").write_bytes(
        data.replace(b'"status":"submitted"', b'"status":"corrupted"', 1)
    )
    with pytest.raises(ValueError, match="committed record"):
        read_bundle(output)


def test_relocated_weights_preserve_identity_but_changed_bytes_reject_resume(tmp_path):
    weights = tmp_path / "original" / "weights.bin"
    weights.parent.mkdir()
    weights.write_bytes(b"model-v1")
    spec = replace(builtin_actor_spec("fixed"), artifact_paths=(weights,))
    output = tmp_path / "bundle"
    evaluate_bundle(episodes()[:1], [spec], output)
    relocated = tmp_path / "moved"
    weights.parent.rename(relocated)
    spec = replace(spec, artifact_paths=(relocated / weights.name,))
    assert evaluate_bundle(episodes()[:1], [spec], output, resume=True).complete
    spec.artifact_paths[0].write_bytes(b"model-v2")
    with pytest.raises(ValueError, match="resume identity"):
        evaluate_bundle(episodes()[:1], [spec], output, resume=True)
    # A completed report is self-contained and does not follow weight locators.
    spec.artifact_paths[0].unlink()
    assert read_bundle(output).complete


@pytest.mark.parametrize("identity", ["source", "objective", "renderer"])
def test_resume_rejects_changed_evaluator_identity(tmp_path, monkeypatch, identity):
    output = tmp_path / "bundle"
    evaluate_bundle(episodes()[:1], [builtin_actor_spec("fixed")], output)
    if identity == "source":
        changed = durable.source_identity()
        changed = json.loads(json.dumps(changed))
        changed["files"][0]["sha256"] = "0" * 64
        changed["sha256"] = hashlib.sha256(
            canonical_json_bytes({"files": changed["files"]})
        ).hexdigest()
        monkeypatch.setattr(durable, "source_identity", lambda: changed)
    else:
        monkeypatch.setattr(durable, identity.upper() + "_ID", "changed-identity")
    with pytest.raises(ValueError, match="resume identity"):
        evaluate_bundle(episodes()[:1], [builtin_actor_spec("fixed")], output, resume=True)


def test_running_weight_change_is_not_committed(tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"before")

    class Actor:
        def decide(self, observation):
            weights.write_bytes(b"after")
            return Submit(0)

    output = tmp_path / "bundle"
    spec = ActorSpec("mutator", Actor, artifact_paths=(weights,))
    with pytest.raises(ValueError, match="running evaluation identity"):
        evaluate_bundle(episodes()[:1], [spec], output)
    partial = read_bundle(output, allow_incomplete=True)
    assert partial.completed == 0 and partial.interrupted_attempts == 1


def test_resume_binds_installed_torch_version_without_importing_it(tmp_path, monkeypatch):
    import sys

    was_imported = "torch" in sys.modules
    monkeypatch.setattr(durable, "version", lambda package: "2.13.0+cpu")
    output = tmp_path / "bundle"
    evaluate_bundle(episodes()[:1], [builtin_actor_spec("fixed")], output)
    assert read_configuration(output)["identity"]["runtime"]["torch"] == "2.13.0+cpu"
    monkeypatch.setattr(durable, "version", lambda package: "2.14.0+cpu")
    with pytest.raises(ValueError, match="resume identity"):
        evaluate_bundle(episodes()[:1], [builtin_actor_spec("fixed")], output, resume=True)
    assert read_bundle(output).complete
    assert ("torch" in sys.modules) == was_imported


@pytest.mark.parametrize("failure", ["actor", "cleanup", "target", "truncate"])
def test_terminal_failures_and_truncations_commit_once_with_truthful_costs(
    tmp_path, monkeypatch, failure
):
    created, closed = [], []

    class Actor:
        def __init__(self):
            created.append(True)

        def decide(self, observation):
            if failure == "actor":
                raise RuntimeError("actor failed")
            return object() if failure == "truncate" else Submit(0)

        def close(self):
            closed.append(True)
            if failure == "cleanup":
                raise RuntimeError("cleanup failed")

    if failure == "target":
        from harpy.tasks.sound_match import runner

        def broken_render(patch):
            raise RuntimeError("renderer failed")

        monkeypatch.setattr(runner, "render", broken_render)
    output = tmp_path / "bundle"
    spec = ActorSpec("failure", Actor)
    episode = replace(episodes()[0], decision_budget=2)
    result = evaluate_bundle([episode], [spec], output)
    assert result.complete and result.completed == result.attempts == 1
    group = result.metrics["groups"][0]
    assert group["truncated" if failure == "truncate" else "error"] == 1
    assert created == closed == ([] if failure == "target" else [True])
    assert evaluate_bundle([episode], [spec], output, resume=True).attempts == 1
    assert created == closed == ([] if failure == "target" else [True])


@pytest.mark.parametrize("field", ["status", "metric", "patch", "target"])
def test_task_validation_rejects_rehashed_malformed_scientific_evidence(tmp_path, field):
    output = tmp_path / "partial"

    def stop(event):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        evaluate_bundle(episodes(), [builtin_actor_spec("fixed")], output, progress=stop)
    payload = read_json_document(output / "records.jsonl")
    if field == "status":
        payload["record"]["status"] = "invented"
    elif field == "metric":
        payload["record"]["initial"]["objective"]["distance"] = -1
    elif field == "patch":
        payload["record"]["initial"]["patch"]["notes_cents"] = [999999]
    else:
        payload["target"]["audio_sha256"] = "0" * 64
    data = canonical_json_bytes(payload)
    (output / "records.jsonl").write_bytes(data)
    commit = read_json_document(output / "commits.jsonl")
    commit.update(length=len(data), sha256=hashlib.sha256(data).hexdigest())
    (output / "commits.jsonl").write_bytes(canonical_json_bytes(commit))
    with pytest.raises(ValueError):
        read_bundle(output, allow_incomplete=True)


def test_export_content_changes_are_rejected(tmp_path):
    output = tmp_path / "bundle"
    evaluate_bundle(episodes()[:1], [builtin_actor_spec("fixed")], output)
    row = next(iter_records(output))
    artifact = output / row["artifacts"][0]["path"]
    artifact.write_bytes(artifact.read_bytes()[:-1] + b"x")
    with pytest.raises(ValueError, match="artifact content"):
        read_bundle(output)
