"""Task-neutral recovery boundaries and bounded streaming aggregation."""

from __future__ import annotations

import hashlib
import tracemalloc
from pathlib import Path

import pytest

from harpy._journal import (
    Journal,
    MetricStore,
    ScanState,
    artifact_inventory,
    lock_bundle,
    manifest_for,
    object_fields,
    scan,
)
from harpy._serialization import canonical_json_bytes, read_json_document


def key(index):
    return [f"episode-{index}"]


def validate(payload, index):
    object_fields(payload, {"key", "artifacts", "value"})
    if payload["key"] != key(index) or type(payload["value"]) is not int:
        raise ValueError("malformed member")


def partial(path, count=1, expected=2):
    manifest = manifest_for("fixture", {"frozen": True}, expected, None)
    with Journal(path, manifest) as journal:
        journal.recover(key_at=key, validate=validate)
        for i in range(count):
            journal.begin(key(i))
            journal.commit({"key": key(i), "artifacts": [], "value": i})
    return manifest


def read(path, manifest):
    state = ScanState()
    with lock_bundle(path, exclusive=False), MetricStore() as references:
        rows = list(
            scan(path, manifest, state, key_at=key, validate=validate, references=references)
        )
    return state, rows


@pytest.mark.parametrize("name", ["commits.jsonl", "attempts.jsonl"])
def test_only_unterminated_tail_is_recoverable_not_complete_malformed_line(tmp_path, name):
    path = tmp_path / "bundle"
    manifest = partial(path)
    original = (path / name).read_bytes()
    (path / name).write_bytes(original + b'{"broken":\n')
    with pytest.raises(ValueError, match="valid JSON"):
        read(path, manifest)
    with (
        pytest.raises(ValueError, match="valid JSON"),
        Journal(path, manifest, resume=True) as journal,
    ):
        journal.recover(key_at=key, validate=validate)
    assert (path / name).read_bytes() == original + b'{"broken":\n'


def test_recovery_keeps_durable_attempts_and_prunes_only_owned_uncommitted_files(tmp_path):
    path = tmp_path / "bundle"
    manifest = partial(path)
    with Journal(path, manifest, resume=True) as journal:
        journal.recover(key_at=key, validate=validate)
        journal.begin(key(1))
        orphan = journal.artifact("unused", b"interrupted export", "wav")
    artifact_temp = path / "artifacts" / f".{'a' * 64}.{'1' * 32}.tmp.wav"
    metadata_temp = path / f".complete.{'2' * 32}.tmp.json"
    artifact_temp.write_bytes(b"torn")
    metadata_temp.write_bytes(b"torn")
    with (path / "records.jsonl").open("ab") as stream:
        stream.write(b'{"torn":')
    with Journal(path, manifest, resume=True) as journal:
        journal.recover(key_at=key, validate=validate)
        assert journal.state.attempts == 2 and journal.state.completed == 1
        journal.begin(key(1))
        journal.commit({"key": key(1), "artifacts": [], "value": 1})
        journal.finish(key_at=key, validate=validate)
    state, rows = read(path, manifest)
    assert state.is_complete and state.interrupted_attempts == 1 and len(rows) == 2
    assert not artifact_temp.exists() and not metadata_temp.exists()
    assert not (path / orphan["path"]).exists()


def test_pruning_rejects_unrelated_user_file_and_never_deletes_it(tmp_path):
    path = tmp_path / "bundle"
    manifest = partial(path, expected=1)
    unrelated = path / "artifacts" / ".private.tmp.wav"
    unrelated.write_bytes(b"do not delete")
    with (
        pytest.raises(ValueError, match="unrecognized"),
        Journal(path, manifest, resume=True) as journal,
    ):
        journal.recover(key_at=key, validate=validate)
        journal.finish(key_at=key, validate=validate)
    assert unrelated.read_bytes() == b"do not delete"


def test_resume_recovers_kill_after_completion_link_before_temporary_unlink(tmp_path):
    path = tmp_path / "bundle"
    manifest = partial(path, expected=1)
    with Journal(path, manifest, resume=True) as journal:
        journal.recover(key_at=key, validate=validate)
        journal.finish(key_at=key, validate=validate)
    temporary = path / f".complete.{'3' * 32}.tmp.json"
    temporary.hardlink_to(path / "complete.json")
    with Journal(path, manifest, resume=True) as journal:
        journal.recover(key_at=key, validate=validate)
        assert journal.state.is_complete
    assert not temporary.exists() and (path / "complete.json").is_file()
    assert read(path, manifest)[0].is_complete


@pytest.mark.parametrize(
    "name", ["records.jsonl", "commits.jsonl", "attempts.jsonl", "manifest.json", "artifacts"]
)
def test_symbolic_link_members_are_rejected(tmp_path, name):
    path = tmp_path / "bundle"
    manifest = partial(path)
    target = path / name
    moved = tmp_path / "outside"
    target.rename(moved)
    target.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(ValueError, match="symbolic links"):
        read(path, manifest)


def test_concurrent_writer_and_reader_reject_without_mutation(tmp_path):
    path = tmp_path / "bundle"
    manifest = partial(path)
    before = (path / "attempts.jsonl").read_bytes()
    with lock_bundle(path):
        with pytest.raises(ValueError, match="in use"), Journal(path, manifest, resume=True):
            pass
        with pytest.raises(ValueError, match="in use"):
            read(path, manifest)
    assert (path / "attempts.jsonl").read_bytes() == before


def test_portable_artifact_inventory_binds_bytes_not_absolute_location(tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    (first / "weights.bin").write_bytes(b"weights")
    original = artifact_inventory((first,))
    moved = tmp_path / "moved"
    first.rename(moved)
    assert artifact_inventory((moved,)) == original
    (moved / "weights.bin").write_bytes(b"changed")
    assert artifact_inventory((moved,)) != original


def test_resume_binds_configuration_but_allows_new_local_locators(tmp_path):
    path = tmp_path / "bundle"
    manifest = manifest_for("fixture", {}, 1, {"dataset_sha256": "a" * 64})
    with Journal(path, manifest, locators={"model": "/old/location"}) as journal:
        journal.recover(key_at=key, validate=validate)
    with Journal(path, manifest, resume=True, locators={"model": "/new/location"}) as journal:
        journal.recover(key_at=key, validate=validate)
    changed = {**manifest, "configuration": {"dataset_sha256": "b" * 64}}
    with pytest.raises(ValueError, match="resume identity"), Journal(path, changed, resume=True):
        pass


def test_forged_hashes_do_not_bypass_task_record_validation(tmp_path):
    path = tmp_path / "bundle"
    manifest = partial(path)
    payload = {"key": key(0), "artifacts": [], "value": "invalid"}
    data = canonical_json_bytes(payload)
    (path / "records.jsonl").write_bytes(data)
    commit = read_json_document(path / "commits.jsonl")
    commit.update(length=len(data), sha256=hashlib.sha256(data).hexdigest())
    (path / "commits.jsonl").write_bytes(canonical_json_bytes(commit))
    with pytest.raises(ValueError, match="malformed member"):
        read(path, manifest)


@pytest.mark.parametrize(
    "relative",
    ["../escape.wav", "/tmp/escape.wav", "artifacts/../escape.wav", "artifacts\\escape.wav"],
)
def test_unsafe_committed_artifact_paths_rejected_before_open(tmp_path, relative):
    path = tmp_path / "bundle"
    manifest = manifest_for("fixture", {}, 1, None)
    with Journal(path, manifest) as journal:
        journal.recover(key_at=key, validate=validate)
        journal.begin(key(0))
        journal.commit(
            {
                "key": key(0),
                "value": 1,
                "artifacts": [
                    {"role": "audio", "path": relative, "sha256": "a" * 64, "size_bytes": 0}
                ],
            }
        )
    with pytest.raises(ValueError, match="portable path"):
        read(path, manifest)


def test_disk_metric_store_retains_exact_quantiles_with_bounded_python_memory():
    tracemalloc.start()
    try:
        with MetricStore() as store:
            location = Path(store._temporary.name)
            for i in range(500):
                store.add("group", "metric", i)
            initial, _ = tracemalloc.get_traced_memory()
            for i in range(500, 20_000):
                store.add("group", "metric", i)
                store.reference(f"artifacts/{i:064x}.wav")
            current, peak = tracemalloc.get_traced_memory()
            distribution = store.distribution("group", "metric")
            assert distribution["count"] == 20_000
            assert distribution["median"] == distribution["mean"] == 9999.5
            assert distribution["p90"] == pytest.approx(17999.1)
            assert current - initial < 1024 * 1024 and peak < 2 * 1024 * 1024
            assert store.connection.execute("PRAGMA cache_size").fetchone()[0] == -1024
        assert not location.exists()
    finally:
        tracemalloc.stop()
