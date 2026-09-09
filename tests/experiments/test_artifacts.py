"""Artifact identity checks require no model deserialization."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from harpy.experiments.artifacts import (
    METADATA_SCHEMA_ID,
    ExperimentManifest,
    import_pitch_artifact,
    load_artifact,
)
from harpy.learning.artifacts import (
    FileRecord,
    PackageSourceStatus,
    RuntimeStatus,
    canonical_json_bytes,
)
from harpy.learning.models import DeviceName, ProfileName


def _artifact(tmp_path):
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "model.pt").write_bytes(b"not loaded by the metadata reader")
    (root / "metadata.json").write_bytes(
        canonical_json_bytes(
            {
                "schema_id": METADATA_SCHEMA_ID,
                "trainer": "pitch",
                "profile": "smoke",
                "seed": 0,
                "device": "cpu",
                "configuration": {},
                "summary": {},
                "runtime": {},
            }
        )
    )
    records = tuple(
        FileRecord(path.name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in root.iterdir()
    )
    manifest = ExperimentManifest(
        "pitch",
        "smoke",
        0,
        "cpu",
        records,
        canonical_json_bytes(
            {"status": "unknown", "reason": "external producer did not record source"}
        ),
    )
    (root / "manifest.json").write_bytes(canonical_json_bytes(manifest.to_document()))
    return load_artifact(root)


def test_unknown_provenance_roundtrip_and_independent_fingerprint(tmp_path):
    artifact = _artifact(tmp_path)
    assert artifact.provenance()["source"]["status"] == "unknown"
    assert ExperimentManifest.from_document(artifact.manifest.to_document()) == artifact.manifest
    fingerprint = artifact.fingerprint
    fingerprint["payload_sha256"].clear()
    assert artifact.fingerprint["payload_sha256"]
    artifact.verify_unchanged()


@pytest.mark.parametrize("payload", ["model.pt", "metadata.json"])
def test_payload_mutation_rejected_without_model_load(tmp_path, payload):
    artifact = _artifact(tmp_path)
    path = artifact.root / payload
    original = path.read_bytes()
    path.write_bytes(b"x" * len(original))
    with pytest.raises(ValueError, match="hash"):
        artifact.verify_unchanged()


def test_manifest_mutation_with_valid_new_payload_hashes_is_still_detected(tmp_path):
    artifact = _artifact(tmp_path)
    path = artifact.root / "manifest.json"
    document = json.loads(path.read_bytes())
    document["source"]["reason"] = "different producer"
    path.write_bytes(canonical_json_bytes(document))
    with pytest.raises(ValueError, match="changed"):
        artifact.verify_unchanged()


def test_extra_files_and_payload_symlinks_are_rejected(tmp_path):
    artifact = _artifact(tmp_path)
    (artifact.root / "extra.txt").write_text("unrecorded")
    with pytest.raises(ValueError, match="exactly"):
        load_artifact(artifact.root)
    (artifact.root / "extra.txt").unlink()
    model = artifact.model_path
    content = model.read_bytes()
    model.unlink()
    other = tmp_path / "other.pt"
    other.write_bytes(content)
    model.symlink_to(other)
    with pytest.raises(ValueError, match="symbolic link"):
        load_artifact(artifact.root)


def test_historical_import_retains_model_bytes_and_original_training_identity(
    monkeypatch, tmp_path
):
    import harpy.learning.pitch_artifacts as historical

    artifact = _artifact(tmp_path)
    old_document = {
        "schema_version": 2,
        "profile": "checkpoint",
        "seed": 2,
        "training_device": "cuda",
    }
    original_bytes = canonical_json_bytes(old_document)
    (artifact.root / "manifest.json").write_bytes(original_bytes)
    runtime = RuntimeStatus(
        "3.12", "Linux", "", "2", "1", "2", "2", DeviceName.CUDA, "GPU", "12", "550"
    )
    legacy = SimpleNamespace(
        root=artifact.root,
        manifest=SimpleNamespace(
            files=artifact.manifest.files,
            profile=ProfileName.CHECKPOINT,
            seed=2,
            runtime=runtime,
            source=PackageSourceStatus("harpy-audio", "1.0.0", "a" * 64),
            to_document=lambda: old_document,
        ),
        file=lambda name: artifact.root / name,
        document=lambda name: {"preserved_payload": name},
    )
    monkeypatch.setattr(historical, "load_pitch_artifact", lambda path: legacy)
    imported = import_pitch_artifact(artifact.root, tmp_path / "imported")
    assert imported.model_path.read_bytes() == artifact.model_path.read_bytes()
    assert (imported.manifest.profile, imported.manifest.seed, imported.manifest.device) == (
        "checkpoint",
        2,
        "cuda",
    )
    metadata = json.loads((imported.root / "metadata.json").read_bytes())
    origin = metadata["configuration"]["origin"]
    assert origin["manifest"] == old_document
    assert origin["manifest_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert origin["operation"] == "exact_model_byte_import_no_training"
    assert imported.provenance()["source"]["package_sha256"] == "a" * 64
    with pytest.raises(FileExistsError):
        import_pitch_artifact(artifact.root, imported.root)
