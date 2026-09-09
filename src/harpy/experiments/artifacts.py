"""Small, create-only model artifacts for the active experiment API.

Hashes establish file identity, not trust. PPO archives must come from a trusted
local producer; pitch checkpoints contain only a validated tensor state dict.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from harpy.learning.artifacts import (
    FileRecord,
    _verify_file_record,
    canonical_json_bytes,
    decode_json_bytes,
)

ARTIFACT_SCHEMA_ID = "harpy-experiment-artifact-v1"
METADATA_SCHEMA_ID = "harpy-experiment-training-v1"


def _object(value: object, keys: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} must contain exactly {sorted(keys)}")
    return value


def validate_training_request(trainer: str, profile: str, seed: int, device: str) -> None:
    if trainer not in ("pitch", "ppo"):
        raise ValueError("trainer must be pitch or ppo")
    if profile not in ("smoke", "checkpoint"):
        raise ValueError("profile must be smoke or checkpoint")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in 0..2**32-1")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    """Immutable canonical manifest with deliberately explicit unknown provenance."""

    trainer: str
    profile: str
    seed: int
    device: str
    files: tuple[FileRecord, ...]
    source_bytes: bytes

    def __post_init__(self) -> None:
        validate_training_request(self.trainer, self.profile, self.seed, self.device)
        files = tuple(self.files)
        expected = {"metadata.json", "model.pt" if self.trainer == "pitch" else "model.zip"}
        if not all(isinstance(record, FileRecord) for record in files) or (
            len(files) != 2 or {record.relative_path for record in files} != expected
        ):
            raise ValueError("manifest must identify exactly its model and metadata payloads")
        object.__setattr__(
            self, "files", tuple(sorted(files, key=lambda record: record.relative_path))
        )
        source = decode_json_bytes(self.source_bytes)
        if not isinstance(source, dict) or source.get("status") not in ("known", "unknown"):
            raise ValueError("source must explicitly be known or unknown")
        if source["status"] == "unknown":
            _object(source, {"status", "reason"}, "unknown source")
            if not isinstance(source["reason"], str) or not source["reason"]:
                raise ValueError("unknown source must explain its missing provenance")
        else:
            from harpy.learning.artifacts import _digest, _source_from_document

            _object(source, {"status", "identity", "package_sha256"}, "known source")
            _source_from_document(source["identity"])
            if source["package_sha256"] is not None:
                _digest(source["package_sha256"], "package_sha256")
        object.__setattr__(self, "source_bytes", canonical_json_bytes(source))

    def to_document(self) -> dict:
        return {
            "schema_id": ARTIFACT_SCHEMA_ID,
            "trainer": self.trainer,
            "profile": self.profile,
            "seed": self.seed,
            "device": self.device,
            "source": decode_json_bytes(self.source_bytes),
            "files": [
                {"path": item.relative_path, "size_bytes": item.size_bytes, "sha256": item.sha256}
                for item in self.files
            ],
        }

    @classmethod
    def from_document(cls, value: object) -> ExperimentManifest:
        doc = _object(
            value,
            {"schema_id", "trainer", "profile", "seed", "device", "source", "files"},
            "manifest",
        )
        if doc["schema_id"] != ARTIFACT_SCHEMA_ID or not isinstance(doc["files"], list):
            raise ValueError("unsupported experiment artifact schema or file inventory")
        records = []
        for item in doc["files"]:
            item = _object(item, {"path", "size_bytes", "sha256"}, "file record")
            records.append(FileRecord(item["path"], item["size_bytes"], item["sha256"]))
        return cls(
            doc["trainer"],
            doc["profile"],
            doc["seed"],
            doc["device"],
            tuple(records),
            canonical_json_bytes(doc["source"]),
        )


@dataclass(frozen=True, slots=True)
class LoadedExperimentArtifact:
    root: Path
    manifest: ExperimentManifest
    manifest_sha256: str

    @property
    def model_path(self) -> Path:
        return self.root / ("model.pt" if self.manifest.trainer == "pitch" else "model.zip")

    @property
    def fingerprint(self) -> dict:
        return {
            "manifest_sha256": self.manifest_sha256,
            "payload_sha256": {item.relative_path: item.sha256 for item in self.manifest.files},
        }

    def provenance(self) -> dict:
        return {
            "artifact_schema": ARTIFACT_SCHEMA_ID,
            "trainer": self.manifest.trainer,
            "profile": self.manifest.profile,
            "seed": self.manifest.seed,
            "training_device": self.manifest.device,
            "source": decode_json_bytes(self.manifest.source_bytes),
            **self.fingerprint,
        }

    def verify_unchanged(self) -> None:
        current = load_artifact(self.root)
        if current.manifest != self.manifest or current.fingerprint != self.fingerprint:
            raise ValueError("artifact changed since it was loaded")

    def load_model(self, *, device: str = "cpu") -> object:
        validate_training_request(
            self.manifest.trainer, self.manifest.profile, self.manifest.seed, device
        )
        self.verify_unchanged()
        model = _load_model_file(self.model_path, trainer=self.manifest.trainer, device=device)
        self.verify_unchanged()
        return model


def _load_model_file(path: Path, *, trainer: str, device: str) -> object:
    if trainer == "pitch":
        from harpy.learning.dependencies import require_pitch_dependencies
        from harpy.learning.pitch import load_pitch_estimator_model

        torch = require_pitch_dependencies().torch
        return load_pitch_estimator_model(path, device=torch.device(device))
    from harpy.learning.ppo import _load_ppo_model, _validate_ppo_model

    model = _load_ppo_model(path, device=device)
    _validate_ppo_model(model, expected_parameter_count=75_816)
    return model


def load_artifact(path: Path) -> LoadedExperimentArtifact:
    """Validate identity and metadata without deserializing either model format."""
    if not isinstance(path, Path) or path.is_symlink():
        raise ValueError("artifact path must be a real directory Path")
    root = path.resolve(strict=True)
    manifest_path = root / "manifest.json"
    if not root.is_dir() or manifest_path.is_symlink():
        raise ValueError("artifact must contain a regular manifest.json")
    content = manifest_path.read_bytes()
    manifest = ExperimentManifest.from_document(decode_json_bytes(content))
    expected = {"manifest.json", *(item.relative_path for item in manifest.files)}
    if {item.name for item in root.iterdir()} != expected:
        raise ValueError("artifact directory must contain exactly its declared payloads")
    for record in manifest.files:
        _verify_file_record(root, record)
    metadata = decode_json_bytes((root / "metadata.json").read_bytes())
    _object(
        metadata,
        {
            "schema_id",
            "trainer",
            "profile",
            "seed",
            "device",
            "configuration",
            "summary",
            "runtime",
        },
        "training metadata",
    )
    if metadata["schema_id"] != METADATA_SCHEMA_ID or any(
        metadata[key] != getattr(manifest, key) for key in ("trainer", "profile", "seed", "device")
    ):
        raise ValueError("training metadata must match its manifest")
    validate_training_request(
        metadata["trainer"], metadata["profile"], metadata["seed"], metadata["device"]
    )
    for key in ("configuration", "summary", "runtime"):
        if not isinstance(metadata[key], dict):
            raise ValueError(f"training {key} must be an object")
    return LoadedExperimentArtifact(root, manifest, hashlib.sha256(content).hexdigest())


def import_pitch_artifact(source: Path, output: Path) -> LoadedExperimentArtifact:
    """Copy a validated historical pitch model without rewriting its training history."""
    from harpy.learning.artifacts import _runtime_to_document, _source_to_document
    from harpy.learning.pitch_artifacts import load_pitch_artifact

    if not isinstance(source, Path) or not isinstance(output, Path):
        raise ValueError("source and output must be Paths")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    original = load_pitch_artifact(source)
    original_bytes = (original.root / "manifest.json").read_bytes()
    original_document = decode_json_bytes(original_bytes)
    if original_document != original.manifest.to_document():
        raise ValueError("source artifact manifest changed during import")
    model_bytes = original.file("model.pt").read_bytes()
    expected_model = next(
        record for record in original.manifest.files if record.relative_path == "model.pt"
    )
    if hashlib.sha256(model_bytes).hexdigest() != expected_model.sha256:
        raise ValueError("source artifact model changed during import")
    manifest = original.manifest
    metadata = {
        "schema_id": METADATA_SCHEMA_ID,
        "trainer": "pitch",
        "profile": manifest.profile.value,
        "seed": manifest.seed,
        "device": manifest.runtime.device.value,
        "configuration": {
            "origin": {
                "manifest_sha256": hashlib.sha256(original_bytes).hexdigest(),
                "manifest": original_document,
                "training_configuration": original.document("training-config.json"),
                "operation": "exact_model_byte_import_no_training",
            },
        },
        "summary": original.document("training-summary.json"),
        "runtime": _runtime_to_document(manifest.runtime),
    }
    for record in manifest.files:
        _verify_file_record(original.root, record)
    if (original.root / "manifest.json").read_bytes() != original_bytes:
        raise ValueError("source artifact manifest changed during import")
    output.mkdir(parents=True, exist_ok=False)
    (output / "model.pt").write_bytes(model_bytes)
    (output / "metadata.json").write_bytes(canonical_json_bytes(metadata))
    files = tuple(
        FileRecord(path.name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(output.iterdir())
    )
    source_status = {
        "status": "known",
        "identity": _source_to_document(manifest.source),
        "package_sha256": getattr(manifest.source, "package_sha256", None),
    }
    imported = ExperimentManifest(
        "pitch",
        manifest.profile.value,
        manifest.seed,
        manifest.runtime.device.value,
        files,
        canonical_json_bytes(source_status),
    )
    (output / "manifest.json").write_bytes(canonical_json_bytes(imported.to_document()))
    return load_artifact(output)


__all__ = [
    "ARTIFACT_SCHEMA_ID",
    "METADATA_SCHEMA_ID",
    "ExperimentManifest",
    "LoadedExperimentArtifact",
    "import_pitch_artifact",
    "load_artifact",
]
