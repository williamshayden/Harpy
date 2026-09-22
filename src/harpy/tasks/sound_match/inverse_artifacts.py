"""Strict, portable inverse artifacts; metadata inspection does not import Torch."""

from __future__ import annotations

import hashlib
import io
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from harpy._serialization import (
    canonical_json_bytes,
    decode_json_bytes,
    sha256_file,
    write_new_bytes,
)
from harpy.tasks.sound_match.inverse_features import (
    ARCHITECTURE_ID,
    DECODER_ID,
    OBSERVATION_ID,
    SUPERVISION_ID,
    architecture_configuration,
    decoder_configuration,
    feature_configuration,
    supervision_configuration,
)
from harpy.tasks.sound_match.models import TASK_ID, _finite, _integer, _object, _plain
from harpy.tasks.sound_match.objective import OBJECTIVE_ID
from harpy.tasks.sound_match.renderer import RENDERER_ID

ARTIFACT_SCHEMA_ID = "harpy.sound-match.inverse-artifact/1"
TRAINING_SCHEMA_ID = "harpy.sound-match.inverse-training/1"
SUMMARY_SCHEMA_ID = "harpy.sound-match.inverse-training-summary/1"
PROFILE_NAMES = ("quick", "reference")
PAYLOAD_NAMES = ("model.pt", "training-manifest.json", "training-summary.json")


def training_configuration(profile: str) -> dict:
    if profile not in PROFILE_NAMES:
        raise ValueError(f"inverse profile must be one of {PROFILE_NAMES}")
    return {
        "profile": profile,
        "purpose": "engineering-smoke-only" if profile == "quick" else "experimental-reference",
        "epochs": 1 if profile == "quick" else 6,
        "batch_size": 4,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_norm_limit": 5.0,
        "training_examples_per_cell": 1 if profile == "quick" else None,
        "validation_examples_per_cell": 1 if profile == "quick" else None,
        "subset_rule": "first-examples-per-cell-in-frozen-manifest-order",
        "epoch_order": "numpy-PCG64-seed-permutation-continuing-stream",
        "torch_deterministic_algorithms": True,
        "torch_threads": 1,
        "loader_workers": 0,
        "waveform_cache": "none-one-waveform-at-a-time",
        "feature_cache": "verified-read-only-npy-disk-cache-batch-only-tensors-v1",
        "selection_epochs": "every-completed-training-epoch",
        "initial_untrained_state_eligible": False,
    }


def _equal(actual: object, expected: object, label: str) -> None:
    if canonical_json_bytes({"value": _plain(actual)}) != canonical_json_bytes(
        {"value": _plain(expected)}
    ):
        raise ValueError(f"unsupported or inconsistent {label}")


def _digest(value: object, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _selected_entries(manifest, per_cell: int | None) -> tuple[Mapping, ...]:
    counts: dict[str, int] = {}
    selected = []
    for entry in manifest.entries:
        cell = entry["cell"]
        if per_cell is None or counts.get(cell, 0) < per_cell:
            selected.append(entry)
            counts[cell] = counts.get(cell, 0) + 1
    return tuple(selected)


def dataset_binding(manifest, selected: tuple[Mapping, ...]) -> dict:
    return {
        "split": manifest.split,
        "manifest_sha256": manifest.manifest_sha256,
        "full_examples": len(manifest.entries),
        "selected": [
            {
                "episode_id": entry["episode"]["episode_id"],
                "cell": entry["cell"],
                "source_family_id": entry["source_family_id"],
                "target_audio_sha256": entry["target_audio_sha256"],
            }
            for entry in selected
        ],
    }


def training_membership(profile: str):
    from harpy.tasks.sound_match.datasets import load_manifest

    configuration = training_configuration(profile)
    train, validation = load_manifest("train"), load_manifest("validation")
    selected_train = _selected_entries(train, configuration["training_examples_per_cell"])
    selected_validation = _selected_entries(
        validation, configuration["validation_examples_per_cell"]
    )
    return train, validation, selected_train, selected_validation


def require_torch():
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Inverse learning requires PyTorch. Install `harpy-audio[learn]`."
        ) from error
    return torch


def state_sha256(state: Mapping) -> str:
    """Stable tensor identity independent of Torch's archive container bytes."""
    inventory = []
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        inventory.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
            }
        )
    return hashlib.sha256(canonical_json_bytes({"tensors": inventory})).hexdigest()


def _validate_source(source: object) -> None:
    source = _object(source, {"kind", "package_version", "files", "sha256"})
    if source["kind"] != "package-python-files" or type(source["package_version"]) is not str:
        raise ValueError("unsupported training source identity")
    if not isinstance(source["files"], list) or not source["files"]:
        raise ValueError("source inventory must contain Python files")
    names = []
    for row in source["files"]:
        row = _object(row, {"path", "size_bytes", "sha256"})
        name = row["path"]
        if (
            type(name) is not str
            or not name.endswith(".py")
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or "\\" in name
            or ":" in name
        ):
            raise ValueError("source inventory paths must be portable Python paths")
        names.append(name)
        _integer(row["size_bytes"], "source size", 1)
        _digest(row["sha256"], "source file hash")
    if names != sorted(set(names)):
        raise ValueError("source inventory must be sorted and unique")
    _equal(
        source["sha256"],
        hashlib.sha256(canonical_json_bytes({"files": source["files"]})).hexdigest(),
        "source inventory digest",
    )


def _validate_training(document: object) -> dict:
    from harpy.tasks.sound_match.datasets import SELECTION_CRITERION
    from harpy.tasks.sound_match.inverse_cache import CACHE_SCHEMA_ID, cache_identity, cache_key

    document = dict(
        _object(
            document,
            {
                "schema_id",
                "seed",
                "configuration",
                "architecture",
                "features",
                "decoder",
                "supervision",
                "datasets",
                "selection",
                "source",
                "runtime",
                "feature_cache",
            },
        )
    )
    _equal(document["schema_id"], TRAINING_SCHEMA_ID, "training schema")
    _integer(document["seed"], "training seed")
    if document["seed"] >= 2**32:
        raise ValueError("training seed must be below 2**32")
    configuration = _object(document["configuration"])
    profile = configuration.get("profile")
    _equal(configuration, training_configuration(profile), "training configuration")
    for name, expected in (
        ("architecture", architecture_configuration()),
        ("features", feature_configuration()),
        ("decoder", decoder_configuration()),
        ("supervision", supervision_configuration()),
        ("selection", SELECTION_CRITERION),
    ):
        _equal(document[name], expected, name)
    train, validation, selected_train, selected_validation = training_membership(profile)
    _equal(
        document["datasets"],
        {
            "train": dataset_binding(train, selected_train),
            "validation": dataset_binding(validation, selected_validation),
        },
        "frozen training and validation membership",
    )
    _validate_source(document["source"])
    runtime = _object(document["runtime"], {"python", "numpy", "torch", "device", "torch_threads"})
    if any(
        type(runtime[name]) is not str or not runtime[name] for name in ("python", "numpy", "torch")
    ):
        raise ValueError("runtime dependency versions must be nonempty strings")
    _equal(runtime["device"], "cpu", "training device")
    _equal(runtime["torch_threads"], 1, "training thread count")
    cache = _object(
        document["feature_cache"], {"schema_id", "key_sha256", "features_sha256", "shape", "dtype"}
    )
    _equal(cache["schema_id"], CACHE_SCHEMA_ID, "feature cache schema")
    _equal(
        cache["key_sha256"],
        cache_key(cache_identity(document["datasets"]["train"], document["source"], runtime)),
        "feature cache input identity",
    )
    _digest(cache["features_sha256"], "training feature cache content")
    _equal(cache["shape"], [len(selected_train), 128, 64], "training feature cache shape")
    _equal(cache["dtype"], "float32", "training feature cache dtype")
    return document


def _validate_summary(document: object, training: dict) -> dict:
    from harpy.tasks.sound_match.datasets import CELL_NAMES

    document = dict(
        _object(
            document,
            {
                "schema_id",
                "epochs",
                "selected_epoch",
                "selected_validation_macro_distance",
                "initial_state_sha256",
                "selected_state_sha256",
                "total_optimizer_updates",
                "training_examples",
                "validation_examples",
                "elapsed_seconds",
                "peak_rss_mib",
            },
        )
    )
    _equal(document["schema_id"], SUMMARY_SCHEMA_ID, "training summary schema")
    training_examples = len(training["datasets"]["train"]["selected"])
    validation_members = training["datasets"]["validation"]["selected"]
    _equal(document["training_examples"], training_examples, "training example count")
    _equal(document["validation_examples"], len(validation_members), "validation example count")
    _finite(document["elapsed_seconds"], "training elapsed seconds", 0)
    if document["peak_rss_mib"] is not None:
        _finite(document["peak_rss_mib"], "training peak RSS", 0)
    initial_hash = _digest(document["initial_state_sha256"], "initial state")
    if initial_hash == _digest(document["selected_state_sha256"], "selected state"):
        raise ValueError("selected model must contain actual parameter updates")
    epochs = document["epochs"]
    count = training["configuration"]["epochs"]
    if not isinstance(epochs, list) or len(epochs) != count:
        raise ValueError("training history must retain every configured epoch")
    updates = math.ceil(training_examples / training["configuration"]["batch_size"])
    for index, epoch in enumerate(epochs, 1):
        epoch = _object(epoch, {"epoch", "mean_training_loss", "optimizer_updates", "validation"})
        _equal(epoch["epoch"], index, "epoch order")
        _finite(epoch["mean_training_loss"], "training loss", 0)
        _equal(epoch["optimizer_updates"], updates, "optimizer update count")
        validation = _object(epoch["validation"], {"macro_distance", "cells", "examples"})
        _equal(validation["examples"], len(validation_members), "validation count")
        cells = _object(validation["cells"], set(CELL_NAMES))
        means = []
        for cell in CELL_NAMES:
            values = _object(cells[cell], {"examples", "mean_distance"})
            _equal(
                values["examples"],
                sum(row["cell"] == cell for row in validation_members),
                "validation cell count",
            )
            means.append(_finite(values["mean_distance"], "validation cell distance", 0))
        _equal(validation["macro_distance"], sum(means) / len(means), "validation macro distance")
    selected = min(epochs, key=lambda row: (row["validation"]["macro_distance"], row["epoch"]))
    _equal(document["selected_epoch"], selected["epoch"], "selected epoch")
    _equal(
        document["selected_validation_macro_distance"],
        selected["validation"]["macro_distance"],
        "selected validation score",
    )
    _equal(document["total_optimizer_updates"], updates * count, "total optimizer updates")
    return document


@dataclass(frozen=True, slots=True)
class LoadedInverseArtifact:
    root: Path
    manifest_bytes: bytes
    training_bytes: bytes
    summary_bytes: bytes

    @property
    def manifest(self) -> dict:
        return decode_json_bytes(self.manifest_bytes)

    @property
    def training(self) -> dict:
        return decode_json_bytes(self.training_bytes)

    @property
    def summary(self) -> dict:
        return decode_json_bytes(self.summary_bytes)

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.manifest_bytes).hexdigest()

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        return tuple(self.root / name for name in ("manifest.json", *PAYLOAD_NAMES))

    def verify_unchanged(self) -> None:
        entries = tuple(self.root.iterdir())
        if {entry.name for entry in entries} != {"manifest.json", *PAYLOAD_NAMES} or any(
            entry.is_symlink() or not entry.is_file() for entry in entries
        ):
            raise ValueError("inverse artifact inventory changed after it was loaded")
        if (self.root / "manifest.json").read_bytes() != self.manifest_bytes:
            raise ValueError("inverse artifact changed after it was loaded")
        for record in self.manifest["files"]:
            path = self.root / record["path"]
            if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
                raise ValueError("inverse artifact payload changed after it was loaded")

    def load_model(self, *, device: str = "cpu"):
        if device != "cpu":
            raise ValueError("sound-match inverse models currently require device='cpu'")
        self.verify_unchanged()
        torch = require_torch()
        from harpy.tasks.sound_match.inverse_network import InverseNetwork

        model = InverseNetwork()
        content = (self.root / "model.pt").read_bytes()
        record = next(row for row in self.manifest["files"] if row["path"] == "model.pt")
        if hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError("inverse weights changed while loading")
        try:
            state = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError(
                "inverse weights could not be loaded as a tensor-only state dict"
            ) from error
        expected = model.state_dict()
        if not isinstance(state, Mapping) or set(state) != set(expected):
            raise ValueError("inverse weights have unknown or missing parameter names")
        for name, value in state.items():
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.float32
                or value.shape != expected[name].shape
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"inverse parameter {name!r} has invalid shape, type, or values")
        _equal(
            state_sha256(state),
            self.summary["selected_state_sha256"],
            "selected parameter identity",
        )
        model.load_state_dict(state, strict=True)
        model.eval()
        self.verify_unchanged()
        return model


def load_inverse_artifact(path: Path) -> LoadedInverseArtifact:
    """Validate all identities and hashes without importing optional dependencies."""
    if not isinstance(path, Path) or path.is_symlink() or not path.is_dir():
        raise ValueError("inverse artifact must be a real directory")
    root = path.resolve(strict=True)
    entries = tuple(root.iterdir())
    if {entry.name for entry in entries} != {"manifest.json", *PAYLOAD_NAMES} or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise ValueError("inverse artifact must contain exactly its regular declared payload files")
    encoded = (root / "manifest.json").read_bytes()
    manifest = decode_json_bytes(encoded)
    _object(
        manifest,
        {
            "schema_id",
            "task_id",
            "renderer_id",
            "objective_id",
            "observation_id",
            "decoder_id",
            "architecture_id",
            "supervision_id",
            "weights_format",
            "quality_status",
            "seed",
            "profile",
            "training_manifest_sha256",
            "source_sha256",
            "files",
        },
    )
    for name, expected in {
        "schema_id": ARTIFACT_SCHEMA_ID,
        "task_id": TASK_ID,
        "renderer_id": RENDERER_ID,
        "objective_id": OBJECTIVE_ID,
        "observation_id": OBSERVATION_ID,
        "decoder_id": DECODER_ID,
        "architecture_id": ARCHITECTURE_ID,
        "supervision_id": SUPERVISION_ID,
        "weights_format": "torch-state-dict-weights-only",
        "quality_status": "experimental",
    }.items():
        _equal(manifest[name], expected, name)
    _integer(manifest["seed"], "artifact seed")
    training_configuration(manifest["profile"])
    files = manifest["files"]
    if not isinstance(files, list) or len(files) != len(PAYLOAD_NAMES):
        raise ValueError("inverse artifact inventory is incomplete")
    names = []
    for row in files:
        row = _object(row, {"path", "size_bytes", "sha256"})
        name = row["path"]
        if type(name) is not str or name not in PAYLOAD_NAMES:
            raise ValueError("unknown inverse artifact payload")
        names.append(name)
        _integer(row["size_bytes"], "payload size", 1)
        _digest(row["sha256"], "payload digest")
        payload = root / name
        if payload.stat().st_size != row["size_bytes"] or sha256_file(payload) != row["sha256"]:
            raise ValueError(f"inverse payload {name!r} does not match its immutable identity")
    if names != sorted(PAYLOAD_NAMES):
        raise ValueError("inverse payload inventory must be sorted and unique")
    training_bytes = (root / "training-manifest.json").read_bytes()
    summary_bytes = (root / "training-summary.json").read_bytes()
    _equal(
        manifest["training_manifest_sha256"],
        hashlib.sha256(training_bytes).hexdigest(),
        "training manifest hash",
    )
    training = _validate_training(decode_json_bytes(training_bytes))
    _validate_summary(decode_json_bytes(summary_bytes), training)
    _equal(manifest["seed"], training["seed"], "seed binding")
    _equal(manifest["profile"], training["configuration"]["profile"], "profile binding")
    _equal(manifest["source_sha256"], training["source"]["sha256"], "source binding")
    return LoadedInverseArtifact(root, encoded, training_bytes, summary_bytes)


def publish_inverse_artifact(
    output: Path, state: Mapping, training: dict, summary: dict
) -> LoadedInverseArtifact:
    """Publish a completed artifact last; interrupted directories remain incomplete."""
    torch = require_torch()
    _validate_training(training)
    _validate_summary(summary, training)
    stream = io.BytesIO()
    torch.save(dict(state), stream)
    payloads = {
        "model.pt": stream.getvalue(),
        "training-manifest.json": canonical_json_bytes(training),
        "training-summary.json": canonical_json_bytes(summary),
    }
    for name, content in payloads.items():
        write_new_bytes(output / name, content)
    manifest = {
        "schema_id": ARTIFACT_SCHEMA_ID,
        "task_id": TASK_ID,
        "renderer_id": RENDERER_ID,
        "objective_id": OBJECTIVE_ID,
        "observation_id": OBSERVATION_ID,
        "decoder_id": DECODER_ID,
        "architecture_id": ARCHITECTURE_ID,
        "supervision_id": SUPERVISION_ID,
        "weights_format": "torch-state-dict-weights-only",
        "quality_status": "experimental",
        "seed": training["seed"],
        "profile": training["configuration"]["profile"],
        "training_manifest_sha256": hashlib.sha256(payloads["training-manifest.json"]).hexdigest(),
        "source_sha256": training["source"]["sha256"],
        "files": [
            {
                "path": name,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, content in sorted(payloads.items())
        ],
    }
    write_new_bytes(output / "manifest.json", canonical_json_bytes(manifest))
    return load_inverse_artifact(output)
