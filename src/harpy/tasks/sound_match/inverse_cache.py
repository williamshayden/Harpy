"""A bounded disk feature cache keyed by frozen data, transform code, and runtime."""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harpy._serialization import (
    canonical_json_bytes,
    decode_json_bytes,
    sha256_file,
    write_new_bytes,
)
from harpy.tasks.sound_match.inverse_features import FEATURE_SHAPE, OBSERVATION_ID, log_spectrogram
from harpy.tasks.sound_match.models import EpisodeSpec, _plain
from harpy.tasks.sound_match.renderer import RENDERER_ID, render
from harpy.tasks.sound_match.results import audio_hash

CACHE_SCHEMA_ID = "harpy.sound-match.inverse-feature-cache/1"


def cache_identity(binding: dict, source: dict, runtime: dict) -> dict:
    source_files = {row["path"]: row["sha256"] for row in source["files"]}
    return {
        "schema_id": CACHE_SCHEMA_ID,
        "dataset_manifest_sha256": binding["manifest_sha256"],
        "observation_id": OBSERVATION_ID,
        "renderer_id": RENDERER_ID,
        "feature_source_sha256": source_files["tasks/sound_match/inverse_features.py"],
        "renderer_source_sha256": source_files["tasks/sound_match/renderer.py"],
        "numpy": runtime["numpy"],
        "python": runtime["python"],
        "members": [
            {"episode_id": row["episode_id"], "target_audio_sha256": row["target_audio_sha256"]}
            for row in binding["selected"]
        ],
    }


def cache_key(identity: dict) -> str:
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


@dataclass(frozen=True, slots=True)
class FeatureCache:
    root: Path
    manifest: dict

    @property
    def identity(self) -> dict:
        return {
            "schema_id": CACHE_SCHEMA_ID,
            "key_sha256": self.manifest["key_sha256"],
            "features_sha256": self.manifest["features_sha256"],
            "shape": list(self.manifest["shape"]),
            "dtype": "float32",
        }

    def open(self) -> np.memmap:
        array = np.load(self.root / "features.npy", mmap_mode="r", allow_pickle=False)
        if (
            array.dtype != np.float32
            or list(array.shape) != self.manifest["shape"]
            or array.flags.writeable
        ):
            raise ValueError("cached feature shape, dtype, or read-only state changed")
        return array


def _load_cache(root: Path, identity: dict) -> FeatureCache:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("feature cache must be a real directory")
    entries = tuple(root.iterdir())
    if {path.name for path in entries} != {"manifest.json", "features.npy"} or any(
        path.is_symlink() or not path.is_file() for path in entries
    ):
        raise ValueError("feature cache is incomplete or contains undeclared files")
    document = decode_json_bytes((root / "manifest.json").read_bytes())
    expected = {"schema_id", "identity", "key_sha256", "features_sha256", "shape", "dtype"}
    if set(document) != expected or canonical_json_bytes(
        document["identity"]
    ) != canonical_json_bytes(identity):
        raise ValueError("feature cache identity does not match the requested frozen inputs")
    if (
        document["schema_id"] != CACHE_SCHEMA_ID
        or document["key_sha256"] != cache_key(identity)
        or document["dtype"] != "float32"
    ):
        raise ValueError("unknown feature cache identity")
    shape = [len(identity["members"]), *FEATURE_SHAPE]
    if canonical_json_bytes({"shape": document["shape"]}) != canonical_json_bytes({"shape": shape}):
        raise ValueError("feature cache shape does not match its membership")
    if sha256_file(root / "features.npy") != document["features_sha256"]:
        raise ValueError("feature cache payload changed")
    cache = FeatureCache(root, document)
    array = cache.open()
    for start in range(0, len(array), 64):
        if not np.isfinite(array[start : start + 64]).all():
            raise ValueError("feature cache contains nonfinite features")
    del array
    return cache


def prepare_feature_cache(
    directory: Path, entries: tuple, identity: dict, progress=None
) -> FeatureCache:
    """Materialize one waveform at a time; publish complete NPY bytes without overwrite."""
    if not isinstance(directory, Path):
        raise ValueError("cache directory must be a Path")
    directory.mkdir(parents=True, exist_ok=True)
    root = directory / cache_key(identity)
    if (root / "manifest.json").exists():
        return _load_cache(root, identity)
    if root.is_symlink():
        raise ValueError("feature cache directory cannot be a symbolic link")
    root.mkdir(exist_ok=True)
    temporary = directory / f".features-{uuid.uuid4().hex}.npy"
    shape = (len(entries), *FEATURE_SHAPE)
    try:
        array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=shape)
        for index, entry in enumerate(entries):
            episode = EpisodeSpec.from_dict(entry["episode"])
            audio = render(episode.target_patch)
            if audio_hash(audio) != entry["target_audio_sha256"]:
                raise ValueError("training renderer does not reproduce the frozen target waveform")
            array[index] = log_spectrogram(audio)
            if progress is not None and (index + 1) % 64 == 0:
                progress({"event": "feature-cache", "completed": index + 1, "total": len(entries)})
        array.flush()
        del array
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        content_hash = sha256_file(temporary)
        destination = root / "features.npy"
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if (
                destination.is_symlink()
                or not destination.is_file()
                or sha256_file(destination) != content_hash
            ):
                raise ValueError(
                    "existing feature cache payload has a conflicting identity"
                ) from None
        document = {
            "schema_id": CACHE_SCHEMA_ID,
            "identity": _plain(identity),
            "key_sha256": cache_key(identity),
            "features_sha256": content_hash,
            "shape": list(shape),
            "dtype": "float32",
        }
        try:
            write_new_bytes(root / "manifest.json", canonical_json_bytes(document))
        except FileExistsError:
            if (root / "manifest.json").read_bytes() != canonical_json_bytes(document):
                raise ValueError(
                    "existing feature cache manifest has a conflicting identity"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)
    return _load_cache(root, identity)
