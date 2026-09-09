"""Small public inputs for ordinary, explicitly identified experiments."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from harpy.envs.models import (
    SOURCE_MAX_CENTS,
    SOURCE_MIN_CENTS,
    TARGET_NOTE_COUNT,
    ObservationMode,
    PitchAction,
)


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _name(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(c in value for c in "\r\n"):
        raise ValueError(f"{name} must be a nonempty single-line string")
    return value


def _finite(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("metadata must contain only finite JSON values")


def _freeze(value: object) -> object:
    value = _plain(value)
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _object(value: object, fields: set[str] | None = None) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or (fields is not None and set(value) != fields):
        raise ValueError("document has missing or unexpected fields")
    _plain(value)
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _json_loads(content: bytes) -> object:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    return json.loads(content, object_pairs_hook=pairs, parse_constant=invalid)


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """An explicit task and nuisance seed; this truth is never passed to actors."""

    id: str
    source_pitch_cents: int
    target_note_index: int
    nuisance_seed: int = 0
    partition: str = "custom"

    def __post_init__(self) -> None:
        _name(self.id, "episode id")
        _name(self.partition, "partition")
        source = _integer(self.source_pitch_cents, "source_pitch_cents")
        target = _integer(self.target_note_index, "target_note_index")
        _integer(self.nuisance_seed, "nuisance_seed")
        if not SOURCE_MIN_CENTS <= source <= SOURCE_MAX_CENTS:
            raise ValueError("source_pitch_cents must be within 4800..7200")
        if target >= TARGET_NOTE_COUNT:
            raise ValueError("target_note_index must be within 0..24")

    def to_document(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_document(cls, value: object) -> EpisodeSpec:
        return cls(**_object(value, set(cls.__dataclass_fields__)))


@dataclass(frozen=True, slots=True)
class Decision:
    """One action, optional current-view pitch estimate, and actual inference count.

    A committed controller reports zero inferences after its initial estimate.
    Custom ``act``-only policies are counted as one policy evaluation per call.
    """

    action: PitchAction
    estimated_candidate_cents: float | None = None
    inference_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.action, PitchAction):
            raise ValueError("action must be a PitchAction")
        _integer(self.inference_count, "inference_count")
        if self.estimated_candidate_cents is not None:
            estimate = _finite(self.estimated_candidate_cents, "estimated_candidate_cents")
            if self.inference_count == 0:
                raise ValueError("a new estimate requires an inference")
            object.__setattr__(self, "estimated_candidate_cents", estimate)


@dataclass(frozen=True, slots=True)
class ActorSpec:
    """A named condition with a fresh episode-controller factory and explicit identity."""

    name: str
    factory: Callable[[], object] = field(repr=False, compare=False)
    observation_mode: ObservationMode
    estimator: str
    decoder: str
    controller: str
    artifact_paths: tuple[Path, ...] = ()
    seed: int | None = None
    device: str = "cpu"
    artifact_provenance: Mapping[str, object] = field(default_factory=dict)
    _verify_artifact: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("name", "estimator", "decoder", "controller"):
            _name(getattr(self, name), name)
        if not callable(self.factory):
            raise ValueError("factory must create a fresh episode actor")
        if not isinstance(self.observation_mode, ObservationMode):
            raise ValueError("observation_mode must be an ObservationMode")
        paths = tuple(self.artifact_paths)
        if not all(isinstance(path, Path) for path in paths):
            raise ValueError("artifact_paths must contain Path values")
        object.__setattr__(self, "artifact_paths", paths)
        if self.seed is not None:
            _integer(self.seed, "seed")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        provenance = _object(self.artifact_provenance)
        if not provenance:
            provenance = {"status": "unknown", "reason": "Actor provenance was not supplied."}
        object.__setattr__(self, "artifact_provenance", _freeze(provenance))
        if self._verify_artifact is not None and not callable(self._verify_artifact):
            raise ValueError("artifact verification must be callable")

    def to_document(self) -> dict[str, object]:
        return {
            "name": self.name,
            "observation_mode": self.observation_mode.value,
            "estimator": self.estimator,
            "decoder": self.decoder,
            "controller": self.controller,
            "seed": self.seed,
            "device": self.device,
            "artifact_provenance": _plain(self.artifact_provenance),
        }


__all__ = ["ActorSpec", "Decision", "EpisodeSpec"]
