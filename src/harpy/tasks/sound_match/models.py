"""Public inputs and owned observations for bounded sound matching."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np

from harpy.tasks.sound_match.patch import PATCH_FIELDS, Patch, default_patch

TASK_ID = "sound-match-v1"
INFORMATION_TRACK = "audio-only-metered"


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, name: str, minimum: float | None = None) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        numeric = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be representable as a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return numeric


def _name(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(c in value for c in "\r\n\x00"):
        raise ValueError(f"{name} must be a nonempty single-line string")
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("metadata keys must be strings")
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


def _object(value: object, fields: set[str] | None = None) -> Mapping:
    if not isinstance(value, Mapping) or (fields is not None and set(value) != fields):
        raise ValueError("document has missing or unexpected fields")
    _plain(value)
    return value


def _patch(value: object) -> Patch:
    # Revalidate even an existing frozen dataclass before trusting external actors.
    return Patch.from_dict(value.to_dict() if isinstance(value, Patch) else value)


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """Evaluator inputs. Never pass this object to an actor factory or decision."""

    episode_id: str
    target_patch: Patch
    initial_patch: Patch = field(default_factory=default_patch)
    editable_fields: tuple[str, ...] = PATCH_FIELDS
    render_budget: int = 32
    decision_budget: int = 64

    def __post_init__(self) -> None:
        _name(self.episode_id, "episode_id")
        if not isinstance(self.target_patch, Patch) or not isinstance(self.initial_patch, Patch):
            raise ValueError("target_patch and initial_patch must be Patch values")
        target, initial = _patch(self.target_patch), _patch(self.initial_patch)
        if len(target.notes_cents) != len(initial.notes_cents):
            raise ValueError("target and initial patch must have the same note count")
        if not isinstance(self.editable_fields, (tuple, list)) or any(
            not isinstance(name, str) for name in self.editable_fields
        ):
            raise ValueError("editable_fields must be an array of field names")
        fields = tuple(self.editable_fields)
        if len(set(fields)) != len(fields) or any(name not in PATCH_FIELDS for name in fields):
            raise ValueError("editable_fields must contain distinct known patch fields")
        fields = tuple(name for name in PATCH_FIELDS if name in fields)
        for name in PATCH_FIELDS:
            if name not in fields and getattr(target, name) != getattr(initial, name):
                raise ValueError(f"target changes locked field {name}")
        _integer(self.render_budget, "render_budget", 1)
        _integer(self.decision_budget, "decision_budget", 1)
        object.__setattr__(self, "target_patch", target)
        object.__setattr__(self, "initial_patch", initial)
        object.__setattr__(self, "editable_fields", fields)

    @property
    def voice_count(self) -> int:
        return len(self.initial_patch.notes_cents)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "target_patch": self.target_patch.to_dict(),
            "initial_patch": self.initial_patch.to_dict(),
            "editable_fields": list(self.editable_fields),
            "render_budget": self.render_budget,
            "decision_budget": self.decision_budget,
        }

    @classmethod
    def from_dict(cls, value: object) -> EpisodeSpec:
        document = dict(_object(value, set(cls.__dataclass_fields__)))
        for name in ("target_patch", "initial_patch"):
            document[name] = _patch(document[name])
        if not isinstance(document["editable_fields"], (list, tuple)):
            raise ValueError("editable_fields must be an array")
        return cls(**document)


@dataclass(frozen=True, slots=True)
class Propose:
    """Propose a full patch. Malformed payloads are rejected by the runner.

    ``model_calls`` is an optional declaration of calls made for this decision.
    Omitting it records unknown cost, rather than an invented zero.
    """

    patch: Patch | Mapping
    model_calls: int | None = None

    def __post_init__(self) -> None:
        if self.model_calls is not None:
            _integer(self.model_calls, "model_calls")


@dataclass(frozen=True, slots=True)
class Submit:
    """Submit the already rendered current candidate without another evaluation."""

    model_calls: int | None = None

    def __post_init__(self) -> None:
        if self.model_calls is not None:
            _integer(self.model_calls, "model_calls")


@dataclass(frozen=True, slots=True)
class Observation:
    """Public state; target/candidate arrays and patch are owned by this observation."""

    episode_id: str
    target_audio: np.ndarray
    candidate_audio: np.ndarray
    patch: Patch
    editable_fields: tuple[str, ...]
    render_budget_remaining: int
    decision_budget_remaining: int
    last_action_status: Mapping[str, object]
    allowed_values: Mapping[str, object]
    sample_rate_hz: int = 48_000
    frame_count: int = 96_000
    note_off_frame: int = 48_000


@dataclass(frozen=True, slots=True)
class ActorSpec:
    """Fresh episode-controller factory with declared portable artifact identity."""

    name: str
    factory: Callable[[], object] = field(repr=False, compare=False)
    metadata: Mapping[str, object] = field(default_factory=dict)
    artifact_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        _name(self.name, "actor name")
        if not callable(self.factory):
            raise ValueError("factory must be callable")
        metadata = _object(self.metadata)
        if not metadata:
            metadata = {"status": "unknown", "reason": "Actor provenance was not supplied."}
        object.__setattr__(self, "metadata", _freeze(metadata))
        paths = tuple(self.artifact_paths)
        if any(not isinstance(path, Path) for path in paths):
            raise ValueError("artifact_paths must contain Path objects")
        object.__setattr__(self, "artifact_paths", paths)


def proposal_error(patch: Patch, episode: EpisodeSpec, evaluations: int) -> str | None:
    """Shared legality rule for runtime decisions and strict ledger readout."""
    if len(patch.notes_cents) != episode.voice_count:
        return "voice_count_changed"
    if any(
        getattr(patch, name) != getattr(episode.initial_patch, name)
        for name in PATCH_FIELDS
        if name not in episode.editable_fields
    ):
        return "locked_field_changed"
    if evaluations >= episode.render_budget:
        return "render_budget_exhausted"
    return None
