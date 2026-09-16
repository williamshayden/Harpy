"""The finite, canonical parameter space of the sound-match-v1 task."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

PATCH_SCHEMA_ID = "harpy.sound-match.patch/1"
WAVEFORMS = ("sine", "triangle", "saw", "square")
PARAMETER_BOUNDS = MappingProxyType(
    {
        "notes_cents": (4800, 7200),
        "brightness_hz": (200, 12000),
        "attack_ms": (5, 250),
        "decay_ms": (20, 500),
        "sustain_db": (-24, 0),
        "release_ms": (20, 750),
        "gain_db": (-18, 0),
    }
)
SCALAR_FIELDS = tuple(name for name in PARAMETER_BOUNDS if name != "notes_cents")
PATCH_FIELDS = ("notes_cents", "waveform", *SCALAR_FIELDS)


def _bounded_integer(value: object, name: str) -> None:
    minimum, maximum = PARAMETER_BOUNDS[name]
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer within {minimum}..{maximum}")


@dataclass(frozen=True, slots=True)
class Patch:
    """A complete legal patch; sorting notes is the only input canonicalization."""

    notes_cents: tuple[int, ...] = (6000,)
    waveform: str = "sine"
    brightness_hz: int = 6000
    attack_ms: int = 10
    decay_ms: int = 100
    sustain_db: int = -6
    release_ms: int = 200
    gain_db: int = -6

    def __post_init__(self) -> None:
        if not isinstance(self.notes_cents, tuple | list) or len(self.notes_cents) not in (1, 3):
            raise ValueError("notes_cents must contain exactly one or three distinct integers")
        for note in self.notes_cents:
            _bounded_integer(note, "notes_cents")
        if len(set(self.notes_cents)) != len(self.notes_cents):
            raise ValueError("notes_cents must contain distinct integers")
        object.__setattr__(self, "notes_cents", tuple(sorted(self.notes_cents)))
        if type(self.waveform) is not str or self.waveform not in WAVEFORMS:
            raise ValueError(f"waveform must be one of {WAVEFORMS}")
        for name in SCALAR_FIELDS:
            _bounded_integer(getattr(self, name), name)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible patch fields; the enclosing record binds the schema."""
        return {
            name: list(self.notes_cents) if name == "notes_cents" else getattr(self, name)
            for name in PATCH_FIELDS
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Patch:
        if not isinstance(value, Mapping) or set(value) != set(PATCH_FIELDS):
            raise ValueError(f"patch must contain exactly these fields: {PATCH_FIELDS}")
        return cls(**value)


def default_patch(voice_count: int = 1) -> Patch:
    if type(voice_count) is not int or voice_count not in (1, 3):
        raise ValueError("voice_count must be 1 or 3")
    return Patch(notes_cents=(6000,) if voice_count == 1 else (6000, 6400, 6700))
