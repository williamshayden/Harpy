"""Validated, GUI-independent commands for controlling audio playback."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from enum import StrEnum

from harpy.synth.models import SynthPatch


class AudioCommandKind(StrEnum):
    """Semantic operations accepted by the audio backend."""

    NOTE_ON = "note_on"
    RETUNE = "retune"
    NOTE_OFF = "note_off"
    CLEAR_CAPTURE = "clear_capture"
    REPLACE_PATCH = "replace_patch"
    RESET = "reset"


@dataclass(frozen=True, slots=True)
class AudioCommand:
    """One fully validated operation crossing the audio-thread boundary."""

    kind: AudioCommandKind
    frequency_hz: float | None = None
    patch: SynthPatch | None = None
    generation: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, AudioCommandKind):
            raise ValueError("kind must be an AudioCommandKind")

        expected_payloads = {
            AudioCommandKind.NOTE_ON: (True, False, True),
            AudioCommandKind.RETUNE: (True, False, False),
            AudioCommandKind.NOTE_OFF: (False, False, False),
            AudioCommandKind.CLEAR_CAPTURE: (False, False, True),
            AudioCommandKind.REPLACE_PATCH: (False, True, True),
            AudioCommandKind.RESET: (False, False, True),
        }
        supplied_payloads = (
            self.frequency_hz is not None,
            self.patch is not None,
            self.generation is not None,
        )
        if supplied_payloads != expected_payloads[self.kind]:
            raise ValueError(f"{self.kind.name} has an invalid payload shape")

        if self.frequency_hz is not None:
            if isinstance(self.frequency_hz, bool):
                raise ValueError("frequency_hz must be positive and finite")
            try:
                frequency_hz = float(self.frequency_hz)
            except (TypeError, ValueError) as error:
                raise ValueError("frequency_hz must be positive and finite") from error
            if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
                raise ValueError("frequency_hz must be positive and finite")
            object.__setattr__(self, "frequency_hz", frequency_hz)

        if self.patch is not None and not isinstance(self.patch, SynthPatch):
            raise ValueError("patch must be a SynthPatch")

        if self.generation is not None:
            if isinstance(self.generation, bool):
                raise ValueError("generation must be a nonnegative integer")
            try:
                generation = operator.index(self.generation)
            except TypeError as error:
                raise ValueError("generation must be a nonnegative integer") from error
            if generation < 0:
                raise ValueError("generation must be a nonnegative integer")
            object.__setattr__(self, "generation", generation)
