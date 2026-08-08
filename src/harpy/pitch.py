from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class MidiNote:
    number: int

    def __post_init__(self) -> None:
        if isinstance(self.number, bool) or not isinstance(self.number, int):
            raise TypeError("MIDI note number must be an integer")
        if not 0 <= self.number <= 127:
            raise ValueError("MIDI note number must be between 0 and 127")


@dataclass(frozen=True, slots=True)
class Pitch:
    cents_from_midi_zero: float

    def __post_init__(self) -> None:
        value = float(self.cents_from_midi_zero)
        if not math.isfinite(value):
            raise ValueError("pitch cents must be finite")
        object.__setattr__(self, "cents_from_midi_zero", value)

    @classmethod
    def from_midi(
        cls,
        note: MidiNote | int,
        detune_cents: float = 0.0,
    ) -> Pitch:
        midi_note = note if isinstance(note, MidiNote) else MidiNote(note)
        detune = float(detune_cents)
        if not math.isfinite(detune):
            raise ValueError("detune_cents must be finite")
        return cls(100.0 * midi_note.number + detune)


@dataclass(frozen=True, slots=True)
class EqualTemperament:
    reference_note: MidiNote = field(default_factory=lambda: MidiNote(69))
    reference_hz: float = 440.0

    def __post_init__(self) -> None:
        if not isinstance(self.reference_note, MidiNote):
            raise TypeError("reference_note must be a MidiNote")
        value = float(self.reference_hz)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("reference_hz must be positive and finite")
        object.__setattr__(self, "reference_hz", value)

    def frequency_hz(self, pitch: Pitch) -> float:
        exponent = (pitch.cents_from_midi_zero - 100.0 * self.reference_note.number) / 1200.0
        try:
            frequency = self.reference_hz * (2.0**exponent)
        except OverflowError as error:
            raise ValueError("pitch maps outside the finite positive frequency range") from error
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("pitch maps outside the finite positive frequency range")
        return frequency
