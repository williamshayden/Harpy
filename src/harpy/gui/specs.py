from dataclasses import dataclass, field

from harpy.pitch import MidiNote


@dataclass(frozen=True, slots=True)
class KeyboardViewSpec:
    minimum_note: MidiNote = field(default_factory=lambda: MidiNote(48))
    initial_note: MidiNote = field(default_factory=lambda: MidiNote(60))
    maximum_note: MidiNote = field(default_factory=lambda: MidiNote(72))
    middle_c_octave: int = 3

    def __post_init__(self) -> None:
        if not (self.minimum_note.number <= self.initial_note.number <= self.maximum_note.number):
            raise ValueError("minimum_note <= initial_note <= maximum_note is required")
        if isinstance(self.middle_c_octave, bool) or not isinstance(self.middle_c_octave, int):
            raise TypeError("middle_c_octave must be an integer")
