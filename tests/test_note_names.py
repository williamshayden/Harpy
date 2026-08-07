import pytest

from harpy.note_names import format_note_name, format_pitch_readout, format_tuning_readout
from harpy.pitch import EqualTemperament, MidiNote


@pytest.mark.parametrize(
    ("number", "expected"),
    [(0, "C-2"), (60, "C3"), (69, "A3"), (127, "G8")],
)
def test_ableton_note_names(number: int, expected: str) -> None:
    assert format_note_name(MidiNote(number)) == expected


def test_pitch_readout_contains_note_midi_and_frequency() -> None:
    assert format_pitch_readout(MidiNote(60), EqualTemperament()) == ("C3 · MIDI 60 · 261.626 Hz")


def test_tuning_readout_avoids_octave_naming_ambiguity() -> None:
    assert format_tuning_readout(EqualTemperament()) == ("Concert A reference (MIDI 69) · 440.0 Hz")
