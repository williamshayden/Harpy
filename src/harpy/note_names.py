from harpy.pitch import EqualTemperament, MidiNote, Pitch

_PITCH_CLASSES = ("C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B")


def format_note_name(note: MidiNote, middle_c_octave: int = 3) -> str:
    octave = note.number // 12 + middle_c_octave - 5
    return f"{_PITCH_CLASSES[note.number % 12]}{octave}"


def format_pitch_readout(
    note: MidiNote,
    tuning: EqualTemperament,
    middle_c_octave: int = 3,
) -> str:
    frequency = tuning.frequency_hz(Pitch.from_midi(note))
    return f"{format_note_name(note, middle_c_octave)} · MIDI {note.number} · {frequency:.3f} Hz"


def format_tuning_readout(tuning: EqualTemperament) -> str:
    return (
        f"Concert A reference (MIDI {tuning.reference_note.number}) · {tuning.reference_hz:.1f} Hz"
    )
