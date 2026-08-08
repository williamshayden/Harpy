import math

import pytest

from harpy.pitch import EqualTemperament, MidiNote, Pitch


@pytest.mark.parametrize("number", [-1, 128, True, 60.0])
def test_midi_note_rejects_invalid_numbers(number: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        MidiNote(number)  # type: ignore[arg-type]


def test_default_tuning_maps_reference_note_exactly() -> None:
    tuning = EqualTemperament()
    assert tuning.frequency_hz(Pitch.from_midi(MidiNote(69))) == 440.0


def test_default_tuning_maps_middle_c() -> None:
    frequency = EqualTemperament().frequency_hz(Pitch.from_midi(60))
    assert frequency == pytest.approx(261.6255653005986)


def test_reference_frequency_is_configurable() -> None:
    tuning = EqualTemperament(reference_hz=442.0)
    assert tuning.frequency_hz(Pitch.from_midi(69)) == 442.0


def test_continuous_cents_preserve_interval_ratios() -> None:
    tuning = EqualTemperament()
    base = tuning.frequency_hz(Pitch.from_midi(60))
    semitone = tuning.frequency_hz(Pitch.from_midi(60, detune_cents=100.0))
    octave = tuning.frequency_hz(Pitch.from_midi(60, detune_cents=1200.0))
    assert semitone / base == pytest.approx(2 ** (1 / 12))
    assert octave / base == pytest.approx(2.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_pitch_rejects_non_finite_cents(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        Pitch(value)


@pytest.mark.parametrize("reference_hz", [0.0, -440.0, math.nan, math.inf])
def test_tuning_rejects_invalid_reference(reference_hz: float) -> None:
    with pytest.raises(ValueError, match="reference_hz"):
        EqualTemperament(reference_hz=reference_hz)


def test_tuning_requires_a_midi_note_reference() -> None:
    with pytest.raises(TypeError, match="reference_note"):
        EqualTemperament(reference_note=69)  # type: ignore[arg-type]
