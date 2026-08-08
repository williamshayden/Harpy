import pytest

from harpy.tuning import NoteReading, Tuning


def test_default_tuning_round_trips_reference_frequency() -> None:
    tuning = Tuning()
    assert tuning.frequency_hz_for_midi_coordinate(69.0) == 440.0
    assert tuning.midi_coordinate_for_frequency_hz(440.0) == 69.0


@pytest.mark.parametrize(
    ("frequency_hz", "expected"),
    [
        (261.6255653005986, NoteReading("C3", 0.0)),
        (440.0, NoteReading("A3", 0.0)),
        (440.0 * 2.0 ** (25.0 / 1200.0), NoteReading("A3", 25.0)),
    ],
)
def test_frequency_reading_uses_ableton_octaves(
    frequency_hz: float,
    expected: NoteReading,
) -> None:
    reading = Tuning().describe_frequency(frequency_hz)
    assert reading.name == expected.name
    assert reading.cents == pytest.approx(expected.cents, abs=1e-9)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_frequency_conversion_rejects_nonphysical_hertz(value: float) -> None:
    with pytest.raises(ValueError):
        Tuning().midi_coordinate_for_frequency_hz(value)
