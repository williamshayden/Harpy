import math
from dataclasses import dataclass

_NOTE_NAMES = ("C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B")


@dataclass(frozen=True, slots=True)
class NoteReading:
    name: str
    cents: float


@dataclass(frozen=True, slots=True)
class Tuning:
    reference_hz: float = 440.0
    reference_midi_coordinate: float = 69.0
    middle_c_octave: int = 3

    def __post_init__(self) -> None:
        reference_hz = _finite_float(self.reference_hz, "reference_hz")
        if reference_hz <= 0.0:
            raise ValueError("reference_hz must be positive and finite")
        reference_coordinate = _finite_float(
            self.reference_midi_coordinate,
            "reference_midi_coordinate",
        )
        if isinstance(self.middle_c_octave, bool) or not isinstance(self.middle_c_octave, int):
            raise TypeError("middle_c_octave must be an integer")
        object.__setattr__(self, "reference_hz", reference_hz)
        object.__setattr__(self, "reference_midi_coordinate", reference_coordinate)

    def frequency_hz_for_midi_coordinate(self, coordinate: float) -> float:
        coordinate = _finite_float(coordinate, "coordinate")
        exponent = (coordinate - self.reference_midi_coordinate) / 12.0
        try:
            frequency_hz = self.reference_hz * (2.0**exponent)
        except OverflowError as error:
            message = "coordinate maps outside the finite positive frequency range"
            raise ValueError(message) from error
        if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
            raise ValueError("coordinate maps outside the finite positive frequency range")
        return frequency_hz

    def midi_coordinate_for_frequency_hz(self, frequency_hz: float) -> float:
        frequency_hz = _positive_finite_hz(frequency_hz)
        coordinate = self.reference_midi_coordinate + 12.0 * (
            math.log2(frequency_hz) - math.log2(self.reference_hz)
        )
        if not math.isfinite(coordinate):
            raise ValueError("frequency_hz maps outside the finite coordinate range")
        return coordinate

    def describe_frequency(self, frequency_hz: float) -> NoteReading:
        coordinate = self.midi_coordinate_for_frequency_hz(frequency_hz)
        nearest_integer = math.floor(coordinate + 0.5)
        octave = nearest_integer // 12 + self.middle_c_octave - 5
        return NoteReading(
            name=f"{_NOTE_NAMES[nearest_integer % 12]}{octave}",
            cents=100.0 * (coordinate - nearest_integer),
        )


def _finite_float(value: float, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _positive_finite_hz(value: float) -> float:
    frequency_hz = _finite_float(value, "frequency_hz")
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive and finite")
    return frequency_hz
