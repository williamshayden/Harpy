from __future__ import annotations

from dataclasses import dataclass

from harpy.tuning import Tuning


@dataclass(frozen=True, slots=True)
class WorkbenchSpec:
    minimum_frequency_hz: float
    center_frequency_hz: float
    maximum_frequency_hz: float

    @classmethod
    def from_tuning(cls, tuning: Tuning) -> WorkbenchSpec:
        return cls(
            minimum_frequency_hz=tuning.frequency_hz_for_midi_coordinate(48.0),
            center_frequency_hz=tuning.frequency_hz_for_midi_coordinate(60.0),
            maximum_frequency_hz=tuning.frequency_hz_for_midi_coordinate(72.0),
        )
