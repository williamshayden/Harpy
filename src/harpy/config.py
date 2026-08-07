from dataclasses import dataclass, field

from harpy.gui.specs import KeyboardViewSpec
from harpy.pitch import EqualTemperament, Pitch
from harpy.synth.specs import RenderSpec, SinePatch


@dataclass(frozen=True, slots=True)
class AppConfig:
    tuning: EqualTemperament = field(default_factory=EqualTemperament)
    keyboard: KeyboardViewSpec = field(default_factory=KeyboardViewSpec)
    render: RenderSpec = field(default_factory=RenderSpec)
    patch: SinePatch = field(default_factory=SinePatch)

    def __post_init__(self) -> None:
        sample_rate = self.render.sample_rate_hz
        pitch_bounds = (
            ("minimum_note", self.keyboard.minimum_note),
            ("maximum_note", self.keyboard.maximum_note),
        )
        frequencies: list[tuple[str, int, float]] = []
        for name, note in pitch_bounds:
            try:
                frequency_hz = self.tuning.frequency_hz(Pitch.from_midi(note))
            except ValueError as error:
                raise ValueError(
                    f"{name} (MIDI {note.number}) must map to a finite, positive frequency"
                ) from error
            frequencies.append((name, note.number, frequency_hz))

        nyquist_hz = sample_rate / 2.0
        for name, note_number, frequency_hz in frequencies:
            if frequency_hz >= nyquist_hz:
                raise ValueError(
                    f"{name} (MIDI {note_number}) maps to {frequency_hz:g} Hz; "
                    f"it must be below Nyquist ({nyquist_hz:g} Hz)"
                )

        frame_counts = {
            "attack": self.patch.envelope.attack_frames(sample_rate),
            "decay": self.patch.envelope.decay_frames(sample_rate),
            "release": self.patch.envelope.release_frames(sample_rate),
        }
        for name, frames in frame_counts.items():
            if frames < 1:
                raise ValueError(f"{name} must produce at least one frame")


DEFAULT_CONFIG = AppConfig()
