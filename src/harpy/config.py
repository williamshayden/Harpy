from dataclasses import dataclass, field

from harpy.gui.specs import KeyboardViewSpec
from harpy.pitch import EqualTemperament
from harpy.synth.specs import RenderSpec, SinePatch


@dataclass(frozen=True, slots=True)
class AppConfig:
    tuning: EqualTemperament = field(default_factory=EqualTemperament)
    keyboard: KeyboardViewSpec = field(default_factory=KeyboardViewSpec)
    render: RenderSpec = field(default_factory=RenderSpec)
    patch: SinePatch = field(default_factory=SinePatch)

    def __post_init__(self) -> None:
        sample_rate = self.render.sample_rate_hz
        frame_counts = {
            "attack": self.patch.envelope.attack_frames(sample_rate),
            "decay": self.patch.envelope.decay_frames(sample_rate),
            "release": self.patch.envelope.release_frames(sample_rate),
        }
        for name, frames in frame_counts.items():
            if frames < 1:
                raise ValueError(f"{name} must produce at least one frame")


DEFAULT_CONFIG = AppConfig()
