from __future__ import annotations

import math
from dataclasses import dataclass, field


def seconds_to_frames(seconds: float, sample_rate_hz: int) -> int:
    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, int)
        or sample_rate_hz <= 0
    ):
        raise ValueError("sample_rate_hz must be a positive integer")
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("seconds must be finite and non-negative")
    return math.floor(seconds * sample_rate_hz + 0.5)


@dataclass(frozen=True, slots=True)
class EnvelopeSpec:
    attack_seconds: float = 0.001
    decay_seconds: float = 0.600
    sustain_db: float = -6.0
    release_seconds: float = 0.600
    curve: str = "linear_amplitude"

    def __post_init__(self) -> None:
        for name in ("attack_seconds", "decay_seconds", "release_seconds"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        sustain = float(self.sustain_db)
        if not math.isfinite(sustain) or sustain > 0.0:
            raise ValueError("sustain_db must be finite and no greater than 0 dB")
        object.__setattr__(self, "sustain_db", sustain)
        if self.curve != "linear_amplitude":
            raise ValueError("v1 supports only the linear_amplitude envelope curve")

    @property
    def sustain_amplitude(self) -> float:
        return 10.0 ** (self.sustain_db / 20.0)

    def attack_frames(self, sample_rate_hz: int) -> int:
        return seconds_to_frames(self.attack_seconds, sample_rate_hz)

    def decay_frames(self, sample_rate_hz: int) -> int:
        return seconds_to_frames(self.decay_seconds, sample_rate_hz)

    def release_frames(self, sample_rate_hz: int) -> int:
        return seconds_to_frames(self.release_seconds, sample_rate_hz)


@dataclass(frozen=True, slots=True)
class RenderSpec:
    sample_rate_hz: int = 48_000
    block_frames: int = 256
    channels: int = 1
    internal_dtype: str = "float32"

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_rate_hz, bool)
            or not isinstance(self.sample_rate_hz, int)
            or self.sample_rate_hz <= 0
        ):
            raise ValueError("sample_rate_hz must be a positive integer")
        if (
            isinstance(self.block_frames, bool)
            or not isinstance(self.block_frames, int)
            or self.block_frames <= 0
        ):
            raise ValueError("block_frames must be a positive integer")
        if self.channels != 1:
            raise ValueError("the authoritative v1 renderer must be mono")
        if self.internal_dtype != "float32":
            raise ValueError("the authoritative v1 renderer must use float32")


@dataclass(frozen=True, slots=True)
class SinePatch:
    peak_gain_dbfs: float = -12.0
    envelope: EnvelopeSpec = field(default_factory=EnvelopeSpec)

    def __post_init__(self) -> None:
        gain = float(self.peak_gain_dbfs)
        if not math.isfinite(gain) or gain > 0.0:
            raise ValueError("peak_gain_dbfs must be finite and no greater than 0 dBFS")
        object.__setattr__(self, "peak_gain_dbfs", gain)

    @property
    def peak_gain(self) -> float:
        return 10.0 ** (self.peak_gain_dbfs / 20.0)
