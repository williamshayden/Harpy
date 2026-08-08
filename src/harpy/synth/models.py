from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum


class OscillatorType(StrEnum):
    SINE = "sine"


@dataclass(frozen=True, slots=True)
class OscillatorConfig:
    type: OscillatorType = OscillatorType.SINE

    def __post_init__(self) -> None:
        if not isinstance(self.type, OscillatorType):
            raise ValueError("type must be an OscillatorType")


@dataclass(frozen=True, slots=True)
class EnvelopeConfig:
    attack_seconds: float = 0.001
    decay_seconds: float = 0.600
    sustain_db: float = -6.0
    release_seconds: float = 0.600
    curve: str = "linear_amplitude"

    def __post_init__(self) -> None:
        for name in ("attack_seconds", "decay_seconds", "release_seconds"):
            value = _finite_float(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)

        sustain = _finite_float(self.sustain_db, "sustain_db")
        if sustain > 0.0:
            raise ValueError("sustain_db must be finite and no greater than 0 dB")
        object.__setattr__(self, "sustain_db", sustain)

        if self.curve != "linear_amplitude":
            raise ValueError("curve must be linear_amplitude")

    @property
    def sustain_amplitude(self) -> float:
        return 10.0 ** (self.sustain_db / 20.0)


@dataclass(frozen=True, slots=True)
class SynthPatch:
    oscillator: OscillatorConfig = field(default_factory=OscillatorConfig)
    envelope: EnvelopeConfig = field(default_factory=EnvelopeConfig)
    output_gain_dbfs: float = -12.0

    def __post_init__(self) -> None:
        if not isinstance(self.oscillator, OscillatorConfig):
            raise ValueError("oscillator must be an OscillatorConfig")
        if not isinstance(self.envelope, EnvelopeConfig):
            raise ValueError("envelope must be an EnvelopeConfig")

        gain = _finite_float(self.output_gain_dbfs, "output_gain_dbfs")
        if gain > 0.0:
            raise ValueError("output_gain_dbfs must be finite and no greater than 0 dBFS")
        object.__setattr__(self, "output_gain_dbfs", gain)

    @property
    def output_gain(self) -> float:
        return 10.0 ** (self.output_gain_dbfs / 20.0)


@dataclass(frozen=True, slots=True)
class RenderConfig:
    sample_rate_hz: int = 48_000
    block_frames: int = 256
    channels: int = 1
    internal_dtype: str = "float32"

    def __post_init__(self) -> None:
        _positive_integer(self.sample_rate_hz, "sample_rate_hz")
        _positive_integer(self.block_frames, "block_frames")
        if (
            isinstance(self.channels, bool)
            or not isinstance(self.channels, int)
            or self.channels != 1
        ):
            raise ValueError("channels must be the authoritative mono count of 1")
        if self.internal_dtype != "float32":
            raise ValueError("internal_dtype must be float32")


def seconds_to_frames(seconds: float, sample_rate_hz: int) -> int:
    _positive_integer(sample_rate_hz, "sample_rate_hz")
    seconds_value = _finite_float(seconds, "seconds")
    if seconds_value < 0.0:
        raise ValueError("seconds must be finite and non-negative")
    return math.floor(seconds_value * sample_rate_hz + 0.5)


def validate_renderable_patch(patch: SynthPatch, render: RenderConfig) -> None:
    for segment in ("attack", "decay", "release"):
        seconds = getattr(patch.envelope, f"{segment}_seconds")
        if seconds_to_frames(seconds, render.sample_rate_hz) < 1:
            raise ValueError(f"{segment} segment must contain at least one frame")


def _finite_float(value: object, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _positive_integer(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
