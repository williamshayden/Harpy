from __future__ import annotations

import math

import numpy as np

from harpy.synth.envelope import LinearEnvelope
from harpy.synth.models import (
    RenderConfig,
    SynthPatch,
    validate_frequency_hz,
    validate_renderable_patch,
)


class SynthEngine:
    def __init__(self, render: RenderConfig, patch: SynthPatch) -> None:
        if not isinstance(render, RenderConfig):
            raise ValueError("render must be a RenderConfig")
        if not isinstance(patch, SynthPatch):
            raise ValueError("patch must be a SynthPatch")
        validate_renderable_patch(patch, render)
        self._render = render
        self._patch = patch
        self._envelope = LinearEnvelope(patch.envelope, render.sample_rate_hz)
        self._phase = 0.0
        self._phase_increment = 0.0

    @property
    def is_idle(self) -> bool:
        return self._envelope.is_idle

    @property
    def patch(self) -> SynthPatch:
        return self._patch

    def note_on(self, frequency_hz: float) -> None:
        phase_increment = self._validated_phase_increment(frequency_hz)
        self._phase = 0.0
        self._phase_increment = phase_increment
        self._envelope.note_on()

    def retune(self, frequency_hz: float) -> None:
        phase_increment = self._validated_phase_increment(frequency_hz)
        if self.is_idle:
            raise RuntimeError("cannot retune an idle synth engine")
        self._phase_increment = phase_increment

    def note_off(self) -> None:
        self._envelope.note_off()

    def replace_patch(self, patch: SynthPatch) -> None:
        if not isinstance(patch, SynthPatch):
            raise ValueError("patch must be a SynthPatch")
        validate_renderable_patch(patch, self._render)
        envelope = LinearEnvelope(patch.envelope, self._render.sample_rate_hz)
        self._patch = patch
        self._envelope = envelope
        self._phase = 0.0
        self._phase_increment = 0.0

    def render(self, frame_count: int) -> np.ndarray:
        if isinstance(frame_count, bool) or not isinstance(frame_count, int):
            raise TypeError("frame_count must be an integer")
        if frame_count < 0:
            raise ValueError("frame_count must be non-negative")
        if frame_count == 0:
            return np.empty(0, dtype=np.float32)
        if self.is_idle:
            return np.zeros(frame_count, dtype=np.float32)

        offsets = np.arange(frame_count, dtype=np.float64)
        phases = self._phase + self._phase_increment * offsets
        oscillator = np.sin(phases)
        envelope = self._envelope.render(frame_count)
        self._phase = math.fmod(
            self._phase + self._phase_increment * frame_count,
            math.tau,
        )
        samples = oscillator * envelope * self._patch.output_gain
        return samples.astype(np.float32)

    def reset(self) -> None:
        self._phase = 0.0
        self._phase_increment = 0.0
        self._envelope.reset()

    def _validated_phase_increment(self, frequency_hz: float) -> float:
        frequency = validate_frequency_hz(frequency_hz, self._render)
        return math.tau * frequency / self._render.sample_rate_hz
