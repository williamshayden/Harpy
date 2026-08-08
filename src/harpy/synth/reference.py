from __future__ import annotations

import math

import numpy as np

from harpy.pitch import EqualTemperament, Pitch
from harpy.synth.envelope import LinearEnvelope
from harpy.synth.specs import RenderSpec, SinePatch


class SineVoice:
    def __init__(
        self,
        tuning: EqualTemperament,
        render_spec: RenderSpec,
        patch: SinePatch,
    ) -> None:
        self.tuning = tuning
        self.render_spec = render_spec
        self.patch = patch
        self._envelope = LinearEnvelope(patch.envelope, render_spec.sample_rate_hz)
        self._phase = 0.0
        self._phase_increment = 0.0

    @property
    def is_idle(self) -> bool:
        return self._envelope.is_idle

    def note_on(self, pitch: Pitch) -> None:
        frequency_hz = self.tuning.frequency_hz(pitch)
        nyquist_hz = self.render_spec.sample_rate_hz / 2.0
        if not math.isfinite(frequency_hz) or frequency_hz <= 0.0 or frequency_hz >= nyquist_hz:
            raise ValueError("frequency must be positive, finite, and below Nyquist")
        self._phase = 0.0
        self._phase_increment = math.tau * frequency_hz / self.render_spec.sample_rate_hz
        self._envelope.note_on()

    def note_off(self) -> None:
        self._envelope.note_off()

    def reset(self) -> None:
        self._phase = 0.0
        self._phase_increment = 0.0
        self._envelope.reset()

    def render_block(self, frame_count: int) -> np.ndarray:
        if isinstance(frame_count, bool) or not isinstance(frame_count, int):
            raise TypeError("frame_count must be an integer")
        if frame_count < 0:
            raise ValueError("frame_count must be non-negative")
        if frame_count == 0:
            return np.empty(0, dtype=np.float32)
        if self._envelope.is_idle:
            return np.zeros(frame_count, dtype=np.float32)
        offsets = np.arange(frame_count, dtype=np.float64)
        phases = self._phase + self._phase_increment * offsets
        oscillator = np.sin(phases)
        envelope = self._envelope.render(frame_count)
        self._phase = math.fmod(
            self._phase + self._phase_increment * frame_count,
            math.tau,
        )
        samples = oscillator * envelope * self.patch.peak_gain
        return samples.astype(np.float32)
