from __future__ import annotations

import math

import numpy as np

from harpy.synth.envelope import AdsrEnvelope, EnvelopeStage
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
        self._envelope = AdsrEnvelope(patch.envelope, render.sample_rate_hz)
        self._phase_anchor = 0.0
        self._phase_increment = 0.0
        self._phase_frame_offset = 0

    @property
    def is_idle(self) -> bool:
        return self._envelope.is_idle

    @property
    def patch(self) -> SynthPatch:
        return self._patch

    def note_on(self, frequency_hz: float) -> None:
        phase_increment = self._validated_phase_increment(frequency_hz)
        self._phase_anchor = 0.0
        self._phase_increment = phase_increment
        self._phase_frame_offset = 0
        self._envelope.note_on()

    def retune(self, frequency_hz: float) -> None:
        phase_increment = self._validated_phase_increment(frequency_hz)
        if self.is_idle:
            raise RuntimeError("cannot retune an idle synth engine")
        self._reanchor_phase()
        self._phase_increment = phase_increment

    def note_off(self) -> None:
        if not self.is_idle and self._envelope.stage is not EnvelopeStage.RELEASE:
            self._reanchor_phase()
        self._envelope.note_off()

    def replace_patch(self, patch: SynthPatch) -> None:
        if not isinstance(patch, SynthPatch):
            raise ValueError("patch must be a SynthPatch")
        validate_renderable_patch(patch, self._render)
        envelope = AdsrEnvelope(patch.envelope, self._render.sample_rate_hz)
        self._patch = patch
        self._envelope = envelope
        self._phase_anchor = 0.0
        self._phase_increment = 0.0
        self._phase_frame_offset = 0

    def render(self, frame_count: int) -> np.ndarray:
        if isinstance(frame_count, bool) or not isinstance(frame_count, int):
            raise TypeError("frame_count must be an integer")
        if frame_count < 0:
            raise ValueError("frame_count must be non-negative")
        if frame_count == 0:
            return np.empty(0, dtype=np.float32)
        if self.is_idle:
            return np.zeros(frame_count, dtype=np.float32)

        release_frames_remaining = self._envelope.release_frames_remaining
        oscillator_frame_count = (
            min(release_frames_remaining, frame_count)
            if release_frames_remaining is not None
            else frame_count
        )
        frame_positions = np.arange(
            self._phase_frame_offset,
            self._phase_frame_offset + oscillator_frame_count,
            dtype=np.float64,
        )
        phases = self._phase_anchor + self._phase_increment * frame_positions
        oscillator = np.sin(phases)
        envelope = self._envelope.render(frame_count)
        self._phase_frame_offset += oscillator_frame_count
        samples = oscillator * envelope[:oscillator_frame_count] * self._patch.output_gain
        result = samples.astype(np.float32)
        if oscillator_frame_count == frame_count:
            return result
        return np.concatenate(
            (result, np.zeros(frame_count - oscillator_frame_count, dtype=np.float32))
        )

    def reset(self) -> None:
        self._phase_anchor = 0.0
        self._phase_increment = 0.0
        self._phase_frame_offset = 0
        self._envelope.reset()

    def _reanchor_phase(self) -> None:
        self._phase_anchor = math.fmod(
            self._phase_anchor + self._phase_increment * self._phase_frame_offset,
            math.tau,
        )
        self._phase_frame_offset = 0

    def _validated_phase_increment(self, frequency_hz: float) -> float:
        frequency = validate_frequency_hz(frequency_hz, self._render)
        return math.tau * frequency / self._render.sample_rate_hz
