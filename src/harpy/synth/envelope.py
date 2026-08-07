from __future__ import annotations

from enum import StrEnum

import numpy as np

from harpy.synth.specs import EnvelopeSpec


class EnvelopeStage(StrEnum):
    IDLE = "idle"
    ATTACK = "attack"
    DECAY = "decay"
    SUSTAIN = "sustain"
    RELEASE = "release"


class LinearEnvelope:
    def __init__(self, spec: EnvelopeSpec, sample_rate_hz: int) -> None:
        self.spec = spec
        self.sample_rate_hz = sample_rate_hz
        self._attack_frames = spec.attack_frames(sample_rate_hz)
        self._decay_frames = spec.decay_frames(sample_rate_hz)
        self._release_frames = spec.release_frames(sample_rate_hz)
        if min(self._attack_frames, self._decay_frames, self._release_frames) < 1:
            raise ValueError("every envelope stage must contain at least one frame")
        self.reset()

    @property
    def level(self) -> float:
        return self._level

    @property
    def stage(self) -> EnvelopeStage:
        return self._stage

    @property
    def is_idle(self) -> bool:
        return self._stage is EnvelopeStage.IDLE

    def reset(self) -> None:
        self._stage = EnvelopeStage.IDLE
        self._level = 0.0
        self._target = 0.0
        self._step = 0.0
        self._remaining = 0

    def note_on(self) -> None:
        self._level = 0.0
        self._begin_segment(EnvelopeStage.ATTACK, 1.0, self._attack_frames)

    def note_off(self) -> None:
        if self._stage in (EnvelopeStage.IDLE, EnvelopeStage.RELEASE):
            return
        if self._level <= 0.0:
            self.reset()
            return
        self._begin_segment(EnvelopeStage.RELEASE, 0.0, self._release_frames)

    def render(self, frame_count: int) -> np.ndarray:
        if isinstance(frame_count, bool) or not isinstance(frame_count, int):
            raise TypeError("frame_count must be an integer")
        if frame_count < 0:
            raise ValueError("frame_count must be non-negative")
        output = np.empty(frame_count, dtype=np.float64)
        for index in range(frame_count):
            if self._stage is EnvelopeStage.IDLE:
                output[index] = 0.0
                continue
            if self._stage is EnvelopeStage.SUSTAIN:
                output[index] = self._level
                continue
            self._level += self._step
            self._remaining -= 1
            if self._remaining == 0:
                self._level = self._target
            output[index] = self._level
            if self._remaining == 0:
                self._finish_segment()
        return output

    def _begin_segment(
        self,
        stage: EnvelopeStage,
        target: float,
        frames: int,
    ) -> None:
        self._stage = stage
        self._target = target
        self._remaining = frames
        self._step = (target - self._level) / frames

    def _finish_segment(self) -> None:
        if self._stage is EnvelopeStage.ATTACK:
            self._begin_segment(
                EnvelopeStage.DECAY,
                self.spec.sustain_amplitude,
                self._decay_frames,
            )
        elif self._stage is EnvelopeStage.DECAY:
            self._stage = EnvelopeStage.SUSTAIN
            self._level = self.spec.sustain_amplitude
            self._remaining = 0
            self._step = 0.0
        elif self._stage is EnvelopeStage.RELEASE:
            self.reset()
