from __future__ import annotations

from enum import StrEnum

import numpy as np

from harpy.synth.curves import evaluate_quadratic_segment
from harpy.synth.models import EnvelopeConfig, seconds_to_frames


class EnvelopeStage(StrEnum):
    IDLE = "idle"
    ATTACK = "attack"
    DECAY = "decay"
    SUSTAIN = "sustain"
    RELEASE = "release"


class AdsrEnvelope:
    def __init__(self, spec: EnvelopeConfig, sample_rate_hz: int) -> None:
        self.spec = spec
        self.sample_rate_hz = sample_rate_hz
        self._attack_frames = seconds_to_frames(spec.attack_seconds, sample_rate_hz)
        self._decay_frames = seconds_to_frames(spec.decay_seconds, sample_rate_hz)
        self._release_frames = seconds_to_frames(spec.release_seconds, sample_rate_hz)
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

    @property
    def release_frames_remaining(self) -> int | None:
        if self._stage is not EnvelopeStage.RELEASE:
            return None
        return self._segment_frames - self._emitted

    def reset(self) -> None:
        self._stage = EnvelopeStage.IDLE
        self._level = 0.0
        self._segment_start = 0.0
        self._target = 0.0
        self._segment_frames = 0
        self._emitted = 0
        self._curvature = 0.0
        self._linear_step = 0.0

    def note_on(self) -> None:
        self._level = 0.0
        self._begin_segment(
            EnvelopeStage.ATTACK,
            1.0,
            self._attack_frames,
            self.spec.attack_curve,
        )

    def note_off(self) -> None:
        if self._stage in (EnvelopeStage.IDLE, EnvelopeStage.RELEASE):
            return
        self._begin_segment(
            EnvelopeStage.RELEASE,
            0.0,
            self._release_frames,
            self.spec.release_curve,
        )

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
            self._emitted += 1
            if self._curvature == 0.0:
                self._level += self._linear_step
            else:
                self._level = evaluate_quadratic_segment(
                    self._segment_start,
                    self._target,
                    self._curvature,
                    self._emitted / self._segment_frames,
                )
            if self._emitted == self._segment_frames:
                self._level = self._target
            output[index] = self._level
            if self._emitted == self._segment_frames:
                self._finish_segment()
        return output

    def _begin_segment(
        self,
        stage: EnvelopeStage,
        target: float,
        frames: int,
        curvature: float,
    ) -> None:
        self._stage = stage
        self._segment_start = self._level
        self._target = target
        self._segment_frames = frames
        self._emitted = 0
        self._curvature = curvature
        self._linear_step = (target - self._level) / frames

    def _finish_segment(self) -> None:
        if self._stage is EnvelopeStage.ATTACK:
            self._begin_segment(
                EnvelopeStage.DECAY,
                self.spec.sustain_amplitude,
                self._decay_frames,
                self.spec.decay_curve,
            )
        elif self._stage is EnvelopeStage.DECAY:
            self._stage = EnvelopeStage.SUSTAIN
            self._level = self.spec.sustain_amplitude
            self._segment_start = self._level
            self._target = self._level
            self._segment_frames = 0
            self._emitted = 0
            self._curvature = 0.0
            self._linear_step = 0.0
        elif self._stage is EnvelopeStage.RELEASE:
            self.reset()
