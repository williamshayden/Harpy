"""Deterministic headless Gymnasium environment for direct sine-pitch control."""

from __future__ import annotations

import copy
import operator
from typing import Any, ClassVar

import gymnasium
import numpy as np

from harpy.envs.models import (
    CENT_MAX,
    CENT_MIN,
    MAX_ABSOLUTE_ERROR_CENTS,
    MAX_STEPS,
    OCTAVE_MAX,
    OCTAVE_MIN,
    SEMITONE_MAX,
    SEMITONE_MIN,
    SOURCE_MAX_CENTS,
    SOURCE_MIN_CENTS,
    SUCCESS_TOLERANCE_CENTS,
    TARGET_MIN_COORDINATE,
    TARGET_NOTE_COUNT,
    ControlState,
    EpisodeResult,
    ObservationMode,
    PitchAction,
    TerminalReason,
)
from harpy.envs.planning import minimum_action_plan
from harpy.envs.spectrum import GYM_ANALYSIS_CONFIG, LOG_SPECTRUM_SIZE, encode_log_spectrum
from harpy.synth import RenderConfig, SynthEngine, SynthPatch
from harpy.synth.models import seconds_to_frames
from harpy.tuning import Tuning

Observation = dict[str, np.ndarray[Any, Any] | np.int64]


class SinePitchEnv(gymnasium.Env[Observation, int]):
    """A deterministic episode over independent octave, semitone, and cent controls."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(
        self,
        observation_mode: ObservationMode = ObservationMode.SPECTRUM,
        render_mode: None = None,
    ) -> None:
        if not isinstance(observation_mode, ObservationMode):
            raise ValueError("observation_mode must be an ObservationMode")
        if render_mode is not None:
            raise ValueError("render_mode must be None")

        self.observation_mode = observation_mode
        self.render_mode = render_mode
        self.action_space = gymnasium.spaces.Discrete(len(PitchAction))
        self.observation_space = self._make_observation_space(observation_mode)

        self._source_pitch_cents: int | None = None
        self._target_note_index: int | None = None
        self._controls = ControlState()
        self._step_count = 0
        self._actions: tuple[PitchAction, ...] = ()
        self._invalid_action_count = 0
        self._total_return = 0.0
        self._episode_result: EpisodeResult | None = None
        self._candidate_audio: np.ndarray[Any, np.dtype[np.float32]] | None = None
        self._spectrum: np.ndarray[Any, np.dtype[np.float32]] | None = None

    @property
    def episode_result(self) -> EpisodeResult:
        if self._episode_result is None:
            raise RuntimeError("episode_result is available only when the episode is done")
        return self._episode_result

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, int]]:
        injected = _validate_options(options)
        validated_seed = _validate_seed(seed)

        previous_rng = getattr(self, "_np_random", None)
        previous_rng_seed = getattr(self, "_np_random_seed", None)
        previous_rng_state = (
            None if previous_rng is None else copy.deepcopy(previous_rng.bit_generator.state)
        )
        try:
            super().reset(seed=validated_seed)
            _ = self.np_random
            if injected is None:
                target_note_index, source_pitch_cents = self._sample_episode()
            else:
                target_note_index, source_pitch_cents = injected
            candidate_audio, spectrum = self._render_candidate(source_pitch_cents)
        except Exception:
            if previous_rng is not None and previous_rng_state is not None:
                previous_rng.bit_generator.state = previous_rng_state
            self._np_random = previous_rng
            self._np_random_seed = previous_rng_seed
            raise

        self._source_pitch_cents = source_pitch_cents
        self._target_note_index = target_note_index
        self._controls = ControlState()
        self._step_count = 0
        self._actions = ()
        self._invalid_action_count = 0
        self._total_return = 0.0
        self._episode_result = None
        self._candidate_audio = candidate_audio
        self._spectrum = spectrum

        return self._observation(), {"step_count": 0, "steps_remaining": MAX_STEPS}

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        if self._source_pitch_cents is None or self._target_note_index is None:
            raise RuntimeError("reset must be called before step")
        if self._episode_result is not None:
            raise RuntimeError("cannot step an episode that is already done")
        pitch_action = _validate_action(action)

        target_pitch_cents = 100 * (TARGET_MIN_COORDINATE + self._target_note_index)
        before_pitch_cents = self._source_pitch_cents + self._controls.offset_cents
        before_absolute_error = abs(before_pitch_cents - target_pitch_cents)
        controls = self._controls
        candidate_audio = self._candidate_audio
        spectrum = self._spectrum
        invalid_action_count = self._invalid_action_count
        action_applied = False
        submitted = pitch_action is PitchAction.SUBMIT
        submitted_success = False

        if submitted:
            submitted_success = before_absolute_error <= SUCCESS_TOLERANCE_CENTS
            reward = 1.0 if submitted_success else -1.0
            terminated = True
            truncated = False
        else:
            controls, action_applied = controls.apply(pitch_action)
            if action_applied:
                after_pitch_cents = self._source_pitch_cents + controls.offset_cents
                after_absolute_error = abs(after_pitch_cents - target_pitch_cents)
                reward = (
                    before_absolute_error - after_absolute_error
                ) / MAX_ABSOLUTE_ERROR_CENTS - 0.00001
                candidate_audio, spectrum = self._render_candidate(after_pitch_cents)
            else:
                reward = -0.01
                invalid_action_count += 1
            terminated = False
            truncated = self._step_count + 1 == MAX_STEPS
            if truncated:
                reward -= 1.0

        step_count = self._step_count + 1
        actions = (*self._actions, pitch_action)
        total_return = self._total_return + reward
        episode_result = None
        if terminated or truncated:
            episode_result = self._make_episode_result(
                submitted_success=submitted_success,
                submitted=submitted,
                controls=controls,
                actions=actions,
                invalid_action_count=invalid_action_count,
                total_return=total_return,
            )

        self._controls = controls
        self._step_count = step_count
        self._actions = actions
        self._invalid_action_count = invalid_action_count
        self._total_return = total_return
        self._episode_result = episode_result
        self._candidate_audio = candidate_audio
        self._spectrum = spectrum

        info = {
            "step_count": step_count,
            "steps_remaining": MAX_STEPS - step_count,
            "action": pitch_action.label,
            "action_applied": action_applied,
            "submitted": submitted,
            "submitted_success": submitted_success,
        }
        return self._observation(), float(reward), terminated, truncated, info

    def render(self) -> None:
        return None

    @staticmethod
    def _make_observation_space(
        observation_mode: ObservationMode,
    ) -> gymnasium.spaces.Dict:
        spaces: dict[str, gymnasium.Space] = {
            "target_note": gymnasium.spaces.Discrete(TARGET_NOTE_COUNT, dtype=np.int64),
            "controls": gymnasium.spaces.Box(
                low=np.array([OCTAVE_MIN, SEMITONE_MIN, CENT_MIN], dtype=np.int16),
                high=np.array([OCTAVE_MAX, SEMITONE_MAX, CENT_MAX], dtype=np.int16),
                dtype=np.int16,
            ),
            "steps_remaining": gymnasium.spaces.Discrete(MAX_STEPS + 1, dtype=np.int64),
        }
        if observation_mode is ObservationMode.SPECTRUM:
            spaces["spectrum"] = gymnasium.spaces.Box(
                0.0, 1.0, shape=(LOG_SPECTRUM_SIZE,), dtype=np.float32
            )
        elif observation_mode is ObservationMode.ORACLE:
            spaces["current_pitch_coordinate"] = gymnasium.spaces.Box(
                11.0, 109.0, shape=(1,), dtype=np.float32
            )
        return gymnasium.spaces.Dict(spaces)

    def _sample_episode(self) -> tuple[int, int]:
        while True:
            target_note_index = int(self.np_random.integers(0, TARGET_NOTE_COUNT))
            source_pitch_cents = int(
                self.np_random.integers(SOURCE_MIN_CENTS, SOURCE_MAX_CENTS + 1)
            )
            target_pitch_cents = 100 * (TARGET_MIN_COORDINATE + target_note_index)
            if abs(source_pitch_cents - target_pitch_cents) > SUCCESS_TOLERANCE_CENTS:
                return target_note_index, source_pitch_cents

    def _render_candidate(
        self, candidate_cents: int
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.float32]],
        np.ndarray[Any, np.dtype[np.float32]],
    ]:
        engine = SynthEngine(RenderConfig(), SynthPatch())
        engine.note_on(Tuning().frequency_hz_for_midi_coordinate(candidate_cents / 100.0))
        settle = seconds_to_frames(0.001, 48_000) + seconds_to_frames(0.600, 48_000)
        rendered = engine.render(settle + GYM_ANALYSIS_CONFIG.fft_frames)
        candidate_audio = np.array(
            rendered[-GYM_ANALYSIS_CONFIG.fft_frames :],
            dtype=np.float32,
            copy=True,
            order="C",
        )
        candidate_audio.setflags(write=False)
        spectrum = encode_log_spectrum(candidate_audio)
        spectrum.setflags(write=False)
        return candidate_audio, spectrum

    def _observation(self) -> Observation:
        if (
            self._source_pitch_cents is None
            or self._target_note_index is None
            or self._candidate_audio is None
            or self._spectrum is None
        ):
            raise RuntimeError("reset must be called before requesting an observation")
        observation: Observation = {
            "target_note": np.int64(self._target_note_index),
            "controls": np.array(
                [self._controls.octaves, self._controls.semitones, self._controls.cents],
                dtype=np.int16,
            ),
            "steps_remaining": np.int64(MAX_STEPS - self._step_count),
        }
        if self.observation_mode is ObservationMode.SPECTRUM:
            observation["spectrum"] = np.array(
                self._spectrum, dtype=np.float32, copy=True, order="C"
            )
        elif self.observation_mode is ObservationMode.ORACLE:
            candidate_pitch_cents = self._source_pitch_cents + self._controls.offset_cents
            observation["current_pitch_coordinate"] = np.array(
                [candidate_pitch_cents / 100.0], dtype=np.float32
            )
        return observation

    def _make_episode_result(
        self,
        *,
        submitted_success: bool,
        submitted: bool,
        controls: ControlState,
        actions: tuple[PitchAction, ...],
        invalid_action_count: int,
        total_return: float,
    ) -> EpisodeResult:
        if self._source_pitch_cents is None or self._target_note_index is None:
            raise RuntimeError("reset must be called before producing a result")
        target_pitch_cents = 100 * (TARGET_MIN_COORDINATE + self._target_note_index)
        initial_signed_error_cents = self._source_pitch_cents - target_pitch_cents
        final_pitch_cents = self._source_pitch_cents + controls.offset_cents
        final_signed_error_cents = final_pitch_cents - target_pitch_cents
        final_absolute_error_cents = abs(final_signed_error_cents)
        optimal_actions = minimum_action_plan(
            initial_signed_error_cents,
            tolerance_cents=SUCCESS_TOLERANCE_CENTS,
        )
        if submitted_success:
            terminal_reason = TerminalReason.SUBMITTED_SUCCESS
            excess_actions = len(actions) - len(optimal_actions)
        elif submitted:
            terminal_reason = TerminalReason.SUBMITTED_FAILURE
            excess_actions = None
        else:
            terminal_reason = TerminalReason.BUDGET_EXHAUSTED
            excess_actions = None
        return EpisodeResult(
            source_pitch_cents=self._source_pitch_cents,
            target_note_index=self._target_note_index,
            target_pitch_cents=target_pitch_cents,
            final_pitch_cents=final_pitch_cents,
            initial_signed_error_cents=initial_signed_error_cents,
            initial_absolute_error_cents=abs(initial_signed_error_cents),
            final_signed_error_cents=final_signed_error_cents,
            final_absolute_error_cents=final_absolute_error_cents,
            submitted_success=submitted_success,
            within_5_cents=final_absolute_error_cents <= SUCCESS_TOLERANCE_CENTS,
            within_1_cent=final_absolute_error_cents <= 1,
            actions=actions,
            invalid_action_count=invalid_action_count,
            optimal_actions=optimal_actions,
            excess_actions=excess_actions,
            total_return=float(total_return),
            terminal_reason=terminal_reason,
        )


def _validate_seed(seed: object) -> int | None:
    if seed is None:
        return None
    value = _index_scalar(seed, "seed")
    if value < 0:
        raise ValueError("seed must be a non-negative integer or None")
    return value


def _validate_action(action: object) -> PitchAction:
    value = _index_scalar(action, "action")
    if not 0 <= value < len(PitchAction):
        raise ValueError(f"action must be an integer within 0..{len(PitchAction) - 1}")
    return PitchAction(value)


def _validate_options(options: object) -> tuple[int, int] | None:
    if options is None:
        return None
    if not isinstance(options, dict):
        raise ValueError("options must be a dict or None")
    if not options:
        return None
    if set(options) != {"target_note_index", "source_pitch_cents"}:
        raise ValueError("options must contain exactly target_note_index and source_pitch_cents")

    target_note_index = _index_scalar(options["target_note_index"], "target_note_index")
    source_pitch_cents = _index_scalar(options["source_pitch_cents"], "source_pitch_cents")
    if not 0 <= target_note_index < TARGET_NOTE_COUNT:
        raise ValueError("target_note_index must be within 0..24")
    if not SOURCE_MIN_CENTS <= source_pitch_cents <= SOURCE_MAX_CENTS:
        raise ValueError("source_pitch_cents must be within 4800..7200")
    return target_note_index, source_pitch_cents


def _index_scalar(value: object, name: str) -> int:
    if isinstance(value, (bool, np.ndarray)):
        raise ValueError(f"{name} must be an integer scalar")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer scalar") from error
