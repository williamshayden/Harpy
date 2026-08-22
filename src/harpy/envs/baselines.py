"""Capability-isolated baselines and summaries for fixed Gymnasium evidence lanes."""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

import gymnasium
import numpy as np

from harpy.envs.models import (
    CENT_MIN,
    OCTAVE_MIN,
    SEMITONE_MIN,
    SOURCE_MAX_CENTS,
    SOURCE_MIN_CENTS,
    SUCCESS_TOLERANCE_CENTS,
    TARGET_MIN_COORDINATE,
    ControlState,
    EpisodeResult,
    ObservationMode,
    PitchAction,
    TerminalReason,
)
from harpy.envs.planning import minimum_action_plan

type JSONScalar = str | int | float | bool | None


class BaselineKind(StrEnum):
    RANDOM = "random"
    SPECTRUM_PEAK = "spectrum_peak"
    ORACLE = "oracle"
    REWARD_SEARCH = "reward_search"

    @property
    def observation_mode(self) -> ObservationMode:
        match self:
            case BaselineKind.RANDOM | BaselineKind.SPECTRUM_PEAK:
                return ObservationMode.SPECTRUM
            case BaselineKind.ORACLE:
                return ObservationMode.ORACLE
            case BaselineKind.REWARD_SEARCH:
                return ObservationMode.REWARD_ONLY

    @property
    def environment_id(self) -> str:
        match self:
            case BaselineKind.RANDOM | BaselineKind.SPECTRUM_PEAK:
                return "Harpy/SinePitch-v0"
            case BaselineKind.ORACLE:
                return "Harpy/SinePitchOracle-v0"
            case BaselineKind.REWARD_SEARCH:
                return "Harpy/SinePitchRewardOnly-v0"


@dataclass(frozen=True, slots=True)
class BaselineSummary:
    baseline: str
    environment_id: str
    observation_mode: str
    episodes: int
    seed: int
    submitted_success_rate: float
    submitted_within_1_cent_rate: float
    final_within_5_cents_rate: float
    final_within_1_cent_rate: float
    mean_absolute_final_error_cents: float
    mean_actions: float
    mean_excess_actions: float | None
    mean_return: float
    truncation_rate: float
    invalid_action_rate: float

    def to_dict(self) -> dict[str, JSONScalar]:
        """Return a fresh JSON-scalar mapping of this summary."""

        return {
            "baseline": self.baseline,
            "environment_id": self.environment_id,
            "observation_mode": self.observation_mode,
            "episodes": self.episodes,
            "seed": self.seed,
            "submitted_success_rate": self.submitted_success_rate,
            "submitted_within_1_cent_rate": self.submitted_within_1_cent_rate,
            "final_within_5_cents_rate": self.final_within_5_cents_rate,
            "final_within_1_cent_rate": self.final_within_1_cent_rate,
            "mean_absolute_final_error_cents": self.mean_absolute_final_error_cents,
            "mean_actions": self.mean_actions,
            "mean_excess_actions": self.mean_excess_actions,
            "mean_return": self.mean_return,
            "truncation_rate": self.truncation_rate,
            "invalid_action_rate": self.invalid_action_rate,
        }


class RandomPolicy:
    """Sample among actions that visible control bounds currently permit."""

    def __init__(self, rng: np.random.Generator) -> None:
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        self._rng = rng

    def act(
        self,
        observation: Mapping[str, object],
        previous_reward: float | None,
        previous_info: Mapping[str, object] | None,
    ) -> PitchAction:
        del previous_reward, previous_info
        controls = _controls(observation)
        legal_actions = [PitchAction.SUBMIT]
        for action in PitchAction:
            if action is PitchAction.SUBMIT:
                continue
            _, applied = controls.apply(action)
            if applied:
                legal_actions.append(action)
        return legal_actions[int(self._rng.integers(len(legal_actions)))]


class RewardSearchPolicy:
    """Search octave, semitone, then cent controls using transition rewards only."""

    _SCALES = (
        (0, OCTAVE_MIN, PitchAction.OCTAVE_DOWN, PitchAction.OCTAVE_UP),
        (1, SEMITONE_MIN, PitchAction.SEMITONE_DOWN, PitchAction.SEMITONE_UP),
        (2, CENT_MIN, PitchAction.CENT_DOWN, PitchAction.CENT_UP),
    )

    def __init__(self) -> None:
        self._scale_index = 0
        self._direction = -1
        self._positive_progress = False
        self._initial_down_probe = True
        self._last_role: str | None = None
        self._after_undo: str | None = None

    def act(
        self,
        observation: Mapping[str, object],
        previous_reward: float | None,
        previous_info: Mapping[str, object] | None,
    ) -> PitchAction:
        if int(observation["steps_remaining"]) == 1:
            return PitchAction.SUBMIT
        if self._scale_index == len(self._SCALES):
            return PitchAction.SUBMIT
        if self._last_role is None:
            return self._start_scale(observation)

        if self._last_role == "undo":
            if self._after_undo == "search_up":
                self._direction = 1
                self._initial_down_probe = False
                return self._emit_move()
            self._advance_scale()
            if self._scale_index == len(self._SCALES):
                return PitchAction.SUBMIT
            return self._start_scale(observation)

        if previous_info is None or previous_reward is None:
            raise ValueError("reward search feedback is required after the first action")
        if not bool(previous_info["action_applied"]):
            self._direction *= -1
            self._initial_down_probe = False
            return self._emit_move()
        if previous_reward > 0.0:
            self._positive_progress = True
            self._initial_down_probe = False
            return self._emit_move()

        if self._initial_down_probe and not self._positive_progress:
            self._after_undo = "search_up"
        else:
            self._after_undo = "advance"
        self._last_role = "undo"
        return self._action(-self._direction)

    def _start_scale(self, observation: Mapping[str, object]) -> PitchAction:
        control_index, minimum, _, _ = self._SCALES[self._scale_index]
        controls = np.asarray(observation["controls"])
        self._positive_progress = False
        self._direction = -1 if int(controls[control_index]) > minimum else 1
        self._initial_down_probe = self._direction == -1
        return self._emit_move()

    def _advance_scale(self) -> None:
        self._scale_index += 1
        self._last_role = None
        self._after_undo = None

    def _emit_move(self) -> PitchAction:
        self._last_role = "move"
        self._after_undo = None
        return self._action(self._direction)

    def _action(self, direction: int) -> PitchAction:
        _, _, down, up = self._SCALES[self._scale_index]
        return down if direction == -1 else up


def oracle_plan(observation: Mapping[str, object]) -> tuple[PitchAction, ...]:
    """Plan from the visible float32 pitch coordinate and public controls."""

    controls = _controls(observation)
    coordinate = np.asarray(observation["current_pitch_coordinate"])[0]
    current_pitch_cents = round(100 * float(coordinate))
    base_error_cents = current_pitch_cents - controls.offset_cents - _target_cents(observation)
    return minimum_action_plan(
        base_error_cents,
        controls,
        tolerance_cents=SUCCESS_TOLERANCE_CENTS,
    )


def spectrum_peak_plan(observation: Mapping[str, object]) -> tuple[PitchAction, ...]:
    """Plan from the first maximum of the visible fixed-grid spectrum."""

    controls = _controls(observation)
    peak_index = int(np.argmax(observation["spectrum"]))
    estimated_current_cents = 1_100 + 5 * peak_index
    estimated_source_cents = min(
        max(estimated_current_cents - controls.offset_cents, SOURCE_MIN_CENTS),
        SOURCE_MAX_CENTS,
    )
    base_error_cents = estimated_source_cents - _target_cents(observation)
    return minimum_action_plan(base_error_cents, controls, tolerance_cents=0)


def evaluate_baseline(kind: BaselineKind, *, episodes: int, seed: int) -> BaselineSummary:
    """Roll out one declared baseline lane and aggregate evaluator-owned results."""

    episode_count, evaluation_seed = _validate_evaluation_arguments(episodes, seed)
    if not isinstance(kind, BaselineKind):
        raise ValueError("kind must be a BaselineKind")

    env = gymnasium.make(kind.environment_id)
    results: list[EpisodeResult] = []
    try:
        for episode_index in range(episode_count):
            observation, _ = env.reset(seed=evaluation_seed + episode_index)
            seed_sequence = np.random.SeedSequence(
                [evaluation_seed, _STABLE_POLICY_CODES[kind], episode_index]
            )
            policy = _make_policy(kind, np.random.default_rng(seed_sequence))
            previous_reward: float | None = None
            previous_info: Mapping[str, object] | None = None

            while True:
                action = policy.act(observation, previous_reward, previous_info)
                observation, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    result = env.unwrapped.episode_result
                    if not isinstance(result, EpisodeResult):
                        raise RuntimeError("environment produced no EpisodeResult after done")
                    results.append(result)
                    break
                previous_reward = reward
                previous_info = info
    finally:
        env.close()

    return _summarize(kind, episode_count, evaluation_seed, tuple(results))


def evaluate_all_baselines(*, episodes: int, seed: int) -> tuple[BaselineSummary, ...]:
    """Evaluate every declared lane in fixed baseline order without pooling rows."""

    episode_count, evaluation_seed = _validate_evaluation_arguments(episodes, seed)
    return tuple(
        evaluate_baseline(kind, episodes=episode_count, seed=evaluation_seed)
        for kind in _BASELINE_ORDER
    )


class _PlannerPolicy:
    def __init__(
        self,
        planner: Callable[[Mapping[str, object]], tuple[PitchAction, ...]],
    ) -> None:
        self._planner = planner
        self._actions: tuple[PitchAction, ...] | None = None
        self._action_index = 0

    def act(
        self,
        observation: Mapping[str, object],
        previous_reward: float | None,
        previous_info: Mapping[str, object] | None,
    ) -> PitchAction:
        del previous_reward, previous_info
        if self._actions is None:
            self._actions = self._planner(observation)
        action = self._actions[self._action_index]
        self._action_index += 1
        return action


def _make_policy(
    kind: BaselineKind, rng: np.random.Generator
) -> RandomPolicy | RewardSearchPolicy | _PlannerPolicy:
    match kind:
        case BaselineKind.RANDOM:
            return RandomPolicy(rng)
        case BaselineKind.SPECTRUM_PEAK:
            return _PlannerPolicy(spectrum_peak_plan)
        case BaselineKind.ORACLE:
            return _PlannerPolicy(oracle_plan)
        case BaselineKind.REWARD_SEARCH:
            return RewardSearchPolicy()


def _summarize(
    kind: BaselineKind,
    episodes: int,
    seed: int,
    results: tuple[EpisodeResult, ...],
) -> BaselineSummary:
    submitted_successes = tuple(result for result in results if result.submitted_success)
    successful_excess_actions = tuple(
        result.excess_actions for result in submitted_successes if result.excess_actions is not None
    )
    total_actions = sum(len(result.actions) for result in results)
    invalid_actions = sum(result.invalid_action_count for result in results)
    return BaselineSummary(
        baseline=kind.value,
        environment_id=kind.environment_id,
        observation_mode=kind.observation_mode.value,
        episodes=episodes,
        seed=seed,
        submitted_success_rate=sum(result.submitted_success for result in results) / episodes,
        submitted_within_1_cent_rate=sum(
            bool(result.actions)
            and result.actions[-1] is PitchAction.SUBMIT
            and result.within_1_cent
            for result in results
        )
        / episodes,
        final_within_5_cents_rate=sum(result.within_5_cents for result in results) / episodes,
        final_within_1_cent_rate=sum(result.within_1_cent for result in results) / episodes,
        mean_absolute_final_error_cents=sum(result.final_absolute_error_cents for result in results)
        / episodes,
        mean_actions=total_actions / episodes,
        mean_excess_actions=(
            sum(successful_excess_actions) / len(successful_excess_actions)
            if successful_excess_actions
            else None
        ),
        mean_return=sum(result.total_return for result in results) / episodes,
        truncation_rate=sum(
            result.terminal_reason is TerminalReason.BUDGET_EXHAUSTED for result in results
        )
        / episodes,
        invalid_action_rate=invalid_actions / total_actions,
    )


def _validate_evaluation_arguments(episodes: object, seed: object) -> tuple[int, int]:
    episode_count = _integer_index(episodes, "episodes must be a positive integer")
    if episode_count <= 0:
        raise ValueError("episodes must be a positive integer")
    evaluation_seed = _integer_index(seed, "seed must be a non-negative integer")
    if evaluation_seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return episode_count, evaluation_seed


def _integer_index(value: object, message: str) -> int:
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(message) from error


_BASELINE_ORDER = (
    BaselineKind.RANDOM,
    BaselineKind.SPECTRUM_PEAK,
    BaselineKind.ORACLE,
    BaselineKind.REWARD_SEARCH,
)
_STABLE_POLICY_CODES = {
    BaselineKind.RANDOM: 1,
    BaselineKind.SPECTRUM_PEAK: 2,
    BaselineKind.ORACLE: 3,
    BaselineKind.REWARD_SEARCH: 4,
}


def _controls(observation: Mapping[str, object]) -> ControlState:
    values = np.asarray(observation["controls"])
    return ControlState(*(int(values[index]) for index in range(3)))


def _target_cents(observation: Mapping[str, object]) -> int:
    return 100 * (TARGET_MIN_COORDINATE + int(observation["target_note"]))
