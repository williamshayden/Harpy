"""Capability-isolated episode policies and deterministic baseline evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields
from typing import ClassVar

import numpy as np
import pytest

import harpy.envs.baselines as baselines
from harpy.envs.baselines import RandomPolicy, RewardSearchPolicy
from harpy.envs.models import (
    EpisodeResult,
    ObservationMode,
    PitchAction,
    TerminalReason,
)
from harpy.envs.sine_pitch import SinePitchEnv
from harpy.envs.spectrum import LOG_SPECTRUM_SIZE


def test_random_policy_requires_a_numpy_generator() -> None:
    with pytest.raises(ValueError, match=r"rng must be a numpy.random.Generator"):
        RandomPolicy(7)  # type: ignore[arg-type]


def test_random_policy_samples_only_visible_legal_actions_and_is_repeatable() -> None:
    lower_bound = _reward_observation(controls=(-2, -12, -100))
    upper_bound = _reward_observation(controls=(2, 12, 100))

    first = RandomPolicy(np.random.default_rng(873))
    second = RandomPolicy(np.random.default_rng(873))
    lower_actions = [first.act(lower_bound, None, None) for _ in range(100)]
    repeated_actions = [second.act(lower_bound, None, None) for _ in range(100)]
    upper_actions = [first.act(upper_bound, None, None) for _ in range(100)]

    assert lower_actions == repeated_actions
    assert set(lower_actions) <= {
        PitchAction.SUBMIT,
        PitchAction.CENT_UP,
        PitchAction.SEMITONE_UP,
        PitchAction.OCTAVE_UP,
    }
    assert set(upper_actions) <= {
        PitchAction.OCTAVE_DOWN,
        PitchAction.SEMITONE_DOWN,
        PitchAction.CENT_DOWN,
        PitchAction.SUBMIT,
    }
    assert PitchAction.SUBMIT in lower_actions
    assert PitchAction.SUBMIT in upper_actions


def test_reward_search_visits_scales_in_order_and_handles_both_applied_directions() -> None:
    policy = RewardSearchPolicy()
    observation = _reward_observation()

    assert policy.act(observation, None, None) is PitchAction.OCTAVE_DOWN
    assert _respond(policy, reward=0.0) is PitchAction.OCTAVE_UP
    assert _respond(policy, reward=0.0) is PitchAction.OCTAVE_UP
    assert _respond(policy, reward=0.01) is PitchAction.OCTAVE_UP
    assert _respond(policy, reward=0.0) is PitchAction.OCTAVE_DOWN
    assert _respond(policy, reward=0.0) is PitchAction.SEMITONE_DOWN
    assert _respond(policy, reward=0.0) is PitchAction.SEMITONE_UP
    assert _respond(policy, reward=0.0) is PitchAction.SEMITONE_UP
    assert _respond(policy, reward=0.0) is PitchAction.SEMITONE_DOWN
    assert _respond(policy, reward=0.0) is PitchAction.CENT_DOWN


def test_reward_search_positive_down_then_nonpositive_continuation_advances() -> None:
    policy = RewardSearchPolicy()

    assert policy.act(_reward_observation(), None, None) is PitchAction.OCTAVE_DOWN
    assert _respond(policy, reward=0.01) is PitchAction.OCTAVE_DOWN
    assert _respond(policy, reward=0.0) is PitchAction.OCTAVE_UP
    assert _respond(policy, reward=0.0) is PitchAction.SEMITONE_DOWN


def test_reward_search_nonpositive_first_down_undoes_then_searches_up() -> None:
    policy = RewardSearchPolicy()

    assert policy.act(_reward_observation(), None, None) is PitchAction.OCTAVE_DOWN
    assert _respond(policy, reward=0.0) is PitchAction.OCTAVE_UP
    assert _respond(policy, reward=0.0) is PitchAction.OCTAVE_UP
    assert _respond(policy, reward=0.01) is PitchAction.OCTAVE_UP
    assert _respond(policy, reward=0.0) is PitchAction.OCTAVE_DOWN
    assert _respond(policy, reward=0.0) is PitchAction.SEMITONE_DOWN


def test_reward_search_blocked_first_down_reverses_without_scheduling_undo() -> None:
    policy = RewardSearchPolicy()

    assert policy.act(_reward_observation(), None, None) is PitchAction.OCTAVE_DOWN
    assert (
        _respond(policy, reward=-0.01, applied=False, controls=(-2, 0, 0)) is PitchAction.OCTAVE_UP
    )
    assert _respond(policy, reward=0.0) is PitchAction.OCTAVE_DOWN
    assert _respond(policy, reward=0.0) is PitchAction.SEMITONE_DOWN


def test_reward_search_blocked_down_continuation_reverses_without_undo() -> None:
    policy = RewardSearchPolicy()

    assert policy.act(_reward_observation(), None, None) is PitchAction.OCTAVE_DOWN
    assert _respond(policy, reward=0.01) is PitchAction.OCTAVE_DOWN
    assert (
        _respond(policy, reward=-0.01, applied=False, controls=(-2, 0, 0)) is PitchAction.OCTAVE_UP
    )
    assert _respond(policy, reward=0.01) is PitchAction.OCTAVE_UP


def test_reward_search_blocked_up_probe_reverses_without_undo() -> None:
    policy = RewardSearchPolicy()

    assert policy.act(_reward_observation(), None, None) is PitchAction.OCTAVE_DOWN
    assert _respond(policy, reward=0.0) is PitchAction.OCTAVE_UP
    assert _respond(policy, reward=0.0) is PitchAction.OCTAVE_UP
    assert (
        _respond(policy, reward=-0.01, applied=False, controls=(2, 0, 0)) is PitchAction.OCTAVE_DOWN
    )
    assert _respond(policy, reward=0.01) is PitchAction.OCTAVE_DOWN


def test_reward_search_blocked_up_continuation_reverses_without_undo() -> None:
    policy = RewardSearchPolicy()

    assert policy.act(_reward_observation(), None, None) is PitchAction.OCTAVE_DOWN
    assert _respond(policy, reward=0.0) is PitchAction.OCTAVE_UP
    assert _respond(policy, reward=0.0) is PitchAction.OCTAVE_UP
    assert _respond(policy, reward=0.01) is PitchAction.OCTAVE_UP
    assert (
        _respond(policy, reward=-0.01, applied=False, controls=(2, 0, 0)) is PitchAction.OCTAVE_DOWN
    )
    assert _respond(policy, reward=0.01) is PitchAction.OCTAVE_DOWN


def test_reward_search_submits_when_only_one_step_remains() -> None:
    policy = RewardSearchPolicy()

    assert policy.act(_reward_observation(steps_remaining=1), None, None) is PitchAction.SUBMIT


@pytest.mark.parametrize(
    ("target_note_index", "source_pitch_cents", "required_action"),
    [
        (0, 6_000, PitchAction.OCTAVE_DOWN),
        (12, 4_800, PitchAction.OCTAVE_UP),
        (12, 6_100, PitchAction.SEMITONE_DOWN),
        (12, 5_900, PitchAction.SEMITONE_UP),
        (12, 6_020, PitchAction.CENT_DOWN),
        (12, 5_980, PitchAction.CENT_UP),
    ],
)
def test_reward_search_real_episodes_use_each_scale_and_direction_without_truncating(
    target_note_index: int,
    source_pitch_cents: int,
    required_action: PitchAction,
) -> None:
    result = _run_reward_episode(target_note_index, source_pitch_cents)

    assert required_action in result.actions
    assert result.actions[-1] is PitchAction.SUBMIT
    assert len(result.actions) <= 64


def test_reward_search_reserves_submit_even_when_refinement_is_incomplete() -> None:
    result = _run_reward_episode(target_note_index=0, source_pitch_cents=6_650)

    assert len(result.actions) == 64
    assert result.actions[-1] is PitchAction.SUBMIT
    assert result.final_absolute_error_cents == 3


def test_baseline_summary_is_frozen_complete_and_returns_fresh_plain_mappings() -> None:
    summary = baselines.BaselineSummary(
        baseline="random",
        environment_id="Harpy/SinePitch-v0",
        observation_mode="spectrum",
        episodes=2,
        seed=91,
        submitted_success_rate=0.5,
        submitted_within_1_cent_rate=0.0,
        final_within_5_cents_rate=0.5,
        final_within_1_cent_rate=0.0,
        mean_absolute_final_error_cents=17.5,
        mean_actions=2.0,
        mean_excess_actions=1.0,
        mean_return=-0.25,
        truncation_rate=0.0,
        invalid_action_rate=0.25,
    )

    assert [field.name for field in fields(summary)] == [
        "baseline",
        "environment_id",
        "observation_mode",
        "episodes",
        "seed",
        "submitted_success_rate",
        "submitted_within_1_cent_rate",
        "final_within_5_cents_rate",
        "final_within_1_cent_rate",
        "mean_absolute_final_error_cents",
        "mean_actions",
        "mean_excess_actions",
        "mean_return",
        "truncation_rate",
        "invalid_action_rate",
    ]
    with pytest.raises(FrozenInstanceError):
        summary.seed = 92  # type: ignore[misc]

    first = summary.to_dict()
    second = summary.to_dict()
    first["baseline"] = "mutated"

    assert type(first) is dict
    assert first is not second
    assert second["baseline"] == "random"
    assert all(
        value is None or isinstance(value, str | int | float | bool) for value in second.values()
    )
    assert not ({"audio", "episode_result", "source_pitch_cents"} & second.keys())


@pytest.mark.parametrize(
    ("episodes", "seed", "message"),
    [
        (True, 0, "episodes must be a positive integer"),
        (0, 0, "episodes must be a positive integer"),
        (-1, 0, "episodes must be a positive integer"),
        (1.5, 0, "episodes must be a positive integer"),
        (1, True, "seed must be a non-negative integer"),
        (1, -1, "seed must be a non-negative integer"),
        (1, 2.5, "seed must be a non-negative integer"),
    ],
)
def test_evaluation_rejects_invalid_counts_and_seeds(
    episodes: object,
    seed: object,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_construction(*args: object, **kwargs: object) -> None:
        raise AssertionError("validation must precede environment and RNG construction")

    monkeypatch.setattr(baselines.gymnasium, "make", forbidden_construction)
    monkeypatch.setattr(baselines.np.random, "SeedSequence", forbidden_construction)

    with pytest.raises(ValueError, match=message):
        baselines.evaluate_baseline(  # type: ignore[arg-type]
            baselines.BaselineKind.RANDOM,
            episodes=episodes,
            seed=seed,
        )


def test_all_baselines_use_fixed_unpooled_lanes_and_matched_episode_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records: list[dict[str, object]] = []
    active: dict[SinePitchEnv, dict[str, object]] = {}
    original_reset = SinePitchEnv.reset
    original_step = SinePitchEnv.step

    def recording_reset(
        env: SinePitchEnv,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ):
        observation, info = original_reset(env, seed=seed, options=options)
        record: dict[str, object] = {
            "mode": env.observation_mode,
            "seed": seed,
            "latent": (env._source_pitch_cents, env._target_note_index),
            "actions": [],
        }
        records.append(record)
        active[env] = record
        return observation, info

    def recording_step(env: SinePitchEnv, action: int):
        actions = active[env]["actions"]
        assert isinstance(actions, list)
        actions.append(int(action))
        return original_step(env, action)

    monkeypatch.setattr(SinePitchEnv, "reset", recording_reset)
    monkeypatch.setattr(SinePitchEnv, "step", recording_step)

    first_summaries = baselines.evaluate_all_baselines(episodes=2, seed=300)
    first_records = [{**record, "actions": tuple(record["actions"])} for record in records]
    records.clear()
    second_summaries = baselines.evaluate_all_baselines(episodes=2, seed=300)
    second_records = [{**record, "actions": tuple(record["actions"])} for record in records]

    assert [
        (summary.baseline, summary.observation_mode, summary.environment_id)
        for summary in first_summaries
    ] == [
        ("random", "spectrum", "Harpy/SinePitch-v0"),
        ("spectrum_peak", "spectrum", "Harpy/SinePitch-v0"),
        ("oracle", "oracle", "Harpy/SinePitchOracle-v0"),
        ("reward_search", "reward_only", "Harpy/SinePitchRewardOnly-v0"),
    ]
    assert all(summary.episodes == 2 for summary in first_summaries)
    assert [record["seed"] for record in first_records] == [300, 301] * 4
    for episode_index in range(2):
        assert (
            len(
                {first_records[lane_index * 2 + episode_index]["latent"] for lane_index in range(4)}
            )
            == 1
        )
    assert first_records == second_records
    assert (
        json.dumps([summary.to_dict() for summary in first_summaries], sort_keys=True).encode()
        == json.dumps([summary.to_dict() for summary in second_summaries], sort_keys=True).encode()
    )


def test_each_lane_uses_its_stable_policy_seed_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entropy_values: list[list[int]] = []
    original_seed_sequence = np.random.SeedSequence

    def recording_seed_sequence(entropy: object) -> np.random.SeedSequence:
        if isinstance(entropy, list):
            entropy_values.append(entropy)
        return original_seed_sequence(entropy)

    monkeypatch.setattr(baselines.np.random, "SeedSequence", recording_seed_sequence)

    baselines.evaluate_all_baselines(episodes=1, seed=712)

    assert entropy_values == [
        [712, 1, 0],
        [712, 2, 0],
        [712, 3, 0],
        [712, 4, 0],
    ]


def test_evaluator_passes_only_immediate_actor_safe_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Mapping[str, object], float | None, Mapping[str, object] | None]] = []

    class FeedbackPolicy:
        def act(
            self,
            observation: Mapping[str, object],
            previous_reward: float | None,
            previous_info: Mapping[str, object] | None,
        ) -> PitchAction:
            calls.append((observation, previous_reward, previous_info))
            assert "episode_result" not in observation
            if len(calls) == 1:
                assert previous_reward is None
                assert previous_info is None
                return PitchAction.CENT_DOWN
            assert isinstance(previous_reward, float)
            assert previous_info is not None
            assert set(previous_info) == {
                "step_count",
                "steps_remaining",
                "action",
                "action_applied",
                "submitted",
                "submitted_success",
            }
            return PitchAction.SUBMIT

    policy = FeedbackPolicy()
    monkeypatch.setattr(baselines, "RandomPolicy", lambda rng: policy)

    baselines.evaluate_baseline(baselines.BaselineKind.RANDOM, episodes=1, seed=4)

    assert len(calls) == 2


def test_summary_metrics_use_exact_episode_and_action_denominators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = _metric_results()
    fake_env = _ScriptedResultEnv(results)
    monkeypatch.setattr(baselines.gymnasium, "make", lambda environment_id: fake_env)

    summary = baselines.evaluate_baseline(baselines.BaselineKind.RANDOM, episodes=3, seed=19)

    assert summary.submitted_success_rate == pytest.approx(1 / 3)
    assert summary.submitted_within_1_cent_rate == pytest.approx(1 / 3)
    assert summary.final_within_5_cents_rate == pytest.approx(2 / 3)
    assert summary.final_within_1_cent_rate == pytest.approx(2 / 3)
    assert summary.mean_absolute_final_error_cents == pytest.approx(20 / 3)
    assert summary.mean_actions == pytest.approx(76 / 3)
    assert summary.mean_excess_actions == 5.0
    successful_return = 1.0 + 10 / 6_100 - 10 * 0.00001
    exhausted_return = 2_400 / 6_100 - 62 * 0.00001 - 2 * 0.01 - 1.0
    assert summary.mean_return == pytest.approx((successful_return + exhausted_return - 1.0) / 3)
    assert summary.truncation_rate == pytest.approx(1 / 3)
    assert summary.invalid_action_rate == pytest.approx(2 / 76)


def test_summary_uses_null_not_nan_when_no_submission_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_env = _ScriptedResultEnv([_metric_results()[-1]])
    monkeypatch.setattr(baselines.gymnasium, "make", lambda environment_id: fake_env)

    summary = baselines.evaluate_baseline(baselines.BaselineKind.RANDOM, episodes=1, seed=8)

    assert summary.mean_excess_actions is None
    assert "NaN" not in json.dumps(summary.to_dict(), allow_nan=False)


def _run_reward_episode(target_note_index: int, source_pitch_cents: int) -> EpisodeResult:
    env = SinePitchEnv(ObservationMode.REWARD_ONLY)
    observation, _ = env.reset(
        options={
            "target_note_index": target_note_index,
            "source_pitch_cents": source_pitch_cents,
        }
    )
    policy = RewardSearchPolicy()
    reward: float | None = None
    info: Mapping[str, object] | None = None
    done = False
    while not done:
        action = policy.act(_NoPrivateCapability(observation), reward, info)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    assert truncated is False
    return env.episode_result


def _respond(
    policy: RewardSearchPolicy,
    *,
    reward: float,
    applied: bool = True,
    steps_remaining: int = 63,
    controls: tuple[int, int, int] = (0, 0, 0),
) -> PitchAction:
    return policy.act(
        _reward_observation(controls=controls, steps_remaining=steps_remaining),
        reward,
        {
            "step_count": 1,
            "steps_remaining": steps_remaining,
            "action": "test action",
            "action_applied": applied,
            "submitted": False,
            "submitted_success": False,
        },
    )


def _reward_observation(
    *, controls: tuple[int, int, int] = (0, 0, 0), steps_remaining: int = 64
) -> _NoPrivateCapability:
    return _NoPrivateCapability(
        target_note=np.int64(12),
        controls=np.array(controls, dtype=np.int16),
        steps_remaining=np.int64(steps_remaining),
        spectrum=_POISON,
        current_pitch_coordinate=_POISON,
        source_pitch_cents=_POISON,
        episode_result=_POISON,
    )


class _ScriptedResultEnv:
    def __init__(self, results: list[EpisodeResult]) -> None:
        self._results = results
        self._episode_index = -1
        self._current_result: EpisodeResult | None = None

    @property
    def unwrapped(self) -> _ScriptedResultEnv:
        return self

    @property
    def episode_result(self) -> EpisodeResult:
        if self._current_result is None:
            raise AssertionError("evaluator read episode_result before done")
        return self._current_result

    def reset(self, *, seed: int):
        self._episode_index += 1
        self._current_result = None
        return self._observation(), {"step_count": 0, "steps_remaining": 64}

    def step(self, action: int):
        result = self._results[self._episode_index]
        self._current_result = result
        truncated = result.terminal_reason is TerminalReason.BUDGET_EXHAUSTED
        info = {
            "step_count": 1,
            "steps_remaining": 63,
            "action": PitchAction(action).label,
            "action_applied": False,
            "submitted": action == PitchAction.SUBMIT,
            "submitted_success": result.submitted_success,
        }
        return self._observation(), 0.0, not truncated, truncated, info

    def close(self) -> None:
        return None

    @staticmethod
    def _observation() -> dict[str, object]:
        return {
            "target_note": np.int64(12),
            "controls": np.zeros(3, dtype=np.int16),
            "steps_remaining": np.int64(64),
            "spectrum": np.zeros(LOG_SPECTRUM_SIZE, dtype=np.float32),
        }


def _metric_results() -> list[EpisodeResult]:
    successful_actions = (PitchAction.CENT_DOWN,) * 10 + (PitchAction.SUBMIT,)
    exhausted_actions = (
        PitchAction.OCTAVE_DOWN,
        PitchAction.OCTAVE_DOWN,
        PitchAction.OCTAVE_DOWN,
        *((PitchAction.CENT_DOWN, PitchAction.CENT_UP) * 30),
        PitchAction.OCTAVE_DOWN,
    )
    return [
        EpisodeResult(
            source_pitch_cents=6_010,
            target_note_index=12,
            target_pitch_cents=6_000,
            final_pitch_cents=6_000,
            initial_signed_error_cents=10,
            initial_absolute_error_cents=10,
            final_signed_error_cents=0,
            final_absolute_error_cents=0,
            submitted_success=True,
            within_5_cents=True,
            within_1_cent=True,
            actions=successful_actions,
            invalid_action_count=0,
            optimal_actions=(PitchAction.CENT_DOWN,) * 5 + (PitchAction.SUBMIT,),
            excess_actions=5,
            total_return=1.0 + 10 / 6_100 - 10 * 0.00001,
            terminal_reason=TerminalReason.SUBMITTED_SUCCESS,
        ),
        EpisodeResult(
            source_pitch_cents=7_200,
            target_note_index=0,
            target_pitch_cents=4_800,
            final_pitch_cents=4_800,
            initial_signed_error_cents=2_400,
            initial_absolute_error_cents=2_400,
            final_signed_error_cents=0,
            final_absolute_error_cents=0,
            submitted_success=False,
            within_5_cents=True,
            within_1_cent=True,
            actions=exhausted_actions,
            invalid_action_count=2,
            optimal_actions=(
                PitchAction.OCTAVE_DOWN,
                PitchAction.OCTAVE_DOWN,
                PitchAction.SUBMIT,
            ),
            excess_actions=None,
            total_return=2_400 / 6_100 - 62 * 0.00001 - 2 * 0.01 - 1.0,
            terminal_reason=TerminalReason.BUDGET_EXHAUSTED,
        ),
        EpisodeResult(
            source_pitch_cents=6_020,
            target_note_index=12,
            target_pitch_cents=6_000,
            final_pitch_cents=6_020,
            initial_signed_error_cents=20,
            initial_absolute_error_cents=20,
            final_signed_error_cents=20,
            final_absolute_error_cents=20,
            submitted_success=False,
            within_5_cents=False,
            within_1_cent=False,
            actions=(PitchAction.SUBMIT,),
            invalid_action_count=0,
            optimal_actions=(PitchAction.CENT_DOWN,) * 15 + (PitchAction.SUBMIT,),
            excess_actions=None,
            total_return=-1.0,
            terminal_reason=TerminalReason.SUBMITTED_FAILURE,
        ),
    ]


class _NoPrivateCapability(dict[str, object]):
    _FORBIDDEN: ClassVar[set[str]] = {
        "spectrum",
        "current_pitch_coordinate",
        "source_pitch_cents",
        "episode_result",
    }

    def __getitem__(self, key: str) -> object:
        if key in self._FORBIDDEN:
            raise AssertionError(f"policy accessed forbidden capability: {key}")
        return super().__getitem__(key)


_POISON = object()
