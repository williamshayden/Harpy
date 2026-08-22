"""Capability-isolated baseline plans derived only from public observations."""

from __future__ import annotations

from inspect import signature
from typing import ClassVar

import numpy as np
import pytest

from harpy.envs.baselines import BaselineKind, oracle_plan, spectrum_peak_plan
from harpy.envs.models import ControlState, ObservationMode, PitchAction
from harpy.envs.sine_pitch import SinePitchEnv
from harpy.envs.spectrum import LOG_SPECTRUM_SIZE


@pytest.mark.parametrize(
    ("kind", "mode", "environment_id"),
    [
        (BaselineKind.RANDOM, ObservationMode.SPECTRUM, "Harpy/SinePitch-v0"),
        (BaselineKind.SPECTRUM_PEAK, ObservationMode.SPECTRUM, "Harpy/SinePitch-v0"),
        (BaselineKind.ORACLE, ObservationMode.ORACLE, "Harpy/SinePitchOracle-v0"),
        (
            BaselineKind.REWARD_SEARCH,
            ObservationMode.REWARD_ONLY,
            "Harpy/SinePitchRewardOnly-v0",
        ),
    ],
)
def test_each_baseline_has_one_declared_lane(
    kind: BaselineKind, mode: ObservationMode, environment_id: str
) -> None:
    assert kind.observation_mode is mode
    assert kind.environment_id == environment_id


def test_baseline_kind_is_exhaustive_and_uses_stable_lane_ids() -> None:
    assert [(kind.name, kind.value) for kind in BaselineKind] == [
        ("RANDOM", "random"),
        ("SPECTRUM_PEAK", "spectrum_peak"),
        ("ORACLE", "oracle"),
        ("REWARD_SEARCH", "reward_search"),
    ]


def test_planners_accept_only_one_observation_capability() -> None:
    assert tuple(signature(oracle_plan).parameters) == ("observation",)
    assert tuple(signature(spectrum_peak_plan).parameters) == ("observation",)


def test_oracle_submits_success_for_every_initial_error() -> None:
    for initial_error_cents in range(-2_400, 2_401):
        observation = {
            "current_pitch_coordinate": np.array(
                [(6_000 + initial_error_cents) / 100.0], dtype=np.float32
            ),
            "target_note": np.int64(12),
            "controls": np.zeros(3, dtype=np.int16),
            "steps_remaining": np.int64(64),
        }

        plan = oracle_plan(observation)

        assert isinstance(plan, tuple)
        assert _final_error(initial_error_cents, ControlState(), plan) in range(-5, 6)


def test_oracle_rounds_float32_coordinate_and_subtracts_visible_controls() -> None:
    rounding_observation = {
        "current_pitch_coordinate": np.array([47.94], dtype=np.float32),
        "target_note": np.int64(0),
        "controls": np.zeros(3, dtype=np.int16),
        "steps_remaining": np.int64(64),
    }
    controlled_observation = {
        "current_pitch_coordinate": np.array([39.85], dtype=np.float32),
        "target_note": np.int64(12),
        "controls": np.array([-1, 4, 90], dtype=np.int16),
        "steps_remaining": np.int64(64),
    }

    assert oracle_plan(rounding_observation) == (
        PitchAction.CENT_UP,
        PitchAction.SUBMIT,
    )
    controlled_plan = oracle_plan(controlled_observation)
    assert _final_error(-1_305, ControlState(-1, 4, 90), controlled_plan) in range(-5, 6)


def test_spectrum_peak_reads_no_oracle_error_reward_or_analyzer_peak_capability() -> None:
    spectrum = np.zeros(LOG_SPECTRUM_SIZE, dtype=np.float32)
    spectrum[(6_025 - 1_100) // 5] = 1.0
    observation = _CapabilityPoisonedObservation(
        spectrum=spectrum,
        target_note=np.int64(12),
        controls=np.array([0, 0, 100], dtype=np.int16),
        steps_remaining=np.int64(64),
        current_pitch_coordinate=_POISON,
        signed_error_cents=_POISON,
        reward=_POISON,
        episode_result=_POISON,
        analyzer_peak_hz=_POISON,
    )

    plan = spectrum_peak_plan(observation)

    assert plan == (PitchAction.CENT_DOWN,) * 25 + (PitchAction.SUBMIT,)


def test_spectrum_peak_maps_only_first_argmax_through_the_five_cent_grid() -> None:
    spectrum = np.zeros(LOG_SPECTRUM_SIZE, dtype=np.float32)
    spectrum[(5_900 - 1_100) // 5] = 0.75
    spectrum[(6_100 - 1_100) // 5] = 0.75
    observation = {
        "spectrum": spectrum,
        "target_note": np.int64(12),
        "controls": np.zeros(3, dtype=np.int16),
        "steps_remaining": np.int64(64),
    }

    assert spectrum_peak_plan(observation) == (
        PitchAction.SEMITONE_UP,
        PitchAction.SUBMIT,
    )


def test_spectrum_peak_clamps_quantization_above_the_physical_source_boundary() -> None:
    spectrum = np.zeros(LOG_SPECTRUM_SIZE, dtype=np.float32)
    spectrum[(7_205 - 1_100) // 5] = 1.0
    observation = {
        "spectrum": spectrum,
        "target_note": np.int64(0),
        "controls": np.array([0, 0, 3], dtype=np.int16),
        "steps_remaining": np.int64(64),
    }

    plan = spectrum_peak_plan(observation)

    assert _final_error(2_400, ControlState(cents=3), plan) == 0


def test_spectrum_peak_clamps_quantization_below_the_physical_source_boundary() -> None:
    spectrum = np.zeros(LOG_SPECTRUM_SIZE, dtype=np.float32)
    spectrum[(4_795 - 1_100) // 5] = 1.0
    observation = {
        "spectrum": spectrum,
        "target_note": np.int64(24),
        "controls": np.array([0, 0, -3], dtype=np.int16),
        "steps_remaining": np.int64(64),
    }

    plan = spectrum_peak_plan(observation)

    assert _final_error(-2_400, ControlState(cents=-3), plan) == 0


@pytest.mark.parametrize(
    ("mode", "planner", "target_note_index", "source_pitch_cents"),
    [
        (ObservationMode.ORACLE, oracle_plan, 24, 4_800),
        (ObservationMode.ORACLE, oracle_plan, 0, 7_200),
        (ObservationMode.SPECTRUM, spectrum_peak_plan, 12, 5_897),
    ],
)
def test_selected_real_observations_produce_one_successful_queued_plan(
    mode: ObservationMode,
    planner: object,
    target_note_index: int,
    source_pitch_cents: int,
) -> None:
    env = SinePitchEnv(mode)
    observation, _ = env.reset(
        options={
            "target_note_index": target_note_index,
            "source_pitch_cents": source_pitch_cents,
        }
    )

    plan = planner(observation)  # type: ignore[operator]
    assert isinstance(plan, tuple)
    for action in plan:
        _, _, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert truncated is False
    assert info["submitted_success"] is True


def _final_error(
    base_error_cents: int, controls: ControlState, plan: tuple[PitchAction, ...]
) -> int:
    assert plan
    assert plan[-1] is PitchAction.SUBMIT
    assert PitchAction.SUBMIT not in plan[:-1]
    final_controls = controls
    for action in plan[:-1]:
        final_controls, applied = final_controls.apply(action)
        assert applied
    return base_error_cents + final_controls.offset_cents


class _CapabilityPoisonedObservation(dict[str, object]):
    _FORBIDDEN: ClassVar[set[str]] = {
        "current_pitch_coordinate",
        "signed_error_cents",
        "reward",
        "episode_result",
        "analyzer_peak_hz",
    }

    def __getitem__(self, key: str) -> object:
        if key in self._FORBIDDEN:
            raise AssertionError(f"planner accessed forbidden capability: {key}")
        return super().__getitem__(key)


_POISON = object()
