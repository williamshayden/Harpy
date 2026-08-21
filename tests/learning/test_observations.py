"""Strict raw-to-policy observation boundary tests."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from harpy.envs.sine_pitch import SinePitchEnv
from harpy.learning.errors import LearningContractError
from harpy.learning.models import PREPROCESSING_SCHEMA_ID
from harpy.learning.observations import (
    POLICY_OBSERVATION_SPACE,
    PolicyObservationWrapper,
    preprocess_observation,
)


def valid_raw_observation() -> dict[str, object]:
    return {
        "spectrum": np.linspace(0, 1, 1_961, dtype=np.float32),
        "target_note": np.int64(24),
        "controls": np.array([-2, 12, -100], dtype=np.int16),
        "steps_remaining": np.int64(32),
    }


def test_preprocessing_has_exact_two_box_values() -> None:
    raw = valid_raw_observation()

    policy = preprocess_observation(raw)

    assert tuple(policy) == ("spectrum", "state")
    np.testing.assert_array_equal(
        policy["state"], np.array([1.0, -1.0, 1.0, -1.0, 0.5], dtype=np.float32)
    )
    assert policy["spectrum"].dtype == np.float32
    assert not np.shares_memory(policy["spectrum"], raw["spectrum"])
    assert POLICY_OBSERVATION_SPACE.contains(policy)


def test_preprocessing_owns_fresh_c_contiguous_arrays() -> None:
    raw = valid_raw_observation()
    raw["spectrum"] = np.linspace(1, 0, 1_961, dtype=np.float32)[::-1]

    first = preprocess_observation(raw)
    second = preprocess_observation(raw)

    for policy in (first, second):
        assert policy["spectrum"].flags.c_contiguous
        assert policy["spectrum"].flags.owndata
        assert policy["state"].flags.c_contiguous
        assert policy["state"].flags.owndata
    assert not np.shares_memory(first["spectrum"], second["spectrum"])
    assert not np.shares_memory(first["state"], second["state"])


def test_preprocessing_accepts_real_frozen_environment_observation() -> None:
    env = SinePitchEnv()
    raw, _ = env.reset(options={"target_note_index": 7, "source_pitch_cents": 5_432})

    policy = preprocess_observation(raw)

    assert POLICY_OBSERVATION_SPACE.contains(policy)
    np.testing.assert_array_equal(
        policy["state"], np.array([-5 / 12, 0, 0, 0, 1], dtype=np.float32)
    )


def test_policy_wrapper_transforms_reset_and_step_observations() -> None:
    wrapped = PolicyObservationWrapper(SinePitchEnv())

    reset_observation, _ = wrapped.reset(
        options={"target_note_index": 12, "source_pitch_cents": 5_994}
    )
    step_observation, _, _, _, _ = wrapped.step(4)

    assert wrapped.observation_space is POLICY_OBSERVATION_SPACE
    assert POLICY_OBSERVATION_SPACE.contains(reset_observation)
    assert POLICY_OBSERVATION_SPACE.contains(step_observation)
    assert tuple(reset_observation) == ("spectrum", "state")
    assert tuple(step_observation) == ("spectrum", "state")


def test_preprocessing_schema_is_reused_from_models() -> None:
    from harpy.learning import observations

    assert observations.PREPROCESSING_SCHEMA_ID is PREPROCESSING_SCHEMA_ID


@pytest.mark.parametrize("missing_key", ["spectrum", "target_note", "controls", "steps_remaining"])
def test_preprocessing_rejects_missing_raw_keys(missing_key: str) -> None:
    raw = valid_raw_observation()
    del raw[missing_key]

    with pytest.raises(LearningContractError, match="exactly"):
        preprocess_observation(raw)


@pytest.mark.parametrize(
    "extra_key",
    [
        "extra",
        "current_pitch_coordinate",
        "source_pitch_cents",
        "final_absolute_error_cents",
    ],
)
def test_preprocessing_rejects_extra_and_truth_alias_keys(extra_key: str) -> None:
    raw = valid_raw_observation()
    raw[extra_key] = 0

    with pytest.raises(LearningContractError, match="exactly"):
        preprocess_observation(raw)


@pytest.mark.parametrize("field", ["target_note", "steps_remaining"])
@pytest.mark.parametrize("value", [0, np.int32(0), np.array(0, dtype=np.int64), True])
def test_preprocessing_requires_numpy_int64_scalars(field: str, value: object) -> None:
    raw = valid_raw_observation()
    raw[field] = value

    with pytest.raises(LearningContractError, match=field):
        preprocess_observation(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target_note", np.int64(-1)),
        ("target_note", np.int64(25)),
        ("steps_remaining", np.int64(-1)),
        ("steps_remaining", np.int64(65)),
        ("controls", np.array([-3, 0, 0], dtype=np.int16)),
        ("controls", np.array([3, 0, 0], dtype=np.int16)),
        ("controls", np.array([0, -13, 0], dtype=np.int16)),
        ("controls", np.array([0, 13, 0], dtype=np.int16)),
        ("controls", np.array([0, 0, -101], dtype=np.int16)),
        ("controls", np.array([0, 0, 101], dtype=np.int16)),
        ("spectrum", np.full(1_961, -0.01, dtype=np.float32)),
        ("spectrum", np.full(1_961, 1.01, dtype=np.float32)),
    ],
)
def test_preprocessing_rejects_every_out_of_bounds_raw_field(field: str, value: object) -> None:
    raw = valid_raw_observation()
    raw[field] = value

    with pytest.raises(LearningContractError, match=field):
        preprocess_observation(raw)


@pytest.mark.parametrize(
    "value",
    [
        [0, 0, 0],
        np.array([0, 0, 0], dtype=np.int32),
        np.array(0, dtype=np.int16),
        np.zeros((1, 3), dtype=np.int16),
        np.zeros(2, dtype=np.int16),
        np.zeros(4, dtype=np.int16),
    ],
)
def test_preprocessing_rejects_wrong_control_type_dtype_rank_or_shape(value: object) -> None:
    raw = valid_raw_observation()
    raw["controls"] = value

    with pytest.raises(LearningContractError, match="controls"):
        preprocess_observation(raw)


@pytest.mark.parametrize(
    "value",
    [
        [0.0] * 1_961,
        np.zeros(1_961, dtype=np.float64),
        np.zeros((1, 1_961), dtype=np.float32),
        np.zeros(1_960, dtype=np.float32),
        np.zeros(1_962, dtype=np.float32),
    ],
)
def test_preprocessing_rejects_wrong_spectrum_type_dtype_rank_or_shape(value: object) -> None:
    raw = valid_raw_observation()
    raw["spectrum"] = value

    with pytest.raises(LearningContractError, match="spectrum"):
        preprocess_observation(raw)


@pytest.mark.parametrize("non_finite", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["spectrum", "controls"])
def test_preprocessing_rejects_non_finite_arrays(field: str, non_finite: float) -> None:
    raw = valid_raw_observation()
    if field == "spectrum":
        value = np.zeros(1_961, dtype=np.float32)
        value[100] = non_finite
    else:
        value = np.zeros(3, dtype=np.float32)
        value[1] = non_finite
    raw[field] = value

    with pytest.raises(LearningContractError, match=field):
        preprocess_observation(raw)


def test_preprocessing_requires_a_mapping() -> None:
    with pytest.raises(LearningContractError, match="mapping"):
        preprocess_observation([("spectrum", np.zeros(1_961, dtype=np.float32))])


def test_preprocessing_accepts_mapping_implementations() -> None:
    class RawMapping(Mapping[str, object]):
        def __init__(self, values: dict[str, object]) -> None:
            self._values = values

        def __getitem__(self, key: str) -> object:
            return self._values[key]

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter(self._values)

        def __len__(self) -> int:
            return len(self._values)

    assert POLICY_OBSERVATION_SPACE.contains(
        preprocess_observation(RawMapping(valid_raw_observation()))
    )
