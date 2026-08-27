"""Narrow learned-actor inference boundary contracts."""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Mapping

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

import harpy.learning.actors as actors
from harpy.envs.models import PitchAction
from harpy.envs.sine_pitch import SinePitchEnv
from harpy.learning.actors import (
    BC_ACTOR_SEMANTICS_ID,
    MASKED_BC_ACTOR_SEMANTICS_ID,
    PPO_ACTOR_SEMANTICS_ID,
    Actor,
    BCActor,
    MaskedBCActor,
    PPOActor,
)
from harpy.learning.errors import LearningContractError, LearningExecutionError
from harpy.learning.network import HarpySineFeaturesExtractor
from harpy.learning.observations import PolicyObservationWrapper


def valid_raw_observation() -> dict[str, object]:
    return {
        "spectrum": np.linspace(0, 1, 1_961, dtype=np.float32),
        "target_note": np.int64(24),
        "controls": np.array([-2, 12, -100], dtype=np.int16),
        "steps_remaining": np.int64(32),
    }


class ConstantLogitBC(torch.nn.Module):
    """A real minimal network whose intentionally tied logits expose argmax ordering."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)
        self.last_observation: dict[str, torch.Tensor] | None = None

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        self.last_observation = observations
        if self.logits.ndim != 1:
            return self.logits
        return self.logits.expand(observations["state"].shape[0], -1)


class MutatingLogitBC(ConstantLogitBC):
    """Mutate caller-owned controls during inference to probe snapshot ownership."""

    def __init__(self, logits: torch.Tensor, controls: np.ndarray) -> None:
        super().__init__(logits)
        self.controls = controls

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        self.controls[:] = (2, 0, 0)
        return super().forward(observations)


class FixedPredictionPPO:
    """A minimal prediction boundary that returns a controlled SB3-shaped action."""

    def __init__(self, action: object) -> None:
        self.action = action
        self.observations: list[dict[str, np.ndarray]] = []
        self.deterministic_values: list[bool] = []

    def predict(
        self, observation: dict[str, np.ndarray], *, deterministic: bool
    ) -> tuple[object, None]:
        self.observations.append(observation)
        self.deterministic_values.append(deterministic)
        return self.action, None


def test_actor_protocol_and_learned_actor_signatures_accept_observation_only() -> None:
    assert tuple(inspect.signature(Actor.act).parameters) == ("self", "observation")
    assert tuple(inspect.signature(BCActor.act).parameters) == ("self", "observation")
    assert tuple(inspect.signature(MaskedBCActor.act).parameters) == ("self", "observation")
    assert tuple(inspect.signature(PPOActor.act).parameters) == ("self", "observation")


def test_actor_semantics_ids_are_stable_and_keep_masked_bc_distinct() -> None:
    assert BC_ACTOR_SEMANTICS_ID == "harpy-sine-policy-bc-v1"
    assert MASKED_BC_ACTOR_SEMANTICS_ID == "harpy-sine-policy-bc-public-bound-mask-v1"
    assert PPO_ACTOR_SEMANTICS_ID == "harpy-sine-policy-ppo-v1"
    assert (
        len(
            {
                BC_ACTOR_SEMANTICS_ID,
                MASKED_BC_ACTOR_SEMANTICS_ID,
                PPO_ACTOR_SEMANTICS_ID,
            }
        )
        == 3
    )


def test_bc_actor_uses_first_argmax_on_a_tie() -> None:
    model = ConstantLogitBC(torch.zeros(7))
    actor = BCActor(model, device=torch.device("cpu"))

    assert isinstance(actor, Actor)
    assert actor.act(valid_raw_observation()) is PitchAction.OCTAVE_DOWN
    assert model.last_observation is not None
    assert tuple(model.last_observation) == ("spectrum", "state")
    assert model.last_observation["spectrum"].shape == (1, 1_961)
    assert model.last_observation["state"].shape == (1, 5)
    assert model.last_observation["spectrum"].dtype is torch.float32
    assert model.logits.device.type == "cpu"
    assert not model.training


def test_bc_actor_validates_raw_observations_before_model_inference() -> None:
    raw = valid_raw_observation()
    raw["source_pitch_cents"] = 6_000
    actor = BCActor(ConstantLogitBC(torch.zeros(7)))

    with pytest.raises(LearningContractError, match="exactly"):
        actor.act(raw)


@pytest.mark.parametrize(
    "logits",
    [
        torch.zeros(6),
        torch.zeros(8),
        torch.zeros((1, 7, 1)),
        torch.full((7,), float("nan")),
    ],
)
def test_bc_actor_rejects_malformed_model_logits(logits: torch.Tensor) -> None:
    actor = BCActor(ConstantLogitBC(logits))

    with pytest.raises(LearningExecutionError, match="logits"):
        actor.act(valid_raw_observation())


@pytest.mark.parametrize(
    ("controls", "logits", "expected"),
    [
        ((-2, -12, -100), (9, 8, 7, 6, 5, 4, 3), PitchAction.SUBMIT),
        ((2, 12, 100), (0, 1, 2, 3, 4, 9, 10), PitchAction.SUBMIT),
        ((0, 0, 0), (0, 0, 0, 9, 0, 0, 0), PitchAction.SUBMIT),
    ],
)
def test_masked_bc_actor_masks_bounds_and_never_masks_submit(
    controls: tuple[int, int, int],
    logits: tuple[int, ...],
    expected: PitchAction,
) -> None:
    raw = valid_raw_observation()
    raw["controls"] = np.asarray(controls, dtype=np.int16)
    actor = MaskedBCActor(ConstantLogitBC(torch.tensor(logits, dtype=torch.float32)))

    assert actor.act(raw) is expected


def test_masked_bc_actor_skips_an_illegal_first_max_without_changing_legacy_bc() -> None:
    raw = valid_raw_observation()
    raw["controls"] = np.array([-2, 0, 0], dtype=np.int16)
    legacy = BCActor(ConstantLogitBC(torch.zeros(7)))
    masked = MaskedBCActor(ConstantLogitBC(torch.zeros(7)))

    assert legacy.act(raw) is PitchAction.OCTAVE_DOWN
    assert masked.act(raw) is PitchAction.SEMITONE_DOWN


@pytest.mark.parametrize(
    "logits",
    [
        torch.zeros(6),
        torch.zeros(8),
        torch.zeros((1, 7, 1)),
        torch.full((7,), float("nan")),
    ],
)
def test_masked_bc_actor_rejects_logits_before_deriving_a_mask(
    logits: torch.Tensor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        actors,
        "legal_action_mask",
        lambda controls: pytest.fail(f"malformed logits reached the mask: {controls}"),
    )
    actor = MaskedBCActor(ConstantLogitBC(logits))

    with pytest.raises(LearningExecutionError, match="logits"):
        actor.act(valid_raw_observation())


def test_masked_bc_actor_validates_the_full_raw_observation_before_inference() -> None:
    raw = valid_raw_observation()
    raw["hidden_pitch"] = 6_000
    model = ConstantLogitBC(torch.zeros(7))
    actor = MaskedBCActor(model)

    with pytest.raises(LearningContractError, match="exactly"):
        actor.act(raw)
    assert model.last_observation is None


class StatefulObservation(Mapping[str, object]):
    """Return conflicting controls on a second read to expose split-snapshot actors."""

    def __init__(self) -> None:
        self._values = valid_raw_observation()
        self.control_reads = 0

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, key: str) -> object:
        if key != "controls":
            return self._values[key]
        self.control_reads += 1
        if self.control_reads == 1:
            return np.array([-2, 0, 0], dtype=np.int16)
        return np.array([2, 0, 0], dtype=np.int16)


def test_masked_bc_actor_uses_one_owned_snapshot_for_inference_and_masking() -> None:
    observation = StatefulObservation()
    model = ConstantLogitBC(torch.tensor([9, 0, 0, 0, 0, 0, 8], dtype=torch.float32))
    actor = MaskedBCActor(model)

    assert actor.act(observation) is PitchAction.OCTAVE_UP
    assert observation.control_reads == 1


def test_masked_bc_actor_owns_controls_before_model_mutates_callers_array() -> None:
    observation = valid_raw_observation()
    controls = np.array([-2, 0, 0], dtype=np.int16)
    observation["controls"] = controls
    model = MutatingLogitBC(torch.tensor([9, 0, 0, 0, 0, 0, 8], dtype=torch.float32), controls)
    actor = MaskedBCActor(model)

    assert actor.act(observation) is PitchAction.OCTAVE_UP
    assert tuple(controls) == (2, 0, 0)


@pytest.mark.parametrize(
    "logits",
    [
        torch.zeros(7, dtype=torch.int64),
        torch.zeros(7, dtype=torch.bool),
        torch.zeros(7, dtype=torch.complex64),
    ],
)
def test_masked_bc_actor_rejects_nonfloating_logits_before_mask_or_argmax(
    logits: torch.Tensor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        actors,
        "legal_action_mask",
        lambda controls: pytest.fail(f"invalid dtype reached the mask: {controls}"),
    )
    actor = MaskedBCActor(ConstantLogitBC(logits))

    with pytest.raises(LearningExecutionError, match="floating-point"):
        actor.act(valid_raw_observation())


def test_ppo_actor_passes_the_transformed_observation_and_uses_deterministic_prediction() -> None:
    model = FixedPredictionPPO(np.array(6, dtype=np.int64))
    actor = PPOActor(model)

    assert isinstance(actor, Actor)
    assert actor.act(valid_raw_observation()) is PitchAction.OCTAVE_UP
    assert model.deterministic_values == [True]
    assert tuple(model.observations[0]) == ("spectrum", "state")
    assert model.observations[0]["spectrum"].shape == (1_961,)
    assert model.observations[0]["state"].shape == (5,)
    assert model.observations[0]["spectrum"].dtype == np.float32


@pytest.mark.parametrize(
    "action",
    [
        np.array([0], dtype=np.int64),
        np.array([[0]], dtype=np.int64),
        np.array(0.0, dtype=np.float32),
        np.array(True),
        np.array(-1, dtype=np.int64),
        np.array(7, dtype=np.int64),
        np.int64(0),
        0,
        False,
    ],
)
def test_ppo_actor_rejects_non_scalar_or_invalid_prediction_actions(action: object) -> None:
    actor = PPOActor(FixedPredictionPPO(action))

    with pytest.raises(LearningExecutionError, match="action"):
        actor.act(valid_raw_observation())


def test_ppo_actor_accepts_an_actual_sb3_unvectorized_discrete_prediction() -> None:
    environment = PolicyObservationWrapper(SinePitchEnv())
    model = PPO(
        "MultiInputPolicy",
        environment,
        device="cpu",
        n_steps=2,
        batch_size=2,
        policy_kwargs={
            "features_extractor_class": HarpySineFeaturesExtractor,
            "net_arch": [],
        },
    )
    actor = PPOActor(model)

    action = actor.act(valid_raw_observation())

    assert isinstance(action, PitchAction)
    assert model.device.type == "cpu"
