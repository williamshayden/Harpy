"""Stateless estimator-plus-planner actor contracts."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import numpy as np
import pytest
import torch

from harpy.envs.models import ZERO_CONTROLS, ControlState, PitchAction
from harpy.envs.planning import minimum_action_plan
from harpy.learning.errors import LearningContractError, LearningExecutionError
from harpy.learning.pitch_actor import PitchDecision, PitchDecoding, PitchPlannerActor
from harpy.learning.pitch_network import PITCH_CLASS_COUNT, PitchEstimatorNetwork


def raw_observation(
    *,
    target_note: int = 12,
    controls: ControlState = ZERO_CONTROLS,
    spectrum_fill: float = 0.25,
) -> dict[str, object]:
    return {
        "spectrum": np.full(PITCH_CLASS_COUNT, spectrum_fill, dtype=np.float32),
        "target_note": np.int64(target_note),
        "controls": np.array(
            [controls.octaves, controls.semitones, controls.cents], dtype=np.int16
        ),
        "steps_remaining": np.int64(64),
    }


class ScriptedPitchEstimator:
    """Test-only controller that scripts an already validated exact estimator."""

    def __init__(self, outputs: list[torch.Tensor]) -> None:
        self.model = PitchEstimatorNetwork()
        self.outputs = outputs
        self.observed_spectra: list[torch.Tensor] = []

    def actor(self, *, decoding: PitchDecoding = PitchDecoding.GLOBAL_ARGMAX) -> PitchPlannerActor:
        actor = PitchPlannerActor(self.model, decoding=decoding)
        self.model.forward = self._forward  # type: ignore[method-assign]
        return actor

    def _forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        self.observed_spectra.append(spectrum.detach().clone())
        return self.outputs.pop(0).to(spectrum.device)


class ChangingTargetObservation(Mapping[str, object]):
    """Return one valid target, then a different value on any later read."""

    def __init__(self, values: dict[str, object]) -> None:
        self._values = values
        self._target_reads = 0

    def __getitem__(self, key: str) -> object:
        if key == "target_note":
            self._target_reads += 1
            return np.int64(12 if self._target_reads == 1 else 0)
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def logits_for(*maximum_indexes: int) -> torch.Tensor:
    logits = torch.zeros((1, PITCH_CLASS_COUNT), dtype=torch.float32)
    logits[0, list(maximum_indexes)] = 1.0
    return logits


def test_pitch_actor_module_import_is_torch_and_sb3_lazy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, harpy.learning.pitch_actor; "
                "assert 'torch' not in sys.modules; "
                "assert 'stable_baselines3' not in sys.modules"
            ),
        ],
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr.decode()


def test_pitch_decision_is_immutable_and_strictly_typed() -> None:
    decision = PitchDecision(
        action=PitchAction.CENT_UP,
        estimated_candidate_cents=6_005,
    )

    assert decision.action is PitchAction.CENT_UP
    assert decision.estimated_candidate_cents == 6_005
    with pytest.raises(FrozenInstanceError):
        decision.action = PitchAction.SUBMIT  # type: ignore[misc]
    with pytest.raises(ValueError, match="PitchAction"):
        PitchDecision(action=4, estimated_candidate_cents=6_000)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer"):
        PitchDecision(
            action=PitchAction.SUBMIT,
            estimated_candidate_cents=True,  # type: ignore[arg-type]
        )


def test_actor_uses_first_argmax_and_passes_only_owned_spectrum_to_estimator() -> None:
    model = ScriptedPitchEstimator([logits_for(4, 5)])
    actor = model.actor()
    observation = raw_observation(target_note=0, spectrum_fill=0.75)

    decision = actor.decide(observation)

    assert decision.estimated_candidate_cents == 1_120
    assert len(model.observed_spectra) == 1
    assert model.observed_spectra[0].shape == (1, PITCH_CLASS_COUNT)
    assert model.observed_spectra[0].dtype is torch.float32
    assert torch.all(model.observed_spectra[0] == 0.75)
    assert (
        model.observed_spectra[0].data_ptr()
        != observation["spectrum"].__array_interface__[  # type: ignore[union-attr]
            "data"
        ][0]
    )


def test_default_decoding_preserves_the_released_actor_identity_and_decision() -> None:
    logits = logits_for(0, 980)
    default = PitchPlannerActor(PitchEstimatorNetwork())
    default._model.forward = lambda _spectrum: logits  # type: ignore[method-assign]
    explicit = ScriptedPitchEstimator([logits]).actor(decoding=PitchDecoding.GLOBAL_ARGMAX)

    assert default.decoding is PitchDecoding.GLOBAL_ARGMAX
    assert default.semantics_id == "harpy-sine-pitch-estimator-planner-v1"
    assert default.decide(raw_observation()) == explicit.decide(raw_observation())


@pytest.mark.parametrize("decoding", list(PitchDecoding))
def test_decoding_identity_is_explicit_and_read_only(decoding: PitchDecoding) -> None:
    actor = ScriptedPitchEstimator([]).actor(decoding=decoding)

    assert actor.decoding is decoding
    assert actor.semantics_id == decoding.semantics_id
    if decoding is PitchDecoding.FEASIBLE_ARGMAX:
        assert actor.semantics_id == "harpy-sine-pitch-feasible-estimator-planner-v1"
    with pytest.raises(AttributeError):
        actor.decoding = PitchDecoding.GLOBAL_ARGMAX  # type: ignore[misc]


@pytest.mark.parametrize("decoding", ["global-argmax", "feasible-argmax", None, True])
def test_actor_rejects_untyped_decoding(decoding: object) -> None:
    with pytest.raises(LearningContractError, match="PitchDecoding"):
        PitchPlannerActor(object(), decoding=decoding)  # type: ignore[arg-type]


@pytest.mark.parametrize("impossible_cents", [1_100, 10_900])
def test_feasible_decoding_selects_the_best_possible_logit_without_clamping_or_mutation(
    impossible_cents: int,
) -> None:
    logits = logits_for((6_600 - 1_100) // 5)
    logits[0, (impossible_cents - 1_100) // 5] = 2.0
    original = logits.clone()
    actor = ScriptedPitchEstimator([logits]).actor(decoding=PitchDecoding.FEASIBLE_ARGMAX)

    decision = actor.decide(raw_observation(target_note=18))

    assert decision == PitchDecision(PitchAction.SUBMIT, 6_600)
    assert torch.equal(logits, original)


@pytest.mark.parametrize(
    ("controls", "lower_cents", "upper_cents"),
    [
        (ControlState(cents=2), 4_800, 7_200),
        (ControlState(cents=3), 4_805, 7_205),
        (ControlState(cents=-2), 4_800, 7_200),
        (ControlState(cents=-3), 4_795, 7_195),
    ],
)
@pytest.mark.parametrize("boundary", ["lower", "upper"])
def test_feasible_decoding_includes_quantized_endpoints_and_excludes_the_next_class(
    controls: ControlState,
    lower_cents: int,
    upper_cents: int,
    boundary: str,
) -> None:
    endpoint = lower_cents if boundary == "lower" else upper_cents
    impossible_neighbor = endpoint + (-5 if boundary == "lower" else 5)
    logits = logits_for((endpoint - 1_100) // 5)
    logits[0, (impossible_neighbor - 1_100) // 5] = 2.0
    actor = ScriptedPitchEstimator([logits]).actor(decoding=PitchDecoding.FEASIBLE_ARGMAX)

    assert actor.decide(raw_observation(controls=controls)).estimated_candidate_cents == endpoint


@pytest.mark.parametrize(
    ("controls", "estimated_cents"),
    [
        (ControlState(octaves=-2, semitones=-12, cents=-100), 1_100),
        (ControlState(octaves=2, semitones=12, cents=100), 10_900),
    ],
)
def test_feasible_decoding_retains_valid_extreme_grid_classes(
    controls: ControlState, estimated_cents: int
) -> None:
    actor = ScriptedPitchEstimator([logits_for((estimated_cents - 1_100) // 5)]).actor(
        decoding=PitchDecoding.FEASIBLE_ARGMAX
    )

    decision = actor.decide(raw_observation(controls=controls))

    assert decision.estimated_candidate_cents == estimated_cents


def test_feasible_decoding_keeps_the_first_maximum_inside_the_possible_interval() -> None:
    actor = ScriptedPitchEstimator([logits_for(0, 740, 741, 1_960)]).actor(
        decoding=PitchDecoding.FEASIBLE_ARGMAX
    )

    assert actor.decide(raw_observation()).estimated_candidate_cents == 4_800


def test_feasible_decoding_uses_only_public_source_bounds_and_total_control_offset() -> None:
    logits = logits_for((8_200 - 1_100) // 5)
    logits[0, 0] = 2.0
    actor = ScriptedPitchEstimator([logits, logits, logits]).actor(
        decoding=PitchDecoding.FEASIBLE_ARGMAX
    )
    observations = [
        raw_observation(target_note=0, controls=ControlState(octaves=1)),
        raw_observation(target_note=24, controls=ControlState(semitones=12)),
        raw_observation(target_note=12, controls=ControlState(octaves=1, semitones=1, cents=-100)),
    ]
    observations[-1]["steps_remaining"] = np.int64(1)

    assert [actor.decide(obs).estimated_candidate_cents for obs in observations] == [8_200] * 3


def test_feasible_decoding_recomputes_the_interval_after_controls_change() -> None:
    logits = logits_for(0)
    logits[0, (6_100 - 1_100) // 5] = 2.0
    actor = ScriptedPitchEstimator([logits, logits]).actor(decoding=PitchDecoding.FEASIBLE_ARGMAX)

    initial = actor.decide(raw_observation())
    shifted = actor.decide(
        raw_observation(controls=ControlState(octaves=-2, semitones=-12, cents=-100))
    )

    assert initial.estimated_candidate_cents == 6_100
    assert shifted.estimated_candidate_cents == 1_100


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_feasible_decoding_rejects_nonfinite_logits_even_outside_the_possible_interval(
    nonfinite: float,
) -> None:
    logits = logits_for(980)
    logits[0, 0] = nonfinite
    actor = ScriptedPitchEstimator([logits]).actor(decoding=PitchDecoding.FEASIBLE_ARGMAX)

    with pytest.raises(LearningExecutionError, match="finite"):
        actor.decide(raw_observation())


@pytest.mark.parametrize("decoding", list(PitchDecoding))
def test_actor_validates_full_raw_observation_before_estimator_inference(
    decoding: PitchDecoding,
) -> None:
    model = ScriptedPitchEstimator([logits_for(980)])
    actor = model.actor(decoding=decoding)
    observation = raw_observation()
    observation["source_pitch_cents"] = 6_000

    with pytest.raises(LearningContractError, match="exactly"):
        actor.decide(observation)
    assert model.observed_spectra == []


@pytest.mark.parametrize("decoding", list(PitchDecoding))
def test_actor_plans_from_one_owned_validated_observation_snapshot(decoding: PitchDecoding) -> None:
    model = ScriptedPitchEstimator([logits_for((6_000 - 1_100) // 5)])
    actor = model.actor(decoding=decoding)
    observation = ChangingTargetObservation(raw_observation(target_note=12))

    assert actor.act(observation) is PitchAction.SUBMIT


def test_actor_reconstructs_public_arithmetic_clamps_and_plans_at_zero_tolerance() -> None:
    controls = ControlState(octaves=-2, semitones=-12, cents=-100)
    model = ScriptedPitchEstimator([logits_for(1_960)])
    actor = model.actor()
    planned = (PitchAction.CENT_UP, PitchAction.SUBMIT)

    with (
        patch("harpy.learning.pitch_actor.minimum_action_plan", return_value=planned) as planner,
        patch(
            "harpy.learning.pitch_actor.legal_action_mask",
            return_value=(True,) * len(PitchAction),
        ) as mask,
    ):
        decision = actor.decide(raw_observation(target_note=0, controls=controls))

    assert decision == PitchDecision(PitchAction.CENT_UP, 10_900)
    planner.assert_called_once_with(2_400, controls, tolerance_cents=0)
    mask.assert_called_once_with(controls)


@pytest.mark.parametrize(
    ("estimated_cents", "target_note", "controls"),
    [
        (6_000, 12, ControlState()),
        (6_305, 12, ControlState(octaves=1, semitones=-2, cents=3)),
        (4_800, 24, ControlState(octaves=-2, semitones=1, cents=-7)),
        (10_900, 0, ControlState(octaves=-2, semitones=-12, cents=-100)),
        (1_100, 24, ControlState(octaves=2, semitones=12, cents=100)),
    ],
)
def test_actor_returns_the_first_real_planned_action_and_it_is_legal(
    estimated_cents: int,
    target_note: int,
    controls: ControlState,
) -> None:
    index = (estimated_cents - 1_100) // 5
    actor = ScriptedPitchEstimator([logits_for(index)]).actor()
    raw_base_error = estimated_cents - controls.offset_cents - (4_800 + 100 * target_note)
    expected_base_error = min(2_400, max(-2_400, raw_base_error))
    expected = minimum_action_plan(expected_base_error, controls, tolerance_cents=0)[0]

    assert actor.act(raw_observation(target_note=target_note, controls=controls)) is expected
    if expected is not PitchAction.SUBMIT:
        _updated, applied = controls.apply(expected)
        assert applied


@pytest.mark.parametrize("decoding", list(PitchDecoding))
def test_actor_reestimates_on_every_call_and_keeps_no_plan_cursor(decoding: PitchDecoding) -> None:
    first = logits_for((6_100 - 1_100) // 5)
    second = logits_for((5_900 - 1_100) // 5)
    model = ScriptedPitchEstimator([first, second])
    actor = model.actor(decoding=decoding)

    first_decision = actor.decide(raw_observation(target_note=12, spectrum_fill=0.1))
    second_decision = actor.decide(raw_observation(target_note=12, spectrum_fill=0.9))

    assert first_decision.estimated_candidate_cents == 6_100
    assert second_decision.estimated_candidate_cents == 5_900
    assert len(model.observed_spectra) == 2
    assert not torch.equal(model.observed_spectra[0], model.observed_spectra[1])


@pytest.mark.parametrize(
    "output",
    [
        torch.zeros(PITCH_CLASS_COUNT),
        torch.zeros((1, PITCH_CLASS_COUNT - 1)),
        torch.zeros((2, PITCH_CLASS_COUNT)),
        torch.zeros((1, PITCH_CLASS_COUNT), dtype=torch.float64),
        torch.full((1, PITCH_CLASS_COUNT), float("nan")),
    ],
)
@pytest.mark.parametrize("decoding", list(PitchDecoding))
def test_actor_rejects_malformed_estimator_output(
    output: torch.Tensor, decoding: PitchDecoding
) -> None:
    actor = ScriptedPitchEstimator([output]).actor(decoding=decoding)

    with pytest.raises(LearningExecutionError, match="logits"):
        actor.decide(raw_observation())


def test_actor_rejects_malformed_or_nonfinite_model_state() -> None:
    with pytest.raises(LearningContractError, match="state"):
        PitchPlannerActor(torch.nn.Linear(1, 1))

    model = PitchEstimatorNetwork()
    with torch.no_grad():
        model.scorer[2].bias[0] = float("inf")
    with pytest.raises(LearningContractError, match="finite"):
        PitchPlannerActor(model)


@pytest.mark.parametrize("decoding", list(PitchDecoding))
def test_actor_rejects_an_illegal_planned_action(decoding: PitchDecoding) -> None:
    actor = ScriptedPitchEstimator([logits_for(980)]).actor(decoding=decoding)

    with (
        patch(
            "harpy.learning.pitch_actor.minimum_action_plan",
            return_value=(PitchAction.OCTAVE_DOWN, PitchAction.SUBMIT),
        ),
        patch(
            "harpy.learning.pitch_actor.legal_action_mask",
            return_value=(False, True, True, True, True, True, True),
        ),
        pytest.raises(LearningExecutionError, match="legal"),
    ):
        actor.decide(raw_observation())
