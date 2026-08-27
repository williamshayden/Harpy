"""Location-preserving Milestone E pitch-estimator contracts."""

from __future__ import annotations

import pytest
import torch

from harpy.learning.errors import LearningContractError
from harpy.learning.pitch_network import (
    PITCH_CLASS_COUNT,
    PITCH_ESTIMATOR_PARAMETER_COUNT,
    PITCH_GRID_MAX_CENTS,
    PITCH_GRID_MIN_CENTS,
    PITCH_GRID_STEP_CENTS,
    PitchEstimatorNetwork,
    pitch_cents_from_index,
    pitch_class_index,
    validate_pitch_estimator_model,
)


def test_pitch_estimator_has_exact_location_preserving_topology() -> None:
    model = PitchEstimatorNetwork()
    layers = list(model.scorer)

    assert len(layers) == 5
    assert isinstance(layers[0], torch.nn.Conv1d)
    assert (layers[0].in_channels, layers[0].out_channels) == (1, 16)
    assert layers[0].kernel_size == (9,)
    assert layers[0].stride == (1,)
    assert layers[0].padding == (4,)
    assert isinstance(layers[1], torch.nn.ReLU)
    assert isinstance(layers[2], torch.nn.Conv1d)
    assert (layers[2].in_channels, layers[2].out_channels) == (16, 16)
    assert layers[2].kernel_size == (9,)
    assert layers[2].stride == (1,)
    assert layers[2].padding == (4,)
    assert isinstance(layers[3], torch.nn.ReLU)
    assert isinstance(layers[4], torch.nn.Conv1d)
    assert (layers[4].in_channels, layers[4].out_channels) == (16, 1)
    assert layers[4].kernel_size == (1,)
    assert layers[4].stride == (1,)
    assert layers[4].padding == (0,)

    forbidden = (
        torch.nn.Linear,
        torch.nn.Flatten,
        torch.nn.modules.pooling._AdaptiveAvgPoolNd,
        torch.nn.modules.pooling._AvgPoolNd,
        torch.nn.modules.pooling._MaxPoolNd,
        torch.nn.RNNBase,
    )
    assert not any(isinstance(module, forbidden) for module in model.modules())


def test_pitch_estimator_state_has_exact_names_shapes_count_and_finite_values() -> None:
    model = PitchEstimatorNetwork()

    assert {name: tuple(value.shape) for name, value in model.state_dict().items()} == {
        "scorer.0.weight": (16, 1, 9),
        "scorer.0.bias": (16,),
        "scorer.2.weight": (16, 16, 9),
        "scorer.2.bias": (16,),
        "scorer.4.weight": (1, 16, 1),
        "scorer.4.bias": (1,),
    }
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_497
    assert PITCH_ESTIMATOR_PARAMETER_COUNT == 2_497
    assert all(value.dtype is torch.float32 for value in model.state_dict().values())
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    validate_pitch_estimator_model(model)


def test_pitch_estimator_returns_one_finite_logit_per_grid_coordinate() -> None:
    model = PitchEstimatorNetwork()
    spectrum = torch.linspace(0, 1, 2 * PITCH_CLASS_COUNT, dtype=torch.float32).reshape(
        2, PITCH_CLASS_COUNT
    )

    logits = model(spectrum)

    assert logits.shape == (2, PITCH_CLASS_COUNT)
    assert logits.dtype is torch.float32
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize(
    "spectrum",
    [
        torch.zeros(PITCH_CLASS_COUNT),
        torch.zeros((1, PITCH_CLASS_COUNT - 1)),
        torch.zeros((1, PITCH_CLASS_COUNT, 1)),
        torch.zeros((1, PITCH_CLASS_COUNT), dtype=torch.float64),
        torch.full((1, PITCH_CLASS_COUNT), float("nan")),
        torch.full((1, PITCH_CLASS_COUNT), float("inf")),
    ],
)
def test_pitch_estimator_rejects_malformed_spectrum_tensors(spectrum: torch.Tensor) -> None:
    with pytest.raises(LearningContractError, match="spectrum"):
        PitchEstimatorNetwork()(spectrum)


def test_pitch_estimator_model_validation_rejects_wrong_or_nonfinite_state() -> None:
    with pytest.raises(LearningContractError, match="state"):
        validate_pitch_estimator_model(torch.nn.Conv1d(1, 1, 1))

    model = PitchEstimatorNetwork()
    with torch.no_grad():
        model.scorer[0].weight[0, 0, 0] = float("nan")
    with pytest.raises(LearningContractError, match="finite"):
        validate_pitch_estimator_model(model)


def test_pitch_estimator_model_validation_rejects_a_state_compatible_impostor() -> None:
    class ConstantLogitImpostor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scorer = PitchEstimatorNetwork().scorer

        def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(spectrum)

    with pytest.raises(LearningContractError, match="type"):
        validate_pitch_estimator_model(ConstantLogitImpostor())


@pytest.mark.parametrize(
    ("layer_index", "attribute", "value"),
    [
        (0, "dilation", (2,)),
        (2, "groups", 2),
        (0, "padding_mode", "reflect"),
        (1, "inplace", True),
    ],
)
def test_pitch_estimator_model_validation_rejects_behavior_changing_topology(
    layer_index: int, attribute: str, value: object
) -> None:
    model = PitchEstimatorNetwork()
    setattr(model.scorer[layer_index], attribute, value)

    with pytest.raises(LearningContractError, match="topology"):
        validate_pitch_estimator_model(model)


@pytest.mark.parametrize(
    ("candidate_cents", "expected_index"),
    [
        (1_100, 0),
        (1_102, 0),
        (1_103, 1),
        (1_107, 1),
        (1_108, 2),
        (10_898, 1_960),
        (10_900, 1_960),
    ],
)
def test_pitch_class_index_uses_the_locked_nearest_five_cent_rule(
    candidate_cents: int, expected_index: int
) -> None:
    assert pitch_class_index(candidate_cents) == expected_index


def test_pitch_grid_index_mapping_is_exact_and_rejects_out_of_range_values() -> None:
    assert PITCH_GRID_MIN_CENTS == 1_100
    assert PITCH_GRID_MAX_CENTS == 10_900
    assert PITCH_GRID_STEP_CENTS == 5
    assert PITCH_CLASS_COUNT == 1_961
    assert pitch_cents_from_index(0) == 1_100
    assert pitch_cents_from_index(1) == 1_105
    assert pitch_cents_from_index(1_960) == 10_900

    for value in (True, 1.5, 1_099, 10_901):
        with pytest.raises(ValueError):
            pitch_class_index(value)  # type: ignore[arg-type]
    for value in (True, 1.5, -1, 1_961):
        with pytest.raises(ValueError):
            pitch_cents_from_index(value)  # type: ignore[arg-type]
