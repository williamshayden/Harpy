"""Versioned shared learned-policy feature-network contracts."""

from __future__ import annotations

import torch

from harpy.learning.models import ARCHITECTURE_SCHEMA_ID
from harpy.learning.network import (
    BCPolicyNetwork,
    HarpySineFeaturesExtractor,
    SineFeatureEncoder,
)
from harpy.learning.observations import POLICY_OBSERVATION_SPACE


def policy_batch() -> dict[str, torch.Tensor]:
    return {
        "spectrum": torch.linspace(0, 1, 2 * 1_961, dtype=torch.float32).reshape(2, 1_961),
        "state": torch.tensor(
            [[-1.0, -1.0, -1.0, -1.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
            dtype=torch.float32,
        ),
    }


def test_network_parameter_counts_are_versioned() -> None:
    encoder = SineFeatureEncoder()
    bc = BCPolicyNetwork()

    assert sum(parameter.numel() for parameter in encoder.parameters()) == 74_784
    assert sum(parameter.numel() for parameter in bc.parameters()) == 75_687


def test_encoder_has_the_versioned_two_branch_topology() -> None:
    encoder = SineFeatureEncoder()

    spectrum_layers = list(encoder.spectrum_encoder)
    assert isinstance(spectrum_layers[0], torch.nn.Conv1d)
    assert spectrum_layers[0].in_channels == 1
    assert spectrum_layers[0].out_channels == 16
    assert spectrum_layers[0].kernel_size == (9,)
    assert spectrum_layers[0].stride == (4,)
    assert isinstance(spectrum_layers[1], torch.nn.ReLU)
    assert isinstance(spectrum_layers[2], torch.nn.Conv1d)
    assert spectrum_layers[2].in_channels == 16
    assert spectrum_layers[2].out_channels == 32
    assert spectrum_layers[2].kernel_size == (7,)
    assert spectrum_layers[2].stride == (4,)
    assert isinstance(spectrum_layers[3], torch.nn.ReLU)
    assert isinstance(spectrum_layers[4], torch.nn.AdaptiveAvgPool1d)
    assert spectrum_layers[4].output_size == 16

    state_layers = list(encoder.state_encoder)
    assert isinstance(state_layers[0], torch.nn.Linear)
    assert (state_layers[0].in_features, state_layers[0].out_features) == (5, 32)
    assert isinstance(state_layers[1], torch.nn.ReLU)
    assert isinstance(state_layers[2], torch.nn.Linear)
    assert (state_layers[2].in_features, state_layers[2].out_features) == (32, 32)
    assert isinstance(state_layers[3], torch.nn.ReLU)

    combined_layers = list(encoder.combined_encoder)
    assert isinstance(combined_layers[0], torch.nn.Linear)
    assert (combined_layers[0].in_features, combined_layers[0].out_features) == (544, 128)
    assert isinstance(combined_layers[1], torch.nn.ReLU)
    assert SineFeatureEncoder.output_dim == 128


def test_network_outputs_are_finite_and_have_expected_shapes_on_cpu() -> None:
    batch = policy_batch()
    encoder = SineFeatureEncoder()
    bc = BCPolicyNetwork()
    extractor = HarpySineFeaturesExtractor(POLICY_OBSERVATION_SPACE)

    features = encoder(batch)
    logits = bc(batch)
    extracted = extractor(batch)

    assert features.shape == (2, 128)
    assert logits.shape == (2, 7)
    assert extracted.shape == (2, 128)
    assert torch.isfinite(features).all()
    assert torch.isfinite(logits).all()
    assert torch.isfinite(extracted).all()
    assert all(parameter.device.type == "cpu" for parameter in bc.parameters())
    assert extractor.features_dim == 128


def test_seeded_network_state_dict_is_deterministic() -> None:
    torch.manual_seed(20_260_810)
    first = BCPolicyNetwork()
    torch.manual_seed(20_260_810)
    second = BCPolicyNetwork()

    assert first.state_dict().keys() == second.state_dict().keys()
    assert all(
        torch.equal(first_value, second.state_dict()[name])
        for name, first_value in first.state_dict().items()
    )


def test_network_reuses_the_architecture_schema() -> None:
    from harpy.learning import network

    assert network.ARCHITECTURE_SCHEMA_ID is ARCHITECTURE_SCHEMA_ID
