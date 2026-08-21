"""Shared PyTorch feature architecture for learned sine policies."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import gymnasium

from harpy.envs.spectrum import LOG_SPECTRUM_SIZE
from harpy.learning.dependencies import require_training_dependencies
from harpy.learning.errors import LearningContractError
from harpy.learning.models import ARCHITECTURE_SCHEMA_ID

_training_stack = require_training_dependencies()
torch = _training_stack.torch
BaseFeaturesExtractor = _training_stack.stable_baselines3.common.torch_layers.BaseFeaturesExtractor


class SineFeatureEncoder(torch.nn.Module):
    """Encode the frozen spectrum and normalized public scalar state."""

    output_dim: ClassVar[int] = 128

    def __init__(self) -> None:
        super().__init__()
        self.spectrum_encoder = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=9, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 32, kernel_size=7, stride=4),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(16),
        )
        self.state_encoder = torch.nn.Sequential(
            torch.nn.Linear(5, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
        )
        self.combined_encoder = torch.nn.Sequential(
            torch.nn.Linear(544, self.output_dim),
            torch.nn.ReLU(),
        )

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return one 128-feature vector for every validated policy observation."""
        spectrum, state = _policy_tensors(observations)
        spectrum_features = self.spectrum_encoder(spectrum.unsqueeze(1)).flatten(start_dim=1)
        state_features = self.state_encoder(state)
        return self.combined_encoder(torch.cat((spectrum_features, state_features), dim=1))


class BCPolicyNetwork(torch.nn.Module):
    """Seven-action behavior-cloning classifier using the shared feature encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = SineFeatureEncoder()
        self.action_head = torch.nn.Linear(SineFeatureEncoder.output_dim, 7)

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return seven action logits for each policy observation."""
        return self.action_head(self.encoder(observations))


class HarpySineFeaturesExtractor(BaseFeaturesExtractor):
    """Expose the shared sine-policy encoder to Stable-Baselines3 PPO."""

    def __init__(self, observation_space: gymnasium.spaces.Dict) -> None:
        super().__init__(observation_space, features_dim=SineFeatureEncoder.output_dim)
        self.encoder = SineFeatureEncoder()

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Return the common 128-feature representation for SB3 policy heads."""
        return self.encoder(observations)


def _policy_tensors(
    observations: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(observations, Mapping) or set(observations) != {"spectrum", "state"}:
        raise LearningContractError("policy observations must contain exactly spectrum and state")
    spectrum = observations["spectrum"]
    state = observations["state"]
    if not isinstance(spectrum, torch.Tensor) or not isinstance(state, torch.Tensor):
        raise LearningContractError("policy observations must contain torch tensors")
    if spectrum.dtype != torch.float32 or state.dtype != torch.float32:
        raise LearningContractError("policy tensors must have dtype float32")
    if spectrum.ndim != 2 or spectrum.shape[1] != LOG_SPECTRUM_SIZE:
        raise LearningContractError(f"spectrum tensor must have shape (batch, {LOG_SPECTRUM_SIZE})")
    if state.ndim != 2 or state.shape[1] != 5:
        raise LearningContractError("state tensor must have shape (batch, 5)")
    if spectrum.shape[0] != state.shape[0] or spectrum.device != state.device:
        raise LearningContractError("policy tensors must share batch size and device")
    if not torch.isfinite(spectrum).all() or not torch.isfinite(state).all():
        raise LearningContractError("policy tensors must contain only finite values")
    return spectrum, state


__all__ = [
    "ARCHITECTURE_SCHEMA_ID",
    "BCPolicyNetwork",
    "HarpySineFeaturesExtractor",
    "SineFeatureEncoder",
]
