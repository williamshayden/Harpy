"""Exploratory text-model actors; separate from the stable harpy CLI/API."""

from research.sound_matching.llm.actor import LLMActor, llm_actor_spec
from research.sound_matching.llm.features import (
    FEATURE_ID,
    feature_configuration,
    public_observation,
)

__all__ = [
    "FEATURE_ID",
    "LLMActor",
    "feature_configuration",
    "llm_actor_spec",
    "public_observation",
]
