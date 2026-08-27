"""Learning-layer exceptions with stable public failure categories."""

from __future__ import annotations


class LearningContractError(ValueError):
    """Raised when data crosses the learning boundary outside its contract."""


class PitchArtifactSetError(LearningContractError):
    """Raised when a pitch checkpoint collection violates its aggregate contract."""


class DependencyUnavailableError(RuntimeError):
    """Raised when an optional training dependency is not installed."""


class ArtifactError(RuntimeError):
    """Raised when a persisted learning artifact is invalid or unusable."""


class LearningExecutionError(RuntimeError):
    """Raised when a learning operation cannot complete safely."""
