"""Lazy access to the optional PyTorch and Stable-Baselines3 stack."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

from harpy.learning.errors import DependencyUnavailableError

TRAIN_INSTALL_INSTRUCTION = "uv sync --group train"


@dataclass(frozen=True, slots=True)
class TrainingStack:
    """The installed optional packages and their resolved module versions."""

    torch: ModuleType
    stable_baselines3: ModuleType
    torch_version: str
    stable_baselines3_version: str


def require_training_dependencies(
    import_module: Callable[[str], ModuleType] = importlib.import_module,
) -> TrainingStack:
    """Load the optional training stack or provide its one installation command."""
    try:
        torch = import_module("torch")
        stable_baselines3 = import_module("stable_baselines3")
    except ModuleNotFoundError as error:
        raise DependencyUnavailableError(
            "The optional training stack is unavailable. Install it with "
            f"`{TRAIN_INSTALL_INSTRUCTION}`."
        ) from error

    return TrainingStack(
        torch=torch,
        stable_baselines3=stable_baselines3,
        torch_version=torch.__version__,
        stable_baselines3_version=stable_baselines3.__version__,
    )
