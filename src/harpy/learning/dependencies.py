"""Lazy access to the optional PyTorch and Stable-Baselines3 stack."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

from harpy.learning.errors import DependencyUnavailableError

TRAIN_INSTALL_INSTRUCTION = "python -m pip install 'harpy-audio[train]'"
CUBLAS_DETERMINISTIC_WORKSPACE_CONFIG = ":4096:8"


def configure_deterministic_cuda_environment() -> None:
    """Pin the cuBLAS workspace before any CUDA runtime interaction.

    NVIDIA requires this process environment setting for deterministic cuBLAS on
    CUDA 10.2 and newer.  The E.1 protocol owns one exact setting rather than
    inheriting a caller's potentially different workspace choice.
    """

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_DETERMINISTIC_WORKSPACE_CONFIG


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
