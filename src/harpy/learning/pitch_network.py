"""Location-preserving neural scorer for the Milestone E pitch grid."""

from __future__ import annotations

from typing import Final

from harpy.envs.spectrum import LOG_SPECTRUM_SIZE
from harpy.learning.dependencies import require_training_dependencies
from harpy.learning.errors import LearningContractError, LearningExecutionError

_training_stack = require_training_dependencies()
torch = _training_stack.torch

PITCH_GRID_MIN_CENTS: Final = 1_100
PITCH_GRID_MAX_CENTS: Final = 10_900
PITCH_GRID_STEP_CENTS: Final = 5
PITCH_CLASS_COUNT: Final = LOG_SPECTRUM_SIZE
PITCH_ESTIMATOR_PARAMETER_COUNT: Final = 2_497
PITCH_ESTIMATOR_ARCHITECTURE_ID: Final = "harpy-sine-pitch-estimator-conv-v1"

_EXPECTED_STATE_SHAPES: Final = {
    "scorer.0.weight": (16, 1, 9),
    "scorer.0.bias": (16,),
    "scorer.2.weight": (16, 16, 9),
    "scorer.2.bias": (16,),
    "scorer.4.weight": (1, 16, 1),
    "scorer.4.bias": (1,),
}


class PitchEstimatorNetwork(torch.nn.Module):
    """Score every public five-cent coordinate without discarding location."""

    def __init__(self) -> None:
        super().__init__()
        self.scorer = torch.nn.Sequential(
            torch.nn.Conv1d(1, 16, kernel_size=9, stride=1, padding=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 16, kernel_size=9, stride=1, padding=4),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        """Return one finite logit per pitch coordinate for each spectrum."""
        _validate_spectrum_tensor(spectrum)
        logits = self.scorer(spectrum.unsqueeze(1)).squeeze(1)
        if logits.shape != spectrum.shape:
            raise LearningExecutionError(
                f"pitch estimator logits must have shape (batch, {PITCH_CLASS_COUNT})"
            )
        if logits.dtype != torch.float32:
            raise LearningExecutionError("pitch estimator logits must have dtype float32")
        if not torch.isfinite(logits).all():
            raise LearningExecutionError("pitch estimator logits must contain only finite values")
        return logits


def pitch_class_index(candidate_cents: int) -> int:
    """Map an integer-cent candidate to its nearest public five-cent class."""
    if (
        isinstance(candidate_cents, bool)
        or not isinstance(candidate_cents, int)
        or not PITCH_GRID_MIN_CENTS <= candidate_cents <= PITCH_GRID_MAX_CENTS
    ):
        raise ValueError(
            "candidate_cents must be an integer within "
            f"{PITCH_GRID_MIN_CENTS}..{PITCH_GRID_MAX_CENTS}"
        )
    return (
        candidate_cents - PITCH_GRID_MIN_CENTS + PITCH_GRID_STEP_CENTS // 2
    ) // PITCH_GRID_STEP_CENTS


def pitch_cents_from_index(index: int) -> int:
    """Map one public class index to its exact five-cent grid center."""
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < PITCH_CLASS_COUNT:
        raise ValueError(f"index must be an integer within 0..{PITCH_CLASS_COUNT - 1}")
    return PITCH_GRID_MIN_CENTS + PITCH_GRID_STEP_CENTS * index


def validate_pitch_estimator_model(model: object) -> None:
    """Reject any estimator whose module state differs from the locked v1 scorer."""
    if type(model) is not PitchEstimatorNetwork:
        raise LearningContractError(
            "pitch estimator type and state must match PitchEstimatorNetwork"
        )
    layers = tuple(model.scorer)
    expected_layer_types = (
        torch.nn.Conv1d,
        torch.nn.ReLU,
        torch.nn.Conv1d,
        torch.nn.ReLU,
        torch.nn.Conv1d,
    )
    if tuple(type(layer) for layer in layers) != expected_layer_types:
        raise LearningContractError("pitch estimator type and topology must match v1")
    convolution_shapes = tuple(
        (
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
            layer.padding_mode,
        )
        for layer in (layers[0], layers[2], layers[4])
    )
    if convolution_shapes != (
        (1, 16, (9,), (1,), (4,), (1,), 1, "zeros"),
        (16, 16, (9,), (1,), (4,), (1,), 1, "zeros"),
        (16, 1, (1,), (1,), (0,), (1,), 1, "zeros"),
    ) or any(layer.inplace for layer in (layers[1], layers[3])):
        raise LearningContractError("pitch estimator type and topology must match v1")
    state = model.state_dict()
    actual_shapes = {name: tuple(value.shape) for name, value in state.items()}
    if actual_shapes != _EXPECTED_STATE_SHAPES:
        raise LearningContractError("pitch estimator state names and shapes must match v1")
    if any(value.dtype != torch.float32 for value in state.values()):
        raise LearningContractError("pitch estimator state must have dtype float32")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != PITCH_ESTIMATOR_PARAMETER_COUNT:
        raise LearningContractError(
            f"pitch estimator state must contain {PITCH_ESTIMATOR_PARAMETER_COUNT} parameters"
        )
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise LearningContractError("pitch estimator state must contain only finite values")


def _validate_spectrum_tensor(spectrum: object) -> None:
    if not isinstance(spectrum, torch.Tensor):
        raise LearningContractError("spectrum must be a torch tensor")
    if spectrum.dtype != torch.float32:
        raise LearningContractError("spectrum tensor must have dtype float32")
    if spectrum.ndim != 2 or spectrum.shape[0] < 1 or spectrum.shape[1] != PITCH_CLASS_COUNT:
        raise LearningContractError(f"spectrum tensor must have shape (batch, {PITCH_CLASS_COUNT})")
    if not torch.isfinite(spectrum).all():
        raise LearningContractError("spectrum tensor must contain only finite values")


__all__ = [
    "PITCH_CLASS_COUNT",
    "PITCH_ESTIMATOR_ARCHITECTURE_ID",
    "PITCH_ESTIMATOR_PARAMETER_COUNT",
    "PITCH_GRID_MAX_CENTS",
    "PITCH_GRID_MIN_CENTS",
    "PITCH_GRID_STEP_CENTS",
    "PitchEstimatorNetwork",
    "pitch_cents_from_index",
    "pitch_class_index",
    "validate_pitch_estimator_model",
]
