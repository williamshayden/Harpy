"""Deterministic supervised training for the Milestone E pitch estimator.

This module owns only coordinate-supervised perception.  It deliberately does not
import episode suites, evaluator records, action labels, rewards, or artifact
workflows, so final evaluation data cannot cross the trainer boundary.
"""

from __future__ import annotations

import math
import operator
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from harpy.envs.spectrum import LOG_SPECTRUM_SIZE
from harpy.learning.cache import SpectrumEvidenceCache
from harpy.learning.dependencies import require_training_dependencies
from harpy.learning.envs import SpectrumEvidenceProvider
from harpy.learning.errors import LearningContractError, LearningExecutionError
from harpy.learning.models import ProfileName
from harpy.learning.pitch_data import pitch_coordinate_split
from harpy.learning.pitch_network import (
    PITCH_CLASS_COUNT,
    PITCH_GRID_MAX_CENTS,
    PITCH_GRID_MIN_CENTS,
    PitchEstimatorNetwork,
    pitch_cents_from_index,
    pitch_class_index,
    validate_pitch_estimator_model,
)

_training_stack = require_training_dependencies()
torch = _training_stack.torch

PITCH_TRAINING_NUM_WORKERS: Final = 0
PITCH_SHUFFLE_GENERATOR_DEVICE: Final = "cpu"
_MAX_TRAINING_COORDINATES: Final = 1_400
_MAX_VALIDATION_COORDINATES: Final = 200
_STATE_SHAPES: Final = {
    "scorer.0.weight": (16, 1, 9),
    "scorer.0.bias": (16,),
    "scorer.2.weight": (16, 16, 9),
    "scorer.2.bias": (16,),
    "scorer.4.weight": (1, 16, 1),
    "scorer.4.bias": (1,),
}


def _integer(
    value: object,
    field: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        normalized = int(operator.index(value))
    except TypeError as error:
        raise ValueError(f"{field} must be an integer") from error
    if normalized < minimum or (maximum is not None and normalized > maximum):
        if maximum is None:
            raise ValueError(f"{field} must be at least {minimum}")
        raise ValueError(f"{field} must be within {minimum}..{maximum}")
    return normalized


def _finite_float(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be a finite number")
    if positive and normalized <= 0.0:
        raise ValueError(f"{field} must be positive")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return normalized


@dataclass(frozen=True, slots=True)
class PitchTrainingProfile:
    """One immutable coordinate count, optimizer, and stopping contract."""

    training_coordinate_count: int
    validation_coordinate_count: int
    max_epochs: int
    early_stopping_patience: int | None
    batch_size: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    eligible_for_aggregate: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "training_coordinate_count",
            _integer(
                self.training_coordinate_count,
                "training_coordinate_count",
                minimum=1,
                maximum=_MAX_TRAINING_COORDINATES,
            ),
        )
        object.__setattr__(
            self,
            "validation_coordinate_count",
            _integer(
                self.validation_coordinate_count,
                "validation_coordinate_count",
                minimum=1,
                maximum=_MAX_VALIDATION_COORDINATES,
            ),
        )
        max_epochs = _integer(self.max_epochs, "max_epochs", minimum=1)
        object.__setattr__(self, "max_epochs", max_epochs)
        if self.early_stopping_patience is not None:
            patience = _integer(
                self.early_stopping_patience,
                "early_stopping_patience",
                minimum=1,
                maximum=max_epochs,
            )
            object.__setattr__(self, "early_stopping_patience", patience)
        object.__setattr__(self, "batch_size", _integer(self.batch_size, "batch_size", minimum=1))
        if self.optimizer != "AdamW":
            raise ValueError("optimizer must be 'AdamW'")
        object.__setattr__(
            self,
            "learning_rate",
            _finite_float(self.learning_rate, "learning_rate", positive=True),
        )
        object.__setattr__(
            self,
            "weight_decay",
            _finite_float(self.weight_decay, "weight_decay", minimum=0.0),
        )
        if type(self.eligible_for_aggregate) is not bool:
            raise ValueError("eligible_for_aggregate must be a bool")


PITCH_PROFILE_CONFIGS: Mapping[ProfileName, PitchTrainingProfile] = MappingProxyType(
    {
        ProfileName.SMOKE: PitchTrainingProfile(
            training_coordinate_count=256,
            validation_coordinate_count=64,
            max_epochs=2,
            early_stopping_patience=None,
            batch_size=64,
            optimizer="AdamW",
            learning_rate=0.001,
            weight_decay=0.0001,
            eligible_for_aggregate=False,
        ),
        ProfileName.CHECKPOINT: PitchTrainingProfile(
            training_coordinate_count=1_400,
            validation_coordinate_count=200,
            max_epochs=50,
            early_stopping_patience=8,
            batch_size=64,
            optimizer="AdamW",
            learning_rate=0.001,
            weight_decay=0.0001,
            eligible_for_aggregate=True,
        ),
    }
)


class PitchCoordinateDataset(torch.utils.data.Dataset):
    """Eagerly own one immutable real-rendered spectrum per ordered coordinate."""

    def __init__(self, coordinates: Sequence[int], evidence_provider: object) -> None:
        normalized = _coordinate_sequence(coordinates)
        if not normalized:
            raise ValueError("coordinates must not be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("coordinates must not contain duplicate values")
        spectrum_for_cents = getattr(evidence_provider, "spectrum_for_cents", None)
        if not callable(spectrum_for_cents):
            raise ValueError("evidence_provider must provide spectrum_for_cents")

        spectra: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        for coordinate in normalized:
            spectra.append(_owned_immutable_spectrum(spectrum_for_cents(coordinate)))
        self._coordinates = normalized
        self._spectra = tuple(spectra)
        self._labels = tuple(np.int64(pitch_class_index(value)) for value in normalized)

    @property
    def coordinates(self) -> tuple[int, ...]:
        """Return the exact ordered candidate-coordinate identity."""
        return self._coordinates

    def __len__(self) -> int:
        return len(self._coordinates)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.int64, np.int64]:
        normalized = _dataset_index(index, len(self))
        return (
            np.array(self._spectra[normalized], dtype=np.float32, copy=True, order="C"),
            self._labels[normalized],
            np.int64(self._coordinates[normalized]),
        )


@dataclass(frozen=True, slots=True)
class PitchPredictionRecord:
    """One ordered raw prediction and its unweighted cross-entropy loss."""

    candidate_cents: int
    predicted_class_index: int
    cross_entropy_loss: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_cents",
            _integer(
                self.candidate_cents,
                "candidate_cents",
                minimum=PITCH_GRID_MIN_CENTS,
                maximum=PITCH_GRID_MAX_CENTS,
            ),
        )
        object.__setattr__(
            self,
            "predicted_class_index",
            _integer(
                self.predicted_class_index,
                "predicted_class_index",
                minimum=0,
                maximum=PITCH_CLASS_COUNT - 1,
            ),
        )
        object.__setattr__(
            self,
            "cross_entropy_loss",
            _finite_float(
                self.cross_entropy_loss,
                "cross_entropy_loss",
                minimum=0.0,
            ),
        )

    @property
    def target_class_index(self) -> int:
        return pitch_class_index(self.candidate_cents)

    @property
    def predicted_cents(self) -> int:
        return pitch_cents_from_index(self.predicted_class_index)

    @property
    def signed_error_cents(self) -> int:
        return self.predicted_cents - self.candidate_cents

    @property
    def absolute_error_cents(self) -> int:
        return abs(self.signed_error_cents)


@dataclass(frozen=True, slots=True)
class PitchDatasetMetrics:
    """Finite loss and coordinate-error metrics for one complete ordered dataset."""

    example_count: int
    loss: float
    mean_absolute_error_cents: float
    median_absolute_error_cents: float
    within_one_count: int
    within_one_rate: float
    within_five_count: int
    within_five_rate: float

    def __post_init__(self) -> None:
        count = _integer(self.example_count, "example_count", minimum=1)
        within_one = _integer(self.within_one_count, "within_one_count", minimum=0, maximum=count)
        within_five = _integer(
            self.within_five_count,
            "within_five_count",
            minimum=within_one,
            maximum=count,
        )
        object.__setattr__(self, "example_count", count)
        object.__setattr__(self, "loss", _finite_float(self.loss, "loss", minimum=0.0))
        object.__setattr__(
            self,
            "mean_absolute_error_cents",
            _finite_float(
                self.mean_absolute_error_cents,
                "mean_absolute_error_cents",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "median_absolute_error_cents",
            _finite_float(
                self.median_absolute_error_cents,
                "median_absolute_error_cents",
                minimum=0.0,
            ),
        )
        object.__setattr__(self, "within_one_count", within_one)
        object.__setattr__(self, "within_five_count", within_five)
        within_one_rate = _finite_float(
            self.within_one_rate,
            "within_one_rate",
            minimum=0.0,
            maximum=1.0,
        )
        within_five_rate = _finite_float(
            self.within_five_rate,
            "within_five_rate",
            minimum=0.0,
            maximum=1.0,
        )
        if within_one_rate != within_one / count or within_five_rate != within_five / count:
            raise ValueError("within-one/five rates must match their derived counts")
        object.__setattr__(self, "within_one_rate", within_one_rate)
        object.__setattr__(self, "within_five_rate", within_five_rate)


@dataclass(frozen=True, slots=True)
class PitchEpochMetrics:
    """Complete ordered train and validation metrics after one epoch."""

    epoch: int
    training: PitchDatasetMetrics
    validation: PitchDatasetMetrics

    def __post_init__(self) -> None:
        object.__setattr__(self, "epoch", _integer(self.epoch, "epoch", minimum=1))
        if not isinstance(self.training, PitchDatasetMetrics):
            raise ValueError("training must be PitchDatasetMetrics")
        if not isinstance(self.validation, PitchDatasetMetrics):
            raise ValueError("validation must be PitchDatasetMetrics")


@dataclass(frozen=True, slots=True)
class PitchTrainingSummary:
    """Selected epoch, full history, and deterministic-runtime settings."""

    history: tuple[PitchEpochMetrics, ...]
    selected_epoch: int
    training_examples: int
    validation_examples: int
    seed: int
    device: str
    deterministic_algorithms: bool
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    data_loader_num_workers: int
    shuffle_generator_device: str
    training_wall_time_seconds: float

    def __post_init__(self) -> None:
        if isinstance(self.history, (str, bytes)):
            raise ValueError("history must contain PitchEpochMetrics")
        try:
            history = tuple(self.history)
        except TypeError as error:
            raise ValueError("history must contain PitchEpochMetrics") from error
        if not history or not all(isinstance(item, PitchEpochMetrics) for item in history):
            raise ValueError("history must contain PitchEpochMetrics")
        if tuple(item.epoch for item in history) != tuple(range(1, len(history) + 1)):
            raise ValueError("history must use consecutive epoch ordering")
        selected_epoch = _integer(self.selected_epoch, "selected_epoch", minimum=1)
        if selected_epoch not in {item.epoch for item in history}:
            raise ValueError("selected_epoch must be present in history")
        best_epoch = min(history, key=pitch_epoch_rank).epoch
        if selected_epoch != best_epoch:
            raise ValueError("selected_epoch must match the complete lexicographic rank")
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "selected_epoch", selected_epoch)
        training_examples = _integer(
            self.training_examples,
            "training_examples",
            minimum=1,
        )
        validation_examples = _integer(
            self.validation_examples,
            "validation_examples",
            minimum=1,
        )
        if any(item.training.example_count != training_examples for item in history):
            raise ValueError("training_examples must match every history row")
        if any(item.validation.example_count != validation_examples for item in history):
            raise ValueError("validation_examples must match every history row")
        object.__setattr__(self, "training_examples", training_examples)
        object.__setattr__(self, "validation_examples", validation_examples)
        object.__setattr__(self, "seed", _integer(self.seed, "seed", minimum=0))
        if not isinstance(self.device, str):
            raise ValueError("device must identify CPU or CUDA")
        try:
            parsed_device = torch.device(self.device)
        except (RuntimeError, ValueError) as error:
            raise ValueError("device must identify CPU or CUDA") from error
        if parsed_device.type not in {"cpu", "cuda"} or str(parsed_device) != self.device:
            raise ValueError("device must identify canonical CPU or CUDA")
        if self.deterministic_algorithms is not True:
            raise ValueError("deterministic_algorithms must be true")
        if self.cudnn_benchmark is not False:
            raise ValueError("cudnn_benchmark must be false")
        if self.cudnn_deterministic is not True:
            raise ValueError("cudnn_deterministic must be true")
        if self.data_loader_num_workers != PITCH_TRAINING_NUM_WORKERS:
            raise ValueError("data_loader_num_workers must be zero")
        if self.shuffle_generator_device != PITCH_SHUFFLE_GENERATOR_DEVICE:
            raise ValueError("shuffle_generator_device must be cpu")
        object.__setattr__(
            self,
            "training_wall_time_seconds",
            _finite_float(
                self.training_wall_time_seconds,
                "training_wall_time_seconds",
                minimum=0.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class PitchTrainingResult:
    """The restored model, independent selected state, and complete summary."""

    model: PitchEstimatorNetwork
    selected_state: dict[str, torch.Tensor]
    summary: PitchTrainingSummary

    def __post_init__(self) -> None:
        validate_pitch_estimator_model(self.model)
        selected_state = _validated_state_copy(self.selected_state)
        model_state = _cpu_state_dict(self.model)
        if any(not torch.equal(selected_state[name], model_state[name]) for name in selected_state):
            raise ValueError("selected_state must match the restored model")
        object.__setattr__(self, "selected_state", selected_state)
        if not isinstance(self.summary, PitchTrainingSummary):
            raise ValueError("summary must be a PitchTrainingSummary")


def build_pitch_coordinate_datasets(
    profile: ProfileName,
    *,
    cache: SpectrumEvidenceCache | None = None,
    evidence_provider: object | None = None,
) -> tuple[PitchCoordinateDataset, PitchCoordinateDataset]:
    """Materialize only the pinned train/validation coordinate prefixes."""
    if not isinstance(profile, ProfileName):
        raise ValueError("profile must be a ProfileName")
    if cache is not None and evidence_provider is not None:
        raise ValueError("cache and evidence_provider cannot both be provided")
    if cache is not None and not isinstance(cache, SpectrumEvidenceCache):
        raise ValueError("cache must be a SpectrumEvidenceCache")
    if evidence_provider is None:
        evidence_provider = SpectrumEvidenceProvider(
            SpectrumEvidenceCache() if cache is None else cache
        )

    config = PITCH_PROFILE_CONFIGS[profile]
    split = pitch_coordinate_split()
    training_coordinates = split.training_coordinates[: config.training_coordinate_count]
    validation_coordinates = split.validation_coordinates[: config.validation_coordinate_count]
    return (
        PitchCoordinateDataset(training_coordinates, evidence_provider),
        PitchCoordinateDataset(validation_coordinates, evidence_provider),
    )


def summarize_pitch_predictions(
    records: Sequence[PitchPredictionRecord],
) -> PitchDatasetMetrics:
    """Recompute every metric from complete ordered raw prediction records."""
    if isinstance(records, (str, bytes)):
        raise ValueError("records must contain PitchPredictionRecord values")
    try:
        normalized = tuple(records)
    except TypeError as error:
        raise ValueError("records must contain PitchPredictionRecord values") from error
    if not normalized or not all(isinstance(item, PitchPredictionRecord) for item in normalized):
        raise ValueError("records must contain PitchPredictionRecord values")

    errors = tuple(item.absolute_error_cents for item in normalized)
    count = len(normalized)
    ordered_errors = sorted(errors)
    midpoint = count // 2
    if count % 2:
        median = float(ordered_errors[midpoint])
    else:
        median = (ordered_errors[midpoint - 1] + ordered_errors[midpoint]) / 2.0
    within_one = sum(error <= 1 for error in errors)
    within_five = sum(error <= 5 for error in errors)
    return PitchDatasetMetrics(
        example_count=count,
        loss=math.fsum(item.cross_entropy_loss for item in normalized) / count,
        mean_absolute_error_cents=math.fsum(errors) / count,
        median_absolute_error_cents=median,
        within_one_count=within_one,
        within_one_rate=within_one / count,
        within_five_count=within_five,
        within_five_rate=within_five / count,
    )


def pitch_epoch_rank(metrics: PitchEpochMetrics) -> tuple[int, int, float, float, int]:
    """Return the exact minimized checkpoint-selection rank."""
    if not isinstance(metrics, PitchEpochMetrics):
        raise ValueError("metrics must be PitchEpochMetrics")
    validation = metrics.validation
    return (
        -validation.within_five_count,
        -validation.within_one_count,
        validation.mean_absolute_error_cents,
        validation.loss,
        metrics.epoch,
    )


@torch.inference_mode()
def evaluate_pitch_dataset(
    model: PitchEstimatorNetwork,
    dataset: PitchCoordinateDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[PitchDatasetMetrics, tuple[PitchPredictionRecord, ...]]:
    """Evaluate one complete dataset in its declared coordinate order."""
    validate_pitch_estimator_model(model)
    if not isinstance(dataset, PitchCoordinateDataset):
        raise ValueError("dataset must be a PitchCoordinateDataset")
    normalized_batch_size = _integer(batch_size, "batch_size", minimum=1)
    normalized_device = _torch_device(device)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=normalized_batch_size,
        shuffle=False,
        num_workers=PITCH_TRAINING_NUM_WORKERS,
    )
    model.to(normalized_device)
    model.eval()
    records: list[PitchPredictionRecord] = []
    for spectra, labels, coordinates in loader:
        tensors, targets, candidate_cents = _batch_to_device(
            spectra,
            labels,
            coordinates,
            normalized_device,
        )
        logits = _validated_logits(model(tensors), targets.shape[0])
        losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        if not torch.isfinite(losses).all():
            raise LearningExecutionError("pitch evaluation loss must contain only finite values")
        predicted = torch.argmax(logits, dim=1)
        records.extend(
            PitchPredictionRecord(
                candidate_cents=int(candidate),
                predicted_class_index=int(prediction),
                cross_entropy_loss=float(loss),
            )
            for candidate, prediction, loss in zip(
                candidate_cents.cpu().tolist(),
                predicted.cpu().tolist(),
                losses.cpu().to(torch.float64).tolist(),
                strict=True,
            )
        )
    normalized_records = tuple(records)
    if len(normalized_records) != len(dataset):
        raise LearningExecutionError("pitch evaluation did not consume the complete dataset")
    if tuple(item.candidate_cents for item in normalized_records) != dataset.coordinates:
        raise LearningExecutionError("pitch evaluation did not preserve dataset ordering")
    return summarize_pitch_predictions(normalized_records), normalized_records


def train_pitch_estimator(
    training_dataset: PitchCoordinateDataset,
    validation_dataset: PitchCoordinateDataset,
    *,
    profile: PitchTrainingProfile,
    seed: int,
    device: torch.device,
) -> PitchTrainingResult:
    """Train, select, restore, and return one deterministic pitch estimator."""
    if not isinstance(training_dataset, PitchCoordinateDataset):
        raise ValueError("training_dataset must be a PitchCoordinateDataset")
    if not isinstance(validation_dataset, PitchCoordinateDataset):
        raise ValueError("validation_dataset must be a PitchCoordinateDataset")
    if not isinstance(profile, PitchTrainingProfile):
        raise ValueError("profile must be a PitchTrainingProfile")
    split = pitch_coordinate_split()
    if (
        training_dataset.coordinates
        != split.training_coordinates[: profile.training_coordinate_count]
    ):
        raise ValueError("training_dataset must match the profile's pinned training prefix")
    if (
        validation_dataset.coordinates
        != split.validation_coordinates[: profile.validation_coordinate_count]
    ):
        raise ValueError("validation_dataset must match the profile's pinned validation prefix")

    normalized_seed = _integer(seed, "seed", minimum=0)
    normalized_device = _torch_device(device)
    _seed_training(normalized_seed, normalized_device)
    generator = torch.Generator(device=PITCH_SHUFFLE_GENERATOR_DEVICE)
    generator.manual_seed(normalized_seed % (2**63 - 1))
    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=profile.batch_size,
        shuffle=True,
        num_workers=PITCH_TRAINING_NUM_WORKERS,
        generator=generator,
    )
    model = PitchEstimatorNetwork().to(normalized_device)
    validate_pitch_estimator_model(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=profile.learning_rate,
        weight_decay=profile.weight_decay,
    )

    started = time.perf_counter()
    history: list[PitchEpochMetrics] = []
    selected_rank: tuple[int, int, float, float, int] | None = None
    selected_state: dict[str, torch.Tensor] | None = None
    consecutive_non_improvements = 0
    for epoch in range(1, profile.max_epochs + 1):
        _train_one_epoch(model, training_loader, optimizer, normalized_device)
        training_metrics, _ = evaluate_pitch_dataset(
            model,
            training_dataset,
            batch_size=profile.batch_size,
            device=normalized_device,
        )
        validation_metrics, _ = evaluate_pitch_dataset(
            model,
            validation_dataset,
            batch_size=profile.batch_size,
            device=normalized_device,
        )
        metrics = PitchEpochMetrics(
            epoch=epoch,
            training=training_metrics,
            validation=validation_metrics,
        )
        history.append(metrics)
        rank = pitch_epoch_rank(metrics)
        if selected_rank is None or rank < selected_rank:
            selected_rank = rank
            selected_state = _cpu_state_dict(model)
            consecutive_non_improvements = 0
        else:
            consecutive_non_improvements += 1
        if (
            profile.early_stopping_patience is not None
            and consecutive_non_improvements >= profile.early_stopping_patience
        ):
            break

    if selected_state is None or selected_rank is None:
        raise LearningExecutionError("pitch training produced no selectable checkpoint")
    selected_epoch = selected_rank[-1]
    model.load_state_dict(selected_state, strict=True)
    model.to(normalized_device)
    validate_pitch_estimator_model(model)
    independent_selected_state = _validated_state_copy(selected_state)
    summary = PitchTrainingSummary(
        history=tuple(history),
        selected_epoch=selected_epoch,
        training_examples=len(training_dataset),
        validation_examples=len(validation_dataset),
        seed=normalized_seed,
        device=str(normalized_device),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
        cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
        data_loader_num_workers=PITCH_TRAINING_NUM_WORKERS,
        shuffle_generator_device=PITCH_SHUFFLE_GENERATOR_DEVICE,
        training_wall_time_seconds=time.perf_counter() - started,
    )
    return PitchTrainingResult(
        model=model,
        selected_state=independent_selected_state,
        summary=summary,
    )


def save_pitch_estimator_model(path: Path, model: PitchEstimatorNetwork) -> None:
    """Persist only an exact finite CPU estimator state dictionary."""
    if not isinstance(path, Path):
        raise ValueError("path must be a pathlib.Path")
    validate_pitch_estimator_model(model)
    state = _cpu_state_dict(model)
    torch.save(state, path)


def load_pitch_estimator_model(
    path: Path,
    *,
    device: torch.device | None = None,
) -> PitchEstimatorNetwork:
    """Weights-only load and validate one trusted-local estimator payload."""
    if not isinstance(path, Path):
        raise ValueError("path must be a pathlib.Path")
    if not path.is_file():
        raise ValueError("pitch estimator model path must be a file")
    normalized_device = torch.device("cpu") if device is None else _torch_device(device)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise LearningContractError("pitch estimator payload could not be loaded") from error
    try:
        state = _validated_state_copy(payload)
        model = PitchEstimatorNetwork()
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise LearningContractError("pitch estimator payload state is invalid") from error
    model.to(normalized_device)
    validate_pitch_estimator_model(model)
    return model


def _train_one_epoch(
    model: PitchEstimatorNetwork,
    loader: object,
    optimizer: object,
    device: torch.device,
) -> None:
    model.train()
    examples = 0
    for spectra, labels, coordinates in loader:  # type: ignore[union-attr]
        tensors, targets, _candidate_cents = _batch_to_device(
            spectra,
            labels,
            coordinates,
            device,
        )
        optimizer.zero_grad(set_to_none=True)  # type: ignore[union-attr]
        logits = _validated_logits(model(tensors), targets.shape[0])
        objective = torch.nn.functional.cross_entropy(logits, targets)
        if not torch.isfinite(objective):
            raise LearningExecutionError("pitch training loss must be finite")
        objective.backward()
        optimizer.step()  # type: ignore[union-attr]
        examples += targets.shape[0]
    if examples < 1:
        raise LearningExecutionError("pitch training loader must not be empty")
    validate_pitch_estimator_model(model)


def _batch_to_device(
    spectra: object,
    labels: object,
    coordinates: object,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not all(isinstance(value, torch.Tensor) for value in (spectra, labels, coordinates)):
        raise LearningExecutionError("pitch loader must collate torch tensors")
    if spectra.dtype != torch.float32 or spectra.ndim != 2:
        raise LearningExecutionError("pitch loader spectra must be two-dimensional float32")
    if spectra.shape[0] < 1 or spectra.shape[1] != LOG_SPECTRUM_SIZE:
        raise LearningExecutionError(
            f"pitch loader spectra must have shape (batch, {LOG_SPECTRUM_SIZE})"
        )
    if not torch.isfinite(spectra).all():
        raise LearningExecutionError("pitch loader spectra must contain only finite values")
    if torch.any(spectra < 0.0) or torch.any(spectra > 1.0):
        raise LearningExecutionError("pitch loader spectra must be within 0..1")
    if labels.dtype != torch.int64 or labels.shape != (spectra.shape[0],):
        raise LearningExecutionError("pitch loader labels must be one-dimensional int64")
    if coordinates.dtype != torch.int64 or coordinates.shape != (spectra.shape[0],):
        raise LearningExecutionError("pitch loader coordinates must be one-dimensional int64")
    expected_labels = torch.tensor(
        [pitch_class_index(int(value)) for value in coordinates.tolist()],
        dtype=torch.int64,
    )
    if not torch.equal(labels.cpu(), expected_labels):
        raise LearningExecutionError("pitch loader labels must match nearest grid coordinates")
    return spectra.to(device), labels.to(device), coordinates.to(device)


def _validated_logits(value: object, batch_size: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != (batch_size, PITCH_CLASS_COUNT):
        raise LearningExecutionError(f"pitch model must return shape (batch, {PITCH_CLASS_COUNT})")
    if value.dtype != torch.float32 or not torch.isfinite(value).all():
        raise LearningExecutionError("pitch model logits must be finite float32 values")
    return value


def _seed_training(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed % (2**63 - 1))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed % (2**63 - 1))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _cpu_state_dict(model: PitchEstimatorNetwork) -> dict[str, torch.Tensor]:
    validate_pitch_estimator_model(model)
    state = {
        name: value.detach().to(device="cpu", dtype=torch.float32, copy=True)
        for name, value in model.state_dict().items()
    }
    return _validated_state_copy(state)


def _validated_state_copy(value: object) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping):
        raise ValueError("pitch estimator state must be a mapping")
    if tuple(value) != tuple(_STATE_SHAPES):
        raise ValueError("pitch estimator state names and order must match v1")
    copied: dict[str, torch.Tensor] = {}
    for name, shape in _STATE_SHAPES.items():
        tensor = value[name]
        if type(tensor) is not torch.Tensor:
            raise ValueError("pitch estimator state values must be torch tensors")
        if tuple(tensor.shape) != shape:
            raise ValueError("pitch estimator state shapes must match v1")
        if tensor.dtype != torch.float32:
            raise ValueError("pitch estimator state values must have dtype float32")
        if not torch.isfinite(tensor).all():
            raise ValueError("pitch estimator state values must be finite")
        copied[name] = tensor.detach().to(device="cpu", dtype=torch.float32, copy=True)
    return copied


def _torch_device(value: object) -> torch.device:
    if not isinstance(value, torch.device) or value.type not in {"cpu", "cuda"}:
        raise ValueError("device must be a CPU or CUDA torch.device")
    if value.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA training requested but CUDA is unavailable")
    return value


def _coordinate_sequence(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError("coordinates must contain integer cents")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError("coordinates must contain integer cents") from error
    return tuple(
        _integer(
            coordinate,
            "coordinate",
            minimum=PITCH_GRID_MIN_CENTS,
            maximum=PITCH_GRID_MAX_CENTS,
        )
        for coordinate in values
    )


def _owned_immutable_spectrum(value: object) -> np.ndarray[Any, np.dtype[np.float32]]:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float32):
        raise ValueError("spectrum must be a float32 ndarray")
    if value.shape != (LOG_SPECTRUM_SIZE,):
        raise ValueError(f"spectrum must have shape ({LOG_SPECTRUM_SIZE},)")
    if not np.isfinite(value).all():
        raise ValueError("spectrum must contain only finite values")
    if np.any(value < np.float32(0.0)) or np.any(value > np.float32(1.0)):
        raise ValueError("spectrum values must be within 0..1")
    spectrum = np.array(value, dtype=np.float32, copy=True, order="C")
    spectrum.setflags(write=False)
    return spectrum


def _dataset_index(value: object, length: int) -> int:
    if isinstance(value, bool):
        raise ValueError("dataset index must be an integer")
    try:
        index = operator.index(value)
    except TypeError as error:
        raise ValueError("dataset index must be an integer") from error
    if index < 0:
        index += length
    if not 0 <= index < length:
        raise IndexError("dataset index out of range")
    return index


__all__ = [
    "PITCH_PROFILE_CONFIGS",
    "PITCH_SHUFFLE_GENERATOR_DEVICE",
    "PITCH_TRAINING_NUM_WORKERS",
    "PitchCoordinateDataset",
    "PitchDatasetMetrics",
    "PitchEpochMetrics",
    "PitchPredictionRecord",
    "PitchTrainingProfile",
    "PitchTrainingResult",
    "PitchTrainingSummary",
    "build_pitch_coordinate_datasets",
    "evaluate_pitch_dataset",
    "load_pitch_estimator_model",
    "pitch_epoch_rank",
    "save_pitch_estimator_model",
    "summarize_pitch_predictions",
    "train_pitch_estimator",
]
