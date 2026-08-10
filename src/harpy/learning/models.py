"""Immutable dependency-free contracts for learned sine-policy training."""

from __future__ import annotations

import math
import operator
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from harpy.envs.models import (
    SOURCE_MAX_CENTS,
    SOURCE_MIN_CENTS,
    TARGET_NOTE_COUNT,
)

ENVIRONMENT_ID = "Harpy/SinePitch-v0"
ENVIRONMENT_CONTRACT_ID = "harpy-sine-pitch-v0-contract-v1"
SPECTRUM_GRID_ID = "harpy-sine-spectrum-grid-v1"
PREPROCESSING_SCHEMA_ID = "harpy-sine-policy-observation-v1"
ARCHITECTURE_SCHEMA_ID = "harpy-sine-policy-conv-v1"

type JSONScalar = bool | int | float | str | None
type JSONValue = JSONScalar | list[JSONValue] | dict[str, JSONValue]

_SUITE_SCHEMA_VERSION = 1
_TRAIN_DISTRIBUTION_ID = "harpy-sine-policy-train-v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ProfileName(StrEnum):
    """The closed set of checked-in training profiles."""

    SMOKE = "smoke"
    CHECKPOINT = "checkpoint"


class TrainerKind(StrEnum):
    """The closed set of supported learned-policy trainers."""

    BC = "bc"
    PPO = "ppo"


class DeviceName(StrEnum):
    """The closed set of user-selectable training devices."""

    CPU = "cpu"
    CUDA = "cuda"


class EvaluationSuiteId(StrEnum):
    """Versioned identities for the fixed final evaluation suites."""

    SMOKE = "harpy-sine-policy-eval-smoke-v1"
    IID = "harpy-sine-policy-eval-iid-v1"
    REGISTER_OOD = "harpy-sine-policy-eval-register-ood-v1"


def _integer(value: object, field: str, *, minimum: int | None = None) -> int:
    """Return an owned integer scalar while rejecting booleans."""
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        normalized = int(operator.index(value))
    except TypeError as error:
        raise ValueError(f"{field} must be an integer") from error
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return normalized


def _bounded_integer(value: object, field: str, minimum: int, maximum: int) -> int:
    """Return an owned integer scalar inside an inclusive range."""
    normalized = _integer(value, field)
    if not minimum <= normalized <= maximum:
        raise ValueError(f"{field} must be within {minimum}..{maximum}")
    return normalized


def _finite_float(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Return an owned finite float inside optional inclusive bounds."""
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{field} must be a finite number")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return normalized


def _episode_tuple(value: object, field: str) -> tuple[EpisodeSpec, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a tuple of EpisodeSpec")
    try:
        episodes = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{field} must be a tuple of EpisodeSpec") from error
    if not all(isinstance(episode, EpisodeSpec) for episode in episodes):
        raise ValueError(f"{field} must contain only EpisodeSpec values")
    return episodes


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    """The two exact values injected through an environment reset."""

    target_note_index: int
    source_pitch_cents: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_note_index",
            _bounded_integer(self.target_note_index, "target_note_index", 0, TARGET_NOTE_COUNT - 1),
        )
        object.__setattr__(
            self,
            "source_pitch_cents",
            _bounded_integer(
                self.source_pitch_cents,
                "source_pitch_cents",
                SOURCE_MIN_CENTS,
                SOURCE_MAX_CENTS,
            ),
        )

    @property
    def pair(self) -> tuple[int, int]:
        """Return the immutable target/source identity pair."""
        return self.target_note_index, self.source_pitch_cents

    def reset_options(self) -> dict[str, int]:
        """Return a freshly owned options mapping for ``SinePitchEnv.reset``."""
        return {
            "target_note_index": self.target_note_index,
            "source_pitch_cents": self.source_pitch_cents,
        }


@dataclass(frozen=True, slots=True)
class EpisodeSuite:
    """A complete ordered fixed evaluation suite with its pinned digest."""

    schema_version: int
    suite_id: EvaluationSuiteId
    suite_seed: int
    episodes: tuple[EpisodeSpec, ...]
    digest_sha256: str

    def __post_init__(self) -> None:
        schema_version = _integer(self.schema_version, "schema_version", minimum=1)
        if schema_version != _SUITE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SUITE_SCHEMA_VERSION}")
        if not isinstance(self.suite_id, EvaluationSuiteId):
            raise ValueError("suite_id must be an EvaluationSuiteId")
        episodes = _episode_tuple(self.episodes, "episodes")
        if not episodes:
            raise ValueError("episodes must not be empty")
        if len({episode.pair for episode in episodes}) != len(episodes):
            raise ValueError("episodes must not contain duplicate pairs")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "suite_seed", _integer(self.suite_seed, "suite_seed", minimum=0))
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "digest_sha256", _digest(self.digest_sha256, "digest_sha256"))


@dataclass(frozen=True, slots=True)
class BCEpisodeSplits:
    """The disjoint finite behavior-cloning train and validation samples."""

    distribution_id: str
    run_seed: int
    profile: ProfileName
    training: tuple[EpisodeSpec, ...]
    validation: tuple[EpisodeSpec, ...]
    training_digest_sha256: str
    validation_digest_sha256: str

    def __post_init__(self) -> None:
        if self.distribution_id != _TRAIN_DISTRIBUTION_ID:
            raise ValueError(f"distribution_id must be {_TRAIN_DISTRIBUTION_ID!r}")
        if not isinstance(self.profile, ProfileName):
            raise ValueError("profile must be a ProfileName")
        training = _episode_tuple(self.training, "training")
        validation = _episode_tuple(self.validation, "validation")
        training_pairs = {episode.pair for episode in training}
        validation_pairs = {episode.pair for episode in validation}
        if len(training_pairs) != len(training):
            raise ValueError("training must not contain duplicate pairs")
        if len(validation_pairs) != len(validation):
            raise ValueError("validation must not contain duplicate pairs")
        if training_pairs & validation_pairs:
            raise ValueError("training and validation must be disjoint")
        expected = PROFILE_CONFIGS[self.profile].bc
        if len(training) != expected.train_episodes:
            raise ValueError("training must match the profile train_episodes")
        if len(validation) != expected.validation_episodes:
            raise ValueError("validation must match the profile validation_episodes")
        object.__setattr__(self, "run_seed", _integer(self.run_seed, "run_seed", minimum=0))
        object.__setattr__(self, "training", training)
        object.__setattr__(self, "validation", validation)
        object.__setattr__(
            self,
            "training_digest_sha256",
            _digest(self.training_digest_sha256, "training_digest_sha256"),
        )
        object.__setattr__(
            self,
            "validation_digest_sha256",
            _digest(self.validation_digest_sha256, "validation_digest_sha256"),
        )


@dataclass(frozen=True, slots=True)
class BCProfile:
    """Checked-in behavior-cloning optimizer and dataset settings."""

    train_episodes: int
    validation_episodes: int
    max_epochs: int
    early_stopping_patience: int | None
    batch_size: int
    optimizer: str
    learning_rate: float
    weight_decay: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "train_episodes", _integer(self.train_episodes, "train_episodes", minimum=1)
        )
        object.__setattr__(
            self,
            "validation_episodes",
            _integer(self.validation_episodes, "validation_episodes", minimum=1),
        )
        object.__setattr__(self, "max_epochs", _integer(self.max_epochs, "max_epochs", minimum=1))
        if self.early_stopping_patience is not None:
            object.__setattr__(
                self,
                "early_stopping_patience",
                _integer(self.early_stopping_patience, "early_stopping_patience", minimum=1),
            )
        object.__setattr__(self, "batch_size", _integer(self.batch_size, "batch_size", minimum=1))
        if self.optimizer != "AdamW":
            raise ValueError("optimizer must be AdamW")
        object.__setattr__(
            self,
            "learning_rate",
            _finite_float(
                self.learning_rate,
                "learning_rate",
                minimum=float.fromhex("0x0.0000000000001p-1022"),
            ),
        )
        object.__setattr__(
            self,
            "weight_decay",
            _finite_float(self.weight_decay, "weight_decay", minimum=0.0),
        )


@dataclass(frozen=True, slots=True)
class PPOProfile:
    """Checked-in PPO rollout and optimizer settings."""

    total_timesteps: int
    n_steps: int
    batch_size: int
    n_epochs: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float

    def __post_init__(self) -> None:
        total_timesteps = _integer(self.total_timesteps, "total_timesteps", minimum=1)
        n_steps = _integer(self.n_steps, "n_steps", minimum=1)
        batch_size = _integer(self.batch_size, "batch_size", minimum=1)
        if total_timesteps % n_steps != 0:
            raise ValueError("total_timesteps must be divisible by n_steps")
        if batch_size > n_steps or n_steps % batch_size != 0:
            raise ValueError("batch_size must evenly divide n_steps")
        gamma = _finite_float(self.gamma, "gamma", minimum=0.0, maximum=1.0)
        if gamma != 1.0:
            raise ValueError("gamma must be 1.0")
        object.__setattr__(self, "total_timesteps", total_timesteps)
        object.__setattr__(self, "n_steps", n_steps)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "n_epochs", _integer(self.n_epochs, "n_epochs", minimum=1))
        object.__setattr__(
            self,
            "learning_rate",
            _finite_float(
                self.learning_rate,
                "learning_rate",
                minimum=float.fromhex("0x0.0000000000001p-1022"),
            ),
        )
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(
            self,
            "gae_lambda",
            _finite_float(self.gae_lambda, "gae_lambda", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "clip_range",
            _finite_float(
                self.clip_range, "clip_range", minimum=float.fromhex("0x0.0000000000001p-1022")
            ),
        )
        object.__setattr__(self, "ent_coef", _finite_float(self.ent_coef, "ent_coef", minimum=0.0))
        object.__setattr__(self, "vf_coef", _finite_float(self.vf_coef, "vf_coef", minimum=0.0))


@dataclass(frozen=True, slots=True)
class TrainingProfile:
    """The paired BC/PPO settings and final suites for one named profile."""

    bc: BCProfile
    ppo: PPOProfile
    evaluation_suites: tuple[EvaluationSuiteId, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.bc, BCProfile):
            raise ValueError("bc must be a BCProfile")
        if not isinstance(self.ppo, PPOProfile):
            raise ValueError("ppo must be a PPOProfile")
        if isinstance(self.evaluation_suites, (str, bytes)):
            raise ValueError("evaluation_suites must be EvaluationSuiteId values")
        try:
            evaluation_suites = tuple(self.evaluation_suites)
        except TypeError as error:
            raise ValueError("evaluation_suites must be EvaluationSuiteId values") from error
        if not evaluation_suites or not all(
            isinstance(suite_id, EvaluationSuiteId) for suite_id in evaluation_suites
        ):
            raise ValueError("evaluation_suites must contain EvaluationSuiteId values")
        if len(set(evaluation_suites)) != len(evaluation_suites):
            raise ValueError("evaluation_suites must not contain duplicates")
        object.__setattr__(self, "evaluation_suites", evaluation_suites)


PROFILE_CONFIGS: Mapping[ProfileName, TrainingProfile] = MappingProxyType(
    {
        ProfileName.SMOKE: TrainingProfile(
            bc=BCProfile(
                train_episodes=128,
                validation_episodes=64,
                max_epochs=2,
                early_stopping_patience=None,
                batch_size=128,
                optimizer="AdamW",
                learning_rate=3e-4,
                weight_decay=1e-4,
            ),
            ppo=PPOProfile(
                total_timesteps=2_048,
                n_steps=256,
                batch_size=64,
                n_epochs=10,
                learning_rate=3e-4,
                gamma=1.0,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
            ),
            evaluation_suites=(EvaluationSuiteId.SMOKE,),
        ),
        ProfileName.CHECKPOINT: TrainingProfile(
            bc=BCProfile(
                train_episodes=4_096,
                validation_episodes=512,
                max_epochs=50,
                early_stopping_patience=5,
                batch_size=256,
                optimizer="AdamW",
                learning_rate=3e-4,
                weight_decay=1e-4,
            ),
            ppo=PPOProfile(
                total_timesteps=256_000,
                n_steps=1_024,
                batch_size=256,
                n_epochs=10,
                learning_rate=3e-4,
                gamma=1.0,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
            ),
            evaluation_suites=(EvaluationSuiteId.IID, EvaluationSuiteId.REGISTER_OOD),
        ),
    }
)


@dataclass(frozen=True, slots=True)
class BCEpochMetrics:
    """Finite validation metrics recorded after a full BC epoch."""

    epoch: int
    training_loss: float
    validation_loss: float
    validation_accuracy: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "epoch", _integer(self.epoch, "epoch", minimum=1))
        object.__setattr__(
            self, "training_loss", _finite_float(self.training_loss, "training_loss", minimum=0.0)
        )
        object.__setattr__(
            self,
            "validation_loss",
            _finite_float(self.validation_loss, "validation_loss", minimum=0.0),
        )
        object.__setattr__(
            self,
            "validation_accuracy",
            _finite_float(
                self.validation_accuracy, "validation_accuracy", minimum=0.0, maximum=1.0
            ),
        )


@dataclass(frozen=True, slots=True)
class BCTrainingSummary:
    """The selected epoch and complete ordered behavior-cloning history."""

    history: tuple[BCEpochMetrics, ...]
    selected_epoch: int
    training_examples: int
    validation_examples: int
    training_wall_time_seconds: float

    def __post_init__(self) -> None:
        if isinstance(self.history, (str, bytes)):
            raise ValueError("history must be BCEpochMetrics values")
        try:
            history = tuple(self.history)
        except TypeError as error:
            raise ValueError("history must be BCEpochMetrics values") from error
        if not history or not all(isinstance(metric, BCEpochMetrics) for metric in history):
            raise ValueError("history must contain BCEpochMetrics values")
        epochs = tuple(metric.epoch for metric in history)
        if epochs != tuple(range(1, len(history) + 1)):
            raise ValueError("history must use consecutive epoch ordering")
        selected_epoch = _integer(self.selected_epoch, "selected_epoch", minimum=1)
        if selected_epoch not in epochs:
            raise ValueError("selected_epoch must be present in history")
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "selected_epoch", selected_epoch)
        object.__setattr__(
            self,
            "training_examples",
            _integer(self.training_examples, "training_examples", minimum=1),
        )
        object.__setattr__(
            self,
            "validation_examples",
            _integer(self.validation_examples, "validation_examples", minimum=1),
        )
        object.__setattr__(
            self,
            "training_wall_time_seconds",
            _finite_float(
                self.training_wall_time_seconds, "training_wall_time_seconds", minimum=0.0
            ),
        )


@dataclass(frozen=True, slots=True)
class PPOTrainingSummary:
    """The exact completed environment-step count for one PPO run."""

    requested_environment_steps: int
    completed_environment_steps: int
    training_wall_time_seconds: float

    def __post_init__(self) -> None:
        requested = _integer(
            self.requested_environment_steps,
            "requested_environment_steps",
            minimum=1,
        )
        completed = _integer(
            self.completed_environment_steps,
            "completed_environment_steps",
            minimum=1,
        )
        if requested != completed:
            raise ValueError("completed_environment_steps must equal requested_environment_steps")
        object.__setattr__(self, "requested_environment_steps", requested)
        object.__setattr__(self, "completed_environment_steps", completed)
        object.__setattr__(
            self,
            "training_wall_time_seconds",
            _finite_float(
                self.training_wall_time_seconds, "training_wall_time_seconds", minimum=0.0
            ),
        )


__all__ = [
    "ARCHITECTURE_SCHEMA_ID",
    "ENVIRONMENT_CONTRACT_ID",
    "ENVIRONMENT_ID",
    "PREPROCESSING_SCHEMA_ID",
    "PROFILE_CONFIGS",
    "SPECTRUM_GRID_ID",
    "BCEpisodeSplits",
    "BCEpochMetrics",
    "BCProfile",
    "BCTrainingSummary",
    "DeviceName",
    "EpisodeSpec",
    "EpisodeSuite",
    "EvaluationSuiteId",
    "JSONScalar",
    "JSONValue",
    "PPOProfile",
    "PPOTrainingSummary",
    "ProfileName",
    "TrainerKind",
    "TrainingProfile",
]
