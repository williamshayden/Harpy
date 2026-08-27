"""Immutable Milestone E pitch-coordinate and evaluation-suite contracts.

These schema-local models deliberately do not extend the Milestone D schema-v1
models.  Keeping the identities separate lets existing BC and PPO artifacts retain
their exact suite and codec behavior while the pitch-estimator lane uses its pinned
v2 evaluation suites.
"""

from __future__ import annotations

import hashlib
import json
import operator
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import cache

import numpy as np

from harpy.envs.models import SOURCE_MAX_CENTS, SOURCE_MIN_CENTS, TARGET_NOTE_COUNT

PITCH_DISTRIBUTION_ID = "harpy-sine-pitch-grid-v1"
PITCH_SPLIT_SCHEMA_VERSION = 1
PITCH_SUITE_SCHEMA_VERSION = 2
PITCH_SPLIT_DIGEST_SHA256 = "e7a111faa7b0722a4ef6d9998c6a69c74221854a05a3661112ff4eb72bef4ef2"

_CENTRAL_MIN_CENTS = 5_000
_CENTRAL_MAX_CENTS = 7_000
_OOD_LOWER_MAX_CENTS = 4_999
_OOD_UPPER_MIN_CENTS = 7_001
_TRAINING_COORDINATE_COUNT = 1_400
_VALIDATION_COORDINATE_COUNT = 200
_SPECTRUM_BIN_COUNT = 1_961
_SHUFFLE_SUITE_CODE = 405
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class PitchTrainerKind(StrEnum):
    """The schema-v2 trainer identity without expanding schema-v1 dispatch."""

    PITCH = "pitch"


class PitchEvaluationSuiteId(StrEnum):
    """Versioned identities for Milestone E episode evaluation suites."""

    SMOKE = "harpy-sine-pitch-e-smoke-v1"
    IID = "harpy-sine-pitch-e-iid-v1"
    OOD_LOWER = "harpy-sine-pitch-e-ood-lower-v1"
    OOD_UPPER = "harpy-sine-pitch-e-ood-upper-v1"


_SUITE_CONFIGS: tuple[tuple[PitchEvaluationSuiteId, int, str, int, str], ...] = (
    (
        PitchEvaluationSuiteId.SMOKE,
        401,
        "validation_coordinates",
        2,
        "32a2761392583677d0e0785d27965a8d139a5475117088beecb413bd1a3ca611",
    ),
    (
        PitchEvaluationSuiteId.IID,
        402,
        "iid_holdout_coordinates",
        10,
        "5b91c98d269a7e9b7319e8827b9eea74a89ab76cabf4a0478746ec3601ee73f3",
    ),
    (
        PitchEvaluationSuiteId.OOD_LOWER,
        403,
        "ood_lower_coordinates",
        8,
        "c8a04e8270e5aa54c17731cd522e0e02b7fa80fa745e5a4c6802480891e73340",
    ),
    (
        PitchEvaluationSuiteId.OOD_UPPER,
        404,
        "ood_upper_coordinates",
        8,
        "1faa402fc3fdebf6a17c758a0647455c8bc6fab129613b919faf6084d0b4da0d",
    ),
)


def _integer(value: object, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        normalized = int(operator.index(value))
    except TypeError as error:
        raise ValueError(f"{field} must be an integer") from error
    if not minimum <= normalized <= maximum:
        raise ValueError(f"{field} must be within {minimum}..{maximum}")
    return normalized


def _coordinate_tuple(value: object, field: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an iterable of coordinates")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{field} must be an iterable of coordinates") from error
    return tuple(
        _integer(coordinate, field, SOURCE_MIN_CENTS, SOURCE_MAX_CENTS) for coordinate in values
    )


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _canonical_digest(payload: dict[str, object]) -> str:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        + b"\n"
    )
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class PitchCoordinateSplit:
    """The complete ordered, seed-neutral pitch-coordinate partition."""

    schema_version: int
    distribution_id: str
    training_coordinates: tuple[int, ...]
    validation_coordinates: tuple[int, ...]
    iid_holdout_coordinates: tuple[int, ...]
    ood_lower_coordinates: tuple[int, ...]
    ood_upper_coordinates: tuple[int, ...]
    digest_sha256: str

    def __post_init__(self) -> None:
        schema_version = _integer(
            self.schema_version,
            "schema_version",
            PITCH_SPLIT_SCHEMA_VERSION,
            PITCH_SPLIT_SCHEMA_VERSION,
        )
        if self.distribution_id != PITCH_DISTRIBUTION_ID:
            raise ValueError(f"distribution_id must be {PITCH_DISTRIBUTION_ID!r}")

        training = _coordinate_tuple(self.training_coordinates, "training_coordinates")
        validation = _coordinate_tuple(self.validation_coordinates, "validation_coordinates")
        iid_holdout = _coordinate_tuple(self.iid_holdout_coordinates, "iid_holdout_coordinates")
        ood_lower = _coordinate_tuple(self.ood_lower_coordinates, "ood_lower_coordinates")
        ood_upper = _coordinate_tuple(self.ood_upper_coordinates, "ood_upper_coordinates")
        if tuple(map(len, (training, validation, iid_holdout))) != (1_400, 200, 401):
            raise ValueError("central coordinate partitions must contain 1400, 200, and 401 items")
        central_sets = tuple(map(set, (training, validation, iid_holdout)))
        if any(
            central_sets[left] & central_sets[right]
            for left in range(len(central_sets))
            for right in range(left + 1, len(central_sets))
        ):
            raise ValueError("central coordinate partitions must be disjoint")
        if set().union(*central_sets) != set(range(_CENTRAL_MIN_CENTS, _CENTRAL_MAX_CENTS + 1)):
            raise ValueError("central coordinate partitions must cover 5000..7000")
        if ood_lower != tuple(range(SOURCE_MIN_CENTS, _OOD_LOWER_MAX_CENTS + 1)):
            raise ValueError("ood_lower_coordinates must be the ordered 4800..4999 pool")
        if ood_upper != tuple(range(_OOD_UPPER_MIN_CENTS, SOURCE_MAX_CENTS + 1)):
            raise ValueError("ood_upper_coordinates must be the ordered 7001..7200 pool")

        digest = _digest(self.digest_sha256, "digest_sha256")
        expected_digest = _canonical_digest(
            _split_payload(training, validation, iid_holdout, ood_lower, ood_upper)
        )
        if expected_digest != PITCH_SPLIT_DIGEST_SHA256 or digest != PITCH_SPLIT_DIGEST_SHA256:
            raise ValueError("coordinate split must match the pinned v1 identity")

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "training_coordinates", training)
        object.__setattr__(self, "validation_coordinates", validation)
        object.__setattr__(self, "iid_holdout_coordinates", iid_holdout)
        object.__setattr__(self, "ood_lower_coordinates", ood_lower)
        object.__setattr__(self, "ood_upper_coordinates", ood_upper)
        object.__setattr__(self, "digest_sha256", digest)


@dataclass(frozen=True, slots=True)
class PitchEpisodeSpec:
    """The public reset truth for one schema-v2 pitch evaluation episode."""

    target_note_index: int
    source_pitch_cents: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_note_index",
            _integer(self.target_note_index, "target_note_index", 0, TARGET_NOTE_COUNT - 1),
        )
        object.__setattr__(
            self,
            "source_pitch_cents",
            _integer(
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
        """Return freshly owned options for ``SinePitchEnv.reset``."""
        return {
            "target_note_index": self.target_note_index,
            "source_pitch_cents": self.source_pitch_cents,
        }


@dataclass(frozen=True, slots=True)
class PitchEvaluationSuite:
    """One ordered schema-v2 pitch evaluation suite and its identity digest."""

    schema_version: int
    suite_id: PitchEvaluationSuiteId
    suite_seed: int
    episodes: tuple[PitchEpisodeSpec, ...]
    digest_sha256: str

    def __post_init__(self) -> None:
        schema_version = _integer(
            self.schema_version,
            "schema_version",
            PITCH_SUITE_SCHEMA_VERSION,
            PITCH_SUITE_SCHEMA_VERSION,
        )
        if not isinstance(self.suite_id, PitchEvaluationSuiteId):
            raise ValueError("suite_id must be a PitchEvaluationSuiteId")
        suite_seed = _integer(self.suite_seed, "suite_seed", 0, 2**63 - 1)
        if isinstance(self.episodes, (str, bytes)):
            raise ValueError("episodes must be an iterable of PitchEpisodeSpec")
        try:
            episodes = tuple(self.episodes)
        except TypeError as error:
            raise ValueError("episodes must be an iterable of PitchEpisodeSpec") from error
        if not episodes or not all(isinstance(episode, PitchEpisodeSpec) for episode in episodes):
            raise ValueError("episodes must contain PitchEpisodeSpec values")
        if len({episode.pair for episode in episodes}) != len(episodes):
            raise ValueError("episodes must not contain duplicate target/source pairs")
        digest = _digest(self.digest_sha256, "digest_sha256")
        expected_digest = _canonical_digest(_suite_payload(self.suite_id, suite_seed, episodes))
        config = next(config for config in _SUITE_CONFIGS if config[0] is self.suite_id)
        _suite_id, pinned_seed, _source_field, per_target, pinned_digest = config
        if (
            suite_seed != pinned_seed
            or len(episodes) != TARGET_NOTE_COUNT * per_target
            or expected_digest != pinned_digest
            or digest != pinned_digest
        ):
            raise ValueError("evaluation suite must match its pinned v2 identity")

        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "suite_seed", suite_seed)
        object.__setattr__(self, "episodes", episodes)
        object.__setattr__(self, "digest_sha256", digest)


def _split_payload(
    training: tuple[int, ...],
    validation: tuple[int, ...],
    iid_holdout: tuple[int, ...],
    ood_lower: tuple[int, ...],
    ood_upper: tuple[int, ...],
) -> dict[str, object]:
    return {
        "schema_version": PITCH_SPLIT_SCHEMA_VERSION,
        "distribution_id": PITCH_DISTRIBUTION_ID,
        "training_coordinates": list(training),
        "validation_coordinates": list(validation),
        "iid_holdout_coordinates": list(iid_holdout),
        "ood_lower_coordinates": list(ood_lower),
        "ood_upper_coordinates": list(ood_upper),
    }


@cache
def pitch_coordinate_split() -> PitchCoordinateSplit:
    """Return the pinned ordered pitch-coordinate split."""
    ordered = tuple(
        sorted(
            range(_CENTRAL_MIN_CENTS, _CENTRAL_MAX_CENTS + 1),
            key=lambda coordinate: (
                hashlib.sha256(f"{PITCH_DISTRIBUTION_ID}:{coordinate}".encode("ascii")).digest(),
                coordinate,
            ),
        )
    )
    training = ordered[:_TRAINING_COORDINATE_COUNT]
    validation_end = _TRAINING_COORDINATE_COUNT + _VALIDATION_COORDINATE_COUNT
    validation = ordered[_TRAINING_COORDINATE_COUNT:validation_end]
    iid_holdout = ordered[validation_end:]
    ood_lower = tuple(range(SOURCE_MIN_CENTS, _OOD_LOWER_MAX_CENTS + 1))
    ood_upper = tuple(range(_OOD_UPPER_MIN_CENTS, SOURCE_MAX_CENTS + 1))
    digest = _canonical_digest(
        _split_payload(training, validation, iid_holdout, ood_lower, ood_upper)
    )
    if digest != PITCH_SPLIT_DIGEST_SHA256:
        raise RuntimeError("pinned pitch-coordinate split digest does not match construction")
    return PitchCoordinateSplit(
        schema_version=PITCH_SPLIT_SCHEMA_VERSION,
        distribution_id=PITCH_DISTRIBUTION_ID,
        training_coordinates=training,
        validation_coordinates=validation,
        iid_holdout_coordinates=iid_holdout,
        ood_lower_coordinates=ood_lower,
        ood_upper_coordinates=ood_upper,
        digest_sha256=digest,
    )


def _suite_payload(
    suite_id: PitchEvaluationSuiteId,
    suite_seed: int,
    episodes: tuple[PitchEpisodeSpec, ...],
) -> dict[str, object]:
    return {
        "schema_version": PITCH_SUITE_SCHEMA_VERSION,
        "suite_id": suite_id.value,
        "suite_seed": suite_seed,
        "episodes": [
            {
                "episode_index": episode_index,
                "target_note_index": episode.target_note_index,
                "source_pitch_cents": episode.source_pitch_cents,
            }
            for episode_index, episode in enumerate(episodes)
        ],
    }


def _suite_episodes(
    *, suite_code: int, source_pool: tuple[int, ...], per_target: int
) -> tuple[PitchEpisodeSpec, ...]:
    episodes: list[PitchEpisodeSpec] = []
    for target_note_index in range(TARGET_NOTE_COUNT):
        target_pitch_cents = 4_800 + 100 * target_note_index
        eligible = tuple(
            source_pitch_cents
            for source_pitch_cents in sorted(source_pool)
            if abs(source_pitch_cents - target_pitch_cents) > 5
        )
        selected = np.random.default_rng(
            np.random.SeedSequence([1, suite_code, target_note_index])
        ).choice(np.asarray(eligible, dtype=np.int64), size=per_target, replace=False)
        episodes.extend(
            PitchEpisodeSpec(target_note_index, int(source_pitch_cents))
            for source_pitch_cents in selected
        )
    return tuple(episodes)


@cache
def _fixed_pitch_evaluation_suites() -> tuple[PitchEvaluationSuite, ...]:
    split = pitch_coordinate_split()
    suites: list[PitchEvaluationSuite] = []
    for suite_id, suite_code, source_field, per_target, expected_digest in _SUITE_CONFIGS:
        episodes = _suite_episodes(
            suite_code=suite_code,
            source_pool=getattr(split, source_field),
            per_target=per_target,
        )
        digest = _canonical_digest(_suite_payload(suite_id, suite_code, episodes))
        if digest != expected_digest:
            raise RuntimeError(f"pinned {suite_id.value} digest does not match construction")
        suites.append(
            PitchEvaluationSuite(
                schema_version=PITCH_SUITE_SCHEMA_VERSION,
                suite_id=suite_id,
                suite_seed=suite_code,
                episodes=episodes,
                digest_sha256=digest,
            )
        )
    return tuple(suites)


def fixed_pitch_evaluation_suite(suite_id: PitchEvaluationSuiteId) -> PitchEvaluationSuite:
    """Return the pinned schema-v2 evaluation suite for ``suite_id``."""
    if not isinstance(suite_id, PitchEvaluationSuiteId):
        raise ValueError("suite_id must be a PitchEvaluationSuiteId")
    return _fixed_pitch_evaluation_suites()[tuple(PitchEvaluationSuiteId).index(suite_id)]


@cache
def iid_shuffled_spectrum_permutation() -> tuple[int, ...]:
    """Return the one pinned v2 IID spectrum permutation."""
    iid_digest = bytes.fromhex(
        fixed_pitch_evaluation_suite(PitchEvaluationSuiteId.IID).digest_sha256
    )
    digest_words = tuple(
        int.from_bytes(iid_digest[index : index + 4], "big")
        for index in range(0, len(iid_digest), 4)
    )
    permutation = np.random.default_rng(
        np.random.SeedSequence([1, _SHUFFLE_SUITE_CODE, *digest_words])
    ).permutation(_SPECTRUM_BIN_COUNT)
    return tuple(int(index) for index in permutation)


__all__ = [
    "PITCH_DISTRIBUTION_ID",
    "PITCH_SPLIT_DIGEST_SHA256",
    "PITCH_SPLIT_SCHEMA_VERSION",
    "PITCH_SUITE_SCHEMA_VERSION",
    "PitchCoordinateSplit",
    "PitchEpisodeSpec",
    "PitchEvaluationSuite",
    "PitchEvaluationSuiteId",
    "PitchTrainerKind",
    "fixed_pitch_evaluation_suite",
    "iid_shuffled_spectrum_permutation",
    "pitch_coordinate_split",
]
