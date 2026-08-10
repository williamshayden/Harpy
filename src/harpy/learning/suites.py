"""Canonical deterministic episode suites for learned sine-policy training."""

from __future__ import annotations

import hashlib
import json
import operator
from functools import cache

import numpy as np

from harpy.envs.models import SOURCE_MAX_CENTS, SOURCE_MIN_CENTS, TARGET_NOTE_COUNT
from harpy.learning.models import (
    PROFILE_CONFIGS,
    BCEpisodeSplits,
    EpisodeSpec,
    EpisodeSuite,
    EvaluationSuiteId,
    ProfileName,
)

SUITE_SCHEMA_VERSION = 1
TRAIN_DISTRIBUTION_ID = "harpy-sine-policy-train-v1"

_BC_TRAIN_CODE = 201
_BC_VALIDATION_CODE = 202
_PPO_TRAIN_CODE = 203
_CENTRAL_SOURCE_MIN = 5_000
_CENTRAL_SOURCE_MAX = 7_000
_LOWER_SOURCE_MIN = SOURCE_MIN_CENTS
_LOWER_SOURCE_MAX = 4_999
_UPPER_SOURCE_MIN = 7_001
_UPPER_SOURCE_MAX = SOURCE_MAX_CENTS
_SUITE_SPECS: dict[EvaluationSuiteId, tuple[int, int]] = {
    EvaluationSuiteId.SMOKE: (32, 202_608_100),
    EvaluationSuiteId.IID: (256, 202_608_101),
    EvaluationSuiteId.REGISTER_OOD: (256, 202_608_102),
}


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer")
    try:
        normalized = int(operator.index(value))
    except TypeError as error:
        raise ValueError(f"{field} must be a non-negative integer") from error
    if normalized < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return normalized


def _require_suite_id(value: object) -> EvaluationSuiteId:
    if not isinstance(value, EvaluationSuiteId):
        raise ValueError("suite_id must be an EvaluationSuiteId")
    return value


def _target_schedule(*, namespace: int, run_seed: int | None, episode_index: int) -> int:
    block_index, position = divmod(episode_index, TARGET_NOTE_COUNT)
    entropy = [1, namespace]
    if run_seed is not None:
        entropy.append(run_seed)
    entropy.append(block_index)
    permutation = np.random.default_rng(np.random.SeedSequence(entropy)).permutation(
        TARGET_NOTE_COUNT
    )
    return int(permutation[position])


def _target_pitch_cents(target_note_index: int) -> int:
    return 100 * (48 + target_note_index)


def _eligible_sources(
    *,
    target_note_index: int,
    minimum: int,
    maximum: int,
    excluded_pairs: set[tuple[int, int]],
) -> list[int]:
    target_pitch_cents = _target_pitch_cents(target_note_index)
    return [
        source_pitch_cents
        for source_pitch_cents in range(minimum, maximum + 1)
        if abs(source_pitch_cents - target_pitch_cents) > 5
        and (target_note_index, source_pitch_cents) not in excluded_pairs
    ]


def _draw_source(seed_sequence: np.random.SeedSequence, eligible: list[int]) -> int:
    if not eligible:
        raise RuntimeError("episode source domain was exhausted")
    return int(
        np.random.default_rng(seed_sequence).choice(np.asarray(sorted(eligible), dtype=np.int64))
    )


def _suite_digest_payload(
    *,
    suite_id: EvaluationSuiteId,
    suite_seed: int,
    episodes: tuple[EpisodeSpec, ...],
) -> bytes:
    payload = {
        "schema_version": SUITE_SCHEMA_VERSION,
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
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        + b"\n"
    )


def suite_digest(
    *,
    suite_id: EvaluationSuiteId,
    suite_seed: int,
    episodes: tuple[EpisodeSpec, ...],
) -> str:
    """Return the versioned SHA-256 identity for an ordered suite."""
    normalized_suite_id = _require_suite_id(suite_id)
    normalized_seed = _integer(suite_seed, "suite_seed")
    try:
        normalized_episodes = tuple(episodes)
    except TypeError as error:
        raise ValueError("episodes must be EpisodeSpec values") from error
    if not all(isinstance(episode, EpisodeSpec) for episode in normalized_episodes):
        raise ValueError("episodes must contain EpisodeSpec values")
    return hashlib.sha256(
        _suite_digest_payload(
            suite_id=normalized_suite_id,
            suite_seed=normalized_seed,
            episodes=normalized_episodes,
        )
    ).hexdigest()


def _fixed_targets(*, suite_seed: int, count: int) -> tuple[int, ...]:
    return tuple(
        _target_schedule(
            namespace=suite_seed,
            run_seed=None,
            episode_index=episode_index,
        )
        for episode_index in range(count)
    )


def _fixed_sources(
    *,
    suite_id: EvaluationSuiteId,
    suite_seed: int,
    targets: tuple[int, ...],
    excluded_pairs: set[tuple[int, int]],
) -> tuple[EpisodeSpec, ...]:
    target_totals = {target: targets.count(target) for target in range(TARGET_NOTE_COUNT)}
    starts_lower: dict[int, bool] = {}
    if suite_id is EvaluationSuiteId.REGISTER_OOD:
        eleven_occurrence_targets = sorted(
            target for target, count in target_totals.items() if count == 11
        )
        starts_lower = {
            target: target not in eleven_occurrence_targets[3:]
            for target in range(TARGET_NOTE_COUNT)
        }

    occurrences: dict[int, int] = {target: 0 for target in range(TARGET_NOTE_COUNT)}
    episodes: list[EpisodeSpec] = []
    for episode_index, target_note_index in enumerate(targets):
        if suite_id is EvaluationSuiteId.REGISTER_OOD:
            is_lower = starts_lower[target_note_index] == (occurrences[target_note_index] % 2 == 0)
            if is_lower:
                minimum, maximum, band_code = (
                    _LOWER_SOURCE_MIN,
                    _LOWER_SOURCE_MAX,
                    0,
                )
            else:
                minimum, maximum, band_code = (
                    _UPPER_SOURCE_MIN,
                    _UPPER_SOURCE_MAX,
                    1,
                )
        else:
            minimum, maximum, band_code = _CENTRAL_SOURCE_MIN, _CENTRAL_SOURCE_MAX, 0
        eligible = _eligible_sources(
            target_note_index=target_note_index,
            minimum=minimum,
            maximum=maximum,
            excluded_pairs=excluded_pairs,
        )
        source_pitch_cents = _draw_source(
            np.random.SeedSequence([1, suite_seed, episode_index, target_note_index, band_code]),
            eligible,
        )
        episode = EpisodeSpec(
            target_note_index=target_note_index,
            source_pitch_cents=source_pitch_cents,
        )
        episodes.append(episode)
        excluded_pairs.add(episode.pair)
        occurrences[target_note_index] += 1
    return tuple(episodes)


@cache
def _fixed_suites() -> tuple[EpisodeSuite, ...]:
    suites: list[EpisodeSuite] = []
    for suite_id in EvaluationSuiteId:
        count, suite_seed = _SUITE_SPECS[suite_id]
        excluded_pairs: set[tuple[int, int]] = set()
        episodes = _fixed_sources(
            suite_id=suite_id,
            suite_seed=suite_seed,
            targets=_fixed_targets(suite_seed=suite_seed, count=count),
            excluded_pairs=excluded_pairs,
        )
        suites.append(
            EpisodeSuite(
                schema_version=SUITE_SCHEMA_VERSION,
                suite_id=suite_id,
                suite_seed=suite_seed,
                episodes=episodes,
                digest_sha256=suite_digest(
                    suite_id=suite_id,
                    suite_seed=suite_seed,
                    episodes=episodes,
                ),
            )
        )
    return tuple(suites)


def fixed_evaluation_suite(suite_id: EvaluationSuiteId) -> EpisodeSuite:
    """Return the immutable deterministic final evaluation suite for ``suite_id``."""
    normalized_suite_id = _require_suite_id(suite_id)
    return _fixed_suites()[tuple(EvaluationSuiteId).index(normalized_suite_id)]


def _fixed_pairs() -> set[tuple[int, int]]:
    return {episode.pair for suite in _fixed_suites() for episode in suite.episodes}


def _bc_digest(
    *,
    split_name: str,
    profile: ProfileName,
    run_seed: int,
    episodes: tuple[EpisodeSpec, ...],
) -> str:
    payload = {
        "schema_version": SUITE_SCHEMA_VERSION,
        "distribution_id": TRAIN_DISTRIBUTION_ID,
        "split": split_name,
        "profile": profile.value,
        "run_seed": run_seed,
        "episodes": [
            {
                "episode_index": episode_index,
                "target_note_index": episode.target_note_index,
                "source_pitch_cents": episode.source_pitch_cents,
            }
            for episode_index, episode in enumerate(episodes)
        ],
    }
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        + b"\n"
    )
    return hashlib.sha256(encoded).hexdigest()


def _bc_split_episodes(
    *,
    split_code: int,
    run_seed: int,
    count: int,
    excluded_pairs: set[tuple[int, int]],
) -> tuple[EpisodeSpec, ...]:
    available_by_target = {
        target_note_index: _eligible_sources(
            target_note_index=target_note_index,
            minimum=_CENTRAL_SOURCE_MIN,
            maximum=_CENTRAL_SOURCE_MAX,
            excluded_pairs=excluded_pairs,
        )
        for target_note_index in range(TARGET_NOTE_COUNT)
    }
    episodes: list[EpisodeSpec] = []
    for episode_index in range(count):
        target_note_index = _target_schedule(
            namespace=split_code,
            run_seed=run_seed,
            episode_index=episode_index,
        )
        eligible = available_by_target[target_note_index]
        source_pitch_cents = _draw_source(
            np.random.SeedSequence([1, split_code, run_seed, episode_index, target_note_index]),
            eligible,
        )
        episode = EpisodeSpec(
            target_note_index=target_note_index,
            source_pitch_cents=source_pitch_cents,
        )
        episodes.append(episode)
        excluded_pairs.add(episode.pair)
        eligible.remove(source_pitch_cents)
    return tuple(episodes)


def build_bc_episode_splits(*, profile: ProfileName, run_seed: int) -> BCEpisodeSplits:
    """Build disjoint, finite BC splits without replacement for a profile and seed."""
    if not isinstance(profile, ProfileName):
        raise ValueError("profile must be a ProfileName")
    normalized_run_seed = _integer(run_seed, "run_seed")
    config = PROFILE_CONFIGS[profile].bc
    excluded_pairs = _fixed_pairs()
    training = _bc_split_episodes(
        split_code=_BC_TRAIN_CODE,
        run_seed=normalized_run_seed,
        count=config.train_episodes,
        excluded_pairs=excluded_pairs,
    )
    validation = _bc_split_episodes(
        split_code=_BC_VALIDATION_CODE,
        run_seed=normalized_run_seed,
        count=config.validation_episodes,
        excluded_pairs=excluded_pairs,
    )
    return BCEpisodeSplits(
        distribution_id=TRAIN_DISTRIBUTION_ID,
        run_seed=normalized_run_seed,
        profile=profile,
        training=training,
        validation=validation,
        training_digest_sha256=_bc_digest(
            split_name="training",
            profile=profile,
            run_seed=normalized_run_seed,
            episodes=training,
        ),
        validation_digest_sha256=_bc_digest(
            split_name="validation",
            profile=profile,
            run_seed=normalized_run_seed,
            episodes=validation,
        ),
    )


def ppo_training_episode(*, run_seed: int, episode_index: int) -> EpisodeSpec:
    """Return one repeatable PPO training episode for an absolute episode index."""
    normalized_run_seed = _integer(run_seed, "run_seed")
    normalized_episode_index = _integer(episode_index, "episode_index")
    target_note_index = _target_schedule(
        namespace=_PPO_TRAIN_CODE,
        run_seed=normalized_run_seed,
        episode_index=normalized_episode_index,
    )
    source_pitch_cents = _draw_source(
        np.random.SeedSequence(
            [1, _PPO_TRAIN_CODE, normalized_run_seed, normalized_episode_index, target_note_index]
        ),
        _eligible_sources(
            target_note_index=target_note_index,
            minimum=_CENTRAL_SOURCE_MIN,
            maximum=_CENTRAL_SOURCE_MAX,
            excluded_pairs=_fixed_pairs(),
        ),
    )
    return EpisodeSpec(
        target_note_index=target_note_index,
        source_pitch_cents=source_pitch_cents,
    )


__all__ = [
    "SUITE_SCHEMA_VERSION",
    "TRAIN_DISTRIBUTION_ID",
    "build_bc_episode_splits",
    "fixed_evaluation_suite",
    "ppo_training_episode",
    "suite_digest",
]
