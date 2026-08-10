"""Contract tests for canonical learned-policy episode generation."""

from __future__ import annotations

from collections import Counter

import pytest

from harpy.learning.models import PROFILE_CONFIGS, EvaluationSuiteId, ProfileName
from harpy.learning.suites import (
    TRAIN_DISTRIBUTION_ID,
    build_bc_episode_splits,
    fixed_evaluation_suite,
    ppo_training_episode,
    suite_digest,
)

EXPECTED_BC_DIGESTS = {
    ProfileName.SMOKE: (
        "ac43951b718486402500191dcbee7b41ef7844dfe16fbd8fb727611cba6a0833",
        "d52d911cde6b941ff329f0531ae168d2867ef06739f7ce93a4cc9439b55414f0",
    ),
    ProfileName.CHECKPOINT: (
        "f9664dc49d64b9e87e9bf5b0a0e823d7c2e78243b9c380bcbc3e10ecf5bd7a26",
        "3904604026b8738cc27801505340cc6517f0dd452cdcf8965d1926cf8f9e964f",
    ),
}


def _target_pitch_cents(target_note_index: int) -> int:
    return 100 * (48 + target_note_index)


@pytest.mark.parametrize(
    ("suite_id", "count", "seed", "digest"),
    [
        (
            EvaluationSuiteId.SMOKE,
            32,
            202_608_100,
            "de8033b443976623b67077e84205795a02278bc53ada3611f0fed628139e75b2",
        ),
        (
            EvaluationSuiteId.IID,
            256,
            202_608_101,
            "302be5ee0eb1556391d60646ef98aa8e7b24e104b4f54e3bf08181d1ccbdcc12",
        ),
        (
            EvaluationSuiteId.REGISTER_OOD,
            256,
            202_608_102,
            "97358f16696601c26fd7721b10810d858bf545c6ca4bb3f82ded2e599b619710",
        ),
    ],
)
def test_fixed_suite_identity(
    suite_id: EvaluationSuiteId, count: int, seed: int, digest: str
) -> None:
    suite = fixed_evaluation_suite(suite_id)

    assert (len(suite.episodes), suite.suite_seed, suite.digest_sha256) == (
        count,
        seed,
        digest,
    )
    assert (
        suite_digest(
            suite_id=suite.suite_id,
            suite_seed=suite.suite_seed,
            episodes=suite.episodes,
        )
        == digest
    )


def test_fixed_suites_are_balanced_disjoint_and_valid_reset_domains() -> None:
    smoke = fixed_evaluation_suite(EvaluationSuiteId.SMOKE)
    iid = fixed_evaluation_suite(EvaluationSuiteId.IID)
    ood = fixed_evaluation_suite(EvaluationSuiteId.REGISTER_OOD)
    all_suites = (smoke, iid, ood)

    assert (
        sorted(Counter(e.target_note_index for e in smoke.episodes).values()) == [1] * 18 + [2] * 7
    )
    assert (
        sorted(Counter(e.target_note_index for e in iid.episodes).values()) == [10] * 19 + [11] * 6
    )
    assert (
        sorted(Counter(e.target_note_index for e in ood.episodes).values()) == [10] * 19 + [11] * 6
    )
    assert sum(4_800 <= e.source_pitch_cents <= 4_999 for e in ood.episodes) == 128
    assert sum(7_001 <= e.source_pitch_cents <= 7_200 for e in ood.episodes) == 128

    target_bands: dict[int, Counter[str]] = {}
    for episode in ood.episodes:
        band = "lower" if episode.source_pitch_cents < 5_000 else "upper"
        target_bands.setdefault(episode.target_note_index, Counter())[band] += 1
    assert all(abs(bands["lower"] - bands["upper"]) <= 1 for bands in target_bands.values())

    seen_pairs: set[tuple[int, int]] = set()
    for suite in all_suites:
        pairs = {episode.pair for episode in suite.episodes}
        assert len(pairs) == len(suite.episodes)
        assert seen_pairs.isdisjoint(pairs)
        seen_pairs |= pairs
        for episode in suite.episodes:
            assert 0 <= episode.target_note_index <= 24
            assert 4_800 <= episode.source_pitch_cents <= 7_200
            assert (
                abs(episode.source_pitch_cents - _target_pitch_cents(episode.target_note_index)) > 5
            )
    assert all(5_000 <= episode.source_pitch_cents <= 7_000 for episode in smoke.episodes)
    assert all(5_000 <= episode.source_pitch_cents <= 7_000 for episode in iid.episodes)


@pytest.mark.parametrize("profile", list(ProfileName))
def test_seed_zero_bc_splits_have_pinned_disjoint_balanced_membership(profile: ProfileName) -> None:
    splits = build_bc_episode_splits(profile=profile, run_seed=0)
    expected_train_digest, expected_validation_digest = EXPECTED_BC_DIGESTS[profile]
    configured = PROFILE_CONFIGS[profile].bc
    fixed_pairs = {
        episode.pair
        for suite_id in EvaluationSuiteId
        for episode in fixed_evaluation_suite(suite_id).episodes
    }

    assert splits.distribution_id == TRAIN_DISTRIBUTION_ID
    assert len(splits.training) == configured.train_episodes
    assert len(splits.validation) == configured.validation_episodes
    assert (splits.training_digest_sha256, splits.validation_digest_sha256) == (
        expected_train_digest,
        expected_validation_digest,
    )
    training_pairs = {episode.pair for episode in splits.training}
    validation_pairs = {episode.pair for episode in splits.validation}
    assert len(training_pairs) == len(splits.training)
    assert len(validation_pairs) == len(splits.validation)
    assert training_pairs.isdisjoint(validation_pairs)
    assert training_pairs.isdisjoint(fixed_pairs)
    assert validation_pairs.isdisjoint(fixed_pairs)
    for episodes in (splits.training, splits.validation):
        target_counts = Counter(episode.target_note_index for episode in episodes)
        assert max(target_counts.values()) - min(target_counts.values()) <= 1
        assert all(5_000 <= episode.source_pitch_cents <= 7_000 for episode in episodes)
        assert all(
            abs(episode.source_pitch_cents - _target_pitch_cents(episode.target_note_index)) > 5
            for episode in episodes
        )


def test_bc_splits_are_repeatable_and_reject_invalid_run_seeds() -> None:
    assert build_bc_episode_splits(
        profile=ProfileName.SMOKE, run_seed=3
    ) == build_bc_episode_splits(
        profile=ProfileName.SMOKE,
        run_seed=3,
    )
    for invalid_seed in (True, -1):
        with pytest.raises(ValueError, match="run_seed"):
            build_bc_episode_splits(profile=ProfileName.SMOKE, run_seed=invalid_seed)


def test_ppo_training_episodes_are_index_deterministic_and_balanced_by_block() -> None:
    episodes = tuple(ppo_training_episode(run_seed=7, episode_index=index) for index in range(75))
    fixed_pairs = {
        episode.pair
        for suite_id in EvaluationSuiteId
        for episode in fixed_evaluation_suite(suite_id).episodes
    }

    assert ppo_training_episode(run_seed=7, episode_index=32) == episodes[32]
    assert ppo_training_episode(run_seed=8, episode_index=32) != episodes[32]
    for block_start in range(0, len(episodes), 25):
        assert sorted(
            episode.target_note_index for episode in episodes[block_start : block_start + 25]
        ) == list(range(25))
    for episode in episodes:
        assert 5_000 <= episode.source_pitch_cents <= 7_000
        assert episode.pair not in fixed_pairs
        assert abs(episode.source_pitch_cents - _target_pitch_cents(episode.target_note_index)) > 5


@pytest.mark.parametrize("episode_index", [True, -1])
def test_ppo_training_episode_rejects_boolean_or_negative_indices(episode_index: object) -> None:
    with pytest.raises(ValueError, match="episode_index"):
        ppo_training_episode(run_seed=0, episode_index=episode_index)
