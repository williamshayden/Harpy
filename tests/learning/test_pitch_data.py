"""Contracts for Milestone E pitch-coordinate data and episode suites."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from harpy.learning.models import TrainerKind
from harpy.learning.pitch_data import (
    PITCH_DISTRIBUTION_ID,
    PITCH_SPLIT_DIGEST_SHA256,
    PitchCoordinateSplit,
    PitchEvaluationSuite,
    PitchEvaluationSuiteId,
    PitchTrainerKind,
    fixed_pitch_evaluation_suite,
    iid_shuffled_spectrum_permutation,
    pitch_coordinate_split,
)

EXPECTED_SPLIT_DIGEST = "e7a111faa7b0722a4ef6d9998c6a69c74221854a05a3661112ff4eb72bef4ef2"
EXPECTED_SHUFFLE_DIGEST = "b3c000befe940a6dcc80f55f6939d7c1bdcbb6899bdd0b03798941ea10749a51"
EXPECTED_SPLIT_PREFIX = (
    6224,
    6377,
    6885,
    6615,
    6421,
    5563,
    6159,
    5083,
    5160,
    6700,
    5058,
    5488,
    6097,
    6580,
    6681,
    6596,
    5105,
    6634,
    6924,
    6008,
)
EXPECTED_SHUFFLE_PREFIX = (
    1636,
    1216,
    1006,
    1068,
    1436,
    1704,
    751,
    1578,
    1007,
    352,
    268,
    1838,
    301,
    156,
    1192,
    1674,
    1014,
    16,
    1575,
    1327,
)


def _canonical_digest(payload: dict[str, object]) -> str:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        + b"\n"
    )
    return hashlib.sha256(encoded).hexdigest()


def _independent_coordinate_split() -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    ordered = tuple(
        sorted(
            range(5_000, 7_001),
            key=lambda coordinate: (
                hashlib.sha256(f"harpy-sine-pitch-grid-v1:{coordinate}".encode("ascii")).digest(),
                coordinate,
            ),
        )
    )
    return (
        ordered[:1_400],
        ordered[1_400:1_600],
        ordered[1_600:],
        tuple(range(4_800, 5_000)),
        tuple(range(7_001, 7_201)),
    )


def _split_payload(
    training: tuple[int, ...],
    validation: tuple[int, ...],
    iid_holdout: tuple[int, ...],
    ood_lower: tuple[int, ...],
    ood_upper: tuple[int, ...],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "distribution_id": "harpy-sine-pitch-grid-v1",
        "training_coordinates": list(training),
        "validation_coordinates": list(validation),
        "iid_holdout_coordinates": list(iid_holdout),
        "ood_lower_coordinates": list(ood_lower),
        "ood_upper_coordinates": list(ood_upper),
    }


def _independent_suite_pairs(
    *, suite_code: int, source_pool: tuple[int, ...], per_target: int
) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for target_note_index in range(25):
        target_pitch_cents = 4_800 + 100 * target_note_index
        eligible = tuple(
            source_pitch_cents
            for source_pitch_cents in sorted(source_pool)
            if abs(source_pitch_cents - target_pitch_cents) > 5
        )
        selected = np.random.default_rng(
            np.random.SeedSequence([1, suite_code, target_note_index])
        ).choice(
            np.asarray(eligible, dtype=np.int64),
            size=per_target,
            replace=False,
        )
        pairs.extend((target_note_index, int(source)) for source in selected)
    return tuple(pairs)


def _suite_payload(
    *, suite_id: PitchEvaluationSuiteId, suite_code: int, pairs: tuple[tuple[int, int], ...]
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "suite_id": suite_id.value,
        "suite_seed": suite_code,
        "episodes": [
            {
                "episode_index": index,
                "target_note_index": target,
                "source_pitch_cents": source,
            }
            for index, (target, source) in enumerate(pairs)
        ],
    }


def test_coordinate_split_matches_independent_order_and_literal_digest() -> None:
    split = pitch_coordinate_split()
    training, validation, iid_holdout, ood_lower, ood_upper = _independent_coordinate_split()

    assert PITCH_DISTRIBUTION_ID == "harpy-sine-pitch-grid-v1"
    assert PITCH_SPLIT_DIGEST_SHA256 == EXPECTED_SPLIT_DIGEST
    assert split.schema_version == 1
    assert split.distribution_id == PITCH_DISTRIBUTION_ID
    assert split.training_coordinates == training
    assert split.validation_coordinates == validation
    assert split.iid_holdout_coordinates == iid_holdout
    assert split.ood_lower_coordinates == ood_lower
    assert split.ood_upper_coordinates == ood_upper
    assert split.training_coordinates[:20] == EXPECTED_SPLIT_PREFIX
    assert split.digest_sha256 == EXPECTED_SPLIT_DIGEST
    assert (
        _canonical_digest(_split_payload(training, validation, iid_holdout, ood_lower, ood_upper))
        == EXPECTED_SPLIT_DIGEST
    )


def test_coordinate_partitions_are_disjoint_complete_and_seed_independent() -> None:
    split = pitch_coordinate_split()
    central_parts = (
        split.training_coordinates,
        split.validation_coordinates,
        split.iid_holdout_coordinates,
    )

    assert tuple(map(len, central_parts)) == (1_400, 200, 401)
    assert len(set().union(*map(set, central_parts))) == 2_001
    assert all(set(left).isdisjoint(right) for left, right in pairwise(central_parts))
    assert set(central_parts[0]).isdisjoint(central_parts[2])
    assert set().union(*map(set, central_parts)) == set(range(5_000, 7_001))
    assert split.ood_lower_coordinates == tuple(range(4_800, 5_000))
    assert split.ood_upper_coordinates == tuple(range(7_001, 7_201))
    assert pitch_coordinate_split() is split


def test_pitch_trainer_identity_is_schema_local_and_does_not_expand_v1() -> None:
    assert tuple(PitchTrainerKind) == (PitchTrainerKind.PITCH,)
    assert PitchTrainerKind.PITCH.value == "pitch"
    with pytest.raises(ValueError):
        TrainerKind("pitch")


def test_coordinate_split_rejects_a_rehashed_noncanonical_partition() -> None:
    split = pitch_coordinate_split()
    training = list(split.training_coordinates)
    validation = list(split.validation_coordinates)
    training[0], validation[0] = validation[0], training[0]
    payload = _split_payload(
        tuple(training),
        tuple(validation),
        split.iid_holdout_coordinates,
        split.ood_lower_coordinates,
        split.ood_upper_coordinates,
    )

    with pytest.raises(ValueError, match="pinned"):
        replace(
            split,
            training_coordinates=tuple(training),
            validation_coordinates=tuple(validation),
            digest_sha256=_canonical_digest(payload),
        )


@pytest.mark.parametrize(
    ("suite_id", "suite_code", "split_field", "per_target", "literal_digest"),
    [
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
    ],
)
def test_fixed_pitch_suite_matches_independent_membership_and_literal_digest(
    suite_id: PitchEvaluationSuiteId,
    suite_code: int,
    split_field: str,
    per_target: int,
    literal_digest: str,
) -> None:
    split = pitch_coordinate_split()
    source_pool = getattr(split, split_field)
    expected_pairs = _independent_suite_pairs(
        suite_code=suite_code,
        source_pool=source_pool,
        per_target=per_target,
    )
    suite = fixed_pitch_evaluation_suite(suite_id)

    assert suite.schema_version == 2
    assert suite.suite_id is suite_id
    assert suite.suite_seed == suite_code
    assert tuple(episode.pair for episode in suite.episodes) == expected_pairs
    assert suite.digest_sha256 == literal_digest
    assert (
        _canonical_digest(
            _suite_payload(suite_id=suite_id, suite_code=suite_code, pairs=expected_pairs)
        )
        == literal_digest
    )


def test_pitch_suite_rejects_rehashed_noncanonical_seed_and_membership() -> None:
    suite = fixed_pitch_evaluation_suite(PitchEvaluationSuiteId.IID)
    changed_seed = 999
    changed_seed_digest = _canonical_digest(
        _suite_payload(
            suite_id=suite.suite_id,
            suite_code=changed_seed,
            pairs=tuple(episode.pair for episode in suite.episodes),
        )
    )
    with pytest.raises(ValueError, match="pinned"):
        replace(suite, suite_seed=changed_seed, digest_sha256=changed_seed_digest)

    one_episode = suite.episodes[:1]
    one_episode_digest = _canonical_digest(
        _suite_payload(
            suite_id=suite.suite_id,
            suite_code=suite.suite_seed,
            pairs=tuple(episode.pair for episode in one_episode),
        )
    )
    with pytest.raises(ValueError, match="pinned"):
        PitchEvaluationSuite(
            schema_version=suite.schema_version,
            suite_id=suite.suite_id,
            suite_seed=suite.suite_seed,
            episodes=one_episode,
            digest_sha256=one_episode_digest,
        )


def test_pitch_coordinate_and_suite_contracts_are_immutable() -> None:
    assert isinstance(pitch_coordinate_split(), PitchCoordinateSplit)
    assert isinstance(
        fixed_pitch_evaluation_suite(PitchEvaluationSuiteId.SMOKE),
        PitchEvaluationSuite,
    )


def test_pitch_suites_are_balanced_valid_and_initial_source_separated() -> None:
    split = pitch_coordinate_split()
    source_pools = {
        PitchEvaluationSuiteId.SMOKE: set(split.validation_coordinates),
        PitchEvaluationSuiteId.IID: set(split.iid_holdout_coordinates),
        PitchEvaluationSuiteId.OOD_LOWER: set(split.ood_lower_coordinates),
        PitchEvaluationSuiteId.OOD_UPPER: set(split.ood_upper_coordinates),
    }
    expected_per_target = {
        PitchEvaluationSuiteId.SMOKE: 2,
        PitchEvaluationSuiteId.IID: 10,
        PitchEvaluationSuiteId.OOD_LOWER: 8,
        PitchEvaluationSuiteId.OOD_UPPER: 8,
    }

    for suite_id in PitchEvaluationSuiteId:
        suite = fixed_pitch_evaluation_suite(suite_id)
        pairs = tuple(episode.pair for episode in suite.episodes)
        target_counts = Counter(target for target, _source in pairs)

        assert target_counts == Counter(
            {target: expected_per_target[suite_id] for target in range(25)}
        )
        assert len(pairs) == len(set(pairs))
        assert [target for target, _source in pairs] == sorted(target for target, _source in pairs)
        assert all(source in source_pools[suite_id] for _target, source in pairs)
        assert all(abs(source - (4_800 + 100 * target)) > 5 for target, source in pairs)

    training = set(split.training_coordinates)
    assert all(
        episode.source_pitch_cents not in training
        for suite_id in PitchEvaluationSuiteId
        for episode in fixed_pitch_evaluation_suite(suite_id).episodes
    )


def test_iid_shuffle_is_the_single_pinned_v2_permutation() -> None:
    permutation = iid_shuffled_spectrum_permutation()

    assert len(permutation) == 1_961
    assert permutation[:20] == EXPECTED_SHUFFLE_PREFIX
    assert set(permutation) == set(range(1_961))
    assert (
        hashlib.sha256(np.asarray(permutation, dtype="<i8").tobytes()).hexdigest()
        == EXPECTED_SHUFFLE_DIGEST
    )
    assert iid_shuffled_spectrum_permutation() is permutation


def test_pitch_data_import_stays_free_of_training_and_gui_dependencies() -> None:
    source = """
import sys
import harpy.learning.pitch_data
import harpy.learning.action_masks
for forbidden in ('torch', 'stable_baselines3', 'PySide6', 'pyqtgraph'):
    assert forbidden not in sys.modules, forbidden
"""

    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
