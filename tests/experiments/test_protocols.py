import copy
from collections import Counter

import pytest

from harpy.experiments.protocols import (
    CONFIRMATION_DIGEST_SHA256,
    benchmark_episodes,
    confirmation_episodes,
    excluded_historical_pairs,
    protocol_document,
    protocol_episodes,
    smoke_episodes,
    validate_protocol_document,
)
from harpy.learning.pitch_data import (
    PitchEvaluationSuiteId,
    fixed_pitch_evaluation_suite,
    pitch_coordinate_split,
)


def test_benchmark_preserves_every_legacy_pair_and_register():
    actual = benchmark_episodes()
    expected = [
        episode
        for suite in (
            PitchEvaluationSuiteId.IID,
            PitchEvaluationSuiteId.OOD_LOWER,
            PitchEvaluationSuiteId.OOD_UPPER,
        )
        for episode in fixed_pitch_evaluation_suite(suite).episodes
    ]
    assert [(e.target_note_index, e.source_pitch_cents) for e in actual] == [
        (e.target_note_index, e.source_pitch_cents) for e in expected
    ]
    assert Counter(e.partition for e in actual) == {"iid": 250, "lower": 200, "upper": 200}
    assert len({e.id for e in actual}) == 650


def test_confirmation_is_frozen_unique_source_heldout_and_never_previously_evaluated():
    episodes = confirmation_episodes()
    pairs = {(e.target_note_index, e.source_pitch_cents) for e in episodes}
    assert len(episodes) == len(pairs) == 1000
    assert len({e.id for e in episodes}) == 1000
    assert not pairs & excluded_historical_pairs()
    assert len(excluded_historical_pairs()) == 1324
    assert Counter(e.partition for e in episodes) == {"iid": 400, "lower": 300, "upper": 300}
    split = pitch_coordinate_split()
    excluded_sources = set(split.training_coordinates) | set(split.validation_coordinates)
    assert not {e.source_pitch_cents for e in episodes} & excluded_sources
    assert all(abs(e.source_pitch_cents - (4800 + e.target_note_index * 100)) > 5 for e in episodes)
    assert protocol_document("confirmation")["digest_sha256"] == CONFIRMATION_DIGEST_SHA256
    for partition, per_target in (("iid", 16), ("lower", 12), ("upper", 12)):
        assert Counter(e.target_note_index for e in episodes if e.partition == partition) == {
            target: per_target for target in range(25)
        }


def test_protocol_documents_are_owned_strict_and_bind_episode_order():
    for name in ("clean", "robustness", "confirmation", "smoke"):
        document = protocol_document(name)
        assert validate_protocol_document(document) == document
        assert document["episodes"] == [e.to_document() for e in protocol_episodes(name)]
        tampered = copy.deepcopy(document)
        tampered["episodes"].reverse()
        with pytest.raises(ValueError, match="differs"):
            validate_protocol_document(tampered)
        assert protocol_document(name) == document
    assert len(smoke_episodes()) == 6
    with pytest.raises(ValueError):
        protocol_episodes("unknown")
