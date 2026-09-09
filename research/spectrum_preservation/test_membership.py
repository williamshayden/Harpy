"""Membership integrity checks; no rendering, training, or study evaluation."""

from collections import Counter
from copy import deepcopy

import pytest

import research.spectrum_preservation.membership as membership
from harpy.experiments.models import _json_bytes
from harpy.experiments.protocols import (
    benchmark_episodes,
    confirmation_episodes,
    excluded_historical_pairs,
)
from harpy.experiments.results import _validate_protocol
from harpy.learning.artifacts import decode_json_bytes
from harpy.learning.pitch_data import pitch_coordinate_split
from research.spectrum_preservation.membership import (
    episodes_from_manifest,
    make_manifest,
    write_manifest,
)


def test_seeded_membership_is_deterministic_owned_and_compatible_with_ordinary_protocol():
    first = make_manifest()
    second = make_manifest()
    assert _json_bytes(first) == _json_bytes(second)
    assert first["digest_sha256"] != make_manifest(20260909)["digest_sha256"]
    episodes = episodes_from_manifest(first)
    _validate_protocol(first, episodes)
    assert first["scope"].startswith(
        "fresh combinations in known rendering family, excluding recorded inventory"
    )
    first["episodes"].clear()
    assert len(second["episodes"]) == 600
    assert len(make_manifest()["episodes"]) == 600


def test_all_partitions_targets_pairs_and_recorded_exclusions_are_exact():
    document = make_manifest()
    episodes = episodes_from_manifest(document)
    pairs = {(item.target_note_index, item.source_pitch_cents) for item in episodes}
    historical = excluded_historical_pairs()
    confirmation = {
        (item.target_note_index, item.source_pitch_cents) for item in confirmation_episodes()
    }
    benchmark = {(item.target_note_index, item.source_pitch_cents) for item in benchmark_episodes()}
    assert len(episodes) == len(pairs) == len({item.id for item in episodes}) == 600
    assert benchmark <= historical
    assert not pairs & (historical | confirmation | benchmark)
    assert Counter(item.partition for item in episodes) == {"iid": 200, "lower": 200, "upper": 200}
    assert set(Counter((item.partition, item.target_note_index) for item in episodes).values()) == {
        8
    }
    split = pitch_coordinate_split()
    pools = {
        "iid": split.iid_holdout_coordinates,
        "lower": split.ood_lower_coordinates,
        "upper": split.ood_upper_coordinates,
    }
    assert all(item.source_pitch_cents in pools[item.partition] for item in episodes)
    assert document["exclusions"]["confirmation"]["pair_count"] == 1000
    assert document["exclusions"]["union_pair_count"] == len(historical | confirmation)
    assert document["source_split_sha256"] == split.digest_sha256
    # The manifest supplies episode seeds, with no actor/condition input or resampling.
    assert episodes == episodes_from_manifest(deepcopy(document))


@pytest.mark.parametrize(
    "change",
    [
        lambda doc: doc.update(extra=True),
        lambda doc: doc["selection"].update(seed=True),
        lambda doc: doc["selection"].update(seed=20260909),
        lambda doc: doc["episodes"].reverse(),
        lambda doc: doc["episodes"].pop(),
        lambda doc: doc["episodes"][0].update(partition="upper"),
        lambda doc: doc["episodes"][0].update(nuisance_seed=0),
        lambda doc: doc["exclusions"]["historical"].update(digest_sha256="0" * 64),
        lambda doc: doc["exclusions"]["confirmation"].update(pair_count=999),
    ],
)
def test_reader_rejects_changed_membership_or_identity_even_with_rehashed_document(change):
    document = make_manifest()
    change(document)
    document["digest_sha256"] = membership._digest(
        {key: value for key, value in document.items() if key != "digest_sha256"}
    )
    with pytest.raises(ValueError):
        episodes_from_manifest(document)


def test_benchmark_coverage_is_required(monkeypatch):
    monkeypatch.setattr(membership, "excluded_historical_pairs", lambda: frozenset())
    with pytest.raises(ValueError, match="all 650 benchmark"):
        make_manifest()


def test_reader_rejects_incorrect_digest():
    document = make_manifest()
    document["digest_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        episodes_from_manifest(document)


def test_atomic_create_only_write_and_validation_before_publication(tmp_path):
    document = make_manifest()
    path = tmp_path / "nested" / "membership.json"
    write_manifest(document, path)
    content = path.read_bytes()
    assert episodes_from_manifest(decode_json_bytes(content)) == episodes_from_manifest(document)
    with pytest.raises(FileExistsError):
        write_manifest(make_manifest(20260909), path)
    assert path.read_bytes() == content
    document["episodes"].clear()
    invalid = tmp_path / "invalid" / "membership.json"
    with pytest.raises(ValueError):
        write_manifest(document, invalid)
    assert not invalid.parent.exists()


@pytest.mark.parametrize("seed", [True, -1, 2**64, 1.5, "20260908", None])
def test_invalid_selection_seed(seed):
    with pytest.raises(ValueError, match="selection seed"):
        make_manifest(seed)
