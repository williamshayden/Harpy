"""Frozen source-family separation, target holdouts and streaming label access."""

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from harpy._serialization import canonical_json_bytes
from harpy.tasks.sound_match import datasets
from harpy.tasks.sound_match.patch import Patch
from harpy.tasks.sound_match.renderer import render


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    directory = tmp_path_factory.mktemp("grouped") / "manifests"
    manifests = datasets.build_manifests(
        directory, counts={"train": 24, "validation": 24, "compositional-test": 24}
    )
    return directory, manifests


def test_memberships_are_balanced_reachable_opaque_and_disjoint(generated):
    directory, manifests = generated
    datasets.validate_disjoint(manifests)
    for manifest in manifests:
        assert datasets.load_manifest(manifest.split, directory=directory) == manifest
        assert len(manifest.episodes) == 24
        assert Counter(entry["cohort"] for entry in manifest.entries) == {
            "single-reconstruction": 6,
            "single-shaping": 6,
            "chord-reconstruction": 6,
            "chord-shaping": 6,
        }
        assert set(entry["cell"] for entry in manifest.entries) == set(datasets.CELL_NAMES)
        assert set(Counter(entry["source_family_id"] for entry in manifest.entries).values()) == {2}
        for entry, episode in zip(manifest.entries, manifest.episodes, strict=True):
            assert episode.episode_id.startswith("e_")
            assert len(episode.episode_id) == 26
            assert datasets.heldout_combination(episode.target_patch) == (
                manifest.split == "compositional-test"
            )
            for name in datasets.PATCH_FIELDS:
                if name not in episode.editable_fields:
                    assert getattr(episode.target_patch, name) == getattr(
                        episode.initial_patch, name
                    )
            assert datasets.audio_hash(render(episode.target_patch)) == entry["target_audio_sha256"]


def test_generation_is_exactly_repeatable(generated, tmp_path):
    directory, _ = generated
    repeated = tmp_path / "repeat"
    datasets.build_manifests(
        repeated, counts={"train": 24, "validation": 24, "compositional-test": 24}
    )
    for path in directory.iterdir():
        assert path.read_bytes() == (repeated / path.name).read_bytes()


def test_structural_alias_matches_actual_inactive_decay_audio():
    first = Patch(sustain_db=0, decay_ms=20)
    second = replace(first, decay_ms=500)
    assert datasets.structural_hash(first) == datasets.structural_hash(second)
    assert np.array_equal(render(first), render(second))
    assert datasets.structural_hash(replace(first, sustain_db=-1)) != datasets.structural_hash(
        replace(second, sustain_db=-1)
    )


def test_chord_holdout_orbits_include_transposition_inversion_and_permutation():
    diminished = (4800, 5100, 5400)
    assert datasets.chord_class(diminished) == datasets.chord_class((5317, 6217, 5617))
    heldout = {datasets.chord_class(value) for value in datasets.HELDOUT_CHORDS.values()}
    assert not heldout & {datasets.chord_class(value) for value in datasets.TRAIN_CHORDS.values()}


@pytest.mark.parametrize(
    "identity", ("source_family_id", "structural_sha256", "target_audio_sha256")
)
def test_cross_split_aliases_are_rejected(generated, identity):
    _, (train, validation, _) = generated
    entries = [dict(entry) for entry in validation.entries]
    entries[0][identity] = train.entries[0][identity]
    invalid = replace(validation, entries=tuple(entries))
    with pytest.raises(ValueError, match="cross-split"):
        datasets.validate_disjoint((train, invalid))


def test_manifest_catalog_binds_exact_membership_bytes(generated, tmp_path):
    directory, _ = generated
    altered = tmp_path / "altered"
    altered.mkdir()
    for name in ("train.json", "catalog.json"):
        (altered / name).write_bytes((directory / name).read_bytes())
    (altered / "train.json").write_bytes((altered / "train.json").read_bytes() + b" ")
    with pytest.raises(ValueError, match="frozen catalog"):
        datasets.load_manifest("train", directory=altered)


def test_malformed_family_or_holdout_cannot_be_rebound(generated):
    _, (train, _, _) = generated
    document = train.to_dict()
    document["entries"][1]["source_family_id"] = "g_" + "a" * 24
    with pytest.raises(ValueError, match="source family"):
        datasets.DatasetManifest.from_dict(document)
    document = train.to_dict()
    document["split"] = "compositional-test"
    with pytest.raises(ValueError, match="holdout"):
        datasets.DatasetManifest.from_dict(document)


@pytest.mark.parametrize("voice_count", (1, 3))
def test_reconstruction_start_cannot_be_rebound_away_from_public_default(voice_count):
    document = datasets.load_manifest("train").to_dict()
    row = next(
        entry
        for entry in document["entries"]
        if entry["mask"] == "all"
        and len(entry["episode"]["initial_patch"]["notes_cents"]) == voice_count
    )
    row["episode"]["initial_patch"]["gain_db"] = -7
    with pytest.raises(ValueError, match="public default"):
        datasets.DatasetManifest.from_dict(document)


def test_training_iterator_yields_one_verified_waveform_at_a_time(generated, monkeypatch):
    directory, (train, _, _) = generated
    calls = []
    original = datasets.render

    def tracked(patch):
        calls.append(patch)
        return original(patch)

    monkeypatch.setattr(datasets, "render", tracked)
    iterator = datasets.iter_training_examples(directory=directory)
    assert not calls
    episode, audio = next(iterator)
    assert len(calls) == 1
    assert episode == train.episodes[0]
    assert audio.shape == (96000,)
    monkeypatch.setattr(datasets, "render", lambda _: np.zeros(96000, dtype=np.float32))
    with pytest.raises(ValueError, match="audio hash"):
        next(iterator)


def test_manifest_is_immutable_and_hash_identifies_canonical_bytes(generated):
    _, (train, _, _) = generated
    with pytest.raises(TypeError):
        train.entries[0]["episode"]["target_patch"]["gain_db"] = 0
    import hashlib

    assert (
        hashlib.sha256(canonical_json_bytes(train.to_dict())).hexdigest() == train.manifest_sha256
    )


def test_generator_refuses_nonbalanced_counts_and_existing_output(tmp_path):
    with pytest.raises(ValueError, match="multiples of 24"):
        datasets.build_manifests(tmp_path / "bad", counts={"train": 25})
    with pytest.raises(FileExistsError):
        datasets.build_manifests(tmp_path, counts={"train": 24})


def test_packaged_memberships_preserve_frozen_counts_and_public_exposure_boundary():
    manifests = tuple(datasets.load_manifest(split) for split in datasets.SPLIT_COUNTS)
    datasets.validate_disjoint(manifests)
    for manifest in manifests:
        summary = datasets.manifest_summary(manifest)
        assert summary["targets"] == datasets.SPLIT_COUNTS[manifest.split]
        assert summary["source_families"] == summary["targets"] // 2
        heldout = manifest.split == "compositional-test"
        assert summary["public_starts_with_heldout_combination"] == (96 if heldout else 0)
        assert summary["public_starts_with_heldout_chord_family"] == (48 if heldout else 0)


def test_recorded_cell_counts_and_catalog_types_are_strict(generated, tmp_path):
    directory, (train, _, _) = generated
    document = train.to_dict()
    document["generation"]["counts"]["1:all"] = 6.0
    with pytest.raises(ValueError, match="generation"):
        datasets.DatasetManifest.from_dict(document)
    document = train.to_dict()
    document["entries"][0]["mask"] = []
    with pytest.raises(ValueError, match="mask"):
        datasets.DatasetManifest.from_dict(document)
    from harpy._serialization import decode_json_bytes

    catalog = decode_json_bytes((directory / "catalog.json").read_bytes())
    catalog["baseline_seeds"][0] = False
    (tmp_path / "catalog.json").write_bytes(canonical_json_bytes(catalog))
    (tmp_path / "train.json").write_bytes((directory / "train.json").read_bytes())
    with pytest.raises(ValueError, match="catalog"):
        datasets.load_manifest("train", directory=tmp_path)
