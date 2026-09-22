"""Objective interpretation is continuous; fixtures do not invent success rates."""

import numpy as np
import pytest

from harpy.tasks.sound_match.calibration import calibration_pairs, write_calibration
from harpy.tasks.sound_match.objective import score


@pytest.fixture(scope="module")
def pairs():
    return {name: (target, candidate) for name, target, candidate in calibration_pairs()}


def test_identity_permutations_and_inactive_labels_have_zero_distance(pairs):
    for name in ("identity", "chord-permutation", "inactive-decay-alias"):
        assert np.array_equal(*pairs[name])
        assert score(*pairs[name])["distance"] == 0


def test_calibration_exposes_phase_insensitivity_and_detects_substantial_errors(pairs):
    assert score(*pairs["phase-polarity-inversion"])["distance"] == 0
    assert (
        score(*pairs["detune-100-cents"])["distance"] > score(*pairs["detune-5-cents"])["distance"]
    )
    for name in (
        "silence",
        "missing-chord-tone",
        "brightness-500hz",
        "attack-250ms",
        "release-750ms",
        "level-minus-12db",
        "clipped-at-0.025",
    ):
        measured = score(*pairs[name])
        assert measured["distance"] > 0
        assert np.isfinite(measured["distance"])


def test_calibration_retains_lossless_pairs_without_acceptance_claim(tmp_path):
    report = write_calibration(tmp_path / "fixtures")
    assert report["perceptual_cutoff"] is None
    assert report["listening_review"] == "not-performed"
    assert len(report["rows"]) == 14
    for row in report["rows"]:
        assert (tmp_path / "fixtures" / row["target_file"]["path"]).stat().st_size == 384056
        assert (tmp_path / "fixtures" / row["candidate_file"]["path"]).stat().st_size == 384056
