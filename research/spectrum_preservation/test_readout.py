"""Pairing must use episode identity and preserve opposing outcome changes."""

from types import SimpleNamespace

import pytest

from research.spectrum_preservation.readout import _paired


def _record(name, *, success, error):
    return SimpleNamespace(
        episode_id=name,
        terminal=SimpleNamespace(
            submitted_success=success, within_1_cent=error <= 1, source_pitch_cents=6000
        ),
        estimates=({"step": 1, "estimated_candidate_cents": 6000 + error},),
    )


def test_pairing_is_order_independent_and_keeps_gains_and_losses_separate():
    legacy = [
        _record("a", success=False, error=10),
        _record("b", success=True, error=0),
        _record("c", success=True, error=4),
    ]
    candidate = [
        _record("c", success=True, error=0),
        _record("b", success=False, error=10),
        _record("a", success=True, error=1),
    ]
    paired = _paired(candidate, legacy, clean=True)
    assert paired == {
        "paired_episodes": 3,
        "rescued_vs_legacy": 1,
        "regressed_vs_legacy": 1,
        "clean_initial_within_1_cent_gains": 2,
        "clean_initial_within_1_cent_losses": 1,
        "clean_submitted_within_1_cent_gains": 2,
        "clean_submitted_within_1_cent_losses": 1,
    }
    assert _paired(candidate, legacy, clean=False)["clean_initial_within_1_cent_gains"] is None
    with pytest.raises(ValueError, match="identical unique"):
        _paired(candidate[:-1], legacy, clean=True)
    with pytest.raises(ValueError, match="identical unique"):
        _paired([candidate[0], candidate[0], candidate[2]], legacy, clean=True)
