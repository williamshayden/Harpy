"""Behavioral safeguards for the representation study, independent of its scores."""

from __future__ import annotations

import numpy as np
import pytest

from harpy.envs.robustness import CLEAN_CONDITION, ROBUSTNESS_CONDITIONS, RobustnessEvidenceCache
from harpy.envs.spectrum import GYM_ANALYSIS_CONFIG, encode_log_spectrum
from harpy.experiments import EpisodeSpec, evaluate, spectrum_peak_actor_spec
from research.spectrum_preservation.encoders import (
    CELL_EDGES_HZ,
    _max_over_cells,
    actor_specs,
    encode_cell_max,
    encode_fft_reference,
)


def test_peak_between_centers_is_retained_and_empty_cells_use_endpoints():
    frequencies = np.array([0.0, 10.0, 20.0])
    levels = np.array([0.0, 20.0, 0.0])
    edges = np.array([1.0, 5.0, 12.0, 18.0])
    assert np.array_equal(_max_over_cells(frequencies, levels, edges), [10.0, 20.0, 16.0])
    assert np.interp(8.5, frequencies, levels) < 20.0


def test_public_grid_really_contains_empty_cells():
    frequencies = np.fft.rfftfreq(GYM_ANALYSIS_CONFIG.fft_frames, 1 / 48_000)[1:]
    counts = np.diff(np.searchsorted(frequencies, CELL_EDGES_HZ, side="left"))
    assert np.count_nonzero(counts == 0) == 228


@pytest.mark.parametrize("source", [4800, 5000, 6857, 7200])
@pytest.mark.parametrize("condition", [CLEAN_CONDITION, ROBUSTNESS_CONDITIONS[6]])
def test_encoded_waveforms_are_finite_owned_and_preserve_point_levels(source, condition):
    evidence = RobustnessEvidenceCache(max_bytes=0).evidence(source, condition, 17)
    legacy = encode_log_spectrum(evidence.waveform)
    candidate = encode_cell_max(evidence.waveform)
    assert candidate.shape == (1961,) and candidate.dtype == np.float32
    assert candidate.flags.owndata and np.isfinite(candidate).all()
    assert np.all((candidate >= 0) & (candidate <= 1))
    assert np.all(candidate >= legacy)
    assert not np.shares_memory(candidate, evidence.waveform)


def test_silence_and_nonfinite_input_follow_analysis_contract():
    waveform = np.zeros(GYM_ANALYSIS_CONFIG.fft_frames, dtype=np.float32)
    controls = np.zeros(3, dtype=np.int16)
    assert not encode_cell_max(waveform).any()
    assert not encode_fft_reference(waveform, controls).any()
    waveform[10] = np.nan
    with pytest.raises(ValueError, match="finite"):
        encode_cell_max(waveform)
    with pytest.raises(ValueError, match="finite"):
        encode_fft_reference(waveform, controls)


def test_legacy_waveform_adapter_matches_native_spectrum_actions_and_estimates():
    episodes = (EpisodeSpec("boundary", 4800, 24, 44), EpisodeSpec("quantized", 6064, 12, 55))
    result = evaluate(
        episodes,
        (spectrum_peak_actor_spec(), actor_specs()[0]),
        conditions=(CLEAN_CONDITION, ROBUSTNESS_CONDITIONS[6]),
    )
    for native, adapted in zip(result.records[::2], result.records[1::2], strict=True):
        assert native.terminal == adapted.terminal
        assert native.estimates == adapted.estimates
        assert native.inference_counts == adapted.inference_counts
        assert native.initial_evidence == adapted.initial_evidence


def test_previously_diagnosed_failure_is_a_retained_development_anchor():
    episode = EpisodeSpec("known-6857", 6857, 13, 14708078973187757233)
    result = evaluate((episode,), actor_specs(), conditions=(ROBUSTNESS_CONDITIONS[6],))
    legacy, candidate, reference = result.records
    assert not legacy.terminal.submitted_success
    assert legacy.terminal.final_absolute_error_cents == 922
    assert candidate.terminal.submitted_success and reference.terminal.submitted_success
    assert all(sum(record.inference_counts) == 1 for record in result.records)
    assert len({actor["controller"] for actor in result.actors}) == 1
