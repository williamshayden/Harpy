"""Independent framing and whole-clip checks for the sound objective."""

import json
import math

import numpy as np
import pytest

from harpy.tasks.sound_match.objective import FFT_SIZES, OBJECTIVE_ID, score
from harpy.tasks.sound_match.renderer import FRAME_COUNT


def _reference_components(target, candidate, size):
    # Deliberately frame individually, independently of the vectorized implementation.
    window = np.hanning(size + 1)[:-1]
    spectra = []
    for audio in (target, candidate):
        frames = []
        for start in range(0, FRAME_COUNT, size // 4):
            segment = np.zeros(size, dtype=np.float64)
            available = audio[start : start + size]
            segment[: len(available)] = available
            frames.append(np.abs(np.fft.rfft(segment * window)) / window.sum())
        spectra.append(np.stack(frames))
    left, right = spectra
    convergence = np.sqrt(((left - right) ** 2).sum()) / max(np.sqrt((left**2).sum()), 1e-7)
    logarithmic = np.mean(np.abs(np.log(np.maximum(left, 1e-7)) - np.log(np.maximum(right, 1e-7))))
    return float(convergence), float(logarithmic)


def test_identity_and_silence_are_exact_finite_json():
    assert OBJECTIVE_ID == "harpy-multires-spectral-v1"
    target = np.zeros(FRAME_COUNT, dtype=np.float32)
    for fixture in (target, np.linspace(-0.5, 0.5, FRAME_COUNT, dtype=np.float32)):
        result = score(fixture, fixture)
        json.dumps(result, allow_nan=False)
        assert result["distance"] == 0.0
        assert result["envelope_rmse"] == 0.0
        assert result["level_error_db"] == 0.0
        assert [part["fft_size"] for part in result["spectral"]] == list(FFT_SIZES)
        assert all(part["convergence"] == part["log_distance"] == 0 for part in result["spectral"])


def test_spectral_components_match_full_clip_formula_including_last_frame():
    target = np.zeros(FRAME_COUNT, dtype=np.float32)
    candidate = target.copy()
    target[[255, 48000, 95999]] = (0.5, 0.25, 0.75)
    candidate[[256, 48000, 95999]] = (0.25, 0.25, -0.25)
    before = target.copy(), candidate.copy()
    result = score(target, candidate)
    expected_distance = 0.0
    for size, part in zip(FFT_SIZES, result["spectral"], strict=True):
        convergence, logarithmic = _reference_components(target, candidate, size)
        assert part["convergence"] == pytest.approx(convergence, rel=1e-12, abs=1e-14)
        assert part["log_distance"] == pytest.approx(logarithmic, rel=1e-12, abs=1e-14)
        expected_distance += convergence + logarithmic
    assert result["distance"] == pytest.approx(expected_distance / 3)
    np.testing.assert_array_equal(target, before[0])
    np.testing.assert_array_equal(candidate, before[1])


def test_early_audio_is_not_discarded_because_the_tail_is_silent():
    target = np.zeros(FRAME_COUNT, dtype=np.float32)
    target[:480] = 0.5
    missing = np.zeros_like(target)
    result = score(target, missing)
    assert result["distance"] > 0
    assert result["envelope_rmse"] == pytest.approx(0.5 / math.sqrt(200))
    delayed = np.roll(target, 48000)
    assert score(target, delayed)["distance"] > 0
    assert score(target, delayed)["envelope_rmse"] > 0


def test_level_difference_has_declared_sign_floor_and_no_gain_normalization():
    target = np.full(FRAME_COUNT, 0.25, dtype=np.float32)
    louder = score(target, target * 2)
    quieter = score(target, target / 2)
    assert louder["level_error_db"] == pytest.approx(20 * math.log10(2))
    assert quieter["level_error_db"] == pytest.approx(-20 * math.log10(2))
    assert louder["envelope_rmse"] == pytest.approx(0.25)
    assert louder["distance"] > 0
    silent = score(target, np.zeros_like(target))
    assert silent["level_error_db"] == pytest.approx(20 * math.log10(1e-7 / 0.25))
    json.dumps(score(np.zeros_like(target), target), allow_nan=False)


def test_phase_inversion_is_not_mistaken_for_a_level_or_envelope_change():
    phase = np.arange(FRAME_COUNT, dtype=np.float64) * (math.tau * 220 / 48000)
    target = (0.25 * np.sin(phase)).astype(np.float32)
    result = score(target, -target)
    assert result["distance"] == 0.0
    assert result["envelope_rmse"] == 0.0
    assert result["level_error_db"] == 0.0


def test_finite_out_of_range_calibration_audio_is_measured_without_clipping():
    target = np.linspace(-2, 2, FRAME_COUNT, dtype=np.float32)
    clipped = np.clip(target, -1.0, 1.0)
    result = score(target, clipped)
    assert result["distance"] > 0
    assert result["envelope_rmse"] > 0
    assert result["level_error_db"] < 0
    json.dumps(result, allow_nan=False)


def test_input_validation_rejects_implicit_conversion_and_nonfinite_audio():
    valid = np.zeros(FRAME_COUNT, dtype=np.float32)
    nonfinite = valid.copy()
    nonfinite[0] = np.nan
    infinite = valid.copy()
    infinite[-1] = np.inf
    for invalid in (
        valid.tolist(),
        valid.astype(np.float64),
        valid[:-1],
        valid.reshape(2, -1),
        nonfinite,
        infinite,
    ):
        with pytest.raises(ValueError, match="target"):
            score(invalid, valid)
        with pytest.raises(ValueError, match="candidate"):
            score(valid, invalid)
