"""Patch and rendering invariants for the fixed sound-matching task."""

import hashlib
import itertools
import json
import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from harpy.tasks.sound_match.patch import (
    PARAMETER_BOUNDS,
    PATCH_FIELDS,
    PATCH_SCHEMA_ID,
    Patch,
    default_patch,
)
from harpy.tasks.sound_match.renderer import FRAME_COUNT, SAMPLE_RATE_HZ, _voice, render


def test_patch_round_trip_is_strict_canonical_and_owned():
    notes = [6700, 6000, 6400]
    patch = Patch(notes_cents=notes)
    notes[0] = 7000
    assert patch.notes_cents == (6000, 6400, 6700)
    assert PATCH_SCHEMA_ID == "harpy.sound-match.patch/1"
    assert tuple(patch.to_dict()) == PATCH_FIELDS
    assert Patch.from_dict(json.loads(json.dumps(patch.to_dict()))) == default_patch(3)
    exported = patch.to_dict()
    exported["notes_cents"][0] = 5000
    assert patch.notes_cents == (6000, 6400, 6700)
    with pytest.raises(FrozenInstanceError):
        patch.gain_db = 0
    for invalid in (
        {},
        {**patch.to_dict(), "schema": PATCH_SCHEMA_ID},
        {name: value for name, value in patch.to_dict().items() if name != "gain_db"},
        [],
    ):
        with pytest.raises(ValueError, match="exactly these fields"):
            Patch.from_dict(invalid)


def test_patch_rejects_invalid_notes_and_never_coerces_values():
    for notes in (
        (),
        (6000, 6400),
        (6000, 6000, 6700),
        (4799,),
        (7201,),
        (True,),
        (6000.0,),
        (float("nan"),),
        "6000",
        {6000},
    ):
        with pytest.raises(ValueError, match="notes_cents"):
            Patch(notes_cents=notes)
    for field, (minimum, maximum) in PARAMETER_BOUNDS.items():
        if field == "notes_cents":
            continue
        for value in (minimum - 1, maximum + 1, float(minimum), True, float("nan")):
            with pytest.raises(ValueError, match=field):
                replace(default_patch(), **{field: value})
        replace(default_patch(), **{field: minimum})
        replace(default_patch(), **{field: maximum})
    for value in ("noise", 1, None):
        with pytest.raises(ValueError, match="waveform"):
            Patch(waveform=value)
    for value in (0, 2, 4, True, 1.0):
        with pytest.raises(ValueError, match="voice_count"):
            default_patch(value)


def test_render_returns_owned_complete_audio_and_restarts_state():
    patch = default_patch()
    first = render(patch)
    expected = first.copy()
    assert first.shape == (FRAME_COUNT,)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert first.flags.owndata
    first[:] = 1.0
    render(replace(patch, waveform="square", notes_cents=(4800,)))
    np.testing.assert_array_equal(render(patch), expected)
    assert not np.any(expected[57_600:])
    with pytest.raises(ValueError, match="Patch"):
        render(patch.to_dict())


def test_render_golden_cpu_fixtures():
    # Frozen on Linux/WSL, Python 3.12.3, NumPy 2.5.1. No cross-platform claim.
    fixtures = (
        (default_patch(), "ac14eeaa7cc9124b85c8044de48938a42e84ccd09a76589d968d2e06a3781250"),
        (
            Patch(
                notes_cents=(4800, 4801, 4802),
                waveform="saw",
                brightness_hz=12000,
                attack_ms=5,
                decay_ms=20,
                sustain_db=0,
                release_ms=750,
                gain_db=0,
            ),
            "881cdfcf8d4bbe37958a47dbb94b68e91da7553c4ebef4640657abee0c590338",
        ),
    )
    for patch, expected in fixtures:
        digest = hashlib.sha256(render(patch).astype("<f4").tobytes()).hexdigest()
        assert digest == expected


@pytest.mark.parametrize("waveform", ("sine", "triangle", "saw", "square"))
def test_bounds_remain_finite_and_unclipped_without_peak_scaling(waveform):
    for notes in ((4800,), (4800, 4801, 4802), (7200,)):
        patch = Patch(
            notes_cents=notes,
            waveform=waveform,
            brightness_hz=12000,
            attack_ms=5,
            decay_ms=20,
            sustain_db=0,
            release_ms=750,
            gain_db=0,
        )
        audio = render(patch)
        assert np.isfinite(audio).all()
        assert 0.0 < np.max(np.abs(audio)) < 1.0
        assert not np.any(audio[84_000:])
    quiet = render(
        Patch(
            waveform=waveform,
            brightness_hz=200,
            attack_ms=250,
            decay_ms=500,
            sustain_db=-24,
            release_ms=20,
            gain_db=-18,
        )
    )
    assert np.isfinite(quiet).all()
    assert not np.any(quiet[48_960:])


def test_chord_permutations_have_identical_patch_bytes_and_audio():
    expected_patch = default_patch(3)
    expected_bytes = json.dumps(expected_patch.to_dict(), sort_keys=True)
    expected_audio = render(expected_patch)
    for ordering in itertools.permutations(expected_patch.notes_cents):
        patch = Patch(notes_cents=ordering)
        assert json.dumps(patch.to_dict(), sort_keys=True) == expected_bytes
        np.testing.assert_array_equal(render(patch), expected_audio)


@pytest.mark.parametrize("waveform", ("triangle", "saw", "square"))
def test_harmonic_recurrence_agrees_with_independent_direct_sine_sum(waveform):
    cents = 4800
    frequency = 440.0 * 2.0 ** ((cents - 6900) / 1200.0)
    indices = (0, 1, 47, 239, 53017, 83999)
    actual = _voice(cents, waveform, 84_000)
    expected = []
    for sample in indices:
        total = 0.0
        for harmonic in range(1, 185):
            if harmonic * frequency >= SAMPLE_RATE_HZ / 2:
                break
            if waveform == "saw":
                weight = (-1.0) ** (harmonic + 1) / harmonic
                factor = 2.0 / math.pi
            elif waveform == "square":
                weight = 1.0 / harmonic if harmonic % 2 else 0.0
                factor = 4.0 / math.pi
            else:
                weight = (-1.0) ** ((harmonic - 1) // 2) / harmonic**2 if harmonic % 2 else 0.0
                factor = 8.0 / math.pi**2
            total += weight * math.sin(math.tau * harmonic * frequency * sample / SAMPLE_RATE_HZ)
        expected.append(total * factor)
    np.testing.assert_allclose(actual[list(indices)], expected, rtol=0, atol=2e-11)


def test_envelope_endpoints_and_release_are_sample_defined():
    reference_patch = Patch(
        notes_cents=(6064,),
        brightness_hz=12000,
        attack_ms=5,
        decay_ms=20,
        sustain_db=0,
        release_ms=750,
        gain_db=0,
    )
    reference = render(reference_patch)
    candidate = render(
        replace(reference_patch, attack_ms=10, decay_ms=100, sustain_db=-6, release_ms=200)
    )
    sustain = 10.0 ** (-6.0 / 20.0)
    endpoints = {
        1: 1 / 480,
        239: 239 / 480,
        479: 479 / 480,
        480: 1.0,
        481: 1.0 + (sustain - 1.0) / 4800,
        5279: 1.0 + (sustain - 1.0) * 4799 / 4800,
        5280: sustain,
        47999: sustain,
        48000: sustain,
        48001: sustain * (1 - 1 / 9600),
        57599: sustain / 9600,
    }
    for index, level in endpoints.items():
        reference_level = (
            index / 240
            if index < 240
            else (1.0 if index < 48000 else 1.0 - (index - 48000) / 36000)
        )
        expected = float(reference[index]) * level / reference_level
        assert candidate[index] == pytest.approx(expected, rel=3e-7, abs=1e-9)
    assert candidate[0] == 0.0
    assert not np.any(candidate[57_600:])


def test_gain_is_absolute_and_does_not_normalize_each_render():
    patch = default_patch(3)
    loud = render(replace(patch, gain_db=0))
    quiet = render(replace(patch, gain_db=-12))
    np.testing.assert_allclose(quiet, loud * 10.0 ** (-12 / 20), rtol=3e-7, atol=1e-8)
