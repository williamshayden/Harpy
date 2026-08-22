from types import SimpleNamespace

import numpy as np
import pytest

import harpy.envs.spectrum as spectrum_module
from harpy.analysis import analyze
from harpy.envs.spectrum import (
    GYM_ANALYSIS_CONFIG,
    LOG_FREQUENCY_GRID_HZ,
    LOG_SPECTRUM_SIZE,
    encode_log_spectrum,
)
from harpy.synth import RenderConfig, SynthEngine, SynthPatch
from harpy.tuning import Tuning


def render_candidate_audio(cents: int, fft_frames: int = 262_144) -> np.ndarray:
    render = RenderConfig()
    patch = SynthPatch()
    engine = SynthEngine(render, patch)
    frequency_hz = Tuning().frequency_hz_for_midi_coordinate(cents / 100.0)
    engine.note_on(frequency_hz)
    attack_frames = int(patch.envelope.attack_seconds * render.sample_rate_hz + 0.5)
    decay_frames = int(patch.envelope.decay_seconds * render.sample_rate_hz + 0.5)
    engine.render(attack_frames + decay_frames)
    return engine.render(fft_frames)


def test_log_grid_is_five_cent_coordinate_grid_covering_every_candidate() -> None:
    assert LOG_SPECTRUM_SIZE == 1_961
    assert LOG_FREQUENCY_GRID_HZ.shape == (1_961,)
    tuning = Tuning()
    assert LOG_FREQUENCY_GRID_HZ[0] == pytest.approx(tuning.frequency_hz_for_midi_coordinate(11.0))
    assert LOG_FREQUENCY_GRID_HZ[-1] == pytest.approx(
        tuning.frequency_hz_for_midi_coordinate(109.0)
    )
    coordinates = [
        tuning.midi_coordinate_for_frequency_hz(float(value)) for value in LOG_FREQUENCY_GRID_HZ
    ]
    assert np.diff(coordinates) == pytest.approx(np.full(1_960, 0.05))
    assert LOG_FREQUENCY_GRID_HZ.flags.owndata
    assert not LOG_FREQUENCY_GRID_HZ.flags.writeable


def test_encoder_returns_owned_bounded_float32_evidence() -> None:
    encoded = encode_log_spectrum(render_candidate_audio(6_000))
    assert encoded.shape == (1_961,)
    assert encoded.dtype == np.float32
    assert encoded.flags.c_contiguous
    assert encoded.flags.owndata
    assert np.all((encoded >= 0.0) & (encoded <= 1.0))


def test_dedicated_analysis_source_strictly_brackets_log_grid() -> None:
    observation = analyze(render_candidate_audio(6_000), 48_000, GYM_ANALYSIS_CONFIG)

    assert observation.spectrum_frequency_hz[0] == 48_000 / 262_144
    assert observation.spectrum_frequency_hz[-1] == 24_000.0
    assert observation.spectrum_frequency_hz[0] < LOG_FREQUENCY_GRID_HZ[0]
    assert observation.spectrum_frequency_hz[-1] > LOG_FREQUENCY_GRID_HZ[-1]


def test_silence_returns_fresh_fixed_zero_evidence() -> None:
    first = encode_log_spectrum(np.zeros(262_144, dtype=np.float32))
    second = encode_log_spectrum(np.zeros(262_144, dtype=np.float32))

    np.testing.assert_array_equal(first, np.zeros(1_961, dtype=np.float32))
    np.testing.assert_array_equal(second, first)
    assert first.flags.owndata
    assert second.flags.owndata
    assert not np.shares_memory(first, second)


def test_encoder_clips_and_maps_decibels_to_the_exact_unit_interval(monkeypatch) -> None:
    sample_indices = np.array([0, 517, 1_203, LOG_SPECTRUM_SIZE - 1])
    levels_dbfs = np.full(LOG_SPECTRUM_SIZE, -120.0)
    levels_dbfs[sample_indices] = [-121.0, -30.0, 0.0, 12.0]
    observation = SimpleNamespace(
        has_signal=True,
        spectrum_frequency_hz=LOG_FREQUENCY_GRID_HZ,
        spectrum_level_dbfs=levels_dbfs,
    )
    monkeypatch.setattr(spectrum_module, "analyze", lambda *_args: observation)

    encoded = encode_log_spectrum(np.zeros(262_144, dtype=np.float32))

    np.testing.assert_array_equal(
        encoded[sample_indices],
        np.array([0.0, 0.75, 1.0, 1.0], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        (np.zeros((262_144, 1), dtype=np.float32), "one-dimensional"),
        (np.zeros(262_143, dtype=np.float32), "at least 262144 samples"),
    ],
)
def test_bad_rank_and_short_inputs_preserve_analyzer_validation(
    samples: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        encode_log_spectrum(samples)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_encoder_rejects_nonfinite_samples(nonfinite: float) -> None:
    samples = np.zeros(262_144, dtype=np.float32)
    samples[17] = nonfinite

    with pytest.raises(ValueError, match="finite"):
        encode_log_spectrum(samples)


@pytest.mark.parametrize("cents", [1_100, 10_900])
def test_candidate_coordinate_extremes_encode_without_extrapolation(cents: int) -> None:
    encoded = encode_log_spectrum(render_candidate_audio(cents))

    assert encoded.shape == (1_961,)
    assert np.all(np.isfinite(encoded))


def test_every_integer_cent_source_is_represented_within_five_cents() -> None:
    maximum_error_cents = 0
    for source_cents in range(4_800, 7_201):
        encoded = encode_log_spectrum(render_candidate_audio(source_cents))
        estimated_cents = 1_100 + 5 * int(np.argmax(encoded))
        maximum_error_cents = max(
            maximum_error_cents,
            abs(estimated_cents - source_cents),
        )

    assert maximum_error_cents <= 5
