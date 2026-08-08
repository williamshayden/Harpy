import math

import numpy as np
import pytest

from harpy.config import DEFAULT_CONFIG
from harpy.pitch import Pitch
from harpy.synth.reference import SineVoice


def make_voice() -> SineVoice:
    return SineVoice(
        tuning=DEFAULT_CONFIG.tuning,
        render_spec=DEFAULT_CONFIG.render,
        patch=DEFAULT_CONFIG.patch,
    )


def test_idle_voice_emits_exact_mono_float32_silence() -> None:
    samples = make_voice().render_block(256)
    assert samples.shape == (256,)
    assert samples.dtype == np.float32
    np.testing.assert_array_equal(samples, np.zeros(256, dtype=np.float32))


def test_note_on_resets_phase_and_respects_peak_gain() -> None:
    voice = make_voice()
    voice.note_on(Pitch.from_midi(60))
    first = voice.render_block(2_048)
    voice.note_on(Pitch.from_midi(60))
    second = voice.render_block(2_048)
    assert first[0] == 0.0
    np.testing.assert_array_equal(first, second)
    assert np.max(np.abs(first)) <= DEFAULT_CONFIG.patch.peak_gain + 1e-7


def test_rendering_is_invariant_to_block_partitioning() -> None:
    whole_voice = make_voice()
    chunked_voice = make_voice()
    pitch = Pitch.from_midi(60)
    whole_voice.note_on(pitch)
    chunked_voice.note_on(pitch)
    whole = whole_voice.render_block(10_000)
    chunked = np.concatenate(
        [
            chunked_voice.render_block(1),
            chunked_voice.render_block(255),
            chunked_voice.render_block(4_096),
            chunked_voice.render_block(5_648),
        ]
    )
    np.testing.assert_allclose(whole, chunked, rtol=0.0, atol=1e-6)


def test_release_reaches_exact_silence() -> None:
    voice = make_voice()
    voice.note_on(Pitch.from_midi(60))
    voice.render_block(1_000)
    voice.note_off()
    voice.render_block(DEFAULT_CONFIG.patch.envelope.release_frames(48_000))
    np.testing.assert_array_equal(
        voice.render_block(256),
        np.zeros(256, dtype=np.float32),
    )


def test_fft_peak_matches_middle_c_within_one_bin() -> None:
    voice = make_voice()
    voice.note_on(Pitch.from_midi(60))
    settle_frames = DEFAULT_CONFIG.patch.envelope.attack_frames(
        48_000
    ) + DEFAULT_CONFIG.patch.envelope.decay_frames(48_000)
    voice.render_block(settle_frames)
    samples = voice.render_block(96_000).astype(np.float64)
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size)))
    frequencies = np.fft.rfftfreq(samples.size, d=1.0 / 48_000)
    peak_hz = frequencies[int(np.argmax(spectrum[1:]) + 1)]
    expected_hz = DEFAULT_CONFIG.tuning.frequency_hz(Pitch.from_midi(60))
    assert abs(peak_hz - expected_hz) <= 0.5


def test_nyquist_frequency_is_rejected() -> None:
    voice = make_voice()
    above_nyquist_cents = 6_900.0 + 1_200.0 * math.log2(25_000.0 / 440.0)
    with pytest.raises(ValueError, match="Nyquist"):
        voice.note_on(Pitch(above_nyquist_cents))
