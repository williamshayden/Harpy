import math
from dataclasses import replace

import numpy as np
import pytest

from harpy.synth.engine import SynthEngine
from harpy.synth.models import EnvelopeConfig, RenderConfig, SynthPatch

INVALID_FREQUENCIES = [math.nan, math.inf, -math.inf, 0.0, -1.0, 24_000.0, 24_001.0]


def short_patch(*, output_gain_dbfs: float = 0.0) -> SynthPatch:
    return SynthPatch(
        envelope=EnvelopeConfig(
            attack_seconds=0.5,
            decay_seconds=0.5,
            sustain_db=20.0 * math.log10(0.5),
            release_seconds=0.5,
        ),
        output_gain_dbfs=output_gain_dbfs,
    )


def test_idle_engine_emits_exact_mono_float32_silence() -> None:
    samples = SynthEngine(RenderConfig(), SynthPatch()).render(256)
    assert samples.shape == (256,)
    assert samples.dtype == np.float32
    assert samples.flags.c_contiguous
    np.testing.assert_array_equal(samples, np.zeros(256, dtype=np.float32))


def test_live_retune_preserves_phase_and_envelope() -> None:
    retuned = SynthEngine(RenderConfig(), SynthPatch())
    control = SynthEngine(RenderConfig(), SynthPatch())
    retuned.note_on(220.0)
    control.note_on(220.0)
    np.testing.assert_array_equal(retuned.render(1_000), control.render(1_000))
    retuned.retune(330.0)
    np.testing.assert_allclose(retuned.render(1), control.render(1), rtol=0.0, atol=1e-7)
    assert not np.array_equal(retuned.render(64), control.render(64))


def test_render_is_block_partition_invariant() -> None:
    whole = SynthEngine(RenderConfig(), SynthPatch())
    chunked = SynthEngine(RenderConfig(), SynthPatch())
    whole.note_on(261.6255653005986)
    chunked.note_on(261.6255653005986)
    expected = whole.render(10_000)
    actual = np.concatenate(
        [chunked.render(1), chunked.render(255), chunked.render(4_096), chunked.render(5_648)]
    )
    np.testing.assert_allclose(expected, actual, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("frequency_hz", INVALID_FREQUENCIES)
def test_note_on_rejects_invalid_frequency_without_mutating_active_state(
    frequency_hz: float,
) -> None:
    engine = SynthEngine(RenderConfig(), SynthPatch())
    control = SynthEngine(RenderConfig(), SynthPatch())
    engine.note_on(220.0)
    control.note_on(220.0)
    np.testing.assert_array_equal(engine.render(1_000), control.render(1_000))

    with pytest.raises(ValueError, match="Nyquist"):
        engine.note_on(frequency_hz)

    np.testing.assert_array_equal(engine.render(256), control.render(256))


@pytest.mark.parametrize("frequency_hz", INVALID_FREQUENCIES)
def test_retune_rejects_invalid_frequency_without_mutating_active_state(
    frequency_hz: float,
) -> None:
    engine = SynthEngine(RenderConfig(), SynthPatch())
    control = SynthEngine(RenderConfig(), SynthPatch())
    engine.note_on(220.0)
    control.note_on(220.0)
    np.testing.assert_array_equal(engine.render(1_000), control.render(1_000))

    with pytest.raises(ValueError, match="Nyquist"):
        engine.retune(frequency_hz)

    np.testing.assert_array_equal(engine.render(256), control.render(256))


def test_retune_raises_clear_state_error_while_idle() -> None:
    engine = SynthEngine(RenderConfig(), SynthPatch())

    with pytest.raises(RuntimeError, match="idle"):
        engine.retune(220.0)


def test_retune_works_during_release_without_retriggering_envelope() -> None:
    retuned = SynthEngine(RenderConfig(), SynthPatch())
    control = SynthEngine(RenderConfig(), SynthPatch())
    retuned.note_on(220.0)
    control.note_on(220.0)
    retuned.render(1_000)
    control.render(1_000)
    retuned.note_off()
    control.note_off()

    retuned.retune(330.0)

    np.testing.assert_allclose(retuned.render(1), control.render(1), rtol=0.0, atol=1e-7)
    assert not np.array_equal(retuned.render(64), control.render(64))


def test_note_on_resets_phase_and_retriggers_envelope_from_silence() -> None:
    engine = SynthEngine(RenderConfig(), SynthPatch())
    fresh = SynthEngine(RenderConfig(), SynthPatch())
    engine.note_on(220.0)
    engine.render(1_000)

    engine.note_on(330.0)
    fresh.note_on(330.0)

    actual = engine.render(256)
    expected = fresh.render(256)
    assert actual[0] == 0.0
    np.testing.assert_array_equal(actual, expected)


def test_note_off_releases_from_last_level_and_reaches_exact_silence() -> None:
    render = RenderConfig(sample_rate_hz=8)
    engine = SynthEngine(render, short_patch())
    engine.note_on(1.0)
    engine.render(2)

    engine.note_off()

    np.testing.assert_allclose(
        engine.render(4),
        [0.375, 0.1767766952966369, 0.0, 0.0],
        rtol=0.0,
        atol=1e-7,
    )
    assert engine.is_idle
    np.testing.assert_array_equal(engine.render(8), np.zeros(8, dtype=np.float32))


def test_repeated_note_off_is_idempotent() -> None:
    repeated = SynthEngine(RenderConfig(), SynthPatch())
    control = SynthEngine(RenderConfig(), SynthPatch())
    repeated.note_on(220.0)
    control.note_on(220.0)
    repeated.render(1_000)
    control.render(1_000)
    repeated.note_off()
    control.note_off()
    np.testing.assert_array_equal(repeated.render(1), control.render(1))

    repeated.note_off()

    np.testing.assert_array_equal(repeated.render(1_000), control.render(1_000))


def test_replace_patch_is_atomic_and_resets_to_exact_silence() -> None:
    render = RenderConfig(sample_rate_hz=8)
    original_patch = short_patch()
    engine = SynthEngine(render, original_patch)
    control = SynthEngine(render, original_patch)
    engine.note_on(1.0)
    control.note_on(1.0)
    np.testing.assert_array_equal(engine.render(2), control.render(2))
    invalid_patch = replace(
        original_patch,
        envelope=replace(original_patch.envelope, attack_seconds=0.01),
    )

    with pytest.raises(ValueError, match="attack"):
        engine.replace_patch(invalid_patch)

    assert engine.patch is original_patch
    np.testing.assert_array_equal(engine.render(2), control.render(2))

    replacement = short_patch(output_gain_dbfs=-6.0)
    engine.replace_patch(replacement)

    assert engine.patch is replacement
    assert engine.is_idle
    np.testing.assert_array_equal(engine.render(8), np.zeros(8, dtype=np.float32))


def test_constructor_rejects_an_unrenderable_patch() -> None:
    patch = replace(
        SynthPatch(),
        envelope=replace(SynthPatch().envelope, release_seconds=0.000001),
    )

    with pytest.raises(ValueError, match="release"):
        SynthEngine(RenderConfig(), patch)


def test_reset_immediately_clears_oscillator_and_envelope_state() -> None:
    engine = SynthEngine(RenderConfig(), SynthPatch())
    engine.note_on(220.0)
    engine.render(1_000)

    engine.reset()

    assert engine.is_idle
    with pytest.raises(RuntimeError, match="idle"):
        engine.retune(330.0)
    np.testing.assert_array_equal(engine.render(256), np.zeros(256, dtype=np.float32))


@pytest.mark.parametrize("frame_count", [True, False, 1.5, "256"])
def test_render_rejects_non_integer_frame_counts(frame_count: object) -> None:
    engine = SynthEngine(RenderConfig(), SynthPatch())

    with pytest.raises(TypeError, match="integer"):
        engine.render(frame_count)  # type: ignore[arg-type]


def test_render_rejects_negative_frame_counts() -> None:
    engine = SynthEngine(RenderConfig(), SynthPatch())

    with pytest.raises(ValueError, match="non-negative"):
        engine.render(-1)


def test_zero_length_render_returns_a_new_contiguous_float32_array() -> None:
    engine = SynthEngine(RenderConfig(), SynthPatch())

    first = engine.render(0)
    second = engine.render(0)

    assert first.shape == (0,)
    assert first.dtype == np.float32
    assert first.flags.c_contiguous
    assert first is not second
