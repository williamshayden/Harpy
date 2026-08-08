import itertools
import math
from dataclasses import replace

import numpy as np
import pytest

from harpy.synth.engine import SynthEngine
from harpy.synth.models import EnvelopeConfig, RenderConfig, SynthPatch

INVALID_FREQUENCIES = [math.nan, math.inf, -math.inf, 0.0, -1.0, 24_000.0, 24_001.0]
EXPECTED_LINEAR_ENGINE_BITS = np.array(
    [
        0,
        1025724698,
        1036759167,
        1041246260,
        1040312847,
        1026296886,
        3183795601,
        3192413184,
        3194009889,
        3190642546,
        3180535862,
        1023623637,
        1036772298,
        1038855006,
        1035049145,
        1025520120,
        584321663,
        3159726069,
        2147483648,
    ],
    dtype=np.uint32,
)


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


def render_partitioned_event(
    engine: SynthEngine,
    *,
    note_off_frame: int,
    total_frames: int,
    partition: list[int],
) -> np.ndarray:
    engine.note_on(3.0)
    rendered: list[np.ndarray] = []
    position = 0
    note_off_sent = False
    chunks = itertools.cycle(partition)
    while position < total_frames:
        if not note_off_sent and position == note_off_frame:
            engine.note_off()
            note_off_sent = True
        requested = next(chunks)
        stop = min(position + requested, total_frames)
        if not note_off_sent:
            stop = min(stop, note_off_frame)
        assert stop > position
        rendered.append(engine.render(stop - position))
        position = stop
    assert note_off_sent
    return np.concatenate(rendered).astype(np.float32, copy=False)


def test_awkward_frame_linear_engine_bits_are_pinned() -> None:
    render = RenderConfig(sample_rate_hz=32)
    patch = SynthPatch(
        envelope=EnvelopeConfig(
            attack_seconds=7 / 32,
            decay_seconds=9 / 32,
            sustain_db=-7.3,
            release_seconds=8 / 32,
        )
    )
    engine = SynthEngine(render, patch)
    engine.note_on(3.0)
    attack_and_decay = engine.render(11)
    engine.note_off()
    release = engine.render(8)

    np.testing.assert_array_equal(
        np.concatenate((attack_and_decay, release)).view(np.uint32),
        EXPECTED_LINEAR_ENGINE_BITS,
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
    expected = whole.render(100_000)
    actual = np.concatenate(
        [chunked.render(1), chunked.render(255), chunked.render(4_096), chunked.render(95_648)]
    )
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("curvature", [-1.0, -0.35, 0.0, 0.65, 1.0])
@pytest.mark.parametrize(
    ("source_stage", "note_off_frame", "release_final_word"),
    [
        ("attack", 3, 0),
        ("decay", 11, 2_147_483_648),
        ("sustain", 18, 0),
    ],
)
def test_curved_events_are_bit_exact_under_every_block_partition(
    curvature: float,
    source_stage: str,
    note_off_frame: int,
    release_final_word: int,
) -> None:
    render = RenderConfig(sample_rate_hz=64)
    patch = SynthPatch(
        envelope=EnvelopeConfig(
            attack_seconds=7 / 64,
            decay_seconds=9 / 64,
            sustain_db=20.0 * math.log10(0.5),
            release_seconds=8 / 64,
            attack_curve=curvature,
            decay_curve=curvature,
            release_curve=curvature,
        ),
        output_gain_dbfs=0.0,
    )
    total_frames = note_off_frame + 8 + 5
    expected = render_partitioned_event(
        SynthEngine(render, patch),
        note_off_frame=note_off_frame,
        total_frames=total_frames,
        partition=[total_frames],
    )

    for partition in ([1], [3, 5, 7], [256], [total_frames]):
        actual = render_partitioned_event(
            SynthEngine(render, patch),
            note_off_frame=note_off_frame,
            total_frames=total_frames,
            partition=list(partition),
        )
        words = actual.view(np.uint32)
        np.testing.assert_array_equal(words, expected.view(np.uint32), err_msg=source_stage)
        assert words[note_off_frame + 8 - 1] == release_final_word
        np.testing.assert_array_equal(
            words[note_off_frame + 8 :],
            np.full(5, np.float32(0.0).view(np.uint32), dtype=np.uint32),
        )


def test_retune_is_bit_exact_at_a_fixed_frame_across_partitions() -> None:
    whole = SynthEngine(RenderConfig(), SynthPatch())
    partitioned = SynthEngine(RenderConfig(), SynthPatch())
    whole.note_on(220.0)
    partitioned.note_on(220.0)

    whole_before = whole.render(997)
    partitioned_before = np.concatenate(
        [partitioned.render(1), partitioned.render(255), partitioned.render(741)]
    )
    whole.retune(330.0)
    partitioned.retune(330.0)
    whole_after = whole.render(321)
    partitioned_after = np.concatenate([partitioned.render(17), partitioned.render(304)])

    np.testing.assert_array_equal(
        partitioned_before.view(np.uint32),
        whole_before.view(np.uint32),
    )
    np.testing.assert_array_equal(
        partitioned_after.view(np.uint32),
        whole_after.view(np.uint32),
    )


def test_active_zero_length_render_does_not_reanchor_phase() -> None:
    engine = SynthEngine(RenderConfig(), SynthPatch())
    control = SynthEngine(RenderConfig(), SynthPatch())
    engine.note_on(220.0)
    control.note_on(220.0)
    engine.render(997)
    control.render(997)

    engine.render(0)

    np.testing.assert_array_equal(engine.render(256), control.render(256))


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
