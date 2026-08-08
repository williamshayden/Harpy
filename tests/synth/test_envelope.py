import math
from dataclasses import replace

import numpy as np
import pytest

from harpy.synth.curves import evaluate_quadratic_segment, sample_envelope_preview
from harpy.synth.envelope import AdsrEnvelope, EnvelopeStage
from harpy.synth.models import EnvelopeConfig, RenderConfig

EXPECTED_LINEAR_ENVELOPE_BITS = np.array(
    [
        4594314991293244562,
        4598818590920615058,
        4601392076421969627,
        4603322190547985554,
        4604608933298662838,
        4605895676049340122,
        4607182418800017408,
        4606613483139180583,
        4606044547478343758,
        4605475611817506933,
        4604906676156670108,
        4604065244080245896,
        4603223812003821684,
        4602085940682148033,
        4600403076529299610,
        4598720212376451187,
        4595899476901929112,
        4591395877274558612,
        0,
    ],
    dtype=np.uint64,
)


def half_sustain_config() -> EnvelopeConfig:
    return EnvelopeConfig(
        attack_seconds=1.0,
        decay_seconds=1.0,
        sustain_db=20.0 * math.log10(0.5),
        release_seconds=1.0,
    )


def test_awkward_frame_linear_envelope_bits_are_pinned() -> None:
    envelope = AdsrEnvelope(
        EnvelopeConfig(
            attack_seconds=0.7,
            decay_seconds=0.9,
            sustain_db=-7.3,
            release_seconds=0.8,
        ),
        sample_rate_hz=10,
    )
    envelope.note_on()
    attack_and_decay = envelope.render(11)
    envelope.note_off()
    release = envelope.render(8)

    np.testing.assert_array_equal(
        np.concatenate((attack_and_decay, release)).view(np.uint64),
        EXPECTED_LINEAR_ENVELOPE_BITS,
    )


def test_attack_decay_and_sustain_emit_exact_endpoints() -> None:
    envelope = AdsrEnvelope(half_sustain_config(), sample_rate_hz=4)
    envelope.note_on()
    np.testing.assert_allclose(envelope.render(4), [0.25, 0.5, 0.75, 1.0])
    assert envelope.stage is EnvelopeStage.DECAY
    np.testing.assert_allclose(envelope.render(4), [0.875, 0.75, 0.625, 0.5])
    assert envelope.stage is EnvelopeStage.SUSTAIN
    np.testing.assert_allclose(envelope.render(3), [0.5, 0.5, 0.5])


def test_early_note_off_releases_from_last_emitted_level() -> None:
    envelope = AdsrEnvelope(half_sustain_config(), sample_rate_hz=4)
    envelope.note_on()
    np.testing.assert_allclose(envelope.render(1), [0.25])
    envelope.note_off()
    np.testing.assert_allclose(envelope.render(4), [0.1875, 0.125, 0.0625, 0.0])
    assert envelope.stage is EnvelopeStage.IDLE
    assert envelope.level == 0.0
    np.testing.assert_array_equal(envelope.render(3), np.zeros(3))


def test_note_off_is_idempotent_during_release() -> None:
    envelope = AdsrEnvelope(half_sustain_config(), sample_rate_hz=4)
    envelope.note_on()
    envelope.render(4)
    envelope.note_off()
    first = envelope.render(1)
    envelope.note_off()
    rest = envelope.render(3)
    np.testing.assert_allclose(np.concatenate((first, rest)), [0.75, 0.5, 0.25, 0.0])


def test_retrigger_restarts_attack_from_zero() -> None:
    envelope = AdsrEnvelope(half_sustain_config(), sample_rate_hz=4)
    envelope.note_on()
    envelope.render(3)
    envelope.note_on()
    np.testing.assert_allclose(envelope.render(2), [0.25, 0.5])


@pytest.mark.parametrize("frame_count", [-1, 1.5, True])
def test_render_rejects_invalid_frame_counts(frame_count: object) -> None:
    envelope = AdsrEnvelope(half_sustain_config(), sample_rate_hz=4)
    with pytest.raises((TypeError, ValueError)):
        envelope.render(frame_count)  # type: ignore[arg-type]


def test_curved_attack_decay_and_release_use_stage_specific_curves() -> None:
    config = replace(
        half_sustain_config(),
        attack_curve=-1.0,
        decay_curve=1.0,
        release_curve=-1.0,
    )
    envelope = AdsrEnvelope(config, sample_rate_hz=4)
    envelope.note_on()
    np.testing.assert_array_equal(envelope.render(4), [0.0625, 0.25, 0.5625, 1.0])
    np.testing.assert_array_equal(envelope.render(4), [0.78125, 0.625, 0.53125, 0.5])
    envelope.note_off()
    np.testing.assert_array_equal(envelope.render(4), [0.46875, 0.375, 0.21875, 0.0])


@pytest.mark.parametrize("curvature", [-1.0, -0.35, 0.0, 0.65, 1.0])
@pytest.mark.parametrize("source_stage", ["attack", "decay", "sustain"])
def test_release_snapshots_last_level_and_runs_its_full_segment(
    curvature: float,
    source_stage: str,
) -> None:
    config = replace(
        half_sustain_config(),
        attack_curve=curvature,
        decay_curve=curvature,
        release_curve=curvature,
    )
    envelope = AdsrEnvelope(config, sample_rate_hz=8)
    envelope.note_on()
    if source_stage == "attack":
        envelope.render(3)
    elif source_stage == "decay":
        envelope.render(8 + 3)
    else:
        envelope.render(8 + 8 + 2)
    last_level = envelope.level

    envelope.note_off()

    assert envelope.stage is EnvelopeStage.RELEASE
    assert envelope.release_frames_remaining == 8
    release = envelope.render(8)
    assert release[0] == pytest.approx(
        evaluate_quadratic_segment(last_level, 0.0, curvature, 1 / 8),
        rel=0.0,
        abs=1e-12,
    )
    assert release[-1] == 0.0
    assert envelope.stage is EnvelopeStage.IDLE


def test_note_off_before_first_attack_sample_emits_full_zero_release() -> None:
    envelope = AdsrEnvelope(half_sustain_config(), sample_rate_hz=4)
    envelope.note_on()

    envelope.note_off()

    assert envelope.stage is EnvelopeStage.RELEASE
    assert envelope.release_frames_remaining == 4
    np.testing.assert_array_equal(envelope.render(3), np.zeros(3))
    assert envelope.stage is EnvelopeStage.RELEASE
    assert envelope.release_frames_remaining == 1
    np.testing.assert_array_equal(envelope.render(1), np.zeros(1))
    assert envelope.stage is EnvelopeStage.IDLE


def test_zero_sustain_still_emits_full_zero_release() -> None:
    config = replace(half_sustain_config(), sustain_db=-10_000.0)
    envelope = AdsrEnvelope(config, sample_rate_hz=4)
    envelope.note_on()
    envelope.render(8)
    assert envelope.stage is EnvelopeStage.SUSTAIN
    assert envelope.level == 0.0

    envelope.note_off()

    assert envelope.stage is EnvelopeStage.RELEASE
    np.testing.assert_array_equal(envelope.render(3), np.zeros(3))
    assert envelope.stage is EnvelopeStage.RELEASE
    np.testing.assert_array_equal(envelope.render(1), np.zeros(1))
    assert envelope.stage is EnvelopeStage.IDLE


@pytest.mark.parametrize("frames", [1, 2, 7, 257])
@pytest.mark.parametrize("curvature", [-1.0, -0.73, 0.0, 0.61, 1.0])
@pytest.mark.parametrize("stage", ["attack", "decay", "release"])
def test_preview_endpoint_excluded_samples_match_renderer(
    frames: int,
    curvature: float,
    stage: str,
) -> None:
    sample_rate_hz = 257
    duration = frames / sample_rate_hz
    config = EnvelopeConfig(
        attack_seconds=duration,
        decay_seconds=duration,
        sustain_db=20.0 * math.log10(0.5),
        release_seconds=duration,
        attack_curve=curvature if stage == "attack" else 0.0,
        decay_curve=curvature if stage == "decay" else 0.0,
        release_curve=curvature if stage == "release" else 0.0,
    )
    preview = sample_envelope_preview(
        config,
        RenderConfig(sample_rate_hz=sample_rate_hz),
        samples_per_stage=frames + 1,
    )
    envelope = AdsrEnvelope(config, sample_rate_hz)
    envelope.note_on()
    if stage == "attack":
        rendered_stage = envelope.render(frames)
        preview_stage = preview.attack_level
    elif stage == "decay":
        envelope.render(frames)
        rendered_stage = envelope.render(frames)
        preview_stage = preview.decay_level
    else:
        envelope.render(frames * 2)
        envelope.note_off()
        rendered_stage = envelope.render(frames)
        preview_stage = preview.release_level
    np.testing.assert_allclose(
        rendered_stage,
        preview_stage[1:],
        rtol=0.0,
        atol=1e-12,
    )


def test_one_frame_stages_emit_targets_before_transitioning() -> None:
    config = EnvelopeConfig(
        attack_seconds=0.25,
        decay_seconds=0.25,
        sustain_db=20.0 * math.log10(0.5),
        release_seconds=0.25,
        attack_curve=-1.0,
        decay_curve=1.0,
        release_curve=-1.0,
    )
    envelope = AdsrEnvelope(config, sample_rate_hz=4)

    envelope.note_on()
    assert envelope.stage is EnvelopeStage.ATTACK
    np.testing.assert_array_equal(envelope.render(1), [1.0])
    assert envelope.stage is EnvelopeStage.DECAY
    np.testing.assert_array_equal(envelope.render(1), [0.5])
    assert envelope.stage is EnvelopeStage.SUSTAIN
    envelope.note_off()
    assert envelope.stage is EnvelopeStage.RELEASE
    np.testing.assert_array_equal(envelope.render(1), [0.0])
    assert envelope.stage is EnvelopeStage.IDLE
