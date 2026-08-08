import math

import numpy as np
import pytest

from harpy.synth.envelope import EnvelopeStage, LinearEnvelope
from harpy.synth.models import EnvelopeConfig


def half_sustain_config() -> EnvelopeConfig:
    return EnvelopeConfig(
        attack_seconds=1.0,
        decay_seconds=1.0,
        sustain_db=20.0 * math.log10(0.5),
        release_seconds=1.0,
    )


def test_attack_decay_and_sustain_emit_exact_endpoints() -> None:
    envelope = LinearEnvelope(half_sustain_config(), sample_rate_hz=4)
    envelope.note_on()
    np.testing.assert_allclose(envelope.render(4), [0.25, 0.5, 0.75, 1.0])
    assert envelope.stage is EnvelopeStage.DECAY
    np.testing.assert_allclose(envelope.render(4), [0.875, 0.75, 0.625, 0.5])
    assert envelope.stage is EnvelopeStage.SUSTAIN
    np.testing.assert_allclose(envelope.render(3), [0.5, 0.5, 0.5])


def test_early_note_off_releases_from_last_emitted_level() -> None:
    envelope = LinearEnvelope(half_sustain_config(), sample_rate_hz=4)
    envelope.note_on()
    np.testing.assert_allclose(envelope.render(1), [0.25])
    envelope.note_off()
    np.testing.assert_allclose(envelope.render(4), [0.1875, 0.125, 0.0625, 0.0])
    assert envelope.stage is EnvelopeStage.IDLE
    assert envelope.level == 0.0
    np.testing.assert_array_equal(envelope.render(3), np.zeros(3))


def test_note_off_is_idempotent_during_release() -> None:
    envelope = LinearEnvelope(half_sustain_config(), sample_rate_hz=4)
    envelope.note_on()
    envelope.render(4)
    envelope.note_off()
    first = envelope.render(1)
    envelope.note_off()
    rest = envelope.render(3)
    np.testing.assert_allclose(np.concatenate((first, rest)), [0.75, 0.5, 0.25, 0.0])


def test_retrigger_restarts_attack_from_zero() -> None:
    envelope = LinearEnvelope(half_sustain_config(), sample_rate_hz=4)
    envelope.note_on()
    envelope.render(3)
    envelope.note_on()
    np.testing.assert_allclose(envelope.render(2), [0.25, 0.5])


@pytest.mark.parametrize("frame_count", [-1, 1.5, True])
def test_render_rejects_invalid_frame_counts(frame_count: object) -> None:
    envelope = LinearEnvelope(half_sustain_config(), sample_rate_hz=4)
    with pytest.raises((TypeError, ValueError)):
        envelope.render(frame_count)  # type: ignore[arg-type]
