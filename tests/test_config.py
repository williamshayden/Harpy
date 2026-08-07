import math

import pytest

from harpy.config import DEFAULT_CONFIG, AppConfig
from harpy.gui.specs import KeyboardViewSpec
from harpy.pitch import EqualTemperament, MidiNote
from harpy.synth.specs import EnvelopeSpec, RenderSpec, SinePatch, seconds_to_frames


def test_default_configuration_matches_approved_values() -> None:
    config = DEFAULT_CONFIG
    assert config.tuning.reference_note == MidiNote(69)
    assert config.tuning.reference_hz == 440.0
    assert config.keyboard == KeyboardViewSpec()
    assert config.render == RenderSpec()
    assert config.patch.envelope == EnvelopeSpec()
    assert config.patch.peak_gain_dbfs == -12.0
    assert config.patch.peak_gain == pytest.approx(10 ** (-12.0 / 20.0))


def test_half_up_frame_rounding() -> None:
    assert seconds_to_frames(0.001, 48_000) == 48
    assert seconds_to_frames(0.0005, 1_000) == 1


def test_default_envelope_frame_counts() -> None:
    envelope = EnvelopeSpec()
    assert envelope.attack_frames(48_000) == 48
    assert envelope.decay_frames(48_000) == 28_800
    assert envelope.release_frames(48_000) == 28_800
    assert envelope.sustain_amplitude == pytest.approx(0.5011872336272722)


@pytest.mark.parametrize("field", ["attack_seconds", "decay_seconds", "release_seconds"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_envelope_rejects_invalid_durations(field: str, value: float) -> None:
    values = {
        "attack_seconds": 0.001,
        "decay_seconds": 0.600,
        "sustain_db": -6.0,
        "release_seconds": 0.600,
    }
    values[field] = value
    with pytest.raises(ValueError):
        EnvelopeSpec(**values)


@pytest.mark.parametrize("sustain_db", [0.1, math.nan, math.inf])
def test_envelope_rejects_invalid_sustain(sustain_db: float) -> None:
    with pytest.raises(ValueError, match="sustain_db"):
        EnvelopeSpec(sustain_db=sustain_db)


def test_keyboard_requires_ordered_range() -> None:
    with pytest.raises(ValueError, match=r"minimum.*initial.*maximum"):
        KeyboardViewSpec(
            minimum_note=MidiNote(61),
            initial_note=MidiNote(60),
            maximum_note=MidiNote(72),
        )


@pytest.mark.parametrize(
    "values",
    [
        {"sample_rate_hz": 0},
        {"block_frames": 0},
        {"channels": 2},
        {"internal_dtype": "float64"},
    ],
)
def test_render_spec_rejects_incompatible_settings(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RenderSpec(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("peak_gain_dbfs", [0.1, math.nan, math.inf])
def test_patch_rejects_invalid_peak_gain(peak_gain_dbfs: float) -> None:
    with pytest.raises(ValueError, match="peak_gain_dbfs"):
        SinePatch(peak_gain_dbfs=peak_gain_dbfs)


def test_application_rejects_sub_frame_envelope_stage() -> None:
    patch = SinePatch(envelope=EnvelopeSpec(attack_seconds=0.00001))
    with pytest.raises(ValueError, match=r"attack.*one frame"):
        AppConfig(patch=patch)


def test_application_rejects_maximum_note_at_or_above_nyquist() -> None:
    tuning = EqualTemperament(reference_hz=48_000.0)

    with pytest.raises(
        ValueError,
        match=r"maximum_note \(MIDI 72\).*below Nyquist \(24000 Hz\)",
    ):
        AppConfig(tuning=tuning)


def test_application_rejects_minimum_note_without_finite_positive_frequency() -> None:
    tuning = EqualTemperament(reference_hz=5e-324)

    with pytest.raises(
        ValueError,
        match=r"minimum_note \(MIDI 48\).*finite, positive frequency",
    ):
        AppConfig(tuning=tuning)
