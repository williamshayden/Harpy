import math
from dataclasses import FrozenInstanceError, replace

import pytest

from harpy.synth.models import (
    EnvelopeConfig,
    OscillatorConfig,
    OscillatorType,
    RenderConfig,
    SynthPatch,
    seconds_to_frames,
    validate_renderable_patch,
)


def test_default_patch_is_the_approved_curve_enabled_sine_patch() -> None:
    patch = SynthPatch()

    assert patch.oscillator == OscillatorConfig(type=OscillatorType.SINE)
    assert patch.envelope == EnvelopeConfig(
        attack_seconds=0.001,
        decay_seconds=0.600,
        sustain_db=-6.0,
        release_seconds=0.600,
        attack_curve=0.0,
        decay_curve=0.0,
        release_curve=0.0,
    )
    assert not hasattr(patch.envelope, "curve")
    assert patch.output_gain_dbfs == -12.0


def test_default_render_config_is_authoritative_mono_float32() -> None:
    assert RenderConfig() == RenderConfig(
        sample_rate_hz=48_000,
        block_frames=256,
        channels=1,
        internal_dtype="float32",
    )


@pytest.mark.parametrize("value", [0.0, -0.1, True, False, float("nan"), float("inf")])
def test_envelope_durations_must_be_positive_finite_numbers(value: object) -> None:
    with pytest.raises(ValueError):
        EnvelopeConfig(attack_seconds=value)


def test_composed_patch_requires_at_least_one_frame_per_segment() -> None:
    patch = replace(SynthPatch(), envelope=EnvelopeConfig(attack_seconds=0.000001))

    with pytest.raises(ValueError, match="attack"):
        validate_renderable_patch(patch, RenderConfig())


def test_seconds_to_frames_uses_half_up_rounding() -> None:
    assert seconds_to_frames(2.5 / 48_000, 48_000) == 3


def test_finite_duration_too_large_for_a_frame_count_is_a_named_value_error() -> None:
    patch = replace(SynthPatch(), envelope=EnvelopeConfig(attack_seconds=1e308))

    with pytest.raises(ValueError, match="attack_seconds"):
        validate_renderable_patch(patch, RenderConfig())


def test_configs_are_immutable() -> None:
    patch = SynthPatch()

    with pytest.raises(FrozenInstanceError):
        patch.output_gain_dbfs = -3.0  # type: ignore[misc]


@pytest.mark.parametrize("value", ["sine", "square", 1])
def test_oscillator_type_rejects_values_outside_the_closed_enum(value: object) -> None:
    with pytest.raises(ValueError, match="type"):
        OscillatorConfig(type=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), 0.1])
def test_sustain_db_must_be_finite_and_no_greater_than_zero(value: object) -> None:
    with pytest.raises(ValueError, match="sustain_db"):
        EnvelopeConfig(sustain_db=value)


@pytest.mark.parametrize("field", ["attack_curve", "decay_curve", "release_curve"])
@pytest.mark.parametrize(
    "value",
    [-1.001, 1.001, True, False, math.nan, math.inf, -math.inf, "0"],
)
def test_envelope_curves_must_be_finite_numbers_in_closed_range(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        EnvelopeConfig(**{field: value})  # type: ignore[arg-type]


def test_negative_zero_patch_numbers_are_stored_as_positive_zero() -> None:
    envelope = EnvelopeConfig(
        sustain_db=-0.0,
        attack_curve=-0.0,
        decay_curve=-0.0,
        release_curve=-0.0,
    )
    patch = SynthPatch(envelope=envelope, output_gain_dbfs=-0.0)

    for value in (
        envelope.sustain_db,
        envelope.attack_curve,
        envelope.decay_curve,
        envelope.release_curve,
        patch.output_gain_dbfs,
    ):
        assert math.copysign(1.0, value) == 1.0


def test_amplitude_properties_convert_decibels_to_linear_amplitude() -> None:
    assert EnvelopeConfig(sustain_db=-6.0).sustain_amplitude == pytest.approx(0.5011872336)
    assert SynthPatch(output_gain_dbfs=-12.0).output_gain == pytest.approx(0.2511886432)


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), 0.1])
def test_output_gain_dbfs_must_be_finite_and_no_greater_than_zero(value: object) -> None:
    with pytest.raises(ValueError, match="output_gain_dbfs"):
        SynthPatch(output_gain_dbfs=value)


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"sample_rate_hz": 0}, "sample_rate_hz"),
        ({"sample_rate_hz": True}, "sample_rate_hz"),
        ({"block_frames": 0}, "block_frames"),
        ({"block_frames": False}, "block_frames"),
        ({"channels": 2}, "channels"),
        ({"channels": True}, "channels"),
        ({"internal_dtype": "float64"}, "internal_dtype"),
    ],
)
def test_render_config_rejects_values_outside_the_authoritative_format(
    kwargs: dict[str, object], field: str
) -> None:
    with pytest.raises(ValueError, match=field):
        RenderConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("segment", ["attack", "decay", "release"])
def test_renderability_validation_names_each_segment_that_rounds_to_zero(segment: str) -> None:
    envelope = replace(EnvelopeConfig(), **{f"{segment}_seconds": 0.000001})
    patch = replace(SynthPatch(), envelope=envelope)

    with pytest.raises(ValueError, match=segment):
        validate_renderable_patch(patch, RenderConfig())
