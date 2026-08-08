import math
from dataclasses import replace

import numpy as np
import pytest

from harpy.synth.curves import (
    EnvelopePreview,
    curvature_from_control_level,
    evaluate_quadratic_segment,
    quadratic_control_level,
    sample_envelope_preview,
)
from harpy.synth.models import EnvelopeConfig, RenderConfig


@pytest.mark.parametrize("curvature", [-1.0, -0.25, 0.0, 0.25, 1.0])
@pytest.mark.parametrize(("start", "end"), [(0.0, 1.0), (1.0, 0.25), (0.25, 0.0)])
def test_quadratic_segment_has_exact_endpoints(
    curvature: float,
    start: float,
    end: float,
) -> None:
    assert evaluate_quadratic_segment(start, end, curvature, 0.0) == start
    assert evaluate_quadratic_segment(start, end, curvature, 1.0) == end


def test_zero_curvature_is_linear_and_sign_reaches_target_early() -> None:
    assert evaluate_quadratic_segment(0.0, 1.0, 0.0, 0.5) == 0.5
    assert evaluate_quadratic_segment(0.0, 1.0, 1.0, 0.5) == 0.75
    assert evaluate_quadratic_segment(0.0, 1.0, -1.0, 0.5) == 0.25
    assert evaluate_quadratic_segment(1.0, 0.0, 1.0, 0.5) == 0.25
    assert evaluate_quadratic_segment(1.0, 0.0, -1.0, 0.5) == 0.75


@pytest.mark.parametrize(("start", "end"), [(0.0, 1.0), (1.0, 0.25), (0.25, 0.0)])
@pytest.mark.parametrize("curvature", np.linspace(-1.0, 1.0, 201))
def test_control_level_and_inverse_round_trip(
    start: float,
    end: float,
    curvature: float,
) -> None:
    control = quadratic_control_level(start, end, float(curvature))
    recovered = curvature_from_control_level(start, end, control)
    assert recovered == pytest.approx(float(curvature), rel=0.0, abs=2e-15)


@pytest.mark.parametrize(
    ("start", "end", "curvature", "expected_control"),
    [
        (1.7976931348623157e308, 1.7976931348623157e308, 0.5, 1.7976931348623157e308),
        (-1e308, 1e308, 0.5, 5e307),
        (1e308, -1e308, 0.5, -5e307),
    ],
    ids=["constant", "rising", "falling"],
)
def test_large_finite_control_levels_do_not_overflow(
    start: float,
    end: float,
    curvature: float,
    expected_control: float,
) -> None:
    assert quadratic_control_level(start, end, curvature) == expected_control


@pytest.mark.parametrize(
    ("start", "end", "control", "expected_curvature"),
    [
        (-1e308, 1e308, 5e307, 0.5),
        (1e308, -1e308, -5e307, 0.5),
    ],
    ids=["rising", "falling"],
)
def test_large_finite_control_levels_have_a_finite_inverse(
    start: float,
    end: float,
    control: float,
    expected_curvature: float,
) -> None:
    assert curvature_from_control_level(start, end, control) == expected_curvature


@pytest.mark.parametrize(
    ("start", "end", "curvature", "position", "expected_level"),
    [
        (
            1.7976931348623157e308,
            1.7976931348623157e308,
            0.5,
            0.1,
            1.7976931348623157e308,
        ),
        (-1e308, 1e308, 0.5, 0.5, 2.5e307),
        (1e308, -1e308, 0.5, 0.5, -2.5e307),
    ],
    ids=["constant", "rising", "falling"],
)
def test_large_finite_scalar_segments_do_not_overflow(
    start: float,
    end: float,
    curvature: float,
    position: float,
    expected_level: float,
) -> None:
    assert evaluate_quadratic_segment(start, end, curvature, position) == expected_level


def test_large_finite_array_segment_does_not_overflow() -> None:
    level = 1.7976931348623157e308
    positions = np.array([0.0, 0.1, 0.5, 1.0], dtype=np.float64)

    values = evaluate_quadratic_segment(level, level, 0.5, positions)

    np.testing.assert_array_equal(values, np.full(positions.shape, level))


def test_inverse_rejects_zero_endpoint_span() -> None:
    with pytest.raises(ValueError, match=r"start_level.*end_level"):
        curvature_from_control_level(0.5, 0.5, 0.5)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("start_level", True),
        ("start_level", float("inf")),
        ("end_level", False),
        ("end_level", float("nan")),
        ("curvature", False),
        ("curvature", float("inf")),
        ("curvature", -1.001),
        ("curvature", 1.001),
    ],
)
def test_control_level_rejects_invalid_inputs(argument: str, value: object) -> None:
    arguments: dict[str, object] = {
        "start_level": 0.0,
        "end_level": 1.0,
        "curvature": 0.0,
    }
    arguments[argument] = value
    with pytest.raises(ValueError, match=argument):
        quadratic_control_level(**arguments)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("start_level", True),
        ("start_level", float("inf")),
        ("end_level", False),
        ("end_level", float("nan")),
        ("control_level", True),
        ("control_level", float("-inf")),
        ("control_level", -0.001),
        ("control_level", 1.001),
    ],
)
def test_inverse_rejects_invalid_inputs(argument: str, value: object) -> None:
    arguments: dict[str, object] = {
        "start_level": 0.0,
        "end_level": 1.0,
        "control_level": 0.5,
    }
    arguments[argument] = value
    with pytest.raises(ValueError, match=argument):
        curvature_from_control_level(**arguments)


@pytest.mark.parametrize(("start", "end"), [(0.0, 1.0), (1.0, 0.25), (0.25, 0.0)])
@pytest.mark.parametrize("curvature", np.linspace(-1.0, 1.0, 201))
def test_dense_curve_is_monotone_and_bounded(
    start: float,
    end: float,
    curvature: float,
) -> None:
    positions = np.linspace(0.0, 1.0, 1_001)
    values = evaluate_quadratic_segment(start, end, float(curvature), positions)
    assert isinstance(values, np.ndarray)
    assert values.shape == positions.shape
    assert values.dtype == np.float64
    assert values.flags.owndata
    assert values.flags.c_contiguous
    assert not values.flags.writeable
    assert values[0] == start
    assert values[-1] == end
    low, high = sorted((start, end))
    assert np.all(values >= low)
    assert np.all(values <= high)
    differences = np.diff(values)
    if end >= start:
        assert np.all(differences >= 0.0)
    else:
        assert np.all(differences <= 0.0)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("start_level", True),
        ("end_level", float("nan")),
        ("curvature", -1.001),
        ("curvature", 1.001),
        ("curvature", False),
        ("position", -0.001),
        ("position", 1.001),
        ("position", np.array([True, False])),
        ("position", np.array([0.0, float("inf")])),
        ("position", np.array([[0.0, 1.0]])),
    ],
)
def test_quadratic_segment_rejects_invalid_inputs(argument: str, value: object) -> None:
    arguments: dict[str, object] = {
        "start_level": 0.0,
        "end_level": 1.0,
        "curvature": 0.0,
        "position": 0.5,
    }
    arguments[argument] = value
    with pytest.raises(ValueError, match=argument):
        evaluate_quadratic_segment(**arguments)


def test_curve_output_is_independent_of_supplied_position_array() -> None:
    positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    values = evaluate_quadratic_segment(0.0, 1.0, 0.5, positions)
    expected = np.array([0.0, 0.34375, 0.625, 0.84375, 1.0], dtype=np.float64)

    positions[:] = 0.0

    np.testing.assert_array_equal(values, expected)


def test_nominal_preview_uses_the_shared_curve_at_renderer_positions() -> None:
    render = RenderConfig(sample_rate_hz=4, block_frames=2)
    envelope = EnvelopeConfig(
        attack_seconds=1.0,
        decay_seconds=1.0,
        sustain_db=20.0 * math.log10(0.5),
        release_seconds=1.0,
        attack_curve=-0.5,
        decay_curve=0.25,
        release_curve=1.0,
    )
    preview = sample_envelope_preview(envelope, render, samples_per_stage=5)
    assert isinstance(preview, EnvelopePreview)
    np.testing.assert_array_equal(preview.position, np.linspace(0.0, 1.0, 5))
    np.testing.assert_array_equal(
        preview.attack_level,
        evaluate_quadratic_segment(0.0, 1.0, -0.5, preview.position),
    )
    np.testing.assert_array_equal(
        preview.decay_level,
        evaluate_quadratic_segment(1.0, 0.5, 0.25, preview.position),
    )
    np.testing.assert_array_equal(
        preview.release_level,
        evaluate_quadratic_segment(0.5, 0.0, 1.0, preview.position),
    )
    assert preview.sustain_level == 0.5
    assert (preview.attack_frames, preview.decay_frames, preview.release_frames) == (4, 4, 4)


def test_preview_arrays_are_owned_read_only_contiguous_float64_vectors() -> None:
    preview = sample_envelope_preview(EnvelopeConfig(), RenderConfig())

    for values in (
        preview.position,
        preview.attack_level,
        preview.decay_level,
        preview.release_level,
    ):
        assert values.ndim == 1
        assert values.dtype == np.float64
        assert values.flags.owndata
        assert values.flags.c_contiguous
        assert not values.flags.writeable


@pytest.mark.parametrize("stage", ["attack", "decay", "release"])
@pytest.mark.parametrize("seconds", [1e-12, 1e308])
def test_preview_rejects_unrenderable_envelope(stage: str, seconds: float) -> None:
    envelope = replace(EnvelopeConfig(), **{f"{stage}_seconds": seconds})

    with pytest.raises(ValueError, match=stage):
        sample_envelope_preview(envelope, RenderConfig())


@pytest.mark.parametrize("samples_per_stage", [True, False, 1, 1.5, "129"])
def test_preview_requires_at_least_two_integer_samples(samples_per_stage: object) -> None:
    with pytest.raises(ValueError, match="samples_per_stage"):
        sample_envelope_preview(
            EnvelopeConfig(),
            RenderConfig(),
            samples_per_stage=samples_per_stage,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("envelope", "render", "field"),
    [(object(), RenderConfig(), "envelope"), (EnvelopeConfig(), object(), "render")],
)
def test_preview_requires_runtime_config_types(
    envelope: object,
    render: object,
    field: str,
) -> None:
    with pytest.raises(ValueError, match=field):
        sample_envelope_preview(envelope, render)  # type: ignore[arg-type]
