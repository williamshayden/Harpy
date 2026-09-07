import numpy as np
import pytest

from harpy.synth.curves import (
    evaluate_quadratic_segment,
    quadratic_control_level,
)


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
