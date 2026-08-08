import numpy as np
import pytest

from harpy.synth.curves import (
    curvature_from_control_level,
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
