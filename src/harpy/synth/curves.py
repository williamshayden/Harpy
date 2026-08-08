from __future__ import annotations

import math

import numpy as np


def _finite_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def quadratic_control_level(
    start_level: float,
    end_level: float,
    curvature: float,
) -> float:
    start = _finite_scalar("start_level", start_level)
    end = _finite_scalar("end_level", end_level)
    curve = _finite_scalar("curvature", curvature)
    if not -1.0 <= curve <= 1.0:
        raise ValueError("curvature must be between -1.0 and 1.0")
    return (start + end) / 2.0 + curve * (end - start) / 2.0


def curvature_from_control_level(
    start_level: float,
    end_level: float,
    control_level: float,
) -> float:
    start = _finite_scalar("start_level", start_level)
    end = _finite_scalar("end_level", end_level)
    control = _finite_scalar("control_level", control_level)
    if start == end:
        raise ValueError("start_level and end_level must differ")
    low, high = sorted((start, end))
    if not low <= control <= high:
        raise ValueError("control_level must be between start_level and end_level")
    return 2.0 * (control - (start + end) / 2.0) / (end - start)


def evaluate_quadratic_segment(
    start_level: float,
    end_level: float,
    curvature: float,
    position: float | np.ndarray,
) -> float | np.ndarray:
    start = _finite_scalar("start_level", start_level)
    end = _finite_scalar("end_level", end_level)
    curve = _finite_scalar("curvature", curvature)
    control_level = quadratic_control_level(start, end, curve)

    if isinstance(position, np.ndarray):
        if position.ndim != 1 or position.dtype != np.float64:
            raise ValueError("position must be a one-dimensional float64 array")
        positions = np.array(position, dtype=np.float64, copy=True, order="C")
        if not np.all(np.isfinite(positions)):
            raise ValueError("position must contain only finite values")
        if np.any((positions < 0.0) | (positions > 1.0)):
            raise ValueError("position must be between 0.0 and 1.0")
        values = (
            (1.0 - positions) ** 2 * start
            + 2.0 * (1.0 - positions) * positions * control_level
            + positions**2 * end
        )
        values[positions == 0.0] = start
        values[positions == 1.0] = end
        result = np.array(values, dtype=np.float64, copy=True, order="C")
        result.flags.writeable = False
        return result

    scalar_position = _finite_scalar("position", position)
    if not 0.0 <= scalar_position <= 1.0:
        raise ValueError("position must be between 0.0 and 1.0")
    positions = np.float64(scalar_position)
    values = (
        (1.0 - positions) ** 2 * start
        + 2.0 * (1.0 - positions) * positions * control_level
        + positions**2 * end
    )
    if positions == 0.0:
        values = np.float64(start)
    elif positions == 1.0:
        values = np.float64(end)
    return float(values)
