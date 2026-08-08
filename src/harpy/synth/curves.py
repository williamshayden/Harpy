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
    scale = max(abs(start), abs(end), 1.0)
    scaled_start = start / scale
    scaled_end = end / scale
    start_weight = (1.0 - curve) / 2.0
    end_weight = (1.0 + curve) / 2.0
    scaled_control = start_weight * scaled_start + end_weight * scaled_end
    low, high = sorted((scaled_start, scaled_end))
    return min(max(scaled_control, low), high) * scale


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
    scale = max(abs(start), abs(end), abs(control), 1.0)
    scaled_start = start / scale
    scaled_end = end / scale
    scaled_control = control / scale
    endpoint_fraction = (scaled_control - scaled_start) / (scaled_end - scaled_start)
    return min(max(2.0 * endpoint_fraction - 1.0, -1.0), 1.0)


def _evaluate_scaled_quadratic(
    start: float,
    end: float,
    control: float,
    positions: np.float64 | np.ndarray,
) -> np.float64 | np.ndarray:
    scale = max(abs(start), abs(end), abs(control), 1.0)
    scaled_start = start / scale
    scaled_end = end / scale
    scaled_control = control / scale
    scaled_values = (
        (1.0 - positions) ** 2 * scaled_start
        + 2.0 * (1.0 - positions) * positions * scaled_control
        + positions**2 * scaled_end
    )
    low, high = sorted((scaled_start, scaled_end))
    return np.clip(scaled_values, low, high) * scale


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
        values = _evaluate_scaled_quadratic(start, end, control_level, positions)
        values[positions == 0.0] = start
        values[positions == 1.0] = end
        result = np.array(values, dtype=np.float64, copy=True, order="C")
        result.flags.writeable = False
        return result

    scalar_position = _finite_scalar("position", position)
    if not 0.0 <= scalar_position <= 1.0:
        raise ValueError("position must be between 0.0 and 1.0")
    positions = np.float64(scalar_position)
    values = _evaluate_scaled_quadratic(start, end, control_level, positions)
    if positions == 0.0:
        values = np.float64(start)
    elif positions == 1.0:
        values = np.float64(end)
    return float(values)
