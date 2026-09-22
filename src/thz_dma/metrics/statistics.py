"""Small statistical helpers shared by experiment and analysis code."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def power_db(value: float) -> float:
    """Convert a non-negative linear power ratio to decibels."""

    return float(10.0 * np.log10(max(float(value), np.finfo(float).tiny)))


def rmse(values) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def mean(values) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def median(values) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def bootstrap_interval(
    values,
    statistic: Callable[[np.ndarray], float],
    *,
    rng: np.random.Generator,
    resamples: int,
) -> list[float]:
    """Return a percentile 95% interval after dropping non-finite values."""

    data = np.asarray(values, dtype=float)
    if data.ndim != 1:
        raise ValueError("bootstrap input must be a vector")
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("bootstrap input has no finite observations")
    if data.size == 1:
        value = float(statistic(data))
        return [value, value]
    if resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100")
    indices = rng.integers(0, data.size, size=(resamples, data.size))
    estimates = np.asarray(
        [statistic(data[index]) for index in indices], dtype=float
    )
    return np.quantile(estimates, [0.025, 0.975]).astype(float).tolist()
