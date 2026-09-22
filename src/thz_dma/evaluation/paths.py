"""Permutation-invariant path matching for estimator evaluation."""

from __future__ import annotations

import itertools

import numpy as np

from thz_dma.channels.planar import PlanarPath


def _match_coordinates(
    estimated_coordinates,
    estimated_gains,
    true_coordinates,
    scales,
) -> dict:
    estimates = np.asarray(estimated_coordinates, dtype=float)
    gains = np.asarray(estimated_gains, dtype=complex)
    truth = np.asarray(true_coordinates, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if estimates.shape != truth.shape or gains.shape != (truth.shape[0],):
        raise ValueError("estimator path count does not match the truth")
    if truth.ndim != 2 or scales.shape != (truth.shape[1],) or np.any(scales <= 0):
        raise ValueError("path coordinates or matching scales are invalid")

    best = None
    best_cost = float("inf")
    for permutation in itertools.permutations(range(truth.shape[0])):
        order = np.asarray(permutation, dtype=int)
        cost = float(np.sum(((estimates[order] - truth) / scales) ** 2))
        if cost < best_cost:
            best_cost = cost
            best = order
    if best is None:
        raise RuntimeError("path matching failed")
    matched = estimates[best]
    return {
        "coordinates": matched,
        "gains": gains[best],
        "errors": matched - truth,
        "matching_cost": best_cost,
    }


def match_two_paths(
    estimated_ranges: np.ndarray,
    estimated_angles_deg: np.ndarray,
    estimated_gains: np.ndarray,
    true_ranges: np.ndarray,
    true_angles_deg: np.ndarray,
    *,
    range_scale_m: float,
    angle_scale_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match two one-dimensional spherical paths and preserve the old tuple API."""

    result = _match_coordinates(
        np.column_stack((estimated_ranges, estimated_angles_deg)),
        estimated_gains,
        np.column_stack((true_ranges, true_angles_deg)),
        np.array([range_scale_m, angle_scale_deg]),
    )
    return result["coordinates"][:, 0], result["coordinates"][:, 1], result["gains"]


def match_planar_paths(
    estimates: dict,
    paths: tuple[PlanarPath, ...],
    config: dict,
) -> dict:
    """Match estimated range/azimuth/elevation tuples to planar path truth."""

    evaluation = config["evaluation"]
    truth = np.array(
        [[path.range_m, path.azimuth_rad, path.elevation_rad] for path in paths],
        dtype=float,
    )
    result = _match_coordinates(
        np.column_stack(
            (
                np.asarray(estimates["range_m"], dtype=float),
                np.asarray(estimates["azimuth_rad"], dtype=float),
                np.asarray(estimates["elevation_rad"], dtype=float),
            )
        ),
        estimates["complex_gain"],
        truth,
        np.array(
            [
                float(evaluation["path_match_range_scale_m"]),
                float(np.deg2rad(evaluation["path_match_azimuth_scale_deg"])),
                float(np.deg2rad(evaluation["path_match_elevation_scale_deg"])),
            ]
        ),
    )
    errors = result.pop("errors")
    return {
        **result,
        "range_error_m": errors[:, 0],
        "azimuth_error_deg": np.rad2deg(errors[:, 1]),
        "elevation_error_deg": np.rad2deg(errors[:, 2]),
    }
