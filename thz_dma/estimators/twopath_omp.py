"""Two-path orthogonal matching pursuit on a known LoS template grid."""

from __future__ import annotations

import numpy as np

from thz_dma.estimators.los_grid import LosTemplateDictionary


def _least_squares_pair(
    templates: np.ndarray,
    observation: np.ndarray,
    first_index: int,
    second_index: int,
) -> tuple[np.ndarray, float, float]:
    matrix = np.column_stack(
        (templates[first_index], templates[second_index])
    )
    gains, _, _, singular_values = np.linalg.lstsq(
        matrix, observation, rcond=None
    )
    residual = observation - matrix @ gains
    residual_energy = float(np.vdot(residual, residual).real)
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else float("inf")
    )
    return gains, residual_energy, condition_number


def omp_two_path_estimate(
    dictionary: LosTemplateDictionary,
    observations,
    *,
    exclusion_range_m: float = 0.0,
    exclusion_angle_rad: float = 0.0,
) -> dict[str, np.ndarray]:
    """Estimate two grid paths and their complex gains with two-step OMP.

    The first atom maximizes the single-path profile likelihood.  The second
    atom is selected from the residual, after suppressing only the local
    range-angle neighborhood of the first atom.  Both gains are then refit
    jointly by complex least squares.  Path order is intentionally left
    unordered; evaluation must match estimates to the two true paths.
    """

    received = np.asarray(observations, dtype=complex)
    scalar_input = received.ndim == 1
    received = np.atleast_2d(received)
    if received.ndim != 2 or received.shape[1] != dictionary.num_pilots:
        raise ValueError("observations must have one column per pilot")
    if exclusion_range_m < 0.0 or exclusion_angle_rad < 0.0:
        raise ValueError("exclusion widths must be non-negative")

    templates = np.asarray(dictionary.templates, dtype=complex)
    energy = np.sum(np.abs(templates) ** 2, axis=1)
    if np.any(energy <= 0.0):
        raise ValueError("dictionary contains a zero-energy template")

    correlation = received @ np.conjugate(templates).T
    first_score = np.abs(correlation) ** 2 / energy[None, :]
    first_index = np.argmax(first_score, axis=1)
    row = np.arange(received.shape[0])
    first_gain = correlation[row, first_index] / energy[first_index]
    residual = received - first_gain[:, None] * templates[first_index]

    residual_correlation = residual @ np.conjugate(templates).T
    second_score = np.abs(residual_correlation) ** 2 / energy[None, :]
    for sample_index, selected in enumerate(first_index):
        local_duplicate = (
            np.abs(dictionary.ranges_m - dictionary.ranges_m[selected])
            <= exclusion_range_m
        ) & (
            np.abs(dictionary.angles_rad - dictionary.angles_rad[selected])
            <= exclusion_angle_rad
        )
        local_duplicate[selected] = True
        second_score[sample_index, local_duplicate] = -np.inf

    second_index = np.argmax(second_score, axis=1)
    gains = np.empty((received.shape[0], 2), dtype=complex)
    residual_energy = np.empty(received.shape[0], dtype=float)
    pair_condition_number = np.empty(received.shape[0], dtype=float)
    for sample_index in range(received.shape[0]):
        fitted, residual_value, condition = _least_squares_pair(
            templates,
            received[sample_index],
            int(first_index[sample_index]),
            int(second_index[sample_index]),
        )
        gains[sample_index] = fitted
        residual_energy[sample_index] = residual_value
        pair_condition_number[sample_index] = condition

    indices = np.column_stack((first_index, second_index))
    total_energy = np.sum(np.abs(received) ** 2, axis=1)
    explained_energy = np.maximum(0.0, total_energy - residual_energy)
    result = {
        "candidate_index": indices,
        "range_m": dictionary.ranges_m[indices],
        "angle_rad": dictionary.angles_rad[indices],
        "complex_gain": gains,
        "explained_energy": explained_energy,
        "residual_energy": residual_energy,
        "pair_condition_number": pair_condition_number,
    }
    if scalar_input:
        return {key: value[0] for key, value in result.items()}
    return result

