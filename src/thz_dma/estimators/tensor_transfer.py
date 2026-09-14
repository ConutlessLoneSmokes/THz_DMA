"""Bounded tensor-structure transfer baselines for Stage 1-D.

The source paper uses a four-way CP model created by microstrip-sequential
training at a transmitting DMA.  The present project has one RF output at a
receiving one-dimensional DMA, so that model cannot be reproduced verbatim.
This module retains only the transferable low-rank frequency mechanism:

* one configuration: rank-P block-Hankel SVD denoising;
* multiple configurations: rank-P CP-ALS denoising of a
  configuration-by-Hankel-frequency tensor.

The denoised observation initializes the existing exact spherical OG-OLS
estimator.  Final path gains and residuals are always refitted against the
original observation, not the denoised surrogate.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from thz_dma.estimators.literature_baselines import ogols_two_path_estimate


def _columnwise_kronecker(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return the Khatri-Rao product with right index varying fastest."""

    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("Khatri-Rao inputs must be matrices")
    if left.shape[1] != right.shape[1]:
        raise ValueError("Khatri-Rao inputs must have the same column count")
    return np.einsum("ir,jr->ijr", left, right).reshape(
        left.shape[0] * right.shape[0], left.shape[1]
    )


def _least_squares_factor(
    unfolding: np.ndarray,
    product: np.ndarray,
    ridge: float,
) -> np.ndarray:
    """Solve ``unfolding ~= factor @ product.T`` for a complex factor."""

    if ridge < 0.0:
        raise ValueError("CP ridge must be non-negative")
    design = product
    target = unfolding.T
    if ridge > 0.0:
        rank = product.shape[1]
        design = np.vstack(
            (product, np.sqrt(ridge) * np.eye(rank, dtype=complex))
        )
        target = np.vstack(
            (target, np.zeros((rank, unfolding.shape[0]), dtype=complex))
        )
    return np.linalg.lstsq(design, target, rcond=None)[0].T


def _cp_reconstruct(factors: tuple[np.ndarray, ...]) -> np.ndarray:
    if len(factors) != 3:
        raise ValueError("Stage 1-D CP reconstruction expects three factors")
    return np.einsum("ir,jr,kr->ijk", *factors)


def _balance_cp_columns(
    factors: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    balanced = [np.asarray(factor, dtype=complex).copy() for factor in factors]
    tiny = np.finfo(float).tiny
    for component in range(balanced[0].shape[1]):
        norms = [
            max(float(np.linalg.norm(factor[:, component])), tiny)
            for factor in balanced
        ]
        shared = float(np.prod(norms) ** (1.0 / 3.0))
        for factor, norm in zip(balanced, norms):
            factor[:, component] *= shared / norm
    return balanced[0], balanced[1], balanced[2]


def complex_cp_als(
    tensor,
    rank: int,
    *,
    max_iterations: int = 40,
    convergence_tolerance: float = 1.0e-6,
    ridge: float = 1.0e-10,
) -> dict:
    """Deterministic complex CP-ALS for a small third-order tensor."""

    values = np.asarray(tensor, dtype=complex)
    if values.ndim != 3 or any(size < 1 for size in values.shape):
        raise ValueError("CP input must be a non-empty third-order tensor")
    if rank < 1 or rank > min(values.shape):
        raise ValueError("CP rank must not exceed the smallest tensor mode")
    if max_iterations < 1:
        raise ValueError("CP max_iterations must be positive")
    if convergence_tolerance <= 0.0:
        raise ValueError("CP convergence_tolerance must be positive")

    mode0 = values.reshape(values.shape[0], -1)
    mode1 = values.transpose(1, 0, 2).reshape(values.shape[1], -1)
    mode2 = values.transpose(2, 0, 1).reshape(values.shape[2], -1)
    factor0 = np.linalg.svd(mode0, full_matrices=False)[0][:, :rank]
    factor1 = np.linalg.svd(mode1, full_matrices=False)[0][:, :rank]
    factor2 = np.linalg.svd(mode2, full_matrices=False)[0][:, :rank]
    factor0, factor1, factor2 = _balance_cp_columns(
        (factor0, factor1, factor2)
    )

    energy = max(float(np.sum(np.abs(values) ** 2)), np.finfo(float).tiny)
    previous = np.inf
    converged = 0
    completed = 0
    reconstruction = _cp_reconstruct((factor0, factor1, factor2))
    residual_fraction = float(
        np.sum(np.abs(values - reconstruction) ** 2) / energy
    )
    for iteration in range(1, max_iterations + 1):
        factor0 = _least_squares_factor(
            mode0, _columnwise_kronecker(factor1, factor2), ridge
        )
        factor1 = _least_squares_factor(
            mode1, _columnwise_kronecker(factor0, factor2), ridge
        )
        factor2 = _least_squares_factor(
            mode2, _columnwise_kronecker(factor0, factor1), ridge
        )
        factor0, factor1, factor2 = _balance_cp_columns(
            (factor0, factor1, factor2)
        )
        reconstruction = _cp_reconstruct((factor0, factor1, factor2))
        residual_fraction = float(
            np.sum(np.abs(values - reconstruction) ** 2) / energy
        )
        if not np.isfinite(residual_fraction):
            raise FloatingPointError("CP-ALS produced a non-finite residual")
        completed = iteration
        if residual_fraction <= convergence_tolerance:
            converged = 1
            break
        if np.isfinite(previous):
            relative_change = abs(previous - residual_fraction) / max(
                previous, np.finfo(float).tiny
            )
            if relative_change <= convergence_tolerance:
                converged = 1
                break
        previous = residual_fraction

    return {
        "reconstruction": reconstruction,
        "factors": (factor0, factor1, factor2),
        "residual_fraction": residual_fraction,
        "iterations": completed,
        "converged": converged,
    }


def _hankelize(vector: np.ndarray, rows: int) -> np.ndarray:
    values = np.asarray(vector, dtype=complex)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("Hankel input must contain at least three samples")
    if rows < 2 or rows >= values.size:
        raise ValueError("Hankel rows must lie in [2, sample_count - 1]")
    columns = values.size - rows + 1
    return np.asarray(
        [[values[row + column] for column in range(columns)] for row in range(rows)],
        dtype=complex,
    )


def _dehankelize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=complex)
    if values.ndim != 2:
        raise ValueError("de-Hankelization expects a matrix")
    output = np.zeros(values.shape[0] + values.shape[1] - 1, dtype=complex)
    counts = np.zeros(output.size, dtype=float)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            output[row + column] += values[row, column]
            counts[row + column] += 1.0
    return output / counts


def _configuration_frequency_matrix(
    observation: np.ndarray,
    frequencies_hz: np.ndarray,
    configuration_index: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], bool]:
    values = np.asarray(observation, dtype=complex)
    frequencies = np.asarray(frequencies_hz, dtype=float)
    configurations = np.asarray(configuration_index, dtype=int)
    if values.ndim != 1:
        raise ValueError("tensor transfer expects one observation vector")
    if frequencies.shape != values.shape or configurations.shape != values.shape:
        raise ValueError("frequency and configuration schedules must match observation")
    labels = np.unique(configurations)
    if not np.array_equal(labels, np.arange(labels.size)):
        raise ValueError("configuration indices must be contiguous from zero")
    groups = []
    frequency_rows = []
    for label in labels:
        indices = np.flatnonzero(configurations == label)
        order = np.argsort(frequencies[indices])
        groups.append(indices[order])
        frequency_rows.append(frequencies[indices][order])
    counts = {indices.size for indices in groups}
    if len(counts) != 1:
        raise ValueError("every tensor configuration must use the same pilot count")
    matrix = np.stack([values[indices] for indices in groups], axis=0)
    aligned = all(
        np.allclose(frequency_rows[0], row, rtol=0.0, atol=1.0e-6)
        for row in frequency_rows[1:]
    )
    return matrix, groups, bool(aligned)


def tensor_low_rank_denoise(
    observation,
    frequencies_hz,
    configuration_index,
    *,
    rank: int = 2,
    hankel_rows: int = 0,
    cp_max_iterations: int = 40,
    cp_convergence_tolerance: float = 1.0e-6,
    cp_ridge: float = 1.0e-10,
) -> dict:
    """Denoise one scheduled observation using its folded frequency structure."""

    values = np.asarray(observation, dtype=complex)
    matrix, groups, aligned = _configuration_frequency_matrix(
        values,
        np.asarray(frequencies_hz, dtype=float),
        np.asarray(configuration_index, dtype=int),
    )
    samples_per_configuration = matrix.shape[1]
    rows = int(hankel_rows) if int(hankel_rows) > 0 else samples_per_configuration // 2
    rows = max(2, min(rows, samples_per_configuration - 1))
    hankel = np.stack([_hankelize(row, rows) for row in matrix], axis=0)
    effective_rank = min(int(rank), rows, hankel.shape[2])
    if effective_rank < 1:
        raise ValueError("tensor rank must be positive")

    if matrix.shape[0] == 1:
        left, singular, right_h = np.linalg.svd(hankel[0], full_matrices=False)
        reconstructed_hankel = (
            left[:, :effective_rank]
            * singular[:effective_rank][None, :]
        ) @ right_h[:effective_rank]
        tensor_reconstruction = reconstructed_hankel[None, :, :]
        tensor_iterations = 1
        tensor_converged = 1
        factorization_order = 2
    else:
        effective_rank = min(effective_rank, matrix.shape[0])
        cp = complex_cp_als(
            hankel,
            effective_rank,
            max_iterations=int(cp_max_iterations),
            convergence_tolerance=float(cp_convergence_tolerance),
            ridge=float(cp_ridge),
        )
        tensor_reconstruction = np.asarray(cp["reconstruction"], dtype=complex)
        tensor_iterations = int(cp["iterations"])
        tensor_converged = int(cp["converged"])
        factorization_order = 3

    denoised_matrix = np.stack(
        [_dehankelize(item) for item in tensor_reconstruction], axis=0
    )
    denoised = np.empty_like(values)
    for row, indices in enumerate(groups):
        denoised[indices] = denoised_matrix[row]
    tensor_energy = max(
        float(np.sum(np.abs(hankel) ** 2)), np.finfo(float).tiny
    )
    observation_energy = max(
        float(np.sum(np.abs(values) ** 2)), np.finfo(float).tiny
    )
    return {
        "denoised_observation": denoised,
        "rank_residual_fraction": float(
            np.sum(np.abs(hankel - tensor_reconstruction) ** 2)
            / tensor_energy
        ),
        "denoise_change_fraction": float(
            np.sum(np.abs(denoised - values) ** 2) / observation_energy
        ),
        "factorization_order": factorization_order,
        "tensor_shape": tuple(int(value) for value in hankel.shape),
        "iterations": tensor_iterations,
        "converged": tensor_converged,
        "frequency_grid_aligned": int(aligned),
    }


def tensor_denoised_ogols_two_path_estimate(
    dictionary,
    observation: np.ndarray,
    template_evaluator: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    frequencies_hz: np.ndarray,
    configuration_index: np.ndarray,
    range_bounds_m: tuple[float, float],
    angle_bounds_rad: tuple[float, float],
    range_grid_step_m: float,
    angle_grid_step_rad: float,
    exclusion_range_m: float,
    exclusion_angle_rad: float,
    refinement_iterations: int,
    derivative_fraction: float,
    maximum_update_grid_units: float,
    convergence_tolerance: float,
    tensor_rank: int = 2,
    hankel_rows: int = 0,
    cp_max_iterations: int = 40,
    cp_convergence_tolerance: float = 1.0e-6,
    cp_ridge: float = 1.0e-10,
) -> dict:
    """Apply tensor denoising, OG-OLS, and an original-data gain refit."""

    values = np.asarray(observation, dtype=complex)
    denoising = tensor_low_rank_denoise(
        values,
        frequencies_hz,
        configuration_index,
        rank=int(tensor_rank),
        hankel_rows=int(hankel_rows),
        cp_max_iterations=int(cp_max_iterations),
        cp_convergence_tolerance=float(cp_convergence_tolerance),
        cp_ridge=float(cp_ridge),
    )
    preliminary = ogols_two_path_estimate(
        dictionary,
        np.asarray(denoising["denoised_observation"], dtype=complex),
        template_evaluator,
        range_bounds_m=range_bounds_m,
        angle_bounds_rad=angle_bounds_rad,
        range_grid_step_m=range_grid_step_m,
        angle_grid_step_rad=angle_grid_step_rad,
        exclusion_range_m=exclusion_range_m,
        exclusion_angle_rad=exclusion_angle_rad,
        refinement_iterations=refinement_iterations,
        derivative_fraction=derivative_fraction,
        maximum_update_grid_units=maximum_update_grid_units,
        convergence_tolerance=convergence_tolerance,
    )
    ranges = np.asarray(preliminary["range_m"], dtype=float)
    angles = np.asarray(preliminary["angle_rad"], dtype=float)
    templates = np.asarray(template_evaluator(ranges, angles), dtype=complex)
    sensing = templates.T
    gains = np.linalg.lstsq(sensing, values, rcond=None)[0]
    residual = values - sensing @ gains
    residual_energy = float(np.sum(np.abs(residual) ** 2))
    total_energy = float(np.sum(np.abs(values) ** 2))
    tensor_shape = denoising["tensor_shape"]
    return {
        **preliminary,
        "complex_gain": gains,
        "residual_energy": residual_energy,
        "explained_energy": max(total_energy - residual_energy, 0.0),
        "pair_condition_number": float(np.linalg.cond(sensing)),
        "iterations": int(denoising["iterations"]),
        "converged": int(denoising["converged"]),
        "offgrid_iterations": int(preliminary.get("iterations", 1)),
        "offgrid_converged": int(preliminary.get("converged", 1)),
        "tensor_rank_residual_fraction": float(
            denoising["rank_residual_fraction"]
        ),
        "tensor_denoise_change_fraction": float(
            denoising["denoise_change_fraction"]
        ),
        "tensor_factorization_order": int(
            denoising["factorization_order"]
        ),
        "tensor_shape_0": int(tensor_shape[0]),
        "tensor_shape_1": int(tensor_shape[1]),
        "tensor_shape_2": int(tensor_shape[2]),
        "tensor_frequency_grid_aligned": int(
            denoising["frequency_grid_aligned"]
        ),
    }
