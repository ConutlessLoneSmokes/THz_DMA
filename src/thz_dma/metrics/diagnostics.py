"""Local identifiability and frequency-projection diagnostics."""

from __future__ import annotations

import numpy as np


def row_space_diagnostics(matrix, *, relative_tolerance: float = 1.0e-8) -> dict:
    """Diagnose how far frequency-indexed rows depart from proportionality."""

    values = np.asarray(matrix, dtype=complex)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("matrix must be two-dimensional and non-empty")
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("matrix contains a zero row")
    normalized = values / norm
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    energy = singular_values**2
    probability = energy / energy.sum()
    numerical_rank = int(np.sum(singular_values > relative_tolerance * singular_values[0]))
    rank_one_residual = float(max(0.0, 1.0 - probability[0]))
    participation_rank = float(1.0 / np.sum(probability**2))
    entropy_rank = float(np.exp(-np.sum(probability * np.log(np.maximum(probability, 1e-300)))))

    gram = np.abs(normalized @ np.conjugate(normalized.T))
    if normalized.shape[0] > 1:
        off_diagonal = gram[~np.eye(gram.shape[0], dtype=bool)]
        adjacent = np.diag(gram, k=1)
        max_coherence = float(np.max(off_diagonal))
        mean_coherence = float(np.mean(off_diagonal))
        mean_adjacent_coherence = float(np.mean(adjacent))
    else:
        max_coherence = mean_coherence = mean_adjacent_coherence = 0.0

    return {
        "numerical_rank": numerical_rank,
        "rank_one_residual_fraction": rank_one_residual,
        "participation_effective_rank": participation_rank,
        "entropy_effective_rank": entropy_rank,
        "max_frequency_row_coherence": max_coherence,
        "mean_frequency_row_coherence": mean_coherence,
        "mean_adjacent_row_coherence": mean_adjacent_coherence,
        "singular_values": singular_values.tolist(),
    }


def finite_difference_real_jacobian(function, parameters, steps) -> np.ndarray:
    """Central-difference Jacobian of stacked real and imaginary observations."""

    point = np.asarray(parameters, dtype=float)
    delta = np.asarray(steps, dtype=float)
    if point.ndim != 1 or delta.shape != point.shape or np.any(delta <= 0.0):
        raise ValueError("parameters and positive steps must be matching vectors")
    columns = []
    for index, step in enumerate(delta):
        plus = point.copy()
        minus = point.copy()
        plus[index] += step
        minus[index] -= step
        derivative = (
            np.asarray(function(plus), dtype=complex)
            - np.asarray(function(minus), dtype=complex)
        ) / (2.0 * step)
        columns.append(np.concatenate([derivative.real.ravel(), derivative.imag.ravel()]))
    return np.column_stack(columns)


def jacobian_diagnostics(jacobian, *, relative_tolerance: float = 1.0e-8) -> dict:
    """Scale-invariant local identifiability diagnostics."""

    values = np.asarray(jacobian, dtype=float)
    if values.ndim != 2:
        raise ValueError("jacobian must be two-dimensional")
    column_norms = np.linalg.norm(values, axis=0)
    if np.any(column_norms == 0.0):
        raise ValueError("jacobian contains an all-zero parameter derivative")
    normalized = values / column_norms[None, :]
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    numerical_rank = int(np.sum(singular_values > relative_tolerance * singular_values[0]))
    condition = (
        float(np.inf)
        if singular_values[-1] <= relative_tolerance * singular_values[0]
        else float(singular_values[0] / singular_values[-1])
    )
    gram_eigenvalues = np.linalg.eigvalsh(normalized.T @ normalized)
    return {
        "numerical_rank": numerical_rank,
        "condition_number": condition,
        "minimum_singular_value": float(singular_values[-1]),
        "maximum_singular_value": float(singular_values[0]),
        "normalized_gram_minimum_eigenvalue": float(gram_eigenvalues[0]),
        "column_norms": column_norms.tolist(),
        "singular_values": singular_values.tolist(),
    }

