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


def tensor_unfolding_rank_residuals(tensor, rank: int) -> dict[str, float]:
    """SVD tail fractions of every unfolding of an ``S x J x K`` tensor."""

    values = np.asarray(tensor, dtype=complex)
    if values.ndim != 3 or values.size == 0:
        raise ValueError("tensor must have three non-empty dimensions")
    if rank < 1:
        raise ValueError("rank must be positive")
    residuals: dict[str, float] = {}
    for mode in range(3):
        unfolding = np.moveaxis(values, mode, 0).reshape(values.shape[mode], -1)
        singular_values = np.linalg.svd(unfolding, compute_uv=False)
        energy = float(np.sum(singular_values**2))
        tail = float(np.sum(singular_values[min(rank, singular_values.size) :] ** 2))
        residuals[f"mode_{mode}_rank_residual_fraction"] = (
            0.0 if energy == 0.0 else float(np.sqrt(tail / energy))
        )
    residuals["max_mode_rank_residual_fraction"] = max(residuals.values())
    return residuals


def stage2_condition_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["physics_level"], row["schedule"]), []).append(row)
    result = []
    for (level, schedule), subset in grouped.items():
        result.append(
            {
                "physics_level": level,
                "schedule": schedule,
                "num_independent_scenes": len(subset),
                "mean_effective_output_snr_db": float(
                    np.mean([item["effective_output_snr_db"] for item in subset])
                ),
                "mean_max_mode_rank_residual_fraction": float(
                    np.mean(
                        [item["max_mode_rank_residual_fraction"] for item in subset]
                    )
                ),
                "max_max_mode_rank_residual_fraction": float(
                    np.max(
                        [item["max_mode_rank_residual_fraction"] for item in subset]
                    )
                ),
                "mean_combiner_row_energy": float(
                    np.mean([item["combiner_row_energy_mean"] for item in subset])
                ),
            }
        )
    return sorted(result, key=lambda item: (item["physics_level"], item["schedule"]))


def channel_domain_metrics(truth, estimate, pilot_indices, legacy_indices) -> dict:
    """Element-space CSI NMSE, summed over each frequency domain and all elements.

    Pilot frequencies are counted once, regardless of configuration reuse.
    Empty held-out domains are absent measurements (None), never zero error.
    """
    truth, estimate = np.asarray(truth), np.asarray(estimate)
    if truth.shape != estimate.shape or truth.ndim < 2 or truth.shape[0] != 128:
        raise ValueError("Stage 2-A domain evaluation requires matching 128-tone CSI")
    pilots = np.unique(np.asarray(pilot_indices, dtype=int))
    legacy = np.asarray(legacy_indices, dtype=int)
    if pilots.size == 0 or any(np.any((x < 0) | (x >= 128)) for x in (pilots, legacy)):
        raise ValueError("invalid CSI evaluation indices")
    domains = {
        "channel_nmse": legacy,
        "pilot_nmse": pilots,
        "fullband_128_nmse": np.arange(128),
        "heldout_112_nmse": np.setdiff1d(np.arange(128), pilots),
    }
    if domains["heldout_112_nmse"].size not in (0, 112):
        raise ValueError("Stage 2-A held-out domain must contain 0 or 112 tones")
    result = {}
    for name, indices in domains.items():
        result[f"{name}_subcarrier_count"] = int(indices.size)
        value = None
        if indices.size:
            energy = float(np.sum(np.abs(truth[indices]) ** 2))
            if energy <= 0 or not np.isfinite(energy):
                raise ValueError("CSI truth energy must be positive and finite")
            value = float(np.sum(np.abs(estimate[indices] - truth[indices]) ** 2) / energy)
            if not np.isfinite(value):
                raise ValueError("nonfinite CSI NMSE")
        result[f"{name}_linear"] = value
        result[f"{name}_db"] = None if value is None else float(
            10 * np.log10(max(value, np.finfo(float).tiny))
        )
    return result
