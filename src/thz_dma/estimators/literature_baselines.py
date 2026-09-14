"""Literature-aligned two-path baselines for Stage 1-C.

The estimators in this module are explicit adaptations to the project's joint
frequency observation model.  They are not claimed to reproduce the numerical
results of the source papers:

* ``ogols_two_path_estimate`` specializes Yang et al.'s off-grid distributed
  OLS idea to one joint frequency observation vector and an exact spherical
  DMA template operator.
* ``screened_sbl_two_path_estimate`` applies the classical SBL/EM parent
  algorithm used by Gao et al. to the same joint-frequency dictionary.  A
  deterministic matched-filter screen keeps the comparison computationally
  bounded and is reported by the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from thz_dma.channels.atmosphere import C0_M_PER_S
from thz_dma.estimators.los_grid import LosTemplateDictionary
from thz_dma.estimators.twopath_omp import _least_squares_pair


@dataclass(frozen=True)
class SphericalDmaTemplateOperator:
    """Evaluate exact spherical-wave templates for one known DMA schedule."""

    frequencies_hz: np.ndarray
    element_positions_m: np.ndarray
    weights: np.ndarray
    absorption_coefficient_per_m: np.ndarray

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.frequencies_hz, dtype=float)
        positions = np.asarray(self.element_positions_m, dtype=float)
        weights = np.asarray(self.weights, dtype=complex)
        absorption = np.asarray(
            self.absorption_coefficient_per_m, dtype=float
        )
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("frequencies_hz must be a non-empty vector")
        if positions.ndim != 1 or positions.size == 0:
            raise ValueError("element_positions_m must be a non-empty vector")
        if weights.shape != (frequencies.size, positions.size):
            raise ValueError("weights must have shape (frequency, element)")
        if absorption.shape != frequencies.shape:
            raise ValueError(
                "absorption coefficients must match the frequency vector"
            )
        if np.any(frequencies <= 0.0) or np.any(absorption < 0.0):
            raise ValueError("frequencies must be positive and absorption non-negative")

    @property
    def num_pilots(self) -> int:
        return int(np.asarray(self.frequencies_hz).size)

    def evaluate(self, ranges_m, angles_rad) -> np.ndarray:
        """Return one combined pilot template per paired range-angle point."""

        ranges = np.atleast_1d(np.asarray(ranges_m, dtype=float))
        angles = np.atleast_1d(np.asarray(angles_rad, dtype=float))
        if ranges.ndim != 1 or angles.shape != ranges.shape:
            raise ValueError("ranges and angles must be matching vectors")
        if np.any(~np.isfinite(ranges)) or np.any(ranges <= 0.0):
            raise ValueError("ranges must be finite and positive")
        if np.any(~np.isfinite(angles)):
            raise ValueError("angles must be finite")

        frequencies = np.asarray(self.frequencies_hz, dtype=float)
        positions = np.asarray(self.element_positions_m, dtype=float)
        weights = np.asarray(self.weights, dtype=complex)
        absorption = np.asarray(
            self.absorption_coefficient_per_m, dtype=float
        )
        distances = np.sqrt(
            ranges[:, None] ** 2
            + positions[None, :] ** 2
            - 2.0
            * ranges[:, None]
            * positions[None, :]
            * np.sin(angles[:, None])
        )
        amplitude = C0_M_PER_S / (
            4.0
            * np.pi
            * frequencies[None, :, None]
            * distances[:, None, :]
        )
        amplitude *= np.exp(
            -0.5 * absorption[None, :, None] * distances[:, None, :]
        )
        channel = amplitude * np.exp(
            -1j
            * 2.0
            * np.pi
            * frequencies[None, :, None]
            * distances[:, None, :]
            / C0_M_PER_S
        )
        return np.einsum(
            "kn,ckn->ck", np.conjugate(weights), channel, optimize=True
        )


def _validate_observations(
    dictionary: LosTemplateDictionary,
    observations,
) -> tuple[np.ndarray, bool]:
    received = np.asarray(observations, dtype=complex)
    scalar_input = received.ndim == 1
    received = np.atleast_2d(received)
    if received.ndim != 2 or received.shape[1] != dictionary.num_pilots:
        raise ValueError("observations must have one column per pilot")
    return received, scalar_input


def _pack_result(result: dict[str, np.ndarray], scalar_input: bool):
    if scalar_input:
        return {key: value[0] for key, value in result.items()}
    return result


def ols_two_path_estimate(
    dictionary: LosTemplateDictionary,
    observations,
    *,
    exclusion_range_m: float = 0.0,
    exclusion_angle_rad: float = 0.0,
) -> dict[str, np.ndarray]:
    """Two-step OLS with joint complex-gain refitting.

    Unlike OMP, the second atom maximizes the energy reduction after the
    candidate is orthogonalized against the selected support.
    """

    received, scalar_input = _validate_observations(dictionary, observations)
    if exclusion_range_m < 0.0 or exclusion_angle_rad < 0.0:
        raise ValueError("exclusion widths must be non-negative")

    templates = np.asarray(dictionary.templates, dtype=complex)
    energy = np.sum(np.abs(templates) ** 2, axis=1)
    if np.any(energy <= 0.0):
        raise ValueError("dictionary contains a zero-energy template")
    correlation = received @ np.conjugate(templates).T
    first_score = np.abs(correlation) ** 2 / energy[None, :]
    first_index = np.argmax(first_score, axis=1)

    second_index = np.empty(received.shape[0], dtype=int)
    gains = np.empty((received.shape[0], 2), dtype=complex)
    residual_energy = np.empty(received.shape[0], dtype=float)
    pair_condition_number = np.empty(received.shape[0], dtype=float)
    for sample_index, selected in enumerate(first_index):
        first = templates[int(selected)]
        projection = (
            templates @ np.conjugate(first) / energy[int(selected)]
        )
        orthogonal = templates - projection[:, None] * first[None, :]
        orthogonal_energy = np.sum(np.abs(orthogonal) ** 2, axis=1)
        residual = received[sample_index] - (
            correlation[sample_index, int(selected)]
            / energy[int(selected)]
        ) * first
        score = np.full(dictionary.num_candidates, -np.inf, dtype=float)
        usable = orthogonal_energy > np.finfo(float).eps * np.max(energy)
        score[usable] = (
            np.abs(orthogonal[usable] @ np.conjugate(residual)) ** 2
            / orthogonal_energy[usable]
        )
        local_duplicate = (
            np.abs(dictionary.ranges_m - dictionary.ranges_m[int(selected)])
            <= exclusion_range_m
        ) & (
            np.abs(dictionary.angles_rad - dictionary.angles_rad[int(selected)])
            <= exclusion_angle_rad
        )
        local_duplicate[int(selected)] = True
        score[local_duplicate] = -np.inf
        candidate = int(np.argmax(score))
        if not np.isfinite(score[candidate]):
            raise RuntimeError("OLS could not find a distinct second atom")
        second_index[sample_index] = candidate
        fitted, residual_value, condition = _least_squares_pair(
            templates,
            received[sample_index],
            int(selected),
            candidate,
        )
        gains[sample_index] = fitted
        residual_energy[sample_index] = residual_value
        pair_condition_number[sample_index] = condition

    indices = np.column_stack((first_index, second_index))
    total_energy = np.sum(np.abs(received) ** 2, axis=1)
    result = {
        "candidate_index": indices,
        "range_m": dictionary.ranges_m[indices],
        "angle_rad": dictionary.angles_rad[indices],
        "complex_gain": gains,
        "explained_energy": np.maximum(0.0, total_energy - residual_energy),
        "residual_energy": residual_energy,
        "pair_condition_number": pair_condition_number,
        "iterations": np.ones(received.shape[0], dtype=int),
        "converged": np.ones(received.shape[0], dtype=int),
        "screened_candidate_count": np.full(
            received.shape[0], dictionary.num_candidates, dtype=int
        ),
    }
    return _pack_result(result, scalar_input)


def _fit_continuous_pair(
    template_evaluator: Callable[[np.ndarray, np.ndarray], np.ndarray],
    observation: np.ndarray,
    ranges_m: np.ndarray,
    angles_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    templates = np.asarray(
        template_evaluator(ranges_m, angles_rad), dtype=complex
    )
    if templates.shape != (2, observation.size):
        raise ValueError("template evaluator must return shape (2, num_pilots)")
    matrix = templates.T
    gains, _, _, singular_values = np.linalg.lstsq(matrix, observation, rcond=None)
    residual = observation - matrix @ gains
    energy = float(np.vdot(residual, residual).real)
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > 0.0
        else float("inf")
    )
    return templates, gains, energy, condition


def _refine_one_pair(
    observation: np.ndarray,
    initial_ranges_m: np.ndarray,
    initial_angles_rad: np.ndarray,
    template_evaluator: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    range_bounds_m: tuple[float, float],
    angle_bounds_rad: tuple[float, float],
    range_grid_step_m: float,
    angle_grid_step_rad: float,
    refinement_iterations: int,
    derivative_fraction: float,
    maximum_update_grid_units: float,
    convergence_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int, int]:
    ranges = np.asarray(initial_ranges_m, dtype=float).copy()
    angles = np.asarray(initial_angles_rad, dtype=float).copy()
    if ranges.shape != (2,) or angles.shape != (2,):
        raise ValueError("off-grid refinement requires exactly two paths")
    if refinement_iterations < 1:
        raise ValueError("refinement_iterations must be positive")
    if not 0.0 < derivative_fraction <= 0.5:
        raise ValueError("derivative_fraction must lie in (0, 0.5]")
    if maximum_update_grid_units <= 0.0:
        raise ValueError("maximum_update_grid_units must be positive")

    templates, gains, residual_energy, condition = _fit_continuous_pair(
        template_evaluator, observation, ranges, angles
    )
    initial_energy = residual_energy
    converged = 0
    completed = 0
    range_delta = derivative_fraction * range_grid_step_m
    angle_delta = derivative_fraction * angle_grid_step_rad

    for iteration in range(1, refinement_iterations + 1):
        residual = observation - templates.T @ gains
        derivative_columns = []
        for path_index in range(2):
            lower_range = max(range_bounds_m[0], ranges[path_index] - range_delta)
            upper_range = min(range_bounds_m[1], ranges[path_index] + range_delta)
            lower_angle = max(
                angle_bounds_rad[0], angles[path_index] - angle_delta
            )
            upper_angle = min(
                angle_bounds_rad[1], angles[path_index] + angle_delta
            )
            range_points = np.array([lower_range, upper_range])
            angle_fixed = np.full(2, angles[path_index])
            range_templates = template_evaluator(range_points, angle_fixed)
            range_derivative = (
                (range_templates[1] - range_templates[0])
                / max(upper_range - lower_range, np.finfo(float).eps)
                * range_grid_step_m
            )
            angle_points = np.array([lower_angle, upper_angle])
            range_fixed = np.full(2, ranges[path_index])
            angle_templates = template_evaluator(range_fixed, angle_points)
            angle_derivative = (
                (angle_templates[1] - angle_templates[0])
                / max(upper_angle - lower_angle, np.finfo(float).eps)
                * angle_grid_step_rad
            )
            derivative_columns.extend(
                [
                    gains[path_index] * range_derivative,
                    gains[path_index] * angle_derivative,
                ]
            )

        derivative = np.column_stack(derivative_columns)
        signal_matrix = templates.T
        derivative -= signal_matrix @ np.linalg.lstsq(
            signal_matrix, derivative, rcond=None
        )[0]
        real_system = np.vstack((derivative.real, derivative.imag))
        real_residual = np.concatenate((residual.real, residual.imag))
        update, _, _, _ = np.linalg.lstsq(
            real_system, real_residual, rcond=None
        )
        update = np.clip(
            update,
            -maximum_update_grid_units,
            maximum_update_grid_units,
        )
        if not np.all(np.isfinite(update)):
            break

        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.125):
            trial_ranges = np.clip(
                ranges + scale * update[[0, 2]] * range_grid_step_m,
                range_bounds_m[0],
                range_bounds_m[1],
            )
            trial_angles = np.clip(
                angles + scale * update[[1, 3]] * angle_grid_step_rad,
                angle_bounds_rad[0],
                angle_bounds_rad[1],
            )
            trial_templates, trial_gains, trial_energy, trial_condition = (
                _fit_continuous_pair(
                    template_evaluator,
                    observation,
                    trial_ranges,
                    trial_angles,
                )
            )
            if trial_energy <= residual_energy * (1.0 + 1.0e-12):
                improvement = residual_energy - trial_energy
                ranges = trial_ranges
                angles = trial_angles
                templates = trial_templates
                gains = trial_gains
                residual_energy = trial_energy
                condition = trial_condition
                completed = iteration
                accepted = True
                if improvement <= convergence_tolerance * max(
                    initial_energy, np.finfo(float).tiny
                ):
                    converged = 1
                break
        if not accepted or converged:
            if not accepted:
                converged = 1
            break

    return (
        ranges,
        angles,
        gains,
        residual_energy,
        condition,
        completed,
        converged,
    )


def ogols_two_path_estimate(
    dictionary: LosTemplateDictionary,
    observations,
    template_evaluator: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    range_bounds_m: tuple[float, float],
    angle_bounds_rad: tuple[float, float],
    range_grid_step_m: float,
    angle_grid_step_rad: float,
    exclusion_range_m: float = 0.0,
    exclusion_angle_rad: float = 0.0,
    refinement_iterations: int = 4,
    derivative_fraction: float = 0.1,
    maximum_update_grid_units: float = 0.5,
    convergence_tolerance: float = 1.0e-6,
) -> dict[str, np.ndarray]:
    """Yang-style OLS support selection followed by off-grid refinement."""

    received, scalar_input = _validate_observations(dictionary, observations)
    initial = ols_two_path_estimate(
        dictionary,
        received,
        exclusion_range_m=exclusion_range_m,
        exclusion_angle_rad=exclusion_angle_rad,
    )
    ranges = np.empty((received.shape[0], 2), dtype=float)
    angles = np.empty((received.shape[0], 2), dtype=float)
    gains = np.empty((received.shape[0], 2), dtype=complex)
    residual_energy = np.empty(received.shape[0], dtype=float)
    condition = np.empty(received.shape[0], dtype=float)
    iterations = np.empty(received.shape[0], dtype=int)
    converged = np.empty(received.shape[0], dtype=int)
    for sample_index in range(received.shape[0]):
        refined = _refine_one_pair(
            received[sample_index],
            np.asarray(initial["range_m"][sample_index]),
            np.asarray(initial["angle_rad"][sample_index]),
            template_evaluator,
            range_bounds_m=range_bounds_m,
            angle_bounds_rad=angle_bounds_rad,
            range_grid_step_m=range_grid_step_m,
            angle_grid_step_rad=angle_grid_step_rad,
            refinement_iterations=refinement_iterations,
            derivative_fraction=derivative_fraction,
            maximum_update_grid_units=maximum_update_grid_units,
            convergence_tolerance=convergence_tolerance,
        )
        (
            ranges[sample_index],
            angles[sample_index],
            gains[sample_index],
            residual_energy[sample_index],
            condition[sample_index],
            iterations[sample_index],
            converged[sample_index],
        ) = refined

    total_energy = np.sum(np.abs(received) ** 2, axis=1)
    result = {
        "candidate_index": np.asarray(initial["candidate_index"], dtype=int),
        "range_m": ranges,
        "angle_rad": angles,
        "complex_gain": gains,
        "explained_energy": np.maximum(0.0, total_energy - residual_energy),
        "residual_energy": residual_energy,
        "pair_condition_number": condition,
        "iterations": iterations,
        "converged": converged,
        "screened_candidate_count": np.full(
            received.shape[0], dictionary.num_candidates, dtype=int
        ),
    }
    return _pack_result(result, scalar_input)


def _select_relevance_pair(
    dictionary: LosTemplateDictionary,
    candidate_indices: np.ndarray,
    relevance: np.ndarray,
    exclusion_range_m: float,
    exclusion_angle_rad: float,
) -> np.ndarray:
    order = np.argsort(relevance)[::-1]
    first = int(candidate_indices[int(order[0])])
    second = None
    for local_index in order[1:]:
        candidate = int(candidate_indices[int(local_index)])
        local_duplicate = (
            abs(dictionary.ranges_m[candidate] - dictionary.ranges_m[first])
            <= exclusion_range_m
            and abs(dictionary.angles_rad[candidate] - dictionary.angles_rad[first])
            <= exclusion_angle_rad
        )
        if not local_duplicate:
            second = candidate
            break
    if second is None:
        for candidate in candidate_indices:
            if int(candidate) != first:
                second = int(candidate)
                break
    if second is None:
        raise RuntimeError("SBL screen does not contain two distinct candidates")
    return np.array([first, second], dtype=int)


def _sbl_one(
    dictionary: LosTemplateDictionary,
    observation: np.ndarray,
    *,
    noise_variance: float | None,
    screening_size: int,
    max_iterations: int,
    convergence_tolerance: float,
    em_damping: float,
    noise_floor_fraction: float,
    exclusion_range_m: float,
    exclusion_angle_rad: float,
) -> tuple[np.ndarray, np.ndarray, float, float, int, int, int]:
    templates = np.asarray(dictionary.templates, dtype=complex)
    energy = np.sum(np.abs(templates) ** 2, axis=1)
    normalized_score = (
        np.abs(templates @ np.conjugate(observation)) ** 2 / energy
    )
    screen_count = min(max(2, int(screening_size)), dictionary.num_candidates)
    screen = np.argpartition(normalized_score, -screen_count)[-screen_count:]

    ols = ols_two_path_estimate(
        dictionary,
        observation,
        exclusion_range_m=exclusion_range_m,
        exclusion_angle_rad=exclusion_angle_rad,
    )
    required = np.asarray(ols["candidate_index"], dtype=int)
    screen = np.unique(np.concatenate((screen, required)))
    if screen.size > screen_count:
        required_unique = np.unique(required)
        ranked = screen[np.argsort(normalized_score[screen])[::-1]]
        optional = ranked[~np.isin(ranked, required_unique)]
        screen = np.concatenate(
            (
                required_unique,
                optional[: max(0, screen_count - required_unique.size)],
            )
        )
    if screen.size < 2:
        raise RuntimeError("SBL candidate screening retained fewer than two atoms")

    selected_templates = templates[screen]
    norms = np.linalg.norm(selected_templates, axis=1)
    sensing = (selected_templates / norms[:, None]).T
    signal_scale = max(
        float(np.sqrt(np.mean(np.abs(observation) ** 2))),
        np.finfo(float).tiny,
    )
    received = observation / signal_scale
    if noise_variance is None or not np.isfinite(noise_variance):
        scaled_noise = noise_floor_fraction
    else:
        scaled_noise = float(noise_variance) / signal_scale**2
        scaled_noise = max(scaled_noise, noise_floor_fraction)

    matched = sensing.conjugate().T @ received
    gamma = np.maximum(np.abs(matched) ** 2, 1.0e-6)
    gram = sensing.conjugate().T @ sensing
    rhs = sensing.conjugate().T @ received / scaled_noise
    converged = 0
    posterior_mean = np.zeros(screen.size, dtype=complex)
    posterior_variance = gamma.copy()
    completed = 0
    for iteration in range(1, max_iterations + 1):
        precision = gram / scaled_noise + np.diag(1.0 / gamma)
        try:
            posterior_covariance = np.linalg.inv(precision)
        except np.linalg.LinAlgError:
            posterior_covariance = np.linalg.pinv(precision, hermitian=True)
        posterior_mean = posterior_covariance @ rhs
        posterior_variance = np.maximum(
            np.real(np.diag(posterior_covariance)), 0.0
        )
        proposed = np.maximum(
            np.abs(posterior_mean) ** 2 + posterior_variance,
            1.0e-12,
        )
        updated = em_damping * gamma + (1.0 - em_damping) * proposed
        change = float(
            np.max(
                np.abs(np.log(updated) - np.log(gamma))
            )
        )
        gamma = updated
        completed = iteration
        if change <= convergence_tolerance:
            converged = 1
            break

    relevance = np.abs(posterior_mean) ** 2 + posterior_variance
    pair = _select_relevance_pair(
        dictionary,
        screen,
        relevance,
        exclusion_range_m,
        exclusion_angle_rad,
    )
    fitted, residual_energy, condition = _least_squares_pair(
        templates, observation, int(pair[0]), int(pair[1])
    )
    return (
        pair,
        fitted,
        residual_energy,
        condition,
        completed,
        converged,
        int(screen.size),
    )


def screened_sbl_two_path_estimate(
    dictionary: LosTemplateDictionary,
    observations,
    *,
    noise_variance: float | np.ndarray | None,
    screening_size: int = 64,
    max_iterations: int = 50,
    convergence_tolerance: float = 1.0e-4,
    em_damping: float = 0.5,
    noise_floor_fraction: float = 1.0e-8,
    exclusion_range_m: float = 0.0,
    exclusion_angle_rad: float = 0.0,
) -> dict[str, np.ndarray]:
    """Screened single-vector SBL/EM on the joint-frequency dictionary."""

    received, scalar_input = _validate_observations(dictionary, observations)
    if screening_size < 2 or max_iterations < 1:
        raise ValueError("screening_size and max_iterations are too small")
    if not 0.0 <= em_damping < 1.0:
        raise ValueError("em_damping must lie in [0, 1)")
    if convergence_tolerance <= 0.0 or noise_floor_fraction <= 0.0:
        raise ValueError("SBL tolerances must be positive")
    if noise_variance is None or np.isscalar(noise_variance):
        noise_values = [noise_variance] * received.shape[0]
    else:
        noise_array = np.asarray(noise_variance, dtype=float)
        if noise_array.shape != (received.shape[0],):
            raise ValueError("noise_variance must be scalar or one value per sample")
        noise_values = noise_array.tolist()

    indices = np.empty((received.shape[0], 2), dtype=int)
    gains = np.empty((received.shape[0], 2), dtype=complex)
    residual_energy = np.empty(received.shape[0], dtype=float)
    condition = np.empty(received.shape[0], dtype=float)
    iterations = np.empty(received.shape[0], dtype=int)
    converged = np.empty(received.shape[0], dtype=int)
    screened_count = np.empty(received.shape[0], dtype=int)
    for sample_index in range(received.shape[0]):
        result = _sbl_one(
            dictionary,
            received[sample_index],
            noise_variance=noise_values[sample_index],
            screening_size=screening_size,
            max_iterations=max_iterations,
            convergence_tolerance=convergence_tolerance,
            em_damping=em_damping,
            noise_floor_fraction=noise_floor_fraction,
            exclusion_range_m=exclusion_range_m,
            exclusion_angle_rad=exclusion_angle_rad,
        )
        (
            indices[sample_index],
            gains[sample_index],
            residual_energy[sample_index],
            condition[sample_index],
            iterations[sample_index],
            converged[sample_index],
            screened_count[sample_index],
        ) = result

    total_energy = np.sum(np.abs(received) ** 2, axis=1)
    result = {
        "candidate_index": indices,
        "range_m": dictionary.ranges_m[indices],
        "angle_rad": dictionary.angles_rad[indices],
        "complex_gain": gains,
        "explained_energy": np.maximum(0.0, total_energy - residual_energy),
        "residual_energy": residual_energy,
        "pair_condition_number": condition,
        "iterations": iterations,
        "converged": converged,
        "screened_candidate_count": screened_count,
    }
    return _pack_result(result, scalar_input)
