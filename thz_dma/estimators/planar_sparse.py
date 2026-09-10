"""Three-dimensional sparse baselines for the Stage 2 planar DMA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from thz_dma.channels.atmosphere import C0_M_PER_S
from thz_dma.channels.planar import direction_unit_vector


@dataclass(frozen=True)
class PlanarDmaTemplateOperator:
    """Exact spherical path templates for one multi-strip acquisition schedule."""

    frequencies_hz: np.ndarray
    element_positions_m: np.ndarray
    weights: np.ndarray
    absorption_coefficient_per_m: np.ndarray
    relative_output_noise_variance: np.ndarray

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.frequencies_hz, dtype=float)
        positions = np.asarray(self.element_positions_m, dtype=float)
        weights = np.asarray(self.weights, dtype=complex)
        absorption = np.asarray(self.absorption_coefficient_per_m, dtype=float)
        relative_noise = np.asarray(self.relative_output_noise_variance, dtype=float)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("frequencies_hz must be a non-empty vector")
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("element_positions_m must have shape (S, E, 3)")
        if weights.ndim != 4:
            raise ValueError("weights must have shape (J, K, S, E)")
        expected_weights = (
            weights.shape[0],
            frequencies.size,
            positions.shape[0],
            positions.shape[1],
        )
        if weights.shape != expected_weights:
            raise ValueError("weights must have shape (J, K, S, E)")
        if absorption.shape != frequencies.shape or np.any(absorption < 0.0):
            raise ValueError("absorption coefficients must match frequencies")
        if relative_noise.shape != (
            positions.shape[0],
            weights.shape[0],
            frequencies.size,
        ):
            raise ValueError("relative noise must have shape (S, J, K)")
        if np.any(relative_noise <= 0.0) or not np.all(np.isfinite(relative_noise)):
            raise ValueError("relative output-noise variance must be positive and finite")

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        weights = np.asarray(self.weights)
        return int(weights.shape[2]), int(weights.shape[0]), int(weights.shape[1])

    @property
    def observation_size(self) -> int:
        return int(np.prod(self.observation_shape))

    def evaluate_raw(
        self,
        ranges_m,
        azimuths_rad,
        elevations_rad,
    ) -> np.ndarray:
        """Return unwhitened templates with one row per geometry candidate."""

        ranges = np.atleast_1d(np.asarray(ranges_m, dtype=float))
        azimuths = np.atleast_1d(np.asarray(azimuths_rad, dtype=float))
        elevations = np.atleast_1d(np.asarray(elevations_rad, dtype=float))
        if ranges.ndim != 1 or azimuths.shape != ranges.shape or elevations.shape != ranges.shape:
            raise ValueError("range, azimuth and elevation must be matching vectors")
        if np.any(ranges <= 0.0) or not np.all(
            np.isfinite(np.column_stack((ranges, azimuths, elevations)))
        ):
            raise ValueError("candidate coordinates must be finite with positive range")

        directions = np.vstack(
            [direction_unit_vector(azimuth, elevation) for azimuth, elevation in zip(azimuths, elevations)]
        )
        points = ranges[:, None] * directions
        positions = np.asarray(self.element_positions_m, dtype=float)
        distances = np.linalg.norm(
            points[:, None, None, :] - positions[None, :, :, :], axis=-1
        )
        frequencies = np.asarray(self.frequencies_hz, dtype=float)
        absorption = np.asarray(self.absorption_coefficient_per_m, dtype=float)
        amplitude = C0_M_PER_S / (
            4.0
            * np.pi
            * frequencies[None, :, None, None]
            * distances[:, None, :, :]
        )
        amplitude *= np.exp(
            -0.5 * absorption[None, :, None, None] * distances[:, None, :, :]
        )
        channel = amplitude * np.exp(
            -1j
            * 2.0
            * np.pi
            * frequencies[None, :, None, None]
            * distances[:, None, :, :]
            / C0_M_PER_S
        )
        combined = np.einsum(
            "jkse,ckse->csjk",
            np.conjugate(np.asarray(self.weights, dtype=complex)),
            channel,
            optimize=True,
        )
        return combined.reshape(ranges.size, -1)

    def evaluate(self, ranges_m, azimuths_rad, elevations_rad) -> np.ndarray:
        """Return templates whitened by the known relative noise covariance."""

        raw = self.evaluate_raw(ranges_m, azimuths_rad, elevations_rad)
        scale = np.sqrt(
            np.asarray(self.relative_output_noise_variance, dtype=float)
        ).reshape(-1)
        return raw / scale[None, :]

    def whiten(self, observation) -> np.ndarray:
        """Flatten one ``S x J x K`` observation and whiten relative variances."""

        values = np.asarray(observation, dtype=complex)
        if values.shape != self.observation_shape:
            raise ValueError("observation does not match the operator schedule")
        scale = np.sqrt(
            np.asarray(self.relative_output_noise_variance, dtype=float)
        )
        return (values / scale).reshape(-1)


@dataclass(frozen=True)
class PlanarTemplateDictionary:
    """Whitened templates over a three-dimensional range-angle grid."""

    ranges_m: np.ndarray
    azimuths_rad: np.ndarray
    elevations_rad: np.ndarray
    templates: np.ndarray

    def __post_init__(self) -> None:
        ranges = np.asarray(self.ranges_m)
        azimuths = np.asarray(self.azimuths_rad)
        elevations = np.asarray(self.elevations_rad)
        templates = np.asarray(self.templates)
        if ranges.ndim != 1 or azimuths.shape != ranges.shape or elevations.shape != ranges.shape:
            raise ValueError("candidate coordinate vectors must match")
        if templates.ndim != 2 or templates.shape[0] != ranges.size:
            raise ValueError("templates must have one row per candidate")
        if ranges.size == 0 or templates.shape[1] == 0:
            raise ValueError("dictionary cannot be empty")

    @property
    def num_candidates(self) -> int:
        return int(np.asarray(self.ranges_m).size)

    @property
    def observation_size(self) -> int:
        return int(np.asarray(self.templates).shape[1])


def _candidate_vectors(
    range_grid_m, azimuth_grid_rad, elevation_grid_rad
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranges = np.asarray(range_grid_m, dtype=float)
    azimuths = np.asarray(azimuth_grid_rad, dtype=float)
    elevations = np.asarray(elevation_grid_rad, dtype=float)
    if any(item.ndim != 1 or item.size == 0 for item in (ranges, azimuths, elevations)):
        raise ValueError("all search grids must be non-empty vectors")
    mesh = np.meshgrid(ranges, azimuths, elevations, indexing="ij")
    return tuple(item.ravel() for item in mesh)  # type: ignore[return-value]


def build_planar_template_dictionary(
    operator: PlanarDmaTemplateOperator,
    range_grid_m,
    azimuth_grid_rad,
    elevation_grid_rad,
    *,
    chunk_size: int = 8,
    storage_dtype: str = "complex64",
    progress: Callable[[str], None] | None = None,
) -> PlanarTemplateDictionary:
    """Build a bounded three-dimensional dictionary in candidate chunks."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if storage_dtype not in {"complex64", "complex128"}:
        raise ValueError("storage_dtype must be complex64 or complex128")
    notify = progress or (lambda _message: None)
    ranges, azimuths, elevations = _candidate_vectors(
        range_grid_m, azimuth_grid_rad, elevation_grid_rad
    )
    templates = np.empty(
        (ranges.size, operator.observation_size), dtype=np.dtype(storage_dtype)
    )
    report_stride = max(1.0, ranges.size / 4.0)
    next_report = report_stride
    for start in range(0, ranges.size, chunk_size):
        stop = min(start + chunk_size, ranges.size)
        templates[start:stop] = operator.evaluate(
            ranges[start:stop], azimuths[start:stop], elevations[start:stop]
        ).astype(storage_dtype)
        if stop >= next_report or stop == ranges.size:
            notify(f"built {stop}/{ranges.size} planar templates")
            next_report += report_stride
    if not np.all(np.isfinite(templates)):
        raise ValueError("dictionary contains non-finite values")
    return PlanarTemplateDictionary(ranges, azimuths, elevations, templates)


def _exclusion_mask(
    dictionary: PlanarTemplateDictionary,
    selected: list[int],
    exclusion_range_m: float,
    exclusion_azimuth_rad: float,
    exclusion_elevation_rad: float,
) -> np.ndarray:
    mask = np.zeros(dictionary.num_candidates, dtype=bool)
    for index in selected:
        mask |= (
            (np.abs(dictionary.ranges_m - dictionary.ranges_m[index]) <= exclusion_range_m)
            & (
                np.abs(dictionary.azimuths_rad - dictionary.azimuths_rad[index])
                <= exclusion_azimuth_rad
            )
            & (
                np.abs(dictionary.elevations_rad - dictionary.elevations_rad[index])
                <= exclusion_elevation_rad
            )
        )
        mask[index] = True
    return mask


def _fit_dictionary_support(
    templates: np.ndarray, observation: np.ndarray, support: list[int]
) -> tuple[np.ndarray, np.ndarray, float, float]:
    matrix = np.asarray(templates[support], dtype=complex).T
    gains, _, _, singular_values = np.linalg.lstsq(matrix, observation, rcond=None)
    residual = observation - matrix @ gains
    energy = float(np.vdot(residual, residual).real)
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else float("inf")
    )
    return gains, residual, energy, condition


def _pack_grid_result(
    dictionary: PlanarTemplateDictionary,
    observation: np.ndarray,
    support: list[int],
    gains: np.ndarray,
    residual_energy: float,
    condition: float,
) -> dict:
    indices = np.asarray(support, dtype=int)
    total_energy = float(np.vdot(observation, observation).real)
    return {
        "candidate_index": indices,
        "range_m": np.asarray(dictionary.ranges_m[indices], dtype=float),
        "azimuth_rad": np.asarray(dictionary.azimuths_rad[indices], dtype=float),
        "elevation_rad": np.asarray(dictionary.elevations_rad[indices], dtype=float),
        "complex_gain": np.asarray(gains, dtype=complex),
        "explained_energy": max(0.0, total_energy - residual_energy),
        "residual_energy": float(residual_energy),
        "support_condition_number": float(condition),
        "iterations": 1,
        "converged": 1,
        "screened_candidate_count": dictionary.num_candidates,
    }


def omp_three_path_estimate(
    dictionary: PlanarTemplateDictionary,
    observation,
    *,
    num_paths: int = 3,
    exclusion_range_m: float = 0.0,
    exclusion_azimuth_rad: float = 0.0,
    exclusion_elevation_rad: float = 0.0,
) -> dict:
    """Grid OMP with joint complex-gain refitting for a planar three-path scene."""

    received = np.asarray(observation, dtype=complex).reshape(-1)
    if received.size != dictionary.observation_size or num_paths < 1:
        raise ValueError("invalid observation size or path count")
    templates = np.asarray(dictionary.templates)
    energy = np.sum(np.abs(templates) ** 2, axis=1, dtype=float)
    if np.any(energy <= 0.0):
        raise ValueError("dictionary contains a zero-energy template")
    support: list[int] = []
    residual = received.copy()
    gains = np.empty(0, dtype=complex)
    residual_energy = float(np.vdot(residual, residual).real)
    condition = 1.0
    for _ in range(num_paths):
        # The conjugate of this product is the desired inner product; only its
        # magnitude is used, avoiding a full conjugated dictionary copy.
        correlation = templates @ np.conjugate(residual)
        score = np.abs(correlation) ** 2 / energy
        score[
            _exclusion_mask(
                dictionary,
                support,
                exclusion_range_m,
                exclusion_azimuth_rad,
                exclusion_elevation_rad,
            )
        ] = -np.inf
        selected = int(np.argmax(score))
        if not np.isfinite(score[selected]):
            raise RuntimeError("OMP could not select a distinct candidate")
        support.append(selected)
        gains, residual, residual_energy, condition = _fit_dictionary_support(
            templates, received, support
        )
    return _pack_grid_result(
        dictionary, received, support, gains, residual_energy, condition
    )


def _select_ols_candidate(
    dictionary: PlanarTemplateDictionary,
    residual: np.ndarray,
    selected_templates: np.ndarray,
    selected_indices: list[int],
    *,
    exclusion_range_m: float,
    exclusion_azimuth_rad: float,
    exclusion_elevation_rad: float,
    candidate_chunk_size: int,
) -> int:
    """Select one grid atom by its conditional residual reduction."""

    if candidate_chunk_size < 1:
        raise ValueError("candidate_chunk_size must be positive")
    templates = np.asarray(dictionary.templates)
    energy = np.sum(np.abs(templates) ** 2, axis=1, dtype=float)
    if np.any(energy <= 0.0):
        raise ValueError("dictionary contains a zero-energy template")
    current = np.asarray(selected_templates, dtype=complex)
    if current.ndim != 2 or current.shape[1] != dictionary.observation_size:
        raise ValueError("selected_templates must have shape (P, observation_size)")

    basis = None
    if current.shape[0]:
        basis = np.linalg.qr(current.T, mode="reduced")[0]
    score = np.full(dictionary.num_candidates, -np.inf, dtype=float)
    for start in range(0, dictionary.num_candidates, candidate_chunk_size):
        stop = min(start + candidate_chunk_size, dictionary.num_candidates)
        candidate_matrix = np.asarray(templates[start:stop], dtype=complex).T
        if basis is not None:
            candidate_matrix = candidate_matrix - basis @ (
                basis.conjugate().T @ candidate_matrix
            )
        conditional_energy = np.sum(np.abs(candidate_matrix) ** 2, axis=0)
        usable = conditional_energy > np.finfo(float).eps * float(np.max(energy))
        local = np.full(stop - start, -np.inf, dtype=float)
        local[usable] = (
            np.abs(candidate_matrix[:, usable].conjugate().T @ residual) ** 2
            / conditional_energy[usable]
        )
        score[start:stop] = local
    score[
        _exclusion_mask(
            dictionary,
            selected_indices,
            exclusion_range_m,
            exclusion_azimuth_rad,
            exclusion_elevation_rad,
        )
    ] = -np.inf
    selected = int(np.argmax(score))
    if not np.isfinite(score[selected]):
        raise RuntimeError("OLS could not select a distinct candidate")
    return selected


def ols_three_path_estimate(
    dictionary: PlanarTemplateDictionary,
    observation,
    *,
    num_paths: int = 3,
    exclusion_range_m: float = 0.0,
    exclusion_azimuth_rad: float = 0.0,
    exclusion_elevation_rad: float = 0.0,
    candidate_chunk_size: int = 256,
) -> dict:
    """Grid OLS whose later atoms maximize conditional residual reduction."""

    received = np.asarray(observation, dtype=complex).reshape(-1)
    if received.size != dictionary.observation_size or num_paths < 1:
        raise ValueError("invalid observation size or path count")
    templates = np.asarray(dictionary.templates)
    support: list[int] = []
    gains = np.empty(0, dtype=complex)
    residual = received.copy()
    residual_energy = float(np.vdot(received, received).real)
    condition = 1.0
    for _ in range(num_paths):
        selected_templates = (
            np.asarray(templates[support], dtype=complex)
            if support
            else np.empty((0, dictionary.observation_size), dtype=complex)
        )
        selected = _select_ols_candidate(
            dictionary,
            residual,
            selected_templates,
            support,
            exclusion_range_m=exclusion_range_m,
            exclusion_azimuth_rad=exclusion_azimuth_rad,
            exclusion_elevation_rad=exclusion_elevation_rad,
            candidate_chunk_size=candidate_chunk_size,
        )
        support.append(selected)
        gains, residual, residual_energy, condition = _fit_dictionary_support(
            templates, received, support
        )
    return _pack_grid_result(
        dictionary, received, support, gains, residual_energy, condition
    )


def _fit_continuous(
    operator: PlanarDmaTemplateOperator,
    observation: np.ndarray,
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    templates = operator.evaluate(
        coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    )
    matrix = templates.T
    gains, _, _, singular_values = np.linalg.lstsq(matrix, observation, rcond=None)
    residual = observation - matrix @ gains
    energy = float(np.vdot(residual, residual).real)
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else float("inf")
    )
    return templates, gains, energy, condition


def _continuous_refinement(
    operator: PlanarDmaTemplateOperator,
    observation: np.ndarray,
    initial_coordinates: np.ndarray,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    grid_steps: np.ndarray,
    refinement_iterations: int,
    derivative_fraction: float,
    maximum_update_grid_units: float,
    convergence_tolerance: float,
) -> dict:
    coordinates = np.asarray(initial_coordinates, dtype=float).copy()
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("initial_coordinates must have shape (P, 3)")
    if refinement_iterations < 1 or not 0.0 < derivative_fraction <= 0.5:
        raise ValueError("invalid refinement settings")
    if maximum_update_grid_units <= 0.0:
        raise ValueError("maximum_update_grid_units must be positive")
    templates, gains, residual_energy, condition = _fit_continuous(
        operator, observation, coordinates
    )
    initial_energy = residual_energy
    completed = 0
    converged = 0

    for iteration in range(1, refinement_iterations + 1):
        residual = observation - templates.T @ gains
        derivative_candidates = []
        derivative_denominators = []
        for path_index in range(coordinates.shape[0]):
            for dimension in range(3):
                delta = derivative_fraction * grid_steps[dimension]
                lower = coordinates[path_index].copy()
                upper = coordinates[path_index].copy()
                lower[dimension] = max(
                    lower_bounds[dimension], lower[dimension] - delta
                )
                upper[dimension] = min(
                    upper_bounds[dimension], upper[dimension] + delta
                )
                derivative_candidates.extend((lower, upper))
                derivative_denominators.append(
                    max(upper[dimension] - lower[dimension], np.finfo(float).eps)
                )
        candidate_array = np.asarray(derivative_candidates, dtype=float)
        candidate_templates = operator.evaluate(
            candidate_array[:, 0], candidate_array[:, 1], candidate_array[:, 2]
        )
        derivative_columns = []
        cursor = 0
        for path_index in range(coordinates.shape[0]):
            for dimension in range(3):
                derivative = (
                    (candidate_templates[cursor + 1] - candidate_templates[cursor])
                    / derivative_denominators[cursor // 2]
                    * grid_steps[dimension]
                )
                derivative_columns.append(gains[path_index] * derivative)
                cursor += 2
        derivative_matrix = np.column_stack(derivative_columns)
        signal_matrix = templates.T
        derivative_matrix -= signal_matrix @ np.linalg.lstsq(
            signal_matrix, derivative_matrix, rcond=None
        )[0]
        real_system = np.vstack((derivative_matrix.real, derivative_matrix.imag))
        real_residual = np.concatenate((residual.real, residual.imag))
        update, _, _, _ = np.linalg.lstsq(real_system, real_residual, rcond=None)
        update = np.clip(
            update.reshape(coordinates.shape),
            -maximum_update_grid_units,
            maximum_update_grid_units,
        )
        if not np.all(np.isfinite(update)):
            break

        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.125):
            trial = np.clip(
                coordinates + scale * update * grid_steps[None, :],
                lower_bounds[None, :],
                upper_bounds[None, :],
            )
            trial_templates, trial_gains, trial_energy, trial_condition = (
                _fit_continuous(operator, observation, trial)
            )
            if trial_energy <= residual_energy * (1.0 + 1.0e-12):
                improvement = residual_energy - trial_energy
                coordinates = trial
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
        if not accepted:
            converged = 1
            break
        if converged:
            break
    return {
        "coordinates": coordinates,
        "complex_gain": gains,
        "initial_residual_energy": float(initial_energy),
        "residual_energy": float(residual_energy),
        "support_condition_number": float(condition),
        "iterations": int(completed),
        "converged": int(converged),
    }


def yang_style_ogols_3d_estimate(
    dictionary: PlanarTemplateDictionary,
    operator: PlanarDmaTemplateOperator,
    observation,
    *,
    num_paths: int,
    range_bounds_m: tuple[float, float],
    azimuth_bounds_rad: tuple[float, float],
    elevation_bounds_rad: tuple[float, float],
    range_grid_step_m: float,
    azimuth_grid_step_rad: float,
    elevation_grid_step_rad: float,
    exclusion_range_m: float,
    exclusion_azimuth_rad: float,
    exclusion_elevation_rad: float,
    refinement_iterations: int,
    derivative_fraction: float,
    maximum_update_grid_units: float,
    convergence_tolerance: float,
    candidate_chunk_size: int = 256,
) -> dict:
    """Alternate 3-D OLS atom selection and off-grid variable projection."""

    received = np.asarray(observation, dtype=complex).reshape(-1)
    if received.size != dictionary.observation_size or num_paths < 1:
        raise ValueError("invalid observation size or path count")
    lower_bounds = np.array(
        [range_bounds_m[0], azimuth_bounds_rad[0], elevation_bounds_rad[0]],
        dtype=float,
    )
    upper_bounds = np.array(
        [range_bounds_m[1], azimuth_bounds_rad[1], elevation_bounds_rad[1]],
        dtype=float,
    )
    grid_steps = np.array(
        [range_grid_step_m, azimuth_grid_step_rad, elevation_grid_step_rad],
        dtype=float,
    )
    support: list[int] = []
    coordinates = np.empty((0, 3), dtype=float)
    selected_templates = np.empty(
        (0, dictionary.observation_size), dtype=complex
    )
    gains = np.empty(0, dtype=complex)
    residual = received.copy()
    residual_energy = float(np.vdot(received, received).real)
    condition = 1.0
    total_refinement_iterations = 0
    all_stages_converged = True
    all_stages_nonincreasing = True
    final_stage_initial_energy = residual_energy

    for _ in range(num_paths):
        selected = _select_ols_candidate(
            dictionary,
            residual,
            selected_templates,
            support,
            exclusion_range_m=exclusion_range_m,
            exclusion_azimuth_rad=exclusion_azimuth_rad,
            exclusion_elevation_rad=exclusion_elevation_rad,
            candidate_chunk_size=candidate_chunk_size,
        )
        support.append(selected)
        grid_coordinate = np.array(
            [
                dictionary.ranges_m[selected],
                dictionary.azimuths_rad[selected],
                dictionary.elevations_rad[selected],
            ],
            dtype=float,
        )
        coordinates = np.vstack((coordinates, grid_coordinate))
        refined = _continuous_refinement(
            operator,
            received,
            coordinates,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            grid_steps=grid_steps,
            refinement_iterations=refinement_iterations,
            derivative_fraction=derivative_fraction,
            maximum_update_grid_units=maximum_update_grid_units,
            convergence_tolerance=convergence_tolerance,
        )
        coordinates = np.asarray(refined["coordinates"], dtype=float)
        gains = np.asarray(refined["complex_gain"], dtype=complex)
        residual_energy = float(refined["residual_energy"])
        condition = float(refined["support_condition_number"])
        total_refinement_iterations += int(refined["iterations"])
        all_stages_converged &= bool(refined["converged"])
        all_stages_nonincreasing &= residual_energy <= float(
            refined["initial_residual_energy"]
        ) * (1.0 + 1.0e-12)
        final_stage_initial_energy = float(refined["initial_residual_energy"])
        selected_templates = operator.evaluate(
            coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        )
        residual = received - selected_templates.T @ gains

    total_energy = float(np.vdot(received, received).real)
    return {
        "candidate_index": np.asarray(support, dtype=int),
        "range_m": coordinates[:, 0],
        "azimuth_rad": coordinates[:, 1],
        "elevation_rad": coordinates[:, 2],
        "complex_gain": gains,
        "explained_energy": max(0.0, total_energy - residual_energy),
        "initial_residual_energy": final_stage_initial_energy,
        "residual_energy": residual_energy,
        "support_condition_number": condition,
        "iterations": total_refinement_iterations,
        "converged": int(all_stages_converged),
        "refinement_stages": num_paths,
        "refinement_all_stages_nonincreasing": int(all_stages_nonincreasing),
        "screened_candidate_count": dictionary.num_candidates,
    }


def oracle_geometry_ls_estimate(
    operator: PlanarDmaTemplateOperator,
    observation,
    ranges_m,
    azimuths_rad,
    elevations_rad,
) -> dict:
    """Genie-aided geometry with complex gains fitted to the same noisy observation."""

    received = np.asarray(observation, dtype=complex).reshape(-1)
    coordinates = np.column_stack(
        (
            np.asarray(ranges_m, dtype=float),
            np.asarray(azimuths_rad, dtype=float),
            np.asarray(elevations_rad, dtype=float),
        )
    )
    templates, gains, residual_energy, condition = _fit_continuous(
        operator, received, coordinates
    )
    total_energy = float(np.vdot(received, received).real)
    return {
        "candidate_index": np.full(coordinates.shape[0], -1, dtype=int),
        "range_m": coordinates[:, 0],
        "azimuth_rad": coordinates[:, 1],
        "elevation_rad": coordinates[:, 2],
        "complex_gain": gains,
        "explained_energy": max(0.0, total_energy - residual_energy),
        "residual_energy": float(residual_energy),
        "initial_residual_energy": float(residual_energy),
        "support_condition_number": float(condition),
        "iterations": 1,
        "converged": 1,
        "screened_candidate_count": 0,
        "oracle_templates_finite": int(np.all(np.isfinite(templates))),
    }
