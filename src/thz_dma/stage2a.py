"""Stage 2-A paired three-dimensional classical baselines."""

from __future__ import annotations

import copy
import itertools
import time
from pathlib import Path
from typing import Callable

import numpy as np

from thz_dma.channels.planar import PlanarPath, multipath_planar_channel
from thz_dma.estimators.planar_sparse import (
    PlanarDmaTemplateOperator,
    PlanarTemplateBatchEvaluator,
    build_planar_template_dictionary,
    build_planar_template_dictionary_from_candidates,
    omp_three_path_estimate,
    oracle_geometry_ls_estimate,
    yang_style_ogols_3d_estimate,
)
from thz_dma.evaluation.beamforming import (
    build_planar_data_codebook,
    feasible_planar_data_metrics,
)
from thz_dma.observations.multistrip import (
    array_input_noise_variance,
    effective_tensor_output_snr_db,
    observe_multistrip_schedule,
)
from thz_dma.stage2 import (
    Stage2Schedule,
    build_stage2_platform,
    sample_stage2_paths,
    stage2_schedule_weights,
)
from thz_dma.surfaces.multistrip import combiner_row_energy


METHOD_PROVENANCE = {
    "oracle_geometry_ls": {
        "label": "Oracle geometry + LS gains",
        "role": "genie-aided geometry benchmark",
        "source_key": "internal_oracle",
        "adaptation": "true path geometry; gains fitted to the same noisy observation",
        "paper_exact_reproduction": False,
        "uses_extra_information": True,
    },
    "grid_omp_3d": {
        "label": "3-D grid OMP",
        "role": "common low-complexity sparse baseline",
        "source_key": "internal_stage2a",
        "adaptation": "three-step range-azimuth-elevation OMP with joint gain refit",
        "paper_exact_reproduction": False,
        "uses_extra_information": False,
    },
    "yang_ogols_3d": {
        "label": "Yang-style 3-D OG-OLS (adapted)",
        "role": "common strong off-grid classical baseline",
        "source_key": "yang2024xldma",
        "adaptation": (
            "three-path receive-DMA specialization: noise-whitened 3-D OLS "
            "atom selection alternated with exact spherical variable-projection "
            "refinement and residual update"
        ),
        "paper_exact_reproduction": False,
        "uses_extra_information": False,
    },
}


def _grid_from_step(lower: float, upper: float, step: float) -> np.ndarray:
    if not np.isfinite([lower, upper, step]).all() or lower >= upper or step <= 0.0:
        raise ValueError("grid bounds and step are invalid")
    intervals = int(round((upper - lower) / step))
    if intervals < 1 or not np.isclose(
        lower + intervals * step, upper, rtol=0.0, atol=step * 1.0e-8
    ):
        raise ValueError("grid interval must be an integer multiple of step")
    return np.linspace(lower, upper, intervals + 1)


def _pilot_indices(num_subcarriers: int, count: int) -> np.ndarray:
    if not 1 <= count <= num_subcarriers:
        raise ValueError("evaluation subcarrier count is invalid")
    indices = np.rint(np.linspace(0, num_subcarriers - 1, count)).astype(int)
    if np.unique(indices).size != count:
        raise RuntimeError("evaluation subcarrier selection contains duplicates")
    return indices


def _operator_for_schedule(platform, schedule, weights, config) -> PlanarDmaTemplateOperator:
    frequencies = platform.frequencies_hz[schedule.pilot_indices]
    if bool(config["channel"]["include_absorption"]):
        absorption = np.asarray(
            platform.absorption_table.coefficient(
                frequencies, method=str(config["atmosphere"]["interpolation"])
            ),
            dtype=float,
        )
    else:
        absorption = np.zeros_like(frequencies, dtype=float)
    energy_jks = combiner_row_energy(weights)
    relative_jks = energy_jks + float(
        config["noise"]["rf_noise_variance_fraction_of_element_noise"]
    )
    relative_sjk = np.transpose(relative_jks, (2, 0, 1))
    return PlanarDmaTemplateOperator(
        frequencies_hz=frequencies,
        element_positions_m=platform.element_positions_m,
        weights=weights,
        absorption_coefficient_per_m=absorption,
        relative_output_noise_variance=relative_sjk,
    )


def build_stage2a_operators(config: dict, project_root: str | Path):
    """Build the shared platform, physical schedule weights, and operators."""

    platform = build_stage2_platform(config, project_root)
    weights = {}
    operators = {}
    for schedule in platform.schedules:
        weights[schedule.name] = stage2_schedule_weights(
            platform, schedule, config, "thz_physical"
        )
        operators[schedule.name] = _operator_for_schedule(
            platform, schedule, weights[schedule.name], config
        )
    return platform, weights, operators


def _validate_accelerated_dictionary(
    operator: PlanarDmaTemplateOperator,
    dictionary,
    *,
    sample_count: int = 16,
    maximum_relative_error: float = 1.0e-4,
    minimum_rho2: float = 0.99999,
) -> dict:
    """Compare a CUDA-built dictionary sample with the NumPy reference."""

    count = min(int(sample_count), int(dictionary.num_candidates))
    indices = np.unique(
        np.rint(np.linspace(0, dictionary.num_candidates - 1, count)).astype(int)
    )
    reference = operator.evaluate(
        np.asarray(dictionary.ranges_m)[indices],
        np.asarray(dictionary.azimuths_rad)[indices],
        np.asarray(dictionary.elevations_rad)[indices],
    ).astype(np.complex64)
    accelerated = np.asarray(dictionary.templates)[indices].astype(np.complex64)
    reference128 = reference.astype(np.complex128)
    accelerated128 = accelerated.astype(np.complex128)
    relative_error = float(
        np.linalg.norm(accelerated128 - reference128)
        / np.linalg.norm(reference128)
    )
    numerator = np.abs(
        np.sum(accelerated128 * np.conjugate(reference128), axis=1)
    ) ** 2
    denominator = np.sum(np.abs(accelerated128) ** 2, axis=1) * np.sum(
        np.abs(reference128) ** 2, axis=1
    )
    rho2 = numerator / denominator
    rho2_min = float(np.min(rho2))
    passed = bool(
        np.isfinite(relative_error)
        and np.isfinite(rho2_min)
        and relative_error <= maximum_relative_error
        and rho2_min >= minimum_rho2
    )
    result = {
        "status": "pass" if passed else "fail",
        "sample_count": int(indices.size),
        "relative_frobenius_error": relative_error,
        "minimum_row_rho2": rho2_min,
        "maximum_relative_error": float(maximum_relative_error),
        "minimum_rho2": float(minimum_rho2),
        "reference_backend": "numpy",
    }
    if not passed:
        raise RuntimeError(
            "accelerated dictionary failed its NumPy equivalence check: "
            f"relative error {relative_error:.3e}, minimum rho^2 {rho2_min:.9f}"
        )
    return result


def build_stage2a_estimators(config: dict, project_root: str | Path, *, progress=None):
    """Build the common platform, schedule operators and coarse 3-D dictionaries."""

    notify = progress or (lambda _message: None)
    platform, weights, operators = build_stage2a_operators(config, project_root)
    search = config["search"]
    ranges = _grid_from_step(
        float(search["range_min_m"]),
        float(search["range_max_m"]),
        float(search["range_step_m"]),
    )
    azimuths_deg = _grid_from_step(
        float(search["azimuth_min_deg"]),
        float(search["azimuth_max_deg"]),
        float(search["azimuth_step_deg"]),
    )
    elevations_deg = _grid_from_step(
        float(search["elevation_min_deg"]),
        float(search["elevation_max_deg"]),
        float(search["elevation_step_deg"]),
    )
    dictionaries = {}
    build_seconds = {}
    dictionary_memory_bytes = {}
    dictionary_compute_backend = {}
    dictionary_compute_device = {}
    dictionary_compute_precision = {}
    dictionary_compute_validation = {}
    for schedule in platform.schedules:
        started = time.perf_counter()
        dictionaries[schedule.name] = build_planar_template_dictionary(
            operators[schedule.name],
            ranges,
            np.deg2rad(azimuths_deg),
            np.deg2rad(elevations_deg),
            chunk_size=int(search["dictionary_chunk_size"]),
            storage_dtype=str(search["dictionary_storage_dtype"]),
            compute_backend=str(
                search.get("dictionary_compute_backend", "numpy")
            ),
            compute_device=str(
                search.get("dictionary_compute_device", "cuda:0")
            ),
            progress=lambda message, name=schedule.name: notify(f"{name}: {message}"),
        )
        build_seconds[schedule.name] = float(time.perf_counter() - started)
        dictionary_memory_bytes[schedule.name] = int(
            dictionaries[schedule.name].templates.nbytes
        )
        dictionary_compute_backend[schedule.name] = dictionaries[
            schedule.name
        ].compute_backend
        dictionary_compute_device[schedule.name] = dictionaries[
            schedule.name
        ].compute_device
        dictionary_compute_precision[schedule.name] = dictionaries[
            schedule.name
        ].compute_precision
        if dictionaries[schedule.name].compute_backend != "numpy":
            dictionary_compute_validation[
                schedule.name
            ] = _validate_accelerated_dictionary(
                operators[schedule.name], dictionaries[schedule.name]
            )
    return (
        platform,
        weights,
        operators,
        dictionaries,
        {
            "range_grid_m": ranges,
            "azimuth_grid_deg": azimuths_deg,
            "elevation_grid_deg": elevations_deg,
            "num_candidates": int(ranges.size * azimuths_deg.size * elevations_deg.size),
            "dictionary_build_seconds": build_seconds,
            "dictionary_memory_bytes": dictionary_memory_bytes,
            "dictionary_compute_backend": dictionary_compute_backend,
            "dictionary_compute_device": dictionary_compute_device,
            "dictionary_compute_precision": dictionary_compute_precision,
            "dictionary_compute_validation": dictionary_compute_validation,
        },
    )


def _estimate(method: str, dictionary, operator, observation, paths, config) -> dict:
    estimator = config["estimator"]
    search = config["search"]
    common = {
        "num_paths": int(config["channel"]["num_paths"]),
        "exclusion_range_m": float(estimator["exclusion_range_m"]),
        "exclusion_azimuth_rad": float(
            np.deg2rad(estimator["exclusion_azimuth_deg"])
        ),
        "exclusion_elevation_rad": float(
            np.deg2rad(estimator["exclusion_elevation_deg"])
        ),
    }
    if method == "grid_omp_3d":
        return omp_three_path_estimate(dictionary, observation, **common)
    if method == "yang_ogols_3d":
        settings = estimator["yang_ogols_3d"]
        return yang_style_ogols_3d_estimate(
            dictionary,
            operator,
            observation,
            **common,
            range_bounds_m=(
                float(search["range_min_m"]),
                float(search["range_max_m"]),
            ),
            azimuth_bounds_rad=(
                float(np.deg2rad(search["azimuth_min_deg"])),
                float(np.deg2rad(search["azimuth_max_deg"])),
            ),
            elevation_bounds_rad=(
                float(np.deg2rad(search["elevation_min_deg"])),
                float(np.deg2rad(search["elevation_max_deg"])),
            ),
            range_grid_step_m=float(search["range_step_m"]),
            azimuth_grid_step_rad=float(np.deg2rad(search["azimuth_step_deg"])),
            elevation_grid_step_rad=float(
                np.deg2rad(search["elevation_step_deg"])
            ),
            refinement_iterations=int(settings["refinement_iterations"]),
            derivative_fraction=float(settings["derivative_fraction"]),
            maximum_update_grid_units=float(settings["maximum_update_grid_units"]),
            convergence_tolerance=float(settings["convergence_tolerance"]),
            candidate_chunk_size=int(settings["candidate_chunk_size"]),
        )
    if method == "oracle_geometry_ls":
        return oracle_geometry_ls_estimate(
            operator,
            observation,
            [path.range_m for path in paths],
            [path.azimuth_rad for path in paths],
            [path.elevation_rad for path in paths],
        )
    raise ValueError(f"unsupported Stage 2-A estimator: {method}")


def _match_paths(estimates: dict, paths: tuple[PlanarPath, ...], config: dict) -> dict:
    evaluation = config["evaluation"]
    true_coordinates = np.array(
        [[path.range_m, path.azimuth_rad, path.elevation_rad] for path in paths],
        dtype=float,
    )
    estimated_coordinates = np.column_stack(
        (
            np.asarray(estimates["range_m"], dtype=float),
            np.asarray(estimates["azimuth_rad"], dtype=float),
            np.asarray(estimates["elevation_rad"], dtype=float),
        )
    )
    estimated_gains = np.asarray(estimates["complex_gain"], dtype=complex)
    path_count = len(paths)
    if estimated_coordinates.shape != (path_count, 3) or estimated_gains.shape != (
        path_count,
    ):
        raise ValueError("estimator path count does not match the truth")
    scales = np.array(
        [
            float(evaluation["path_match_range_scale_m"]),
            float(np.deg2rad(evaluation["path_match_azimuth_scale_deg"])),
            float(np.deg2rad(evaluation["path_match_elevation_scale_deg"])),
        ]
    )
    best = None
    best_cost = float("inf")
    for permutation in itertools.permutations(range(path_count)):
        candidate = estimated_coordinates[np.asarray(permutation)]
        cost = float(np.sum(((candidate - true_coordinates) / scales[None, :]) ** 2))
        if cost < best_cost:
            best_cost = cost
            best = np.asarray(permutation, dtype=int)
    if best is None:
        raise RuntimeError("path matching failed")
    matched_coordinates = estimated_coordinates[best]
    matched_gains = estimated_gains[best]
    errors = matched_coordinates - true_coordinates
    return {
        "coordinates": matched_coordinates,
        "gains": matched_gains,
        "range_error_m": errors[:, 0],
        "azimuth_error_deg": np.rad2deg(errors[:, 1]),
        "elevation_error_deg": np.rad2deg(errors[:, 2]),
        "matching_cost": best_cost,
    }


def _channel_from_estimate(frequencies, platform, coordinates, gains, config):
    paths = tuple(
        PlanarPath(
            range_m=float(coordinates[index, 0]),
            azimuth_rad=float(coordinates[index, 1]),
            elevation_rad=float(coordinates[index, 2]),
            complex_gain=complex(gains[index]),
        )
        for index in range(coordinates.shape[0])
    )
    channel, _ = multipath_planar_channel(
        frequencies,
        platform.element_positions_m,
        paths,
        model="spherical_thz",
        carrier_frequency_hz=float(config["system"]["carrier_frequency_hz"]),
        absorption_table=platform.absorption_table,
        include_absorption=bool(config["channel"]["include_absorption"]),
        interpolation=str(config["atmosphere"]["interpolation"]),
    )
    return channel


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


def _method_names(config: dict) -> list[str]:
    methods = [str(item) for item in config["experiment"]["estimators"]]
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("Stage 2-A estimators must be non-empty and unique")
    unknown = sorted(set(methods) - set(METHOD_PROVENANCE))
    if unknown:
        raise ValueError(f"unsupported Stage 2-A estimators: {unknown}")
    required = {"oracle_geometry_ls", "grid_omp_3d", "yang_ogols_3d"}
    if set(methods) != required:
        raise ValueError("Stage 2-A requires Oracle, grid OMP and Yang-style OG-OLS")
    return methods


def _declared_channel_support_bounds(config: dict) -> dict[str, tuple[float, float]]:
    """Return conservative geometry bounds for every generated Stage 2 path."""

    channel = config["channel"]
    has_reflections = int(channel["num_paths"]) > 1
    range_extension = (
        float(channel["reflection_excess_range_max_m"])
        if has_reflections
        else 0.0
    )
    azimuth_extension = (
        float(channel["reflection_azimuth_separation_max_deg"])
        if has_reflections
        else 0.0
    )
    elevation_extension = (
        float(channel["reflection_elevation_separation_max_deg"])
        if has_reflections
        else 0.0
    )
    return {
        "range_m": (
            float(channel["range_min_m"]),
            float(channel["range_max_m"]) + range_extension,
        ),
        "azimuth_deg": (
            float(channel["azimuth_min_deg"]) - azimuth_extension,
            float(channel["azimuth_max_deg"]) + azimuth_extension,
        ),
        "elevation_deg": (
            float(channel["elevation_min_deg"]) - elevation_extension,
            float(channel["elevation_max_deg"]) + elevation_extension,
        ),
    }


def _validate_search_contains_channel_support(config: dict) -> dict:
    """Fail before dictionary construction if generated paths can lie outside it."""

    support = _declared_channel_support_bounds(config)
    search = config["search"]
    search_bounds = {
        "range_m": (
            float(search["range_min_m"]),
            float(search["range_max_m"]),
        ),
        "azimuth_deg": (
            float(search["azimuth_min_deg"]),
            float(search["azimuth_max_deg"]),
        ),
        "elevation_deg": (
            float(search["elevation_min_deg"]),
            float(search["elevation_max_deg"]),
        ),
    }
    uncovered = []
    for dimension, truth_bounds in support.items():
        lower, upper = search_bounds[dimension]
        if truth_bounds[0] < lower - 1.0e-12 or truth_bounds[1] > upper + 1.0e-12:
            uncovered.append(dimension)
    if uncovered:
        raise ValueError(
            "search dictionary does not cover all declared path support dimensions: "
            + ", ".join(uncovered)
        )
    return {
        "declared_channel_support_bounds": {
            key: list(value) for key, value in support.items()
        },
        "fixed_search_bounds": {
            key: list(value) for key, value in search_bounds.items()
        },
        "search_contains_declared_channel_support": True,
    }


def _search_mode(config: dict) -> str:
    mode = str(config["search"].get("mode", "fixed"))
    if mode not in {"fixed", "beam_center_adaptive"}:
        raise ValueError(f"unsupported Stage 2-A search mode: {mode}")
    return mode


def _search_bounds_radians(config: dict) -> tuple[np.ndarray, np.ndarray]:
    search = config["search"]
    lower = np.array(
        [
            float(search["range_min_m"]),
            float(np.deg2rad(search["azimuth_min_deg"])),
            float(np.deg2rad(search["elevation_min_deg"])),
        ],
        dtype=float,
    )
    upper = np.array(
        [
            float(search["range_max_m"]),
            float(np.deg2rad(search["azimuth_max_deg"])),
            float(np.deg2rad(search["elevation_max_deg"])),
        ],
        dtype=float,
    )
    return lower, upper


def _adaptive_settings(config: dict) -> dict:
    if "adaptive_search" not in config:
        raise ValueError("beam_center_adaptive mode requires [adaptive_search]")
    raw = config["adaptive_search"]
    settings = {
        "correlation_threshold": float(raw["correlation_threshold"]),
        "window_fullwidth_multiplier": float(raw["window_fullwidth_multiplier"]),
        "grid_points_per_3db_width": float(raw["grid_points_per_3db_width"]),
        "probe_steps": np.array(
            [
                float(raw["probe_range_step_m"]),
                float(np.deg2rad(raw["probe_azimuth_step_deg"])),
                float(np.deg2rad(raw["probe_elevation_step_deg"])),
            ],
            dtype=float,
        ),
        "bisection_iterations": int(raw["bisection_iterations"]),
        "probe_batch_size": int(raw.get("probe_batch_size", 32)),
        "truth_offset_fraction": float(raw["truth_offset_halfpower_fraction"]),
        "truth_offset_caps": np.array(
            [
                float(raw["truth_offset_cap_range_m"]),
                float(np.deg2rad(raw["truth_offset_cap_azimuth_deg"])),
                float(np.deg2rad(raw["truth_offset_cap_elevation_deg"])),
            ],
            dtype=float,
        ),
        "preserve_channel_relationships": bool(
            raw.get("preserve_channel_relationships", True)
        ),
        "maximum_candidates": int(raw["maximum_candidates_per_scene"]),
    }
    if not 0.0 < settings["correlation_threshold"] < 1.0:
        raise ValueError("adaptive correlation threshold must lie in (0, 1)")
    if settings["window_fullwidth_multiplier"] < 1.0:
        raise ValueError("adaptive window multiplier must be at least one")
    if settings["grid_points_per_3db_width"] < 2.0:
        raise ValueError("adaptive grid requires at least two points per 3 dB width")
    if np.any(settings["probe_steps"] <= 0.0):
        raise ValueError("adaptive profile probe steps must be positive")
    if settings["bisection_iterations"] < 0 or settings["probe_batch_size"] < 1:
        raise ValueError("invalid adaptive profile refinement settings")
    if not 0.0 < settings["truth_offset_fraction"] <= 1.0:
        raise ValueError("truth offset fraction must lie in (0, 1]")
    if np.any(settings["truth_offset_caps"] <= 0.0):
        raise ValueError("truth offset caps must be positive")
    if settings["maximum_candidates"] < 1:
        raise ValueError("maximum adaptive candidate count must be positive")
    return settings


def _normalized_template_rho2(
    evaluator: PlanarTemplateBatchEvaluator,
    reference: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    candidates = evaluator.evaluate(
        coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    )
    reference_energy = float(np.vdot(reference, reference).real)
    candidate_energy = np.sum(np.abs(candidates) ** 2, axis=1, dtype=float)
    if reference_energy <= 0.0 or np.any(candidate_energy <= 0.0):
        raise ValueError("adaptive profile contains a zero-energy template")
    return (
        np.abs(candidates @ np.conjugate(reference)) ** 2
        / candidate_energy
        / reference_energy
    )


def _nearest_profile_crossing(
    evaluator: PlanarTemplateBatchEvaluator,
    center: np.ndarray,
    reference: np.ndarray,
    *,
    dimension: int,
    direction: float,
    bound: float,
    probe_step: float,
    threshold: float,
    bisection_iterations: int,
    batch_size: int,
) -> tuple[float, bool]:
    maximum_distance = float(direction * (bound - center[dimension]))
    if maximum_distance < -1.0e-14:
        raise ValueError("adaptive profile bound is on the wrong side of its center")
    if maximum_distance <= 1.0e-14:
        return 0.0, True
    count = int(np.ceil(maximum_distance / probe_step))
    previous_distance = 0.0
    for first in range(1, count + 1, batch_size):
        indices = np.arange(first, min(count, first + batch_size - 1) + 1)
        distances = np.minimum(indices * probe_step, maximum_distance)
        points = np.repeat(center[None, :], distances.size, axis=0)
        points[:, dimension] += direction * distances
        rho2 = _normalized_template_rho2(evaluator, reference, points)
        crossings = np.flatnonzero(rho2 <= threshold)
        if crossings.size:
            crossing_index = int(crossings[0])
            high = float(distances[crossing_index])
            low = float(
                distances[crossing_index - 1]
                if crossing_index > 0
                else previous_distance
            )
            for _ in range(bisection_iterations):
                middle = 0.5 * (low + high)
                point = center.copy()
                point[dimension] += direction * middle
                value = float(
                    _normalized_template_rho2(
                        evaluator, reference, point[None, :]
                    )[0]
                )
                if value > threshold:
                    low = middle
                else:
                    high = middle
            return high, False
        previous_distance = float(distances[-1])
    return maximum_distance, True


def _measure_adaptive_profiles(
    operators: dict[str, PlanarDmaTemplateOperator],
    centers: np.ndarray,
    config: dict,
    evaluators: dict[str, PlanarTemplateBatchEvaluator] | None = None,
) -> dict:
    settings = _adaptive_settings(config)
    lower, upper = _search_bounds_radians(config)
    schedule_profiles: dict[str, list[dict]] = {
        name: [] for name in operators
    }
    for schedule_name, operator in operators.items():
        evaluator = (
            evaluators[schedule_name]
            if evaluators is not None
            else PlanarTemplateBatchEvaluator(operator)
        )
        for center in centers:
            reference = evaluator.evaluate(
                np.array([center[0]]),
                np.array([center[1]]),
                np.array([center[2]]),
            )[0]
            negative = []
            positive = []
            truncated = []
            for dimension in range(3):
                left, left_truncated = _nearest_profile_crossing(
                    evaluator,
                    center,
                    reference,
                    dimension=dimension,
                    direction=-1.0,
                    bound=lower[dimension],
                    probe_step=float(settings["probe_steps"][dimension]),
                    threshold=float(settings["correlation_threshold"]),
                    bisection_iterations=int(settings["bisection_iterations"]),
                    batch_size=int(settings["probe_batch_size"]),
                )
                right, right_truncated = _nearest_profile_crossing(
                    evaluator,
                    center,
                    reference,
                    dimension=dimension,
                    direction=1.0,
                    bound=upper[dimension],
                    probe_step=float(settings["probe_steps"][dimension]),
                    threshold=float(settings["correlation_threshold"]),
                    bisection_iterations=int(settings["bisection_iterations"]),
                    batch_size=int(settings["probe_batch_size"]),
                )
                negative.append(left)
                positive.append(right)
                truncated.append([left_truncated, right_truncated])
            full_width = np.asarray(negative) + np.asarray(positive)
            if np.any(full_width <= 0.0):
                raise RuntimeError("adaptive 3 dB profile has zero full width")
            schedule_profiles[schedule_name].append(
                {
                    "negative_halfwidth_rad_units": list(map(float, negative)),
                    "positive_halfwidth_rad_units": list(map(float, positive)),
                    "fullwidth_rad_units": full_width.astype(float).tolist(),
                    "crossing_truncated": truncated,
                }
            )
    return {
        "settings": settings,
        "schedule_profiles": schedule_profiles,
    }


def _path_relationships_valid(coordinates: np.ndarray, config: dict) -> bool:
    channel = config["channel"]
    los = coordinates[0]
    if not (
        float(channel["range_min_m"]) <= los[0] <= float(channel["range_max_m"])
        and np.deg2rad(float(channel["azimuth_min_deg"]))
        <= los[1]
        <= np.deg2rad(float(channel["azimuth_max_deg"]))
        and np.deg2rad(float(channel["elevation_min_deg"]))
        <= los[2]
        <= np.deg2rad(float(channel["elevation_max_deg"]))
    ):
        return False
    for reflection in coordinates[1:]:
        excess_range = reflection[0] - los[0]
        azimuth_separation = abs(np.rad2deg(reflection[1] - los[1]))
        elevation_separation = abs(np.rad2deg(reflection[2] - los[2]))
        if not (
            float(channel["reflection_excess_range_min_m"])
            <= excess_range
            <= float(channel["reflection_excess_range_max_m"])
            and float(channel["reflection_azimuth_separation_min_deg"])
            <= azimuth_separation
            <= float(channel["reflection_azimuth_separation_max_deg"])
            and float(channel["reflection_elevation_separation_min_deg"])
            <= elevation_separation
            <= float(channel["reflection_elevation_separation_max_deg"])
        ):
            return False
    return True


def _sample_truth_around_centers(
    center_paths: tuple[PlanarPath, ...],
    profile: dict,
    config: dict,
    rng: np.random.Generator,
) -> tuple[tuple[PlanarPath, ...], dict]:
    centers = np.array(
        [
            [path.range_m, path.azimuth_rad, path.elevation_rad]
            for path in center_paths
        ],
        dtype=float,
    )
    settings = profile["settings"]
    schedules = list(profile["schedule_profiles"])
    negative = np.array(
        [
            [
                min(
                    profile["schedule_profiles"][schedule][path_index][
                        "negative_halfwidth_rad_units"
                    ][dimension]
                    for schedule in schedules
                )
                for dimension in range(3)
            ]
            for path_index in range(len(center_paths))
        ],
        dtype=float,
    )
    positive = np.array(
        [
            [
                min(
                    profile["schedule_profiles"][schedule][path_index][
                        "positive_halfwidth_rad_units"
                    ][dimension]
                    for schedule in schedules
                )
                for dimension in range(3)
            ]
            for path_index in range(len(center_paths))
        ],
        dtype=float,
    )
    caps = np.asarray(settings["truth_offset_caps"], dtype=float)[None, :]
    negative = np.minimum(
        negative * float(settings["truth_offset_fraction"]), caps
    )
    positive = np.minimum(
        positive * float(settings["truth_offset_fraction"]), caps
    )
    lower, upper = _search_bounds_radians(config)
    for _ in range(10_000):
        offsets = rng.uniform(-negative, positive)
        coordinates = centers + offsets
        if np.any(coordinates < lower[None, :]) or np.any(
            coordinates > upper[None, :]
        ):
            continue
        if bool(settings["preserve_channel_relationships"]) and not (
            _path_relationships_valid(coordinates, config)
        ):
            continue
        paths = tuple(
            PlanarPath(
                range_m=float(coordinates[index, 0]),
                azimuth_rad=float(coordinates[index, 1]),
                elevation_rad=float(coordinates[index, 2]),
                complex_gain=center_paths[index].complex_gain,
            )
            for index in range(len(center_paths))
        )
        return paths, {
            "offsets_m_rad_rad": offsets.astype(float).tolist(),
            "negative_offset_limits_m_rad_rad": negative.astype(float).tolist(),
            "positive_offset_limits_m_rad_rad": positive.astype(float).tolist(),
        }
    raise RuntimeError("could not sample a beam-conditioned path scene")


def _lattice_indices(
    lower: float,
    upper: float,
    *,
    anchor: float,
    step: float,
    maximum_index: int,
) -> np.ndarray:
    tolerance = 1.0e-10
    first = max(0, int(np.ceil((lower - anchor) / step - tolerance)))
    last = min(
        maximum_index, int(np.floor((upper - anchor) / step + tolerance))
    )
    if first <= last:
        return np.arange(first, last + 1, dtype=int)
    middle = int(np.clip(np.rint((0.5 * (lower + upper) - anchor) / step), 0, maximum_index))
    return np.array([middle], dtype=int)


def _adaptive_candidate_union(
    centers: np.ndarray,
    profile: dict,
    truth_coordinates: np.ndarray,
    config: dict,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    settings = profile["settings"]
    schedule_profiles = profile["schedule_profiles"]
    schedule_names = list(schedule_profiles)
    full_widths = np.array(
        [
            [
                max(
                    schedule_profiles[name][path_index]["fullwidth_rad_units"][dimension]
                    for name in schedule_names
                )
                for dimension in range(3)
            ]
            for path_index in range(centers.shape[0])
        ],
        dtype=float,
    )
    minimum_widths = np.min(
        np.array(
            [
                [item["fullwidth_rad_units"] for item in schedule_profiles[name]]
                for name in schedule_names
            ],
            dtype=float,
        ),
        axis=(0, 1),
    )
    step_caps = np.array(
        [
            float(config["search"]["range_step_m"]),
            float(np.deg2rad(config["search"]["azimuth_step_deg"])),
            float(np.deg2rad(config["search"]["elevation_step_deg"])),
        ],
        dtype=float,
    )
    steps = np.minimum(
        minimum_widths / float(settings["grid_points_per_3db_width"]),
        step_caps,
    )
    if np.any(steps <= 0.0) or not np.all(np.isfinite(steps)):
        raise RuntimeError("adaptive grid step is invalid")
    global_lower, global_upper = _search_bounds_radians(config)
    maximum_indices = np.floor(
        (global_upper - global_lower) / steps + 1.0e-10
    ).astype(int)
    half_span = (
        0.5
        * float(settings["window_fullwidth_multiplier"])
        * full_widths
    )
    window_lower = np.maximum(centers - half_span, global_lower[None, :])
    window_upper = np.minimum(centers + half_span, global_upper[None, :])
    index_blocks = []
    windows = []
    for path_index in range(centers.shape[0]):
        axes = [
            _lattice_indices(
                float(window_lower[path_index, dimension]),
                float(window_upper[path_index, dimension]),
                anchor=float(global_lower[dimension]),
                step=float(steps[dimension]),
                maximum_index=int(maximum_indices[dimension]),
            )
            for dimension in range(3)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        indices = np.column_stack([item.ravel() for item in mesh])
        index_blocks.append(indices)
        windows.append(
            {
                "path_center_index": path_index,
                "continuous_lower_m_rad_rad": window_lower[path_index].astype(float).tolist(),
                "continuous_upper_m_rad_rad": window_upper[path_index].astype(float).tolist(),
                "lattice_shape": [int(axis.size) for axis in axes],
                "lattice_candidates_before_union": int(indices.shape[0]),
            }
        )
    union_indices = np.unique(np.vstack(index_blocks), axis=0)
    if union_indices.shape[0] > int(settings["maximum_candidates"]):
        raise RuntimeError(
            "adaptive dictionary exceeds maximum_candidates_per_scene: "
            f"{union_indices.shape[0]} > {settings['maximum_candidates']}"
        )
    coordinates = global_lower[None, :] + union_indices * steps[None, :]
    coverage = np.all(
        (truth_coordinates >= window_lower)
        & (truth_coordinates <= window_upper),
        axis=1,
    )
    if not np.all(coverage):
        raise RuntimeError("beam-conditioned truth escaped its associated local window")
    nearest_error = np.empty_like(truth_coordinates)
    for path_index, truth in enumerate(truth_coordinates):
        distance = np.abs(coordinates - truth[None, :])
        nearest_error[path_index] = distance[int(np.argmin(np.sum((distance / steps) ** 2, axis=1)))]
    return (
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
    ), {
        "grid_steps_m_rad_rad": steps.astype(float).tolist(),
        "combined_fullwidths_m_rad_rad": full_widths.astype(float).tolist(),
        "windows": windows,
        "num_candidates": int(coordinates.shape[0]),
        "truth_associated_window_coverage": coverage.astype(bool).tolist(),
        "nearest_union_candidate_abs_error_m_rad_rad": nearest_error.astype(float).tolist(),
    }


def prepare_adaptive_stage2a_scene(
    config: dict,
    operators: dict[str, PlanarDmaTemplateOperator],
    *,
    center_rng: np.random.Generator,
    truth_rng: np.random.Generator,
    profile_evaluators: dict[str, PlanarTemplateBatchEvaluator] | None = None,
) -> tuple[tuple[PlanarPath, ...], tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    """Generate known coarse centers, continuous truth, and a local union grid."""

    center_paths = sample_stage2_paths(config, center_rng)
    centers = np.array(
        [
            [path.range_m, path.azimuth_rad, path.elevation_rad]
            for path in center_paths
        ],
        dtype=float,
    )
    profile = _measure_adaptive_profiles(
        operators, centers, config, profile_evaluators
    )
    paths, truth_metadata = _sample_truth_around_centers(
        center_paths, profile, config, truth_rng
    )
    truth_coordinates = np.array(
        [[path.range_m, path.azimuth_rad, path.elevation_rad] for path in paths],
        dtype=float,
    )
    candidates, grid_metadata = _adaptive_candidate_union(
        centers, profile, truth_coordinates, config
    )
    serializable_profiles = copy.deepcopy(profile["schedule_profiles"])
    return paths, candidates, {
        "center_source": "pre-observation random coarse path-beam centers",
        "centers_m_rad_rad": centers.astype(float).tolist(),
        "truth": truth_metadata,
        "schedule_profiles": serializable_profiles,
        **grid_metadata,
    }


def run_stage2a(
    config: dict,
    project_root: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict]:
    """Run paired 3-D classical baselines on both native acquisition schedules."""

    notify = progress or (lambda _message: None)
    started = time.perf_counter()
    if not np.isclose(float(config["acquisition"]["pilot_energy_per_re"]), 1.0):
        raise ValueError("Stage 2-A currently requires pilot_energy_per_re = 1.0")
    methods = _method_names(config)
    support_metadata = _validate_search_contains_channel_support(config)
    search_mode = _search_mode(config)
    if search_mode == "fixed":
        (
            platform,
            schedule_weights,
            operators,
            dictionaries,
            search_metadata,
        ) = build_stage2a_estimators(config, project_root, progress=notify)
    else:
        # Adaptive candidates depend on each pre-observation beam center, so only
        # the schedule operators can be reused across scenes.
        _adaptive_settings(config)
        platform, schedule_weights, operators = build_stage2a_operators(
            config, project_root
        )
        adaptive_profile_evaluators = {
            name: PlanarTemplateBatchEvaluator(
                operator,
                compute_backend=str(
                    config["search"].get(
                        "dictionary_compute_backend", "numpy"
                    )
                ),
                compute_device=str(
                    config["search"].get(
                        "dictionary_compute_device", "cuda:0"
                    )
                ),
            )
            for name, operator in operators.items()
        }
        dictionaries = {}
        search_metadata = None
    experiment = config["experiment"]
    channel_config = config["channel"]
    noise_config = config["noise"]
    evaluation = config["evaluation"]
    data_config = config["data_evaluation"]
    snr_values = [float(value) for value in noise_config["array_reference_snr_db"]]
    if not snr_values or len(set(snr_values)) != len(snr_values):
        raise ValueError("array_reference_snr_db must be non-empty and unique")

    dma = config["dma"]
    data_codebook, data_codebook_metadata = build_planar_data_codebook(
        carrier_frequency_hz=float(config["system"]["carrier_frequency_hz"]),
        num_microstrips=int(dma["num_microstrips"]),
        elements_per_microstrip=int(dma["elements_per_microstrip"]),
        tuning_bandwidth_hz=float(dma["tuning_bandwidth_hz"]),
        feed_distances_m=platform.feed_distances_m,
        quality_factor=float(dma["quality_factor"]),
        waveguide_effective_index=float(dma["waveguide_effective_index"]),
        waveguide_end_power_fraction=float(dma["waveguide_end_power_fraction"]),
        codebook_size=int(data_config["codebook_size"]),
        seed_start=int(data_config["codebook_seed_start"]),
    )
    evaluation_indices = _pilot_indices(
        platform.frequencies_hz.size,
        int(evaluation["channel_nmse_subcarrier_count"]),
    )
    center_frequency = np.array(
        [float(config["system"]["carrier_frequency_hz"])], dtype=float
    )

    rows: list[dict] = []
    execution_schedule: list[dict] = []
    timings = {method: 0.0 for method in methods}
    calls = {method: 0 for method in methods}
    adaptive_scene_search: list[dict] = []
    adaptive_build_seconds = {
        schedule.name: [] for schedule in platform.schedules
    }
    adaptive_memory_bytes = {
        schedule.name: [] for schedule in platform.schedules
    }
    adaptive_compute_metadata = {
        schedule.name: set() for schedule in platform.schedules
    }
    adaptive_compute_validation = {}
    replicate_seeds = [int(value) for value in experiment["replicate_seeds"]]
    samples_per_seed = int(experiment["samples_per_seed"])
    if not replicate_seeds or samples_per_seed < 1:
        raise ValueError("replicate seeds and samples_per_seed must be non-empty")

    for seed_index, seed in enumerate(replicate_seeds, start=1):
        scene_rng = np.random.default_rng(seed)
        for sample_index in range(samples_per_seed):
            scene_id = f"{seed}:{sample_index}"
            adaptive_metadata = None
            adaptive_candidates = None
            case_config = config
            adaptive_row_metadata: dict[str, float | int] = {
                "adaptive_search_enabled": 0
            }
            if search_mode == "beam_center_adaptive":
                center_rng = np.random.default_rng(
                    np.random.SeedSequence([seed, sample_index, 2_026_091_101])
                )
                truth_rng = np.random.default_rng(
                    np.random.SeedSequence([seed, sample_index, 2_026_091_102])
                )
                paths, adaptive_candidates, adaptive_metadata = (
                    prepare_adaptive_stage2a_scene(
                        config,
                        operators,
                        center_rng=center_rng,
                        truth_rng=truth_rng,
                        profile_evaluators=adaptive_profile_evaluators,
                    )
                )
                case_config = copy.deepcopy(config)
                adaptive_steps = np.asarray(
                    adaptive_metadata["grid_steps_m_rad_rad"], dtype=float
                )
                case_config["search"]["range_step_m"] = float(adaptive_steps[0])
                case_config["search"]["azimuth_step_deg"] = float(
                    np.rad2deg(adaptive_steps[1])
                )
                case_config["search"]["elevation_step_deg"] = float(
                    np.rad2deg(adaptive_steps[2])
                )
                adaptive_row_metadata = {
                    "adaptive_search_enabled": 1,
                    "adaptive_candidate_count": int(
                        adaptive_metadata["num_candidates"]
                    ),
                    "adaptive_range_step_m": float(adaptive_steps[0]),
                    "adaptive_azimuth_step_deg": float(
                        np.rad2deg(adaptive_steps[1])
                    ),
                    "adaptive_elevation_step_deg": float(
                        np.rad2deg(adaptive_steps[2])
                    ),
                    "adaptive_truth_window_coverage": int(
                        all(adaptive_metadata["truth_associated_window_coverage"])
                    ),
                }
                centers = np.asarray(
                    adaptive_metadata["centers_m_rad_rad"], dtype=float
                )
                offsets = np.asarray(
                    adaptive_metadata["truth"]["offsets_m_rad_rad"], dtype=float
                )
                for path_index in range(len(paths)):
                    adaptive_row_metadata.update(
                        {
                            f"coarse_center_{path_index}_range_m": float(
                                centers[path_index, 0]
                            ),
                            f"coarse_center_{path_index}_azimuth_deg": float(
                                np.rad2deg(centers[path_index, 1])
                            ),
                            f"coarse_center_{path_index}_elevation_deg": float(
                                np.rad2deg(centers[path_index, 2])
                            ),
                            f"truth_offset_{path_index}_range_m": float(
                                offsets[path_index, 0]
                            ),
                            f"truth_offset_{path_index}_azimuth_deg": float(
                                np.rad2deg(offsets[path_index, 1])
                            ),
                            f"truth_offset_{path_index}_elevation_deg": float(
                                np.rad2deg(offsets[path_index, 2])
                            ),
                        }
                    )
                adaptive_record = {
                    "scene_id": scene_id,
                    **adaptive_metadata,
                    "dictionary_by_schedule": {},
                }
                adaptive_scene_search.append(adaptive_record)
            else:
                paths = sample_stage2_paths(config, scene_rng)
            full_channel, _ = multipath_planar_channel(
                platform.frequencies_hz,
                platform.element_positions_m,
                paths,
                model="spherical_thz",
                carrier_frequency_hz=float(config["system"]["carrier_frequency_hz"]),
                absorption_table=platform.absorption_table,
                include_absorption=bool(channel_config["include_absorption"]),
                interpolation=str(config["atmosphere"]["interpolation"]),
            )
            true_center_channel = _channel_from_estimate(
                center_frequency,
                platform,
                np.array(
                    [[path.range_m, path.azimuth_rad, path.elevation_rad] for path in paths]
                ),
                np.array([path.complex_gain for path in paths]),
                config,
            )[0]

            for schedule_index, schedule in enumerate(platform.schedules):
                weights = schedule_weights[schedule.name]
                operator = operators[schedule.name]
                if search_mode == "beam_center_adaptive":
                    if adaptive_candidates is None or adaptive_metadata is None:
                        raise RuntimeError("adaptive scene preparation is missing")
                    dictionary_started = time.perf_counter()
                    dictionary = build_planar_template_dictionary_from_candidates(
                        operator,
                        *adaptive_candidates,
                        chunk_size=int(config["search"]["dictionary_chunk_size"]),
                        storage_dtype=str(
                            config["search"]["dictionary_storage_dtype"]
                        ),
                        compute_backend=str(
                            config["search"].get(
                                "dictionary_compute_backend", "numpy"
                            )
                        ),
                        compute_device=str(
                            config["search"].get(
                                "dictionary_compute_device", "cuda:0"
                            )
                        ),
                        progress=lambda message, name=schedule.name, scene=scene_id: notify(
                            f"{scene} / {name}: {message}"
                        ),
                    )
                    build_seconds = float(time.perf_counter() - dictionary_started)
                    memory_bytes = int(dictionary.templates.nbytes)
                    adaptive_build_seconds[schedule.name].append(build_seconds)
                    adaptive_memory_bytes[schedule.name].append(memory_bytes)
                    adaptive_compute_metadata[schedule.name].add(
                        (
                            dictionary.compute_backend,
                            dictionary.compute_device,
                            dictionary.compute_precision,
                        )
                    )
                    if (
                        dictionary.compute_backend != "numpy"
                        and schedule.name not in adaptive_compute_validation
                    ):
                        adaptive_compute_validation[
                            schedule.name
                        ] = _validate_accelerated_dictionary(
                            operator, dictionary
                        )
                    adaptive_record["dictionary_by_schedule"][schedule.name] = {
                        "build_seconds": build_seconds,
                        "memory_bytes": memory_bytes,
                        "compute_backend": dictionary.compute_backend,
                        "compute_device": dictionary.compute_device,
                        "compute_precision": dictionary.compute_precision,
                    }
                else:
                    dictionary = dictionaries[schedule.name]
                pilot_channel = full_channel[schedule.pilot_indices]
                for snr_index, snr_db in enumerate(snr_values):
                    element_variance = array_input_noise_variance(full_channel, snr_db)
                    rf_variance = element_variance * float(
                        noise_config["rf_noise_variance_fraction_of_element_noise"]
                    )
                    # Excluding SNR from this seed keeps the standard noise draw
                    # paired; only its scale changes across SNR levels.
                    noise_rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [seed, sample_index, schedule_index, 2_026_090_921]
                        )
                    )
                    observation = observe_multistrip_schedule(
                        weights,
                        pilot_channel,
                        pilot_energy_per_re=float(
                            config["acquisition"]["pilot_energy_per_re"]
                        ),
                        element_noise_variance=element_variance,
                        rf_noise_variance=rf_variance,
                        rng=noise_rng,
                    )
                    whitened = operator.whiten(observation.observed)
                    order_rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [
                                int(experiment["randomization_seed"]),
                                seed,
                                sample_index,
                                schedule_index,
                                snr_index,
                            ]
                        )
                    )
                    order = [methods[index] for index in order_rng.permutation(len(methods))]
                    execution_schedule.append(
                        {
                            "scene_id": scene_id,
                            "schedule": schedule.name,
                            "array_reference_snr_db": snr_db,
                            "method_order": order,
                        }
                    )
                    for order_index, method in enumerate(order):
                        method_started = time.perf_counter()
                        estimates = _estimate(
                            method,
                            dictionary,
                            operator,
                            whitened,
                            paths,
                            case_config,
                        )
                        elapsed = time.perf_counter() - method_started
                        timings[method] += elapsed
                        calls[method] += 1

                        matched = _match_paths(estimates, paths, case_config)
                        estimated_full_channel = _channel_from_estimate(
                            platform.frequencies_hz,
                            platform,
                            matched["coordinates"],
                            matched["gains"],
                            config,
                        )
                        estimated_center_channel = _channel_from_estimate(
                            center_frequency,
                            platform,
                            matched["coordinates"],
                            matched["gains"],
                            config,
                        )[0]
                        domain_metrics = channel_domain_metrics(
                            full_channel, estimated_full_channel,
                            schedule.pilot_indices, evaluation_indices,
                        )
                        estimated_clean = np.asarray(estimates["complex_gain"]) @ (
                            operator.evaluate_raw(
                                estimates["range_m"],
                                estimates["azimuth_rad"],
                                estimates["elevation_rad"],
                            )
                        )
                        clean_vector = observation.clean.reshape(-1)
                        observation_nmse = float(
                            np.sum(np.abs(estimated_clean - clean_vector) ** 2)
                            / np.sum(np.abs(clean_vector) ** 2)
                        )
                        range_errors = np.asarray(matched["range_error_m"])
                        azimuth_errors = np.asarray(matched["azimuth_error_deg"])
                        elevation_errors = np.asarray(matched["elevation_error_deg"])
                        per_path_success = (
                            (np.abs(range_errors) <= float(evaluation["path_success_range_tolerance_m"]))
                            & (
                                np.abs(azimuth_errors)
                                <= float(evaluation["path_success_azimuth_tolerance_deg"])
                            )
                            & (
                                np.abs(elevation_errors)
                                <= float(evaluation["path_success_elevation_tolerance_deg"])
                            )
                        )
                        switch_guard = schedule.num_switches * float(
                            config["acquisition"]["switch_guard_symbol_equivalents"]
                        )
                        data_metrics = feasible_planar_data_metrics(
                            true_center_channel,
                            estimated_center_channel,
                            data_codebook,
                            element_noise_variance=element_variance,
                            rf_noise_variance=rf_variance,
                            coherence_symbols=float(data_config["coherence_symbols"]),
                            training_symbols=float(schedule.training_symbols),
                            switch_guard_symbol_equivalents=switch_guard,
                        )
                        total_white_energy = float(np.vdot(whitened, whitened).real)
                        initial_residual = float(
                            estimates.get(
                                "initial_residual_energy",
                                estimates["residual_energy"],
                            )
                        )
                        coordinates = np.asarray(matched["coordinates"])
                        true_coordinates = np.array(
                            [
                                [
                                    path.range_m,
                                    path.azimuth_rad,
                                    path.elevation_rad,
                                ]
                                for path in paths
                            ],
                            dtype=float,
                        )
                        path_diagnostics = {
                            "selected_candidate_indices": ";".join(
                                str(int(value))
                                for value in np.asarray(
                                    estimates["candidate_index"], dtype=int
                                )
                            )
                        }
                        for path_index in range(len(paths)):
                            path_diagnostics.update(
                                {
                                    f"true_path_{path_index}_range_m": float(
                                        true_coordinates[path_index, 0]
                                    ),
                                    f"true_path_{path_index}_azimuth_deg": float(
                                        np.rad2deg(true_coordinates[path_index, 1])
                                    ),
                                    f"true_path_{path_index}_elevation_deg": float(
                                        np.rad2deg(true_coordinates[path_index, 2])
                                    ),
                                    f"estimated_path_{path_index}_range_m": float(
                                        coordinates[path_index, 0]
                                    ),
                                    f"estimated_path_{path_index}_azimuth_deg": float(
                                        np.rad2deg(coordinates[path_index, 1])
                                    ),
                                    f"estimated_path_{path_index}_elevation_deg": float(
                                        np.rad2deg(coordinates[path_index, 2])
                                    ),
                                }
                            )
                        search_bounds = np.array(
                            [
                                [
                                    float(config["search"]["range_min_m"]),
                                    float(config["search"]["range_max_m"]),
                                ],
                                [
                                    float(config["search"]["azimuth_min_deg"]),
                                    float(config["search"]["azimuth_max_deg"]),
                                ],
                                [
                                    float(config["search"]["elevation_min_deg"]),
                                    float(config["search"]["elevation_max_deg"]),
                                ],
                            ]
                        )
                        coordinates_for_boundary = coordinates.copy()
                        coordinates_for_boundary[:, 1:] = np.rad2deg(
                            coordinates_for_boundary[:, 1:]
                        )
                        boundary_hit = bool(
                            any(
                                np.any(
                                    np.isclose(
                                        coordinates_for_boundary[:, dimension, None],
                                        search_bounds[dimension][None, :],
                                        rtol=0.0,
                                        atol=1.0e-10,
                                    )
                                )
                                for dimension in range(3)
                            )
                        )
                        estimate_outside_local_union = False
                        if adaptive_metadata is not None:
                            local_bounds = [
                                (
                                    np.asarray(
                                        window["continuous_lower_m_rad_rad"],
                                        dtype=float,
                                    ),
                                    np.asarray(
                                        window["continuous_upper_m_rad_rad"],
                                        dtype=float,
                                    ),
                                )
                                for window in adaptive_metadata["windows"]
                            ]
                            estimate_outside_local_union = any(
                                not any(
                                    np.all(point >= lower - 1.0e-12)
                                    and np.all(point <= upper + 1.0e-12)
                                    for lower, upper in local_bounds
                                )
                                for point in coordinates
                            )
                        rows.append(
                            {
                                "scene_id": scene_id,
                                "replicate_seed": seed,
                                "sample_index": sample_index,
                                "physics_level": "thz_physical",
                                "schedule": schedule.name,
                                "schedule_family": schedule.family,
                                "array_reference_snr_db": snr_db,
                                "effective_output_snr_db": effective_tensor_output_snr_db(
                                    observation.clean,
                                    observation.output_noise_variance,
                                ),
                                "estimator": method,
                                "method_source_key": METHOD_PROVENANCE[method][
                                    "source_key"
                                ],
                                "uses_extra_information": int(
                                    METHOD_PROVENANCE[method]["uses_extra_information"]
                                ),
                                **adaptive_row_metadata,
                                "adaptive_estimate_outside_local_union": int(
                                    estimate_outside_local_union
                                ),
                                **domain_metrics,
                                "observation_nmse_linear": observation_nmse,
                                "observation_nmse_db": float(
                                    10.0
                                    * np.log10(
                                        max(observation_nmse, np.finfo(float).tiny)
                                    )
                                ),
                                "range_rmse_m": float(
                                    np.sqrt(np.mean(range_errors**2))
                                ),
                                "azimuth_rmse_deg": float(
                                    np.sqrt(np.mean(azimuth_errors**2))
                                ),
                                "elevation_rmse_deg": float(
                                    np.sqrt(np.mean(elevation_errors**2))
                                ),
                                "los_range_abs_error_m": float(abs(range_errors[0])),
                                "los_azimuth_abs_error_deg": float(
                                    abs(azimuth_errors[0])
                                ),
                                "los_elevation_abs_error_deg": float(
                                    abs(elevation_errors[0])
                                ),
                                "los_path_success": int(per_path_success[0]),
                                "all_paths_success": int(np.all(per_path_success)),
                                "matching_cost": float(matched["matching_cost"]),
                                **path_diagnostics,
                                **data_metrics,
                                "residual_energy": float(estimates["residual_energy"]),
                                "residual_energy_fraction": float(
                                    estimates["residual_energy"]
                                    / max(total_white_energy, np.finfo(float).tiny)
                                ),
                                "initial_residual_energy": initial_residual,
                                "refinement_nonincreasing": int(
                                    estimates.get(
                                        "refinement_all_stages_nonincreasing",
                                        estimates["residual_energy"]
                                        <= initial_residual * (1.0 + 1.0e-10),
                                    )
                                ),
                                "refinement_stages": int(
                                    estimates.get("refinement_stages", 0)
                                ),
                                "support_condition_number": float(
                                    estimates["support_condition_number"]
                                ),
                                "algorithm_iterations": int(estimates["iterations"]),
                                "algorithm_converged": int(estimates["converged"]),
                                "screened_candidate_count": int(
                                    estimates["screened_candidate_count"]
                                ),
                                "estimator_runtime_seconds": float(elapsed),
                                "search_boundary_hit": int(boundary_hit),
                                "execution_order_index": order_index,
                                "training_symbols": schedule.training_symbols,
                                "pilot_resource_elements": schedule.pilot_resource_elements,
                                "num_configurations": schedule.num_configurations,
                                "num_switches": schedule.num_switches,
                                "switch_guard_symbol_equivalents": switch_guard,
                                "num_rf_outputs": int(
                                    config["acquisition"]["num_rf_outputs"]
                                ),
                                "element_noise_variance": element_variance,
                                "rf_noise_variance": rf_variance,
                            }
                        )
                if search_mode == "beam_center_adaptive":
                    # Drop each scene dictionary before building the next one to
                    # bound peak host and device memory.
                    del dictionary
        notify(f"completed Stage 2-A seed {seed_index}/{len(replicate_seeds)}")

    expected_rows = (
        len(replicate_seeds)
        * samples_per_seed
        * len(platform.schedules)
        * len(snr_values)
        * len(methods)
    )
    if len(rows) != expected_rows:
        raise RuntimeError("Stage 2-A row count does not match the declared design")
    if search_mode == "fixed":
        if search_metadata is None:
            raise RuntimeError("fixed search metadata is missing")
        search_summary = {
            "mode": "fixed",
            "range_grid_points": int(search_metadata["range_grid_m"].size),
            "azimuth_grid_points": int(search_metadata["azimuth_grid_deg"].size),
            "elevation_grid_points": int(
                search_metadata["elevation_grid_deg"].size
            ),
            "num_candidates": search_metadata["num_candidates"],
            "dictionary_build_seconds": search_metadata[
                "dictionary_build_seconds"
            ],
            "dictionary_memory_bytes": search_metadata[
                "dictionary_memory_bytes"
            ],
            "dictionary_compute_backend": search_metadata[
                "dictionary_compute_backend"
            ],
            "dictionary_compute_device": search_metadata[
                "dictionary_compute_device"
            ],
            "dictionary_compute_precision": search_metadata[
                "dictionary_compute_precision"
            ],
            "dictionary_compute_validation": search_metadata[
                "dictionary_compute_validation"
            ],
            "total_dictionary_memory_bytes": int(
                sum(search_metadata["dictionary_memory_bytes"].values())
            ),
            "dictionary_storage_dtype": str(
                config["search"]["dictionary_storage_dtype"]
            ),
        }
    else:
        candidate_counts = np.asarray(
            [item["num_candidates"] for item in adaptive_scene_search], dtype=int
        )
        grid_steps = np.asarray(
            [item["grid_steps_m_rad_rad"] for item in adaptive_scene_search],
            dtype=float,
        )
        nearest_errors = np.asarray(
            [
                item["nearest_union_candidate_abs_error_m_rad_rad"]
                for item in adaptive_scene_search
            ],
            dtype=float,
        )
        if candidate_counts.size != len(replicate_seeds) * samples_per_seed:
            raise RuntimeError("adaptive search scene records are incomplete")
        search_summary = {
            "mode": "beam_center_adaptive",
            "center_source": "pre-observation random coarse path-beam centers",
            "per_path_center_count": int(channel_config["num_paths"]),
            "truth_offsets": "continuous random draws; no forced grid or half-grid placement",
            "candidate_union": "three local Cartesian windows on one global anchored lattice; overlaps deduplicated",
            "candidate_count_min": int(np.min(candidate_counts)),
            "candidate_count_median": float(np.median(candidate_counts)),
            "candidate_count_mean": float(np.mean(candidate_counts)),
            "candidate_count_max": int(np.max(candidate_counts)),
            "grid_step_min_m_deg_deg": [
                float(np.min(grid_steps[:, 0])),
                float(np.rad2deg(np.min(grid_steps[:, 1]))),
                float(np.rad2deg(np.min(grid_steps[:, 2]))),
            ],
            "grid_step_max_m_deg_deg": [
                float(np.max(grid_steps[:, 0])),
                float(np.rad2deg(np.max(grid_steps[:, 1]))),
                float(np.rad2deg(np.max(grid_steps[:, 2]))),
            ],
            "maximum_nearest_candidate_error_m_deg_deg": [
                float(np.max(nearest_errors[:, :, 0])),
                float(np.rad2deg(np.max(nearest_errors[:, :, 1]))),
                float(np.rad2deg(np.max(nearest_errors[:, :, 2]))),
            ],
            "all_truth_paths_inside_associated_windows": bool(
                all(
                    all(item["truth_associated_window_coverage"])
                    for item in adaptive_scene_search
                )
            ),
            "dictionary_build_seconds_total": {
                name: float(np.sum(values))
                for name, values in adaptive_build_seconds.items()
            },
            "dictionary_build_seconds_mean_per_scene": {
                name: float(np.mean(values))
                for name, values in adaptive_build_seconds.items()
            },
            "peak_dictionary_memory_bytes": {
                name: int(np.max(values))
                for name, values in adaptive_memory_bytes.items()
            },
            "dictionary_compute": {
                name: [
                    {
                        "backend": backend,
                        "device": device,
                        "precision": precision,
                    }
                    for backend, device, precision in sorted(values)
                ]
                for name, values in adaptive_compute_metadata.items()
            },
            "dictionary_compute_validation": adaptive_compute_validation,
            "profile_compute": {
                name: {
                    "backend": evaluator.compute_backend,
                    "device": evaluator.compute_device,
                    "precision": evaluator.compute_precision,
                }
                for name, evaluator in adaptive_profile_evaluators.items()
            },
            "schedules_built_sequentially_per_scene": True,
            "dictionary_storage_dtype": str(
                config["search"]["dictionary_storage_dtype"]
            ),
            "settings": {
                key: value
                for key, value in config["adaptive_search"].items()
            },
            "scene_records": adaptive_scene_search,
        }
    summary = {
        "experiment": {
            "name": str(experiment["name"]),
            "status": "Stage 2-A paired 3-D classical-baseline validation",
            "research_decision_enabled": bool(
                experiment["research_decision_enabled"]
            ),
            "diagnostic_only": bool(experiment.get("diagnostic_only", False)),
            "diagnostic_scope": str(experiment.get("diagnostic_scope", "")),
            "replicate_seeds": replicate_seeds,
            "samples_per_seed": samples_per_seed,
            "independent_scene_count": len(replicate_seeds) * samples_per_seed,
            "array_reference_snr_db": snr_values,
            "expected_raw_metric_rows": expected_rows,
            "actual_raw_metric_rows": len(rows),
        },
        "scope": {
            "truth": (
                "three exact spherical THz paths drawn continuously around known pre-observation coarse centers"
                if search_mode == "beam_center_adaptive"
                else "three exact spherical THz paths on the Stage 2 planar receive DMA"
            ),
            "comparison_axis": "paired estimator replacement within each native acquisition schedule",
            "learning_model_used": False,
            "paper_exact_reproduction": False,
            "oracle_is_genie_reference_not_competitor": True,
            "channel_nmse_evaluation_subcarriers": evaluation_indices.tolist(),
            "csi_metric_version": "frequency-domains-v2",
            "csi_metric_space": "uncombined element CSI; sum squared error / sum truth energy",
            "fullband_evaluation_subcarriers": list(range(128)),
            "heldout_empty_domain": "null in JSON / blank in CSV; not zero",
            "search_mode": search_mode,
            "artificial_half_grid_offset_used": False,
            **support_metadata,
        },
        "model_metadata": {
            **platform.metadata,
            "bandwidth_hz": float(config["system"]["bandwidth_hz"]),
            "num_subcarriers": int(config["system"]["num_subcarriers"]),
            "quality_factor": float(config["dma"]["quality_factor"]),
            "absorption_enabled": bool(channel_config["include_absorption"]),
            "absorption_table_path": str(platform.absorption_table.source_path),
        },
        "method_provenance": {method: METHOD_PROVENANCE[method] for method in methods},
        "resource_fairness": {
            "same_observation_across_estimators_within_schedule_snr_scene": True,
            "common_standard_noise_across_snr_levels_within_schedule_scene": True,
            "independent_unit": "one generated multipath scene",
            "paired_block": "schedule, SNR and estimator within scene",
            "method_order_randomized": True,
            "common_pilot_resource_elements": int(
                config["acquisition"]["total_pilot_resource_elements"]
            ),
            "pilot_energy_per_re": float(
                config["acquisition"]["pilot_energy_per_re"]
            ),
            "switch_guard_symbol_equivalents_per_switch": float(
                config["acquisition"]["switch_guard_symbol_equivalents"]
            ),
            "combiner_row_normalization": False,
            "noise_whitening": "known diagonal post-DMA covariance",
        },
        "schedules": [
            {
                "name": schedule.name,
                "family": schedule.family,
                "pilot_indices": schedule.pilot_indices.tolist(),
                "heldout_indices": np.setdiff1d(np.arange(128), schedule.pilot_indices).tolist(),
                "training_symbols": schedule.training_symbols,
                "pilot_resource_elements": schedule.pilot_resource_elements,
                "num_configurations": schedule.num_configurations,
                "num_switches": schedule.num_switches,
            }
            for schedule in platform.schedules
        ],
        "search": search_summary,
        "data_codebook": data_codebook_metadata,
        "timing": {
            method: {
                "calls": calls[method],
                "total_seconds": float(timings[method]),
                "mean_batch1_seconds": float(timings[method] / calls[method]),
            }
            for method in methods
        },
        "execution_schedule": execution_schedule,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return rows, summary
