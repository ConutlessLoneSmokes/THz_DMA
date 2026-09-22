"""Configured planar sparse-estimator construction and dispatch."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from thz_dma.estimators.los_grid import regular_grid as _grid_from_step
from thz_dma.estimators.planar_sparse import (
    PlanarDmaTemplateOperator,
    build_planar_template_dictionary,
    omp_three_path_estimate,
    oracle_geometry_ls_estimate,
    yang_style_ogols_3d_estimate,
)
from thz_dma.observations.planar import (
    build_stage2_platform,
    stage2_schedule_weights,
)
from thz_dma.surfaces.multistrip import combiner_row_energy


def operator_for_schedule(platform, schedule, weights, config) -> PlanarDmaTemplateOperator:
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
        operators[schedule.name] = operator_for_schedule(
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


def estimate_planar_baseline(
    method: str, dictionary, operator, observation, paths, config
) -> dict:
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
