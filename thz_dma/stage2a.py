"""Stage 2-A paired three-dimensional classical baselines."""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Callable

import numpy as np

from thz_dma.channels.planar import PlanarPath, multipath_planar_channel
from thz_dma.estimators.planar_sparse import (
    PlanarDmaTemplateOperator,
    build_planar_template_dictionary,
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


def build_stage2a_estimators(config: dict, project_root: str | Path, *, progress=None):
    """Build the common platform, schedule operators and coarse 3-D dictionaries."""

    notify = progress or (lambda _message: None)
    platform = build_stage2_platform(config, project_root)
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
    weights = {}
    operators = {}
    dictionaries = {}
    build_seconds = {}
    dictionary_memory_bytes = {}
    for schedule in platform.schedules:
        weights[schedule.name] = stage2_schedule_weights(
            platform, schedule, config, "thz_physical"
        )
        operators[schedule.name] = _operator_for_schedule(
            platform, schedule, weights[schedule.name], config
        )
        started = time.perf_counter()
        dictionaries[schedule.name] = build_planar_template_dictionary(
            operators[schedule.name],
            ranges,
            np.deg2rad(azimuths_deg),
            np.deg2rad(elevations_deg),
            chunk_size=int(search["dictionary_chunk_size"]),
            storage_dtype=str(search["dictionary_storage_dtype"]),
            progress=lambda message, name=schedule.name: notify(f"{name}: {message}"),
        )
        build_seconds[schedule.name] = float(time.perf_counter() - started)
        dictionary_memory_bytes[schedule.name] = int(
            dictionaries[schedule.name].templates.nbytes
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
    (
        platform,
        schedule_weights,
        operators,
        dictionaries,
        search_metadata,
    ) = build_stage2a_estimators(config, project_root, progress=notify)
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
    replicate_seeds = [int(value) for value in experiment["replicate_seeds"]]
    samples_per_seed = int(experiment["samples_per_seed"])
    if not replicate_seeds or samples_per_seed < 1:
        raise ValueError("replicate seeds and samples_per_seed must be non-empty")

    for seed_index, seed in enumerate(replicate_seeds, start=1):
        scene_rng = np.random.default_rng(seed)
        for sample_index in range(samples_per_seed):
            scene_id = f"{seed}:{sample_index}"
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
                dictionary = dictionaries[schedule.name]
                pilot_channel = full_channel[schedule.pilot_indices]
                for snr_index, snr_db in enumerate(snr_values):
                    element_variance = array_input_noise_variance(full_channel, snr_db)
                    rf_variance = element_variance * float(
                        noise_config["rf_noise_variance_fraction_of_element_noise"]
                    )
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
                            method, dictionary, operator, whitened, paths, config
                        )
                        elapsed = time.perf_counter() - method_started
                        timings[method] += elapsed
                        calls[method] += 1

                        matched = _match_paths(estimates, paths, config)
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
            "truth": "three exact spherical THz paths on the Stage 2 planar receive DMA",
            "comparison_axis": "paired estimator replacement within each native acquisition schedule",
            "learning_model_used": False,
            "paper_exact_reproduction": False,
            "oracle_is_genie_reference_not_competitor": True,
            "channel_nmse_evaluation_subcarriers": evaluation_indices.tolist(),
            "csi_metric_version": "frequency-domains-v2",
            "csi_metric_space": "uncombined element CSI; sum squared error / sum truth energy",
            "fullband_evaluation_subcarriers": list(range(128)),
            "heldout_empty_domain": "null in JSON / blank in CSV; not zero",
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
        "search": {
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
            "total_dictionary_memory_bytes": int(
                sum(search_metadata["dictionary_memory_bytes"].values())
            ),
            "dictionary_storage_dtype": str(
                config["search"]["dictionary_storage_dtype"]
            ),
        },
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
