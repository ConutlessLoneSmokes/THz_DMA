"""Stage 2-A paired three-dimensional classical baselines."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from thz_dma.channels.planar import (
    channel_from_planar_estimate as _channel_from_estimate,
    multipath_planar_channel,
    sample_stage2_paths,
)
from thz_dma.estimators.adaptive_grid import (
    adaptive_settings as _adaptive_settings,
    prepare_adaptive_stage2a_scene,
    search_mode as _search_mode,
    validate_search_contains_channel_support as _validate_search_contains_channel_support,
)
from thz_dma.estimators.planar_baselines import (
    _validate_accelerated_dictionary,
    build_stage2a_estimators,
    build_stage2a_operators,
    estimate_planar_baseline as _estimate,
)
from thz_dma.estimators.planar_sparse import (
    PlanarTemplateBatchEvaluator,
    build_planar_template_dictionary_from_candidates,
)
from thz_dma.evaluation.beamforming import (
    build_planar_data_codebook,
    feasible_planar_data_metrics,
)
from thz_dma.evaluation.paths import match_planar_paths as _match_paths
from thz_dma.metrics.diagnostics import channel_domain_metrics
from thz_dma.observations.multistrip import (
    array_input_noise_variance,
    effective_tensor_output_snr_db,
    observe_multistrip_schedule,
)
from thz_dma.observations.pilots import full_band_pilot_indices as _pilot_indices


METHOD_PROVENANCE = {
    "sbl_jointfreq_3d": {
        "label": "Joint-frequency SBL posterior mean",
        "role": "classical sparse Bayesian baseline",
        "source_key": "tipping2003fastsbl",
        "adaptation": "complex known-noise sequential evidence maximisation on the full physical dictionary; all active posterior coefficients reconstruct CSI; three peaks for geometry diagnostics only",
        "paper_exact_reproduction": False,
        "uses_extra_information": False,
    },
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
    "grid_somp_3d": {
        "label": "3-D SOMP support + shared-gain LS",
        "role": "common-support wideband sparse baseline",
        "source_key": "internal_stage2a",
        "adaptation": (
            "noise-whitened per-tone SOMP support selection with independent "
            "selection gains; shared physical gain refit for fullband CSI"
        ),
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


def _method_names(config: dict) -> list[str]:
    methods = [str(item) for item in config["experiment"]["estimators"]]
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("Stage 2-A estimators must be non-empty and unique")
    if set(methods) & {"grid_somp_3d", "ridge"}:
        raise ValueError("SOMP and Tikhonov are retired from new runs; use OMP/Oracle/SBL")
    unknown = sorted(set(methods) - set(METHOD_PROVENANCE))
    if unknown:
        raise ValueError(f"unsupported Stage 2-A estimators: {unknown}")
    if not {"oracle_geometry_ls", "grid_omp_3d"} <= set(methods):
        raise ValueError("Stage 2-A requires Oracle and grid OMP")
    return methods


def run_stage2a(
    config: dict,
    project_root: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
    checkpoint_dir: Path | None = None,
) -> tuple[list[dict], dict]:
    """Run paired 3-D classical baselines on both native acquisition schedules."""

    notify = progress or (lambda _message: None)
    started = time.perf_counter()
    resumed_seconds = 0.0
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
            scene_started = time.perf_counter()
            row_start, order_start = len(rows), len(execution_schedule)
            checkpoint = (checkpoint_dir / f"{seed}_{sample_index}.json"
                          if checkpoint_dir is not None else None)
            if checkpoint is not None and checkpoint.exists():
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                expected = len(platform.schedules) * len(snr_values) * len(methods)
                if len(saved["rows"]) != expected or any(row["scene_id"] != scene_id for row in saved["rows"]):
                    raise ValueError(f"invalid scene checkpoint: {checkpoint}")
                rows.extend(saved["rows"])
                execution_schedule.extend(saved["execution_schedule"])
                scene_rng.bit_generator.state = saved["scene_rng_state"]
                resumed_seconds += saved["runtime_seconds"]
                for row in saved["rows"]:
                    timings[row["estimator"]] += row["estimator_runtime_seconds"]
                    calls[row["estimator"]] += 1
                if search_mode == "beam_center_adaptive":
                    record = saved["adaptive_record"]
                    adaptive_scene_search.append(record)
                    adaptive_compute_validation.update(saved["compute_validation"])
                    for name, meta in record["dictionary_by_schedule"].items():
                        adaptive_build_seconds[name].append(meta["build_seconds"])
                        adaptive_memory_bytes[name].append(meta["memory_bytes"])
                        adaptive_compute_metadata[name].add((meta["compute_backend"], meta["compute_device"], meta["compute_precision"]))
                notify(f"restored {scene_id}: {len(rows)} rows")
                continue
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
                            noise_variance=element_variance,
                        )
                        elapsed = time.perf_counter() - method_started
                        timings[method] += elapsed
                        calls[method] += 1

                        matched = _match_paths(estimates, paths, case_config)
                        reconstruction_started = time.perf_counter()
                        reconstruction_coordinates = estimates.get("reconstruction_coordinates", matched["coordinates"])
                        reconstruction_gains = estimates.get("reconstruction_gains", matched["gains"])
                        estimated_full_channel = _channel_from_estimate(
                            platform.frequencies_hz,
                            platform,
                            reconstruction_coordinates,
                            reconstruction_gains,
                            config,
                        )
                        estimated_center_channel = _channel_from_estimate(
                            center_frequency,
                            platform,
                            reconstruction_coordinates,
                            reconstruction_gains,
                            config,
                        )[0]
                        domain_metrics = channel_domain_metrics(
                            full_channel, estimated_full_channel,
                            schedule.pilot_indices, evaluation_indices,
                        )
                        estimated_clean = np.asarray(reconstruction_gains) @ (
                            operator.evaluate_raw(
                                *reconstruction_coordinates.T,
                            )
                        )
                        clean_vector = observation.clean.reshape(-1)
                        reconstruction_seconds = time.perf_counter() - reconstruction_started
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
                                "sbl_state_json": estimates.get("sbl_state_json", ""),
                                "screened_candidate_count": int(
                                    estimates["screened_candidate_count"]
                                ),
                                "estimator_runtime_seconds": float(elapsed),
                                "channel_reconstruction_seconds": float(reconstruction_seconds),
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
            if checkpoint is not None:
                saved = {
                    "rows": rows[row_start:],
                    "execution_schedule": execution_schedule[order_start:],
                    "scene_rng_state": scene_rng.bit_generator.state,
                    "adaptive_record": adaptive_record if search_mode == "beam_center_adaptive" else None,
                    "compute_validation": adaptive_compute_validation,
                    "runtime_seconds": time.perf_counter() - scene_started,
                }
                temporary = checkpoint.with_suffix(".tmp")
                temporary.write_text(json.dumps(saved, ensure_ascii=False, allow_nan=False), encoding="utf-8")
                temporary.replace(checkpoint)
                notify(f"saved {scene_id}: {len(rows)} rows")
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
        "runtime_seconds": float(time.perf_counter() - started + resumed_seconds),
    }
    return rows, summary
