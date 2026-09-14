"""Stage 1-C paired comparison of literature-aligned classical estimators."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from thz_dma.channels.atmosphere import C0_M_PER_S
from thz_dma.channels.geometry import rayleigh_distance_m
from thz_dma.estimators.literature_baselines import (
    SphericalDmaTemplateOperator,
    ogols_two_path_estimate,
    screened_sbl_two_path_estimate,
)
from thz_dma.estimators.tensor_transfer import (
    tensor_denoised_ogols_two_path_estimate,
)
from thz_dma.estimators.twopath_omp import omp_two_path_estimate
from thz_dma.observations.pilots import (
    combine_pilot_observation,
    complex_standard_normal,
    effective_output_snr_db,
    ideal_array_noise_variance,
    matched_output_noise_variance,
)
from thz_dma.stage1a2 import (
    MeasurementDesign,
    _build_dictionaries,
    _db,
    _grid_from_step,
    _ideal_digital_beam_gain,
    _noise_conditions,
    build_stage1a2_designs,
)
from thz_dma.stage1b import (
    _build_data_codebook,
    _channel_from_paths,
    _feasible_data_metrics,
    _match_two_paths,
    _scenario_samples,
)


METHOD_PROVENANCE = {
    "grid_omp": {
        "label": "Grid OMP",
        "role": "internal lower-complexity sanity baseline",
        "source_key": "internal_stage1b",
        "adaptation": "unchanged two-step grid OMP with joint gain refit",
        "paper_exact_reproduction": False,
    },
    "yang_ogols_adapted": {
        "label": "Yang-style OG-OLS (adapted)",
        "role": "near-field off-grid literature-transfer baseline",
        "source_key": "yang2024xldma",
        "adaptation": (
            "single joint-frequency-vector specialization of OG-DOLS; exact "
            "spherical DMA templates, OLS atom selection, first-order off-grid "
            "refinement; no UPA EL-AZ decoupling or measurement-matrix optimization"
        ),
        "paper_exact_reproduction": False,
    },
    "sbl_jointfreq": {
        "label": "Joint-frequency SBL (adapted)",
        "role": "classical Bayesian sparse-recovery baseline",
        "source_key": "gao2024thzunfolding_parent_sbl",
        "adaptation": (
            "classical complex SBL/EM applied to a deterministic screened joint-"
            "frequency dictionary; not Gao et al.'s MMV AMP-SBL or unfolded network"
        ),
        "paper_exact_reproduction": False,
    },
    "zhang_cpd_ogols_adapted": {
        "label": "Zhang-style CPD -> OG-OLS (adapted)",
        "role": "low-rank frequency-tensor transfer baseline",
        "source_key": "zhang2025tensorofdm",
        "adaptation": (
            "rank-2 block-Hankel SVD/CP denoising of the actual receive-DMA "
            "observation, followed by exact spherical OG-OLS and an original-"
            "observation gain refit; not the source paper's transmit-DMA four-way "
            "CP model, microstrip-sequential training, algebraic parameter "
            "extraction, or uniqueness result"
        ),
        "paper_exact_reproduction": False,
    },
}


def _select_designs(
    designs: list[MeasurementDesign], config: dict
) -> list[MeasurementDesign]:
    requested = [str(value) for value in config["experiment"]["design_families"]]
    selected = [design for design in designs if design.family in requested]
    if len(selected) != len(requested):
        counts = {family: 0 for family in requested}
        for design in selected:
            counts[design.family] += 1
        if any(value != 1 for value in counts.values()):
            raise ValueError(
                "Stage 1-C requires exactly one design for every requested family"
            )
    if len({design.name for design in selected}) != len(selected):
        raise ValueError("Stage 1-C design names must be unique")
    pilot_counts = {design.pilot_resource_elements for design in selected}
    if len(pilot_counts) != 1:
        raise ValueError("Stage 1-C designs must use the same pilot RE count")
    return selected


def _method_names(config: dict) -> list[str]:
    names = [str(value) for value in config["experiment"]["estimators"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Stage 1-C estimators must be non-empty and unique")
    unknown = sorted(set(names) - set(METHOD_PROVENANCE))
    if unknown:
        raise ValueError(f"unsupported Stage 1-C estimators: {unknown}")
    if "grid_omp" not in names:
        raise ValueError("grid_omp must be retained as the paired reference")
    return names


def _template_operators(
    designs: list[MeasurementDesign],
    positions_m: np.ndarray,
    absorption,
    *,
    include_absorption: bool,
    interpolation: str,
) -> dict[str, SphericalDmaTemplateOperator]:
    operators = {}
    for design in designs:
        if include_absorption:
            coefficient = np.asarray(
                absorption.coefficient(
                    design.frequencies_hz, method=interpolation
                ),
                dtype=float,
            )
        else:
            coefficient = np.zeros(design.frequencies_hz.size, dtype=float)
        operators[design.name] = SphericalDmaTemplateOperator(
            frequencies_hz=design.frequencies_hz,
            element_positions_m=positions_m,
            weights=design.weights,
            absorption_coefficient_per_m=coefficient,
        )
    return operators


def _estimate(
    method: str,
    dictionary,
    observation: np.ndarray,
    noise_variance: float | None,
    template_operator: SphericalDmaTemplateOperator,
    design: MeasurementDesign,
    config: dict,
) -> dict:
    estimator = config["estimator"]
    exclusion_range = float(estimator["exclusion_range_m"])
    exclusion_angle = float(np.deg2rad(estimator["exclusion_angle_deg"]))
    if method == "grid_omp":
        result = omp_two_path_estimate(
            dictionary,
            observation,
            exclusion_range_m=exclusion_range,
            exclusion_angle_rad=exclusion_angle,
        )
        return {
            **result,
            "iterations": 1,
            "converged": 1,
            "screened_candidate_count": dictionary.num_candidates,
        }
    if method == "yang_ogols_adapted":
        settings = estimator["yang_ogols_adapted"]
        search = config["search"]
        return ogols_two_path_estimate(
            dictionary,
            observation,
            template_operator.evaluate,
            range_bounds_m=(
                float(search["range_min_m"]),
                float(search["range_max_m"]),
            ),
            angle_bounds_rad=(
                float(np.deg2rad(search["angle_min_deg"])),
                float(np.deg2rad(search["angle_max_deg"])),
            ),
            range_grid_step_m=float(search["range_step_m"]),
            angle_grid_step_rad=float(np.deg2rad(search["angle_step_deg"])),
            exclusion_range_m=exclusion_range,
            exclusion_angle_rad=exclusion_angle,
            refinement_iterations=int(settings["refinement_iterations"]),
            derivative_fraction=float(settings["derivative_fraction"]),
            maximum_update_grid_units=float(
                settings["maximum_update_grid_units"]
            ),
            convergence_tolerance=float(settings["convergence_tolerance"]),
        )
    if method == "sbl_jointfreq":
        settings = estimator["sbl_jointfreq"]
        return screened_sbl_two_path_estimate(
            dictionary,
            observation,
            noise_variance=noise_variance,
            screening_size=int(settings["screening_size"]),
            max_iterations=int(settings["max_iterations"]),
            convergence_tolerance=float(settings["convergence_tolerance"]),
            em_damping=float(settings["em_damping"]),
            noise_floor_fraction=float(settings["noise_floor_fraction"]),
            exclusion_range_m=exclusion_range,
            exclusion_angle_rad=exclusion_angle,
        )
    if method == "zhang_cpd_ogols_adapted":
        tensor_settings = estimator["zhang_cpd_ogols_adapted"]
        offgrid_settings = estimator["yang_ogols_adapted"]
        search = config["search"]
        return tensor_denoised_ogols_two_path_estimate(
            dictionary,
            observation,
            template_operator.evaluate,
            frequencies_hz=design.frequencies_hz,
            configuration_index=design.configuration_index,
            range_bounds_m=(
                float(search["range_min_m"]),
                float(search["range_max_m"]),
            ),
            angle_bounds_rad=(
                float(np.deg2rad(search["angle_min_deg"])),
                float(np.deg2rad(search["angle_max_deg"])),
            ),
            range_grid_step_m=float(search["range_step_m"]),
            angle_grid_step_rad=float(np.deg2rad(search["angle_step_deg"])),
            exclusion_range_m=exclusion_range,
            exclusion_angle_rad=exclusion_angle,
            refinement_iterations=int(offgrid_settings["refinement_iterations"]),
            derivative_fraction=float(offgrid_settings["derivative_fraction"]),
            maximum_update_grid_units=float(
                offgrid_settings["maximum_update_grid_units"]
            ),
            convergence_tolerance=float(
                offgrid_settings["convergence_tolerance"]
            ),
            tensor_rank=int(tensor_settings["rank"]),
            hankel_rows=int(tensor_settings["hankel_rows"]),
            cp_max_iterations=int(tensor_settings["cp_max_iterations"]),
            cp_convergence_tolerance=float(
                tensor_settings["cp_convergence_tolerance"]
            ),
            cp_ridge=float(tensor_settings["cp_ridge"]),
        )
    raise ValueError(f"unsupported estimator: {method}")


def run_stage1c(
    config: dict,
    project_root: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict]:
    """Run paired estimator comparisons on fixed THz DMA observations."""

    notify = progress or (lambda _message: None)
    started = time.perf_counter()
    root = Path(project_root).resolve()
    experiment = config["experiment"]
    system = config["system"]
    channel_config = config["channel"]
    search = config["search"]
    atmosphere_config = config["atmosphere"]
    resources = config["resources"]
    evaluation = config["evaluation"]
    methods = _method_names(config)

    if not np.isclose(float(resources["pilot_energy_per_re"]), 1.0):
        raise ValueError("Stage 1-C currently requires pilot_energy_per_re = 1.0")
    if int(resources["num_rf_outputs"]) != 1:
        raise ValueError("Stage 1-C currently implements one RF output")

    frequencies, positions, absorption, all_designs, design_metadata = (
        build_stage1a2_designs(config, root)
    )
    designs = _select_designs(all_designs, config)
    ranges = _grid_from_step(
        float(search["range_min_m"]),
        float(search["range_max_m"]),
        float(search["range_step_m"]),
    )
    angles_deg = _grid_from_step(
        float(search["angle_min_deg"]),
        float(search["angle_max_deg"]),
        float(search["angle_step_deg"]),
    )
    dictionaries, dictionary_timings = _build_dictionaries(
        designs,
        ["spherical"],
        positions,
        ranges,
        angles_deg,
        absorption,
        include_absorption=bool(channel_config["include_absorption"]),
        interpolation=str(atmosphere_config["interpolation"]),
        chunk_size=int(search["dictionary_chunk_size"]),
        notify=notify,
    )
    operators = _template_operators(
        designs,
        positions,
        absorption,
        include_absorption=bool(channel_config["include_absorption"]),
        interpolation=str(atmosphere_config["interpolation"]),
    )
    data_codebook, data_codebook_metadata = _build_data_codebook(config, positions)
    conditions = _noise_conditions(config["noise"])
    scenarios = list(config["scenarios"])
    if not scenarios or len({item["name"] for item in scenarios}) != len(scenarios):
        raise ValueError("Stage 1-C scenarios must be non-empty and uniquely named")
    replicate_seeds = [int(value) for value in experiment["replicate_seeds"]]
    samples_per_seed = int(experiment["samples_per_seed"])
    if not replicate_seeds or samples_per_seed < 1:
        raise ValueError("replicate seeds and samples_per_seed cannot be empty")

    include_absorption = bool(channel_config["include_absorption"])
    interpolation = str(atmosphere_config["interpolation"])
    carrier_hz = float(system["carrier_frequency_hz"])
    center_frequency = np.array([carrier_hz])
    range_match_scale = float(evaluation["path_match_range_scale_m"])
    angle_match_scale = float(evaluation["path_match_angle_scale_deg"])
    range_tolerance = float(evaluation["path_success_range_tolerance_m"])
    angle_tolerance = float(evaluation["path_success_angle_tolerance_deg"])
    data_input_snr_db = float(config["data_evaluation"]["input_snr_db"])

    rows: list[dict] = []
    execution_schedule: list[dict] = []
    estimator_timings = {
        f"{design.name}|{method}": 0.0
        for design in designs
        for method in methods
    }
    estimator_calls = {key: 0 for key in estimator_timings}
    scenario_truth_summary: dict[str, dict] = {}

    for scenario_index, scenario in enumerate(scenarios):
        scenario_name = str(scenario["name"])
        received_reflection_ratios = []
        for seed_index, seed in enumerate(replicate_seeds, start=1):
            rng = np.random.default_rng(seed + scenario_index * 1_000_003)
            scene = _scenario_samples(
                rng, channel_config, scenario, samples_per_seed
            )
            full_channels = []
            center_channels = []
            standard_noise = []
            for sample_index in range(samples_per_seed):
                full, path_channels = _channel_from_paths(
                    frequencies,
                    positions,
                    scene["ranges_m"][sample_index],
                    scene["angles_deg"][sample_index],
                    scene["gains"][sample_index],
                    absorption=absorption,
                    include_absorption=include_absorption,
                    interpolation=interpolation,
                )
                center, _ = _channel_from_paths(
                    center_frequency,
                    positions,
                    scene["ranges_m"][sample_index],
                    scene["angles_deg"][sample_index],
                    scene["gains"][sample_index],
                    absorption=absorption,
                    include_absorption=include_absorption,
                    interpolation=interpolation,
                )
                full_channels.append(full)
                center_channels.append(center[0])
                standard_noise.append(
                    complex_standard_normal(
                        rng, (frequencies.size, positions.size)
                    )
                )
                los_component = scene["gains"][sample_index, 0] * path_channels[0]
                reflection_component = (
                    scene["gains"][sample_index, 1] * path_channels[1]
                )
                received_reflection_ratios.append(
                    _db(
                        np.sum(np.abs(reflection_component) ** 2)
                        / np.sum(np.abs(los_component) ** 2)
                    )
                )

            measurement_standard_noise: dict[str, np.ndarray] = {}
            for design_index, design in enumerate(designs):
                if np.unique(design.pilot_indices).size == design.pilot_indices.size:
                    measurement_standard_noise[design.name] = np.stack(
                        [
                            standard_noise[sample_index][design.pilot_indices]
                            for sample_index in range(samples_per_seed)
                        ],
                        axis=0,
                    )
                else:
                    repeated_noise_rng = np.random.default_rng(
                        seed
                        + scenario_index * 1_000_003
                        + (design_index + 1) * 10_000_019
                        + 97_531
                    )
                    measurement_standard_noise[design.name] = (
                        complex_standard_normal(
                            repeated_noise_rng,
                            (
                                samples_per_seed,
                                design.pilot_resource_elements,
                                positions.size,
                            ),
                        )
                    )

            for condition_index, condition in enumerate(conditions):
                observations: dict[str, np.ndarray] = {}
                diagnostics: dict[str, list[dict]] = {}
                for design in designs:
                    values = np.empty(
                        (samples_per_seed, design.pilot_resource_elements),
                        dtype=complex,
                    )
                    design_diagnostics = []
                    for sample_index in range(samples_per_seed):
                        pilot_channel = full_channels[sample_index][
                            design.pilot_indices
                        ]
                        clean = combine_pilot_observation(
                            design.weights, pilot_channel
                        )
                        reference_power = float(
                            np.mean(
                                np.sum(
                                    np.abs(full_channels[sample_index]) ** 2,
                                    axis=1,
                                )
                            )
                        )
                        if condition["mode"] == "input_array":
                            noise_variance = ideal_array_noise_variance(
                                full_channels[sample_index],
                                float(condition["target_snr_db"]),
                            )
                        elif condition["mode"] == "matched_output":
                            noise_variance = matched_output_noise_variance(
                                clean, float(condition["target_snr_db"])
                            )
                        else:
                            noise_variance = None
                        if noise_variance is None:
                            observation = clean
                            output_snr_db = None
                            input_snr_db = None
                        else:
                            noise = (
                                np.sqrt(noise_variance)
                                * measurement_standard_noise[design.name][
                                    sample_index
                                ]
                            )
                            observation = combine_pilot_observation(
                                design.weights,
                                pilot_channel,
                                element_noise=noise,
                            )
                            output_snr_db = effective_output_snr_db(
                                clean, noise_variance
                            )
                            input_snr_db = float(
                                10.0
                                * np.log10(reference_power / noise_variance)
                            )
                        values[sample_index] = observation
                        design_diagnostics.append(
                            {
                                "noise_variance": noise_variance,
                                "effective_output_snr_db": output_snr_db,
                                "defined_input_array_snr_db": input_snr_db,
                            }
                        )
                    observations[design.name] = values
                    diagnostics[design.name] = design_diagnostics

                for sample_index in range(samples_per_seed):
                    combinations = [
                        (design, method)
                        for design in designs
                        for method in methods
                    ]
                    order_rng = np.random.default_rng(
                        int(experiment["randomization_seed"])
                        + seed * 1009
                        + scenario_index * 65537
                        + condition_index * 9176
                        + sample_index * 131
                    )
                    execution_order = [
                        combinations[index]
                        for index in order_rng.permutation(len(combinations))
                    ]
                    execution_schedule.append(
                        {
                            "scenario": scenario_name,
                            "replicate_seed": seed,
                            "sample_index": sample_index,
                            "snr_condition": condition["label"],
                            "method_order": [
                                f"{design.name}|{method}"
                                for design, method in execution_order
                            ],
                        }
                    )

                    for order_index, (design, method) in enumerate(
                        execution_order, start=1
                    ):
                        diagnostic = diagnostics[design.name][sample_index]
                        estimate_started = time.perf_counter()
                        estimates = _estimate(
                            method,
                            dictionaries[(design.name, "spherical")],
                            observations[design.name][sample_index],
                            diagnostic["noise_variance"],
                            operators[design.name],
                            design,
                            config,
                        )
                        elapsed = time.perf_counter() - estimate_started
                        timing_key = f"{design.name}|{method}"
                        estimator_timings[timing_key] += elapsed
                        estimator_calls[timing_key] += 1

                        estimated_ranges = np.asarray(
                            estimates["range_m"], dtype=float
                        )
                        estimated_angles = np.rad2deg(
                            np.asarray(estimates["angle_rad"], dtype=float)
                        )
                        estimated_gains = np.asarray(
                            estimates["complex_gain"], dtype=complex
                        )
                        matched_ranges, matched_angles, matched_gains = (
                            _match_two_paths(
                                estimated_ranges,
                                estimated_angles,
                                estimated_gains,
                                scene["ranges_m"][sample_index],
                                scene["angles_deg"][sample_index],
                                range_scale_m=range_match_scale,
                                angle_scale_deg=angle_match_scale,
                            )
                        )
                        reconstructed, _ = _channel_from_paths(
                            frequencies,
                            positions,
                            matched_ranges,
                            matched_angles,
                            matched_gains,
                            absorption=absorption,
                            include_absorption=include_absorption,
                            interpolation=interpolation,
                        )
                        estimated_center, _ = _channel_from_paths(
                            center_frequency,
                            positions,
                            matched_ranges,
                            matched_angles,
                            matched_gains,
                            absorption=absorption,
                            include_absorption=include_absorption,
                            interpolation=interpolation,
                        )
                        truth = full_channels[sample_index]
                        nmse = float(
                            np.sum(np.abs(reconstructed - truth) ** 2)
                            / np.sum(np.abs(truth) ** 2)
                        )
                        true_ranges = scene["ranges_m"][sample_index]
                        true_angles = scene["angles_deg"][sample_index]
                        range_errors = matched_ranges - true_ranges
                        angle_errors = matched_angles - true_angles
                        per_path_success = (
                            (np.abs(range_errors) <= range_tolerance)
                            & (np.abs(angle_errors) <= angle_tolerance)
                        )
                        true_center = center_channels[sample_index]
                        ideal_beam_gain = _ideal_digital_beam_gain(
                            true_center, estimated_center[0]
                        )
                        data_metrics = _feasible_data_metrics(
                            true_center,
                            estimated_center[0],
                            data_codebook,
                            data_input_snr_db,
                        )
                        boundary_hit = bool(
                            np.any(np.isclose(matched_ranges, ranges[0]))
                            or np.any(np.isclose(matched_ranges, ranges[-1]))
                            or np.any(np.isclose(matched_angles, angles_deg[0]))
                            or np.any(np.isclose(matched_angles, angles_deg[-1]))
                        )
                        total_observation_energy = float(
                            np.sum(
                                np.abs(observations[design.name][sample_index]) ** 2
                            )
                        )
                        rows.append(
                            {
                                "scenario": scenario_name,
                                "replicate_seed": seed,
                                "sample_index": sample_index,
                                "snr_condition": condition["label"],
                                "noise_mode": condition["mode"],
                                "target_snr_db": condition["target_snr_db"],
                                "design": design.name,
                                "design_family": design.family,
                                "estimator": method,
                                "method": method,
                                "method_source_key": METHOD_PROVENANCE[method][
                                    "source_key"
                                ],
                                "true_los_range_m": float(true_ranges[0]),
                                "estimated_los_range_m": float(matched_ranges[0]),
                                "los_range_error_m": float(range_errors[0]),
                                "true_los_angle_deg": float(true_angles[0]),
                                "estimated_los_angle_deg": float(matched_angles[0]),
                                "los_angle_error_deg": float(angle_errors[0]),
                                "true_reflection_range_m": float(true_ranges[1]),
                                "estimated_reflection_range_m": float(
                                    matched_ranges[1]
                                ),
                                "reflection_range_error_m": float(range_errors[1]),
                                "true_reflection_angle_deg": float(true_angles[1]),
                                "estimated_reflection_angle_deg": float(
                                    matched_angles[1]
                                ),
                                "reflection_angle_error_deg": float(
                                    angle_errors[1]
                                ),
                                "true_reflection_gain_relative_db": float(
                                    scene["reflection_gain_relative_db"][sample_index]
                                ),
                                "estimated_reflection_gain_relative_db": float(
                                    20.0
                                    * np.log10(
                                        max(
                                            abs(matched_gains[1])
                                            / max(
                                                abs(matched_gains[0]),
                                                np.finfo(float).tiny,
                                            ),
                                            np.finfo(float).tiny,
                                        )
                                    )
                                ),
                                "los_path_success": int(per_path_success[0]),
                                "reflection_path_success": int(per_path_success[1]),
                                "both_paths_success": int(np.all(per_path_success)),
                                "channel_nmse_linear": nmse,
                                "channel_nmse_db": _db(nmse),
                                "ideal_digital_beam_gain_linear": ideal_beam_gain,
                                "ideal_digital_beam_loss_db": -_db(ideal_beam_gain),
                                **data_metrics,
                                "effective_output_snr_db": diagnostic[
                                    "effective_output_snr_db"
                                ],
                                "defined_input_array_snr_db": diagnostic[
                                    "defined_input_array_snr_db"
                                ],
                                "element_noise_variance": diagnostic[
                                    "noise_variance"
                                ],
                                "explained_energy": float(
                                    estimates["explained_energy"]
                                ),
                                "residual_energy": float(
                                    estimates["residual_energy"]
                                ),
                                "residual_energy_fraction": float(
                                    estimates["residual_energy"]
                                    / max(
                                        total_observation_energy,
                                        np.finfo(float).tiny,
                                    )
                                ),
                                "selected_pair_condition_number": float(
                                    estimates["pair_condition_number"]
                                ),
                                "algorithm_iterations": int(
                                    estimates.get("iterations", 1)
                                ),
                                "algorithm_converged": int(
                                    estimates.get("converged", 1)
                                ),
                                "screened_candidate_count": int(
                                    estimates.get(
                                        "screened_candidate_count",
                                        dictionaries[
                                            (design.name, "spherical")
                                        ].num_candidates,
                                    )
                                ),
                                "offgrid_iterations": estimates.get(
                                    "offgrid_iterations"
                                ),
                                "offgrid_converged": estimates.get(
                                    "offgrid_converged"
                                ),
                                "tensor_rank_residual_fraction": estimates.get(
                                    "tensor_rank_residual_fraction"
                                ),
                                "tensor_denoise_change_fraction": estimates.get(
                                    "tensor_denoise_change_fraction"
                                ),
                                "tensor_factorization_order": estimates.get(
                                    "tensor_factorization_order"
                                ),
                                "tensor_shape_0": estimates.get("tensor_shape_0"),
                                "tensor_shape_1": estimates.get("tensor_shape_1"),
                                "tensor_shape_2": estimates.get("tensor_shape_2"),
                                "tensor_frequency_grid_aligned": estimates.get(
                                    "tensor_frequency_grid_aligned"
                                ),
                                "estimator_runtime_seconds": float(elapsed),
                                "search_boundary_hit": int(boundary_hit),
                                "execution_order_index": order_index,
                                "training_symbols": design.training_symbols,
                                "pilot_resource_elements": (
                                    design.pilot_resource_elements
                                ),
                                "num_configurations": design.num_configurations,
                                "num_switches": design.num_switches,
                                "switch_guard_symbol_equivalents": float(
                                    design.num_switches
                                    * float(
                                        resources[
                                            "switch_guard_symbol_equivalents"
                                        ]
                                    )
                                ),
                                "num_rf_outputs": int(resources["num_rf_outputs"]),
                            }
                        )
                notify(
                    f"completed {scenario_name}, seed "
                    f"{seed_index}/{len(replicate_seeds)}, "
                    f"condition {condition['label']}"
                )

        scenario_truth_summary[scenario_name] = {
            "received_reflection_to_los_power_db_mean": float(
                np.mean(received_reflection_ratios)
            ),
            "received_reflection_to_los_power_db_range": [
                float(np.min(received_reflection_ratios)),
                float(np.max(received_reflection_ratios)),
            ],
        }

    aperture_m = float(np.ptp(positions))
    expected_rows = (
        len(scenarios)
        * len(replicate_seeds)
        * samples_per_seed
        * len(conditions)
        * len(designs)
        * len(methods)
    )
    timing_summary = {
        key: {
            "total_seconds": float(value),
            "calls": int(estimator_calls[key]),
            "mean_batch1_seconds": float(value / estimator_calls[key]),
        }
        for key, value in estimator_timings.items()
    }
    summary = {
        "experiment": {
            "name": str(experiment["name"]),
            "status": "Stage 1-C paired literature-method transfer screen",
            "replicate_seeds": replicate_seeds,
            "samples_per_seed": samples_per_seed,
            "independent_scene_count_per_scenario": (
                len(replicate_seeds) * samples_per_seed
            ),
            "scenarios": [str(item["name"]) for item in scenarios],
            "noise_conditions": conditions,
            "expected_raw_metric_rows": expected_rows,
            "actual_raw_metric_rows": len(rows),
        },
        "scope": {
            "truth": "two exact spherical paths: LoS plus one reflection",
            "comparison_axis": (
                "paired estimator replacement on identical observations within design"
            ),
            "primary_design_family": str(
                config["analysis"]["primary_design_family"]
            ),
            "resource_reference_design_families": [
                str(value)
                for value in config["analysis"].get(
                    "resource_reference_design_families", []
                )
            ],
            "learning_model_used": False,
            "paper_exact_reproduction": False,
        },
        "method_provenance": {
            method: METHOD_PROVENANCE[method] for method in methods
        },
        "resource_fairness": {
            "designs": [
                {
                    "name": design.name,
                    "family": design.family,
                    "training_symbols": design.training_symbols,
                    "pilot_resource_elements": design.pilot_resource_elements,
                    "num_configurations": design.num_configurations,
                    "num_switches": design.num_switches,
                }
                for design in designs
            ],
            "same_observation_across_estimators_within_design": True,
            "pilot_energy_per_re": float(resources["pilot_energy_per_re"]),
            "num_rf_outputs": int(resources["num_rf_outputs"]),
            "combiner_row_norm": 1.0,
        },
        "estimator": {
            "methods": methods,
            "range_grid_points": int(ranges.size),
            "angle_grid_points": int(angles_deg.size),
            "num_candidates": int(ranges.size * angles_deg.size),
            "range_resolution_m": float(search["range_step_m"]),
            "angle_resolution_deg": float(search["angle_step_deg"]),
            "dictionary_build_seconds": dictionary_timings,
            "batch1_timing": timing_summary,
        },
        "model_metadata": {
            "carrier_frequency_hz": carrier_hz,
            "bandwidth_hz": float(system["bandwidth_hz"]),
            "system_subcarriers": int(frequencies.size),
            "num_elements": int(positions.size),
            "element_spacing_m": float(
                config["dma"]["spacing_wavelengths"]
                * C0_M_PER_S
                / carrier_hz
            ),
            "aperture_m": aperture_m,
            "rayleigh_distance_m": float(rayleigh_distance_m(carrier_hz, positions)),
            "absorption_enabled": include_absorption,
            "absorption_table_path": str(absorption.source_path),
            "measurement_design": design_metadata,
            "data_codebook": data_codebook_metadata,
        },
        "scenario_truth_summary": scenario_truth_summary,
        "execution_schedule": execution_schedule,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    if len(rows) != expected_rows:
        raise RuntimeError("Stage 1-C row count does not match the declared design")
    return rows, summary
