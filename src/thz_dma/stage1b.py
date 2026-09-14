"""Stage 1-B minimal sufficient near-field LoS-plus-reflection experiment."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from thz_dma.channels.atmosphere import AbsorptionTable, C0_M_PER_S
from thz_dma.channels.geometry import los_element_channel, rayleigh_distance_m
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
from thz_dma.surfaces.dma import (
    dma_weight_matrix,
    frequency_tiled_resonances,
    row_normalize,
)


def _build_data_codebook(
    config: dict,
    positions_m: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Build an independent feasible center-frequency DMA data codebook."""

    system = config["system"]
    hardware = config["dma"]
    data_config = config["data_evaluation"]
    carrier_hz = float(system["carrier_frequency_hz"])
    spacing_m = float(hardware["spacing_wavelengths"]) * C0_M_PER_S / carrier_hz
    feed_positions = np.arange(positions_m.size, dtype=float) * spacing_m
    count = int(data_config["codebook_size"])
    seed_start = int(data_config["codebook_seed_start"])
    if count < 2:
        raise ValueError("data codebook must contain at least two states")

    rows = []
    for offset in range(count):
        resonances = frequency_tiled_resonances(
            positions_m.size,
            carrier_hz,
            float(hardware["tuning_bandwidth_hz"]),
            seed=seed_start + offset,
        )
        response = dma_weight_matrix(
            np.array([carrier_hz]),
            resonances,
            feed_positions,
            quality_factor=float(hardware["quality_factor"]),
            reference_frequency_hz=carrier_hz,
            waveguide_effective_index=float(hardware["waveguide_effective_index"]),
            waveguide_end_power_fraction=float(
                hardware["waveguide_end_power_fraction"]
            ),
        )
        rows.append(row_normalize(response)[0])
    return np.asarray(rows, dtype=complex), {
        "type": "independent feasible center-frequency DMA states",
        "size": count,
        "seed_start": seed_start,
        "selection_rule": "maximize predicted center-frequency output power",
    }


def _validate_minimal_design(designs: list[MeasurementDesign]) -> None:
    signatures = {
        (design.family, design.num_configurations) for design in designs
    }
    expected = {
        ("physical_frequency_response", 1),
        ("physical_time_scan", 2),
        ("flat_timescan", 4),
    }
    if signatures != expected or len(designs) != 3:
        raise ValueError(
            "Stage 1-B requires exactly physical J1, physical J2, and flat J4"
        )
    pilot_counts = {design.pilot_resource_elements for design in designs}
    if len(pilot_counts) != 1:
        raise ValueError("all Stage 1-B designs must use the same pilot RE count")


def _scenario_samples(
    rng: np.random.Generator,
    channel_config: dict,
    scenario: dict,
    count: int,
) -> dict[str, np.ndarray]:
    los_range = rng.uniform(
        float(channel_config["range_min_m"]),
        float(channel_config["range_max_m"]),
        size=count,
    )
    los_angle = rng.uniform(
        float(channel_config["angle_min_deg"]),
        float(channel_config["angle_max_deg"]),
        size=count,
    )
    excess_range = rng.uniform(
        float(scenario["reflection_excess_range_min_m"]),
        float(scenario["reflection_excess_range_max_m"]),
        size=count,
    )
    angle_separation = rng.uniform(
        float(scenario["reflection_angle_separation_min_deg"]),
        float(scenario["reflection_angle_separation_max_deg"]),
        size=count,
    )
    angle_sign = rng.choice(np.array([-1.0, 1.0]), size=count)
    reflection_angle = los_angle + angle_sign * angle_separation
    relative_gain_db = rng.uniform(
        float(scenario["reflection_gain_relative_min_db"]),
        float(scenario["reflection_gain_relative_max_db"]),
        size=count,
    )
    path_phase = rng.uniform(-np.pi, np.pi, size=(count, 2))
    los_magnitude = float(channel_config["los_complex_gain_magnitude"])
    gains = np.empty((count, 2), dtype=complex)
    gains[:, 0] = los_magnitude * np.exp(1j * path_phase[:, 0])
    gains[:, 1] = (
        los_magnitude
        * 10.0 ** (relative_gain_db / 20.0)
        * np.exp(1j * path_phase[:, 1])
    )
    return {
        "ranges_m": np.column_stack((los_range, los_range + excess_range)),
        "angles_deg": np.column_stack((los_angle, reflection_angle)),
        "gains": gains,
        "reflection_gain_relative_db": relative_gain_db,
    }


def _match_two_paths(
    estimated_ranges: np.ndarray,
    estimated_angles_deg: np.ndarray,
    estimated_gains: np.ndarray,
    true_ranges: np.ndarray,
    true_angles_deg: np.ndarray,
    *,
    range_scale_m: float,
    angle_scale_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if range_scale_m <= 0.0 or angle_scale_deg <= 0.0:
        raise ValueError("path matching scales must be positive")
    direct = np.sum(
        ((estimated_ranges - true_ranges) / range_scale_m) ** 2
        + ((estimated_angles_deg - true_angles_deg) / angle_scale_deg) ** 2
    )
    swapped_ranges = estimated_ranges[::-1]
    swapped_angles = estimated_angles_deg[::-1]
    swapped = np.sum(
        ((swapped_ranges - true_ranges) / range_scale_m) ** 2
        + ((swapped_angles - true_angles_deg) / angle_scale_deg) ** 2
    )
    if swapped < direct:
        return swapped_ranges, swapped_angles, estimated_gains[::-1]
    return estimated_ranges, estimated_angles_deg, estimated_gains


def _feasible_data_metrics(
    true_channel: np.ndarray,
    estimated_channel: np.ndarray,
    data_codebook: np.ndarray,
    input_snr_db: float,
) -> dict[str, float | int]:
    true_channel = np.asarray(true_channel, dtype=complex).ravel()
    estimated_channel = np.asarray(estimated_channel, dtype=complex).ravel()
    predicted_power = np.abs(np.conjugate(data_codebook) @ estimated_channel) ** 2
    true_power = np.abs(np.conjugate(data_codebook) @ true_channel) ** 2
    selected_index = int(np.argmax(predicted_power))
    oracle_index = int(np.argmax(true_power))
    selected_power = float(true_power[selected_index])
    oracle_power = float(true_power[oracle_index])
    digital_power = float(np.vdot(true_channel, true_channel).real)
    if digital_power <= 0.0 or oracle_power <= 0.0:
        raise ValueError("data channel or feasible codebook has zero gain")
    input_snr = 10.0 ** (float(input_snr_db) / 10.0)
    selected_snr = input_snr * selected_power / digital_power
    oracle_snr = input_snr * oracle_power / digital_power
    ratio = float(np.clip(selected_power / oracle_power, 0.0, 1.0))
    return {
        "selected_data_state_index": selected_index,
        "oracle_data_state_index": oracle_index,
        "data_state_selection_correct": int(selected_index == oracle_index),
        "feasible_beam_gain_ratio_linear": ratio,
        "feasible_beam_loss_db": -_db(ratio),
        "oracle_feasible_fraction_of_digital_linear": float(
            np.clip(oracle_power / digital_power, 0.0, 1.0)
        ),
        "feasible_gross_rate_bps_hz": float(np.log2(1.0 + selected_snr)),
        "oracle_feasible_gross_rate_bps_hz": float(
            np.log2(1.0 + oracle_snr)
        ),
    }


def _channel_from_paths(
    frequencies_hz: np.ndarray,
    positions_m: np.ndarray,
    ranges_m: np.ndarray,
    angles_deg: np.ndarray,
    gains: np.ndarray,
    *,
    absorption: AbsorptionTable,
    include_absorption: bool,
    interpolation: str,
) -> tuple[np.ndarray, list[np.ndarray]]:
    path_channels = []
    total = np.zeros(
        (frequencies_hz.size, positions_m.size), dtype=complex
    )
    for path_index in range(2):
        channel = los_element_channel(
            frequencies_hz,
            positions_m,
            float(ranges_m[path_index]),
            float(np.deg2rad(angles_deg[path_index])),
            absorption_table=absorption,
            include_absorption=include_absorption,
            interpolation=interpolation,
        )
        path_channels.append(channel)
        total += complex(gains[path_index]) * channel
    return total, path_channels


def run_stage1b(
    config: dict,
    project_root: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict]:
    """Run the blocked minimal Stage 1-B two-path experiment."""

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

    if not np.isclose(float(resources["pilot_energy_per_re"]), 1.0):
        raise ValueError("Stage 1-B currently requires pilot_energy_per_re = 1.0")
    if int(resources["num_rf_outputs"]) != 1:
        raise ValueError("Stage 1-B currently implements one RF output")

    frequencies, positions, absorption, designs, design_metadata = (
        build_stage1a2_designs(config, root)
    )
    _validate_minimal_design(designs)
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
    data_codebook, data_codebook_metadata = _build_data_codebook(
        config, positions
    )
    conditions = _noise_conditions(config["noise"])
    scenarios = list(config["scenarios"])
    if not scenarios or len({item["name"] for item in scenarios}) != len(scenarios):
        raise ValueError("Stage 1-B scenarios must be non-empty and uniquely named")
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
    exclusion_range = float(config["estimator"]["exclusion_range_m"])
    exclusion_angle = float(
        np.deg2rad(config["estimator"]["exclusion_angle_deg"])
    )

    rows: list[dict] = []
    execution_schedule: list[dict] = []
    estimator_timings = {design.name: 0.0 for design in designs}
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
                                * standard_noise[sample_index][
                                    design.pilot_indices
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

                order_rng = np.random.default_rng(
                    int(experiment["randomization_seed"])
                    + seed * 1009
                    + scenario_index * 65537
                    + condition_index * 9176
                )
                execution_order = [
                    designs[index] for index in order_rng.permutation(len(designs))
                ]
                execution_schedule.append(
                    {
                        "scenario": scenario_name,
                        "replicate_seed": seed,
                        "snr_condition": condition["label"],
                        "method_order": [item.name for item in execution_order],
                    }
                )

                for order_index, design in enumerate(execution_order, start=1):
                    estimate_started = time.perf_counter()
                    estimates = omp_two_path_estimate(
                        dictionaries[(design.name, "spherical")],
                        observations[design.name],
                        exclusion_range_m=exclusion_range,
                        exclusion_angle_rad=exclusion_angle,
                    )
                    estimator_timings[design.name] += (
                        time.perf_counter() - estimate_started
                    )
                    for sample_index in range(samples_per_seed):
                        estimated_ranges = np.asarray(
                            estimates["range_m"][sample_index], dtype=float
                        )
                        estimated_angles = np.rad2deg(
                            estimates["angle_rad"][sample_index]
                        ).astype(float)
                        estimated_gains = np.asarray(
                            estimates["complex_gain"][sample_index],
                            dtype=complex,
                        )
                        (
                            matched_ranges,
                            matched_angles,
                            matched_gains,
                        ) = _match_two_paths(
                            estimated_ranges,
                            estimated_angles,
                            estimated_gains,
                            scene["ranges_m"][sample_index],
                            scene["angles_deg"][sample_index],
                            range_scale_m=range_match_scale,
                            angle_scale_deg=angle_match_scale,
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
                        diagnostic = diagnostics[design.name][sample_index]
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
                                "method": design.name,
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
                                    estimates["explained_energy"][sample_index]
                                ),
                                "residual_energy": float(
                                    estimates["residual_energy"][sample_index]
                                ),
                                "selected_pair_condition_number": float(
                                    estimates["pair_condition_number"][sample_index]
                                ),
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
    )
    summary = {
        "experiment": {
            "name": str(experiment["name"]),
            "status": "exploratory Stage 1-B minimal sufficient screen",
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
            "estimator": "two-step grid OMP with joint complex-gain refit",
            "learning_model_used": False,
            "main_gate_scenario": str(config["decision"]["primary_scenario"]),
            "boundary_only_scenarios": [
                str(item["name"])
                for item in scenarios
                if str(item["name"])
                != str(config["decision"]["primary_scenario"])
            ],
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
            "pilot_energy_per_re": float(resources["pilot_energy_per_re"]),
            "num_rf_outputs": int(resources["num_rf_outputs"]),
            "combiner_row_norm": 1.0,
        },
        "estimator": {
            "range_grid_points": int(ranges.size),
            "angle_grid_points": int(angles_deg.size),
            "num_candidates": int(ranges.size * angles_deg.size),
            "range_resolution_m": float(search["range_step_m"]),
            "angle_resolution_deg": float(search["angle_step_deg"]),
            "local_exclusion_range_m": exclusion_range,
            "local_exclusion_angle_deg": float(
                config["estimator"]["exclusion_angle_deg"]
            ),
            "dictionary_build_seconds": dictionary_timings,
            "estimation_seconds_by_design": estimator_timings,
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
            "rayleigh_distance_m": float(
                rayleigh_distance_m(carrier_hz, positions)
            ),
            "absorption_enabled": include_absorption,
            "absorption_table_path": str(absorption.source_path),
            "measurement_design": design_metadata,
            "data_codebook": data_codebook_metadata,
            "data_metric": (
                "center-frequency feasible DMA state selected from estimated CSI; "
                "gross rate uses ideal-array input SNR reference"
            ),
        },
        "scenario_truth_summary": scenario_truth_summary,
        "execution_schedule": execution_schedule,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    if len(rows) != expected_rows:
        raise RuntimeError("Stage 1-B row count does not match the declared design")
    return rows, summary
