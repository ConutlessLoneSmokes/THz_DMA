"""Stage 1-A2 fairness, resource, and near-field closure experiments."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from thz_dma.analysis.stage1a2 import make_stage1a2_summary as _make_summary
from thz_dma.channels.geometry import los_channel_for_model as _channel_for_model
from thz_dma.channels.geometry import los_element_channel
from thz_dma.estimators.los_grid import (
    build_los_dictionaries_for_designs as _build_dictionaries,
    regular_grid as _grid_from_step,
    variable_projection_estimate,
)
from thz_dma.evaluation.beamforming import (
    ideal_digital_beam_gain as _ideal_digital_beam_gain,
)
from thz_dma.metrics.statistics import power_db as _db
from thz_dma.observations.frequency_diverse import (
    MeasurementDesign,
    build_stage1a2_designs,
)
from thz_dma.observations.pilots import (
    combine_pilot_observation,
    complex_standard_normal,
    effective_output_snr_db,
    ideal_array_noise_variance,
    matched_output_noise_variance,
    noise_conditions as _noise_conditions,
)


__all__ = ["MeasurementDesign", "build_stage1a2_designs", "run_stage1a2"]


def run_stage1a2(
    config: dict,
    project_root: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict]:
    """Run one blocked Stage 1-A2 configuration."""

    notify = progress or (lambda _message: None)
    started = time.perf_counter()
    experiment = config["experiment"]
    system = config["system"]
    channel_config = config["channel"]
    search = config["search"]
    atmosphere_config = config["atmosphere"]
    resources = config["resources"]
    if not np.isclose(float(resources["pilot_energy_per_re"]), 1.0):
        raise ValueError("Stage 1-A2 currently requires pilot_energy_per_re = 1.0")
    if int(resources["num_rf_outputs"]) != 1:
        raise ValueError("Stage 1-A2 currently implements one RF output")

    all_frequencies, positions, absorption, designs, design_metadata = (
        build_stage1a2_designs(config, project_root)
    )
    estimator_models = [str(value) for value in config["estimator"]["models"]]
    if not estimator_models or any(
        value not in {"spherical", "far_field"} for value in estimator_models
    ):
        raise ValueError("estimator.models must contain spherical and/or far_field")
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
        estimator_models,
        positions,
        ranges,
        angles_deg,
        absorption,
        include_absorption=bool(channel_config["include_absorption"]),
        interpolation=str(atmosphere_config["interpolation"]),
        chunk_size=int(search["dictionary_chunk_size"]),
        notify=notify,
    )
    conditions = _noise_conditions(config["noise"])
    replicate_seeds = [int(value) for value in experiment["replicate_seeds"]]
    samples_per_seed = int(experiment["samples_per_seed"])
    if not replicate_seeds or samples_per_seed < 1:
        raise ValueError("replicate seeds and samples_per_seed cannot be empty")

    include_absorption = bool(channel_config["include_absorption"])
    interpolation = str(atmosphere_config["interpolation"])
    carrier_hz = float(system["carrier_frequency_hz"])
    rows: list[dict] = []
    execution_schedule: list[dict] = []
    estimator_timings = {
        f"{design.name}__{model}": 0.0
        for design in designs
        for model in estimator_models
    }

    for seed_index, seed in enumerate(replicate_seeds, start=1):
        rng = np.random.default_rng(seed)
        scene_ranges = rng.uniform(
            float(channel_config["range_min_m"]),
            float(channel_config["range_max_m"]),
            size=samples_per_seed,
        )
        scene_angles_deg = rng.uniform(
            float(channel_config["angle_min_deg"]),
            float(channel_config["angle_max_deg"]),
            size=samples_per_seed,
        )
        gain_phase = rng.uniform(-np.pi, np.pi, size=samples_per_seed)
        true_gain = float(channel_config["complex_gain_magnitude"]) * np.exp(
            1j * gain_phase
        )
        full_channels = []
        center_channels = []
        standard_noise = []
        for sample_index in range(samples_per_seed):
            angle_rad = float(np.deg2rad(scene_angles_deg[sample_index]))
            full_channels.append(
                los_element_channel(
                    all_frequencies,
                    positions,
                    float(scene_ranges[sample_index]),
                    angle_rad,
                    absorption_table=absorption,
                    include_absorption=include_absorption,
                    interpolation=interpolation,
                )
            )
            center_channels.append(
                los_element_channel(
                    np.array([carrier_hz]),
                    positions,
                    float(scene_ranges[sample_index]),
                    angle_rad,
                    absorption_table=absorption,
                    include_absorption=include_absorption,
                    interpolation=interpolation,
                )[0]
            )
            standard_noise.append(
                complex_standard_normal(rng, (all_frequencies.size, positions.size))
            )

        for condition_index, condition in enumerate(conditions):
            observations: dict[str, np.ndarray] = {}
            diagnostic: dict[str, list[dict]] = {}
            for design in designs:
                design_observations = np.empty(
                    (samples_per_seed, design.pilot_resource_elements), dtype=complex
                )
                design_diagnostic = []
                for sample_index in range(samples_per_seed):
                    channel = full_channels[sample_index][design.pilot_indices]
                    clean = combine_pilot_observation(
                        design.weights,
                        channel,
                        complex_gain=true_gain[sample_index],
                    )
                    reference_channel = true_gain[sample_index] * full_channels[sample_index]
                    reference_power = float(
                        np.mean(np.sum(np.abs(reference_channel) ** 2, axis=1))
                    )
                    if condition["mode"] == "input_array":
                        noise_variance = ideal_array_noise_variance(
                            reference_channel, float(condition["target_snr_db"])
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
                            * standard_noise[sample_index][design.pilot_indices]
                        )
                        observation = combine_pilot_observation(
                            design.weights,
                            channel,
                            complex_gain=true_gain[sample_index],
                            element_noise=noise,
                        )
                        output_snr_db = effective_output_snr_db(clean, noise_variance)
                        input_snr_db = float(
                            10.0 * np.log10(reference_power / noise_variance)
                        )
                    design_observations[sample_index] = observation
                    design_diagnostic.append(
                        {
                            "noise_variance": noise_variance,
                            "effective_output_snr_db": output_snr_db,
                            "defined_input_array_snr_db": input_snr_db,
                        }
                    )
                observations[design.name] = design_observations
                diagnostic[design.name] = design_diagnostic

            analysis_keys = [
                (design.name, model)
                for design in designs
                for model in estimator_models
            ]
            order_rng = np.random.default_rng(
                int(experiment["randomization_seed"])
                + seed * 1009
                + condition_index * 9176
            )
            execution_order = [analysis_keys[index] for index in order_rng.permutation(len(analysis_keys))]
            execution_schedule.append(
                {
                    "replicate_seed": seed,
                    "snr_condition": condition["label"],
                    "method_order": [f"{name}__{model}" for name, model in execution_order],
                }
            )

            design_lookup = {design.name: design for design in designs}
            for order_index, (design_name, estimator_model) in enumerate(
                execution_order, start=1
            ):
                design = design_lookup[design_name]
                method = f"{design_name}__{estimator_model}"
                estimate_started = time.perf_counter()
                estimates = variable_projection_estimate(
                    dictionaries[(design_name, estimator_model)],
                    observations[design_name],
                )
                estimator_timings[method] += time.perf_counter() - estimate_started
                for sample_index in range(samples_per_seed):
                    estimated_range = float(estimates["range_m"][sample_index])
                    estimated_angle_rad = float(estimates["angle_rad"][sample_index])
                    estimated_angle_deg = float(np.rad2deg(estimated_angle_rad))
                    estimated_gain = complex(estimates["complex_gain"][sample_index])
                    reconstructed = estimated_gain * _channel_for_model(
                        estimator_model,
                        all_frequencies,
                        positions,
                        estimated_range,
                        estimated_angle_rad,
                        absorption=absorption,
                        include_absorption=include_absorption,
                        interpolation=interpolation,
                    )
                    truth = true_gain[sample_index] * full_channels[sample_index]
                    nmse = float(
                        np.sum(np.abs(reconstructed - truth) ** 2)
                        / np.sum(np.abs(truth) ** 2)
                    )
                    estimated_center = estimated_gain * _channel_for_model(
                        estimator_model,
                        np.array([carrier_hz]),
                        positions,
                        estimated_range,
                        estimated_angle_rad,
                        absorption=absorption,
                        include_absorption=include_absorption,
                        interpolation=interpolation,
                    )[0]
                    true_center = true_gain[sample_index] * center_channels[sample_index]
                    beam_gain = _ideal_digital_beam_gain(true_center, estimated_center)
                    boundary_hit = bool(
                        np.isclose(estimated_range, ranges[0])
                        or np.isclose(estimated_range, ranges[-1])
                        or np.isclose(estimated_angle_deg, angles_deg[0])
                        or np.isclose(estimated_angle_deg, angles_deg[-1])
                    )
                    diag = diagnostic[design_name][sample_index]
                    rows.append(
                        {
                            "replicate_seed": seed,
                            "sample_index": sample_index,
                            "snr_condition": condition["label"],
                            "noise_mode": condition["mode"],
                            "target_snr_db": condition["target_snr_db"],
                            "design": design_name,
                            "design_family": design.family,
                            "estimator_model": estimator_model,
                            "method": method,
                            "true_range_m": float(scene_ranges[sample_index]),
                            "estimated_range_m": estimated_range,
                            "range_error_m": estimated_range
                            - float(scene_ranges[sample_index]),
                            "true_angle_deg": float(scene_angles_deg[sample_index]),
                            "estimated_angle_deg": estimated_angle_deg,
                            "angle_error_deg": estimated_angle_deg
                            - float(scene_angles_deg[sample_index]),
                            "true_gain_real": float(true_gain[sample_index].real),
                            "true_gain_imag": float(true_gain[sample_index].imag),
                            "estimated_gain_real": float(estimated_gain.real),
                            "estimated_gain_imag": float(estimated_gain.imag),
                            "channel_nmse_linear": nmse,
                            "channel_nmse_db": _db(nmse),
                            "ideal_digital_beam_gain_linear": beam_gain,
                            "ideal_digital_beam_loss_db": -_db(beam_gain),
                            "effective_output_snr_db": diag[
                                "effective_output_snr_db"
                            ],
                            "defined_input_array_snr_db": diag[
                                "defined_input_array_snr_db"
                            ],
                            "element_noise_variance": diag["noise_variance"],
                            "profile_score": float(
                                estimates["profile_score"][sample_index]
                            ),
                            "residual_energy": float(
                                estimates["residual_energy"][sample_index]
                            ),
                            "search_boundary_hit": int(boundary_hit),
                            "execution_order_index": order_index,
                            "training_symbols": design.training_symbols,
                            "pilot_resource_elements": design.pilot_resource_elements,
                            "num_configurations": design.num_configurations,
                            "num_switches": design.num_switches,
                            "switch_guard_symbol_equivalents": float(
                                design.num_switches
                                * float(resources["switch_guard_symbol_equivalents"])
                            ),
                            "num_rf_outputs": int(resources["num_rf_outputs"]),
                        }
                    )
            notify(
                f"completed seed {seed_index}/{len(replicate_seeds)}, "
                f"condition {condition['label']}"
            )

    elapsed_seconds = time.perf_counter() - started
    summary = _make_summary(
        rows,
        config,
        designs,
        estimator_models,
        conditions,
        positions,
        all_frequencies,
        absorption,
        design_metadata,
        dictionary_timings,
        estimator_timings,
        execution_schedule,
        elapsed_seconds,
    )
    return rows, summary
