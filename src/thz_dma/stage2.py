"""Stage 2 unified two-dimensional THz-DMA platform validation."""

from __future__ import annotations

import time
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np

from thz_dma.channels.planar import multipath_planar_channel, sample_stage2_paths
from thz_dma.metrics.diagnostics import (
    stage2_condition_summary as _condition_summary,
    tensor_unfolding_rank_residuals,
)
from thz_dma.observations.multistrip import (
    array_input_noise_variance,
    effective_tensor_output_snr_db,
    observe_multistrip_schedule,
)
from thz_dma.observations.planar import (
    PHYSICS_MODELS,
    Stage2Platform,
    Stage2Schedule,
    build_stage2_platform,
    stage2_schedule_weights,
)
from thz_dma.surfaces.multistrip import combiner_row_energy


__all__ = [
    "Stage2Platform",
    "Stage2Schedule",
    "build_stage2_platform",
    "run_stage2_validation",
    "sample_stage2_paths",
    "stage2_schedule_weights",
    "tensor_unfolding_rank_residuals",
]


def run_stage2_validation(
    config: dict,
    project_root: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict]:
    """Validate the shared Stage 2 platform before estimator adapters are added."""

    notify = progress or (lambda _message: None)
    started = time.perf_counter()
    platform = build_stage2_platform(config, project_root)
    experiment = config["experiment"]
    channel_config = config["channel"]
    atmosphere_config = config["atmosphere"]
    acquisition = config["acquisition"]
    noise_config = config["noise"]
    analysis = config["analysis"]
    levels = [str(item) for item in experiment["physics_levels"]]
    if set(levels) != set(PHYSICS_MODELS):
        raise ValueError("Stage 2 validation requires compatibility and thz_physical")
    if not bool(noise_config["element_noise_enabled"]):
        raise ValueError("Stage 2 validation requires element-domain noise")

    weights = {
        (level, schedule.name): stage2_schedule_weights(
            platform, schedule, config, level
        )
        for level, schedule in product(levels, platform.schedules)
    }
    row_energies = {key: combiner_row_energy(value) for key, value in weights.items()}
    all_energy = np.concatenate([value.reshape(-1) for value in row_energies.values()])
    rows: list[dict] = []
    execution_schedule: list[dict] = []
    replicate_seeds = [int(item) for item in experiment["replicate_seeds"]]
    samples_per_seed = int(experiment["samples_per_seed"])
    if not replicate_seeds or samples_per_seed < 1:
        raise ValueError("replicate seeds and samples_per_seed must be non-empty")

    conditions = list(product(range(len(levels)), range(len(platform.schedules))))
    for seed_index, seed in enumerate(replicate_seeds, start=1):
        scenario_rng = np.random.default_rng(seed)
        for sample_index in range(samples_per_seed):
            paths = sample_stage2_paths(config, scenario_rng)
            order_rng = np.random.default_rng(
                np.random.SeedSequence(
                    [int(experiment["randomization_seed"]), seed, sample_index]
                )
            )
            order = list(conditions)
            order_rng.shuffle(order)
            scene_id = f"{seed}:{sample_index}"
            execution_schedule.append(
                {
                    "scene_id": scene_id,
                    "condition_order": [
                        f"{levels[level_index]}::{platform.schedules[schedule_index].name}"
                        for level_index, schedule_index in order
                    ],
                }
            )

            channels = {}
            noise_variances = {}
            for level in levels:
                channel, _ = multipath_planar_channel(
                    platform.frequencies_hz,
                    platform.element_positions_m,
                    paths,
                    model=PHYSICS_MODELS[level],
                    carrier_frequency_hz=float(
                        config["system"]["carrier_frequency_hz"]
                    ),
                    absorption_table=platform.absorption_table,
                    include_absorption=(
                        level == "thz_physical"
                        and bool(channel_config["include_absorption"])
                    ),
                    interpolation=str(atmosphere_config["interpolation"]),
                )
                channels[level] = channel
                noise_variances[level] = array_input_noise_variance(
                    channel, float(noise_config["array_input_snr_db"])
                )

            for execution_index, (level_index, schedule_index) in enumerate(order):
                level = levels[level_index]
                schedule = platform.schedules[schedule_index]
                channel = channels[level][schedule.pilot_indices]
                element_variance = noise_variances[level]
                rf_variance = element_variance * float(
                    noise_config["rf_noise_variance_fraction_of_element_noise"]
                )
                noise_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [seed, sample_index, level_index, schedule_index, 2_026_090_902]
                    )
                )
                observation = observe_multistrip_schedule(
                    weights[(level, schedule.name)],
                    channel,
                    pilot_energy_per_re=float(acquisition["pilot_energy_per_re"]),
                    element_noise_variance=element_variance,
                    rf_noise_variance=rf_variance,
                    rng=noise_rng,
                )
                residuals = tensor_unfolding_rank_residuals(
                    observation.clean, int(analysis["tensor_rank"])
                )
                energy = row_energies[(level, schedule.name)]
                finite = bool(
                    np.all(np.isfinite(observation.clean))
                    and np.all(np.isfinite(observation.observed))
                    and np.all(np.isfinite(observation.output_noise_variance))
                )
                rows.append(
                    {
                        "scene_id": scene_id,
                        "replicate_seed": seed,
                        "sample_index": sample_index,
                        "physics_level": level,
                        "channel_model": PHYSICS_MODELS[level],
                        "schedule": schedule.name,
                        "schedule_family": schedule.family,
                        "execution_order_index": execution_index,
                        "num_paths": len(paths),
                        "num_rf_outputs": int(acquisition["num_rf_outputs"]),
                        "num_configurations": schedule.num_configurations,
                        "pilots_per_configuration": schedule.pilots_per_configuration,
                        "training_symbols": schedule.training_symbols,
                        "pilot_resource_elements": schedule.pilot_resource_elements,
                        "num_switches": schedule.num_switches,
                        "pilot_energy_per_re": float(
                            acquisition["pilot_energy_per_re"]
                        ),
                        "array_input_snr_db": float(
                            noise_config["array_input_snr_db"]
                        ),
                        "element_noise_variance": float(element_variance),
                        "rf_noise_variance": float(rf_variance),
                        "combiner_row_energy_min": float(np.min(energy)),
                        "combiner_row_energy_mean": float(np.mean(energy)),
                        "combiner_row_energy_max": float(np.max(energy)),
                        "clean_output_power_mean": float(
                            np.mean(np.abs(observation.clean) ** 2)
                        ),
                        "output_noise_variance_mean": float(
                            np.mean(observation.output_noise_variance)
                        ),
                        "effective_output_snr_db": effective_tensor_output_snr_db(
                            observation.clean, observation.output_noise_variance
                        ),
                        **residuals,
                        "all_finite": int(finite),
                    }
                )
        notify(
            f"completed Stage 2 platform seed {seed_index}/{len(replicate_seeds)}"
        )

    expected_rows = (
        len(replicate_seeds)
        * samples_per_seed
        * len(levels)
        * len(platform.schedules)
    )
    if len(rows) != expected_rows:
        raise RuntimeError("Stage 2 validation row count does not match the design")
    condition_summary = _condition_summary(rows)
    tensor_family = str(analysis["tensor_schedule_family"])
    compatibility_tensor_rows = [
        row
        for row in rows
        if row["physics_level"] == "compatibility"
        and row["schedule_family"] == tensor_family
    ]
    if not compatibility_tensor_rows:
        raise ValueError("no compatibility tensor-schedule rows were produced")
    compatibility_residual = max(
        item["max_mode_rank_residual_fraction"]
        for item in compatibility_tensor_rows
    )
    threshold = float(analysis["compatibility_rank_residual_threshold"])
    checks = {
        "equal_pilot_resource_elements": len(
            {item.pilot_resource_elements for item in platform.schedules}
        )
        == 1,
        "one_rf_output_per_microstrip": int(acquisition["num_rf_outputs"])
        == int(config["dma"]["num_microstrips"]),
        "cyclic_prefix_covers_maximum_excess_delay": platform.metadata[
            "cyclic_prefix_margin_s"
        ]
        >= 0.0,
        "nominal_range_straddles_rayleigh_distance": float(
            channel_config["range_min_m"]
        )
        <= platform.metadata["rayleigh_distance_m"]
        <= float(channel_config["range_max_m"]),
        "compatibility_tensor_is_rank_p": compatibility_residual <= threshold,
        "absolute_combiner_energy_preserved": not np.allclose(
            all_energy, 1.0, rtol=1.0e-5, atol=1.0e-8
        ),
        "all_outputs_finite": all(bool(item["all_finite"]) for item in rows),
    }
    status = (
        "platform_ready_for_method_adapters"
        if all(checks.values())
        else "platform_validation_failed"
    )
    summary = {
        "experiment": {
            "name": str(experiment["name"]),
            "status": status,
            "stage": "Stage 2-0 unified 2-D platform validation",
            "replicate_seeds": replicate_seeds,
            "samples_per_seed": samples_per_seed,
            "independent_scene_count": len(replicate_seeds) * samples_per_seed,
            "expected_raw_metric_rows": expected_rows,
            "actual_raw_metric_rows": len(rows),
        },
        "scope": {
            "truth": "single-user uplink OFDM; planar receive DMA; LoS plus two specular paths",
            "comparison_axis": (
                "common physical truth and hardware with schedule-specific acquisition; "
                "no estimator comparison in Stage 2-0"
            ),
            "physics_levels": {
                "compatibility": (
                    "carrier-spatial plane wave plus frequency delay and frequency-flat DMA"
                ),
                "thz_physical": (
                    "exact per-element spherical distance, spreading, absorption, "
                    "Lorentzian response and dispersive lossy waveguide"
                ),
            },
            "learning_model_used": False,
            "paper_exact_reproduction": False,
        },
        "platform_checks": checks,
        "platform_ready": bool(all(checks.values())),
        "compatibility_tensor_max_rank_residual_fraction": float(
            compatibility_residual
        ),
        "compatibility_tensor_rank_threshold": threshold,
        "resource_fairness": {
            "common_pilot_resource_elements": int(
                platform.schedules[0].pilot_resource_elements
            ),
            "pilot_energy_per_re": float(acquisition["pilot_energy_per_re"]),
            "same_truth_hardware_and_noise_definition": True,
            "same_observation_required_within_schedule_for_estimator_comparisons": True,
            "method_specific_schedule_allowed_for_full_method_comparison": True,
            "switches_and_training_symbols_must_be_charged": True,
            "combiner_row_normalization": False,
            "independent_unit": "one generated multipath scene",
            "paired_block": "all physics-level and schedule conditions within a scene",
            "condition_order_randomized": True,
        },
        "schedules": [
            {
                "name": item.name,
                "family": item.family,
                "num_configurations": item.num_configurations,
                "pilots_per_configuration": item.pilots_per_configuration,
                "pilot_resource_elements": item.pilot_resource_elements,
                "training_symbols": item.training_symbols,
                "num_switches": item.num_switches,
                "pilot_indices": item.pilot_indices.astype(int).tolist(),
            }
            for item in platform.schedules
        ],
        "model_metadata": {
            **platform.metadata,
            "bandwidth_hz": float(config["system"]["bandwidth_hz"]),
            "num_subcarriers": int(config["system"]["num_subcarriers"]),
            "quality_factor": float(config["dma"]["quality_factor"]),
            "waveguide_effective_index": float(
                config["dma"]["waveguide_effective_index"]
            ),
            "waveguide_end_power_fraction": float(
                config["dma"]["waveguide_end_power_fraction"]
            ),
            "absorption_enabled_in_target_model": bool(
                channel_config["include_absorption"]
            ),
            "absorption_table_path": str(platform.absorption_table.source_path),
        },
        "method_adapter_contract": {
            "current_status": "interfaces only; estimators are intentionally not yet compared",
            "frequency_single_shot_candidates": [
                "Deshpande-style single-shot decoding",
                "Yang-style OG-OLS",
                "grid OMP",
                "joint-frequency SBL",
            ],
            "tensor_schedule_candidates": [
                "Zhang-style receive-side tensor adapter",
                "Yang-style OG-OLS",
                "grid OMP",
                "joint-frequency SBL",
            ],
            "common_downstream_endpoint": (
                "feasible DMA data codebook beam loss and net spectral efficiency"
            ),
        },
        "condition_summary": condition_summary,
        "execution_schedule": execution_schedule,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    return rows, summary
