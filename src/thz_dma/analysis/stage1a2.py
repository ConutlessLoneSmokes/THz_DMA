"""Stage 1-A2 statistical summaries."""

from __future__ import annotations

import math

import numpy as np

from thz_dma.channels.atmosphere import AbsorptionTable, C0_M_PER_S
from thz_dma.channels.geometry import rayleigh_distance_m
from thz_dma.metrics.statistics import (
    bootstrap_interval as _bootstrap_interval,
    power_db as _db,
    rmse as _rmse,
)
from thz_dma.observations.frequency_diverse import MeasurementDesign


def _summarize_group(rows: list[dict], rng, resamples: int) -> dict:
    range_error = np.asarray([row["range_error_m"] for row in rows], dtype=float)
    angle_error = np.asarray([row["angle_error_deg"] for row in rows], dtype=float)
    nmse = np.asarray([row["channel_nmse_linear"] for row in rows], dtype=float)
    beam_gain = np.asarray(
        [row["ideal_digital_beam_gain_linear"] for row in rows], dtype=float
    )
    output_snr = [
        float(row["effective_output_snr_db"])
        for row in rows
        if row["effective_output_snr_db"] is not None
    ]
    input_snr = [
        float(row["defined_input_array_snr_db"])
        for row in rows
        if row["defined_input_array_snr_db"] is not None
    ]
    nmse_ci = _bootstrap_interval(
        nmse, lambda value: float(np.mean(value)), rng=rng, resamples=resamples
    )
    return {
        "num_independent_scenes": len(rows),
        "range_rmse_m": _rmse(range_error),
        "range_rmse_ci95_m": _bootstrap_interval(
            range_error, _rmse, rng=rng, resamples=resamples
        ),
        "angle_rmse_deg": _rmse(angle_error),
        "angle_rmse_ci95_deg": _bootstrap_interval(
            angle_error, _rmse, rng=rng, resamples=resamples
        ),
        "mean_channel_nmse_db": _db(float(np.mean(nmse))),
        "mean_channel_nmse_ci95_db": [_db(value) for value in nmse_ci],
        "median_channel_nmse_db": float(
            np.median([row["channel_nmse_db"] for row in rows])
        ),
        "median_ideal_digital_beam_loss_db": float(
            np.median([row["ideal_digital_beam_loss_db"] for row in rows])
        ),
        "fifth_percentile_ideal_digital_beam_gain_linear": float(
            np.quantile(beam_gain, 0.05)
        ),
        "mean_effective_output_snr_db": (
            float(np.mean(output_snr)) if output_snr else None
        ),
        "mean_defined_input_array_snr_db": (
            float(np.mean(input_snr)) if input_snr else None
        ),
        "search_boundary_hit_rate": float(
            np.mean([row["search_boundary_hit"] for row in rows])
        ),
    }


def _paired_contrast(
    rows: list[dict],
    condition_label: str,
    left_method: str,
    right_method: str,
    *,
    rng: np.random.Generator,
    resamples: int,
) -> dict:
    relevant = [row for row in rows if row["snr_condition"] == condition_label]
    lookup = {
        (row["replicate_seed"], row["sample_index"], row["method"]): row
        for row in relevant
    }
    scene_keys = sorted(
        {
            (row["replicate_seed"], row["sample_index"])
            for row in relevant
            if row["method"] == left_method
        }
    )
    pairs = [
        (lookup[(seed, sample, left_method)], lookup[(seed, sample, right_method)])
        for seed, sample in scene_keys
        if (seed, sample, right_method) in lookup
    ]
    if not pairs:
        raise ValueError(f"no paired rows for {left_method} versus {right_method}")
    nmse_delta = np.asarray(
        [left["channel_nmse_db"] - right["channel_nmse_db"] for left, right in pairs]
    )
    range_delta = np.asarray(
        [
            abs(left["range_error_m"]) - abs(right["range_error_m"])
            for left, right in pairs
        ]
    )
    angle_delta = np.asarray(
        [
            abs(left["angle_error_deg"]) - abs(right["angle_error_deg"])
            for left, right in pairs
        ]
    )
    beam_delta = np.asarray(
        [
            left["ideal_digital_beam_loss_db"]
            - right["ideal_digital_beam_loss_db"]
            for left, right in pairs
        ]
    )
    return {
        "left_method": left_method,
        "right_method": right_method,
        "num_paired_scenes": len(pairs),
        "left_minus_right_channel_nmse_db_mean": float(np.mean(nmse_delta)),
        "left_minus_right_channel_nmse_db_ci95": _bootstrap_interval(
            nmse_delta,
            lambda value: float(np.mean(value)),
            rng=rng,
            resamples=resamples,
        ),
        "left_better_channel_nmse_fraction": float(np.mean(nmse_delta < 0.0)),
        "left_minus_right_absolute_range_error_m_mean": float(
            np.mean(range_delta)
        ),
        "left_minus_right_absolute_angle_error_deg_mean": float(
            np.mean(angle_delta)
        ),
        "left_minus_right_ideal_beam_loss_db_mean": float(np.mean(beam_delta)),
    }


def _frequency_alias_period_m(
    pilot_indices: np.ndarray, subcarrier_spacing_hz: float
) -> float | None:
    indices = np.asarray(pilot_indices, dtype=int)
    if indices.size < 2:
        return None
    offsets = np.abs(indices - int(indices[0]))
    gcd_index = 0
    for value in offsets:
        gcd_index = math.gcd(gcd_index, int(value))
    if gcd_index == 0:
        return None
    return float(C0_M_PER_S / (gcd_index * subcarrier_spacing_hz))


def make_stage1a2_summary(
    rows: list[dict],
    config: dict,
    designs: list[MeasurementDesign],
    estimator_models: list[str],
    conditions: list[dict],
    positions_m: np.ndarray,
    frequencies_hz: np.ndarray,
    absorption: AbsorptionTable,
    design_metadata: dict,
    dictionary_timings: dict[str, float],
    estimator_timings: dict[str, float],
    execution_schedule: list[dict],
    elapsed_seconds: float,
) -> dict:
    experiment = config["experiment"]
    system = config["system"]
    channel = config["channel"]
    search = config["search"]
    resources = config["resources"]
    rng = np.random.default_rng(int(experiment["bootstrap_seed"]))
    resamples = int(experiment["bootstrap_resamples"])
    method_order = [
        f"{design.name}__{model}" for design in designs for model in estimator_models
    ]
    grouped = {}
    for condition in conditions:
        label = condition["label"]
        grouped[label] = {}
        for method in method_order:
            subset = [
                row
                for row in rows
                if row["snr_condition"] == label and row["method"] == method
            ]
            grouped[label][method] = _summarize_group(subset, rng, resamples)

    physical_designs = [
        design for design in designs if design.family == "physical_frequency_response"
    ]
    physical_full = max(
        physical_designs, key=lambda design: design.pilot_resource_elements
    )
    comparator_designs = [
        design
        for design in designs
        if design.family.startswith("flat") or design.family == "physical_time_scan"
    ]
    paired = {}
    for condition in conditions:
        label = condition["label"]
        paired[label] = {}
        for model in estimator_models:
            left = f"{physical_full.name}__{model}"
            for comparator in comparator_designs:
                right = f"{comparator.name}__{model}"
                key = f"physical_full_vs_{comparator.name}__{model}"
                paired[label][key] = _paired_contrast(
                    rows,
                    label,
                    left,
                    right,
                    rng=rng,
                    resamples=resamples,
                )
        if {"spherical", "far_field"}.issubset(estimator_models):
            for design in designs:
                left = f"{design.name}__far_field"
                right = f"{design.name}__spherical"
                key = f"far_field_mismatch_vs_spherical__{design.name}"
                paired[label][key] = _paired_contrast(
                    rows,
                    label,
                    left,
                    right,
                    rng=rng,
                    resamples=resamples,
                )

    spacing_hz = float(system["bandwidth_hz"]) / int(system["num_subcarriers"])
    design_records = []
    for design in designs:
        design_records.append(
            {
                "name": design.name,
                "family": design.family,
                "training_symbols": design.training_symbols,
                "pilot_resource_elements": design.pilot_resource_elements,
                "pilot_energy_per_re": float(resources["pilot_energy_per_re"]),
                "total_normalized_pilot_energy": float(
                    design.pilot_resource_elements
                    * float(resources["pilot_energy_per_re"])
                ),
                "num_configurations": design.num_configurations,
                "num_switches": design.num_switches,
                "switch_guard_symbol_equivalents_per_switch": float(
                    resources["switch_guard_symbol_equivalents"]
                ),
                "total_switch_guard_symbol_equivalents": float(
                    design.num_switches
                    * float(resources["switch_guard_symbol_equivalents"])
                ),
                "training_plus_guard_symbol_equivalents": float(
                    design.training_symbols
                    + design.num_switches
                    * float(resources["switch_guard_symbol_equivalents"])
                ),
                "num_rf_outputs": int(resources["num_rf_outputs"]),
                "configuration_seeds": list(design.configuration_seeds),
                "pilot_indices": design.pilot_indices.astype(int).tolist(),
                "configuration_index_by_pilot": design.configuration_index.astype(
                    int
                ).tolist(),
                "fundamental_delay_alias_period_m": _frequency_alias_period_m(
                    design.pilot_indices, spacing_hz
                ),
            }
        )

    rayleigh_m = rayleigh_distance_m(
        float(system["carrier_frequency_hz"]), positions_m
    )
    return {
        "experiment": {
            "name": str(experiment["name"]),
            "status": "exploratory Stage 1-A2",
            "independent_scene_count_per_condition": len(
                experiment["replicate_seeds"]
            )
            * int(experiment["samples_per_seed"]),
            "replicate_seeds": [int(value) for value in experiment["replicate_seeds"]],
            "samples_per_seed": int(experiment["samples_per_seed"]),
            "randomization_seed": int(experiment["randomization_seed"]),
            "experimental_unit": "one independently drawn range-angle-gain scene",
            "blocking_and_pairing": (
                "replicate seed is a block; all methods share each scene and the "
                "same per-frequency standardized element-noise realization"
            ),
        },
        "noise_conditions": conditions,
        "measurement_designs": design_records,
        "estimator": {
            "name": "2D grid variable projection with analytic complex-gain elimination",
            "truth_model": "spherical",
            "candidate_models": estimator_models,
            "range_grid_points": int(
                round(
                    (float(search["range_max_m"]) - float(search["range_min_m"]))
                    / float(search["range_step_m"])
                )
                + 1
            ),
            "angle_grid_points": int(
                round(
                    (float(search["angle_max_deg"]) - float(search["angle_min_deg"]))
                    / float(search["angle_step_deg"])
                )
                + 1
            ),
            "range_resolution_m": float(search["range_step_m"]),
            "angle_resolution_deg": float(search["angle_step_deg"]),
            "dictionary_build_seconds": dictionary_timings,
            "estimation_seconds_by_method": estimator_timings,
        },
        "results_by_condition": grouped,
        "paired_contrasts": paired,
        "model_metadata": {
            "carrier_frequency_hz": float(system["carrier_frequency_hz"]),
            "bandwidth_hz": float(system["bandwidth_hz"]),
            "system_subcarriers": int(frequencies_hz.size),
            "subcarrier_spacing_hz": spacing_hz,
            "num_elements": int(positions_m.size),
            "aperture_m": float(np.ptp(positions_m)),
            "rayleigh_distance_m": float(rayleigh_m),
            "scene_range_over_rayleigh_min": float(channel["range_min_m"])
            / rayleigh_m,
            "scene_range_over_rayleigh_max": float(channel["range_max_m"])
            / rayleigh_m,
            "near_field_by_rayleigh_boundary": bool(
                float(channel["range_max_m"]) < rayleigh_m
            ),
            "absorption_enabled": bool(channel["include_absorption"]),
            "absorption_table_path": str(absorption.source_path),
            "snr_definitions": {
                "input_array": (
                    "average ideal matched-array SNR before candidate combiner; "
                    "one element-noise variance is shared by every design in a scene; "
                    "unit row normalization retains projection gain but excludes a "
                    "receiver-noise penalty from hardware insertion loss"
                ),
                "matched_output": (
                    "each design's expected post-combiner SNR is set to the target; "
                    "this removes projection-gain differences but not template diversity"
                ),
                "noiseless": "off-grid numerical identifiability and model-mismatch diagnostic",
            },
            "data_beam_metric": (
                "ideal fully-digital center-frequency combiner; upper-bound diagnostic"
            ),
            "time_scan_channel_assumption": (
                "the LoS channel is constant across all J training symbols; each "
                "configuration occupies one symbol and observes its assigned pilot subset"
            ),
            **design_metadata,
        },
        "execution_schedule": execution_schedule,
        "runtime_seconds": float(elapsed_seconds),
    }
