"""Stage 1-A paired Monte Carlo experiment for single-path LoS estimation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from thz_dma.channels.atmosphere import AbsorptionTable, C0_M_PER_S
from thz_dma.channels.geometry import los_element_channel, ofdm_frequencies, ula_positions
from thz_dma.estimators.los_grid import (
    build_los_template_dictionaries,
    variable_projection_estimate,
)
from thz_dma.observations.pilots import (
    combine_pilot_observation,
    complex_standard_normal,
    effective_output_snr_db,
    ideal_array_noise_variance,
)
from thz_dma.surfaces.dma import (
    dma_weight_matrix,
    frequency_tiled_resonances,
    normalized_lorentzian_response,
    row_normalize,
    waveguide_response,
)


FLAT_METHOD = "flat_shared_projection"
PHYSICAL_METHOD = "physical_frequency_response"


def _grid_from_step(lower: float, upper: float, step: float) -> np.ndarray:
    if not np.isfinite([lower, upper, step]).all() or lower >= upper or step <= 0.0:
        raise ValueError("grid bounds and step are invalid")
    intervals = int(round((upper - lower) / step))
    if intervals < 1 or not np.isclose(
        lower + intervals * step, upper, rtol=0.0, atol=step * 1.0e-8
    ):
        raise ValueError("grid interval must be an integer multiple of step")
    return np.linspace(lower, upper, intervals + 1)


def _bootstrap_interval(
    values,
    statistic: Callable[[np.ndarray], float],
    *,
    rng: np.random.Generator,
    resamples: int,
) -> list[float]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size < 2:
        raise ValueError("bootstrap data must contain at least two values")
    if resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100")
    indices = rng.integers(0, data.size, size=(resamples, data.size))
    sampled = data[indices]
    estimates = np.asarray([statistic(row) for row in sampled], dtype=float)
    return np.quantile(estimates, [0.025, 0.975]).astype(float).tolist()


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _mean(values: np.ndarray) -> float:
    return float(np.mean(np.asarray(values, dtype=float)))


def _db(value: float) -> float:
    return float(10.0 * np.log10(max(float(value), np.finfo(float).tiny)))


def _ideal_digital_beam_gain(true_channel, estimated_channel) -> float:
    truth = np.asarray(true_channel, dtype=complex).ravel()
    estimate = np.asarray(estimated_channel, dtype=complex).ravel()
    estimate_norm = np.linalg.norm(estimate)
    truth_energy = float(np.vdot(truth, truth).real)
    if estimate_norm == 0.0 or truth_energy == 0.0:
        return 0.0
    combiner = estimate / estimate_norm
    gain = float(abs(np.vdot(combiner, truth)) ** 2 / truth_energy)
    return float(np.clip(gain, 0.0, 1.0))


def build_stage1_weight_models(config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return full frequencies, centered aperture positions, and paired combiners."""

    system = config["system"]
    hardware = config["dma"]
    carrier_hz = float(system["carrier_frequency_hz"])
    frequencies = ofdm_frequencies(
        carrier_hz,
        float(system["bandwidth_hz"]),
        int(system["num_subcarriers"]),
    )
    num_elements = int(hardware["num_elements"])
    spacing_m = (
        float(hardware["spacing_wavelengths"]) * C0_M_PER_S / carrier_hz
    )
    positions = ula_positions(num_elements, spacing_m)
    feed_positions = np.arange(num_elements, dtype=float) * spacing_m
    resonances = frequency_tiled_resonances(
        num_elements,
        carrier_hz,
        float(hardware["tuning_bandwidth_hz"]),
        seed=int(hardware["configuration_seed"]),
    )
    full_response = dma_weight_matrix(
        frequencies,
        resonances,
        feed_positions,
        quality_factor=float(hardware["quality_factor"]),
        reference_frequency_hz=carrier_hz,
        waveguide_effective_index=float(hardware["waveguide_effective_index"]),
        waveguide_end_power_fraction=float(hardware["waveguide_end_power_fraction"]),
    )
    center_element = normalized_lorentzian_response(
        np.array([carrier_hz]),
        resonances,
        quality_factor=float(hardware["quality_factor"]),
        reference_frequency_hz=carrier_hz,
    )
    center_waveguide = waveguide_response(
        np.array([carrier_hz]),
        feed_positions,
        effective_index=float(hardware["waveguide_effective_index"]),
        end_power_fraction=float(hardware["waveguide_end_power_fraction"]),
    )
    flat_response = np.repeat(
        center_element * center_waveguide, frequencies.size, axis=0
    )
    return frequencies, positions, {
        FLAT_METHOD: row_normalize(flat_response),
        PHYSICAL_METHOD: row_normalize(full_response),
    }


def _summarize(
    rows: list[dict],
    *,
    methods: list[str],
    snr_values: list[float],
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> tuple[dict, dict]:
    rng = np.random.default_rng(bootstrap_seed)
    grouped: dict = {}
    for snr_db in snr_values:
        grouped[str(snr_db)] = {}
        for method in methods:
            subset = [
                row
                for row in rows
                if row["method"] == method and row["input_snr_db"] == snr_db
            ]
            range_error = np.asarray([row["range_error_m"] for row in subset])
            angle_error = np.asarray([row["angle_error_deg"] for row in subset])
            channel_nmse = np.asarray([row["channel_nmse_linear"] for row in subset])
            beam_gain = np.asarray([row["ideal_digital_beam_gain_linear"] for row in subset])
            range_ci = _bootstrap_interval(
                range_error,
                _rmse,
                rng=rng,
                resamples=bootstrap_resamples,
            )
            angle_ci = _bootstrap_interval(
                angle_error,
                _rmse,
                rng=rng,
                resamples=bootstrap_resamples,
            )
            nmse_ci_linear = _bootstrap_interval(
                channel_nmse,
                _mean,
                rng=rng,
                resamples=bootstrap_resamples,
            )
            grouped[str(snr_db)][method] = {
                "num_independent_scenes": len(subset),
                "range_rmse_m": _rmse(range_error),
                "range_rmse_ci95_m": range_ci,
                "angle_rmse_deg": _rmse(angle_error),
                "angle_rmse_ci95_deg": angle_ci,
                "mean_channel_nmse_db": _db(_mean(channel_nmse)),
                "mean_channel_nmse_ci95_db": [_db(value) for value in nmse_ci_linear],
                "median_channel_nmse_db": float(
                    np.median([row["channel_nmse_db"] for row in subset])
                ),
                "median_ideal_digital_beam_loss_db": float(
                    np.median([row["ideal_digital_beam_loss_db"] for row in subset])
                ),
                "fifth_percentile_ideal_digital_beam_gain_linear": float(
                    np.quantile(beam_gain, 0.05)
                ),
                "mean_effective_output_snr_db": float(
                    np.mean([row["effective_output_snr_db"] for row in subset])
                ),
                "search_boundary_hit_rate": float(
                    np.mean([row["search_boundary_hit"] for row in subset])
                ),
            }

    paired: dict = {}
    lookup = {
        (
            row["replicate_seed"],
            row["sample_index"],
            row["input_snr_db"],
            row["method"],
        ): row
        for row in rows
    }
    for snr_db in snr_values:
        flat_rows = [
            row
            for row in rows
            if row["method"] == FLAT_METHOD and row["input_snr_db"] == snr_db
        ]
        channel_delta = []
        range_delta = []
        angle_delta = []
        beam_delta = []
        for flat in flat_rows:
            physical = lookup[
                (
                    flat["replicate_seed"],
                    flat["sample_index"],
                    snr_db,
                    PHYSICAL_METHOD,
                )
            ]
            channel_delta.append(
                physical["channel_nmse_db"] - flat["channel_nmse_db"]
            )
            range_delta.append(
                abs(physical["range_error_m"]) - abs(flat["range_error_m"])
            )
            angle_delta.append(
                abs(physical["angle_error_deg"]) - abs(flat["angle_error_deg"])
            )
            beam_delta.append(
                physical["ideal_digital_beam_loss_db"]
                - flat["ideal_digital_beam_loss_db"]
            )
        channel_delta_array = np.asarray(channel_delta)
        paired[str(snr_db)] = {
            "num_paired_scenes": len(channel_delta),
            "physical_minus_flat_channel_nmse_db_mean": _mean(channel_delta_array),
            "physical_minus_flat_channel_nmse_db_ci95": _bootstrap_interval(
                channel_delta_array,
                _mean,
                rng=rng,
                resamples=bootstrap_resamples,
            ),
            "physical_better_channel_nmse_fraction": float(
                np.mean(channel_delta_array < 0.0)
            ),
            "physical_minus_flat_absolute_range_error_m_mean": _mean(
                np.asarray(range_delta)
            ),
            "physical_minus_flat_absolute_angle_error_deg_mean": _mean(
                np.asarray(angle_delta)
            ),
            "physical_minus_flat_ideal_beam_loss_db_mean": _mean(
                np.asarray(beam_delta)
            ),
        }
    return grouped, paired


def run_stage1_los(
    config: dict,
    project_root: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict]:
    """Run the Stage 1-A paired LoS estimation experiment."""

    notify = progress or (lambda _message: None)
    started = time.perf_counter()
    root = Path(project_root).resolve()
    experiment = config["experiment"]
    system = config["system"]
    channel_config = config["channel"]
    search = config["search"]
    atmosphere_config = config["atmosphere"]
    resources = config["resources"]

    all_frequencies, positions, all_weights = build_stage1_weight_models(config)
    stride = int(system["pilot_subcarrier_stride"])
    if stride < 1:
        raise ValueError("pilot_subcarrier_stride must be positive")
    pilot_indices = np.arange(0, all_frequencies.size, stride, dtype=int)
    pilot_frequencies = all_frequencies[pilot_indices]
    weights = {name: value[pilot_indices] for name, value in all_weights.items()}
    for name, value in weights.items():
        row_norm = np.linalg.norm(value, axis=1)
        if not np.allclose(row_norm, 1.0, atol=1.0e-12):
            raise ValueError(f"combiner {name} is not row-normalized")

    absorption = AbsorptionTable.from_txt(
        root / atmosphere_config["table_path"],
        max_frequency_hz=float(atmosphere_config["max_model_frequency_hz"]),
    )
    include_absorption = bool(channel_config["include_absorption"])
    interpolation = str(atmosphere_config["interpolation"])
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
    notify(
        f"building {ranges.size * angles_deg.size} templates for "
        f"{len(weights)} paired methods"
    )
    dictionary_started = time.perf_counter()
    dictionaries = build_los_template_dictionaries(
        pilot_frequencies,
        positions,
        ranges,
        np.deg2rad(angles_deg),
        weights,
        absorption_table=absorption,
        include_absorption=include_absorption,
        interpolation=interpolation,
        chunk_size=int(search["dictionary_chunk_size"]),
    )
    dictionary_seconds = time.perf_counter() - dictionary_started

    replicate_seeds = [int(value) for value in experiment["replicate_seeds"]]
    samples_per_seed = int(experiment["samples_per_seed"])
    snr_values = [float(value) for value in experiment["input_snr_db"]]
    if len(replicate_seeds) < 1 or samples_per_seed < 1 or len(snr_values) < 1:
        raise ValueError("replicate seeds, samples and SNR values cannot be empty")

    rows: list[dict] = []
    estimator_seconds = {name: 0.0 for name in weights}
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
        pilot_channels = []
        full_channels = []
        center_channels = []
        standard_noise = []
        for sample_index in range(samples_per_seed):
            angle_rad = np.deg2rad(scene_angles_deg[sample_index])
            pilot_channels.append(
                los_element_channel(
                    pilot_frequencies,
                    positions,
                    float(scene_ranges[sample_index]),
                    float(angle_rad),
                    absorption_table=absorption,
                    include_absorption=include_absorption,
                    interpolation=interpolation,
                )
            )
            full_channels.append(
                los_element_channel(
                    all_frequencies,
                    positions,
                    float(scene_ranges[sample_index]),
                    float(angle_rad),
                    absorption_table=absorption,
                    include_absorption=include_absorption,
                    interpolation=interpolation,
                )
            )
            center_channels.append(
                los_element_channel(
                    np.array([float(system["carrier_frequency_hz"])]),
                    positions,
                    float(scene_ranges[sample_index]),
                    float(angle_rad),
                    absorption_table=absorption,
                    include_absorption=include_absorption,
                    interpolation=interpolation,
                )[0]
            )
            standard_noise.append(
                complex_standard_normal(
                    rng, (pilot_frequencies.size, positions.size)
                )
            )

        for snr_db in snr_values:
            observations = {
                name: np.empty(
                    (samples_per_seed, pilot_frequencies.size), dtype=complex
                )
                for name in weights
            }
            noise_variances = np.empty(samples_per_seed, dtype=float)
            effective_snr = {
                name: np.empty(samples_per_seed, dtype=float) for name in weights
            }
            for sample_index in range(samples_per_seed):
                channel = pilot_channels[sample_index]
                noise_variance = ideal_array_noise_variance(channel, snr_db)
                noise_variances[sample_index] = noise_variance
                noise = np.sqrt(noise_variance) * standard_noise[sample_index]
                for name, method_weights in weights.items():
                    clean = combine_pilot_observation(
                        method_weights,
                        channel,
                        complex_gain=true_gain[sample_index],
                    )
                    observations[name][sample_index] = combine_pilot_observation(
                        method_weights,
                        channel,
                        complex_gain=true_gain[sample_index],
                        element_noise=noise,
                    )
                    effective_snr[name][sample_index] = effective_output_snr_db(
                        clean, noise_variance
                    )

            for name in weights:
                estimate_started = time.perf_counter()
                estimates = variable_projection_estimate(
                    dictionaries[name], observations[name]
                )
                elapsed = time.perf_counter() - estimate_started
                estimator_seconds[name] += elapsed
                for sample_index in range(samples_per_seed):
                    estimated_range = float(estimates["range_m"][sample_index])
                    estimated_angle_rad = float(estimates["angle_rad"][sample_index])
                    estimated_angle_deg = float(np.rad2deg(estimated_angle_rad))
                    estimated_gain = complex(estimates["complex_gain"][sample_index])
                    reconstructed = estimated_gain * los_element_channel(
                        all_frequencies,
                        positions,
                        estimated_range,
                        estimated_angle_rad,
                        absorption_table=absorption,
                        include_absorption=include_absorption,
                        interpolation=interpolation,
                    )
                    truth = true_gain[sample_index] * full_channels[sample_index]
                    nmse = float(
                        np.sum(np.abs(reconstructed - truth) ** 2)
                        / np.sum(np.abs(truth) ** 2)
                    )
                    estimated_center = estimated_gain * los_element_channel(
                        np.array([float(system["carrier_frequency_hz"])]),
                        positions,
                        estimated_range,
                        estimated_angle_rad,
                        absorption_table=absorption,
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
                    rows.append(
                        {
                            "replicate_seed": seed,
                            "sample_index": sample_index,
                            "input_snr_db": snr_db,
                            "method": name,
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
                            "effective_output_snr_db": float(
                                effective_snr[name][sample_index]
                            ),
                            "element_noise_variance": float(
                                noise_variances[sample_index]
                            ),
                            "profile_score": float(
                                estimates["profile_score"][sample_index]
                            ),
                            "residual_energy": float(
                                estimates["residual_energy"][sample_index]
                            ),
                            "search_boundary_hit": int(boundary_hit),
                            "training_symbols": int(resources["training_symbols"]),
                            "pilot_resource_elements": int(pilot_frequencies.size),
                            "num_configurations": int(resources["num_configurations"]),
                            "num_switches": int(resources["num_switches"]),
                            "num_rf_outputs": int(resources["num_rf_outputs"]),
                        }
                    )
            notify(
                f"completed seed {seed_index}/{len(replicate_seeds)}, "
                f"input SNR {snr_db:g} dB"
            )

    grouped, paired = _summarize(
        rows,
        methods=list(weights),
        snr_values=snr_values,
        bootstrap_seed=int(experiment["bootstrap_seed"]),
        bootstrap_resamples=int(experiment["bootstrap_resamples"]),
    )
    elapsed_seconds = time.perf_counter() - started
    summary = {
        "experiment": {
            "name": str(experiment["name"]),
            "status": "exploratory Stage 1-A",
            "independent_scene_count_per_snr": len(replicate_seeds)
            * samples_per_seed,
            "replicate_seeds": replicate_seeds,
            "samples_per_seed": samples_per_seed,
            "input_snr_db": snr_values,
        },
        "resource_fairness": {
            "training_symbols": int(resources["training_symbols"]),
            "pilot_resource_elements": int(pilot_frequencies.size),
            "num_configurations": int(resources["num_configurations"]),
            "num_switches": int(resources["num_switches"]),
            "num_rf_outputs": int(resources["num_rf_outputs"]),
            "pilot_energy_per_re": float(resources["pilot_energy_per_re"]),
            "combiner_row_norm": 1.0,
            "noise_pairing": "same element-domain noise for both methods",
            "snr_definition": "average ideal matched-array SNR before candidate combiner",
        },
        "estimator": {
            "name": "2D grid variable projection with analytic complex-gain elimination",
            "range_grid_points": int(ranges.size),
            "angle_grid_points": int(angles_deg.size),
            "num_candidates": int(ranges.size * angles_deg.size),
            "range_resolution_m": float(search["range_step_m"]),
            "angle_resolution_deg": float(search["angle_step_deg"]),
            "dictionary_build_seconds": float(dictionary_seconds),
            "estimation_seconds_by_method": estimator_seconds,
        },
        "results_by_input_snr": grouped,
        "paired_effects": paired,
        "model_metadata": {
            "carrier_frequency_hz": float(system["carrier_frequency_hz"]),
            "bandwidth_hz": float(system["bandwidth_hz"]),
            "system_subcarriers": int(all_frequencies.size),
            "pilot_subcarriers": int(pilot_frequencies.size),
            "pilot_subcarrier_stride": stride,
            "subcarrier_spacing_hz": float(
                float(system["bandwidth_hz"]) / int(system["num_subcarriers"])
            ),
            "delay_alias_period_m": float(
                C0_M_PER_S
                / (
                    float(system["bandwidth_hz"])
                    / int(system["num_subcarriers"])
                    * stride
                )
            ),
            "num_elements": int(positions.size),
            "aperture_m": float(np.ptp(positions)),
            "absorption_enabled": include_absorption,
            "absorption_table_path": str(absorption.source_path),
            "configuration_label": "single unoptimized frequency-tiled resonance state",
            "data_beam_metric": "ideal fully-digital center-frequency combiner; upper-bound diagnostic",
        },
        "runtime_seconds": float(elapsed_seconds),
    }
    return rows, summary

