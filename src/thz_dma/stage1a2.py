"""Stage 1-A2 fairness, resource, and near-field closure experiments."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from thz_dma.channels.atmosphere import AbsorptionTable, C0_M_PER_S
from thz_dma.channels.geometry import (
    far_field_los_element_channel,
    los_element_channel,
    ofdm_frequencies,
    rayleigh_distance_m,
    ula_positions,
)
from thz_dma.estimators.los_grid import (
    LosTemplateDictionary,
    build_los_template_dictionaries,
    variable_projection_estimate,
)
from thz_dma.observations.pilots import (
    combine_pilot_observation,
    complex_standard_normal,
    effective_output_snr_db,
    ideal_array_noise_variance,
    matched_output_noise_variance,
)
from thz_dma.surfaces.dma import (
    dma_weight_matrix,
    frequency_tiled_resonances,
    row_normalize,
)


@dataclass(frozen=True)
class MeasurementDesign:
    """One known pilot/combiner schedule."""

    name: str
    family: str
    frequencies_hz: np.ndarray
    pilot_indices: np.ndarray
    weights: np.ndarray
    configuration_index: np.ndarray
    configuration_seeds: tuple[int, ...]
    training_symbols: int
    num_configurations: int
    num_switches: int

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.frequencies_hz)
        indices = np.asarray(self.pilot_indices)
        weights = np.asarray(self.weights)
        configuration = np.asarray(self.configuration_index)
        if frequencies.ndim != 1 or frequencies.size == 0:
            raise ValueError("design frequencies must be a non-empty vector")
        if indices.shape != frequencies.shape or configuration.shape != frequencies.shape:
            raise ValueError("pilot and configuration indices must match frequencies")
        if weights.ndim != 2 or weights.shape[0] != frequencies.size:
            raise ValueError("design weights must have one row per pilot")
        if not np.allclose(np.linalg.norm(weights, axis=1), 1.0, atol=1.0e-12):
            raise ValueError(f"design {self.name} has non-unit combiner rows")
        if self.training_symbols < 1 or self.num_configurations < 1:
            raise ValueError("training symbols and configurations must be positive")
        if self.num_switches != self.num_configurations - 1:
            raise ValueError("Stage 1-A2 assumes one contiguous block per configuration")

    @property
    def pilot_resource_elements(self) -> int:
        return int(np.asarray(self.frequencies_hz).size)


def _grid_from_step(lower: float, upper: float, step: float) -> np.ndarray:
    if not np.isfinite([lower, upper, step]).all() or lower >= upper or step <= 0.0:
        raise ValueError("grid bounds and step are invalid")
    intervals = int(round((upper - lower) / step))
    if intervals < 1 or not np.isclose(
        lower + intervals * step, upper, rtol=0.0, atol=step * 1.0e-8
    ):
        raise ValueError("grid interval must be an integer multiple of step")
    return np.linspace(lower, upper, intervals + 1)


def _pilot_indices(num_subcarriers: int, pilot_count: int) -> np.ndarray:
    """Select deterministic full-band pilots without exact decimation aliases."""

    if not 1 <= pilot_count <= num_subcarriers:
        raise ValueError("pilot count must lie in [1, num_subcarriers]")
    if pilot_count == 1:
        return np.array([num_subcarriers // 2], dtype=int)
    indices = np.rint(
        np.linspace(0, num_subcarriers - 1, pilot_count, dtype=float)
    ).astype(int)
    if np.unique(indices).size != pilot_count:
        raise RuntimeError("pilot selection produced duplicate indices")
    return indices


def _db(value: float) -> float:
    return float(10.0 * np.log10(max(float(value), np.finfo(float).tiny)))


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _bootstrap_interval(
    values,
    statistic: Callable[[np.ndarray], float],
    *,
    rng: np.random.Generator,
    resamples: int,
) -> list[float]:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or data.size == 0:
        raise ValueError("bootstrap data must be a non-empty vector")
    if data.size == 1:
        value = float(statistic(data))
        return [value, value]
    if resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100")
    indices = rng.integers(0, data.size, size=(resamples, data.size))
    estimates = np.asarray(
        [statistic(data[index]) for index in indices], dtype=float
    )
    return np.quantile(estimates, [0.025, 0.975]).astype(float).tolist()


def _ideal_digital_beam_gain(true_channel, estimated_channel) -> float:
    truth = np.asarray(true_channel, dtype=complex).ravel()
    estimate = np.asarray(estimated_channel, dtype=complex).ravel()
    truth_energy = float(np.vdot(truth, truth).real)
    estimate_norm = float(np.linalg.norm(estimate))
    if truth_energy == 0.0 or estimate_norm == 0.0:
        return 0.0
    combiner = estimate / estimate_norm
    gain = float(abs(np.vdot(combiner, truth)) ** 2 / truth_energy)
    return float(np.clip(gain, 0.0, 1.0))


def _channel_for_model(
    model: str,
    frequencies_hz,
    positions_m,
    range_m: float,
    angle_rad: float,
    *,
    absorption: AbsorptionTable,
    include_absorption: bool,
    interpolation: str,
) -> np.ndarray:
    kwargs = {
        "absorption_table": absorption,
        "include_absorption": include_absorption,
        "interpolation": interpolation,
    }
    if model == "spherical":
        return los_element_channel(
            frequencies_hz, positions_m, range_m, angle_rad, **kwargs
        )
    if model == "far_field":
        return far_field_los_element_channel(
            frequencies_hz, positions_m, range_m, angle_rad, **kwargs
        )
    raise ValueError(f"unsupported propagation model: {model}")


def _build_candidate_codebook(
    config: dict,
    frequencies_hz: np.ndarray,
    positions_m: np.ndarray,
    absorption: AbsorptionTable,
) -> tuple[np.ndarray, list[np.ndarray], list[int], dict]:
    """Select feasible static states by greedy sector-coverage maximization."""

    system = config["system"]
    channel = config["channel"]
    hardware = config["dma"]
    design_config = config["design"]
    carrier_hz = float(system["carrier_frequency_hz"])
    spacing_m = float(hardware["spacing_wavelengths"]) * C0_M_PER_S / carrier_hz
    feed_positions = np.arange(positions_m.size, dtype=float) * spacing_m
    pool_size = int(hardware["candidate_pool_size"])
    seed_start = int(hardware["candidate_seed_start"])
    flat_counts = [int(value) for value in design_config["flat_num_configurations"]]
    physical_timescan_counts = [
        int(value)
        for value in design_config.get("physical_timescan_num_configurations", [])
    ]
    physical_tensor_counts = [
        int(value)
        for value in design_config.get("physical_tensor_num_configurations", [])
    ]
    max_codebook_size = max(
        flat_counts + physical_timescan_counts + physical_tensor_counts,
        default=1,
    )
    if pool_size < max_codebook_size:
        raise ValueError("candidate_pool_size is smaller than the requested codebook")

    candidate_rows = []
    candidate_resonances = []
    candidate_seeds = []
    for offset in range(pool_size):
        seed = seed_start + offset
        resonances = frequency_tiled_resonances(
            positions_m.size,
            carrier_hz,
            float(hardware["tuning_bandwidth_hz"]),
            seed=seed,
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
        candidate_rows.append(row_normalize(response)[0])
        candidate_resonances.append(resonances)
        candidate_seeds.append(seed)
    rows = np.asarray(candidate_rows, dtype=complex)

    num_ranges = int(hardware["selection_range_points"])
    num_angles = int(hardware["selection_angle_points"])
    if num_ranges < 1 or num_angles < 2:
        raise ValueError("codebook selection grid is too small")
    training_ranges = np.linspace(
        float(hardware.get("selection_range_min_m", channel["range_min_m"])),
        float(hardware.get("selection_range_max_m", channel["range_max_m"])),
        num_ranges,
    )
    training_angles_deg = np.linspace(
        float(hardware.get("selection_angle_min_deg", channel["angle_min_deg"])),
        float(hardware.get("selection_angle_max_deg", channel["angle_max_deg"])),
        num_angles,
    )
    steering = []
    for range_m in training_ranges:
        for angle_deg in training_angles_deg:
            vector = los_element_channel(
                np.array([carrier_hz]),
                positions_m,
                float(range_m),
                float(np.deg2rad(angle_deg)),
                absorption_table=absorption,
                include_absorption=bool(channel["include_absorption"]),
                interpolation=str(config["atmosphere"]["interpolation"]),
            )[0]
            steering.append(vector / np.linalg.norm(vector))
    steering_matrix = np.asarray(steering, dtype=complex)
    candidate_gain = np.abs(steering_matrix @ np.conjugate(rows).T) ** 2

    quantile = float(hardware["selection_quantile"])
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("selection_quantile must lie in [0, 1]")
    selected: list[int] = []
    aggregate_gain = np.zeros(steering_matrix.shape[0], dtype=float)
    coverage_history = []
    for _ in range(max_codebook_size):
        trial = aggregate_gain[:, None] + candidate_gain
        primary = np.quantile(trial, quantile, axis=0)
        primary[selected] = -np.inf
        best_value = float(np.max(primary))
        tied = np.flatnonzero(np.isclose(primary, best_value, rtol=1.0e-12, atol=0.0))
        if tied.size > 1:
            secondary = np.mean(trial[:, tied], axis=0)
            best = int(tied[int(np.argmax(secondary))])
        else:
            best = int(np.argmax(primary))
        selected.append(best)
        aggregate_gain += candidate_gain[:, best]
        coverage_history.append(
            {
                "size": len(selected),
                "quantile_aggregate_gain_linear": float(
                    np.quantile(aggregate_gain, quantile)
                ),
                "mean_aggregate_gain_linear": float(np.mean(aggregate_gain)),
            }
        )

    selected_rows = rows[selected]
    selected_resonances = [candidate_resonances[index] for index in selected]
    selected_seeds = [candidate_seeds[index] for index in selected]
    metadata = {
        "algorithm": "greedy sector-coverage selection over feasible frequency-tiled states",
        "candidate_pool_size": pool_size,
        "candidate_seed_start": seed_start,
        "selection_quantile": quantile,
        "selection_ranges_m": training_ranges.astype(float).tolist(),
        "selection_angles_deg": training_angles_deg.astype(float).tolist(),
        "selected_candidate_indices": selected,
        "selected_configuration_seeds": selected_seeds,
        "coverage_history": coverage_history,
    }
    return selected_rows, selected_resonances, selected_seeds, metadata


def build_stage1a2_designs(
    config: dict,
    project_root: str | Path,
) -> tuple[np.ndarray, np.ndarray, AbsorptionTable, list[MeasurementDesign], dict]:
    """Build optimized, center-matched physical and flat measurement schedules."""

    root = Path(project_root).resolve()
    system = config["system"]
    hardware = config["dma"]
    atmosphere_config = config["atmosphere"]
    design_config = config["design"]
    carrier_hz = float(system["carrier_frequency_hz"])
    frequencies = ofdm_frequencies(
        carrier_hz,
        float(system["bandwidth_hz"]),
        int(system["num_subcarriers"]),
    )
    spacing_m = float(hardware["spacing_wavelengths"]) * C0_M_PER_S / carrier_hz
    positions = ula_positions(int(hardware["num_elements"]), spacing_m)
    feed_positions = np.arange(positions.size, dtype=float) * spacing_m
    absorption = AbsorptionTable.from_txt(
        root / atmosphere_config["table_path"],
        max_frequency_hz=float(atmosphere_config["max_model_frequency_hz"]),
    )
    center_rows, resonances, selected_seeds, codebook_metadata = (
        _build_candidate_codebook(config, frequencies, positions, absorption)
    )

    physical_responses = []
    for resonance_state in resonances:
        physical_responses.append(
            row_normalize(
                dma_weight_matrix(
                    frequencies,
                    resonance_state,
                    feed_positions,
                    quality_factor=float(hardware["quality_factor"]),
                    reference_frequency_hz=carrier_hz,
                    waveguide_effective_index=float(
                        hardware["waveguide_effective_index"]
                    ),
                    waveguide_end_power_fraction=float(
                        hardware["waveguide_end_power_fraction"]
                    ),
                )
            )
        )
    designs: list[MeasurementDesign] = []
    physical_counts = sorted(
        {int(value) for value in design_config["physical_pilot_counts"]}
    )
    for count in physical_counts:
        indices = _pilot_indices(frequencies.size, count)
        designs.append(
            MeasurementDesign(
                name=f"physical_freq_J1_K{count}",
                family="physical_frequency_response",
                frequencies_hz=frequencies[indices],
                pilot_indices=indices,
                weights=physical_responses[0][indices],
                configuration_index=np.zeros(count, dtype=int),
                configuration_seeds=(selected_seeds[0],),
                training_symbols=1,
                num_configurations=1,
                num_switches=0,
            )
        )

    flat_total = int(design_config["flat_total_pilot_re"])
    flat_indices = _pilot_indices(frequencies.size, flat_total)
    for num_configurations in sorted(
        {
            int(value)
            for value in design_config.get(
                "physical_timescan_num_configurations", []
            )
        }
    ):
        if num_configurations < 2:
            raise ValueError("physical time scan requires at least two configurations")
        if num_configurations > len(physical_responses):
            raise ValueError("requested physical time scan exceeds selected states")
        assignment = np.arange(flat_total, dtype=int) % num_configurations
        response_stack = np.stack(
            physical_responses[:num_configurations], axis=0
        )
        designs.append(
            MeasurementDesign(
                name=f"physical_timescan_J{num_configurations}_K{flat_total}",
                family="physical_time_scan",
                frequencies_hz=frequencies[flat_indices],
                pilot_indices=flat_indices,
                weights=response_stack[assignment, flat_indices],
                configuration_index=assignment,
                configuration_seeds=tuple(
                    selected_seeds[:num_configurations]
                ),
                training_symbols=num_configurations,
                num_configurations=num_configurations,
                num_switches=num_configurations - 1,
            )
        )

    tensor_counts = sorted(
        {
            int(value)
            for value in design_config.get(
                "physical_tensor_num_configurations", []
            )
        }
    )
    tensor_pilots = int(
        design_config.get("physical_tensor_pilots_per_configuration", 0)
    )
    if tensor_counts and tensor_pilots < 2:
        raise ValueError(
            "physical tensor scan requires at least two pilots per configuration"
        )
    if tensor_pilots > frequencies.size:
        raise ValueError(
            "physical tensor pilots per configuration exceed system subcarriers"
        )
    tensor_indices = (
        _pilot_indices(frequencies.size, tensor_pilots)
        if tensor_counts
        else np.empty(0, dtype=int)
    )
    for num_configurations in tensor_counts:
        if num_configurations < 2:
            raise ValueError(
                "physical tensor scan requires at least two configurations"
            )
        if num_configurations > len(physical_responses):
            raise ValueError(
                "requested physical tensor scan exceeds selected states"
            )
        repeated_indices = np.tile(tensor_indices, num_configurations)
        assignment = np.repeat(
            np.arange(num_configurations, dtype=int), tensor_pilots
        )
        tensor_weights = np.concatenate(
            [
                physical_responses[index][tensor_indices]
                for index in range(num_configurations)
            ],
            axis=0,
        )
        total_re = int(num_configurations * tensor_pilots)
        designs.append(
            MeasurementDesign(
                name=(
                    f"physical_tensor_J{num_configurations}_"
                    f"K{tensor_pilots}_RE{total_re}"
                ),
                family="physical_tensor_scan",
                frequencies_hz=frequencies[repeated_indices],
                pilot_indices=repeated_indices,
                weights=tensor_weights,
                configuration_index=assignment,
                configuration_seeds=tuple(
                    selected_seeds[:num_configurations]
                ),
                training_symbols=num_configurations,
                num_configurations=num_configurations,
                num_switches=num_configurations - 1,
            )
        )

    for num_configurations in sorted(
        {int(value) for value in design_config["flat_num_configurations"]}
    ):
        if num_configurations > center_rows.shape[0]:
            raise ValueError("requested flat codebook exceeds selected states")
        assignment = np.arange(flat_total, dtype=int) % num_configurations
        label = "flat_static" if num_configurations == 1 else "flat_timescan"
        designs.append(
            MeasurementDesign(
                name=(
                    f"{label}_J{num_configurations}_K{flat_total}"
                ),
                family=label,
                frequencies_hz=frequencies[flat_indices],
                pilot_indices=flat_indices,
                weights=center_rows[assignment],
                configuration_index=assignment,
                configuration_seeds=tuple(
                    selected_seeds[:num_configurations]
                ),
                training_symbols=num_configurations,
                num_configurations=num_configurations,
                num_switches=num_configurations - 1,
            )
        )

    if len({design.name for design in designs}) != len(designs):
        raise ValueError("measurement design names are not unique")
    metadata = {
        "codebook_selection": codebook_metadata,
        "center_match_statement": (
            "physical J1 and flat J1 use the same selected feasible state at the "
            "carrier; only the physical design retains its frequency response"
        ),
    }
    return frequencies, positions, absorption, designs, metadata


def _build_dictionaries(
    designs: list[MeasurementDesign],
    estimator_models: list[str],
    positions_m: np.ndarray,
    ranges_m: np.ndarray,
    angles_deg: np.ndarray,
    absorption: AbsorptionTable,
    *,
    include_absorption: bool,
    interpolation: str,
    chunk_size: int,
    notify: Callable[[str], None],
) -> tuple[dict[tuple[str, str], LosTemplateDictionary], dict[str, float]]:
    grouped: dict[tuple[int, bytes], list[MeasurementDesign]] = {}
    for design in designs:
        frequencies = np.ascontiguousarray(design.frequencies_hz, dtype=np.float64)
        grouped.setdefault((frequencies.size, frequencies.tobytes()), []).append(design)

    dictionaries: dict[tuple[str, str], LosTemplateDictionary] = {}
    timings: dict[str, float] = {}
    for model in estimator_models:
        for group_index, group in enumerate(grouped.values(), start=1):
            frequencies = np.asarray(group[0].frequencies_hz, dtype=float)
            weights = {design.name: design.weights for design in group}
            notify(
                f"building dictionary {group_index}/{len(grouped)} for {model}: "
                f"K={frequencies.size}, designs={len(group)}"
            )
            started = time.perf_counter()
            built = build_los_template_dictionaries(
                frequencies,
                positions_m,
                ranges_m,
                np.deg2rad(angles_deg),
                weights,
                absorption_table=absorption,
                include_absorption=include_absorption,
                interpolation=interpolation,
                chunk_size=chunk_size,
                propagation_model=model,
            )
            elapsed = time.perf_counter() - started
            timings[f"{model}:group{group_index}:K{frequencies.size}"] = float(elapsed)
            for name, dictionary in built.items():
                dictionaries[(name, model)] = dictionary
    return dictionaries, timings


def _noise_conditions(noise_config: dict) -> list[dict]:
    conditions = []
    for value in noise_config.get("input_array_snr_db", []):
        target = float(value)
        conditions.append(
            {
                "label": f"input_array:{target:g}dB",
                "mode": "input_array",
                "target_snr_db": target,
            }
        )
    for value in noise_config.get("matched_output_snr_db", []):
        target = float(value)
        conditions.append(
            {
                "label": f"matched_output:{target:g}dB",
                "mode": "matched_output",
                "target_snr_db": target,
            }
        )
    if bool(noise_config.get("include_noiseless", False)):
        conditions.append(
            {"label": "noiseless", "mode": "noiseless", "target_snr_db": None}
        )
    if not conditions:
        raise ValueError("at least one noise condition is required")
    if len({item["label"] for item in conditions}) != len(conditions):
        raise ValueError("noise conditions contain duplicates")
    return conditions


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


def _make_summary(
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
