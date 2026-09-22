"""Frequency-diverse one-dimensional DMA measurement schedules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from thz_dma.channels.atmosphere import AbsorptionTable, C0_M_PER_S
from thz_dma.channels.geometry import los_element_channel, ofdm_frequencies, ula_positions
from thz_dma.observations.pilots import full_band_pilot_indices as _pilot_indices
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
