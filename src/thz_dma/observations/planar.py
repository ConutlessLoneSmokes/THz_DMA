"""Planar DMA acquisition platform and schedule construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from thz_dma.channels.atmosphere import AbsorptionTable, C0_M_PER_S
from thz_dma.channels.geometry import ofdm_frequencies
from thz_dma.channels.planar import (
    planar_aperture_diagonal_m,
    planar_aperture_extents_m,
    planar_dma_positions,
    planar_rayleigh_distance_m,
)
from thz_dma.observations.pilots import full_band_pilot_indices as _pilot_indices
from thz_dma.surfaces.multistrip import (
    cyclic_frequency_resonance_states,
    feed_distances_m,
    grouped_sector_resonances,
    multistrip_dma_weights,
)


PHYSICS_MODELS = {
    "compatibility": "separable_compatibility",
    "thz_physical": "spherical_thz",
}


@dataclass(frozen=True)
class Stage2Schedule:
    """Known acquisition schedule shared by all compatible method adapters."""

    name: str
    family: str
    pilot_indices: np.ndarray
    resonant_frequencies_hz: np.ndarray

    def __post_init__(self) -> None:
        indices = np.asarray(self.pilot_indices)
        resonances = np.asarray(self.resonant_frequencies_hz)
        if indices.ndim != 1 or indices.size == 0:
            raise ValueError("pilot_indices must be a non-empty vector")
        if np.unique(indices).size != indices.size or np.any(indices < 0):
            raise ValueError("pilot_indices must be unique and non-negative")
        if resonances.ndim != 3 or resonances.shape[0] < 1:
            raise ValueError("resonances must have shape (J, S, E)")
        if not self.name or not self.family:
            raise ValueError("schedule name and family must be non-empty")

    @property
    def num_configurations(self) -> int:
        return int(self.resonant_frequencies_hz.shape[0])

    @property
    def pilots_per_configuration(self) -> int:
        return int(self.pilot_indices.size)

    @property
    def pilot_resource_elements(self) -> int:
        return self.num_configurations * self.pilots_per_configuration

    @property
    def training_symbols(self) -> int:
        return self.num_configurations

    @property
    def num_switches(self) -> int:
        return self.num_configurations - 1


@dataclass(frozen=True)
class Stage2Platform:
    """Resolved common truth, hardware and acquisition contracts."""

    frequencies_hz: np.ndarray
    element_positions_m: np.ndarray
    feed_distances_m: np.ndarray
    absorption_table: AbsorptionTable
    schedules: tuple[Stage2Schedule, ...]
    metadata: dict


def _build_schedule(state: dict, config: dict) -> Stage2Schedule:
    system = config["system"]
    dma = config["dma"]
    configurations = int(state["num_configurations"])
    pilots = int(state["pilots_per_configuration"])
    indices = _pilot_indices(int(system["num_subcarriers"]), pilots)
    design = str(state["state_design"])
    common = (
        int(dma["num_microstrips"]),
        int(dma["elements_per_microstrip"]),
        float(system["carrier_frequency_hz"]),
        float(dma["tuning_bandwidth_hz"]),
    )
    if design == "grouped_frequency_sectors":
        if configurations != 1:
            raise ValueError("grouped_frequency_sectors requires one configuration")
        resonances = grouped_sector_resonances(
            *common, num_sectors=int(state["num_sectors"])
        )
    elif design == "cyclic_frequency_tiles":
        resonances = cyclic_frequency_resonance_states(configurations, *common)
    else:
        raise ValueError(f"unsupported Stage 2 state design: {design}")
    return Stage2Schedule(
        name=str(state["name"]),
        family=str(state["family"]),
        pilot_indices=indices,
        resonant_frequencies_hz=resonances,
    )


def build_stage2_platform(
    config: dict, project_root: str | Path
) -> Stage2Platform:
    """Resolve the common 2-D geometry, OFDM grid and acquisition schedules."""

    root = Path(project_root).resolve()
    system = config["system"]
    dma = config["dma"]
    channel = config["channel"]
    atmosphere = config["atmosphere"]
    acquisition = config["acquisition"]

    carrier_hz = float(system["carrier_frequency_hz"])
    bandwidth_hz = float(system["bandwidth_hz"])
    subcarriers = int(system["num_subcarriers"])
    nfft = int(system["nfft"])
    cp_samples = int(system["cyclic_prefix_samples"])
    if nfft != subcarriers:
        raise ValueError("Stage 2 currently requires nfft == num_subcarriers")
    if cp_samples < 0:
        raise ValueError("cyclic_prefix_samples must be non-negative")
    frequencies = ofdm_frequencies(carrier_hz, bandwidth_hz, subcarriers)
    wavelength = C0_M_PER_S / carrier_hz
    element_spacing = float(dma["element_spacing_wavelengths"]) * wavelength
    strip_spacing = float(dma["microstrip_spacing_wavelengths"]) * wavelength
    positions = planar_dma_positions(
        int(dma["num_microstrips"]),
        int(dma["elements_per_microstrip"]),
        element_spacing,
        strip_spacing,
    )
    feed_distances = feed_distances_m(
        int(dma["elements_per_microstrip"]), element_spacing
    )
    table = AbsorptionTable.from_txt(
        root / atmosphere["table_path"],
        max_frequency_hz=float(atmosphere["max_model_frequency_hz"]),
    )
    table.coefficient(frequencies, method=str(atmosphere["interpolation"]))

    schedules = tuple(
        _build_schedule(item, config) for item in acquisition["schedules"]
    )
    if len(schedules) < 2 or len({item.name for item in schedules}) != len(schedules):
        raise ValueError("Stage 2 requires at least two uniquely named schedules")
    resource_counts = {item.pilot_resource_elements for item in schedules}
    if len(resource_counts) != 1:
        raise ValueError("all Stage 2 schedules must use equal pilot RE")
    expected_re = int(acquisition["total_pilot_resource_elements"])
    if resource_counts != {expected_re}:
        raise ValueError("schedule pilot RE does not match the declared common budget")
    if int(acquisition["num_rf_outputs"]) != int(dma["num_microstrips"]):
        raise ValueError("Stage 2 requires one RF output per microstrip")

    diagonal = planar_aperture_diagonal_m(positions)
    extents = planar_aperture_extents_m(positions)
    rayleigh = planar_rayleigh_distance_m(carrier_hz, positions)
    subcarrier_spacing = bandwidth_hz / nfft
    useful_symbol_duration = 1.0 / subcarrier_spacing
    sampling_period = 1.0 / bandwidth_hz
    cp_duration = cp_samples * sampling_period
    maximum_excess_delay = float(channel["reflection_excess_range_max_m"]) / C0_M_PER_S
    metadata = {
        "carrier_wavelength_m": wavelength,
        "element_spacing_m": element_spacing,
        "microstrip_spacing_m": strip_spacing,
        "num_microstrips": int(dma["num_microstrips"]),
        "elements_per_microstrip": int(dma["elements_per_microstrip"]),
        "num_elements": int(dma["num_microstrips"])
        * int(dma["elements_per_microstrip"]),
        "aperture_extents_xyz_m": extents.tolist(),
        "aperture_diagonal_m": diagonal,
        "rayleigh_distance_m": rayleigh,
        "frequency_min_hz": float(frequencies.min()),
        "frequency_max_hz": float(frequencies.max()),
        "subcarrier_spacing_hz": subcarrier_spacing,
        "useful_symbol_duration_s": useful_symbol_duration,
        "sampling_period_s": sampling_period,
        "cyclic_prefix_duration_s": cp_duration,
        "maximum_excess_delay_s": maximum_excess_delay,
        "cyclic_prefix_margin_s": cp_duration - maximum_excess_delay,
    }
    return Stage2Platform(
        frequencies_hz=frequencies,
        element_positions_m=positions,
        feed_distances_m=feed_distances,
        absorption_table=table,
        schedules=schedules,
        metadata=metadata,
    )


def stage2_schedule_weights(
    platform: Stage2Platform,
    schedule: Stage2Schedule,
    config: dict,
    physics_level: str,
) -> np.ndarray:
    """Build one schedule's unnormalised hardware weights."""

    if physics_level not in PHYSICS_MODELS:
        raise ValueError(f"unsupported physics level: {physics_level}")
    dma = config["dma"]
    pilot_frequencies = platform.frequencies_hz[schedule.pilot_indices]
    return multistrip_dma_weights(
        pilot_frequencies,
        schedule.resonant_frequencies_hz,
        platform.feed_distances_m,
        quality_factor=float(dma["quality_factor"]),
        reference_frequency_hz=float(config["system"]["carrier_frequency_hz"]),
        waveguide_effective_index=float(dma["waveguide_effective_index"]),
        waveguide_end_power_fraction=float(dma["waveguide_end_power_fraction"]),
        frequency_flat=physics_level == "compatibility",
    )
