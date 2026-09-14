"""Stage 2 unified two-dimensional THz-DMA platform validation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np

from thz_dma.channels.atmosphere import AbsorptionTable, C0_M_PER_S
from thz_dma.channels.geometry import ofdm_frequencies
from thz_dma.channels.planar import (
    PlanarPath,
    multipath_planar_channel,
    planar_aperture_diagonal_m,
    planar_aperture_extents_m,
    planar_dma_positions,
    planar_rayleigh_distance_m,
)
from thz_dma.observations.multistrip import (
    array_input_noise_variance,
    effective_tensor_output_snr_db,
    observe_multistrip_schedule,
)
from thz_dma.surfaces.multistrip import (
    combiner_row_energy,
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


def _pilot_indices(num_subcarriers: int, pilot_count: int) -> np.ndarray:
    """Deterministically sample the full data band without duplicate indices."""

    if not 1 <= pilot_count <= num_subcarriers:
        raise ValueError("pilot count must lie in [1, num_subcarriers]")
    if pilot_count == 1:
        return np.array([num_subcarriers // 2], dtype=int)
    indices = np.rint(np.linspace(0, num_subcarriers - 1, pilot_count)).astype(int)
    if np.unique(indices).size != pilot_count:
        raise RuntimeError("pilot selection produced duplicate indices")
    return indices


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


def sample_stage2_paths(config: dict, rng: np.random.Generator) -> tuple[PlanarPath, ...]:
    """Draw LoS plus specular reflections as one independent scene unit."""

    channel = config["channel"]
    paths = int(channel["num_paths"])
    if paths < 1:
        raise ValueError("num_paths must be positive")
    los_range = rng.uniform(float(channel["range_min_m"]), float(channel["range_max_m"]))
    los_azimuth = np.deg2rad(
        rng.uniform(float(channel["azimuth_min_deg"]), float(channel["azimuth_max_deg"]))
    )
    los_elevation = np.deg2rad(
        rng.uniform(
            float(channel["elevation_min_deg"]),
            float(channel["elevation_max_deg"]),
        )
    )
    los_magnitude = float(channel["los_complex_gain_magnitude"])
    if los_magnitude <= 0.0:
        raise ValueError("los_complex_gain_magnitude must be positive")
    generated = [
        PlanarPath(
            range_m=float(los_range),
            azimuth_rad=float(los_azimuth),
            elevation_rad=float(los_elevation),
            complex_gain=los_magnitude * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi)),
        )
    ]
    for _ in range(paths - 1):
        azimuth_sign = rng.choice(np.array([-1.0, 1.0]))
        elevation_sign = rng.choice(np.array([-1.0, 1.0]))
        gain_db = rng.uniform(
            float(channel["reflection_gain_relative_min_db"]),
            float(channel["reflection_gain_relative_max_db"]),
        )
        generated.append(
            PlanarPath(
                range_m=float(
                    los_range
                    + rng.uniform(
                        float(channel["reflection_excess_range_min_m"]),
                        float(channel["reflection_excess_range_max_m"]),
                    )
                ),
                azimuth_rad=float(
                    los_azimuth
                    + azimuth_sign
                    * np.deg2rad(
                        rng.uniform(
                            float(channel["reflection_azimuth_separation_min_deg"]),
                            float(channel["reflection_azimuth_separation_max_deg"]),
                        )
                    )
                ),
                elevation_rad=float(
                    los_elevation
                    + elevation_sign
                    * np.deg2rad(
                        rng.uniform(
                            float(channel["reflection_elevation_separation_min_deg"]),
                            float(channel["reflection_elevation_separation_max_deg"]),
                        )
                    )
                ),
                complex_gain=float(10.0 ** (gain_db / 20.0))
                * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi)),
            )
        )
    return tuple(generated)


def tensor_unfolding_rank_residuals(tensor, rank: int) -> dict[str, float]:
    """SVD tail fractions of every unfolding of an ``S x J x K`` tensor."""

    values = np.asarray(tensor, dtype=complex)
    if values.ndim != 3 or values.size == 0:
        raise ValueError("tensor must have three non-empty dimensions")
    if rank < 1:
        raise ValueError("rank must be positive")
    residuals: dict[str, float] = {}
    for mode in range(3):
        unfolding = np.moveaxis(values, mode, 0).reshape(values.shape[mode], -1)
        singular_values = np.linalg.svd(unfolding, compute_uv=False)
        energy = float(np.sum(singular_values**2))
        tail = float(np.sum(singular_values[min(rank, singular_values.size) :] ** 2))
        residuals[f"mode_{mode}_rank_residual_fraction"] = (
            0.0 if energy == 0.0 else float(np.sqrt(tail / energy))
        )
    residuals["max_mode_rank_residual_fraction"] = max(residuals.values())
    return residuals


def _condition_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["physics_level"], row["schedule"]), []).append(row)
    result = []
    for (level, schedule), subset in grouped.items():
        result.append(
            {
                "physics_level": level,
                "schedule": schedule,
                "num_independent_scenes": len(subset),
                "mean_effective_output_snr_db": float(
                    np.mean([item["effective_output_snr_db"] for item in subset])
                ),
                "mean_max_mode_rank_residual_fraction": float(
                    np.mean(
                        [item["max_mode_rank_residual_fraction"] for item in subset]
                    )
                ),
                "max_max_mode_rank_residual_fraction": float(
                    np.max(
                        [item["max_mode_rank_residual_fraction"] for item in subset]
                    )
                ),
                "mean_combiner_row_energy": float(
                    np.mean([item["combiner_row_energy_mean"] for item in subset])
                ),
            }
        )
    return sorted(result, key=lambda item: (item["physics_level"], item["schedule"]))


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
