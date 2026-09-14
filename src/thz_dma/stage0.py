"""Direction-one deterministic Stage 0 physics checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from thz_dma.channels.atmosphere import AbsorptionTable, C0_M_PER_S, spreading_loss_db
from thz_dma.channels.geometry import (
    far_field_array_response,
    find_focus_peak_2d,
    focusing_gain,
    los_element_channel,
    ofdm_frequencies,
    rayleigh_distance_m,
    spherical_array_response,
    ula_positions,
)
from thz_dma.metrics.diagnostics import (
    finite_difference_real_jacobian,
    jacobian_diagnostics,
    row_space_diagnostics,
)
from thz_dma.surfaces.dma import (
    dma_weight_matrix,
    frequency_tiled_resonances,
    normalized_lorentzian_response,
    row_normalize,
    waveguide_response,
)


def _gain_loss_db(gain_linear: float) -> float:
    return float(-10.0 * np.log10(max(gain_linear, np.finfo(float).tiny)))


def run_stage0(config: dict, project_root: str | Path) -> tuple[list[dict], dict]:
    root = Path(project_root).resolve()
    seed = int(config["experiment"]["seed"])
    system = config["system"]
    atmosphere_config = config["atmosphere"]

    table_path = root / atmosphere_config["table_path"]
    absorption = AbsorptionTable.from_txt(
        table_path,
        max_frequency_hz=float(atmosphere_config["max_model_frequency_hz"]),
    )
    interpolation = str(atmosphere_config["interpolation"])
    raw: list[dict] = []
    summary: dict = {
        "experiment": config["experiment"],
        "absorption": {},
        "beam_defocus": {},
        "dma_projection_diversity": {},
        "local_observability": {},
    }

    num_subcarriers = int(system["num_subcarriers"])
    for window in atmosphere_config["windows"]:
        frequencies = ofdm_frequencies(
            float(window["carrier_frequency_hz"]),
            float(window["bandwidth_hz"]),
            num_subcarriers,
        )
        distance_m = float(window["distance_m"])
        kappa = np.asarray(
            absorption.coefficient(frequencies, method=interpolation), dtype=float
        )
        absorption_loss = np.asarray(
            absorption.loss_db(frequencies, distance_m, method=interpolation), dtype=float
        )
        center_frequency_hz = float(window["carrier_frequency_hz"])
        center_absorption_loss_db = float(
            absorption.loss_db(
                center_frequency_hz,
                distance_m,
                method=interpolation,
            )
        )
        spreading_loss = np.asarray(spreading_loss_db(frequencies, distance_m), dtype=float)
        for frequency, coefficient, loss_abs, loss_spread in zip(
            frequencies, kappa, absorption_loss, spreading_loss
        ):
            raw.append(
                {
                    "category": "absorption",
                    "scenario": window["name"],
                    "frequency_hz": float(frequency),
                    "distance_m": distance_m,
                    "metric": "kappa_per_m",
                    "value": float(coefficient),
                }
            )
            raw.append(
                {
                    "category": "absorption",
                    "scenario": window["name"],
                    "frequency_hz": float(frequency),
                    "distance_m": distance_m,
                    "metric": "absorption_loss_db",
                    "value": float(loss_abs),
                }
            )
            raw.append(
                {
                    "category": "absorption",
                    "scenario": window["name"],
                    "frequency_hz": float(frequency),
                    "distance_m": distance_m,
                    "metric": "spreading_loss_db",
                    "value": float(loss_spread),
                }
            )
        summary["absorption"][window["name"]] = {
            "distance_m": distance_m,
            "frequency_min_hz": float(frequencies[0]),
            "frequency_max_hz": float(frequencies[-1]),
            "center_absorption_loss_db": center_absorption_loss_db,
            "absorption_loss_min_db": float(np.min(absorption_loss)),
            "absorption_loss_max_db": float(np.max(absorption_loss)),
            "absorption_loss_span_db": float(np.ptp(absorption_loss)),
            "spreading_loss_min_db": float(np.min(spreading_loss)),
            "spreading_loss_max_db": float(np.max(spreading_loss)),
        }

    carrier_hz = float(system["carrier_frequency_hz"])
    bandwidth_hz = float(system["bandwidth_hz"])
    frequencies = ofdm_frequencies(carrier_hz, bandwidth_hz, num_subcarriers)
    wavelength_m = C0_M_PER_S / carrier_hz

    aperture = config["aperture"]
    aperture_positions = ula_positions(
        int(aperture["num_elements"]),
        float(aperture["spacing_wavelengths"]) * wavelength_m,
    )
    focus_range_m = float(aperture["focus_range_m"])
    focus_angle_rad = np.deg2rad(float(aperture["focus_angle_deg"]))
    focus_weights = spherical_array_response(
        carrier_hz,
        aperture_positions,
        focus_range_m,
        focus_angle_rad,
        normalize=True,
    )
    gain = focusing_gain(
        focus_weights,
        frequencies,
        aperture_positions,
        focus_range_m,
        focus_angle_rad,
    )
    minimum_gain_index = int(np.argmin(gain))
    far_field_weights = far_field_array_response(
        carrier_hz, aperture_positions, focus_angle_rad, normalize=True
    )
    far_field_gain = np.array(
        [
            abs(
                far_field_array_response(
                    frequency, aperture_positions, focus_angle_rad, normalize=True
                )
                @ np.conjugate(far_field_weights)
            )
            ** 2
            for frequency in frequencies
        ]
    )
    broadside_weights = spherical_array_response(
        carrier_hz,
        aperture_positions,
        focus_range_m,
        0.0,
        normalize=True,
    )
    broadside_spherical_gain = focusing_gain(
        broadside_weights,
        frequencies,
        aperture_positions,
        focus_range_m,
        0.0,
    )
    for frequency, spherical_gain, plane_gain, broadside_gain in zip(
        frequencies,
        gain,
        far_field_gain,
        broadside_spherical_gain,
    ):
        for scenario, gain_value in (
            ("off_axis_spherical", spherical_gain),
            ("off_axis_plane_wave", plane_gain),
            ("broadside_spherical", broadside_gain),
        ):
            raw.append(
                {
                    "category": "beam_defocus",
                    "scenario": scenario,
                    "frequency_hz": float(frequency),
                    "distance_m": focus_range_m,
                    "metric": "target_gain_linear",
                    "value": float(gain_value),
                }
            )

    ranges = np.linspace(
        float(aperture["range_grid_min_m"]),
        float(aperture["range_grid_max_m"]),
        int(aperture["range_grid_points"]),
    )
    angles = np.deg2rad(
        np.linspace(
            float(aperture["angle_grid_min_deg"]),
            float(aperture["angle_grid_max_deg"]),
            int(aperture["angle_grid_points"]),
        )
    )
    peaks = {}
    peak_frequencies = {
        "lower_edge": float(frequencies[0]),
        "center": carrier_hz,
        "upper_edge": float(frequencies[-1]),
    }
    target_gains = {
        "lower_edge": float(gain[0]),
        "center": 1.0,
        "upper_edge": float(gain[-1]),
    }
    for label, frequency in peak_frequencies.items():
        peak = find_focus_peak_2d(
            focus_weights, frequency, aperture_positions, ranges, angles
        )
        peak["angle_deg"] = float(np.rad2deg(peak.pop("angle_rad")))
        peak["frequency_hz"] = float(frequency)
        peak["target_point_gain_linear"] = target_gains[label]
        peak["target_point_gain_loss_db"] = _gain_loss_db(target_gains[label])
        peak["tracked_peak_gain_loss_db"] = _gain_loss_db(peak["gain_linear"])
        peak["range_shift_m"] = float(peak["range_m"] - focus_range_m)
        peak["angle_shift_deg"] = float(
            peak["angle_deg"] - float(aperture["focus_angle_deg"])
        )
        peaks[label] = peak
        for metric in (
            "target_point_gain_linear",
            "target_point_gain_loss_db",
            "tracked_peak_gain_loss_db",
            "range_shift_m",
            "angle_shift_deg",
        ):
            raw.append(
                {
                    "category": "beam_defocus_peak",
                    "scenario": label,
                    "frequency_hz": float(frequency),
                    "distance_m": focus_range_m,
                    "metric": metric,
                    "value": float(peak[metric]),
                }
            )
    rayleigh_distance = rayleigh_distance_m(carrier_hz, aperture_positions)
    summary["beam_defocus"] = {
        "aperture_m": float(np.ptp(aperture_positions)),
        "rayleigh_distance_m": rayleigh_distance,
        "focus_range_m": focus_range_m,
        "focus_angle_deg": float(aperture["focus_angle_deg"]),
        "focus_inside_rayleigh_region": bool(focus_range_m < rayleigh_distance),
        "relative_bandwidth": float(bandwidth_hz / carrier_hz),
        "lower_edge_target_gain_loss_db": _gain_loss_db(float(gain[0])),
        "upper_edge_target_gain_loss_db": _gain_loss_db(float(gain[-1])),
        "minimum_in_band_target_gain_linear": float(np.min(gain)),
        "minimum_in_band_target_gain_loss_db": _gain_loss_db(float(np.min(gain))),
        "minimum_in_band_frequency_hz": float(frequencies[minimum_gain_index]),
        "diagnostic_comparison": {
            "off_axis_spherical_lower_edge_loss_db": _gain_loss_db(float(gain[0])),
            "off_axis_spherical_upper_edge_loss_db": _gain_loss_db(float(gain[-1])),
            "off_axis_plane_wave_lower_edge_loss_db": _gain_loss_db(
                float(far_field_gain[0])
            ),
            "off_axis_plane_wave_upper_edge_loss_db": _gain_loss_db(
                float(far_field_gain[-1])
            ),
            "broadside_spherical_lower_edge_loss_db": _gain_loss_db(
                float(broadside_spherical_gain[0])
            ),
            "broadside_spherical_upper_edge_loss_db": _gain_loss_db(
                float(broadside_spherical_gain[-1])
            ),
        },
        "peaks": peaks,
    }

    dma = config["dma"]
    dma_num_elements = int(dma["num_elements"])
    dma_spacing_m = float(dma["spacing_wavelengths"]) * wavelength_m
    feed_positions = np.arange(dma_num_elements, dtype=float) * dma_spacing_m
    spatial_positions = feed_positions - feed_positions.mean()
    resonances = frequency_tiled_resonances(
        dma_num_elements,
        carrier_hz,
        float(dma["tuning_bandwidth_hz"]),
        seed=seed,
    )
    response_full = dma_weight_matrix(
        frequencies,
        resonances,
        feed_positions,
        quality_factor=float(dma["quality_factor"]),
        reference_frequency_hz=carrier_hz,
        waveguide_effective_index=float(dma["waveguide_effective_index"]),
        waveguide_end_power_fraction=float(dma["waveguide_end_power_fraction"]),
    )
    element_response = normalized_lorentzian_response(
        frequencies,
        resonances,
        quality_factor=float(dma["quality_factor"]),
        reference_frequency_hz=carrier_hz,
    )
    waveguide_frequency_response = waveguide_response(
        frequencies,
        feed_positions,
        effective_index=float(dma["waveguide_effective_index"]),
        end_power_fraction=float(dma["waveguide_end_power_fraction"]),
    )
    center_element = normalized_lorentzian_response(
        np.array([carrier_hz]),
        resonances,
        quality_factor=float(dma["quality_factor"]),
        reference_frequency_hz=carrier_hz,
    )
    center_waveguide = waveguide_response(
        np.array([carrier_hz]),
        feed_positions,
        effective_index=float(dma["waveguide_effective_index"]),
        end_power_fraction=float(dma["waveguide_end_power_fraction"]),
    )
    response_models = {
        "proportional_flat": np.repeat(center_element * center_waveguide, num_subcarriers, axis=0),
        "lorentzian_only": element_response * np.repeat(center_waveguide, num_subcarriers, axis=0),
        "waveguide_only": np.repeat(center_element, num_subcarriers, axis=0)
        * waveguide_frequency_response,
        "lorentzian_plus_waveguide": response_full,
    }
    normalized_models = {}
    for name, response in response_models.items():
        normalized = row_normalize(response)
        normalized_models[name] = normalized
        diagnostics = row_space_diagnostics(normalized)
        summary["dma_projection_diversity"][name] = diagnostics
        for index, singular_value in enumerate(diagnostics["singular_values"]):
            raw.append(
                {
                    "category": "dma_projection_diversity",
                    "scenario": name,
                    "frequency_hz": "",
                    "distance_m": "",
                    "metric": f"singular_value_{index}",
                    "value": float(singular_value),
                }
            )

    observability = config["observability"]
    parameter_point = np.array(
        [
            float(observability["gain_real"]),
            float(observability["gain_imag"]),
            float(observability["range_m"]),
            np.deg2rad(float(observability["angle_deg"])),
        ],
        dtype=float,
    )
    steps = np.array(
        [
            float(observability["step_gain"]),
            float(observability["step_gain"]),
            float(observability["step_range_m"]),
            np.deg2rad(float(observability["step_angle_deg"])),
        ],
        dtype=float,
    )

    for response_name in ("proportional_flat", "lorentzian_plus_waveguide"):
        weights = normalized_models[response_name]
        for include_absorption in (False, True):
            scenario = f"{response_name}__absorption_{'on' if include_absorption else 'off'}"

            def observation(parameters):
                complex_gain = parameters[0] + 1j * parameters[1]
                channel = los_element_channel(
                    frequencies,
                    spatial_positions,
                    float(parameters[2]),
                    float(parameters[3]),
                    absorption_table=absorption,
                    include_absorption=include_absorption,
                    interpolation=interpolation,
                )
                return complex_gain * np.sum(np.conjugate(weights) * channel, axis=1)

            jacobian = finite_difference_real_jacobian(observation, parameter_point, steps)
            diagnostics = jacobian_diagnostics(jacobian)
            summary["local_observability"][scenario] = diagnostics
            for metric in (
                "numerical_rank",
                "condition_number",
                "minimum_singular_value",
                "normalized_gram_minimum_eigenvalue",
            ):
                raw.append(
                    {
                        "category": "local_observability",
                        "scenario": scenario,
                        "frequency_hz": "",
                        "distance_m": float(observability["range_m"]),
                        "metric": metric,
                        "value": float(diagnostics[metric]),
                    }
                )

    summary["model_metadata"] = {
        "absorption_table_path": str(absorption.source_path),
        "absorption_table_frequency_range_hz": list(absorption.frequency_range_hz),
        "ofdm_frequency_min_hz": float(frequencies[0]),
        "ofdm_frequency_max_hz": float(frequencies[-1]),
        "ofdm_subcarrier_spacing_hz": float(bandwidth_hz / num_subcarriers),
        "dma_resonance_min_hz": float(np.min(resonances)),
        "dma_resonance_max_hz": float(np.max(resonances)),
    }
    return raw, summary
