"""Wideband spherical-wave geometry and focusing diagnostics."""

from __future__ import annotations

import numpy as np

from .atmosphere import AbsorptionTable, C0_M_PER_S, free_space_amplitude


def ofdm_frequencies(
    carrier_frequency_hz: float,
    bandwidth_hz: float,
    num_subcarriers: int,
) -> np.ndarray:
    """Return a centered OFDM frequency grid with Delta-f = B/K."""

    if carrier_frequency_hz <= 0.0 or bandwidth_hz <= 0.0:
        raise ValueError("Carrier frequency and bandwidth must be positive")
    if num_subcarriers < 1:
        raise ValueError("num_subcarriers must be positive")
    spacing = bandwidth_hz / num_subcarriers
    index = np.arange(num_subcarriers, dtype=float) - (num_subcarriers - 1) / 2.0
    frequency = carrier_frequency_hz + index * spacing
    if frequency[0] <= 0.0:
        raise ValueError("OFDM grid contains a non-positive frequency")
    return frequency


def ula_positions(num_elements: int, spacing_m: float, *, centered: bool = True) -> np.ndarray:
    if num_elements < 1 or spacing_m <= 0.0:
        raise ValueError("num_elements and spacing_m must be positive")
    positions = np.arange(num_elements, dtype=float) * spacing_m
    if centered:
        positions -= positions.mean()
    return positions


def spherical_distances(
    element_positions_m,
    range_m: float,
    angle_rad: float,
) -> np.ndarray:
    """Distance from ULA elements to a point; angle is measured from broadside."""

    if range_m <= 0.0:
        raise ValueError("range_m must be positive")
    positions = np.asarray(element_positions_m, dtype=float)
    radicand = (
        range_m**2
        + positions**2
        - 2.0 * range_m * positions * np.sin(angle_rad)
    )
    if np.any(radicand <= 0.0):
        raise ValueError("Point coincides with or crosses an array element")
    return np.sqrt(radicand)


def spherical_array_response(
    frequency_hz,
    element_positions_m,
    range_m: float,
    angle_rad: float,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Array response using only element-to-reference distance differences."""

    original_scalar = np.isscalar(frequency_hz)
    frequency = np.atleast_1d(np.asarray(frequency_hz, dtype=float))
    if np.any(frequency <= 0.0):
        raise ValueError("Frequencies must be positive")
    distance = spherical_distances(element_positions_m, range_m, angle_rad)
    relative_distance = distance - range_m
    response = np.exp(
        -1j * 2.0 * np.pi * frequency[:, None] * relative_distance[None, :] / C0_M_PER_S
    )
    if normalize:
        response /= np.sqrt(response.shape[1])
    return response[0] if original_scalar else response


def far_field_array_response(
    frequency_hz,
    element_positions_m,
    angle_rad: float,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Plane-wave limit consistent with ``spherical_array_response``."""

    positions = np.asarray(element_positions_m, dtype=float)
    response = np.exp(
        1j * 2.0 * np.pi * float(frequency_hz) * positions * np.sin(angle_rad) / C0_M_PER_S
    )
    if normalize:
        response /= np.sqrt(response.size)
    return response


def rayleigh_distance_m(frequency_hz: float, element_positions_m) -> float:
    aperture_m = float(np.max(element_positions_m) - np.min(element_positions_m))
    wavelength_m = C0_M_PER_S / frequency_hz
    return 2.0 * aperture_m**2 / wavelength_m


def focusing_gain(
    focus_weights,
    frequencies_hz,
    element_positions_m,
    range_m: float,
    angle_rad: float,
) -> np.ndarray:
    """Normalized gain at a fixed physical point for center-frequency weights."""

    weights = np.asarray(focus_weights, dtype=complex)
    weights = weights / np.linalg.norm(weights)
    response = spherical_array_response(
        frequencies_hz, element_positions_m, range_m, angle_rad, normalize=True
    )
    return np.abs(response @ np.conjugate(weights)) ** 2


def find_focus_peak_2d(
    focus_weights,
    frequency_hz: float,
    element_positions_m,
    ranges_m,
    angles_rad,
) -> dict[str, float]:
    """Find the strongest point on a supplied range-angle grid."""

    weights = np.asarray(focus_weights, dtype=complex)
    weights = weights / np.linalg.norm(weights)
    positions = np.asarray(element_positions_m, dtype=float)
    ranges = np.asarray(ranges_m, dtype=float)
    angles = np.asarray(angles_rad, dtype=float)
    best_gain = -np.inf
    best_range = np.nan
    best_angle = np.nan
    for angle in angles:
        distance = np.sqrt(
            ranges[:, None] ** 2
            + positions[None, :] ** 2
            - 2.0 * ranges[:, None] * positions[None, :] * np.sin(angle)
        )
        relative_distance = distance - ranges[:, None]
        response = np.exp(
            -1j
            * 2.0
            * np.pi
            * frequency_hz
            * relative_distance
            / C0_M_PER_S
        ) / np.sqrt(positions.size)
        gain = np.abs(response @ np.conjugate(weights)) ** 2
        index = int(np.argmax(gain))
        if gain[index] > best_gain:
            best_gain = float(gain[index])
            best_range = float(ranges[index])
            best_angle = float(angle)
    return {
        "range_m": best_range,
        "angle_rad": best_angle,
        "gain_linear": best_gain,
    }


def los_element_channel(
    frequencies_hz,
    element_positions_m,
    range_m: float,
    angle_rad: float,
    *,
    absorption_table: AbsorptionTable | None = None,
    include_absorption: bool = True,
    interpolation: str = "linear",
) -> np.ndarray:
    """Element-domain coherent LoS channel with spreading and absorption."""

    frequencies = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
    distance = spherical_distances(element_positions_m, range_m, angle_rad)
    amplitude = free_space_amplitude(frequencies[:, None], distance[None, :])
    if include_absorption:
        if absorption_table is None:
            raise ValueError("absorption_table is required when absorption is enabled")
        kappa = np.asarray(
            absorption_table.coefficient(frequencies, method=interpolation), dtype=float
        )
        amplitude *= np.exp(-0.5 * kappa[:, None] * distance[None, :])
    phase = np.exp(
        -1j * 2.0 * np.pi * frequencies[:, None] * distance[None, :] / C0_M_PER_S
    )
    return amplitude * phase


def far_field_los_element_channel(
    frequencies_hz,
    element_positions_m,
    range_m: float,
    angle_rad: float,
    *,
    absorption_table: AbsorptionTable | None = None,
    include_absorption: bool = True,
    interpolation: str = "linear",
) -> np.ndarray:
    """Plane-wave LoS channel retaining wideband delay and path loss.

    This is the far-field approximation of :func:`los_element_channel`: all
    elements share the reference distance for spreading and absorption, while
    the linear spatial phase is retained.  It is primarily used as a deliberate
    model-mismatch baseline for large-aperture near-field experiments.
    """

    frequencies = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
    positions = np.asarray(element_positions_m, dtype=float)
    if np.any(frequencies <= 0.0) or positions.ndim != 1 or positions.size == 0:
        raise ValueError("invalid frequencies or empty aperture")
    if range_m <= 0.0:
        raise ValueError("range_m must be positive")

    amplitude = free_space_amplitude(frequencies, float(range_m))
    if include_absorption:
        if absorption_table is None:
            raise ValueError("absorption_table is required when absorption is enabled")
        kappa = np.asarray(
            absorption_table.coefficient(frequencies, method=interpolation), dtype=float
        )
        amplitude *= np.exp(-0.5 * kappa * float(range_m))

    delay_phase = np.exp(
        -1j * 2.0 * np.pi * frequencies * float(range_m) / C0_M_PER_S
    )
    spatial_phase = np.exp(
        1j
        * 2.0
        * np.pi
        * frequencies[:, None]
        * positions[None, :]
        * np.sin(angle_rad)
        / C0_M_PER_S
    )
    return amplitude[:, None] * delay_phase[:, None] * spatial_phase
