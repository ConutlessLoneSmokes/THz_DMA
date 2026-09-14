"""Normalized Lorentzian DMA and dispersive waveguide models."""

from __future__ import annotations

import numpy as np

from thz_dma.channels.atmosphere import C0_M_PER_S


def normalized_lorentzian_response(
    frequencies_hz,
    resonant_frequencies_hz,
    *,
    quality_factor: float,
    reference_frequency_hz: float,
) -> np.ndarray:
    """Normalized magnetic polarizability used by Gavras/Carlson models.

    With Gamma = 2 pi f_ref / Q, the response is

        Gamma f / (2 pi (f_r^2 - f^2) + j Gamma f),

    and has unit magnitude at f = f_r = f_ref.
    """

    if quality_factor <= 0.0 or reference_frequency_hz <= 0.0:
        raise ValueError("quality_factor and reference_frequency_hz must be positive")
    frequency = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))[:, None]
    resonance = np.atleast_1d(np.asarray(resonant_frequencies_hz, dtype=float))[None, :]
    if np.any(frequency <= 0.0) or np.any(resonance <= 0.0):
        raise ValueError("Frequencies must be positive")
    damping_hz = 2.0 * np.pi * reference_frequency_hz / quality_factor
    denominator = (
        2.0 * np.pi * (resonance**2 - frequency**2)
        + 1j * damping_hz * frequency
    )
    return damping_hz * frequency / denominator


def waveguide_response(
    frequencies_hz,
    distances_from_feed_m,
    *,
    effective_index: float,
    end_power_fraction: float,
) -> np.ndarray:
    """Dispersive phase and exponential field attenuation along one feed."""

    if effective_index <= 0.0:
        raise ValueError("effective_index must be positive")
    if not 0.0 < end_power_fraction <= 1.0:
        raise ValueError("end_power_fraction must lie in (0, 1]")
    frequency = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))[:, None]
    distance = np.atleast_1d(np.asarray(distances_from_feed_m, dtype=float))[None, :]
    if np.any(frequency <= 0.0) or np.any(distance < 0.0):
        raise ValueError("Frequencies must be positive and distances non-negative")
    length = float(np.max(distance))
    attenuation_np_per_m = (
        0.0
        if length == 0.0 or end_power_fraction == 1.0
        else -np.log(end_power_fraction) / (2.0 * length)
    )
    beta_rad_per_m = 2.0 * np.pi * frequency * effective_index / C0_M_PER_S
    return np.exp(-attenuation_np_per_m * distance - 1j * beta_rad_per_m * distance)


def dma_weight_matrix(
    frequencies_hz,
    resonant_frequencies_hz,
    distances_from_feed_m,
    *,
    quality_factor: float,
    reference_frequency_hz: float,
    waveguide_effective_index: float,
    waveguide_end_power_fraction: float,
) -> np.ndarray:
    element = normalized_lorentzian_response(
        frequencies_hz,
        resonant_frequencies_hz,
        quality_factor=quality_factor,
        reference_frequency_hz=reference_frequency_hz,
    )
    waveguide = waveguide_response(
        frequencies_hz,
        distances_from_feed_m,
        effective_index=waveguide_effective_index,
        end_power_fraction=waveguide_end_power_fraction,
    )
    if element.shape != waveguide.shape:
        raise ValueError("Element and waveguide response shapes do not match")
    return element * waveguide


def frequency_tiled_resonances(
    num_elements: int,
    center_frequency_hz: float,
    tuning_bandwidth_hz: float,
    *,
    seed: int,
) -> np.ndarray:
    """Deterministic exploratory shared-bias configuration across the tuning band."""

    if num_elements < 1 or center_frequency_hz <= 0.0 or tuning_bandwidth_hz < 0.0:
        raise ValueError("Invalid resonance configuration")
    low = center_frequency_hz - tuning_bandwidth_hz / 2.0
    high = center_frequency_hz + tuning_bandwidth_hz / 2.0
    if low <= 0.0:
        raise ValueError("Tuning range contains a non-positive frequency")
    resonances = np.linspace(low, high, num_elements, dtype=float)
    return np.random.default_rng(seed).permutation(resonances)


def row_normalize(matrix) -> np.ndarray:
    """Noise-whitening proxy for a single element-noise-limited RF output."""

    values = np.asarray(matrix, dtype=complex)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("Cannot normalize a zero DMA response row")
    return values / norm

