"""Multi-microstrip DMA state designs and frequency-dependent weights."""

from __future__ import annotations

import numpy as np

from .dma import dma_weight_matrix


def feed_distances_m(elements_per_microstrip: int, element_spacing_m: float) -> np.ndarray:
    """Distance of each meta-atom from the feed at one strip edge."""

    if elements_per_microstrip < 1 or element_spacing_m <= 0.0:
        raise ValueError("invalid microstrip geometry")
    return np.arange(elements_per_microstrip, dtype=float) * float(element_spacing_m)


def grouped_sector_resonances(
    num_microstrips: int,
    elements_per_microstrip: int,
    center_frequency_hz: float,
    tuning_bandwidth_hz: float,
    *,
    num_sectors: int,
) -> np.ndarray:
    """One-shot state whose strip groups cover distinct frequency sectors.

    The returned state has shape ``(1, S, E)``.  It is deliberately simple and
    deterministic so a later single-shot method adapter can use the same state.
    """

    if num_microstrips < 1 or elements_per_microstrip < 1:
        raise ValueError("array dimensions must be positive")
    if not 1 <= num_sectors <= num_microstrips:
        raise ValueError("num_sectors must lie in [1, num_microstrips]")
    if center_frequency_hz <= 0.0 or tuning_bandwidth_hz < 0.0:
        raise ValueError("invalid tuning band")
    low = center_frequency_hz - tuning_bandwidth_hz / 2.0
    high = center_frequency_hz + tuning_bandwidth_hz / 2.0
    if low <= 0.0:
        raise ValueError("tuning band contains a non-positive frequency")
    edges = np.linspace(low, high, num_sectors + 1, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    sector_index = np.minimum(
        np.floor(np.arange(num_microstrips) * num_sectors / num_microstrips).astype(int),
        num_sectors - 1,
    )
    strip_resonance = centers[sector_index]
    return np.broadcast_to(
        strip_resonance[None, :, None],
        (1, num_microstrips, elements_per_microstrip),
    ).copy()


def cyclic_frequency_resonance_states(
    num_configurations: int,
    num_microstrips: int,
    elements_per_microstrip: int,
    center_frequency_hz: float,
    tuning_bandwidth_hz: float,
) -> np.ndarray:
    """Structured multi-configuration states for a receive-side tensor schedule.

    Every strip uses the same within-strip resonance pattern.  Configurations
    cyclically shift that pattern.  Under the compatibility plane-wave channel
    and frequency-flat hardware switch this makes the ``S x J x K`` clean
    observation exactly rank-P; the target THz model is free to break it.
    """

    if num_configurations < 1 or num_microstrips < 1 or elements_per_microstrip < 1:
        raise ValueError("array and schedule dimensions must be positive")
    if center_frequency_hz <= 0.0 or tuning_bandwidth_hz < 0.0:
        raise ValueError("invalid tuning band")
    low = center_frequency_hz - tuning_bandwidth_hz / 2.0
    high = center_frequency_hz + tuning_bandwidth_hz / 2.0
    if low <= 0.0:
        raise ValueError("tuning band contains a non-positive frequency")
    base = np.linspace(low, high, elements_per_microstrip, endpoint=True, dtype=float)
    states = np.empty(
        (num_configurations, num_microstrips, elements_per_microstrip), dtype=float
    )
    for configuration in range(num_configurations):
        shift = int(np.floor(configuration * elements_per_microstrip / num_configurations))
        pattern = np.roll(base, shift)
        states[configuration] = np.broadcast_to(
            pattern[None, :], (num_microstrips, elements_per_microstrip)
        )
    return states


def multistrip_dma_weights(
    frequencies_hz,
    resonant_frequencies_hz,
    distances_from_feed_m,
    *,
    quality_factor: float,
    reference_frequency_hz: float,
    waveguide_effective_index: float,
    waveguide_end_power_fraction: float,
    frequency_flat: bool = False,
) -> np.ndarray:
    """Return unnormalised DMA weights with shape ``(J, K, S, E)``.

    No row normalisation is applied: Lorentzian attenuation, waveguide loss and
    the resulting method-dependent output-noise gain remain in the experiment.
    """

    frequencies = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
    resonances = np.asarray(resonant_frequencies_hz, dtype=float)
    distances = np.atleast_1d(np.asarray(distances_from_feed_m, dtype=float))
    if frequencies.ndim != 1 or frequencies.size == 0 or np.any(frequencies <= 0.0):
        raise ValueError("frequencies_hz must be a non-empty positive vector")
    if resonances.ndim != 3:
        raise ValueError("resonant_frequencies_hz must have shape (J, S, E)")
    configurations, strips, elements = resonances.shape
    if distances.shape != (elements,) or np.any(distances < 0.0):
        raise ValueError("distances_from_feed_m must contain one value per element")
    evaluation_frequencies = (
        np.array([float(reference_frequency_hz)], dtype=float)
        if frequency_flat
        else frequencies
    )
    weights = np.empty(
        (configurations, frequencies.size, strips, elements), dtype=complex
    )
    for configuration in range(configurations):
        for strip in range(strips):
            response = dma_weight_matrix(
                evaluation_frequencies,
                resonances[configuration, strip],
                distances,
                quality_factor=quality_factor,
                reference_frequency_hz=reference_frequency_hz,
                waveguide_effective_index=waveguide_effective_index,
                waveguide_end_power_fraction=waveguide_end_power_fraction,
            )
            if frequency_flat:
                response = np.repeat(response, frequencies.size, axis=0)
            weights[configuration, :, strip, :] = response
    if not np.all(np.isfinite(weights)):
        raise ValueError("DMA weights contain non-finite values")
    return weights


def combiner_row_energy(weights) -> np.ndarray:
    """Return ``sum_e |w|^2`` for each configuration/frequency/RF output."""

    values = np.asarray(weights, dtype=complex)
    if values.ndim != 4 or values.shape[-1] == 0:
        raise ValueError("weights must have shape (J, K, S, E)")
    return np.sum(np.abs(values) ** 2, axis=-1)
