"""Feasible planar-DMA data-state selection and link metrics."""

from __future__ import annotations

import numpy as np

from thz_dma.surfaces.dma import frequency_tiled_resonances
from thz_dma.surfaces.multistrip import multistrip_dma_weights


def build_planar_data_codebook(
    *,
    carrier_frequency_hz: float,
    num_microstrips: int,
    elements_per_microstrip: int,
    tuning_bandwidth_hz: float,
    feed_distances_m,
    quality_factor: float,
    waveguide_effective_index: float,
    waveguide_end_power_fraction: float,
    codebook_size: int,
    seed_start: int,
) -> tuple[np.ndarray, dict]:
    """Build independent feasible center-frequency states without normalization."""

    if codebook_size < 2:
        raise ValueError("data codebook must contain at least two states")
    resonances = np.empty(
        (codebook_size, num_microstrips, elements_per_microstrip), dtype=float
    )
    for state in range(codebook_size):
        for strip in range(num_microstrips):
            resonances[state, strip] = frequency_tiled_resonances(
                elements_per_microstrip,
                carrier_frequency_hz,
                tuning_bandwidth_hz,
                seed=seed_start + state * num_microstrips + strip,
            )
    weights = multistrip_dma_weights(
        np.array([carrier_frequency_hz], dtype=float),
        resonances,
        feed_distances_m,
        quality_factor=quality_factor,
        reference_frequency_hz=carrier_frequency_hz,
        waveguide_effective_index=waveguide_effective_index,
        waveguide_end_power_fraction=waveguide_end_power_fraction,
        frequency_flat=False,
    )[:, 0]
    return weights, {
        "type": "independent feasible center-frequency multi-strip DMA states",
        "size": int(codebook_size),
        "seed_start": int(seed_start),
        "row_normalized": False,
        "analog_state_selection_rule": "maximize predicted noise-whitened RF-output SNR",
        "digital_combiner": "estimated effective-channel MVDR with diagonal known noise",
    }


def feasible_planar_data_metrics(
    true_channel,
    estimated_channel,
    codebook_weights,
    *,
    element_noise_variance: float,
    rf_noise_variance: float,
    coherence_symbols: float,
    training_symbols: float,
    switch_guard_symbol_equivalents: float,
) -> dict[str, float | int]:
    """Select a feasible analog state and evaluate its actual digital-combined SNR."""

    truth = np.asarray(true_channel, dtype=complex)
    estimate = np.asarray(estimated_channel, dtype=complex)
    weights = np.asarray(codebook_weights, dtype=complex)
    if truth.ndim != 2 or estimate.shape != truth.shape:
        raise ValueError("true and estimated channels must share shape (S, E)")
    if weights.ndim != 3 or weights.shape[1:] != truth.shape:
        raise ValueError("codebook weights must have shape (M, S, E)")
    if element_noise_variance <= 0.0 or rf_noise_variance < 0.0:
        raise ValueError("noise variances are invalid")
    if coherence_symbols <= 0.0 or training_symbols < 0.0 or switch_guard_symbol_equivalents < 0.0:
        raise ValueError("resource durations are invalid")

    effective_true = np.einsum(
        "mse,se->ms", np.conjugate(weights), truth, optimize=True
    )
    effective_estimate = np.einsum(
        "mse,se->ms", np.conjugate(weights), estimate, optimize=True
    )
    output_variance = (
        float(element_noise_variance) * np.sum(np.abs(weights) ** 2, axis=-1)
        + float(rf_noise_variance)
    )
    predicted_snr = np.sum(
        np.abs(effective_estimate) ** 2 / output_variance, axis=1
    )
    oracle_snr_by_state = np.sum(
        np.abs(effective_true) ** 2 / output_variance, axis=1
    )
    selected_index = int(np.argmax(predicted_snr))
    oracle_index = int(np.argmax(oracle_snr_by_state))
    digital_weights = effective_estimate[selected_index] / output_variance[selected_index]
    digital_noise = float(
        np.sum(np.abs(digital_weights) ** 2 * output_variance[selected_index])
    )
    if digital_noise <= 0.0 or not np.isfinite(digital_noise):
        raise ValueError("estimated digital combiner has zero or invalid energy")
    selected_signal = float(
        np.abs(np.vdot(digital_weights, effective_true[selected_index])) ** 2
    )
    selected_snr = selected_signal / digital_noise
    oracle_snr = float(oracle_snr_by_state[oracle_index])
    ideal_digital_snr = float(
        np.sum(np.abs(truth) ** 2) / float(element_noise_variance)
    )
    if selected_snr < 0.0 or oracle_snr <= 0.0 or ideal_digital_snr <= 0.0:
        raise ValueError("data SNR is invalid")
    beam_ratio = float(np.clip(selected_snr / oracle_snr, 0.0, 1.0))
    overhead = float(training_symbols + switch_guard_symbol_equivalents)
    payload_fraction = float(np.clip(1.0 - overhead / coherence_symbols, 0.0, 1.0))
    gross_rate = float(np.log2(1.0 + selected_snr))
    oracle_rate = float(np.log2(1.0 + oracle_snr))
    return {
        "selected_data_state_index": selected_index,
        "oracle_data_state_index": oracle_index,
        "data_state_selection_correct": int(selected_index == oracle_index),
        "feasible_data_snr_linear": float(selected_snr),
        "oracle_feasible_data_snr_linear": oracle_snr,
        "ideal_digital_data_snr_linear": ideal_digital_snr,
        "feasible_beam_gain_ratio_linear": beam_ratio,
        "feasible_beam_loss_db": float(
            -10.0 * np.log10(max(beam_ratio, np.finfo(float).tiny))
        ),
        "oracle_feasible_fraction_of_digital_snr": float(
            np.clip(oracle_snr / ideal_digital_snr, 0.0, 1.0)
        ),
        "feasible_gross_rate_bps_hz": gross_rate,
        "oracle_feasible_gross_rate_bps_hz": oracle_rate,
        "payload_fraction": payload_fraction,
        "net_rate_bps_hz": payload_fraction * gross_rate,
        "oracle_net_rate_bps_hz": payload_fraction * oracle_rate,
    }
