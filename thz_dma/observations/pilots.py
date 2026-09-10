"""Single-RF-output pilot observation utilities."""

from __future__ import annotations

import numpy as np


def complex_standard_normal(rng: np.random.Generator, shape) -> np.ndarray:
    """Return circular complex Gaussian samples with unit variance."""

    return (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    ) / np.sqrt(2.0)


def ideal_array_noise_variance(element_channel, snr_db: float) -> float:
    """Set element-noise variance from average ideal matched-array SNR.

    For a unit-norm ideal combiner, the maximum per-subcarrier signal power is
    ``||h_k||^2`` and the output noise variance equals the element-noise
    variance.  This definition provides one common input-SNR block for all
    candidate DMA combiners without normalizing each method separately.
    """

    channel = np.asarray(element_channel, dtype=complex)
    if channel.ndim != 2 or channel.size == 0:
        raise ValueError("element_channel must be a non-empty K-by-N matrix")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    reference_power = float(np.mean(np.sum(np.abs(channel) ** 2, axis=1)))
    if reference_power <= 0.0:
        raise ValueError("element_channel has zero reference power")
    return reference_power / (10.0 ** (float(snr_db) / 10.0))


def matched_output_noise_variance(clean_observation, snr_db: float) -> float:
    """Set output-noise variance to a requested post-combiner SNR.

    Combiner rows are assumed to have unit norm, so the returned output
    variance is also the variance of independent element-domain noise.  This
    diagnostic removes method-dependent projection gain while preserving each
    method's frequency-domain template shape.
    """

    signal = np.asarray(clean_observation, dtype=complex)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("clean_observation must be a non-empty vector")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    signal_power = float(np.mean(np.abs(signal) ** 2))
    if signal_power <= 0.0:
        raise ValueError("clean_observation has zero power")
    return signal_power / (10.0 ** (float(snr_db) / 10.0))


def combine_pilot_observation(
    weights,
    element_channel,
    *,
    complex_gain: complex = 1.0 + 0.0j,
    element_noise=None,
    pilot_symbol: complex = 1.0 + 0.0j,
) -> np.ndarray:
    """Apply frequency-indexed unit-RF-chain combining to pilots.

    ``weights`` and ``element_channel`` both have shape ``(K, N)`` and the
    returned observation is ``y_k = w_k^H (alpha h_k x_k + n_k)``.
    """

    combiner = np.asarray(weights, dtype=complex)
    channel = np.asarray(element_channel, dtype=complex)
    if combiner.ndim != 2 or combiner.shape != channel.shape:
        raise ValueError("weights and element_channel must have the same K-by-N shape")
    received = complex(complex_gain) * complex(pilot_symbol) * channel
    if element_noise is not None:
        noise = np.asarray(element_noise, dtype=complex)
        if noise.shape != channel.shape:
            raise ValueError("element_noise must match element_channel")
        received = received + noise
    return np.sum(np.conjugate(combiner) * received, axis=1)


def effective_output_snr_db(clean_observation, noise_variance: float) -> float:
    """Average post-combiner SNR for unit-norm frequency rows."""

    signal = np.asarray(clean_observation, dtype=complex)
    if signal.ndim != 1 or signal.size == 0:
        raise ValueError("clean_observation must be a non-empty vector")
    if noise_variance <= 0.0 or not np.isfinite(noise_variance):
        raise ValueError("noise_variance must be positive and finite")
    return float(10.0 * np.log10(np.mean(np.abs(signal) ** 2) / noise_variance))
