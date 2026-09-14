"""Pilot observations for a planar DMA with one RF output per microstrip."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pilots import complex_standard_normal


@dataclass(frozen=True)
class MultiStripObservation:
    """Clean/noisy observations in canonical ``(S, J, K_p)`` tensor order."""

    clean: np.ndarray
    observed: np.ndarray
    output_noise_variance: np.ndarray


def array_input_noise_variance(element_channel, snr_db: float) -> float:
    """Noise per element from ideal full-digital matched-array input SNR."""

    channel = np.asarray(element_channel, dtype=complex)
    if channel.ndim != 3 or channel.size == 0:
        raise ValueError("element_channel must have shape (K, S, E)")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    reference_power = float(np.mean(np.sum(np.abs(channel) ** 2, axis=(1, 2))))
    if reference_power <= 0.0:
        raise ValueError("element_channel has zero reference power")
    return reference_power / (10.0 ** (float(snr_db) / 10.0))


def observe_multistrip_schedule(
    weights,
    element_channel,
    *,
    pilot_energy_per_re: float,
    element_noise_variance: float,
    rf_noise_variance: float = 0.0,
    rng: np.random.Generator | None = None,
) -> MultiStripObservation:
    """Apply a ``J x K_p`` acquisition schedule without combiner normalisation.

    ``weights`` has shape ``(J, K_p, S, E)`` and ``element_channel`` has shape
    ``(K_p, S, E)``.  Element-domain thermal noise is independently generated
    for every configuration/resource element.  RF-chain noise is added after
    combining.  Pilot energy scales the signal but not either receiver-noise
    contribution.
    """

    combiner = np.asarray(weights, dtype=complex)
    channel = np.asarray(element_channel, dtype=complex)
    if combiner.ndim != 4:
        raise ValueError("weights must have shape (J, K_p, S, E)")
    if channel.ndim != 3 or combiner.shape[1:] != channel.shape:
        raise ValueError("channel shape must match the final three weight dimensions")
    if pilot_energy_per_re <= 0.0 or not np.isfinite(pilot_energy_per_re):
        raise ValueError("pilot_energy_per_re must be positive and finite")
    if element_noise_variance < 0.0 or not np.isfinite(element_noise_variance):
        raise ValueError("element_noise_variance must be non-negative and finite")
    if rf_noise_variance < 0.0 or not np.isfinite(rf_noise_variance):
        raise ValueError("rf_noise_variance must be non-negative and finite")

    clean_jks = np.sqrt(float(pilot_energy_per_re)) * np.sum(
        np.conjugate(combiner) * channel[None, :, :, :], axis=-1
    )
    output_variance_jks = (
        float(element_noise_variance) * np.sum(np.abs(combiner) ** 2, axis=-1)
        + float(rf_noise_variance)
    )
    noisy_jks = clean_jks.copy()
    if element_noise_variance > 0.0 or rf_noise_variance > 0.0:
        generator = rng if rng is not None else np.random.default_rng()
        if element_noise_variance > 0.0:
            element_noise = np.sqrt(float(element_noise_variance)) * (
                complex_standard_normal(generator, combiner.shape)
            )
            noisy_jks += np.sum(np.conjugate(combiner) * element_noise, axis=-1)
        if rf_noise_variance > 0.0:
            noisy_jks += np.sqrt(float(rf_noise_variance)) * complex_standard_normal(
                generator, clean_jks.shape
            )

    # Tensor algorithms use receive-output/configuration/frequency order.
    return MultiStripObservation(
        clean=np.transpose(clean_jks, (2, 0, 1)),
        observed=np.transpose(noisy_jks, (2, 0, 1)),
        output_noise_variance=np.transpose(output_variance_jks, (2, 0, 1)),
    )


def effective_tensor_output_snr_db(
    clean_observation, output_noise_variance
) -> float:
    """Average post-combiner SNR across all scheduled RF outputs and pilots."""

    clean = np.asarray(clean_observation, dtype=complex)
    variance = np.asarray(output_noise_variance, dtype=float)
    if clean.shape != variance.shape or clean.size == 0:
        raise ValueError("clean observation and noise variance must share a non-empty shape")
    total_noise = float(np.sum(variance))
    if total_noise <= 0.0 or not np.isfinite(total_noise):
        raise ValueError("total output-noise variance must be positive and finite")
    return float(10.0 * np.log10(np.sum(np.abs(clean) ** 2) / total_noise))
