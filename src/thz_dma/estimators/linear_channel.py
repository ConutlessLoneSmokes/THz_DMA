"""Noise-aware linear CSI estimators; no sparse path fitting."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh

from thz_dma.channels.planar import PlanarPath, spherical_planar_path_channel
from thz_dma.estimators.adaptive_grid import adaptive_settings, _sample_truth_around_centers


def conditional_planar_covariance_basis(platform, config, scene_record, *, draws, rng):
    """Independent conditional geometry integration, with analytic gain moments.

    Only pre-observation centers/profiles are read; test offsets and gains are not.
    Independent uniform gain phases make the mean and cross-path moments zero.
    """
    if draws < 2:
        raise ValueError("at least two covariance draws are required")
    centers = tuple(PlanarPath(*point) for point in scene_record["centers_m_rad_rad"])
    profile = {"settings": adaptive_settings(config),
               "schedule_profiles": scene_record["schedule_profiles"]}
    channel = config["channel"]
    low, high = (float(channel[f"reflection_gain_relative_{edge}_db"])
                 for edge in ("min", "max"))
    reflection_power = (10 ** (high / 10) - 10 ** (low / 10)) / ((high - low) * np.log(10) / 10) if high != low else 10 ** (low / 10)
    powers = [float(channel["los_complex_gain_magnitude"]) ** 2] + [reflection_power] * (len(centers) - 1)
    basis = np.empty((draws * len(centers), len(platform.frequencies_hz),
                      *platform.element_positions_m.shape[:2]), dtype=np.complex64)
    for draw in range(draws):
        paths, _ = _sample_truth_around_centers(centers, profile, config, rng)
        for index, path in enumerate(paths):
            basis[draw * len(paths) + index] = np.sqrt(powers[index] / draws) * spherical_planar_path_channel(
                platform.frequencies_hz, platform.element_positions_m, path,
                absorption_table=platform.absorption_table,
                include_absorption=bool(channel["include_absorption"]),
                interpolation=config["atmosphere"]["interpolation"],
            )
    return basis


def lmmse_coefficients(observed_basis, observation, noise_variance):
    """Weights for R_h = H.T @ H.conj(), given observed basis B = (A H.T).T.

    The prior mean is zero. The caller returns H.T @ weights, avoiding a dense
    channel covariance. Each row of H already includes its covariance weight.
    """
    basis = np.asarray(observed_basis, dtype=complex)
    received = np.asarray(observation, dtype=complex)
    variance = np.asarray(noise_variance, dtype=float)
    if basis.ndim != 2 or received.shape[0] != basis.shape[1]:
        raise ValueError("basis and observation dimensions disagree")
    if np.any(variance <= 0) or not np.isfinite(variance).all():
        raise ValueError("noise variance must be positive and finite")
    if not np.isfinite(basis).all() or not np.isfinite(received).all():
        raise ValueError("basis and observations must be finite")
    eigenvalues, vectors = eigh(basis.conj() @ basis.T)
    right = vectors.conj().T @ (basis.conj() @ received)
    denominator = np.maximum(eigenvalues, 0)
    if received.ndim == 2:
        denominator = denominator[:, None]
    return vectors @ (right / (denominator + variance))


def ridge_channel(operator, observation, regularization):
    """Exact white-noise Tikhonov solution at the observed pilot tones.

    Minimizes ||y_white - A_white h||^2 + lambda ||h||^2. Without a
    cross-frequency prior, the minimizer is zero at unobserved tones.
    """
    if regularization <= 0 or not np.isfinite(regularization):
        raise ValueError("regularization must be positive and finite")
    received = np.asarray(observation, dtype=complex).reshape(operator.observation_shape)
    if not np.isfinite(received).all():
        raise ValueError("observation must be finite")
    # K,S,J,E blocks: different microstrips and frequencies have disjoint unknowns.
    matrix = np.conjugate(operator.weights).transpose(1, 2, 0, 3)
    scale = np.sqrt(operator.relative_output_noise_variance).transpose(2, 0, 1)
    matrix = matrix / scale[..., None]
    gram = matrix @ matrix.conj().swapaxes(-1, -2)
    gram += regularization * np.eye(gram.shape[-1])
    dual = np.linalg.solve(gram, received.transpose(2, 0, 1)[..., None])
    return (matrix.conj().swapaxes(-1, -2) @ dual)[..., 0]
