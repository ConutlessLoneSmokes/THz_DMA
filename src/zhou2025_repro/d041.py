"""D-041 paired no-absorption physics and full-tone diagnostic operators.

The state bank is fixed independently of all channel/noise samples. Raw hardware
weights are retained; only the observation and its operator are noise-whitened.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch

from thz_dma.channels.geometry import ofdm_frequencies
from thz_dma.channels.planar import (
    PlanarPath, planar_dma_positions, sample_stage2_paths,
    spherical_planar_path_channel,
)
from thz_dma.surfaces.multistrip import feed_distances_m, multistrip_dma_weights
from .data import decode, encode
from .wideband import DigitalArrayOperator


@dataclass(frozen=True)
class PairedArray:
    frequencies: np.ndarray
    positions: np.ndarray
    feed_distances: np.ndarray


def paired_array(config):
    system, dma, settings = config["system"], config["dma"], config["diagnostic"]
    carrier = float(system["carrier_frequency_hz"])
    if config["channel"]["include_absorption"]:
        raise ValueError("D-041 requires freshly generated absorption-free CSI")
    frequencies = (np.array([carrier]) if config["band"] == "narrow" else
                   ofdm_frequencies(carrier, system["bandwidth_hz"], system["num_subcarriers"]))
    positions = planar_dma_positions(dma["num_microstrips"], dma["elements_per_microstrip"],
        settings["element_spacing_m"], settings["microstrip_spacing_m"])
    return PairedArray(frequencies, positions,
        feed_distances_m(dma["elements_per_microstrip"], settings["element_spacing_m"]))


def paired_channel(config, array, seed, index):
    paths = sample_stage2_paths(config, np.random.default_rng(np.random.SeedSequence([seed, index])))
    templates = np.stack([spherical_planar_path_channel(array.frequencies, array.positions,
        PlanarPath(p.range_m, p.azimuth_rad, p.elevation_rad), include_absorption=False) for p in paths])
    record = {"scene_id": f"{seed}:{index}",
        "coordinates": [[p.range_m, p.azimuth_rad, p.elevation_rad] for p in paths],
        "gain_real": [p.complex_gain.real for p in paths],
        "gain_imag": [p.complex_gain.imag for p in paths]}
    gains = np.array([p.complex_gain for p in paths])
    return np.einsum("l,lkse->kse", gains, templates), record


def geometry_templates(array, record):
    """Unit-gain templates; the geometry reference never reads saved true gains."""
    return np.stack([spherical_planar_path_channel(array.frequencies, array.positions,
        PlanarPath(*point), include_absorption=False) for point in record["coordinates"]])


def state_bank(config):
    dma, settings = config["dma"], config["diagnostic"]
    center, width = config["system"]["carrier_frequency_hz"], dma["tuning_bandwidth_hz"]
    # Same within-strip pattern on all strips, as in the established platform.
    return np.random.default_rng(settings["state_seed"]).uniform(center-width/2, center+width/2,
        (max(settings["configuration_counts"]), 1, dma["elements_per_microstrip"]))


def hardware_weights(config, array, bank):
    dma = config["dma"]
    return multistrip_dma_weights(array.frequencies, bank, array.feed_distances,
        quality_factor=dma["quality_factor"],
        reference_frequency_hz=config["system"]["carrier_frequency_hz"],
        waveguide_effective_index=dma["waveguide_effective_index"],
        waveguide_end_power_fraction=dma["waveguide_end_power_fraction"])[:, :, 0]


class SpectralDmaOperator:
    """Shared-strip A[k,j,e], with exact SVD VP likelihood and row projection.

    Caching a complex128 SVD avoids repeated J x J Cholesky solves and numerical
    loss with small likelihood variance. This is the same linear VP likelihood,
    rather than an additional channel prior or an alternative recovery method.
    """
    def __init__(self, weights, strips, *, rf_noise_fraction=0., rtol=1e-10,
                 device="cpu", dtype=torch.complex64):
        weights = np.asarray(weights)
        if (weights.ndim != 3 or min(weights.shape) < 1 or strips < 1 or
                not np.isfinite(weights).all() or not 0 < rtol < 1 or
                not math.isfinite(rf_noise_fraction) or rf_noise_fraction < 0):
            raise ValueError("finite (J,K,E) weights, strips and valid noise/rank settings required")
        relative = np.sum(abs(weights)**2, axis=-1) + rf_noise_fraction
        if np.any(relative <= 0):
            raise ValueError("strictly positive relative output noise variance required")
        matrix = (weights.conj()/np.sqrt(relative)[..., None]).transpose(1, 0, 2).copy()
        # All strips share weights, so each frequency needs only one decomposition.
        u, s, vh = torch.linalg.svd(torch.as_tensor(matrix, dtype=torch.complex128), full_matrices=False)
        self.keep = s > rtol*s[:, :1]
        self.singular_values = s
        self.ranks = self.keep.sum(dim=-1)
        self.matrix = torch.as_tensor(matrix, dtype=dtype, device=device)
        self.u, self.vh = u.to(device=device, dtype=dtype), vh.to(device=device, dtype=dtype)
        self.s = s.to(device=device, dtype=self.matrix.real.dtype)
        self.mask = self.keep.to(device=device)
        self.relative_variance = torch.as_tensor(relative, device=device,
            dtype=self.matrix.real.dtype)[None].expand(strips, -1, -1)
        self.image_shape = (2*strips, matrix.shape[-1], matrix.shape[0])
        self.channel_shape = (matrix.shape[0], strips, matrix.shape[-1])
        self.observation_shape = (strips, matrix.shape[1], matrix.shape[0])

    def forward(self, image):
        h = decode(image, "dma")
        return torch.einsum("kje,bkse->bksj", self.matrix, h).permute(0, 2, 3, 1)

    def adjoint(self, observation):
        h = torch.einsum("kje,bksj->bkse", self.matrix.conj(), observation.permute(0, 3, 1, 2))
        return encode(h, "dma")

    def likelihood_score(self, image, observation, variance, alpha_bar):
        if not math.isfinite(variance) or variance <= 0 or not 0 < alpha_bar < 1:
            raise ValueError("positive complex variance and alpha_bar in (0,1) required")
        root = math.sqrt(alpha_bar)
        residual = (observation-self.forward(image)/root).permute(0, 3, 1, 2)
        projected = torch.einsum("kjr,bksj->bksr", self.u.conj(), residual)
        denominator = variance+2*(1-alpha_bar)/alpha_bar*self.s.square()
        coefficients = projected*self.s[None, :, None]/denominator[None, :, None]
        h = torch.einsum("kre,bksr->bkse", self.vh.conj(), coefficients)
        return 2*encode(h, "dma")/root

    def least_squares(self, observation):
        projected = torch.einsum("kjr,bksj->bksr", self.u.conj(), observation.permute(0, 3, 1, 2))
        inverse = torch.where(self.mask, 1/self.s.clamp_min(torch.finfo(self.s.dtype).tiny), 0)
        return torch.einsum("kre,bksr->bkse", self.vh.conj(), projected*inverse[None, :, None])

    def project_row(self, channel):
        coordinates = torch.einsum("kre,bkse->bksr", self.vh, channel)
        return torch.einsum("kre,bksr->bkse", self.vh.conj(), coordinates*self.mask[None, :, None])


class DiagnosticDigitalOperator(DigitalArrayOperator):
    def __init__(self, shape, *, device="cpu"):
        super().__init__(list(range(shape[0])), shape, device=device)

    def least_squares(self, observation):
        return observation[:, 0]

    def project_row(self, channel):
        return channel


def geometry_ls(operator, observation, templates):
    images = encode(templates.to(observation.dtype), "dma")
    measured = operator.forward(images).reshape(len(templates), -1).T
    # Least squares fits gains using only observations; geometry is the extra information.
    measured = measured.to(torch.complex128)
    coefficients = torch.linalg.lstsq(measured, observation.flatten().to(torch.complex128)).solution
    return torch.einsum("l,lkse->kse", coefficients, templates.to(torch.complex128))[None]


def diagnostics(operator, truth, estimate, observation, clean):
    estimate = estimate.to(truth)
    error = estimate-truth
    energy = float(truth.abs().square().sum())
    if not energy > 0 or not torch.isfinite(estimate).all():
        raise ArithmeticError("finite estimate and nonzero truth energy required")
    row = operator.project_row(error)
    null = error-row
    predicted = operator.forward(encode(estimate, "dma"))
    observed_energy = float(observation.abs().square().sum())
    clean_energy = float(clean.abs().square().sum())
    predicted_energy = float(estimate.abs().square().sum())
    correlation = (truth.conj()*estimate).sum()
    error_ratio = float(error.abs().square().sum())/energy
    return {
        "nmse_linear": error_ratio, "nmse_db": 10*math.log10(max(error_ratio, 1e-30)),
        "observation_residual_relative": float((predicted-observation).abs().square().sum())/max(observed_energy, 1e-30),
        "clean_observation_residual_relative": float((predicted-clean).abs().square().sum())/max(clean_energy, 1e-30),
        "prediction_energy_ratio": predicted_energy/energy,
        "normalized_correlation": float(correlation.abs())/math.sqrt(max(energy*predicted_energy, 1e-60)),
        "coherent_correlation_real": float(correlation.real)/math.sqrt(max(energy*predicted_energy, 1e-60)),
        "row_space_nmse_linear": float(row.abs().square().sum())/energy,
        "null_space_nmse_linear": float(null.abs().square().sum())/energy,
        "error_decomposition_relative_gap": abs(float(row.abs().square().sum()+null.abs().square().sum())/energy-error_ratio),
        "truth_energy_normalized": energy, "prediction_energy_normalized": predicted_energy,
    }
