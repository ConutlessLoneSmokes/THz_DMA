"""VP/angle-domain adapter around the unchanged physical DMA operator."""
import math
import torch
import numpy as np
import time

from thz_dma.estimators.diffusion import TorchDmaOperator
from thz_dma.channels.planar import PlanarPath, channel_from_planar_estimate
from thz_dma.estimators.adaptive_grid import adaptive_settings, _adaptive_candidate_union
from thz_dma.estimators.planar_sparse import build_planar_template_dictionary_from_candidates
from thz_dma.estimators.planar_baselines import estimate_planar_baseline
from thz_dma.estimators.linear_channel import conditional_planar_covariance_basis, lmmse_coefficients
from .data import dma_spatial


class DmaOperator:
    def __init__(self, operator, pilot_indices, *, tones=128, device="cpu", dtype=None):
        options = {} if dtype is None else {"dtype": dtype}
        self.source = operator
        self.physical = TorchDmaOperator(operator, pilot_indices, tones=tones, device=device, **options)
        self.relative_variance = torch.as_tensor(operator.relative_output_noise_variance, device=device)
        matrix = self.physical.matrix
        self.image_shape = (2 * matrix.shape[1], matrix.shape[3], tones)

    def forward(self, image):
        return self.physical.forward(dma_spatial(image, inverse=True))

    def adjoint(self, observation):
        return dma_spatial(self.physical.adjoint(observation))

    def likelihood_score(self, image, observation, variance, alpha_bar):
        if not 0 < alpha_bar < 1:
            raise ValueError("alpha_bar in (0,1) required")
        factor = self.physical.noise_factor(variance, math.sqrt((1 - alpha_bar) / alpha_bar))
        physical = dma_spatial(image, inverse=True) / math.sqrt(alpha_bar)
        score = self.physical.likelihood_score(physical, observation, factor)
        return dma_spatial(score) / math.sqrt(alpha_bar)


class CoarseReferences:
    """Existing additional-information baselines, kept separate from global DDPM."""
    def __init__(self, operator, schedule, platform, record, config, methods, *, device, draws, seed):
        self.operator, self.platform, self.record, self.config = operator, platform, record, config
        self.dictionary, self.basis, self.observed_basis = None, None, None
        started = time.perf_counter()
        if set(methods) & {"grid_omp_3d", "sbl_jointfreq_3d"}:
            centers = np.asarray(record["centers_m_rad_rad"])
            profile = {"settings": adaptive_settings(config), "schedule_profiles": record["schedule_profiles"]}
            # The helper's truth argument only audits coverage; do not read truth for candidate selection.
            candidates, _ = _adaptive_candidate_union(centers, profile, centers.copy(), config)
            self.dictionary = build_planar_template_dictionary_from_candidates(
                operator.source, *candidates, chunk_size=config["search"]["dictionary_chunk_size"],
                storage_dtype=config["search"]["dictionary_storage_dtype"],
                compute_backend="torch_cuda" if str(device).startswith("cuda") else "numpy",
                compute_device=device)
        self.dictionary_seconds = time.perf_counter()-started
        started = time.perf_counter()
        if "conditional_lmmse" in methods:
            self.basis = conditional_planar_covariance_basis(platform, config, record, draws=draws,
                rng=np.random.default_rng(seed))
            selected = self.basis[:, schedule.pilot_indices]
            raw = np.einsum("jkse,bkse->bsjk", np.conjugate(operator.source.weights), selected, optimize=True)
            self.observed_basis = (raw/np.sqrt(operator.source.relative_output_noise_variance)).reshape(len(raw), -1)
        self.covariance_seconds = time.perf_counter()-started

    def estimate(self, method, observation, variance):
        white = np.asarray(observation).reshape(-1)
        if method == "conditional_lmmse":
            coefficients = lmmse_coefficients(self.observed_basis, white, variance)
            channel = (self.basis.reshape(len(self.basis), -1).T@coefficients).reshape(self.basis.shape[1:])
            return channel, {"algorithm_converged": True, "algorithm_iterations": 1,
                             "preparation_seconds": self.covariance_seconds}
        paths = tuple(PlanarPath(*point) for point in self.record["coordinates"]) if method == "oracle_geometry_ls" else ()
        fit = estimate_planar_baseline(method, self.dictionary, self.operator.source, white, paths,
                                       self.config, noise_variance=variance)
        coordinates = fit.get("reconstruction_coordinates", np.column_stack(
            (fit["range_m"], fit["azimuth_rad"], fit["elevation_rad"])))
        gains = fit.get("reconstruction_gains", fit["complex_gain"])
        channel = channel_from_planar_estimate(self.platform.frequencies_hz, self.platform,
                                               coordinates, gains, self.config)
        return channel, {"algorithm_converged": bool(fit["converged"]),
                         "algorithm_iterations": int(fit["iterations"]),
                         "preparation_seconds": self.dictionary_seconds if self.dictionary is not None else 0.}
