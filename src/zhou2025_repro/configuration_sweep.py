"""Compact observations for fixed-RE DMA configuration/frequency controls."""
import math

import numpy as np
import torch

from thz_dma.observations.pilots import full_band_pilot_indices
from .d041 import SpectralDmaOperator
from .data import encode


class ScheduledDmaOperator(SpectralDmaOperator):
    """Keep full CSI, but expose only the actually sampled (configuration, tone) REs."""
    def __init__(self, weights, strips, active, **kwargs):
        active = np.asarray(active)
        if (active.dtype != np.bool_ or active.shape != np.shape(weights)[:2]
                or not active.any(axis=1).all()):
            raise ValueError("boolean (J,K) mask with at least one RE per configuration required")
        super().__init__(weights, strips, **kwargs)
        relative = self.relative_variance
        self.active = active.copy()
        self.indices = torch.as_tensor(np.flatnonzero(active), device=self.matrix.device)
        # Decompose before casting to complex64, including the unobserved zero rows.
        raw_relative = np.sum(abs(weights)**2, axis=-1) + kwargs.get("rf_noise_fraction", 0.)
        matrix = torch.as_tensor((weights.conj()/np.sqrt(raw_relative)[..., None]).transpose(1, 0, 2),
            dtype=torch.complex128) * torch.as_tensor(active.T[..., None])
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        self.keep = s > kwargs.get("rtol", 1e-10)*s[:, :1]
        self.singular_values, self.ranks = s, self.keep.sum(-1)
        self.matrix = matrix.to(self.matrix)
        self.u, self.vh = u.to(self.matrix), vh.to(self.matrix)
        self.s, self.mask = s.to(self.matrix.real), self.keep.to(self.matrix.device)
        self.relative_variance = relative.flatten(1).index_select(1, self.indices)
        self.observation_shape = (strips, int(active.sum()))

    def expand(self, observation):
        if tuple(observation.shape[1:]) != self.observation_shape:
            raise ValueError("compact observation shape differs from schedule")
        result = observation.new_zeros((len(observation), self.channel_shape[1], self.active.size))
        result.index_copy_(2, self.indices, observation)
        return result.reshape(len(observation), self.channel_shape[1], *self.active.shape)

    def forward(self, image):
        return super().forward(image).flatten(2).index_select(2, self.indices)

    def adjoint(self, observation):
        return super().adjoint(self.expand(observation))

    def least_squares(self, observation):
        return super().least_squares(self.expand(observation))

    def likelihood_score(self, image, observation, variance, alpha_bar):
        if not math.isfinite(variance) or variance <= 0 or not 0 < alpha_bar < 1:
            raise ValueError("positive complex variance and alpha_bar in (0,1) required")
        root = math.sqrt(alpha_bar)
        residual = (self.expand(observation)-super().forward(image)/root).permute(0, 3, 1, 2)
        projected = torch.einsum("kjr,bksj->bksr", self.u.conj(), residual)
        denominator = variance+2*(1-alpha_bar)/alpha_bar*self.s.square()
        coefficients = projected*self.s[None, :, None]/denominator[None, :, None]
        channel = torch.einsum("kre,bksr->bkse", self.vh.conj(), coefficients)
        return 2*encode(channel, "dma")/root


def re128_mask(configurations, *, interleaved=False):
    if configurations not in (1, 2, 4, 8, 16):
        raise ValueError("RE128 supports J=1/2/4/8/16")
    active = np.zeros((configurations, 128), dtype=bool)
    for index in range(configurations):
        indices = (np.arange(index, 128, configurations) if interleaved else
                   full_band_pilot_indices(128, 128//configurations))
        active[index, indices] = True
    return active
