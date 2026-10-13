"""Full-digital selected-tone reception of the unchanged wideband physical CSI."""
import math
from numbers import Integral

import torch

from .data import decode, encode


class DigitalArrayOperator:
    """CSI (B,K,S,E) -> observations (B,J,Kp,S,E), with independent noise per J.

    Each repetition receives every element at the selected pilot tones. Noise
    variance is E|n|² per complex observation; its real-component variance is half.
    """
    def __init__(self, pilot_indices, channel_shape, *, repetitions=1, device="cpu"):
        self.channel_shape = tuple(channel_shape)
        if (len(self.channel_shape) != 3 or any(
                not isinstance(n, Integral) or isinstance(n, bool) or n < 1
                for n in self.channel_shape)):
            raise ValueError("positive (tones, strips, elements) channel shape required")
        if not isinstance(repetitions, Integral) or isinstance(repetitions, bool) or repetitions < 1:
            raise ValueError("positive integer repetitions required")
        indices = torch.as_tensor(pilot_indices, device=device)
        if (indices.ndim != 1 or indices.numel() < 1
                or indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
                or indices.unique().numel() != indices.numel()
                or indices.min() < 0 or indices.max() >= self.channel_shape[0]):
            raise ValueError("unique in-bounds integer pilot indices required")
        self.indices, self.repetitions = indices.long(), int(repetitions)
        tones, strips, elements = self.channel_shape
        self.image_shape = (2 * strips, elements, tones)
        self.observation_shape = (self.repetitions, len(indices), strips, elements)
        self.relative_variance = torch.ones(self.observation_shape, device=device)

    def forward(self, image):
        if (image.ndim != 4 or tuple(image.shape[1:]) != self.image_shape
                or not image.is_floating_point()):
            raise ValueError("real image dimensions differ from the declared channel shape")
        selected = decode(image, "dma")[:, self.indices]
        return selected[:, None].expand(-1, self.repetitions, -1, -1, -1)

    def adjoint(self, observation):
        if (observation.ndim != 5 or tuple(observation.shape[1:]) != self.observation_shape
                or not observation.is_complex()):
            raise ValueError("complex observation dimensions differ from the acquisition")
        channel = observation.new_zeros((len(observation), *self.channel_shape))
        channel[:, self.indices] = observation.sum(dim=1)
        return encode(channel, "dma")

    def likelihood_score(self, image, observation, variance, alpha_bar):
        if not (math.isfinite(variance) and variance > 0 and 0 < alpha_bar < 1):
            raise ValueError("positive variance and alpha_bar in (0,1) required")
        root = math.sqrt(alpha_bar)
        residual = observation - self.forward(image) / root
        # AA^H has eigenvalue J on repeated signals; A^H removes the orthogonal residual.
        denominator = variance + 2 * (1 - alpha_bar) / alpha_bar * self.repetitions
        return 2 * self.adjoint(residual) / (root * denominator)
