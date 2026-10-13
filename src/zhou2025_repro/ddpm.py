"""Published VP/DDPM equations and deterministic Algorithm 1."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DenoisingCNN(nn.Module):
    """Two pre-convolutions, time scale/bias, three post-convolutions."""
    def __init__(self, channels=2, width=64, time_dim=16):
        super().__init__()
        if channels < 2 or width < channels or width < 4 or time_dim < 4 or time_dim % 2:
            raise ValueError("valid feature width and even time dimension required")
        self.pre = nn.Sequential(nn.Conv2d(channels, width // 2, 3, padding=1),
                                 nn.ReLU(), nn.Conv2d(width // 2, width, 3, padding=1))
        self.time = nn.Linear(time_dim, 2 * width)
        self.register_buffer("frequencies", torch.exp(
            -math.log(10000) * torch.arange(time_dim // 2) / (time_dim // 2 - 1)))
        first, second = int((2 * width + channels) / 3), int((width + 2 * channels) / 3)
        self.post = nn.Sequential(nn.Conv2d(width, first, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(first, second, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(second, channels, 3, padding=1))

    def forward(self, image, time):
        phase = time[:, None] * self.frequencies[None]
        scale, shift = self.time(torch.cat((phase.sin(), phase.cos()), dim=1)).chunk(2, dim=1)
        features = self.pre(image)
        return self.post(features * (1 + scale[:, :, None, None]) + shift[:, :, None, None])


class VPSchedule(nn.Module):
    def __init__(self, steps=100, beta_start=1e-4, beta_end=.1):
        super().__init__()
        if not isinstance(steps, int) or steps < 2 or not 0 < beta_start <= beta_end < 1:
            raise ValueError("at least two steps and 0 < beta_start <= beta_end < 1 required")
        beta = torch.linspace(beta_start, beta_end, steps, dtype=torch.float64)
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", 1 - beta)
        self.register_buffer("alpha_bar", torch.cumprod(1 - beta, dim=0))

    def perturb(self, clean, indices, noise):
        if clean.shape != noise.shape or indices.shape != (len(clean),):
            raise ValueError("diffusion sample dimensions disagree")
        a = self.alpha_bar[indices].to(clean)[:, None, None, None]
        return a.sqrt() * clean + (1 - a).sqrt() * noise


def noise_loss(model, schedule, clean, *, generator=None):
    time = torch.randint(len(schedule.beta), (len(clean),), device=clean.device, generator=generator)
    noise = torch.randn(clean.shape, dtype=clean.dtype, device=clean.device, generator=generator)
    return F.mse_loss(model(schedule.perturb(clean, time, noise), time + 1), noise)


@torch.no_grad()
def recover(model, schedule, operator, observation, variance, *, scale=1., seed=0, refinement=1):
    """Frozen final endpoint; no truth-based choice of an intermediate iterate."""
    if (not np.isfinite(scale) or scale < 0 or not isinstance(refinement, int) or refinement < 1
            or not np.isfinite(variance) or variance <= 0 or not torch.isfinite(observation).all()):
        raise ValueError("finite observation/noise, nonnegative guidance and positive refinement required")
    model.eval()
    rng = torch.Generator(device=observation.device).manual_seed(seed)
    state = torch.randn((len(observation), *operator.image_shape),
                        device=observation.device, generator=rng, dtype=observation.real.dtype)
    trace, calls = [], 0
    for index in reversed(range(len(schedule.beta))):
        alpha, bar = float(schedule.alpha[index]), float(schedule.alpha_bar[index])
        repeats = refinement if index < len(schedule.beta) // 2 else 1
        for _ in range(repeats):
            epsilon = model(state, torch.full((len(state),), index + 1, device=state.device))
            likelihood = operator.likelihood_score(state, observation, variance, bar)
            state = (state - (1 - alpha) / math.sqrt(1 - bar) * epsilon
                     + scale * (1 - alpha) * likelihood) / math.sqrt(alpha)
            calls += 1
            if not torch.isfinite(state).all():
                raise ArithmeticError("nonfinite DDPM reverse iterate")
        trace.append({"t": index + 1, "real_rms": float(state.square().mean().sqrt()),
                      "network_calls": calls})
    return state, trace
