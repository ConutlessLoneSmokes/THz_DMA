"""Full-CSI AWGN SURE, with paper Eq.34 and the chain-rule-corrected variant."""
import math
import numpy as np
import torch


def matched_time(schedule, variance):
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("positive real-component noise variance required")
    target = 1 / (1 + variance)
    index = int((schedule.alpha_bar - target).abs().argmin())
    return index, target, float(schedule.alpha_bar[index]) - target


def denoise(model, scaled_noisy, variance, time_index):
    a = 1 / (1 + variance)
    time = torch.full((len(scaled_noisy),), time_index + 1, device=scaled_noisy.device)
    return (scaled_noisy - math.sqrt(1 - a) * model(scaled_noisy, time)) / math.sqrt(a)


def risk(denoiser, noisy, variance, *, variant="corrected", epsilon=1e-5, probe=None):
    """Per-real-component SURE including its constant, for finite-difference probes."""
    if (variant not in ("paper", "corrected") or not np.isfinite(variance) or variance <= 0
            or not np.isfinite(epsilon) or epsilon <= 0 or not torch.isfinite(noisy).all()):
        raise ValueError("valid SURE variant, finite data and positive noise/epsilon required")
    a = 1 / (1 + variance)
    q = math.sqrt(a) * noisy
    probe = torch.randn_like(q) if probe is None else probe
    if probe.shape != q.shape:
        raise ValueError("SURE probe shape differs from the observation")
    estimate = denoiser(q)
    divergence = (probe * (denoiser(q + epsilon * probe) - estimate) / epsilon).flatten(1).mean(1)
    error = (estimate - noisy).square().flatten(1).mean(1)
    coefficient = math.sqrt(a) if variant == "corrected" else 1.
    return error + 2 * variance * coefficient * divergence - variance
