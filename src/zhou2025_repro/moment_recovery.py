"""Independent VP conditional-mean prototype; existing Zhou recovery stays frozen."""
import math

import numpy as np
import torch

from .configuration_sweep import ScheduledDmaOperator
from .d041 import SpectralDmaOperator
from .data import encode


def clean_prediction(state, epsilon, alpha_bar):
    if not 0 < alpha_bar < 1 or state.shape != epsilon.shape:
        raise ValueError("matching VP images and alpha_bar in (0,1) required")
    return (state-math.sqrt(1-alpha_bar)*epsilon)/math.sqrt(alpha_bar)


def condition_gaussian(mean, covariance, matrix, observed, noise_covariance):
    """Small real-valued reference only; never materialize the full DMA covariance."""
    n, m = mean.numel(), observed.numel()
    if (mean.shape != (n,) or observed.shape != (m,) or matrix.shape != (m, n)
            or covariance.shape != (n, n) or noise_covariance.shape != (m, m)
            or any(q.is_complex() or not torch.isfinite(q).all()
                   for q in (mean, covariance, matrix, observed, noise_covariance))):
        raise ValueError("finite, real, dimension-compatible Gaussian inputs required")
    if (not torch.allclose(covariance, covariance.T) or
            not torch.allclose(noise_covariance, noise_covariance.T)):
        raise ValueError("symmetric covariance required")
    torch.linalg.cholesky(covariance)
    if torch.linalg.eigvalsh(noise_covariance).min() < 0:
        raise ValueError("positive semidefinite noise covariance required")
    cross = covariance@matrix.T
    factor = torch.linalg.cholesky(noise_covariance+matrix@cross)
    gain = torch.cholesky_solve(cross.T, factor).T
    return mean+gain@(observed-matrix@mean), covariance-gain@cross.T


def fuse_scalar(mean, operator, observed, noise_complex_variance, variance_real):
    """Fuse V=v I in real VP coordinates; proper complex covariance is 2v I."""
    if (not isinstance(operator, SpectralDmaOperator) or
            not math.isfinite(variance_real) or variance_real <= 0 or
            not math.isfinite(noise_complex_variance) or noise_complex_variance <= 0 or
            tuple(mean.shape[1:]) != operator.image_shape or
            tuple(observed.shape[1:]) != operator.observation_shape or
            len(mean) != len(observed) or not torch.isfinite(mean).all() or
            not torch.isfinite(observed).all()):
        raise ValueError("compatible finite DMA inputs and positive real/complex variances required")
    residual = observed-operator.forward(mean)
    if isinstance(operator, ScheduledDmaOperator):
        residual = operator.expand(residual)
    projected = torch.einsum("kjr,bksj->bksr", operator.u.conj(), residual.permute(0, 3, 1, 2))
    gain = 2*variance_real*operator.s/(noise_complex_variance+2*variance_real*operator.s.square())
    correction = torch.einsum("kre,bksr->bkse", operator.vh.conj(), projected*gain[None, :, None])
    return mean+encode(correction, "dma")


@torch.no_grad()
def recover_moments(model, schedule, operator, observation, variance, *, variances_real,
                    seed=0, trace_observer=None):
    """Deterministic posterior-mean recursion; no extra likelihood term or claim of sampling."""
    values = np.asarray(variances_real, dtype=float)
    if (values.shape != (len(schedule.beta),) or not np.isfinite(values).all() or
            (values <= 0).any() or not math.isfinite(variance) or variance <= 0):
        raise ValueError("one positive training-calibrated real variance per VP step required")
    model.eval()
    rng = torch.Generator(device=observation.device).manual_seed(seed)
    state = torch.randn((len(observation), *operator.image_shape), device=observation.device,
                        generator=rng, dtype=observation.real.dtype)
    if trace_observer is not None:
        trace_observer(state, {"phase": "initial", "t": len(values),
            "state_alpha_bar": float(schedule.alpha_bar[-1]), "network_calls": 0})
    for index in reversed(range(len(values))):
        alpha, bar = float(schedule.alpha[index]), float(schedule.alpha_bar[index])
        epsilon = model(state, torch.full((len(state),), index+1, device=state.device))
        mean = clean_prediction(state, epsilon, bar)
        fused = fuse_scalar(mean, operator, observation, variance, float(values[index]))
        previous = float(schedule.alpha_bar[index-1]) if index else 1.
        state = (math.sqrt(previous)*(1-alpha)/(1-bar)*fused +
                 math.sqrt(alpha)*(1-previous)/(1-bar)*state)
        if not torch.isfinite(state).all():
            raise ArithmeticError("nonfinite conditional-mean reverse iterate")
        if trace_observer is not None:
            trace_observer(state, {"phase": "updated", "t": index,
                "state_alpha_bar": previous, "network_calls": len(values)-index})
    return state


def gaussian_control():
    """Compare local fusion with the direct Gaussian joint conditional, not a full trajectory."""
    rng = torch.Generator().manual_seed(2026101607)
    dtype = torch.float64
    n, m = 7, 4
    base = torch.randn(n, n, generator=rng, dtype=dtype)
    covariance = base@base.T+.3*torch.eye(n, dtype=dtype)
    matrix = torch.randn(m, n, generator=rng, dtype=dtype)
    state, observation = torch.randn(n+m, generator=rng, dtype=dtype).split([n, m])
    noise = .07*torch.eye(m, dtype=dtype)
    errors = []
    for bar in (.01, .4, .99):
        perturbation = bar*covariance+(1-bar)*torch.eye(n, dtype=dtype)
        mean = math.sqrt(bar)*covariance@torch.linalg.solve(perturbation, state)
        conditional = covariance-bar*covariance@torch.linalg.solve(perturbation, covariance)
        fused, fused_covariance = condition_gaussian(mean, conditional, matrix, observation, noise)
        stacked = torch.cat([math.sqrt(bar)*torch.eye(n, dtype=dtype), matrix])
        block_noise = torch.block_diag((1-bar)*torch.eye(n, dtype=dtype), noise)
        direct, direct_covariance = condition_gaussian(torch.zeros(n, dtype=dtype), covariance,
            stacked, torch.cat([state, observation]), block_noise)
        errors.append(max(float((fused-direct).norm()/direct.norm()),
            float((fused_covariance-direct_covariance).norm()/direct_covariance.norm())))
    if max(errors) > 1e-8:
        raise ArithmeticError("Gaussian conditional fusion control failed")
    return {"passed": True, "relative_errors": errors, "tolerance": 1e-8,
            "scope": "local means/covariances; no finite-trajectory MMSE assertion"}
