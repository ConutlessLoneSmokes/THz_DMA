"""Conditional VE score prior and approximate annealed Langevin DMA recovery.

Real/imaginary components each use noise standard deviation sigma. Thus their
complex perturbation has variance 2*sigma**2. Positive-sigma likelihood below is
a Gaussian surrogate, not the exact likelihood conditioned on a diffusion state.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def pack_channel(channel):
    """Complex (B,K,S,E) -> real (B,2S,E,K), real strips then imaginary strips."""
    channel = channel.permute(0, 2, 3, 1)
    return torch.cat((channel.real, channel.imag), dim=1)


def unpack_channel(tensor):
    real, imag = tensor.chunk(2, dim=1)
    return torch.complex(real, imag).permute(0, 3, 1, 2)


def center_condition(centers, config):
    search = config["search"]
    low = np.array([search["range_min_m"], np.deg2rad(search["azimuth_min_deg"]),
                    np.deg2rad(search["elevation_min_deg"])])
    high = np.array([search["range_max_m"], np.deg2rad(search["azimuth_max_deg"]),
                     np.deg2rad(search["elevation_max_deg"])])
    points = np.asarray(centers)
    if points.shape != (3, 3) or not np.isfinite(points).all() or np.any(high <= low):
        raise ValueError("expected three finite pre-observation centers and valid bounds")
    return (2 * (points - low) / (high - low) - 1).reshape(-1).astype(np.float32)


class ConditionalBlock(nn.Module):
    def __init__(self, inputs, outputs, embedding):
        super().__init__()
        self.first = nn.Conv2d(inputs, outputs, 3, padding=1)
        self.norm1, self.norm2 = nn.GroupNorm(4, outputs), nn.GroupNorm(4, outputs)
        self.condition = nn.Linear(embedding, outputs)
        self.second = nn.Conv2d(outputs, outputs, 3, padding=1)

    def forward(self, x, embedding):
        x = F.silu(self.norm1(self.first(x)) + self.condition(embedding)[:, :, None, None])
        return F.silu(self.norm2(self.second(x)))


class ConditionalScoreNet(nn.Module):
    """Small two-scale U-Net predicts Gaussian corruption epsilon, not channel gain."""
    def __init__(self, channels=16, width=24, noise_embedding_scale=1.):
        super().__init__()
        if channels < 2 or channels % 2 or width < 4 or width % 4:
            raise ValueError("even real/imag channels and width divisible by four required")
        embedding = width * 4
        self.noise_embedding_scale = noise_embedding_scale
        self.register_buffer("noise_frequencies", torch.logspace(0, 2, 8))
        self.embedding = nn.Sequential(nn.Linear(25, embedding), nn.SiLU(),
                                       nn.Linear(embedding, embedding))
        self.enc1 = ConditionalBlock(channels, width, embedding)
        self.enc2 = ConditionalBlock(width, 2*width, embedding)
        self.middle = ConditionalBlock(2*width, 4*width, embedding)
        self.dec2 = ConditionalBlock(6*width, 2*width, embedding)
        self.dec1 = ConditionalBlock(3*width, width, embedding)
        self.output = nn.Conv2d(width, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, sigma, centers):
        phase = self.noise_embedding_scale * sigma.log()[:, None] * self.noise_frequencies[None]
        embedding = self.embedding(torch.cat((phase.sin(), phase.cos(), centers), dim=1))
        first = self.enc1(x, embedding)
        second = self.enc2(F.avg_pool2d(first, 2), embedding)
        middle = self.middle(F.avg_pool2d(second, 2), embedding)
        up = self.dec2(torch.cat((F.interpolate(middle, size=second.shape[-2:], mode="nearest"), second), dim=1), embedding)
        up = self.dec1(torch.cat((F.interpolate(up, size=first.shape[-2:], mode="nearest"), first), dim=1), embedding)
        return self.output(up)


def fourier_channel(tensor, *, inverse=False):
    """Unitary transform of the complete CSI; no bins, paths or energy discarded."""
    transform = torch.fft.ifft2 if inverse else torch.fft.fft2
    return pack_channel(transform(unpack_channel(tensor), dim=(1, 3), norm="ortho"))


class PreconditionedScoreNet(nn.Module):
    """EDM-style sigma_data=1 preconditioning in unitary frequency/spatial DFT coordinates.

    Global real-component RMS is fixed by the training set. This is not a
    per-example normalization and the network still returns element-domain epsilon.
    """
    def __init__(self, channels=16, width=24):
        super().__init__()
        self.backbone = ConditionalScoreNet(channels, width, noise_embedding_scale=.25)

    def forward(self, x, sigma, centers):
        s = sigma[:, None, None, None]
        z = fourier_channel(x)
        denominator = torch.sqrt(1+s*s)
        residual = self.backbone(z/denominator, sigma, centers)
        # Return the predicted noise directly to avoid subtractive cancellation.
        noise = s/(1+s*s)*z - residual/denominator
        return fourier_channel(noise, inverse=True)


def build_score_model(settings):
    kind = settings.get("kind", "epsilon")
    if kind == "epsilon":
        return ConditionalScoreNet(width=settings["width"])
    if kind == "edm_fourier":
        return PreconditionedScoreNet(width=settings["width"])
    raise ValueError(f"unknown score network: {kind}")


class TorchDmaOperator:
    """Matrix-free fullband CSI operator, exactly matching relative whitening."""
    def __init__(self, operator, pilot_indices, *, tones=128, device="cpu", dtype=torch.complex64):
        if tones < 1:
            raise ValueError("positive fullband tone count required")
        self.tones = tones
        self.indices = torch.as_tensor(pilot_indices, dtype=torch.long, device=device)
        weights = torch.as_tensor(operator.weights, dtype=dtype, device=device)
        scale = torch.as_tensor(np.sqrt(operator.relative_output_noise_variance),
                                dtype=weights.real.dtype, device=device).permute(2, 0, 1)
        self.matrix = weights.conj().permute(1, 2, 0, 3) / scale[..., None]  # K,S,J,E
        self.gram = self.matrix @ self.matrix.conj().transpose(-1, -2)
        if (len(self.indices) != self.matrix.shape[0] or len(self.indices.unique()) != len(self.indices)
                or self.indices.min() < 0 or self.indices.max() >= tones):
            raise ValueError("invalid pilot indices")

    def forward(self, x):
        h = unpack_channel(x)[:, self.indices]
        return torch.einsum("ksje,bkse->bksj", self.matrix, h).permute(0, 2, 3, 1)

    def adjoint(self, y):
        selected = torch.einsum("ksje,bksj->bkse", self.matrix.conj(), y.permute(0, 3, 1, 2))
        h = selected.new_zeros((y.shape[0], self.tones, self.matrix.shape[1], self.matrix.shape[3]))
        h[:, self.indices] = selected
        return pack_channel(h)

    def noise_factor(self, variance, sigma):
        if not math.isfinite(variance) or variance <= 0 or not math.isfinite(sigma) or sigma < 0:
            raise ValueError("positive complex noise variance and nonnegative sigma required")
        covariance = 2 * sigma**2 * self.gram + variance * torch.eye(
            self.gram.shape[-1], device=self.gram.device, dtype=self.gram.dtype)
        return torch.linalg.cholesky(covariance)

    def likelihood_score(self, x, y, factor):
        residual = (y - self.forward(x)).permute(0, 3, 1, 2)
        dual = torch.cholesky_solve(residual[..., None], factor)[..., 0].permute(0, 2, 3, 1)
        return 2 * self.adjoint(dual)  # real gradient of a complex Gaussian log likelihood


@torch.no_grad()
def posterior_samples(model, operator, observation, centers, variance, settings, *, seed, gaussian_prior=None):
    """Finite-step approximate annealed Langevin chains; no truth or NMSE input.

    Gaussian surrogate at sigma: C_y = variance*I + 2*sigma**2*A*A^H.
    Its score is added to the learned smoothed prior score. This is an adaptation
    of annealed score-based channel recovery, not an exact calibrated posterior.
    """
    if settings.get("sampler", "ald") == "guided_heun":
        return guided_heun_samples(model, operator, observation, centers, variance, settings,
                                   seed=seed, gaussian_prior=gaussian_prior)
    if settings.get("sampler", "ald") != "ald":
        raise ValueError("unknown diffusion sampler")
    sigma_min, sigma_max = settings["sigma_min"], settings["sigma_max"]
    levels, steps, samples = settings["levels"], settings["steps_per_level"], settings["samples"]
    step_scale = settings["step_scale"]
    if not (math.isfinite(sigma_max) and 0 < sigma_min < sigma_max
            and all(isinstance(n, int) for n in (levels, steps, samples))
            and levels >= 2 and steps >= 1 and samples >= 1
            and 0 < step_scale <= 0.25):
        raise ValueError("invalid sampler schedule")
    model.eval()
    device = observation.device
    rng = torch.Generator(device=device).manual_seed(seed)
    shape = (samples, 2*operator.matrix.shape[1], operator.matrix.shape[3], operator.tones)
    x = torch.randn(shape, device=device, generator=rng) * sigma_max
    condition = centers.expand(samples, -1)
    history = []
    for sigma_tensor in torch.logspace(math.log10(sigma_max), math.log10(sigma_min), levels, device=device):
        sigma = float(sigma_tensor)
        noise_level = x.new_full((samples,), sigma)
        factor = operator.noise_factor(variance, sigma)
        eta = step_scale * sigma**2
        for _ in range(steps):
            score = -model(x, noise_level, condition) / sigma
            score += operator.likelihood_score(x, observation, factor)
            x = x + eta * score + math.sqrt(2*eta) * torch.randn(shape, device=device, generator=rng)
        if not torch.isfinite(x).all():
            raise ArithmeticError("nonfinite posterior iterate")
        history.append({"sigma": sigma, "residual_mean_squared": float(
            (operator.forward(x)-observation).abs().square().mean())})
    # Tweedie final denoising; averaging all chains defines the reported estimate.
    x -= sigma_min * model(x, x.new_full((samples,), sigma_min), condition)
    if not torch.isfinite(x).all():
        raise ArithmeticError("nonfinite final posterior estimate")
    return unpack_channel(x), history


def denoised_likelihood_score(model, operator, x, observation, centers, variance, sigma, conditional_solve=None):
    """PiGDM-style Gaussian likelihood at denoised CSI, differentiated through D.

    Conditional real-component covariance is approximated by sigma²/(1+sigma²) I.
    Exact for an N(0,I) real prior with its exact denoiser; not exact for learned priors.
    """
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("positive finite diffusion sigma required")
    with torch.enable_grad():
        state = x.detach().requires_grad_(True)
        denoised = state - sigma * model(state, state.new_full((len(state),), sigma), centers)
        residual = observation - operator.forward(denoised)
        if conditional_solve is None:
            tau_squared = sigma**2 / (1 + sigma**2)
            factor = operator.noise_factor(variance, math.sqrt(tau_squared))
            residual = residual.permute(0, 3, 1, 2)[..., None]
            dual = torch.cholesky_solve(residual, factor)
        else:
            dual = conditional_solve(residual, sigma)
        log_likelihood = -(residual.conj()*dual).sum().real
        guidance, = torch.autograd.grad(log_likelihood, state)
    return (denoised.detach()-x)/sigma**2 + guidance.detach()


@torch.no_grad()
def conditional_moment_score(model, operator, x, observation, centers, variance, sigma, prior, conditional_solve):
    """Conditional Tweedie score with h|x approximated as CN(D(x), S_sigma).

    S_sigma is the existing Gaussian conditional covariance. This is exact for
    its Gaussian denoiser and an uncalibrated moment approximation for learned D.
    """
    if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(variance) or variance <= 0:
        raise ValueError("positive finite noise levels required")
    level = x.new_full((len(x),), sigma)
    denoised = x-sigma*model(x, level, centers)
    dual = conditional_solve(observation-operator.forward(denoised), sigma)
    # S_sigma = 2 sigma² R(R + 2 sigma² I)^-1 in complex coordinates.
    return (denoised-x)/sigma**2 + 2*prior.denoise(operator.adjoint(dual), level)


@torch.no_grad()
def heun_flow(initial, sigmas, score):
    """VE probability-flow ODE dx/dsigma=-sigma*score, ending at sigma=0.

    The caller supplies score(x, sigma). No additional unconditional denoising is
    applied after the last conditional Euler step.
    """
    levels = [float(s) for s in sigmas]
    if (len(levels) < 2 or not all(math.isfinite(s) and s > 0 for s in levels)
            or any(a <= b for a, b in zip(levels, levels[1:]))):
        raise ValueError("strictly decreasing positive finite sigma levels required")
    state, trace = initial.clone(), []
    for sigma, next_sigma in zip(levels, levels[1:]+[0.]):
        drift = -sigma*score(state, sigma)
        trial = state + (next_sigma-sigma)*drift
        if next_sigma:
            trial = state + (next_sigma-sigma)*.5*(drift-next_sigma*score(trial, next_sigma))
        state = trial
        if not torch.isfinite(state).all():
            raise ArithmeticError("nonfinite Heun iterate")
        trace.append({"sigma": next_sigma, "state_real_rms": float(state.square().mean().sqrt())})
    return state, trace


@torch.no_grad()
def guided_heun_samples(model, operator, observation, centers, variance, settings, *, seed, gaussian_prior=None):
    lo, hi, levels, samples = (settings[k] for k in ("sigma_min", "sigma_max", "levels", "samples"))
    if not (math.isfinite(hi) and 0 < lo < hi and isinstance(levels, int) and levels >= 2
            and isinstance(samples, int) and samples >= 1):
        raise ValueError("invalid guided Heun schedule")
    guidance = settings.get("guidance", "denoiser_vjp")
    if guidance not in ("denoiser_vjp", "conditional_moment_mean"):
        raise ValueError("unknown guidance method")
    if guidance == "conditional_moment_mean" and gaussian_prior is None:
        raise ValueError("conditional moment guidance requires a covariance prior")
    conditional_solve = None
    if gaussian_prior is not None:
        model = SubspaceScoreModel(model, gaussian_prior)
        conditional_solve = gaussian_prior.observation_solver(operator, variance)
    model.eval()
    rng = torch.Generator(device=observation.device).manual_seed(seed)
    shape = (samples, 2*operator.matrix.shape[1], operator.matrix.shape[3], operator.tones)
    initialization = settings.get("initialization", "sigma_gaussian")
    antithetic = settings.get("antithetic", False)
    if antithetic and samples % 2:
        raise ValueError("antithetic sampling requires an even sample count")
    draws = samples//2 if antithetic else samples
    noise = torch.randn((draws,*shape[1:]), device=observation.device,
                        dtype=operator.matrix.real.dtype, generator=rng)*hi
    mean = torch.zeros_like(noise[:1])
    if initialization == "conditional_gaussian":
        if gaussian_prior is None:
            raise ValueError("conditional Gaussian initialization requires a covariance prior")
        latent_mean, covariance = gaussian_prior.posterior_moments(operator, observation, variance)
        z = torch.randn((draws,gaussian_prior.rank,2), device=observation.device,
                        dtype=torch.float64, generator=rng)
        z = torch.complex(z[...,0],z[...,1])/math.sqrt(2)
        perturbation = z@torch.linalg.cholesky(covariance).T
        noise += gaussian_prior.synthesize(perturbation.to(gaussian_prior.u.dtype))
        mean = gaussian_prior.synthesize(latent_mean[None].to(gaussian_prior.u.dtype))
    elif initialization != "sigma_gaussian":
        raise ValueError("unknown initial distribution")
    initial = mean + (torch.cat((noise,-noise)) if antithetic else noise)
    condition = centers.expand(samples, -1)
    sigmas = torch.logspace(math.log10(hi), math.log10(lo), levels, dtype=torch.float64)
    if guidance == "conditional_moment_mean":
        score = lambda x,s: conditional_moment_score(model, operator, x, observation,
            condition, variance, s, gaussian_prior, conditional_solve)
    else:
        score = lambda x,s: denoised_likelihood_score(model, operator, x, observation,
            condition, variance, s, conditional_solve)
    state, trace = heun_flow(initial, sigmas, score)
    trace[-1]["residual_mean_squared"] = float((operator.forward(state)-observation).abs().square().mean())
    trace[-1].update(initialization=initialization, antithetic=bool(antithetic), guidance=guidance)
    return unpack_channel(state), trace


class LowRankChannelPrior:
    """Independent conditional MC covariance R=U diag(eigenvalues) U^H.

    Compact eigensystem avoids an element-space covariance. A relative eigenvalue
    cutoff is explicit and the retained energy is recorded, not inferred from truth.
    """
    def __init__(self, basis, *, device="cpu", relative_cutoff=1e-5):
        if not 0 < relative_cutoff < 1:
            raise ValueError("relative covariance cutoff must be in (0,1)")
        self.shape = tuple(basis.shape[1:])
        b = torch.as_tensor(basis.reshape(len(basis), -1), device=device, dtype=torch.complex64)
        if not torch.isfinite(b).all():
            raise ValueError("nonfinite conditional covariance basis")
        gram = (b.conj()@b.T).to(torch.complex128)
        values, vectors = torch.linalg.eigh((gram+gram.H)/2)
        keep = values > values[-1]*relative_cutoff
        if not keep.any() or values[-1] <= 0:
            raise ValueError("empty conditional prior")
        span = b.T@vectors[:,keep].to(b.dtype)
        q,_ = torch.linalg.qr(span,mode="reduced")
        reduced = (b.conj()@q).to(torch.complex128)
        eigenvalues, rotation = torch.linalg.eigh(reduced.H@reduced)
        self.u = q@rotation.to(q.dtype)
        self.values = eigenvalues.real.to(torch.float32)
        self.retained_energy = float(eigenvalues.sum()/b.abs().square().sum())
        self.rank = len(eigenvalues)

    def coefficients(self, x):
        return unpack_channel(x).reshape(len(x),-1)@self.u.conj()

    def synthesize(self, coefficients):
        return pack_channel((coefficients@self.u.T).reshape(len(coefficients),*self.shape))

    def project(self, x):
        return self.synthesize(self.coefficients(x))

    def denoise(self, x, sigma):
        return self.synthesize(self.coefficients(x)*self.values[None]/(self.values[None]+2*sigma[:,None]**2))

    def posterior_moments(self, operator, observation, variance):
        """Complex latent posterior moments under this Gaussian prior; no channel truth."""
        if not math.isfinite(variance) or variance <= 0 or len(observation) != 1:
            raise ValueError("one observation and positive finite noise variance required")
        v = operator.forward(pack_channel(self.u.T.reshape(self.rank,*self.shape))).flatten(1).T.to(torch.complex128)
        precision = torch.diag(1/self.values.double()).to(v.dtype) + v.H@v/variance
        covariance = torch.cholesky_inverse(torch.linalg.cholesky((precision+precision.H)/2))
        mean = covariance@(v.H@observation.flatten().to(v.dtype))/variance
        return mean, covariance

    def observation_solver(self, operator, variance):
        if not math.isfinite(variance) or variance <= 0:
            raise ValueError("positive finite noise variance required")
        v = operator.forward(pack_channel(self.u.T.reshape(self.rank,*self.shape))).flatten(1).T.to(torch.complex128)
        gram = v.H@v
        def solve(residual, sigma):
            # Woodbury in rank-dimensional space, using double precision for subtraction.
            covariance_eigenvalues = self.values.double()*2*sigma*sigma/(self.values.double()+2*sigma*sigma)
            system = gram+torch.diag(variance/covariance_eigenvalues).to(gram.dtype)
            flat = residual.flatten(1).to(torch.complex128)
            coefficients = torch.linalg.solve(system, v.H@flat.T)
            result = (flat.T-v@coefficients).T/variance
            return result.to(residual.dtype).reshape_as(residual)
        return solve


class SubspaceScoreModel(nn.Module):
    """Projected learned denoiser, or an exact Gaussian denoiser when core is None."""
    def __init__(self, core, prior):
        super().__init__()
        self.core, self.prior = core, prior

    def forward(self, x, sigma, centers):
        s = sigma[:,None,None,None]
        if self.core is None:
            clean = self.prior.denoise(x,sigma)
        else:
            clean = self.prior.project(x-s*self.core(x,sigma,centers))
        return (x-clean)/s
