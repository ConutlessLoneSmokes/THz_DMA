"""Existing complex LASSO and real sample-covariance LMMSE computations."""
import numpy as np
import torch


def real_operator(a):
    return torch.cat((torch.cat((a.real, -a.imag), dim=1),
                      torch.cat((a.imag, a.real), dim=1)), dim=0)


def lasso(y, b, penalty, *, steps=1000, tolerance=1e-5):
    """Complex FISTA for .5 ||Z B-Y||² + lambda ||Z||_1."""
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("finite nonnegative LASSO penalty required")
    lipschitz = torch.linalg.matrix_norm(b, ord=2).square()
    if not torch.isfinite(lipschitz) or lipschitz <= 0:
        raise ValueError("nonzero finite measurement matrix required")
    x = torch.zeros((*y.shape[:-1], b.shape[0]), device=y.device, dtype=y.dtype)
    z = x.clone()
    momentum, converged = 1., False
    if steps < 1 or tolerance <= 0:
        raise ValueError("positive iteration budget and tolerance required")
    for iteration in range(steps):
        proposal = z - (z @ b - y) @ b.H / lipschitz
        updated = proposal * (1 - penalty / lipschitz / proposal.abs().clamp_min(1e-20)).clamp_min(0)
        error = torch.linalg.vector_norm(updated - x) / (1 + torch.linalg.vector_norm(x))
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        z = updated + (momentum - 1) / next_momentum * (updated - x)
        x, momentum = updated, next_momentum
        if iteration >= 5 and error < tolerance:
            converged = True
            break
    return x, iteration + 1, converged


def real_observation(y):
    flat = y.flatten(1)
    return torch.cat((flat.real, flat.imag), dim=1)


class EmpiricalLMMSE:
    """Zero-mean real sample covariance, using the smaller observed/latent Gram.

    The basis is whitened forward images divided by sqrt(training count).
    Synthesis uses the same training images and weight; no full CSI covariance.
    """
    def __init__(self, observed_basis):
        self.basis = observed_basis
        if observed_basis.ndim != 2 or not torch.isfinite(observed_basis).all():
            raise ValueError("finite two-dimensional observed covariance basis required")
        self.latent = len(observed_basis) < observed_basis.shape[1]
        gram = (observed_basis @ observed_basis.T if self.latent
                else observed_basis.T @ observed_basis)
        values, self.vectors = torch.linalg.eigh((gram + gram.T) / 2)
        self.values = values.clamp_min(0)

    def coefficients(self, y, complex_variance):
        if not np.isfinite(complex_variance) or complex_variance <= 0:
            raise ValueError("positive complex observation noise variance required")
        real_y = real_observation(y).to(self.basis)
        right = real_y @ self.basis.T if self.latent else real_y
        solved = ((right @ self.vectors) / (self.values + complex_variance / 2)) @ self.vectors.T
        return solved if self.latent else solved @ self.basis.T
