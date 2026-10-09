"""Complex SBL by sequential marginal-likelihood maximisation (Tipping/Faul).

Known white noise; independent complex Gaussian coefficient priors. All atoms
are considered at each step. Return the empirical-Bayes posterior mean, without
top-L truncation or an LS refit. No test geometry is used.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular


def sparse_bayesian_estimate(templates, observation, noise_variance, *,
                             max_iterations=2000, evidence_tolerance=1e-4):
    """Templates are rows; gamma denotes variance, not precision.

    Optimise log|C| + y^H C^-1 y, C = noise*I + D.T @ Gamma @ D.conj().
    Only Gram columns for currently active atoms are stored (no N x N Gram).
    The complex likelihood removes the real Gaussian's common factor 1/2.
    """
    rows = np.asarray(templates, dtype=np.complex128)
    y = np.asarray(observation, dtype=np.complex128)
    if rows.ndim != 2 or y.shape != (rows.shape[1],) or not rows.size:
        raise ValueError("templates and observation dimensions disagree")
    if not np.isfinite(rows).all() or not np.isfinite(y).all():
        raise ValueError("templates and observation must be finite")
    if noise_variance <= 0 or not np.isfinite(noise_variance):
        raise ValueError("noise variance must be positive and finite")
    if max_iterations < 1 or evidence_tolerance <= 0 or not np.isfinite(evidence_tolerance):
        raise ValueError("invalid SBL stopping settings")
    norms = np.linalg.norm(rows, axis=1)
    if np.any(norms == 0):
        raise ValueError("zero-energy dictionary atom")
    conjugated_rows = np.conjugate(rows / norms[:, None])
    del rows
    scale = max(float(np.linalg.norm(y)), np.sqrt(noise_variance))
    y = y / scale
    noise = float(noise_variance) / scale**2
    matched = conjugated_rows @ y
    active, gamma = [], np.empty(0)
    gram_columns = np.empty((len(norms), 0), dtype=complex)
    global_s = np.full(len(norms), 1 / noise)
    global_q = matched / noise
    history, mean, covariance = [], np.empty(0, complex), np.empty((0, 0), complex)
    converged, updates = False, 0
    best_improvement = float("inf")
    for iteration in range(max_iterations + 1):
        if active:
            gram = gram_columns[active]
            precision = gram / noise + np.diag(1 / gamma)
            factor = cho_factor(precision, lower=True, check_finite=False)
            covariance = cho_solve(factor, np.eye(len(active)), check_finite=False)
            mean = cho_solve(factor, matched[active] / noise, check_finite=False)
            # Periodically refresh the rank-one statistics to control roundoff.
            if iteration % 50 == 0:
                projected = solve_triangular(factor[0], gram_columns.conj().T,
                                             lower=True, check_finite=False) / noise
                global_s = 1 / noise - np.sum(np.abs(projected)**2, axis=0)
                global_q = (matched - gram_columns @ mean) / noise
            s, q = global_s.copy(), global_q.copy()
            logdet = (len(y) * np.log(noise) + np.log(gamma).sum()
                      + 2 * np.log(np.diag(factor[0]).real).sum())
            objective = logdet + (np.vdot(y, y).real - np.vdot(matched[active], mean).real) / noise
            # Leave-one-out terms for active atoms, avoiding 1 - gamma*S cancellation.
            variance = covariance.diagonal().real
            s[active] = 1 / variance - 1 / gamma
            q[active] = mean / variance
        else:
            s = np.full(len(norms), 1 / noise)
            q = matched / noise
            global_s, global_q = s.copy(), q.copy()
            objective = len(y) * np.log(noise) + np.vdot(y, y).real / noise
            mean, covariance = np.empty(0, complex), np.empty((0, 0), complex)
        if history and objective > history[-1] + 1e-7 * max(1, abs(history[-1])):
            raise ArithmeticError("SBL marginal likelihood decreased")
        history.append(float(objective))
        s = np.maximum(s, np.finfo(float).eps / noise)
        q2 = np.abs(q)**2
        proposed = np.maximum(q2 - s, 0) / s**2
        old = np.zeros(len(norms))
        old[active] = gamma
        improvement = (q2 * proposed / (1 + proposed*s) - np.log1p(proposed*s)
                       - q2 * old / (1 + old*s) + np.log1p(old*s))
        chosen = int(np.argmax(improvement))
        best_improvement = float(improvement[chosen])
        if best_improvement <= evidence_tolerance:
            converged = True
            break
        if iteration == max_iterations:
            break
        if chosen in active:
            column = gram_columns[:, active.index(chosen)]
        else:
            column = conjugated_rows @ conjugated_rows[chosen].conj()
        projection = column / noise
        if active:
            projection = projection - gram_columns @ (covariance @ gram_columns[chosen].conj()) / noise**2
        change = proposed[chosen] - old[chosen]
        denominator = 1 + change * global_s[chosen]
        if denominator <= 0:
            raise ArithmeticError("nonpositive SBL rank-one update denominator")
        ratio = change / denominator
        global_q -= ratio * projection * global_q[chosen]
        global_s -= ratio * np.abs(projection)**2
        if chosen in active:
            position = active.index(chosen)
            if proposed[chosen] == 0:
                active.pop(position)
                gamma = np.delete(gamma, position)
                gram_columns = np.delete(gram_columns, position, axis=1)
            else:
                gamma[position] = proposed[chosen]
        else:
            active.append(chosen)
            gamma = np.append(gamma, proposed[chosen])
            gram_columns = np.column_stack((gram_columns, column))
        updates += 1
    indices = np.asarray(active, dtype=int)
    coefficient_scale = scale / norms[indices]
    return {"indices": indices, "mean": mean * coefficient_scale,
            "variance": covariance.diagonal().real * coefficient_scale**2,
            "gamma": gamma * coefficient_scale**2, "iterations": updates,
            "converged": converged, "objective_history": np.asarray(history),
            "max_evidence_improvement": best_improvement}
