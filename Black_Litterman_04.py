"""Black–Litterman absolute views from a fixed-order VAR."""
import numpy as np

from Portfolio_Methods_03 import mv_weights, regularize_cov


def forecast_view(values, steps):
    """Iterate a VAR(2) to the holding-period endpoint with forecast-error variance."""
    design = np.column_stack((np.ones(len(values) - 2), values[1:-1], values[:-2]))
    response = values[2:]
    n = values.shape[1]
    coeff, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    a1, a2 = coeff[1:n + 1].T, coeff[n + 1:].T
    companion = np.block([[a1, a2], [np.eye(n), np.zeros((n, n))]])
    if rank < design.shape[1] or np.max(np.abs(np.linalg.eigvals(companion))) >= 1:
        return values.mean(axis=0), np.diag(np.maximum(values.var(axis=0, ddof=1), 1e-6)), "sample_mean"
    residuals = response - design @ coeff
    innovation_cov = residuals.T @ residuals / (len(response) - rank)
    state = np.r_[values[-1], values[-2]]
    intercept = np.r_[coeff[0], np.zeros(n)]
    error_cov = np.zeros((n, n))
    propagation = np.eye(2 * n)
    for _ in range(steps):
        effect = propagation[:n, :n]
        error_cov += effect @ innovation_cov @ effect.T
        propagation = propagation @ companion
        state = intercept + companion @ state
    return state[:n], np.diag(np.maximum(np.diag(error_cov), 1e-6)), "VAR2"


def posterior_moments(cov, view, omega, gamma, sample_size, repairs, context):
    """Use an equal-weight reference prior and the posterior predictive covariance."""
    tau_cov = cov / sample_size
    prior = gamma * cov @ np.full(len(view), 1 / len(view))
    update = np.linalg.solve(tau_cov + omega, view - prior)
    mean = prior + tau_cov @ update
    mean_cov = tau_cov - tau_cov @ np.linalg.solve(tau_cov + omega, tau_cov)
    predictive = regularize_cov(cov + mean_cov, repairs, {**context, "matrix": "BL predictive"})
    return mean, predictive

