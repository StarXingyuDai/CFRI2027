from __future__ import annotations

from typing import List, Optional

import numpy as np

from scipy.optimize import minimize


def regularize_cov(cov: np.ndarray, diagnostics: List[dict], context: dict) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = (cov + cov.T) / 2
    min_eig = float(np.min(np.linalg.eigvalsh(cov))) if cov.size else 0.0
    if min_eig <= 1e-8:
        jitter = abs(min_eig) + 1e-6
        cov = cov + np.eye(cov.shape[0]) * jitter
        row = dict(context)
        row.update({"min_eigenvalue": min_eig, "jitter_added": jitter})
        diagnostics.append(row)
    return cov


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w[w < 0] = 0.0
    total = float(w.sum())
    if total <= 0:
        return np.ones_like(w) / len(w)
    return w / total


def naive_weights(n: int) -> np.ndarray:
    return np.ones(n) / n


def inverse_variance_weights(cov: np.ndarray, diagnostics: List[dict], context: dict) -> np.ndarray:
    variances = np.diag(cov).copy()
    min_var = float(np.min(variances))
    if min_var < 1e-8:
        row = dict(context)
        row.update({"warning": "very_small_variance_in_inverse_variance_weights", "min_variance": min_var})
        diagnostics.append(row)
    inv = 1.0 / np.maximum(variances, 1e-8)
    return normalize_weights(inv)


def constrained_optimize(
    objective,
    n: int,
    start: Optional[np.ndarray],
    failures: List[dict],
    context: dict,
    fallback: np.ndarray,
    maxiter: int = 100,
) -> np.ndarray:
    x0 = normalize_weights(start if start is not None else fallback)
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0.0, 1.0)] * n
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": maxiter, "ftol": 1e-9, "disp": False})
    if not result.success:
        row = dict(context)
        row.update({"failure_reason": str(result.message), "fallback_type": "previous_or_naive"})
        failures.append(row)
        return normalize_weights(fallback)
    return normalize_weights(result.x)


def gmv_weights(cov: np.ndarray, failures: List[dict], context: dict, start: Optional[np.ndarray] = None) -> np.ndarray:
    n = cov.shape[0]
    fallback = naive_weights(n)
    return constrained_optimize(lambda w: float(w @ cov @ w), n, start, failures, context, fallback)


def mv_weights(mu: np.ndarray, cov: np.ndarray, gamma: float, failures: List[dict], context: dict, start: Optional[np.ndarray] = None) -> np.ndarray:
    n = len(mu)
    fallback = normalize_weights(start if start is not None else naive_weights(n))
    return constrained_optimize(lambda w: -(float(w @ mu) - gamma / 2 * float(w @ cov @ w)), n, start, failures, context, fallback)


def fund2_weights(mu: np.ndarray, cov: np.ndarray, gamma: float, w_naive: np.ndarray, w_gmv: np.ndarray, failures: List[dict], context: dict) -> np.ndarray:
    base = np.vstack([w_naive, w_gmv])
    def objective(alpha):
        w = alpha @ base
        return -(float(w @ mu) - gamma / 2 * float(w @ cov @ w))
    alpha = constrained_optimize(objective, 2, np.ones(2) / 2, failures, context, np.ones(2) / 2)
    return normalize_weights(alpha @ base)


def lower_partial_moment(portfolio_returns: np.ndarray, threshold: float) -> float:
    downside = np.minimum(portfolio_returns - threshold, 0.0)
    return float(np.mean(downside ** 2))


def ml_weights(returns: np.ndarray, threshold: float, lpm_penalty: float, failures: List[dict], context: dict, start: Optional[np.ndarray] = None) -> np.ndarray:
    n = returns.shape[1]
    values = np.asarray(returns, dtype=float)
    fallback = normalize_weights(start if start is not None else naive_weights(n))
    def objective(w):
        rp = values @ w
        return -float(np.mean(rp)) + lpm_penalty * lower_partial_moment(rp, threshold)
    return constrained_optimize(objective, n, start, failures, context, fallback)


def mls_weights(returns: np.ndarray, threshold: float, lpm_penalty: float, skew_penalty: float, failures: List[dict], warnings: List[dict], context: dict, start: Optional[np.ndarray] = None) -> np.ndarray:
    n = returns.shape[1]
    values = np.asarray(returns, dtype=float)
    fallback = normalize_weights(start if start is not None else naive_weights(n))
    def objective(w):
        rp = values @ w
        std = float(np.std(rp, ddof=1))
        if std <= 1e-8:
            skew = 0.0
        else:
            z = (rp - np.mean(rp)) / std
            skew = float(np.mean(z ** 3))
        return -float(np.mean(rp)) + lpm_penalty * lower_partial_moment(rp, threshold) - skew_penalty * skew
    w = constrained_optimize(objective, n, start, failures, context, fallback)
    if not np.isfinite(w).all():
        row = dict(context)
        row.update({"warning": "nonfinite_mls_weights_replaced_by_fallback"})
        warnings.append(row)
        return fallback
    return w



