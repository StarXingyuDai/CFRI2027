from __future__ import annotations

import numpy as np
from scipy.stats import norm


def jobson_korkie_test(case_returns, bench_returns, rf_returns):
    raw_case = np.asarray(case_returns, dtype=float)
    raw_bench = np.asarray(bench_returns, dtype=float)
    rf = np.asarray(rf_returns, dtype=float)
    mask = np.isfinite(raw_case) & np.isfinite(raw_bench) & np.isfinite(rf)
    raw_case = raw_case[mask]
    raw_bench = raw_bench[mask]
    rf = rf[mask]
    x = raw_case - rf
    y = raw_bench - rf
    n = len(x)
    if n < 5:
        return np.nan, np.nan, np.nan
    sx = np.std(raw_case, ddof=1)
    sy = np.std(raw_bench, ddof=1)
    if sx <= 0 or sy <= 0:
        return np.nan, np.nan, np.nan
    sr_x = np.mean(x) / sx
    sr_y = np.mean(y) / sy
    rho = np.corrcoef(raw_case, raw_bench)[0, 1]
    var = (1 / n) * (2 * (1 - rho) + 0.5 * (sr_x ** 2 + sr_y ** 2 - 2 * sr_x * sr_y * rho ** 2))
    if var <= 0 or not np.isfinite(var):
        return np.nan, np.nan, np.nan
    z = (sr_x - sr_y) / np.sqrt(var)
    return float(z), float(norm.sf(z)), float(2 * (norm.sf(abs(z))))

