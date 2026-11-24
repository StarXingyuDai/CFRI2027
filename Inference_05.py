"""Matched inference for the difference between two unannualized Sharpe ratios."""
import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from Jobson_Korkie_06 import jobson_korkie_test


def sharpe_hac(case_excess, benchmark_excess, max_lag):
    """Delta-method HAC test for the difference, using paired weekly observations."""
    x = np.column_stack((case_excess, benchmark_excess))
    n = len(x)
    means = x.mean(axis=0)
    centered = x - means
    variances = np.mean(centered ** 2, axis=0)
    if np.any(variances == 0):
        return np.nan, np.nan, np.nan
    sigmas = np.sqrt(variances)
    # The Sharpe denominator uses sample SD (ddof=1), as in the descriptive outputs.
    finite_sample = np.sqrt((n - 1) / n)
    contributions = finite_sample * (
        centered / sigmas - means * (centered ** 2 - variances) / (2 * sigmas ** 3)
    )
    influence = contributions[:, 0] - contributions[:, 1]
    long_run_variance = np.dot(influence, influence) / n
    for lag in range(1, max_lag + 1):
        covariance = np.dot(influence[lag:], influence[:-lag]) / n
        long_run_variance += 2 * (1 - lag / (max_lag + 1)) * covariance
    difference = finite_sample * (means[0] / sigmas[0] - means[1] / sigmas[1])
    se = np.sqrt(long_run_variance / n) if long_run_variance > 0 else np.nan
    return difference, se, float(2 * norm.sf(abs(difference / se)))


def pair_inference(case_excess, benchmark_excess, horizon, window):
    # With h-period overlap and shared estimation data, dependence can extend beyond h-1.
    # The prior same-phase weights also enter net returns, which we analyze descriptively.
    main_lag = min(window + 2 * horizon, len(case_excess) - 1)
    alt_lag = min(2 * main_lag, len(case_excess) - 1)
    diff, se, p = sharpe_hac(case_excess, benchmark_excess, main_lag)
    _, se_alt, p_alt = sharpe_hac(case_excess, benchmark_excess, alt_lag)
    jk_z, jk_one, jk_two = jobson_korkie_test(case_excess, benchmark_excess, np.zeros(len(case_excess)))
    return {
        "sharpe_difference": diff, "hac_lag": main_lag, "hac_se": se, "hac_p_two_sided": p,
        "ci95_low": diff - 1.96 * se, "ci95_high": diff + 1.96 * se, "hac_alt_lag": alt_lag,
        "hac_alt_se": se_alt, "hac_alt_p_two_sided": p_alt, "jk_z": jk_z,
        "jk_p_one_sided_positive": jk_one, "jk_p_two_sided": jk_two,
    }


def adjust_family(frame):
    for raw, adjusted in [
        ("hac_p_two_sided", "holm_p"),
        ("hac_alt_p_two_sided", "holm_alt_p"),
    ]:
        valid = frame[raw].notna()
        if valid.any():
            frame.loc[valid, adjusted] = multipletests(frame.loc[valid, raw], method="holm")[1]
    return frame

