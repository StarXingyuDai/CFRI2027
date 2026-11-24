"""Reproduce the revised analysis from the two daily input workbooks."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from Data_Preparation_02 import ASSETS, weekly_inputs
from Portfolio_Methods_03 import (
    fund2_weights, gmv_weights, inverse_variance_weights, ml_weights, mls_weights,
    mv_weights, naive_weights, regularize_cov,
)
from Black_Litterman_04 import forecast_view, posterior_moments
from Inference_05 import adjust_family, pair_inference
from DEA_07 import dea_efficiency


ROOT = Path(__file__).resolve().parent
HORIZONS = [1, 2, 4, 7, 13, 26, 52]
WINDOW = 52
GAMMAS = (2, 5, 10)
METHODS = {
    "1/N": (), "IV": (), "MV": GAMMAS, "BL": GAMMAS, "Fund2": GAMMAS, "ML": (), "MLS": (),
}
COSTS = (20, 50)
KEYS = ["horizon", "method", "gamma"]
PORTFOLIOS = ("Benchmark", "Renewable")
# region 1. Estimation, holding periods, and costs
def turnover_from_drift(previous_weights, asset_returns_pct, targets):
    gross = 1 + asset_returns_pct / 100
    drift = previous_weights * gross
    drift /= drift.sum(axis=1, keepdims=True)
    return np.abs(targets - drift).sum(axis=1)


def allocate(values, forecast_steps, previous, diag, context):
    mu = values.mean(axis=0)
    cov = regularize_cov(np.cov(values, rowvar=False), diag["covariance_repairs"], context)
    n = values.shape[1]
    base_equal = naive_weights(n)
    base_gmv = gmv_weights(cov, diag["optimizer_failures"], {**context, "method": "GMV"})
    views, omega, status = forecast_view(values, forecast_steps)
    diag["var_status"].append({**context, "status": status, "training_n": len(values), "forecast_steps": forecast_steps})
    allocations = {}
    for method, gammas in METHODS.items():
        for gamma in gammas or ("NA",):
            key = (method, gamma)
            start = previous.get(key)
            local = {**context, "method": method, "gamma": gamma}
            failures = diag["optimizer_failures"]
            if method == "1/N":
                target = base_equal
            elif method == "IV":
                target = inverse_variance_weights(cov, diag["other_warnings"], local)
            elif method == "MV":
                target = mv_weights(mu, cov, gamma, failures, local, start=start)
            elif method == "BL":
                bl_mu, bl_cov = posterior_moments(cov, views, omega, gamma, len(values), diag["covariance_repairs"], local)
                target = mv_weights(bl_mu, bl_cov, gamma, failures, local, start=start)
            elif method == "Fund2":
                target = fund2_weights(mu, cov, gamma, base_equal, base_gmv, failures, local)
            elif method == "ML":
                target = ml_weights(values, 0.0, 1.0, failures, local, start=start)
            elif method == "MLS":
                target = mls_weights(values, 0.0, 1.0, 0.1, failures, diag["other_warnings"], local, start=start)
            allocations[key] = target
    return allocations


def run_horizon(weekly, horizon, diag):
    prices = weekly[ASSETS]
    # Simple asset returns in percent. The same returns feed estimation and realized performance.
    returns = (100 * (prices / prices.shift(horizon) - 1)).to_numpy()
    dates = weekly["date"].reset_index(drop=True)
    annual_yields_pct = weekly["yield_1y_pct"].to_numpy()
    # First training endpoints are h,...,h+51. Formation is one mark later.
    formations = np.arange(horizon + WINDOW, len(weekly) - horizon)
    maturity = formations + horizon
    if len(formations) <= horizon:
        raise ValueError(f"At h={horizon}, a holding phase would lack a turnover observation.")
    days = (dates.iloc[maturity].to_numpy() - dates.iloc[formations].to_numpy()) / np.timedelta64(1, "D")
    # Equation (1): Compound the annual yield quoted at formation over actual calendar days.
    # The result is a simple holding-period return in percent. log1p/expm1 evaluate the power stably.
    rf = 100 * np.expm1(np.log1p(annual_yields_pct[formations] / 100) * days / 365)
    schedule = pd.DataFrame({
        "horizon": horizon, "phase": np.arange(len(formations)) % horizon,
        "formation": dates.iloc[formations].to_numpy(), "maturity": dates.iloc[maturity].to_numpy(),
        "first_training_endpoint": dates.iloc[formations - WINDOW].to_numpy(),
        "last_training_endpoint": dates.iloc[formations - 1].to_numpy(),
        "holding_days": days.astype(int), "risk_free_pct": rf,
    })
    allocations = {group: defaultdict(list) for group in PORTFOLIOS}
    for i in formations:
        assert i - WINDOW >= horizon
        for group, count in zip(PORTFOLIOS, (len(ASSETS) - 1, len(ASSETS))):
            context = {"portfolio": group, "horizon": horizon, "formation": dates.iloc[i]}
            previous = {key: history[-horizon] for key, history in allocations[group].items()
                        if len(history) >= horizon}
            targets = allocate(returns[i - WINDOW:i, :count], horizon + 1, previous, diag, context)
            for key, target in targets.items():
                allocations[group][key].append(target)

    detail = []
    weights = []
    for group, records in allocations.items():
        count = len(ASSETS) if group == "Renewable" else len(ASSETS) - 1
        actual = returns[maturity, :count]
        for (method, gamma), history in records.items():
            targets = np.stack(history)
            if np.any(targets < -1e-10) or not np.allclose(targets.sum(axis=1), 1):
                raise ValueError(f"Long-only budget violation at h={horizon}, {group}, {method}, {gamma}.")
            gross = (targets * actual).sum(axis=1)
            turn = np.r_[np.full(horizon, np.nan), turnover_from_drift(targets[:-horizon], actual[:-horizon], targets[horizon:])]
            assert np.all((turn[horizon:] >= 0) & (turn[horizon:] <= 2 + 1e-10))
            frame = schedule.assign(portfolio=group, method=method, gamma=str(gamma),
                                    gross_pct=gross, excess_pct=gross - rf, turnover=turn)
            for bps in COSTS:
                drag = (100 + gross) * (bps / 10000) * np.nan_to_num(turn)
                frame[f"cost_drag_{bps}"] = drag
                frame[f"net_{bps}_pct"] = gross - drag
                assert np.allclose(1 + frame[f"net_{bps}_pct"] / 100,
                                   (1 + gross / 100) * (1 - bps / 10000 * np.nan_to_num(turn)))
            detail.append(frame)
            weight_frame = pd.DataFrame(targets, columns=ASSETS[:count])
            for col in ["horizon", "phase", "formation", "maturity"]:
                weight_frame[col] = schedule[col]
            weights.append(weight_frame.assign(portfolio=group, method=method, gamma=str(gamma)))
    phases = schedule.groupby("phase").size()
    metadata = {
        "horizon": horizon, "training_observations": WINDOW, "var_usable_rows": WINDOW - 2,
        "var_forecast_steps": horizon + 1, "first_formation": dates.iloc[formations[0]],
        "first_maturity": dates.iloc[maturity[0]], "last_maturity": dates.iloc[maturity[-1]],
        "out_of_sample_n": len(formations), "number_of_phases": horizon,
        "phase_n_min": phases.min(), "phase_n_max": phases.max(),
        "holding_days_min": int(days.min()), "holding_days_max": int(days.max()),
    }
    return pd.concat(detail, ignore_index=True), pd.concat(weights, ignore_index=True), metadata, schedule, returns
# endregion


# region 2. Performance, matched inference, and non-overlapping phase checks
def excess_return_moments(excess):
    """Return the numerator and denominators of the gross risk-adjusted ratios."""
    volatility = np.std(excess, ddof=1)
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    return excess.mean(), volatility, downside


def risk_adjusted_components(returns):
    """Report the return and risk components used to interpret Sharpe and Sortino."""
    columns = [*KEYS, "portfolio"]
    rows = []
    for key, frame in returns.groupby(columns, sort=False):
        excess = frame["excess_pct"].to_numpy()
        mean, volatility, downside = excess_return_moments(excess)
        shortfalls = excess[excess < 0]
        rows.append({
            **dict(zip(columns, key)),
            "observations": len(excess),
            "mean_excess_pct": mean,
            "excess_volatility_pct": volatility,
            "downside_deviation_pct": downside,
            "shortfall_share": len(shortfalls) / len(excess),
            "shortfall_rms_pct": np.sqrt(np.mean(shortfalls ** 2)) if len(shortfalls) else np.nan,
        })
    return pd.DataFrame(rows)


def performance_metrics(frame):
    excess = frame["excess_pct"].to_numpy()
    mean, volatility, downside = excess_return_moments(excess)
    out = {
        "MeanGross": frame["gross_pct"].mean(), "GrossVolatility": frame["gross_pct"].std(ddof=1),
        "MeanExcess": mean, "Sharpe": mean / volatility if volatility > 0 else np.nan,
        "Sortino": mean / downside if downside > 0 else np.nan,
        "Turnover": frame["turnover"].mean(), "observations": len(frame),
        "first_maturity": frame["maturity"].iloc[0], "last_maturity": frame["maturity"].iloc[-1],
    }
    for bps in COSTS:
        out[f"CostDrag_{bps}"] = frame[f"cost_drag_{bps}"].mean()
        out[f"MeanNet_{bps}"] = frame[f"net_{bps}_pct"].mean()
        out[f"NetVolatility_{bps}"] = frame[f"net_{bps}_pct"].std(ddof=1)
    return out


def economic_comparison(base, case):
    result = {
        "sharpe_delta": case["Sharpe"] - base["Sharpe"],
        "sortino_delta": case["Sortino"] - base["Sortino"],
        "turnover_delta": case["Turnover"] - base["Turnover"],
        "gross_volatility_scale": case["GrossVolatility"] / base["GrossVolatility"],
    }
    scale = result["gross_volatility_scale"]
    result["fixed_risk_gross_gain_pct"] = case["MeanGross"] - scale * base["MeanGross"]
    for bps in COSTS:
        key = f"NetVolatility_{bps}"
        result[f"return_loss_{bps}_pct"] = base[f"MeanNet_{bps}"] * case[key] / base[key] - case[f"MeanNet_{bps}"]
        result[f"fixed_risk_extra_cost_{bps}_pct"] = case[f"CostDrag_{bps}"] - scale * base[f"CostDrag_{bps}"]
        result[f"fixed_risk_net_gain_{bps}_pct"] = result["fixed_risk_gross_gain_pct"] - result[f"fixed_risk_extra_cost_{bps}_pct"]
        assert np.isclose(result[f"fixed_risk_net_gain_{bps}_pct"], case[f"MeanNet_{bps}"] - scale * base[f"MeanNet_{bps}"])
    return result


def summarize(returns):
    all_perf, inference, contrasts, phases = [], [], [], []
    for keys, pair in returns.groupby(KEYS, sort=False):
        meta = dict(zip(KEYS, keys))
        groups = {name: frame.sort_values("maturity") for name, frame in pair.groupby("portfolio")}
        base, case = (groups[group] for group in PORTFOLIOS)
        for col in ["formation", "maturity", "phase", "risk_free_pct"]:
            assert np.array_equal(base[col], case[col]), f"Unpaired {col} at {meta}"
        metrics = {name: performance_metrics(frame) for name, frame in groups.items()}
        all_perf.extend({**meta, "case_type": name, **stats} for name, stats in metrics.items())
        contrasts.append({**meta, **economic_comparison(*(metrics[group] for group in PORTFOLIOS))})
        inference.append({**meta, **pair_inference(case["excess_pct"], base["excess_pct"], keys[0], WINDOW)})
        for phase, base_phase in base.groupby("phase"):
            case_phase = case[case["phase"].eq(phase)]
            n = len(base_phase)
            assert np.array_equal(base_phase["maturity"].to_numpy()[:-1], base_phase["formation"].to_numpy()[1:])
            phases.append({
                **meta, "phase": phase, "observations": n,
                "first_formation": base_phase["formation"].iloc[0],
                "first_maturity": base_phase["maturity"].iloc[0],
                **economic_comparison(performance_metrics(base_phase), performance_metrics(case_phase)),
            })
    contrasts = pd.DataFrame(contrasts)
    inference = adjust_family(pd.DataFrame(inference))
    assert np.allclose(contrasts["sharpe_delta"], inference["sharpe_difference"])
    return pd.DataFrame(all_perf), contrasts, inference, pd.DataFrame(phases)


def phase_summary(phases):
    rows = []
    for keys, group in phases.groupby(KEYS, sort=False):
        delta = group["sharpe_delta"]
        loss = group["return_loss_50_pct"]
        rows.append({
            **dict(zip(KEYS, keys)), "phases": len(group), "n_min": group["observations"].min(),
            "n_max": group["observations"].max(), "positive_sharpe_phases": int(delta.gt(0).sum()),
            "sharpe_delta_min": delta.min(), "sharpe_delta_median": delta.median(), "sharpe_delta_max": delta.max(),
            "negative_loss_50_phases": int(loss.lt(0).sum()), "return_loss_50_median_pct": loss.median(),
        })
    return pd.DataFrame(rows)


def weight_summary(weights):
    indicators = []
    for label in ASSETS:
        valid = weights[weights[label].notna()]
        for key, group in valid.groupby([*KEYS, "portfolio"], sort=False):
            values = group[label]
            indicators.append({
                **dict(zip([*KEYS, "portfolio"], key)), "asset": label,
                "mean_weight": values.mean(), "positive_weight_share": values.gt(1e-8).mean(),
            })
    return pd.DataFrame(indicators)
# endregion


# region 3. Export tables, figures, and the verification trail
def save_csv(frame, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, date_format="%Y-%m-%d", float_format="%.15g")


def run(output):
    weekly = weekly_inputs().reset_index(drop=True)
    assert (weekly["yield_date"] < weekly["date"]).all()
    save_csv(weekly, output / "data" / "weekly_inputs.csv")
    diag = defaultdict(list)
    return_sets, weight_sets, coverage, schedules, descriptions = [], [], [], [], []
    for horizon in HORIZONS:
        print(f"Estimating h={horizon}", flush=True)
        ret, weights, counts, schedule, asset_returns = run_horizon(weekly, horizon, diag)
        return_sets.append(ret)
        weight_sets.append(weights)
        coverage.append(counts)
        schedules.append(schedule)
        for idx, asset in enumerate(ASSETS):
            s = pd.Series(asset_returns[horizon:, idx])
            descriptions.append({"horizon": horizon, "asset": asset, "n": len(s), "mean_pct": s.mean(),
                                 "sd_pct": s.std(), "min_pct": s.min(), "max_pct": s.max(),
                                 "skewness": s.skew(), "excess_kurtosis": s.kurt()})
    returns = pd.concat(return_sets, ignore_index=True)
    weights = pd.concat(weight_sets, ignore_index=True)
    print("Summarizing matched results", flush=True)
    perf, differences, inference, phases = summarize(returns)
    weights_agg = weight_summary(weights)
    dea, dea_diag = dea_efficiency(perf, include_sortino=True)
    dea_no_sortino, dea_no_diag = dea_efficiency(perf, include_sortino=False)
    dea_pivot = dea.pivot(index=KEYS, columns="case_type", values="DEA_efficiency").reset_index()
    dea_pivot["dea_delta"] = dea_pivot["Renewable"] - dea_pivot["Benchmark"]
    dea_alt = dea_no_sortino.pivot(index=KEYS, columns="case_type", values="DEA_efficiency").reset_index()
    dea_pivot["dea_delta_without_sortino"] = dea_alt["Renewable"] - dea_alt["Benchmark"]
    dea_summary = dea_pivot.groupby("method").agg(
        mean_delta=("dea_delta", "mean"), median_delta=("dea_delta", "median"),
        mean_delta_without_sortino=("dea_delta_without_sortino", "mean"),
    ).reset_index()
    tables = {
        "table_02_sample": pd.DataFrame(coverage), "table_03_sharpe_inference": inference,
        "table_04_economic_comparison": differences, "table_05_dea_summary": dea_summary,
        "nonoverlapping_phase_summary": phase_summary(phases), "table_A1_asset_statistics": pd.DataFrame(descriptions),
        "performance_by_portfolio": perf, "risk_adjusted_components": risk_adjusted_components(returns),
        "dea_by_portfolio": dea, "dea_alternative": dea_no_sortino,
        "dea_matched_differences": dea_pivot, "nonoverlapping_phase_results": phases,
        "asset_weights_summary": weights_agg,
        "dea_diagnostics": pd.concat([dea_diag, dea_no_diag], ignore_index=True),
    }
    for name, frame in tables.items():
        save_csv(frame, output / "tables" / f"{name}.csv")
    save_csv(pd.concat(schedules, ignore_index=True), output / "data" / "formation_calendar.csv")
    save_csv(returns, output / "data" / "portfolio_returns.csv")
    save_csv(weights, output / "data" / "portfolio_weights.csv")
    for name in ["var_status", "covariance_repairs", "optimizer_failures", "other_warnings"]:
        save_csv(pd.DataFrame(diag[name]) if diag[name] else pd.DataFrame({"status": ["no incidents"]}),
                 output / "tables" / "diagnostics" / f"{name}.csv")
    assert len(inference) == len(HORIZONS) * sum(max(len(g), 1) for g in METHODS.values())
    assert len(returns) == len(weights)
    assert len(dea_pivot) == len(inference)
    from Figures_08 import draw_figures
    draw_figures(differences, dea_pivot, weights_agg, output / "figures")
    print(f"Completed {len(inference)} matched comparisons. Outputs: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT, help="Optional directory for generated data, tables, and figures.")
    run(parser.parse_args().output.resolve())
# endregion

