from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def _optimization_setting(row) -> str:
    return row["method"] if row["gamma"] == "NA" else f"{row['method']}_gamma{int(float(row['gamma']))}"


def _rdm_group(data: pd.DataFrame, input_cols: list[str], output_cols: list[str], model_name: str) -> tuple[pd.DataFrame, list[dict]]:
    diagnostics = []
    original_n = len(data)
    data = data.copy()
    data = data.dropna(subset=input_cols + output_cols).reset_index(drop=True)
    removed_n = original_n - len(data)
    horizon = data["horizon"].iloc[0] if not data.empty else np.nan
    diagnostics.append(
        {
            "DEA_model": model_name,
            "horizon": horizon,
            "dmu_count_original": original_n,
            "dmu_count_used": len(data),
            "dmu_count_removed_missing": removed_n,
            "infeasible_count": 0,
        }
    )
    if data.empty:
        diagnostics.append({"DEA_model": model_name, "horizon": horizon, "warning": "empty_dea_input"})
        return data.assign(DEA_efficiency=np.nan, theta=np.nan), diagnostics

    x = data[input_cols].astype(float).to_numpy()
    y = data[output_cols].astype(float).to_numpy()
    x_min = np.min(x, axis=0)
    y_max = np.max(y, axis=0)
    dx = x - x_min
    dy = y_max - y
    for col, rng in zip(input_cols, np.max(x, axis=0) - x_min):
        if abs(float(rng)) <= 1e-12:
            diagnostics.append({"DEA_model": model_name, "horizon": horizon, "metric": col, "warning": "zero_input_range_direction_set_to_zero"})
    for col, rng in zip(output_cols, y_max - np.min(y, axis=0)):
        if abs(float(rng)) <= 1e-12:
            diagnostics.append({"DEA_model": model_name, "horizon": horizon, "metric": col, "warning": "zero_output_range_direction_set_to_zero"})

    n = len(data)
    theta_values = []
    infeasible_count = 0
    for k in range(n):
        # Variables are lambda_1..lambda_n and theta. Max theta => min -theta.
        c = np.r_[np.zeros(n), -1.0]
        a_ub = []
        b_ub = []
        for m in range(len(input_cols)):
            # sum lambda*x_jm <= x_km - theta*(x_km - min_m)
            a_ub.append(np.r_[x[:, m], dx[k, m]])
            b_ub.append(x[k, m])
        for r in range(len(output_cols)):
            # sum lambda*y_jr >= y_kr + theta*(max_r - y_kr)
            # -sum lambda*y_jr + theta*(max_r - y_kr) <= -y_kr
            a_ub.append(np.r_[-y[:, r], dy[k, r]])
            b_ub.append(-y[k, r])
        a_eq = [np.r_[np.ones(n), 0.0]]
        b_eq = [1.0]
        bounds = [(0.0, None)] * n + [(0.0, 1.0)]
        res = linprog(c, A_ub=np.asarray(a_ub), b_ub=np.asarray(b_ub), A_eq=np.asarray(a_eq), b_eq=np.asarray(b_eq), bounds=bounds, method="highs")
        if not res.success:
            theta_values.append(np.nan)
            infeasible_count += 1
            diagnostics.append(
                {
                    "DEA_model": model_name,
                    "horizon": data.loc[k, "horizon"],
                    "case_type": data.loc[k, "case_type"],
                    "optimization_setting": data.loc[k, "optimization_setting"],
                    "warning": "dea_infeasible",
                    "failure_reason": res.message,
                }
            )
        else:
            theta_values.append(float(res.x[-1]))
    out = data.copy()
    out["theta"] = theta_values
    out["DEA_efficiency"] = 1 - out["theta"]
    out["DEA_model"] = model_name
    diagnostics[0]["infeasible_count"] = infeasible_count
    return out, diagnostics


def dea_efficiency(performance: pd.DataFrame, include_sortino: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_name = "DEA_Sharpe_Sortino" if include_sortino else "DEA_Sharpe_Only"
    input_cols = ["Turnover", "CostDrag_50"]
    output_cols = ["Sharpe"] + (["Sortino"] if include_sortino else [])
    data = performance.copy()
    data["optimization_setting"] = data.apply(_optimization_setting, axis=1)
    outputs = []
    diagnostics = []
    for horizon, group in data.groupby("horizon", sort=True):
        out, diag = _rdm_group(group.copy(), input_cols, output_cols, model_name)
        outputs.append(out)
        diagnostics.extend(diag)
    result = pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
    diag_df = pd.DataFrame(diagnostics) if diagnostics else pd.DataFrame({"message": ["No DEA diagnostics."]})
    return result, diag_df

