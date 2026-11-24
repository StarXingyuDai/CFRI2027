"""Generate the numbered manuscript figures and their plotting data."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


METHOD_ORDER = ["1/N", "IV", "MV", "BL", "Fund2", "ML", "MLS"]
HORIZONS = [1, 2, 4, 7, 13, 26, 52]


def moderate(frame):
    frame = frame[frame["gamma"].astype(str).isin(["NA", "5"])].copy()
    return frame


def heatmap(frame, metric, title, color_label, root, number, favor_negative=False, percentage_max=None):
    panel = moderate(frame).pivot(index="method", columns="horizon", values=metric).reindex(index=METHOD_ORDER, columns=HORIZONS)
    panel.to_csv(root / f"figure_{number:02d}_data.csv")
    z = panel.to_numpy() * (100 if metric == "turnover_delta" else 1)
    vmax = max(np.nanmax(np.abs(z)), 1e-5)
    fig, ax = plt.subplots(figsize=(8.1, 3.6), layout="constrained")
    if percentage_max is None:
        im = ax.imshow(z, cmap="RdBu_r" if favor_negative else "RdBu",
                       norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax), aspect="auto")
        label_limit = vmax
    else:
        percentage_max = max(percentage_max, vmax)
        im = ax.imshow(z, cmap="Blues", vmin=0, vmax=percentage_max, aspect="auto")
        label_limit = percentage_max
    labels = [str(h) for h in HORIZONS]
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(METHOD_ORDER)), METHOD_ORDER)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("Holding horizon (retained weeks)")
    ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=10)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            value = z[i, j]
            if np.isfinite(value):
                if percentage_max is None:
                    label = "0.000" if abs(value) < .0005 else f"{value:+.3f}"
                else:
                    label = f"{value:.1f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8,
                        color="white" if abs(value) > .64 * label_limit else "#152331")
    color = fig.colorbar(im, ax=ax, pad=.024, aspect=22)
    color.set_label(color_label, fontsize=9)
    fig.get_layout_engine().set(rect=(0, 0.055, 1, 0.945))
    fig.text(0.015, 0.024, "Source: Authors' calculations.", fontsize=8)
    fig.savefig(root / f"figure_{number:02d}.png", dpi=260, facecolor="white")
    plt.close(fig)


def draw_figures(differences, efficiencies, weights, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "text.color": "#152331",
                         "axes.labelcolor": "#152331", "axes.titlecolor": "#152331"}):
        heatmap(differences, "sharpe_delta", "(a) Difference in Sharpe ratios", "Difference (unitless)", root, 1)
        heatmap(differences, "sortino_delta", "(b) Difference in Sortino ratios", "Difference (unitless)", root, 2)
        heatmap(differences, "turnover_delta", "Difference in turnover", "pp per rebalance", root, 3, True)
        heatmap(differences, "return_loss_50_pct", "(a) Return loss at 50 bp per side", "pp per holding", root, 4, True)
        heatmap(differences, "return_loss_20_pct", "(b) Return loss at 20 bp per side", "pp per holding", root, 5, True)
        heatmap(efficiencies, "dea_delta", "Difference in DEA efficiency", "Difference (score units)", root, 6)
        target = weights.loc[weights["portfolio"].eq("Renewable") & weights["asset"].eq("new_energy")].copy()
        target["mean_weight_pct"] = 100 * target["mean_weight"]
        target["positive_weight_pct"] = 100 * target["positive_weight_share"]
        heatmap(target, "mean_weight_pct", "(a) Mean weight in the proxy index", "Percent", root, 7, percentage_max=25)
        heatmap(target, "positive_weight_pct", "(b) Frequency of positive weights in the proxy index", "Percent of formations", root, 8, percentage_max=100)

