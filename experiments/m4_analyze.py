"""
M4 analysis: tables and figures from the CSVs of m4b (supervised / ANN), m4c (GPO) and m4d (RL).
Run from the repo root:  python experiments/m4_analyze.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tuners.plots import _save

OUT = "results/methods/M4"
FIG = f"{OUT}/figures"
ORDER = ["constant", "ridge-poly2", "knn", "svr", "mlp", "extra-trees", "ann-supervised", "ann-decision", "ann"]
LABEL = {"ann": "ANN (combined loss)", "ann-supervised": "ANN (supervised)", "ann-decision": "ANN (decision-focused)"}


def md(df, fmt="{:.4g}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join([df.index.name or ""] + [str(c) for c in cols]) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]) + " |")
    return "\n".join(lines)


def fig_compare():
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
    for ax, suf, title in zip(axes, ("", "_noisy"), ("ideal identification (noise-free bump test)", "realistic: noisy bump test")):
        p = f"{OUT}/M4_compare{suf}.csv"
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        x = np.arange(len(ORDER))
        for k, (split, c) in enumerate((("val", "#1f77b4"), ("test", "#2ca02c"), ("ood", "#d62728"))):
            g = d[d.split == split].set_index("model").reindex(ORDER)
            ax.bar(x + (k - 1) * 0.27, g["median"], 0.27, color=c, label=f"{split} median")
            ax.scatter(x + (k - 1) * 0.27, g["p90"], color=c, marker="_", s=120, zorder=3)
        ax.set_xticks(x, [LABEL.get(m, m) for m in ORDER], rotation=40, ha="right")
        ax.set_yscale("log"); ax.axhline(1.0, color="k", lw=0.8); ax.grid(alpha=0.3, which="both", axis="y")
        ax.set_title(title)
    axes[0].set_ylabel("regret = ITAE / label (bar: median, tick: 90th percentile)")
    axes[0].legend(fontsize=8)
    _save(fig, f"{FIG}/M4_compare_regret.png")


def tables_compare():
    for suf in ("", "_noisy"):
        p = f"{OUT}/M4_compare{suf}.csv"
        if os.path.exists(p):
            d = pd.read_csv(p)
            t = d.pivot(index="model", columns="split", values="median").reindex(ORDER)
            t.to_csv(f"{OUT}/M4_table_median{suf}.csv")
            d.pivot(index="model", columns="split", values="p90").reindex(ORDER).to_csv(f"{OUT}/M4_table_p90{suf}.csv")
            d.pivot(index="model", columns="split", values="within1p5").reindex(ORDER).to_csv(f"{OUT}/M4_table_within1p5{suf}.csv")
        pp = f"{OUT}/M4_compare_perplant{suf}.csv"
        if os.path.exists(pp):
            d = pd.read_csv(pp)
            t = d[d.model.isin(["constant", "svr", "extra-trees", "ann-decision", "ann"])].groupby(["model", "ptype"]).ratio.median().unstack("ptype")
            t.to_csv(f"{OUT}/M4_regret_by_type{suf}.csv")


def fig_ablations(suf):
    p = f"{OUT}/M4_ablations{suf}.csv"
    if not os.path.exists(p):
        return
    a = pd.read_csv(p)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    g = a[a.ablation == "train size"].copy(); g["value"] = g.value.astype(float)
    for m, c in (("svr", "#1f77b4"), ("ann", "#ff7f0e")):
        s = g[g.model == m].sort_values("value")
        axes[0, 0].plot(s.value, s["median"], marker="o", color=c, label=m)
        axes[0, 0].fill_between(s.value, s["median"], s["p90"], color=c, alpha=0.12)
    axes[0, 0].set_xscale("log"); axes[0, 0].set_yscale("log"); axes[0, 0].set_title("learning curve (median .. p90)"); axes[0, 0].legend()
    for ax, name in zip(axes.ravel()[1:], ["feature set", "label variant", "ANN loss", "ANN features", "feature noise"]):
        s = a[a.ablation == name]
        if len(s):
            ax.bar([str(v) for v in s.value], s["median"], color="#1f77b4", alpha=0.8)
            ax.scatter([str(v) for v in s.value], s["p90"], marker="_", s=150, color="k")
            ax.set_yscale("log"); ax.set_title(name); ax.tick_params(axis="x", rotation=25)
    for ax in axes.ravel():
        ax.axhline(1.0, color="k", lw=0.6); ax.grid(alpha=0.3, which="both", axis="y")
    _save(fig, f"{FIG}/M4_ablations{suf}.png")


def fig_noise_robust():
    p = f"{OUT}/M4_noise_robust.csv"
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    d = d[~d.drop_fit_error]
    d["cond"] = "train " + d.train + " / test " + d.test
    t = d.pivot(index="cond", columns="model", values="median")
    t.to_csv(f"{OUT}/M4_noise_robust_table.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    t.plot.bar(ax=ax, logy=True, rot=20)
    ax.axhline(1.0, color="k", lw=0.6); ax.grid(alpha=0.3, which="both", axis="y")
    ax.set_ylabel("median regret"); ax.set_title("Clean-trained models collapse on noisy bump-test features; noisy-trained ones recover")
    _save(fig, f"{FIG}/M4_noise_robust.png")


def final_summary():
    for suf in ("", "_noisy"):
        p = f"{OUT}/M4_final_eval{suf}.csv"
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        for sc in ("step", "load", "noise"):
            d[f"{sc}_ratio"] = d[sc] / d[f"{sc}_ref"]
        t = d.groupby(["split", "model"])[["step_ratio", "load_ratio", "noise_ratio"]].median()
        t.to_csv(f"{OUT}/M4_final_summary{suf}.csv")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    tables_compare(); fig_compare(); fig_ablations(""); fig_ablations("_noisy"); fig_noise_robust(); final_summary()
    print("M4 analysis written")
