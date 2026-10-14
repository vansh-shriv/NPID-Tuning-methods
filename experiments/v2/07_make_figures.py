"""
Experiment 07 - Figures for the v2 report, built only from the CSVs in results/v2/.

    fig1_fair_comparison.png   clean vs noisy ITAE, every tuning treatment (Phase 1)
    fig2_multi_plant.png       ITAE relative to the best controller on each plant (Phase 4)
    fig3_ml_regret.png         held-out regret per model + extrapolation sweep (Phase 5)
    fig4_drift.png             drift baselines (Phase 6)

Run from repo root:  python experiments/v2/07_make_figures.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = "results/v2"
OUT = f"{R}/figures"
os.makedirs(OUT, exist_ok=True)


def fig1():
    d = pd.read_csv(f"{R}/01_fair_comparison.csv")
    keep = ["L-PID ZN [raw]", "L-PID PSO [raw]", "L-PID PSO [filt10ms]", "L-PID PSO-noisy [filt10ms]",
            "N-PID ZN [raw]", "N-PID PSO [raw]", "N-PID(scaled) PSO [raw]", "N-PID(scaled) PSO [filt10ms]",
            "N-PID PSO-noisy [filt10ms]", "N-PID(scaled) PSO-noisy [filt10ms]"]
    step = d[d.scenario == "step"].set_index("variant").loc[keep, "ITAE"]
    noise = d[d.scenario == "noise"].set_index("variant").loc[keep, "ITAE"]
    y = np.arange(len(keep))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(y - 0.2, step.values, 0.4, label="clean step", color="#2a6f97")
    ax.barh(y + 0.2, noise.values, 0.4, label="noisy feedback (10 seeds)", color="#e07a1f")
    ax.set_yticks(y, keep)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("ITAE (log scale, lower is better)")
    ax.set_title("DC motor: L-PID vs N-PID under equal tuning treatment")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_fair_comparison.png", dpi=150)
    plt.close(fig)


def fig2():
    d = pd.read_csv(f"{R}/05_multi_plant.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, scen in zip(axes, ("step", "load", "noise")):
        t = d[d.scenario == scen].pivot(index="variant", columns="plant", values="ITAE")
        rel = t / t.min()
        im = ax.imshow(np.log10(rel.values), cmap="viridis_r", vmin=0, vmax=3, aspect="auto")
        ax.set_xticks(range(rel.shape[1]), rel.columns, rotation=30)
        ax.set_yticks(range(rel.shape[0]), rel.index)
        ax.set_title(f"{scen}: ITAE / best on that plant")
        for i in range(rel.shape[0]):
            for j in range(rel.shape[1]):
                v = rel.values[i, j]
                ax.text(j, i, f"{v:.1f}" if v < 100 else ">100", ha="center", va="center", fontsize=7,
                        color="white" if np.log10(v) < 1.5 else "black")
    fig.colorbar(im, ax=axes, label="log10 ratio to best", shrink=0.8)
    fig.savefig(f"{OUT}/fig2_multi_plant.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig3():
    m = pd.read_csv(f"{R}/04_ml_comparison.csv")
    e = pd.read_csv(f"{R}/04_extrapolation.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(m))
    a.bar(x - 0.2, m.median_ratio, 0.4, label="median", color="#2a6f97")
    a.bar(x + 0.2, m.p90_ratio, 0.4, label="90th percentile", color="#e07a1f")
    a.axhline(1.0, color="k", lw=0.8)
    a.set_xticks(x, m.model, rotation=30, ha="right")
    a.set_ylim(0.95, 1.25)
    a.set_ylabel("ITAE(ML gains) / ITAE(PSO gains)")
    a.set_title("Held-out plants: regret of each predictor\n('constant' = one fixed gain set)")
    a.legend()
    colors = ["#2a6f97" if r else "#c1121f" for r in e.in_range]
    b.bar(range(len(e)), e.ratio, color=colors)
    b.axhline(1.0, color="k", lw=0.8)
    b.set_xticks(range(len(e)), e.case, rotation=60, ha="right")
    b.set_ylabel("ITAE(ML) / ITAE(fresh PSO)")
    b.set_title("Extrapolation (red = outside training range)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_ml_regret.png", dpi=150)
    plt.close(fig)


def fig4():
    d = pd.read_csv(f"{R}/06_drift_baselines.csv")
    d = d[d.k_heat <= 0.10]
    t = d.pivot(index="controller", columns="k_heat", values="ITAE_total")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t.plot.bar(ax=ax, logy=True, rot=20)
    ax.set_ylabel("total ITAE over 20 s (log)")
    ax.set_title("Resistance drift: frozen gains vs scheduling vs RL")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_drift.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4):
        f()
        print("wrote", f.__name__)
