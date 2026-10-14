"""
Shared figures for the method phases (docs/PROTOCOL.md section 10). All functions take plain data (DataFrames,
dicts) and save a PNG; they never run experiments themselves.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
           "#bcbd22", "#17becf", "#393b79", "#ad494a"]


def _color_map(names):
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_step_overlays(traces, path, title="", ncols=5):
    """
    traces: {plant: {"ref": float, "methods": {method: (time, output) or None}}}. One panel per plant; methods
    that failed / are not applicable (None) are listed in the legend as such and not drawn.
    """
    plants = list(traces)
    names = sorted({m for p in plants for m in traces[p]["methods"]})
    cmap = _color_map(names)
    nrows = int(np.ceil(len(plants) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows), squeeze=False)
    for ax, plant in zip(axes.ravel(), plants):
        ref = traces[plant]["ref"]
        ax.axhline(ref, color="k", lw=0.8, ls="--")
        for m, tr in traces[plant]["methods"].items():
            if tr is None:
                continue
            t, y = tr
            ax.plot(t, np.clip(y, -3 * abs(ref), 3 * abs(ref)), color=cmap[m], lw=1.1, label=m)
        ax.set_title(plant)
        ax.set_ylim(-0.15 * abs(ref), 1.8 * abs(ref))
        ax.grid(alpha=0.3)
        ax.set_xlabel("time (s)")
    for ax in axes.ravel()[len(plants):]:
        ax.axis("off")
    handles = [plt.Line2D([0], [0], color=cmap[m], lw=2) for m in names]
    fig.legend(handles, names, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)


def plot_heatmap(table, path, title="", fmt="{:.3g}", cmap="viridis_r", note=None):
    """
    table: DataFrame index=method, columns=plant, values = metric (NaN = not applicable). Colors are log-scaled
    and normalised per column (best = 1.0) so that plants with different scales are comparable; the cell text
    shows the raw value.
    """
    vals = table.values.astype(float)
    best = np.nanmin(np.where(vals > 0, vals, np.nan), axis=0)
    rel = np.log10(vals / best)
    fig, ax = plt.subplots(figsize=(1.6 * table.shape[1] + 3, 0.5 * table.shape[0] + 1.8))
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#dddddd")
    im = ax.imshow(np.ma.masked_invalid(rel), cmap=cm, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=20)
    ax.set_yticks(range(table.shape[0]), table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = vals[i, j]
            txt = "n/a" if np.isnan(v) else (">1e3" if v > 1e3 else fmt.format(v))
            c = "black" if (np.isnan(rel[i, j]) or rel[i, j] < 1.5) else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=c)
    fig.colorbar(im, ax=ax, label="log10(value / best on that plant)")
    ax.set_title(title + (f"\n{note}" if note else ""), fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def plot_cost_vs_performance(df, path, title="", perf="ITAE", cost="sim_seconds"):
    """df rows: method, plant, `cost`, `perf`. Marker per plant, color per method, log axes."""
    methods = sorted(df["method"].unique())
    plants = sorted(df["plant"].unique())
    cmap = _color_map(methods)
    markers = dict(zip(plants, "osD^v<>p*h"))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for _, r in df.iterrows():
        if not (np.isfinite(r[perf]) and r[cost] > 0):
            continue
        ax.scatter(r[cost], r[perf], color=cmap[r["method"]], marker=markers[r["plant"]], s=45, alpha=0.85,
                   edgecolor="k", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("tuning cost: simulated plant seconds (experiment time)")
    ax.set_ylabel(f"{perf} of the returned gains (step scenario)")
    ax.grid(alpha=0.3, which="both")
    h1 = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap[m], markeredgecolor="k", markersize=8) for m in methods]
    h2 = [plt.Line2D([0], [0], marker=markers[p], color="w", markerfacecolor="#999", markeredgecolor="k", markersize=8) for p in plants]
    fig.subplots_adjust(right=0.62)
    fig.legend(h1, methods, loc="upper left", bbox_to_anchor=(0.64, 0.95), fontsize=7.5, title="method", frameon=True)
    fig.legend(h2, plants, loc="lower left", bbox_to_anchor=(0.64, 0.08), fontsize=7.5, title="plant", frameon=True)
    ax.set_title(title)
    _save(fig, path)


def plot_identification(models, path, title=""):
    """models: {plant: ProcessModel} - measured unit-step response vs fitted model."""
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.1), squeeze=False)
    for ax, (plant, m) in zip(axes[0], models.items()):
        if m is None:
            ax.set_title(f"{plant}\n(not identifiable)")
            ax.axis("off")
            continue
        d = m.data
        ax.plot(d["t"], d["s"], color="#1f77b4", lw=1.5, label="measured")
        ax.plot(d["t"], d["s_fit"], color="#d62728", lw=1.2, ls="--", label=m.kind.upper())
        ax.set_title(f"{plant}: {m.kind}, nrmse {m.nrmse:.3f}", fontsize=9)
        ax.set_xlabel("time since step (s)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("unit-step response")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)
