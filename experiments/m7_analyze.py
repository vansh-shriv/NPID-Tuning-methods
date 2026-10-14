"""M7 analysis: tables and figures of the cross-method benchmark (called by `m7_benchmark.py analyze`)."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tuners.benchmark import annotate_failures

OUT = "results/methods/M7"
FIG = f"{OUT}/figures"
PLANTS = ["dc_speed", "dc_position", "link", "fopdt", "tank"]
SCEN = ["step", "load", "noise", "change"]
FAMILY_ORDER = ["classical", "intelligent", "optimization", "ml", "hybrid", "adaptive", "reference"]
FAM_COLOR = {"classical": "#4c78a8", "intelligent": "#f58518", "optimization": "#54a24b", "ml": "#e45756", "hybrid": "#b279a2",
             "adaptive": "#9d755d", "reference": "#7f7f7f"}
CAP = 100.0                                  # failures count as this ratio in aggregates


def load():
    df = pd.read_csv(f"{OUT}/M7_benchmark.csv")
    if "diverged" not in df:
        df["diverged"] = False
    df["id"] = df["id"].astype(str).str.zfill(2) if df["id"].dtype != object else df["id"].astype(str)
    df = annotate_failures(df)
    df["label"] = df["id"] + " " + df["method"]
    return df


def aggregate(df):
    g = df.groupby(["id", "label", "family", "plant", "kind", "scenario"])
    a = g.agg(itae=("ITAE", "median"), q25=("ITAE", lambda x: x.quantile(0.25)), q75=("ITAE", lambda x: x.quantile(0.75)),
              n=("ITAE", "size"), failed=("failed", "mean"), evals=("evals", "median"), sim_seconds=("sim_seconds", "median"),
              wall=("wall_time", "median"), overshoot=("overshoot_pct", "median") if "overshoot_pct" in df else ("ITAE", "size")).reset_index()
    return a


def ratios(a, kind="npid"):
    """ITAE of each method / best ITAE on the same (plant, scenario); failures -> CAP."""
    r = a[a.kind == kind].copy()
    best = r[(r.failed < 0.5) & np.isfinite(r.itae)].groupby(["plant", "scenario"]).itae.min()
    r["ratio"] = [CAP if (f >= 0.5 or not np.isfinite(i)) else min(i / best.get((p, s), np.nan), CAP) for i, f, p, s in zip(r.itae, r.failed, r.plant, r.scenario)]
    return r


def significant_rank(r):
    """Rank per (plant, scenario): 1 + number of methods that are better by more than the seed spread (>= 5% floor)."""
    rows = []
    for (p, s), g in r.groupby(["plant", "scenario"]):
        g = g[np.isfinite(g.itae) & (g.failed < 0.5)]
        spread = np.maximum((g.q75 - g.q25).values, 0.05 * g.itae.values)
        med = g.itae.values
        for i, (lab, fam) in enumerate(zip(g.label, g.family)):
            better = sum(1 for k in range(len(g)) if k != i and med[i] - med[k] > max(spread[i], spread[k]))
            rows.append(dict(plant=p, scenario=s, label=lab, family=fam, rank=1 + better))
    return pd.DataFrame(rows)


def geomean(x):
    x = np.clip(np.asarray(x, dtype=float), 1e-9, None)
    return float(np.exp(np.mean(np.log(x))))


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    df = load()
    a = aggregate(df)
    a.to_csv(f"{OUT}/M7_aggregate.csv", index=False)
    r = ratios(a)
    r.to_csv(f"{OUT}/M7_ratios.csv", index=False)
    order = [l for f in FAMILY_ORDER for l in sorted(r[r.family == f].label.unique())]
    fam_of = dict(zip(r.label, r.family))

    # best-known gap (step, N-PID)
    bk = pd.read_csv("results/methods/M3/best_known.csv")
    bk = bk[(bk.kind == "npid") & (~bk.tune_shape)].set_index("plant").cost
    gap = a[(a.kind == "npid") & (a.scenario == "step")].copy()
    gap["gap_to_best_known"] = [CAP if (f >= 0.5 or not np.isfinite(i)) else min(i / bk[p], CAP) for i, f, p in zip(gap.itae, gap.failed, gap.plant)]
    gap.pivot(index="label", columns="plant", values="gap_to_best_known").reindex(order)[PLANTS].round(2).to_csv(f"{OUT}/M7_gap_to_best_known.csv")

    # ---- tables: ratio per scenario, rank, summary per method
    for s in SCEN:
        t = r[r.scenario == s].pivot(index="label", columns="plant", values="ratio").reindex(order)
        t[[c for c in PLANTS if c in t]].round(2).to_csv(f"{OUT}/M7_ratio_{s}.csv")
    rk = significant_rank(r)
    rk.to_csv(f"{OUT}/M7_ranks.csv", index=False)
    summ = []
    for lab in order:
        row = dict(label=lab, family=fam_of[lab])
        for s in SCEN:
            x = r[(r.label == lab) & (r.scenario == s)]
            row[f"geomean_ratio_{s}"] = geomean(x.ratio) if len(x) else np.nan
            rr = rk[(rk.label == lab) & (rk.scenario == s)]
            row[f"mean_rank_{s}"] = rr["rank"].mean() if len(rr) else np.nan
        g = gap[gap.label == lab]
        row["geomean_gap_best_known_step"] = geomean(g.gap_to_best_known) if len(g) else np.nan
        row["median_evals"] = float(a[(a.label == lab) & (a.kind == "npid") & (a.scenario == "step")].evals.median())
        row["median_sim_seconds"] = float(a[(a.label == lab) & (a.kind == "npid") & (a.scenario == "step")].sim_seconds.median())
        ok = a[(a.label == lab) & (a.kind == "npid") & (a.scenario == "step")]
        row["plants_ok"] = int((ok.failed < 0.5).sum()) if len(ok) else np.nan
        summ.append(row)
    summ = pd.DataFrame(summ)
    summ.to_csv(f"{OUT}/M7_summary_per_method.csv", index=False)
    fam = summ.groupby("family").agg({c: "median" for c in summ.columns if c.startswith(("geomean", "mean_rank", "median_"))}).reindex(FAMILY_ORDER)
    fam.round(2).to_csv(f"{OUT}/M7_summary_per_family.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(summ.round(2).to_string(index=False)); print(fam.round(2).to_string())

    # ---- applicability
    app = []
    for lab in order:
        row = dict(label=lab)
        for p in PLANTS:
            x = df[(df.label == lab) & (df.plant == p) & (df.scenario == "step") & (df.kind == "npid")]
            if len(x) == 0:
                row[p] = "-"
            elif not x.success.astype(bool).all():
                row[p] = "not applicable"
            elif x.failed.mean() >= 0.5:
                row[p] = "fails"
            else:
                row[p] = "ok" if x.failed.mean() == 0 else "partly fails"
        app.append(row)
    pd.DataFrame(app).to_csv(f"{OUT}/M7_applicability.csv", index=False)
    print(pd.DataFrame(app).to_string(index=False))

    # ---- RQ5: same gains on N-PID vs L-PID
    both = a[a.scenario.isin(["step", "load", "noise"])].pivot_table(index=["label", "plant", "scenario"], columns="kind", values="itae").dropna()
    both["ratio"] = both["npid"] / both["lpid"]
    both["verdict"] = np.where(both.ratio < 0.9, "N-PID better", np.where(both.ratio > 1.1, "L-PID better", "similar"))
    rq5 = both.groupby(["scenario", "verdict"]).size().unstack(fill_value=0)
    rq5.to_csv(f"{OUT}/M7_rq5_npid_vs_lpid.csv"); print(rq5.to_string())

    # ---- figure 1: heat maps
    fig, axes = plt.subplots(1, 4, figsize=(21, 9), sharey=True)
    for ax, s in zip(axes, SCEN):
        t = r[r.scenario == s].pivot(index="label", columns="plant", values="ratio").reindex(order)[PLANTS]
        im = ax.imshow(np.log10(t.values.astype(float)), cmap="RdYlGn_r", vmin=0, vmax=2, aspect="auto")
        ax.set_xticks(range(len(PLANTS))); ax.set_xticklabels(PLANTS, rotation=30, fontsize=8)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                v = t.values[i, j]
                ax.text(j, i, "F" if v >= CAP else (f"{v:.1f}" if v < 10 else f"{v:.0f}"), ha="center", va="center", fontsize=7)
        ax.set_title(f"{s}", fontsize=11)
    for tl in axes[0].get_yticklabels():
        tl.set_color(FAM_COLOR[fam_of[tl.get_text()]])
    fig.suptitle("ITAE of each method / best method on the same plant and scenario (F = failed or not applicable, blank = not run)", fontsize=11); fig.colorbar(im, ax=axes, shrink=0.5, label="log10 ratio")
    fig.savefig(f"{FIG}/M7_heatmap.png", dpi=120, bbox_inches="tight")

    # ---- figure 2: cost vs performance (Pareto)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for k, (col, xl) in enumerate((("median_evals", "closed-/open-loop simulations used (median over plants)"),
                                   ("median_sim_seconds", "simulated plant time used (s, median over plants)"))):
        pts = summ[np.isfinite(summ[col]) & np.isfinite(summ.geomean_gap_best_known_step)]
        for _, p_ in pts.iterrows():
            ax[k].scatter(max(p_[col], 0.5), p_.geomean_gap_best_known_step, color=FAM_COLOR[p_.family], s=45)
            ax[k].annotate(p_.label[:2], (max(p_[col], 0.5), p_.geomean_gap_best_known_step), fontsize=7, xytext=(3, 3), textcoords="offset points")
        xs = pts.sort_values(col)
        front = xs[xs.geomean_gap_best_known_step <= xs.geomean_gap_best_known_step.cummin() + 1e-12]
        ax[k].step([max(v, 0.5) for v in front[col]], front.geomean_gap_best_known_step, where="post", color="k", lw=0.8, ls="--")
        ax[k].set_xscale("log"); ax[k].set_yscale("log"); ax[k].set_xlabel(xl); ax[k].set_ylabel("geometric-mean gap to best known (step ITAE)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in FAM_COLOR.items()]
    ax[1].legend(handles=handles, fontsize=8)
    ax[0].set_title("Tuning cost vs performance (labels = method id; dashed = Pareto front)")
    fig.tight_layout(); fig.savefig(f"{FIG}/M7_pareto.png", dpi=130)

    # ---- figure 3: robustness (clean vs noise)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for _, p_ in summ.iterrows():
        ax.scatter(p_.geomean_ratio_step, p_.geomean_ratio_noise, color=FAM_COLOR[p_.family], s=45)
        ax.annotate(p_.label[:2], (p_.geomean_ratio_step, p_.geomean_ratio_noise), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.plot([1, CAP], [1, CAP], "k:", lw=0.6); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("clean step: geometric-mean ITAE ratio to best"); ax.set_ylabel("noisy feedback: geometric-mean ITAE ratio to best")
    ax.legend(handles=handles, fontsize=8); ax.set_title("Robustness: clean vs noise")
    fig.tight_layout(); fig.savefig(f"{FIG}/M7_robustness.png", dpi=130)

    # ---- figure 4: per-family summary
    fig, ax = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    for k, s in enumerate(SCEN):
        vals = [summ[summ.family == f][f"geomean_ratio_{s}"].values for f in FAMILY_ORDER]
        ax[k].boxplot(vals, tick_labels=[f[:6] for f in FAMILY_ORDER], showfliers=True)
        ax[k].set_yscale("log"); ax[k].set_title(f"{s}: geometric-mean ratio per method"); ax[k].tick_params(axis="x", rotation=40, labelsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/M7_families.png", dpi=130)

    # ---- figure 5: change scenario, first post-change window vs adapted windows
    ch = df[(df.scenario == "change") & (df.kind == "npid") & df.itae_first.notna()] if "itae_first" in df else pd.DataFrame()
    if len(ch):
        t = ch.groupby(["label", "plant"]).agg(first=("itae_first", "median"), post=("ITAE", "median")).reset_index()
        t.to_csv(f"{OUT}/M7_change_first_vs_adapted.csv", index=False)


if __name__ == "__main__":
    analyze()
