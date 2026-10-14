"""
M3 - Optimization-based tuning: PSO, GA, ABC, SA (+ random search baseline).

Stages
  run      main benchmark (5 optimizers x {N-PID, L-PID, N-PID+shape} x 5 plants x 5 seeds, 400 simulations),
           budget sweep (100/200/400/800), hyper-parameter sensitivity, noise-aware training
  analyze  gap to the best-known optimum (needs results/methods/M3/best_known.csv from m3a_best_known.py),
           statistics, tables, figures

Run from the repo root:   python experiments/m3_optimizers.py run   |   python experiments/m3_optimizers.py analyze
"""

import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark_plants import PLANT_NAMES
from src.tuners.benchmark import run_benchmark
from src.tuners.optimizers import ABC, GA, PSO, SA, RandomSearch

warnings.filterwarnings("ignore")
OUT = "results/methods/M3"
FIG = f"{OUT}/figures"
SEEDS = tuple(range(5))
ALGOS = [("PSO", PSO), ("GA", GA), ("ABC", ABC), ("SA", SA), ("Random search", RandomSearch)]
WORKERS = 12


# ----------------------------------------------------------------------------- run
def stage_main():
    tuners = []
    for _, cls in ALGOS:
        tuners += [cls(), cls(kind="lpid"), cls(tune_shape=True)]
    df = run_benchmark(tuners, PLANT_NAMES, seeds=SEEDS, budget=400, workers=WORKERS)
    df.to_csv(f"{OUT}/M3_main.csv", index=False)


def stage_budget():
    rows = []
    fast = ["dc_speed", "dc_position", "link"]
    for b in (100, 200, 400, 800):
        df = run_benchmark([cls() for _, cls in ALGOS], fast, seeds=SEEDS, budget=b, kinds=("npid",), workers=WORKERS)
        df["budget"] = b
        rows.append(df)
    pd.concat(rows).to_csv(f"{OUT}/M3_budget.csv", index=False)


HYPER = {                                   # one-at-a-time sweeps, 3 values each
    "PSO": [("n", [10, 20, 40]), ("c1", [1.0, 2.0, 3.0])],
    "GA": [("pm", [0.05, 0.2, 0.5]), ("n", [10, 20, 40])],
    "ABC": [("n_sources", [5, 10, 20]), ("limit", [2, 8, 40])],
    "SA": [("sigma0", [0.1, 0.25, 0.5]), ("t_final_ratio", [0.001, 0.01, 0.1])],
}


def stage_hyper():
    rows = []
    for name, cls in ALGOS[:4]:
        for param, values in HYPER[name]:
            for v in values:
                tuner = cls(name=f"{name} {param}={v}", **{param: v})
                df = run_benchmark([tuner], ["dc_speed", "link"], seeds=SEEDS, budget=400, kinds=("npid",), workers=WORKERS)
                df["algo"], df["param"], df["value"] = name, param, v
                rows.append(df)
    pd.concat(rows).to_csv(f"{OUT}/M3_hyper.csv", index=False)


def stage_noise_aware():
    tuners = [PSO(), PSO(scenario="noise", name="PSO noise-aware"), GA(), GA(scenario="noise", name="GA noise-aware")]
    df = run_benchmark(tuners, PLANT_NAMES, seeds=SEEDS, budget=400, kinds=("npid",), workers=WORKERS)
    df.to_csv(f"{OUT}/M3_noise_aware.csv", index=False)


def stage_noise_aware_lpid():
    """Fair noise comparison: the L-PID also gets noise-aware tuning (same optimizers, same budget)."""
    tuners = [PSO(kind="lpid", scenario="noise", name="PSO noise-aware [L-PID]"),
              GA(kind="lpid", scenario="noise", name="GA noise-aware [L-PID]"),
              PSO(kind="lpid", name="PSO [L-PID]"), GA(kind="lpid", name="GA [L-PID]")]
    df = run_benchmark(tuners, PLANT_NAMES, seeds=SEEDS, budget=400, kinds=("lpid",), workers=WORKERS)
    df.to_csv(f"{OUT}/M3_noise_aware_lpid.csv", index=False)


# ----------------------------------------------------------------------------- analyze
def own_kind(method):
    return "lpid" if "[L-PID]" in method else "npid"


def base_name(method):
    return method.replace(" [L-PID]", "").replace(" +shape", "").replace(" noise-aware", "")


def variant(method):
    return "L-PID" if "[L-PID]" in method else ("N-PID +shape" if "+shape" in method else "N-PID")


def stage_analyze():
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu

    from src.tuners.plots import _save, plot_heatmap

    os.makedirs(FIG, exist_ok=True)
    main = pd.read_csv(f"{OUT}/M3_main.csv")
    bk = pd.read_csv(f"{OUT}/best_known.csv")
    bk["variant"] = np.where(bk.kind == "lpid", "L-PID", np.where(bk.tune_shape, "N-PID +shape", "N-PID"))
    best = bk.set_index(["plant", "variant"]).cost

    d = main[main.success & (main.scenario == "step")].copy()
    d["algo"], d["variant"] = d.method.map(base_name), d.method.map(variant)
    d["own"] = d.method.map(own_kind)
    d = d[d.kind == d.own]                                            # score each tuner with its own controller
    d["best_known"] = [best.get((p, v), np.nan) for p, v in zip(d.plant, d.variant)]
    d["gap"] = d.ITAE / d.best_known
    d.to_csv(f"{OUT}/M3_step_with_gap.csv", index=False)

    # ---- table 1: median gap per algorithm x plant (N-PID)
    n = d[d.variant == "N-PID"]
    gap = n.groupby(["algo", "plant"]).gap.median().unstack().reindex(columns=PLANT_NAMES).reindex([a for a, _ in ALGOS])
    gap.to_csv(f"{OUT}/M3_gap_median.csv")
    plot_heatmap(gap, f"{FIG}/M3_heatmap_gap.png", title="M3: median ITAE / best-known (N-PID, 400 simulations, 5 seeds)", fmt="{:.3g}")
    spread = n.groupby(["algo", "plant"]).gap.agg(["min", "median", "max"]).unstack("plant")
    spread.to_csv(f"{OUT}/M3_gap_spread.csv")

    # ---- success rates
    succ = n.assign(w5=n.gap <= 1.05, w20=n.gap <= 1.2).groupby(["algo", "plant"])[["w5", "w20"]].mean().unstack("plant")
    succ.to_csv(f"{OUT}/M3_success_rate.csv")

    # ---- statistics: pairwise Mann-Whitney per plant, count of significant wins
    algos = [a for a, _ in ALGOS]
    rows = []
    for plant in PLANT_NAMES:
        for a, b in itertools.combinations(algos, 2):
            x, y = n[(n.plant == plant) & (n.algo == a)].gap, n[(n.plant == plant) & (n.algo == b)].gap
            if len(x) and len(y):
                p = mannwhitneyu(x, y, alternative="two-sided").pvalue
                rows.append(dict(plant=plant, a=a, b=b, median_a=x.median(), median_b=y.median(), p=p,
                                 winner=(a if x.median() < y.median() else b) if p < 0.05 else "tie"))
    pd.DataFrame(rows).to_csv(f"{OUT}/M3_pairwise.csv", index=False)

    # ---- figure: gap boxplots
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6), sharey=False)
    for ax, plant in zip(axes, PLANT_NAMES):
        data = [n[(n.plant == plant) & (n.algo == a)].gap.values for a in algos]
        ax.boxplot(data, tick_labels=algos)
        ax.set_yscale("log")
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_title(plant)
        ax.tick_params(axis="x", rotation=40)
        ax.grid(alpha=0.3, which="both", axis="y")
    axes[0].set_ylabel("ITAE / best-known")
    _save(fig, f"{FIG}/M3_gap_boxplots.png")

    # ---- convergence curves
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    colors = dict(zip(algos, ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#7f7f7f"]))
    grid = np.arange(10, 401, 10)
    # histories are not stored in the CSV; re-run one seed per optimizer is done in the 'curves' stage
    curves = pd.read_csv(f"{OUT}/M3_curves.csv")
    for ax, plant in zip(axes, PLANT_NAMES):
        for a in algos:
            c = curves[(curves.plant == plant) & (curves.algo == a)].pivot(index="evals", columns="seed", values="best")
            c = c.reindex(grid, method="ffill")
            bkv = best.get((plant, "N-PID"), np.nan)
            med = (c / bkv).median(axis=1)
            ax.plot(grid, med, color=colors[a], label=a)
            ax.fill_between(grid, (c / bkv).quantile(0.25, axis=1), (c / bkv).quantile(0.75, axis=1), color=colors[a], alpha=0.15)
        ax.set_yscale("log")
        ax.set_title(plant)
        ax.set_xlabel("simulations")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("best-so-far ITAE / best-known")
    axes[0].legend(fontsize=8)
    _save(fig, f"{FIG}/M3_convergence.png")

    # ---- budget sweep
    bud = pd.read_csv(f"{OUT}/M3_budget.csv")
    bud = bud[bud.success & (bud.scenario == "step") & (bud.kind == "npid")].copy()
    bud["gap"] = [r.ITAE / best.get((r.plant, "N-PID"), np.nan) for r in bud.itertuples()]
    bud.to_csv(f"{OUT}/M3_budget_gap.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, plant in zip(axes, ["dc_speed", "dc_position", "link"]):
        for a in algos:
            g = bud[(bud.plant == plant) & (bud.method == a)].groupby("budget").gap
            ax.errorbar(g.median().index, g.median(), yerr=[g.median() - g.min(), g.max() - g.median()], marker="o",
                        capsize=3, color=colors[a], label=a)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(plant)
        ax.set_xlabel("budget (simulations)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("ITAE / best-known (median, min-max)")
    axes[0].legend(fontsize=8)
    _save(fig, f"{FIG}/M3_budget_sweep.png")

    # ---- hyper-parameter sensitivity
    hy = pd.read_csv(f"{OUT}/M3_hyper.csv")
    hy = hy[hy.success & (hy.scenario == "step") & (hy.kind == "npid")].copy()
    hy["gap"] = [r.ITAE / best.get((r.plant, "N-PID"), np.nan) for r in hy.itertuples()]
    hy.to_csv(f"{OUT}/M3_hyper_gap.csv", index=False)
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for col, (name, params) in enumerate(HYPER.items()):
        for row, (param, values) in enumerate(params):
            ax = axes[row, col]
            for plant, mk in (("dc_speed", "o"), ("link", "s")):
                g = hy[(hy.algo == name) & (hy.param == param) & (hy.plant == plant)].groupby("value").gap
                ax.errorbar(range(len(values)), [g.median()[v] for v in values], marker=mk, capsize=3, label=plant,
                            yerr=[[g.median()[v] - g.min()[v] for v in values], [g.max()[v] - g.median()[v] for v in values]])
            ax.set_xticks(range(len(values)), [str(v) for v in values])
            ax.set_yscale("log")
            ax.set_title(f"{name}: {param}")
            ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=8)
    _save(fig, f"{FIG}/M3_hyper_sensitivity.png")

    # ---- N-PID vs L-PID vs N-PID+shape, each tuned by the same optimizer (RQ5)
    full = main[main.success].copy()
    full["algo"], full["variant"], full["own"] = full.method.map(base_name), full.method.map(variant), full.method.map(own_kind)
    full = full[(full.kind == full.own)]
    rq5 = full.groupby(["scenario", "plant", "variant"]).ITAE.median().unstack("variant")
    rq5.to_csv(f"{OUT}/M3_rq5_variants.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, scen in zip(axes, ("step", "load", "noise")):
        t = rq5.xs(scen, level="scenario").reindex(PLANT_NAMES)
        t.plot.bar(ax=ax, logy=True, rot=20)
        ax.set_title(f"{scen} scenario")
        ax.grid(alpha=0.3, axis="y", which="both")
    axes[0].set_ylabel("ITAE (median over 4 optimizers x 5 seeds)")
    fig.suptitle("Each controller variant tuned by the same optimizers with the same budget (own controller evaluated)")
    _save(fig, f"{FIG}/M3_rq5_variants.png")
    shape = full[(full.variant == "N-PID +shape") & (full.scenario == "step")]
    shape[["plant", "algo", "seed", "gain_n", "gain_m", "gain_p", "ITAE"]].to_csv(f"{OUT}/M3_shape_exponents.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, k in zip(axes, ("gain_n", "gain_m", "gain_p")):
        ax.boxplot([shape[shape.plant == pl][k].dropna().values for pl in PLANT_NAMES], tick_labels=PLANT_NAMES)
        ax.set_title(f"tuned shape exponent {k[-1]}")
        ax.axhline(2.0, color="k", ls="--", lw=0.8)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3)
    _save(fig, f"{FIG}/M3_shape_exponents.png")

    # ---- noise-aware
    na = pd.read_csv(f"{OUT}/M3_noise_aware.csv")
    na = na[na.success]
    na["trained"] = np.where(na.method.str.contains("noise-aware"), "noise-aware", "clean")
    na["algo"] = na.method.map(base_name)
    t = na.pivot_table(index=["scenario", "algo", "trained"], columns="plant", values="ITAE", aggfunc="median")
    t.to_csv(f"{OUT}/M3_noise_aware_table.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, scen in zip(axes, ("step", "noise")):
        tt = na[na.scenario == scen].pivot_table(index="plant", columns=["algo", "trained"], values="ITAE", aggfunc="median").reindex(PLANT_NAMES)
        tt.plot.bar(ax=ax, logy=True, rot=20)
        ax.set_title(f"{scen} ITAE: clean-trained vs noise-aware")
        ax.grid(alpha=0.3, axis="y", which="both")
    _save(fig, f"{FIG}/M3_noise_aware.png")

    # ---- noise-aware N-PID vs noise-aware L-PID (RQ5, fair version)
    if os.path.exists(f"{OUT}/M3_noise_aware_lpid.csv"):
        nl = pd.read_csv(f"{OUT}/M3_noise_aware_lpid.csv")
        nl = nl[nl.success].assign(algo=lambda x: x.method.map(base_name))
        nl["trained"] = np.where(nl.method.str.contains("noise-aware"), "noise-aware", "clean")
        nl["variant"] = "L-PID"
        nn = na.assign(variant="N-PID")
        both = pd.concat([nn[["plant", "scenario", "algo", "trained", "variant", "ITAE", "seed"]],
                          nl[["plant", "scenario", "algo", "trained", "variant", "ITAE", "seed"]]])
        t2 = both.pivot_table(index=["scenario", "trained", "variant"], columns="plant", values="ITAE", aggfunc="median")
        t2.to_csv(f"{OUT}/M3_rq5_noise_aware_fair.csv")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, scen in zip(axes, ("step", "noise")):
            tt = both[both.scenario == scen].pivot_table(index="plant", columns=["trained", "variant"], values="ITAE",
                                                         aggfunc="median").reindex(PLANT_NAMES)
            tt.plot.bar(ax=ax, logy=True, rot=20)
            ax.set_title(f"{scen} ITAE (median over PSO, GA, seeds)")
            ax.grid(alpha=0.3, axis="y", which="both")
        _save(fig, f"{FIG}/M3_rq5_noise_aware_fair.png")

    # ---- identifiability: gains scatter per plant
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
    for ax, plant in zip(axes, PLANT_NAMES):
        for a in algos[:4]:
            s = n[(n.plant == plant) & (n.algo == a)]
            ax.scatter(s.gain_Kp, s.gain_Ki, color=colors[a], s=28, alpha=0.8, label=a)
        b = bk[(bk.plant == plant) & (bk.variant == "N-PID")]
        ax.scatter(b.gain_Kp, b.gain_Ki, marker="*", color="k", s=120, label="best known")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(plant)
        ax.set_xlabel("Kp")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("Ki")
    axes[0].legend(fontsize=7)
    _save(fig, f"{FIG}/M3_gain_scatter.png")
    cv = n.groupby(["plant", "algo"]).agg(cost_cv=("ITAE", lambda x: x.std() / x.mean()),
                                          Kp_cv=("gain_Kp", lambda x: x.std() / x.mean()),
                                          Ki_cv=("gain_Ki", lambda x: x.std() / x.mean()),
                                          Kd_cv=("gain_Kd", lambda x: x.std() / x.mean()))
    cv.to_csv(f"{OUT}/M3_identifiability.csv")

    # ---- runtime / overhead
    rt = n.groupby("algo").wall_time.median()
    rt.to_csv(f"{OUT}/M3_wall_time.csv")
    print("analysis written to", OUT)


def stage_curves():
    """Histories for convergence figure: one run per (plant, optimizer, seed) with the history kept."""
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(p, a, s) for p in PLANT_NAMES for a, _ in ALGOS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        res = list(ex.map(_curve_job, jobs))
    pd.DataFrame([r for rs in res for r in rs]).to_csv(f"{OUT}/M3_curves.csv", index=False)


def _curve_job(args):
    from src.tuners import TuningProblem
    plant, algo, seed = args
    cls = dict(ALGOS)[algo]
    r = cls().tune(TuningProblem.from_scenario(plant), budget=400, seed=seed)
    return [dict(plant=plant, algo=algo, seed=seed, evals=e, best=b) for e, b in r.history]


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "run"
    if stage == "run":
        for fn in (stage_main, stage_budget, stage_hyper, stage_noise_aware, stage_curves):
            print("running", fn.__name__, flush=True)
            fn()
    elif stage in ("main", "budget", "hyper", "noise_aware", "noise_aware_lpid", "curves"):
        globals()[f"stage_{stage}"]()
    elif stage == "analyze":
        stage_analyze()
