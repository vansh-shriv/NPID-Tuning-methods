"""
M4c - Gaussian-process (Bayesian) optimisation: sample efficiency against the optimizers of M3.

A. The 5 benchmark plants: GPO with 60 and 400 simulations (5 seeds) against the M3 curves of GA/PSO/ABC/SA/random
   search (gap to the best-known optimum of M3a).
B. Plants of the M4 family (60 val+test plants): GPO, GA, PSO, random search with histories up to 120 simulations;
   regret vs the label, as a function of the budget.
C. GP diagnostics on the benchmark plants: fitted length scales (which gain matters) and leave-one-out error.

Run from the repo root:  python experiments/m4c_bayesopt.py run | analyze
"""

import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark_plants import PLANT_NAMES
from src.plant_family import problem_from_spec, spec_from_dict
from src.tuners import TuningProblem
from src.tuners.ml import BayesOptTuner, load_bundle
from src.tuners.optimizers import GA, PSO, RandomSearch

warnings.filterwarnings("ignore")
OUT = "results/methods/M4"
FIG = f"{OUT}/figures"
DATA = "data/m4/dataset.jsonl"
CHECK = (20, 40, 60, 90, 120, 200, 400)


def _history_at(history, budgets):
    """Best-so-far cost at the given simulation counts (last known value)."""
    h = np.array(history, dtype=float)
    out = {}
    for b in budgets:
        i = np.searchsorted(h[:, 0], b, side="right") - 1
        out[b] = h[i, 1] if i >= 0 else np.nan
    return out


def _bench_job(args):
    plant, seed, budget = args
    p = TuningProblem.from_scenario(plant)
    t = BayesOptTuner()
    r = t.tune(p, budget=budget, seed=seed)
    gp = t.last_gp
    k = gp.kernel_
    ls = k.k1.k2.length_scale if hasattr(k, "k1") else [np.nan] * 3
    # leave-one-out style check: refit-free predictive error on the last 10% of the sampled points
    n = len(t.last_Y)
    y = np.log10(np.minimum(t.last_Y, 1e3 * t.last_Y.min()))
    y_n = (y - y.mean()) / (y.std() + 1e-9)
    mu = gp.predict(t.last_U)
    rmse = float(np.sqrt(np.mean((mu - y_n) ** 2)))
    rows = [dict(plant=plant, seed=seed, budget=budget, evals=b, best=v) for b, v in _history_at(r.history, CHECK).items() if b <= budget]
    diag = dict(plant=plant, seed=seed, budget=budget, ls_Kp=float(ls[0]), ls_Ki=float(ls[1]), ls_Kd=float(ls[2]), train_rmse_norm=rmse)
    return rows, diag


def _family_job(args):
    spec_d, ident, label_cost, seed = args
    spec = spec_from_dict(spec_d)
    rows = []
    for name, tuner in (("GPO", BayesOptTuner()), ("GA", GA()), ("PSO", PSO()), ("Random search", RandomSearch())):
        r = tuner.tune(problem_from_spec(spec), budget=120, seed=seed)
        for b, v in _history_at(r.history, CHECK).items():
            if b <= 120:
                rows.append(dict(ident=ident, method=name, evals=b, best=v, label=label_cost, ratio=v / label_cost))
    return rows


def stage_run():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(p, s, 60) for p in PLANT_NAMES for s in range(5)] + [(p, s, 400) for p in PLANT_NAMES for s in range(5)]
    with ProcessPoolExecutor(max_workers=12) as ex:
        res = list(ex.map(_bench_job, jobs))
    pd.DataFrame([r for rows, _ in res for r in rows]).to_csv(f"{OUT}/M4_gpo_benchmark.csv", index=False)
    pd.DataFrame([d for _, d in res]).to_csv(f"{OUT}/M4_gpo_diagnostics.csv", index=False)
    print("benchmark plants done", flush=True)

    b = load_bundle(DATA)
    sel = np.where(np.isin(b.split, ["val", "test"]))[0][:60]
    jobs = [(b.specs[i], int(b.ident[i]), float(b.best_cost[i]), 7) for i in sel]
    with ProcessPoolExecutor(max_workers=12) as ex:
        rows = [r for rs in ex.map(_family_job, jobs) for r in rs]
    pd.DataFrame(rows).to_csv(f"{OUT}/M4_gpo_family.csv", index=False)
    print("family plants done", flush=True)


def stage_analyze():
    import matplotlib.pyplot as plt

    from src.tuners.plots import _save
    bench = pd.read_csv(f"{OUT}/M4_gpo_benchmark.csv")
    bk = pd.read_csv("results/methods/M3/best_known.csv")
    best = bk[(bk.kind == "npid") & (~bk.tune_shape)].set_index("plant").cost
    curves = pd.read_csv("results/methods/M3/M3_curves.csv")               # GA/PSO/ABC/SA/random histories (5 seeds, 400)
    algos = ["PSO", "GA", "ABC", "SA", "Random search"]
    grid = np.array(CHECK)

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
    rows = []
    for ax, plant in zip(axes, PLANT_NAMES):
        for a in algos:
            c = curves[(curves.plant == plant) & (curves.algo == a)].pivot(index="evals", columns="seed", values="best")
            c = c.reindex(np.arange(1, 401), method="ffill")
            med = [c.loc[b_].median() / best[plant] if b_ in c.index else np.nan for b_ in grid]
            ax.plot(grid, med, marker="o", ms=3, label=a, alpha=0.8)
            rows += [dict(plant=plant, method=a, evals=int(b_), gap=m) for b_, m in zip(grid, med)]
        g = bench[(bench.plant == plant) & (bench.budget == 400)].groupby("evals").best.median() / best[plant]
        ax.plot(g.index, g.values, "k-", marker="s", lw=2, label="GPO")
        rows += [dict(plant=plant, method="GPO", evals=int(e), gap=float(v)) for e, v in g.items()]
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(plant); ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("simulations")
    axes[0].set_ylabel("ITAE / best-known (median of 5 seeds)")
    axes[0].legend(fontsize=7)
    _save(fig, f"{FIG}/M4_gpo_sample_efficiency.png")
    pd.DataFrame(rows).to_csv(f"{OUT}/M4_gpo_vs_optimizers.csv", index=False)

    fam = pd.read_csv(f"{OUT}/M4_gpo_family.csv")
    t = fam.groupby(["method", "evals"]).ratio.median().unstack("evals")
    t.to_csv(f"{OUT}/M4_gpo_family_median_regret.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in t.index:
        ax.plot(t.columns, t.loc[m], marker="o", label=m)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
    ax.set_xlabel("simulations"); ax.set_ylabel("median regret vs label (60 plants)"); ax.legend()
    _save(fig, f"{FIG}/M4_gpo_family_regret.png")

    diag = pd.read_csv(f"{OUT}/M4_gpo_diagnostics.csv")
    diag.groupby(["plant", "budget"])[["ls_Kp", "ls_Ki", "ls_Kd", "train_rmse_norm"]].median().to_csv(f"{OUT}/M4_gpo_diagnostics_median.csv")
    print("analysis written")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    {"run": stage_run, "analyze": stage_analyze}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
