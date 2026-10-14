"""
M5a - GA-enhanced ML (method 16).

A GA (the M3 implementation on the unit cube) selects the model family, hyper-parameters and feature subset of the
supervised tuner; the fitness of a configuration is its 3-fold cross-validated *closed-loop* regret (geometric mean of
ITAE/ITAE_label, clipped) on a fixed subsample of the training plants, every prediction being simulated.
Compared, at the same number of fitness evaluations, with random search over the same genome space, and with the
M4 defaults (SVR with inner grid-search, extra-trees, MLP). Selected models are retrained on the whole training split
and evaluated on val / test / ood (primary protocol: noisy bump-test features).

Run from the repo root:  python experiments/m5a_ga_ml.py run [n_evals] | analyze
"""

import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.m5_common import FIG, OUT, WORKERS, load_main, regret, summarize
from src.tuners.hybrid import GENOME_DIM, build_model, cv_regret_fitness, decode_genome
from src.tuners.ml import make_model
from src.tuners.optimizers import ga, random_search

warnings.filterwarnings("ignore")
N_FIT_PLANTS = 420            # training plants used by the fitness (fixed subsample)
SEEDS = (0, 1, 2)


def run_search(name, search, seed, n_evals, b_fit, X, Y, pool):
    cache, hist, records = {}, [], []
    t0 = time.perf_counter()

    def fitness(u):
        cfg = decode_genome(u)
        key = cfg.describe()
        if key not in cache:
            cache[key] = cv_regret_fitness(cfg, X, Y, lambda idx, Z: regret(b_fit.select(np.isin(np.arange(len(b_fit)), idx)), Z, pool),
                                           k=3, seed=seed)
        f = cache[key]
        records.append((len(records) + 1, f))
        hist.append((len(records), min(r[1] for r in records)))
        if len(records) % 10 == 0:
            print(f"  {name} seed {seed}: {len(records)}/{n_evals}  best fitness {hist[-1][1]:.4f}  ({time.perf_counter() - t0:.0f} s)", flush=True)
        return f

    u, f = search(fitness, GENOME_DIM, n_evals, np.random.default_rng(seed))
    return u, f, hist, time.perf_counter() - t0


def stage_run(n_evals=160):
    os.makedirs(FIG, exist_ok=True)
    b = load_main()
    tr = b.select(b.split == "train")
    rng = np.random.default_rng(0)
    fit_idx = np.sort(rng.permutation(len(tr))[:N_FIT_PLANTS])
    b_fit = tr.select(np.isin(np.arange(len(tr)), fit_idx))
    X, Y = b_fit.X("engineered"), b_fit.targets("best")
    evalset = {s: b.select(b.split == s) for s in ("val", "test", "ood")}
    rows, curves, configs = [], [], []

    with Pool(WORKERS) as pool:
        def evaluate(tag, seed, model, extra=None):
            m = model.fit(tr.X("engineered"), tr.targets("best")) if tag.startswith("default") else model
            for split, ev in evalset.items():
                Z = m.predict(ev.X("engineered"))
                rows.append(dict(method=tag, seed=seed, split=split, **summarize(regret(ev, Z, pool)), **(extra or {})))
            r = {x["split"]: x for x in rows[-3:]}
            print(f"{tag} seed {seed}: val {r['val']['median']:.3f}/{r['val']['p90']:.2f}  test {r['test']['median']:.3f}/{r['test']['p90']:.2f}"
                  f"  ood {r['ood']['median']:.2f}", flush=True)

        for name in ("svr", "extra-trees", "mlp"):                                   # M4 defaults
            evaluate(f"default {name}", 0, make_model(name))

        for method, search in (("GA", lambda f, d, n, r: ga(f, d, n, r, n=16)),
                               ("Random search", lambda f, d, n, r: random_search(f, d, n, r))):
            for seed in SEEDS:
                u, f, hist, secs = run_search(method, search, seed, n_evals, b_fit, X, Y, pool)
                cfg = decode_genome(u)
                configs.append(dict(method=method, seed=seed, cv_fitness=f, seconds=secs, model=cfg.model,
                                    params=json.dumps(cfg.params), mask=json.dumps(list(cfg.mask)), description=cfg.describe()))
                for n, v in hist:
                    curves.append(dict(method=method, seed=seed, evals=n, best_fitness=v))
                m = build_model(cfg, seed).fit(tr.X("engineered"), tr.targets("best"))
                evaluate(f"{method}-selected", seed, m, dict(cv_fitness=f, model=cfg.model, n_features=len(cfg.mask)))
                pd.DataFrame(rows).to_csv(f"{OUT}/M5a_results.csv", index=False)
                pd.DataFrame(curves).to_csv(f"{OUT}/M5a_curves.csv", index=False)
                pd.DataFrame(configs).to_csv(f"{OUT}/M5a_configs.csv", index=False)
    print("done", flush=True)


def stage_analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = pd.read_csv(f"{OUT}/M5a_results.csv")
    cur = pd.read_csv(f"{OUT}/M5a_curves.csv")
    cfg = pd.read_csv(f"{OUT}/M5a_configs.csv")
    summ = (res.groupby(["method", "split"]).agg(median_of_median=("median", "median"), min_median=("median", "min"),
                                                 max_median=("median", "max"), p90=("p90", "median"), geo_mean=("geo_mean", "median"),
                                                 within1p5=("within1p5", "median")).reset_index())
    summ.to_csv(f"{OUT}/M5a_summary.csv", index=False)
    print(summ.pivot(index="method", columns="split", values="median_of_median").round(3))
    print(summ.pivot(index="method", columns="split", values="p90").round(2))
    print(cfg[["method", "seed", "cv_fitness", "model", "n_features", "description"]] if "n_features" in cfg else cfg[["method", "seed", "cv_fitness", "description"]])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for method, col in (("GA", "C0"), ("Random search", "C1")):
        g = cur[cur.method == method].pivot(index="evals", columns="seed", values="best_fitness")
        ax[0].plot(g.index, g.median(axis=1), color=col, label=method)
        ax[0].fill_between(g.index, g.min(axis=1), g.max(axis=1), color=col, alpha=0.2)
    ax[0].set_xlabel("fitness evaluations (3-fold CV closed-loop regret)"); ax[0].set_ylabel("best CV fitness (geometric regret)")
    ax[0].set_title("Search progress (median and range of 3 seeds)"); ax[0].legend()
    methods = ["default svr", "default extra-trees", "default mlp", "Random search-selected", "GA-selected"]
    w = 0.25
    for j, split in enumerate(("val", "test", "ood")):
        vals = [summ[(summ.method == m) & (summ.split == split)].median_of_median.iloc[0] for m in methods]
        ax[1].bar(np.arange(len(methods)) + (j - 1) * w, vals, w, label=split)
    ax[1].set_xticks(range(len(methods))); ax[1].set_xticklabels([m.replace("-selected", "\nselected").replace("default ", "default\n") for m in methods], fontsize=8)
    ax[1].set_yscale("log"); ax[1].set_ylabel("median regret"); ax[1].set_title("Held-out regret (noisy protocol)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/M5a_ga_ml.png", dpi=130)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "run"
    if stage == "run":
        stage_run(int(sys.argv[2]) if len(sys.argv) > 2 else 160)
    else:
        stage_analyze()
