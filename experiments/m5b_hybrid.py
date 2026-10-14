"""
M5b - Hybrid ML + classical (method 17).

Components, switched on one by one (ablation), each run end to end as a deployable tuner (`ClassicalMLTuner`: noisy
bump test -> base -> optional ML correction -> optional 20-simulation local refinement), with its real tuning cost:
    constant start            median normalised label of the training plants (the M4 baseline start), no search
    Cohen-Coon                classical rule alone
    SVR                       M4 regressor alone (absolute z)
    CC + SVR correction       classical start + SVR trained on the residual label - Cohen-Coon
    + 20-sim refinement       each of the above followed by the local search (3 seeds, median)
Plants of the M4 family (val / test / ood), regret = ITAE / ITAE(label) on the clean step; plus the five benchmark
plants (full protocol: step / load / noise; gap to the best-known optimum of M3a).

Run from the repo root:  python experiments/m5b_hybrid.py run | bench | analyze
"""

import json
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.m5_common import FIG, OUT, WORKERS, load_main, summarize
from src.benchmark_plants import PLANT_NAMES
from src.plant_family import problem_from_spec, spec_from_dict
from src.tuners import TuningProblem
from src.tuners.benchmark import evaluate_gains
from src.tuners.hybrid import ClassicalMLTuner, HybridModelTuner, ModelConfig, base_matrix, build_model
from src.tuners.ml import make_model

warnings.filterwarnings("ignore")
REFINE = 20
SEEDS = (0, 1, 2)
VARIANTS = ["constant start", "Cohen-Coon", "SVR", "Cohen-Coon + SVR",
            "constant start + refine", "Cohen-Coon + refine", "SVR + refine", "Cohen-Coon + SVR + refine"]


def build_variants(tr):
    """Fit the models on the training plants and return {variant: (tuner, n_seeds)}."""
    X = tr.X("engineered")
    z_const = np.median(tr.best_z, axis=0)
    rule_const = "constant:" + ",".join(f"{v:.6f}" for v in z_const)
    svr_abs = make_model("svr").fit(X, tr.best_z)
    svr_res = make_model("svr").fit(X, tr.best_z - base_matrix(X, "cohen-coon"))
    out = {}
    for refine in (0, REFINE):
        sfx = " + refine" if refine else ""
        out["constant start" + sfx] = ClassicalMLTuner(None, rule_const, refine)
        out["Cohen-Coon" + sfx] = ClassicalMLTuner(None, "cohen-coon", refine)
        out["SVR" + sfx] = ClassicalMLTuner(svr_abs, "constant:0,0,0", refine)
        out["Cohen-Coon + SVR" + sfx] = ClassicalMLTuner(svr_res, "cohen-coon", refine)
    return out


_TUNERS = {}


def _init(tuners):
    _TUNERS.update(tuners)


def _plant_job(args):
    variant, spec_d, label_cost, seed = args
    p = problem_from_spec(spec_from_dict(spec_d))
    r = _TUNERS[variant].tune(p, seed=seed)
    if not r.success:
        return dict(variant=variant, seed=seed, ratio=np.nan, evals=r.evals, sim_seconds=r.sim_seconds, wall=r.wall_time)
    itae = p.itae(p.simulate(r.gains, "npid", "step", count=False))
    return dict(variant=variant, seed=seed, ratio=itae / label_cost, evals=r.evals, sim_seconds=r.sim_seconds, wall=r.wall_time)


def stage_run():
    os.makedirs(FIG, exist_ok=True)
    b = load_main()
    tr = b.select(b.split == "train")
    tuners = build_variants(tr)
    rows = []
    with Pool(WORKERS, initializer=_init, initargs=(tuners,)) as pool:
        for split in ("val", "test", "ood"):
            ev = b.select(b.split == split)
            jobs = [(v, ev.specs[i], float(ev.best_cost[i]), s) for v in VARIANTS for i in range(len(ev))
                    for s in (SEEDS if "refine" in v else (0,))]
            idents = [int(ev.ident[i]) for v in VARIANTS for i in range(len(ev)) for s in (SEEDS if "refine" in v else (0,))]
            ptypes = [str(ev.ptype[i]) for v in VARIANTS for i in range(len(ev)) for s in (SEEDS if "refine" in v else (0,))]
            for r, i_, t_ in zip(pool.map(_plant_job, jobs, chunksize=6), idents, ptypes):
                rows.append(dict(split=split, ident=i_, ptype=t_, **r))
            print("done", split, flush=True)
            pd.DataFrame(rows).to_csv(f"{OUT}/M5b_perplant.csv", index=False)


# ----------------------------------------------------------------------------- benchmark plants
def _bench_job(args):
    variant, plant, seed = args
    p = TuningProblem.from_scenario(plant)
    r = _TUNERS[variant].tune(p, seed=seed)
    if not r.success:
        return dict(variant=variant, plant=plant, seed=seed, success=False, message=r.message, evals=r.evals)
    ev = evaluate_gains(p, r.gains, "npid")
    return dict(variant=variant, plant=plant, seed=seed, success=True, evals=r.evals, sim_seconds=r.sim_seconds,
                itae_step=ev["step"]["ITAE"], itae_load=ev.get("load", {}).get("ITAE", np.nan), itae_noise=ev["noise"]["ITAE"],
                overshoot=ev["step"]["overshoot_pct"],**{f"gain_{k}": v for k, v in r.gains.items()})


def stage_bench():
    b = load_main()
    tr = b.select(b.split == "train")
    tuners = build_variants(tr)
    cfgs = pd.read_csv(f"{OUT}/M5a_configs.csv")
    ga_row = cfgs[cfgs.method == "GA"].sort_values("cv_fitness").iloc[0]
    cfg = ModelConfig(ga_row.model, json.loads(ga_row.params), tuple(json.loads(ga_row["mask"])))
    ga_model = build_model(cfg).fit(tr.X("engineered"), tr.best_z)
    tuners["GA-enhanced SVR/ML"] = HybridModelTuner(ga_model, name="GA-enhanced ML")
    print("GA-selected:", cfg.describe(), flush=True)
    names = list(tuners)
    jobs = [(v, plant, s) for v in names for plant in PLANT_NAMES for s in (SEEDS if "refine" in v else (0,))]
    with Pool(WORKERS, initializer=_init, initargs=(tuners,)) as pool:
        rows = pool.map(_bench_job, jobs, chunksize=2)
    pd.DataFrame(rows).to_csv(f"{OUT}/M5b_benchmark.csv", index=False)
    print(pd.DataFrame(rows).groupby(["plant", "variant"]).itae_step.median().unstack().round(3).to_string())


# ----------------------------------------------------------------------------- analysis
def stage_analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = pd.read_csv(f"{OUT}/M5b_perplant.csv")
    d["failed"] = d.ratio.isna()
    d["ratio"] = d.ratio.fillna(100.0)
    # per (variant, split): pool plants x seeds (each seed is a separate run), summarise
    rows = []
    for (v, s), g in d.groupby(["variant", "split"]):
        rows.append(dict(variant=v, split=s, **summarize(g.ratio.values), failed=int(g.failed.sum()),
                         evals_median=float(g.evals.median()), sim_seconds_median=float(g.sim_seconds.median())))
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/M5b_summary.csv", index=False)
    for col in ("median", "p90", "within1p5"):
        print(col); print(t.pivot(index="variant", columns="split", values=col).loc[VARIANTS].round(3).to_string())
    print(t.pivot(index="variant", columns="split", values="evals_median").loc[VARIANTS].to_string())
    # by plant structure (val + test)
    bt = (d[d.split.isin(["val", "test"])].groupby(["variant", "ptype"]).ratio.median().unstack().loc[VARIANTS].round(2))
    bt.to_csv(f"{OUT}/M5b_by_type.csv"); print(bt.to_string())

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    vt = t[t.split.isin(["val", "test"])].groupby("variant").agg(median=("median", "mean"), p90=("p90", "mean"), evals=("evals_median", "mean")).loc[VARIANTS]
    for v in VARIANTS:
        ax[0].scatter(vt.loc[v, "evals"], vt.loc[v, "median"], s=60)
        ax[0].annotate(v, (vt.loc[v, "evals"], vt.loc[v, "median"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("closed-loop simulations used (incl. bump test)")
    ax[0].set_ylabel("median regret (val+test)"); ax[0].set_title("Tuning cost vs accuracy")
    for v in VARIANTS:
        ax[1].scatter(vt.loc[v, "evals"], vt.loc[v, "p90"], s=60)
        ax[1].annotate(v, (vt.loc[v, "evals"], vt.loc[v, "p90"]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("closed-loop simulations used (incl. bump test)")
    ax[1].set_ylabel("90th-percentile regret (val+test)"); ax[1].set_title("Tuning cost vs worst cases")
    fig.tight_layout(); fig.savefig(f"{FIG}/M5b_cost_vs_regret.png", dpi=130)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for v in VARIANTS:
        r = np.sort(d[(d.variant == v) & d.split.isin(["val", "test"])].ratio.values)
        ax.plot(r, np.arange(1, len(r) + 1) / len(r), label=v)
    ax.set_xscale("log"); ax.set_xlim(0.9, 100); ax.set_xlabel("regret"); ax.set_ylabel("fraction of plants"); ax.legend(fontsize=7)
    ax.set_title("Regret distribution (val + test)"); fig.tight_layout(); fig.savefig(f"{FIG}/M5b_regret_cdf.png", dpi=130)

    if os.path.exists(f"{OUT}/M5b_benchmark.csv"):
        bm = pd.read_csv(f"{OUT}/M5b_benchmark.csv")
        bk = pd.read_csv("results/methods/M3/best_known.csv")
        bk = bk[(bk.kind == "npid") & (~bk.tune_shape)].set_index("plant").cost
        ok = bm[bm.success.astype(bool)]
        tab = ok.groupby(["plant", "variant"]).agg(step=("itae_step", "median"), load=("itae_load", "median"),
                                                  noise=("itae_noise", "median"), evals=("evals", "median")).reset_index()
        tab["gap_to_best_known"] = tab.apply(lambda r: r.step / bk[r.plant], axis=1)
        tab.to_csv(f"{OUT}/M5b_benchmark_summary.csv", index=False)
        print(tab.pivot(index="variant", columns="plant", values="gap_to_best_known").round(2).to_string())
        print("not applicable / failed:", int((~bm.success.astype(bool)).sum()))


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    {"run": stage_run, "bench": stage_bench, "analyze": stage_analyze}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
