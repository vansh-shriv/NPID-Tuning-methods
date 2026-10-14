"""
M6 - Adaptive ML-based tuning (method 18) under plant changes.

Scenario (per plant of the M4 family): the closed loop runs 5 windows of H = 10 time scales, the setpoint changing
every H (1.0, 0.8, 1.0, 0.8, 1.0 x y_ref); the plant changes at 1.95 H, i.e. just before window 2:
    none        no change (does the adaptation harm?)           gain x0.4 / x2.5   plant gain change
    lag x3 / x0.4   all lags (inertia) tripled / cut to 40%
Methods (all start from the same gains: SVR on a noisy bump test, 3 experiments):
    frozen                       the initial gains forever
    adaptive ML (M6)             gated RLS estimator -> FOPDT/IPDT -> M4 SVR -> rate-limited gain update
    self-tuning pole placement   the same estimator, M2's pole-placement design instead of the ML predictor
    retune by experiment         M5 hybrid (Cohen-Coon + SVR + 20 simulations) on the plant before / after the
                                 change, switched exactly at the change (reference: knows when, costs 2 x 23 experiments)
Metric: ITAE of each window (from that window's reference step) / ITAE of the reference method in the same window.
Two protocols: noisy measurements (the plant's own sensor noise) and noise-free.

Run from the repo root:  python experiments/m6_adaptive.py run <split> <n_plants> [tag] [offset] | analyze <tag>
"""

import os
import sys
import warnings
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.m5_common import DATA, load_main
from src.tuners.ml import load_bundle
from experiments.m5b_hybrid import build_variants
from src.plant_family import problem_from_spec, spec_from_dict
from src.tuners.adaptive import (AdaptiveConfig, ClippedModel, AdaptivePolicy, FrozenPolicy, N_WINDOWS, ScheduledPolicy, change_feasible,
                                 run_scenario, sampling_steps, scaled_spec)
from src.tuners.features import denormalise_gains, extract_features

warnings.filterwarnings("ignore")
OUT = "results/methods/M6"
FIG = f"{OUT}/figures"
WORKERS = 12
CHANGES = [("none", 1.0), ("gain", 0.4), ("gain", 2.5), ("lag", 3.0), ("lag", 0.4)]
T_SWITCH = 1.95
CFG = AdaptiveConfig(detune=float(os.environ.get("M6_DETUNE", "0.5")))
_STATE = {}


def _init(tuners, cfg, X_train_clean):
    _STATE.update(tuners=tuners, cfg=cfg, X_train_clean=X_train_clean)          # tuners = {'noisy': variants trained on noisy features, 'clean': on clean ones}


def _job(args):
    spec_d, change, noise, seed = args
    spec = spec_from_dict(spec_d)
    cfg = _STATE["cfg"]
    # Feature protocol: a bump test with measurement noise needs the model trained on noisy features (M4); the models
    # on the *estimator* side always use clean-feature models (an ARX step response is smooth: its features look like
    # a noise-free bump test, as the fit-error / peak / t63 features show), see report section 4.
    tuners = _STATE["tuners"]["noisy" if noise else "clean"]
    svr_est = ClippedModel(_STATE["tuners"]["clean"]["SVR"].model, _STATE["X_train_clean"])
    sigma_kw = {} if noise else dict(noise_sigma=0.0)
    p = problem_from_spec(spec, **sigma_kw)
    base = dict(ident=spec.ident, ptype=spec.ptype, change=f"{change[0]} x{change[1]}", noise=noise, seed=seed)
    try:
        f = extract_features(p, noise_sigma=p.noise_sigma, seed=seed)
    except Exception as e:
        return [dict(**base, method="all", failed=True, note=f"identification: {type(e).__name__}")]
    svr = tuners["SVR"].model
    g0 = denormalise_gains(np.asarray(svr.predict(f.vec[None, :]))[0], f.K_eff, f.T_n)
    lb, ub = p.bounds
    g0 = {k: float(np.clip(v, lo, hi)) for (k, v), lo, hi in zip(g0.items(), lb, ub)}
    m = sampling_steps(f.T_n, f.model["theta"], p.dt, cfg.delta_div)
    H = spec.horizon()

    # reference: experiment-based retune before and after the change
    hyb = tuners["Cohen-Coon + SVR + refine"]
    g_b = hyb.tune(p, seed=seed).gains
    g_a = hyb.tune(problem_from_spec(scaled_spec(spec, change), **sigma_kw), seed=seed).gains if change[0] != "none" else g_b
    policies = {"frozen": FrozenPolicy(),
                "adaptive ML": AdaptivePolicy(cfg, svr_est),
                "self-tuning pole placement": AdaptivePolicy(replace(cfg, mode="pp")),
                "retune by experiment": ScheduledPolicy([(0.0, g_b), (T_SWITCH * H, g_a)])}
    rows = []
    for name, pol in policies.items():
        r = run_scenario(spec, change, pol, g0, m, noise=noise, seed=seed, t_switch_frac=T_SWITCH, kind0=f.kind, T_n0=f.T_n, problem=p)
        row = dict(**base, method=name, failed=r.failed, n_updates=len(r.gain_trace) - 1, delta_over_Tn=m * p.dt / f.T_n)
        if not r.failed:
            for w, wm in enumerate(r.windows):
                row.update({f"itae_w{w}": wm["itae"], f"os_w{w}": wm["overshoot_pct"]})
            for k in ("Kp", "Ki", "Kd"):
                row[f"final_{k}"] = r.gain_trace[-1][1][k]
        rows.append(row)
    return rows


def stage_run(split="val", n=12, tag=None, offset=0):
    os.makedirs(FIG, exist_ok=True)
    tag = tag or split
    b = load_main()
    tr = b.select(b.split == "train")
    bc = load_bundle(DATA)
    tuners = {"noisy": build_variants(tr), "clean": build_variants(bc.select(bc.split == "train"))}
    ev = b.select(b.split == split)
    idx = np.random.default_rng(0).permutation(len(ev))[offset:offset + n]
    jobs = []
    for i in idx:
        spec = spec_from_dict(ev.specs[i])
        for change in CHANGES:
            if not change_feasible(spec, change):
                continue
            for noise in (True, False):
                jobs.append((ev.specs[i], change, noise, 0))
    print(f"{len(jobs)} jobs", flush=True)
    with Pool(WORKERS, initializer=_init, initargs=(tuners, CFG, bc.select(bc.split == "train").X_eng)) as pool:
        out = pool.map(_job, jobs, chunksize=1)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_csv(f"{OUT}/M6_runs_{tag}.csv", index=False)
    print("saved", len(df), flush=True)


def summarize(df, ref="retune by experiment"):
    """Per (noise, change, method, window): median ITAE ratio to the reference method; failures count as ratio 100."""
    d = df[df.method != "all"].copy()
    wcols = [f"itae_w{w}" for w in range(N_WINDOWS)]
    for c in wcols:
        d[c] = d[c].where(~d.failed.astype(bool), np.nan)
    key = ["ident", "change", "noise", "seed"]
    refd = d[d.method == ref].set_index(key)[wcols]
    rows = []
    for _, r in d.iterrows():
        k = tuple(r[c] for c in key)
        for w, c in enumerate(wcols):
            ratio = (r[c] / refd.loc[k, c]) if (k in refd.index and np.isfinite(r[c]) and np.isfinite(refd.loc[k, c])) else 100.0
            rows.append(dict(ident=r.ident, ptype=r.ptype, change=r.change, noise=r.noise, method=r.method, window=w,
                             ratio=float(np.clip(ratio, 0.0, 100.0)), failed=bool(r.failed)))
    return pd.DataFrame(rows)


def stage_analyze(tag="val"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.read_csv(f"{OUT}/M6_runs_{tag}.csv")
    ratios = summarize(df)
    ratios.to_csv(f"{OUT}/M6_ratios_{tag}.csv", index=False)
    methods = ["frozen", "adaptive ML", "self-tuning pole placement", "retune by experiment"]
    g = ratios.groupby(["noise", "change", "method", "window"]).ratio.median().reset_index()
    pd.set_option("display.width", 250)
    for noise in (True, False):
        print(f"\n=== noise={noise}: median ITAE ratio to 'retune by experiment' per window (w2 = first window after the change)")
        t = g[g.noise == noise].pivot_table(index=["change", "method"], columns="window", values="ratio").round(2)
        print(t.loc[[(c, m) for c in [f"{a} x{b}" for a, b in CHANGES] for m in methods if (c, m) in t.index]].to_string())
    fl = df[(df.method != "all")].groupby(["noise", "method"]).failed.mean().round(3)
    print("\nfailure rate:\n", fl.to_string())
    # figure: windows 3-4 (adapted) median ratio per change, noisy protocol
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, noise in zip(axes, (True, False)):
        sub = g[(g.noise == noise) & (g.window >= 3)].groupby(["change", "method"]).ratio.mean().unstack()
        sub = sub.loc[[f"{a} x{b}" for a, b in CHANGES if f"{a} x{b}" in sub.index], methods]
        sub.plot.bar(ax=ax, rot=20)
        ax.set_yscale("log"); ax.axhline(1, color="k", lw=0.5)
        ax.set_title(f"windows 3-4 (after adaptation), {'noisy' if noise else 'noise-free'}"); ax.set_ylabel("median ITAE / retune by experiment")
    fig.tight_layout(); fig.savefig(f"{FIG}/M6_ratio_{tag}.png", dpi=130)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    stage = sys.argv[1]
    if stage == "run":
        stage_run(sys.argv[2] if len(sys.argv) > 2 else "val", int(sys.argv[3]) if len(sys.argv) > 3 else 12, sys.argv[4] if len(sys.argv) > 4 else None,
                  int(sys.argv[5]) if len(sys.argv) > 5 else 0)
    else:
        stage_analyze(sys.argv[2] if len(sys.argv) > 2 else "val")
