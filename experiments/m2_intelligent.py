"""
M2 - Intelligent tuning methods: self-tuning (pole placement) and model matching.

Stages: run (all computations) | analyze (tables and figures). Outputs: results/methods/M2/ (+ figures/).
Run from the repo root:  python experiments/m2_intelligent.py run  |  python experiments/m2_intelligent.py analyze
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark_plants import PLANT_NAMES
from src.tuners import TuningProblem
from src.tuners.benchmark import evaluate_gains, run_benchmark
from src.tuners.classical import TimeDomainTuning
from src.tuners.modelmatching import ModelMatching, reference_response
from src.tuners.optimizers import PSO
from src.tuners.selftuning import SelfTuningPolePlacement

warnings.filterwarnings("ignore")
OUT = "results/methods/M2"
FIG = f"{OUT}/figures"
WORKERS = 12


def main_tuners():
    return [SelfTuningPolePlacement(), ModelMatching("frequency"), ModelMatching("time"),
            TimeDomainTuning("simc"), PSO()]


# ----------------------------------------------------------------------------- run
def stage_main():
    df = run_benchmark(main_tuners(), PLANT_NAMES, seeds=tuple(range(5)), budget=400, workers=WORKERS)
    df.to_csv(f"{OUT}/M2_main.csv", index=False)


def _ablation_job(args):
    plant, label, value, kind, kw, seed = args
    p = TuningProblem.from_scenario(plant)
    if kind == "str":
        if "noise_rel" in kw:
            rel = kw.pop("noise_rel")
            p = TuningProblem.from_scenario(plant, noise_sigma=rel * abs(p.y_ref))
            kw["noisy"] = rel > 0
        t = SelfTuningPolePlacement(**kw)
    else:
        t = ModelMatching(**kw)
    r = t.tune(p, seed=seed)
    row = dict(plant=plant, ablation=label, value=value, seed=seed, success=r.success, message=r.message,
               evals=r.evals, sim_seconds=r.sim_seconds)
    if r.success:
        ref = TuningProblem.from_scenario(plant)
        ev = evaluate_gains(ref, r.gains)
        row.update(ITAE=ev["step"]["ITAE"], overshoot=ev["step"]["overshoot_pct"], ITAE_noise=ev["noise"]["ITAE"],
                   settling=ev["step"]["settling_time"], **{f"gain_{k}": v for k, v in r.gains.items()},
                   **{f"meta_{k}": v for k, v in r.meta.items() if np.isscalar(v)})
    return row


def stage_ablations():
    from concurrent.futures import ProcessPoolExecutor
    jobs = []
    for plant in PLANT_NAMES:
        for lam in (0.98, 0.99, 0.995, 0.999):
            jobs.append((plant, "STR lambda", lam, "str", dict(lam=lam), 0))
        for amp in (0.02, 0.05, 0.1, 0.2):
            jobs.append((plant, "STR excitation amp", amp, "str", dict(amp=amp), 0))
        for sp in (1.5, 3.0, 6.0):
            jobs.append((plant, "STR speedup", sp, "str", dict(speedup=sp), 0))
        for z in (0.5, 0.7, 0.9, 1.0):
            jobs.append((plant, "STR zeta", z, "str", dict(zeta=z), 0))
        for nk in (1, 3, 6):
            jobs.append((plant, "STR nk_max", nk, "str", dict(nk_max=nk), 0))
        for per in (4, 8, 16):
            jobs.append((plant, "STR periods", per, "str", dict(periods=per), 0))
        for rel in (0.0, 0.01, 0.02, 0.05):
            for seed in range(5 if rel > 0 else 1):
                jobs.append((plant, "STR noise", rel, "str", dict(noise_rel=rel), seed))
        for mode in ("frequency", "time"):
            for sp in (1.5, 3.0, 6.0):
                jobs.append((plant, "MM speedup", f"{mode}/{sp}", "mm", dict(mode=mode, speedup=sp), 0))
            for z in (0.7, 0.9, 1.0):
                jobs.append((plant, "MM zeta", f"{mode}/{z}", "mm", dict(mode=mode, zeta=z), 0))
            for order in (1, 2):
                jobs.append((plant, "MM order", f"{mode}/{order}", "mm", dict(mode=mode, order=order), 0))
        for ms in (30, 60, 120, 240):
            jobs.append((plant, "MM max_sims", ms, "mm", dict(mode="time", max_sims=ms), 0))
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        rows = list(ex.map(_ablation_job, jobs, chunksize=2))
    pd.DataFrame(rows).to_csv(f"{OUT}/M2_ablations.csv", index=False)


def stage_traces():
    """In-process STR runs keeping the parameter / gain traces; MM closed loops vs reference."""
    import pickle
    out = {"str": {}, "mm": {}}
    for plant in PLANT_NAMES:
        p = TuningProblem.from_scenario(plant)
        t = SelfTuningPolePlacement()
        r = t.tune(p)
        if r.success:
            truth = TuningProblem.from_scenario(plant)
            from src.tuners.identify import identify_process
            m = identify_process(truth)
            out["str"][plant] = dict(trace=t.trace, gains=r.gains, meta=r.meta, bump=m.data, bump_kind=m.kind,
                                     Ts=p.Ts, dt=p.dt, y_ref=p.y_ref)
        for mode in ("frequency", "time"):
            p = TuningProblem.from_scenario(plant)
            mm = ModelMatching(mode)
            r = mm.tune(p)
            if r.success:
                res = TuningProblem.from_scenario(plant).simulate(r.gains, count=False)
                tr, ym = reference_response(p, mm.last["ref"], 2)
                out["mm"][(plant, mode)] = dict(t=res.time, y=res.omega, t_ref=tr, y_ref=ym, gains=r.gains)
    with open(f"{OUT}/M2_traces.pkl", "wb") as f:
        pickle.dump(out, f)


# ----------------------------------------------------------------------------- analyze
def stage_analyze():
    import pickle

    import matplotlib.pyplot as plt

    from src.tuners.plots import _save, plot_cost_vs_performance, plot_heatmap

    os.makedirs(FIG, exist_ok=True)
    df = pd.read_csv(f"{OUT}/M2_main.csv")
    order = [t.name for t in main_tuners()]

    def pivot(kind, scen, value="ITAE"):
        d = df[(df.kind == kind) & (df.scenario == scen) & df.success]
        return d.groupby(["method", "plant"])[value].median().unstack("plant").reindex(index=order, columns=PLANT_NAMES)

    for kind in ("npid", "lpid"):
        for scen in ("step", "load", "noise"):
            t = pivot(kind, scen)
            t.to_csv(f"{OUT}/M2_ITAE_{kind}_{scen}.csv")
            plot_heatmap(t, f"{FIG}/M2_heatmap_ITAE_{kind}_{scen}.png",
                         title=f"M2: ITAE, {kind.upper()}, {scen} (median; colour = x best on the plant)")
    pivot("npid", "step", "overshoot_pct").to_csv(f"{OUT}/M2_overshoot_npid_step.csv")
    pivot("npid", "step", "settling_time").to_csv(f"{OUT}/M2_settling_npid_step.csv")
    (pivot("npid", "step") / pivot("lpid", "step")).to_csv(f"{OUT}/M2_ratio_npid_lpid_step.csv")
    (pivot("npid", "noise") / pivot("lpid", "noise")).to_csv(f"{OUT}/M2_ratio_npid_lpid_noise.csv")
    cost = df[(df.kind == "npid") & (df.scenario == "step")].groupby(["method", "plant"]).agg(
        evals=("evals", "median"), sim_seconds=("sim_seconds", "median"), wall_time=("wall_time", "median"),
        ITAE=("ITAE", "median"), success=("success", "all")).reset_index()
    cost.to_csv(f"{OUT}/M2_cost.csv", index=False)
    plot_cost_vs_performance(cost[cost.success], f"{FIG}/M2_cost_vs_performance.png", title="M2: tuning cost vs result quality")
    app = df[(df.kind == "npid") & (df.scenario == "step")].groupby(["method", "plant"]).apply(
        lambda g: "ok" if g.success.all() else "n/a: " + str(g.message.iloc[0])[:45], include_groups=False
    ).unstack("plant").reindex(index=order, columns=PLANT_NAMES)
    app.to_csv(f"{OUT}/M2_applicability.csv")

    # step overlays
    from src.tuners.plots import plot_step_overlays
    traces = {}
    for plant in PLANT_NAMES:
        p = TuningProblem.from_scenario(plant)
        entry = {"ref": p.y_ref, "methods": {}}
        for m in order:
            r = df[(df.plant == plant) & (df.method == m) & (df.scenario == "step") & (df.kind == "npid")]
            r = r[r.seed == r.seed.min()]
            if not len(r) or not bool(r.success.iloc[0]):
                entry["methods"][m] = None
                continue
            g = {k: float(r[f"gain_{k}"].iloc[0]) for k in ("Kp", "Ki", "Kd")}
            res = p.simulate(g, "npid", count=False)
            entry["methods"][m] = (res.time, res.omega)
        traces[plant] = entry
    plot_step_overlays(traces, f"{FIG}/M2_step_overlay_npid.png", title="M2: N-PID step responses")

    # STR traces and identification quality
    with open(f"{OUT}/M2_traces.pkl", "rb") as f:
        tr = pickle.load(f)
    fig, axes = plt.subplots(3, 5, figsize=(18, 8.5))
    for j, plant in enumerate(PLANT_NAMES):
        s = tr["str"].get(plant)
        if s is None:
            for i in range(3):
                axes[i, j].axis("off")
            axes[0, j].set_title(f"{plant}: n/a")
            continue
        t = np.array(s["trace"]["t"])
        th = np.array(s["trace"]["theta"])
        for i, nm in enumerate(["a1", "a2", "b0", "b1"]):
            axes[0, j].plot(t, th[:, i], label=nm)
        axes[0, j].set_title(f"{plant}: RLS parameters (nk={s['trace']['nk']}, Delta={s['trace']['delta']:.3g}s)", fontsize=8)
        gs = pd.DataFrame(s["trace"]["gains"])
        for nm in ("Kp", "Ki", "Kd"):
            axes[1, j].plot(t, gs[nm], label=nm)
        axes[1, j].set_yscale("log")
        axes[1, j].set_title("designed gains", fontsize=8)
        a1, a2, b0, b1 = th[-1]
        from src.tuners.selftuning import arx_step_response
        n = int(s["bump"]["t"][-1] / s["trace"]["delta"])
        y = arx_step_response(a1, a2, b0, b1, n=max(n, 10))
        axes[2, j].plot(s["bump"]["t"], s["bump"]["s"], label="bump test (measured)")
        axes[2, j].plot(np.arange(len(y)) * s["trace"]["delta"], y, "--", label="identified ARX")
        axes[2, j].set_title("unit-step response", fontsize=8)
        for i in range(3):
            axes[i, j].grid(alpha=0.3)
        axes[2, j].set_xlabel("time (s)")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    axes[2, 0].legend(fontsize=7)
    _save(fig, f"{FIG}/M2_str_identification.png")

    # model matching overlays
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.4))
    for ax, plant in zip(axes, PLANT_NAMES):
        for mode, c in (("frequency", "#1f77b4"), ("time", "#d62728")):
            d = tr["mm"].get((plant, mode))
            if d is None:
                continue
            ax.plot(d["t"], d["y"], color=c, label=f"MM {mode}")
        d = tr["mm"].get((plant, "time")) or tr["mm"].get((plant, "frequency"))
        if d is not None:
            ax.plot(d["t_ref"], d["y_ref"], "k--", label="reference model")
        ax.set_title(plant)
        ax.grid(alpha=0.3)
        ax.set_xlabel("time (s)")
    axes[0].legend(fontsize=8)
    _save(fig, f"{FIG}/M2_mm_vs_reference.png")

    # ablations
    ab = pd.read_csv(f"{OUT}/M2_ablations.csv")
    ab = ab[ab.success].copy()
    ab["value"] = ab.value.astype(str)
    for name, g in ab.groupby("ablation"):
        t = g.groupby(["plant", "value"]).ITAE.median().unstack("value").reindex(PLANT_NAMES)
        t.to_csv(f"{OUT}/M2_ablation_{name.replace(' ', '_')}.csv")
    fig, axes = plt.subplots(2, 4, figsize=(18, 7))
    names = ["STR lambda", "STR excitation amp", "STR speedup", "STR zeta", "STR nk_max", "STR periods", "STR noise", "MM max_sims"]
    for ax, name in zip(axes.ravel(), names):
        g = ab[ab.ablation == name]
        t = g.groupby(["plant", "value"]).ITAE.median().unstack("value").reindex(PLANT_NAMES)
        cols = sorted(t.columns, key=lambda v: float(v))
        t[cols].T.plot(ax=ax, marker="o", logy=True, legend=False)
        ax.set_title(name)
        ax.set_xlabel("")
        ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend(PLANT_NAMES, fontsize=7)
    _save(fig, f"{FIG}/M2_ablations.png")
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    for ax, name in zip(axes, ["MM speedup", "MM zeta", "MM order", "MM max_sims"]):
        g = ab[ab.ablation == name]
        t = g.groupby(["plant", "value"]).ITAE.median().unstack("value").reindex(PLANT_NAMES)
        t.plot.bar(ax=ax, logy=True, rot=20, legend=(name == "MM speedup"))
        ax.set_title(name)
        ax.grid(alpha=0.3, which="both", axis="y")
    _save(fig, f"{FIG}/M2_mm_ablations.png")
    print("analysis written to", OUT)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "run"
    if stage == "run":
        for fn in (stage_main, stage_traces, stage_ablations):
            print("running", fn.__name__, flush=True)
            fn()
    elif stage in ("main", "traces", "ablations"):
        globals()[f"stage_{stage}"]()
    elif stage == "analyze":
        stage_analyze()
