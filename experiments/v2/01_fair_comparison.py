"""
Experiment 01 - Fair L-PID vs N-PID comparison (ROADMAP Phase 1)

Question: does N-PID beat L-PID once *both* get the same tuning treatment?

Variants (each with an unfiltered and a low-pass-filtered derivative):
    L-PID  ZN | PSO
    N-PID  (paper-literal shaping, e_scale=1)       ZN | PSO | PSO + n,m,p
    N-PID  (scale-free shaping, e_scale=w_ref)      ZN | PSO

PSO is trained on the noise-free step only (ITAE, same box/budget for all) and
then evaluated on: clean step, load-torque step, and noisy feedback (mean/std
over NOISE_SEEDS noise realizations).

Run from repo root:  python experiments/v2/01_fair_comparison.py
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.metrics import compute_metrics
from src.plant import DCMotorParams
from src.simulate import run_simulation
from src.tuning_generic import make_controller, tune_pso
from src.tuning_zn import find_critical_gain, ziegler_nichols_gains

W_REF = 20.0
SIGMA = 0.3
NOISE_SEEDS = list(range(10))
TAU_FILTER = 0.01
DT = 0.001
OUT_DIR = "results/v2"
LOAD = lambda t: 0.05 if t >= 1.5 else 0.0

SCALED = dict(e_scale=W_REF, de_scale=W_REF / 0.1)   # shaping acts at 100% error / 100%-per-0.1s rate


def noise_aware_variants():
    """Same controllers, but PSO fitness averages ITAE over noisy runs (seeds disjoint from eval seeds)."""
    for tau in (0.0, TAU_FILTER):
        f = {"tau_d": tau}
        tag = f"filt{tau * 1000:.0f}ms" if tau else "raw"
        yield dict(name=f"L-PID PSO-noisy [{tag}]", kind="lpid", fixed=f, tune="pso", train_sigma=SIGMA)
        yield dict(name=f"N-PID PSO-noisy [{tag}]", kind="npid", fixed=f, tune="pso", train_sigma=SIGMA)
        yield dict(name=f"N-PID(scaled) PSO-noisy [{tag}]", kind="npid", fixed={**f, **SCALED},
                   tune="pso", train_sigma=SIGMA)


def variants():
    for tau in (0.0, TAU_FILTER):
        f = {"tau_d": tau}
        tag = f"filt{tau * 1000:.0f}ms" if tau else "raw"
        yield dict(name=f"L-PID ZN [{tag}]", kind="lpid", fixed=f, tune="zn")
        yield dict(name=f"L-PID PSO [{tag}]", kind="lpid", fixed=f, tune="pso")
        yield dict(name=f"N-PID ZN [{tag}]", kind="npid", fixed=f, tune="zn")
        yield dict(name=f"N-PID PSO [{tag}]", kind="npid", fixed=f, tune="pso")
        yield dict(name=f"N-PID PSO+shape [{tag}]", kind="npid", fixed=f, tune="pso", shape=True)
        yield dict(name=f"N-PID(scaled) ZN [{tag}]", kind="npid", fixed={**f, **SCALED}, tune="zn")
        yield dict(name=f"N-PID(scaled) PSO [{tag}]", kind="npid", fixed={**f, **SCALED}, tune="pso")


def evaluate(kind, gains, fixed, plant):
    """Return {scenario: metrics-dict} for a tuned controller."""
    mk = lambda: make_controller(kind, gains, plant, DT, **fixed)
    out = {}
    r = run_simulation(mk(), plant, W_REF, t_end=4.0, dt=DT)
    out["step"] = compute_metrics(r.time, W_REF, r.omega)
    r = run_simulation(mk(), plant, W_REF, t_end=4.0, dt=DT, T_load_func=LOAD)
    out["load"] = compute_metrics(r.time, W_REF, r.omega)
    runs = []
    for s in NOISE_SEEDS:
        r = run_simulation(mk(), plant, W_REF, t_end=4.0, dt=DT, noise_sigma=SIGMA,
                           rng=np.random.default_rng(s))
        m = compute_metrics(r.time, W_REF, r.omega)
        m["u_chatter"] = float(np.mean(np.abs(np.diff(r.voltage))))  # control activity (V/step)
        runs.append(m)
    out["noise"] = {k: float(np.nanmean([m[k] for m in runs])) for k in runs[0]}
    out["noise_std"] = {k: float(np.nanstd([m[k] for m in runs])) for k in runs[0]}
    return out


def run_variant(spec, zn, seed=0):
    plant = DCMotorParams()
    gains = dict(zip(("Kp", "Ki", "Kd"), zn))
    cost = np.nan
    if spec["tune"] == "pso":
        gains, cost = tune_pso(spec["kind"], plant, W_REF, center=zn, tune_shape=spec.get("shape", False),
                               seed=seed, restarts=3, noise_sigma=spec.get("train_sigma", 0.0), **spec["fixed"])
    res = evaluate(spec["kind"], gains, spec["fixed"], plant)
    rows = []
    for scen in ("step", "load", "noise"):
        row = dict(variant=spec["name"], scenario=scen, train_ITAE=cost, **{k: round(v, 3) for k, v in gains.items()})
        row.update(res[scen])
        if scen == "noise":
            row.update({f"{k}_std": v for k, v in res["noise_std"].items()})
        rows.append(row)
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plant = DCMotorParams()
    Kcr, Tcr = find_critical_gain(plant, W_REF, verbose=False)
    zn = ziegler_nichols_gains(Kcr, Tcr)
    print(f"ZN: Kcr={Kcr:.2f} Tcr={Tcr:.4f}  ->  Kp={zn[0]:.3f} Ki={zn[1]:.3f} Kd={zn[2]:.4f}")

    specs = list(variants()) + list(noise_aware_variants())
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(run_variant, specs, [zn] * len(specs)))
    df = pd.DataFrame([r for rows in results for r in rows])
    df.to_csv(f"{OUT_DIR}/01_fair_comparison.csv", index=False)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    cols = ["variant", "Kp", "Ki", "Kd", "overshoot_pct", "settling_time", "IAE", "ITAE"]
    for scen in ("step", "load", "noise"):
        print(f"\n=== {scen} ===")
        print(df[df.scenario == scen][cols + (["u_chatter"] if scen == "noise" else [])].to_string(index=False))


if __name__ == "__main__":
    main()
