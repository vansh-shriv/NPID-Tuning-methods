"""
Experiment 05 - Multi-plant evaluation (ROADMAP Phase 4)

The source review covers process control, motor drives and robotic manipulators. This runs the
same pipeline (ZN where it exists, PSO-tuned L-PID, PSO-tuned N-PID with paper-literal and
scale-free shaping, with/without a derivative filter) on five plants of different character:

    dc_speed      DC motor speed, Coulomb friction (anchor plant, relative degree 2)
    dc_position   DC motor position (adds an integrator), Coulomb friction
    link          single-link manipulator with gravity torque m*g*l*sin(q) (strong nonlinearity)
    fopdt         first-order plus dead time, K=2, tau=5, theta=1 (classic process model)
    tank          nonlinear tank level, sqrt outflow (state-dependent gain)

Everything is generic: PSO boxes are derived from the plant's actuator range, setpoint and a
time scale Ts (no hand-tuned gains), N-PID shaping scales are e_scale=|ref|, de_scale=|ref|/Ts.
PSO (best of 3 restarts: single seeds sometimes converge prematurely) trains on the clean step; evaluation = clean step, input/load disturbance, noisy feedback
(10 seeds).

Run from repo root:  python experiments/v2/05_multi_plant.py
"""

import math
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.metrics import compute_metrics
from src.plant import (DCMotorParams, DelayedPlant, FunctionPlant, TransferFunctionPlant,
                       dc_motor_dynamics)
from src.simulate import run_simulation
from src.tuning_generic import make_controller, tune_pso
from src.tuning_zn import find_critical_gain, ziegler_nichols_gains

NOISE_SEEDS = list(range(10))
OUT = "results/v2"


# ----------------------------------------------------------------------------- plants
def make_plant(name):
    if name == "dc_speed":
        return DCMotorParams()
    if name == "dc_position":
        m = DCMotorParams()
        f = lambda t, x, u, d: [*dc_motor_dynamics(t, x[:2], u, d, m), x[1]]
        return FunctionPlant(f, [0.0, 0.0, 0.0], output_fn=lambda x: x[2], u_limits=(-m.Vmax, m.Vmax), name=name)
    if name == "link":
        J, b, mgl, Tc = 0.05, 0.1, 1.47, 0.02

        def f(t, x, u, d):
            q, w = x
            return [w, (u - d - b * w - mgl * math.sin(q) - Tc * math.tanh(w / 0.01)) / J]
        return FunctionPlant(f, [0.0, 0.0], output_fn=lambda x: x[0], u_limits=(-5.0, 5.0), name=name)
    if name == "fopdt":
        return DelayedPlant(TransferFunctionPlant([2.0], [5.0, 1.0], u_limits=(-10.0, 10.0), name="fopdt"), 1.0)
    if name == "tank":
        f = lambda t, x, u, d: [-0.5 * math.sqrt(max(x[0], 0.0)) + 0.4 * (u + d)]
        return FunctionPlant(f, [0.0], u_limits=(0.0, 10.0), name=name)
    raise KeyError(name)


# ref, dt, t_end, time scale Ts, (disturbance magnitude, onset), noise sigma
SCEN = {
    "dc_speed":    dict(ref=20.0, dt=0.002, t_end=3.0,  Ts=0.1,  dist=(0.05, 1.5), sigma=0.3),
    "dc_position": dict(ref=1.0,  dt=0.002, t_end=2.0,  Ts=0.3,  dist=(0.05, 1.0), sigma=0.01),
    "link":        dict(ref=1.0,  dt=0.002, t_end=4.0,  Ts=0.5,  dist=(0.3, 2.0),   sigma=0.01),
    "fopdt":       dict(ref=1.0,  dt=0.02,  t_end=60.0, Ts=5.0,  dist=(0.5, 30.0),  sigma=0.02),
    "tank":        dict(ref=4.0,  dt=0.02,  t_end=60.0, Ts=5.0,  dist=(0.5, 30.0),  sigma=0.04),
}


def auto_box(plant, ref, Ts):
    """Generic log-space gain box from actuator range, setpoint and time scale (no per-plant tuning)."""
    umax = max(abs(plant.u_min), abs(plant.u_max))
    kp0 = umax / abs(ref)
    return ((0.05 * kp0, 0.01 * kp0 / Ts, 0.01 * kp0 * Ts), (20 * kp0, 10 * kp0 / Ts, 10 * kp0 * Ts))


def zn_gains(plant, sc):
    """ZN gains (Kp, Ki, Kd) and the mode that produced them, or (None, reason)."""
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Kcr, Tcr = find_critical_gain(plant, sc["ref"], t_end=sc["t_end"], dt=sc["dt"],
                                          kp_start=0.1 * max(abs(plant.u_max), abs(plant.u_min)) / abs(sc["ref"]),
                                          kp_max=1e4, verbose=False)
            mode = "limit-cycle" if w else "linear"
        return ziegler_nichols_gains(Kcr, Tcr), mode
    except RuntimeError as e:
        return None, "none (no ultimate gain)"


# ----------------------------------------------------------------------------- jobs
def variants(Ts):
    for filt in (False, True):
        tau = Ts / 20 if filt else 0.0
        tag = "filt" if filt else "raw"
        yield dict(name=f"L-PID PSO [{tag}]", kind="lpid", fixed=dict(tau_d=tau))
        yield dict(name=f"N-PID(literal) PSO [{tag}]", kind="npid", fixed=dict(tau_d=tau))
        yield dict(name=f"N-PID(scaled) PSO [{tag}]", kind="npid", fixed=dict(tau_d=tau, _scaled=True))


def resolve_fixed(fixed, sc):
    f = {k: v for k, v in fixed.items() if not k.startswith("_")}
    if fixed.get("_scaled"):
        f.update(e_scale=abs(sc["ref"]), de_scale=abs(sc["ref"]) / sc["Ts"])
    return f


def evaluate(plant_name, kind, gains, fixed):
    sc = SCEN[plant_name]
    plant = make_plant(plant_name)
    mk = lambda: make_controller(kind, gains, plant, sc["dt"], **fixed)
    mag, onset = sc["dist"]
    kw = dict(t_end=sc["t_end"], dt=sc["dt"])
    out = {}
    r = run_simulation(mk(), plant, sc["ref"], **kw)
    out["step"] = compute_metrics(r.time, sc["ref"], r.omega)
    r = run_simulation(mk(), plant, sc["ref"], T_load_func=lambda t: mag if t >= onset else 0.0, **kw)
    out["load"] = compute_metrics(r.time, sc["ref"], r.omega)
    runs = []
    for s in NOISE_SEEDS:
        r = run_simulation(mk(), plant, sc["ref"], noise_sigma=sc["sigma"], rng=np.random.default_rng(s), **kw)
        runs.append(compute_metrics(r.time, sc["ref"], r.omega))
    out["noise"] = {k: float(np.nanmean([m[k] for m in runs])) for k in runs[0]}
    return out


def job(spec):
    plant_name, v, zn = spec["plant"], spec["variant"], spec.get("zn")
    sc = SCEN[plant_name]
    plant = make_plant(plant_name)
    fixed = resolve_fixed(v["fixed"], sc)
    if zn is not None:                                  # ZN baseline row, no tuning
        gains, cost = dict(zip(("Kp", "Ki", "Kd"), zn)), np.nan
    else:
        lb, ub = auto_box(plant, sc["ref"], sc["Ts"])
        gains, cost = tune_pso(v["kind"], plant, sc["ref"], None, n_particles=16, iters=25, t_end=sc["t_end"],
                               dt=sc["dt"], seed=0, restarts=3, bounds=(lb, ub), log_space=True, **fixed)
    res = evaluate(plant_name, v["kind"], gains, fixed)
    return [dict(plant=plant_name, variant=v["name"], scenario=s, train_ITAE=cost,
                 **{k: float(f"{x:.4g}") for k, x in gains.items()}, **res[s]) for s in ("step", "load", "noise")]


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs, zn_info = [], {}
    for pn, sc in SCEN.items():
        plant = make_plant(pn)
        zn, mode = zn_gains(plant, sc)
        zn_info[pn] = mode
        print(f"{pn:12s} ZN: {mode}" + (f"  Kp={zn[0]:.4g} Ki={zn[1]:.4g} Kd={zn[2]:.4g}" if zn else ""), flush=True)
        if zn:
            jobs.append(dict(plant=pn, zn=zn, variant=dict(name="L-PID ZN", kind="lpid", fixed={})))
        jobs += [dict(plant=pn, variant=v) for v in variants(sc["Ts"])]

    with ProcessPoolExecutor(max_workers=14) as ex:
        rows = [r for rs in ex.map(job, jobs) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/05_multi_plant.csv", index=False)
    pd.DataFrame([dict(plant=k, zn_mode=v) for k, v in zn_info.items()]).to_csv(f"{OUT}/05_zn_modes.csv", index=False)

    pd.set_option("display.width", 220, "display.max_columns", 30)
    for scen in ("step", "load", "noise"):
        print(f"\n=== {scen}: ITAE (overshoot %) ===")
        t = df[df.scenario == scen].copy()
        t["cell"] = t.ITAE.map("{:.4g}".format) + " (" + t.overshoot_pct.map("{:.1f}".format) + ")"
        print(t.pivot(index="variant", columns="plant", values="cell").to_string())


if __name__ == "__main__":
    main()
