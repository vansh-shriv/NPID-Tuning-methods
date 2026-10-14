"""
Experiment 03 - Dataset v2: plant parameters -> PSO-optimal gains (ROADMAP Phase 5)

Fixes found in the audit of the legacy dataset (see progress.md):
  * 42% of legacy rows asked for a setpoint above the motor's maximum speed at 24 V,
    so their "optimal" gains were arbitrary. v2 samples the setpoint as a fraction of the
    plant's maximum reachable speed:  w_ref = frac * w_max(plant, T_load).
  * PSO is seeded per trial, runs 16 particles x 25 iterations (cost CV <= 1.3% across seeds,
    see 03a), searches log10(gains) in a fixed wide box, and uses scale-free N-PID shaping.
  * Armature resistance R is randomized too, so one model can later cover drifting plants.
  * Every trial is checkpointed to JSONL; the run is resumable and deterministic per trial index.

Run from repo root:   python experiments/v2/03_generate_dataset_v2.py --samples 1200 --workers 14
"""

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.metrics import compute_metrics
from src.plant import DCMotorParams
from src.simulate import run_simulation
from src.tuning_generic import make_controller, tune_pso

BOUNDS = ((10.0, 100.0, 0.05), (500.0, 15000.0, 5.0))
RANGES = dict(J=(0.005, 0.02), B=(0.05, 0.2), Tc=(0.02, 0.1), T_load=(0.0, 0.05), R=(0.5, 2.0))
FRAC = (0.15, 0.85)           # w_ref as a fraction of the plant's maximum speed
T_END, T_LOAD_ON = 3.0, 1.5
N_PARTICLES, ITERS = 16, 25
COLUMNS = ["J", "B", "Tc", "T_load", "R", "omega_ref", "Kp", "Ki", "Kd",
           "achieved_ITAE", "overshoot_pct", "settling_time", "at_bound"]
FEATURES = ["J", "B", "Tc", "T_load", "R", "omega_ref"]


def w_max(p, T_load):
    """Maximum steady speed at full voltage with Coulomb friction and load torque."""
    return (p.Kt * p.Vmax / p.R - p.Tc - T_load) / (p.B + p.Kt * p.Ke / p.R)


def sample_plant(seed, i):
    rng = np.random.default_rng([seed, i])
    v = {k: rng.uniform(*r) for k, r in RANGES.items()}
    p = DCMotorParams(J=v["J"], B=v["B"], Tc=v["Tc"], R=v["R"])
    w_ref = float(rng.uniform(*FRAC) * w_max(p, v["T_load"]))
    return p, v["T_load"], w_ref


def label_trial(task):
    seed, i = task
    p, T_load, w_ref = sample_plant(seed, i)
    load = lambda t: T_load if t >= T_LOAD_ON else 0.0
    fixed = dict(e_scale=w_ref, de_scale=10.0 * w_ref)
    gains, cost = tune_pso("npid", p, w_ref, None, n_particles=N_PARTICLES, iters=ITERS, t_end=T_END,
                           seed=int(seed * 100003 + i), T_load_func=load, bounds=BOUNDS, log_space=True, **fixed)
    res = run_simulation(make_controller("npid", gains, p, 0.001, **fixed), p, w_ref, t_end=T_END, dt=0.001,
                         T_load_func=load)
    m = compute_metrics(res.time, w_ref, res.omega)
    lo, hi = np.log10(BOUNDS[0]), np.log10(BOUNDS[1])
    g = np.log10([gains["Kp"], gains["Ki"], gains["Kd"]])
    at_bound = bool(np.any(g < lo + 0.02 * (hi - lo)) or np.any(g > hi - 0.02 * (hi - lo)))
    return dict(trial=i, J=p.J, B=p.B, Tc=p.Tc, T_load=T_load, R=p.R, omega_ref=w_ref, **gains,
                achieved_ITAE=cost, overshoot_pct=float(m["overshoot_pct"]),
                settling_time=float(m["settling_time"]), at_bound=at_bound)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--output", default="data/v2/dataset_npid.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    ckpt = a.output + ".partial.jsonl"

    rows = []
    if os.path.exists(ckpt):
        rows = [json.loads(l) for l in open(ckpt) if l.strip()]
    done = {r["trial"] for r in rows}
    todo = [(a.seed, i) for i in range(a.samples) if i not in done]
    print(f"{len(done)} done, {len(todo)} to run on {a.workers} workers", flush=True)

    t0 = time.perf_counter()
    with Pool(a.workers) as pool, open(ckpt, "a") as f:
        for k, r in enumerate(pool.imap_unordered(label_trial, todo)):
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            if (k + 1) % 20 == 0 or k + 1 == len(todo):
                el = time.perf_counter() - t0
                print(f"  {k + 1}/{len(todo)}  elapsed {el / 60:.1f} min  eta {(el / (k + 1)) * (len(todo) - k - 1) / 60:.1f} min",
                      flush=True)

    with open(a.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for r in sorted(rows, key=lambda r: r["trial"]):
            w.writerow([r[c] for c in COLUMNS])
    os.remove(ckpt)
    print("saved", a.output, len(rows), "rows")


if __name__ == "__main__":
    main()
