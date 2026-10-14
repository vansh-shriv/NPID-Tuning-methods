"""
Generic PSO tuner (Phase 1 of docs/ROADMAP_v2.md)

Unlike `tuning_pso.py` (N-PID only, fixed n=m=p=2, DC motor), this tunes either
controller family on any `Plant`, with the same objective, bounds and budget, so
L-PID and N-PID can be compared like for like.

    tune_pso("lpid", plant, 20.0, center=zn_gains)
    tune_pso("npid", plant, 20.0, center=zn_gains, tune_shape=True)
"""

import logging

import numpy as np
import pyswarms as ps

from src.controllers import LPID, NPID
from src.metrics import compute_metrics
from src.simulate import run_simulation

# Search ranges for the NPID shape exponents when tune_shape=True
SHAPE_BOUNDS = {"n": (0.5, 4.0), "m": (0.5, 4.0), "p": (0.5, 4.0)}
FAIL_COST = 1e6


def make_controller(kind, gains, plant, dt=0.001, **fixed):
    """Build an LPID/NPID from a gains dict {Kp, Ki, Kd[, n, m, p]} and fixed kwargs."""
    kw = dict(u_min=plant.u_min, u_max=plant.u_max)
    kw.update(fixed)
    if kind == "lpid":
        return LPID(gains["Kp"], gains["Ki"], gains["Kd"], dt, **kw)
    if kind == "npid":
        shape = {k: gains[k] for k in ("n", "m", "p") if k in gains}
        return NPID(gains["Kp"], gains["Ki"], gains["Kd"], dt, **shape, **kw)
    raise ValueError(f"unknown controller kind {kind!r}")


def step_itae(kind, gains, plant, y_ref, t_end=4.0, dt=0.001, T_load_func=None,
              noise_sigma=0.0, noise_seeds=(0,), **fixed):
    """ITAE of closed-loop step response(s) (the PSO fitness).

    With noise_sigma > 0 the ITAE is averaged over `noise_seeds` noise realizations
    (common random numbers, so every candidate sees the same noise).
    """
    if min(gains["Kp"], gains["Ki"], gains["Kd"]) < 0:
        return FAIL_COST
    seeds = noise_seeds if noise_sigma > 0 else (0,)
    total = 0.0
    for sd in seeds:
        ctrl = make_controller(kind, gains, plant, dt, **fixed)
        try:
            res = run_simulation(ctrl, plant, y_ref, t_end=t_end, dt=dt, T_load_func=T_load_func,
                                 noise_sigma=noise_sigma, rng=np.random.default_rng(sd))
        except Exception:
            return FAIL_COST
        if not np.all(np.isfinite(res.omega)) or np.max(np.abs(res.omega)) > 10.0 * abs(y_ref):
            return FAIL_COST
        total += float(compute_metrics(res.time, y_ref, res.omega)["ITAE"])
    return total / len(seeds)


def tune_pso(kind, plant, y_ref, center, tune_shape=False, n_particles=20, iters=40,
             t_end=4.0, dt=0.001, T_load_func=None, seed=0, lo=0.1, hi=3.0,
             noise_sigma=0.0, noise_seeds=(100, 101, 102), bounds=None, log_space=False, restarts=1, **fixed):
    """
    PSO over [Kp, Ki, Kd] (+ [n, m, p] if tune_shape and kind == "npid").

    Args:
        bounds: explicit ((Kp,Ki,Kd) lower, (Kp,Ki,Kd) upper); overrides `center`.
        restarts: independent PSO runs (seeds seed, seed+1, ...); the best is returned.
        log_space: search log10(Kp, Ki, Kd) instead of the gains themselves.
        center: (Kp, Ki, Kd) around which the search box [lo*center, hi*center] is built
            (spec §3.3: 0.1x-3x of the Ziegler-Nichols gains).
        noise_sigma: if > 0, fitness is averaged over noisy runs (noise-aware tuning).
        fixed: extra controller kwargs held constant (tau_d, e_scale, de_scale, ...).

    Returns:
        (gains_dict, best_cost)
    """
    names = ["Kp", "Ki", "Kd"]
    if bounds is not None:                      # explicit (lower, upper) per gain
        lb, ub = list(bounds[0]), list(bounds[1])
    else:
        lb = [lo * c for c in center]
        ub = [hi * c for c in center]
    n_gain = 3
    if log_space:                               # search log10(gain): gains span decades
        lb = list(np.log10(lb))
        ub = list(np.log10(ub))
    if tune_shape and kind == "npid":
        for k in ("n", "m", "p"):
            names.append(k)
            lb.append(SHAPE_BOUNDS[k][0])
            ub.append(SHAPE_BOUNDS[k][1])

    def decode(x):
        x = np.asarray(x, dtype=float).copy()
        if log_space:
            x[:n_gain] = 10.0 ** x[:n_gain]
        return dict(zip(names, x))

    def objective(X):
        return np.array([
            step_itae(kind, decode(x), plant, y_ref, t_end, dt, T_load_func,
                      noise_sigma, noise_seeds, **fixed)
            for x in X
        ])

    logging.getLogger("pyswarms").setLevel(logging.WARNING)
    best_cost, best_pos = np.inf, None
    for r in range(restarts):                       # multi-start: PSO can converge prematurely on one seed
        np.random.seed(seed + r)                    # pyswarms draws from the global numpy RNG
        opt = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=len(names),
                                      options={"c1": 0.5, "c2": 0.3, "w": 0.9},
                                      bounds=(np.array(lb), np.array(ub)))
        cost, pos = opt.optimize(objective, iters=iters, verbose=False)
        if cost < best_cost:
            best_cost, best_pos = cost, pos
    return {k: float(v) for k, v in decode(best_pos).items()}, float(best_cost)
