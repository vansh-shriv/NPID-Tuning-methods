"""
Benchmark runner (docs/PROTOCOL.md sections 4-8).

`evaluate_gains` scores a gain set on the fixed scenarios WITHOUT counting experiments (this is the
referee, not part of any method). `run_benchmark` runs tuners over plants and seeds, evaluates the returned
gains on both the N-PID and the L-PID, and returns one flat DataFrame row per
(plant, method, seed, controller kind, scenario).
"""

import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from src.metrics import compute_metrics
from src.tuners.base import FAIL_COST, TuningProblem

METRIC_KEYS = ("rise_time", "overshoot_pct", "settling_time", "IAE", "ISE", "ITAE")
NOISE_SEEDS = tuple(range(10))


def _scenario_metrics(problem, res):
    ok = np.all(np.isfinite(res.omega)) and np.max(np.abs(res.omega)) <= 10.0 * abs(problem.y_ref)
    if not ok:
        return {**{k: np.nan for k in METRIC_KEYS}, "ITAE": FAIL_COST, "effort": np.nan, "diverged": True}
    m = compute_metrics(res.time, problem.y_ref, res.omega)
    m = {k: float(m[k]) for k in METRIC_KEYS}
    m["effort"] = float(np.mean(np.abs(np.diff(res.voltage))))      # control activity per step
    m["diverged"] = False
    return m


def evaluate_gains(problem, gains, kind="npid", noise_seeds=NOISE_SEEDS):
    """Return {"step": metrics, "load": metrics, "noise": mean-metrics, "noise_std": std-metrics}."""
    out = {}
    out["step"] = _scenario_metrics(problem, problem.simulate(gains, kind, "step", count=False))
    if problem.disturbance is not None:
        out["load"] = _scenario_metrics(problem, problem.simulate(gains, kind, "load", count=False))
    runs = [_scenario_metrics(problem, problem.simulate(gains, kind, "noise", noise_seed=s, count=False))
            for s in noise_seeds]
    numeric = [k for k in runs[0] if k != "diverged"]
    out["noise"] = {k: float(np.nanmean([r[k] for r in runs])) if not all(np.isnan(r[k]) for r in runs) else np.nan
                    for k in numeric}
    out["noise"]["diverged"] = any(r["diverged"] for r in runs)
    out["noise_std"] = {k: float(np.nanstd([r[k] for r in runs])) for k in numeric}
    return out


def _job(args):
    plant, tuner, seed, budget, kinds = args
    problem = TuningProblem.from_scenario(plant)
    res = tuner.tune(problem, budget=budget, seed=seed)
    rows = []
    base = dict(plant=plant, method=tuner.name, family=tuner.family, seed=seed, evals=res.evals,
                sim_seconds=res.sim_seconds, wall_time=res.wall_time, success=res.success, message=res.message,
                **{f"gain_{k}": v for k, v in res.gains.items()})
    base.update({f"meta_{k}": v for k, v in res.meta.items() if np.isscalar(v)})
    if not res.success:
        return [dict(base, kind=k, scenario="step", ITAE=np.nan) for k in kinds], res
    for kind in kinds:
        ev = evaluate_gains(problem, res.gains, kind)
        for scen in ("step", "load", "noise"):
            if scen in ev:
                row = dict(base, kind=kind, scenario=scen, **ev[scen])
                if scen == "noise":
                    row.update({f"{k}_std": v for k, v in ev["noise_std"].items()})
                rows.append(row)
    return rows, res


def run_benchmark(tuners, plant_names, seeds=(0,), budget=None, kinds=("npid", "lpid"), workers=1,
                  return_results=False):
    """
    Run every tuner on every plant. Deterministic tuners should be given `seeds=(0,)`; stochastic ones use all
    seeds. A tuner attribute `stochastic = True` makes the runner use all `seeds` for it, otherwise only seeds[0].
    """
    jobs = []
    for plant, tuner in itertools.product(plant_names, tuners):
        for s in (seeds if getattr(tuner, "stochastic", False) else seeds[:1]):
            jobs.append((plant, tuner, s, budget, kinds))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(_job, jobs))
    else:
        out = [_job(j) for j in jobs]
    df = pd.DataFrame([r for rows, _ in out for r in rows])
    df = annotate_failures(df)
    return (df, [res for _, res in out]) if return_results else df


def annotate_failures(df, factor=100.0):
    """Failure flag (docs/PROTOCOL.md section 6): diverged/non-finite, tuner not applicable, or ITAE > factor x best on that plant/scenario."""
    df = df.copy()
    if "diverged" not in df:
        df["diverged"] = False
    df["diverged"] = df["diverged"].fillna(False).astype(bool)
    ok = df["success"].astype(bool) & ~df["diverged"] & np.isfinite(df["ITAE"])
    best = df[ok].groupby(["plant", "scenario", "kind"])["ITAE"].transform("min")
    best = best.reindex(df.index)
    df["failed"] = ~ok | (df["ITAE"] > factor * best.fillna(np.inf))
    return df
