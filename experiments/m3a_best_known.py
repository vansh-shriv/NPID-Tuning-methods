"""
M3a - best-known optimum per plant and controller variant (reference for the optimizer gaps).

Variants: N-PID (3-D), L-PID (3-D), N-PID with shape exponents (6-D, PSO route only).
Slow (tens of minutes on the long-horizon plants); resumable per (plant, variant) via the output CSV.
Run from the repo root:  python experiments/m3a_best_known.py
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark_plants import PLANT_NAMES
from src.tuners import TuningProblem
from src.tuners.reference import best_known

OUT = "results/methods/M3/best_known.csv"
VARIANTS = [("npid", False), ("lpid", False), ("npid", True)]


def job(args):
    plant, kind, shape = args
    t0 = time.perf_counter()
    r = best_known(lambda: TuningProblem.from_scenario(plant), kind, shape)
    return dict(plant=plant, kind=kind, tune_shape=shape, cost=r["cost"], source=r["source"], n_evals=r["n_evals"],
                seconds=time.perf_counter() - t0, **{f"gain_{k}": v for k, v in r["gains"].items()})


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = pd.read_csv(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["plant", "kind", "tune_shape"])
    have = {(r.plant, r.kind, bool(r.tune_shape)) for r in done.itertuples()}
    todo = [(p, k, s) for p in PLANT_NAMES for k, s in VARIANTS if (p, k, s) not in have]
    print(len(todo), "jobs", flush=True)
    rows = done.to_dict("records")
    with ProcessPoolExecutor(max_workers=12) as ex:
        for r in ex.map(job, todo):
            rows.append(r)
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(r["plant"], r["kind"], r["tune_shape"], round(r["cost"], 5), f"{r['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
