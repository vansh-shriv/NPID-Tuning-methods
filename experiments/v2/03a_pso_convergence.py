"""
Experiment 03a - PSO repeatability / budget choice (ROADMAP Phase 2 item 3, needed for Phase 5 labels)

For several plants, run PSO with different seeds and budgets and compare the
achieved cost and the variation in gains. Decides the particles x iterations used
for dataset labels.

Run from repo root:  python experiments/v2/03a_pso_convergence.py
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.plant import DCMotorParams
from src.tuning_generic import tune_pso

BOUNDS = ((10.0, 100.0, 0.05), (500.0, 15000.0, 5.0))
PLANTS = [dict(J=0.005, B=0.05, Tc=0.02, w=5.0), dict(J=0.01, B=0.1, Tc=0.05, w=20.0),
          dict(J=0.02, B=0.2, Tc=0.1, w=50.0), dict(J=0.012, B=0.15, Tc=0.03, w=35.0)]
BUDGETS = [(8, 8), (16, 25), (24, 40)]
SEEDS = [0, 1, 2]
LOAD = 0.03


def run(job):
    pi, (npart, iters), seed = job
    pl = PLANTS[pi]
    plant = DCMotorParams(J=pl["J"], B=pl["B"], Tc=pl["Tc"])
    w = pl["w"]
    g, c = tune_pso("npid", plant, w, None, n_particles=npart, iters=iters, t_end=3.0, seed=seed,
                    T_load_func=lambda t: LOAD if t >= 1.5 else 0.0,
                    bounds=BOUNDS, log_space=True, e_scale=w, de_scale=10 * w)
    return dict(plant=pi, budget=f"{npart}x{iters}", seed=seed, cost=c, cost_norm=c / w ** 2 * 100, **g)


if __name__ == "__main__":
    jobs = [(p, b, s) for p in range(len(PLANTS)) for b in BUDGETS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=14) as ex:
        df = pd.DataFrame(list(ex.map(run, jobs)))
    os.makedirs("results/v2", exist_ok=True)
    df.to_csv("results/v2/03a_pso_convergence.csv", index=False)
    pd.set_option("display.width", 200)
    g = df.groupby(["plant", "budget"]).agg(cost_mean=("cost", "mean"), cost_cv_pct=("cost", lambda x: 100 * x.std() / x.mean()),
                                           Kp_cv_pct=("Kp", lambda x: 100 * x.std() / x.mean()),
                                           Ki_cv_pct=("Ki", lambda x: 100 * x.std() / x.mean()),
                                           Kd_cv_pct=("Kd", lambda x: 100 * x.std() / x.mean())).round(3)
    print(g)
