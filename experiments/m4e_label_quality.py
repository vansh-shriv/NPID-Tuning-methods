"""
M4e - How good are the dataset labels? A long search on test plants (GA and PSO, 1500 simulations each, new seeds)
compared with the label (best of GA 400 + PSO 400). Regret in M4 is measured against the label, so this calibrates it.

Run from the repo root:  python experiments/m4e_label_quality.py
"""

import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plant_family import problem_from_spec, spec_from_dict
from src.tuners.ml import load_bundle
from src.tuners.optimizers import GA, PSO

warnings.filterwarnings("ignore")
OUT = "results/methods/M4"


def job(args):
    spec_d, ident, ptype, label = args
    spec = spec_from_dict(spec_d)
    res = [cls().tune(problem_from_spec(spec), budget=1500, seed=seed) for cls, seed in ((GA, 7001 + ident), (PSO, 8001 + ident))]
    best = min(r.cost for r in res)
    return dict(ident=ident, ptype=ptype, label=label, long_best=best, ga_long=res[0].cost, pso_long=res[1].cost,
                label_over_long=label / best)


def main():
    b = load_bundle("data/m4/dataset.jsonl")
    idx = np.where(b.split == "test")[0][:48]
    jobs = [(b.specs[i], int(b.ident[i]), str(b.ptype[i]), float(b.best_cost[i])) for i in idx]
    with ProcessPoolExecutor(max_workers=12) as ex:
        rows = list(ex.map(job, jobs))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/M4_label_quality.csv", index=False)
    r = df.label_over_long
    print("label / long-search cost: median %.3f mean %.3f p90 %.3f max %.3f ; label better in %.0f%%" %
          (r.median(), r.mean(), r.quantile(0.9), r.max(), 100 * (r < 1).mean()))
    print(df.groupby("ptype").label_over_long.median())


if __name__ == "__main__":
    main()
