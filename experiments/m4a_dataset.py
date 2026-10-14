"""
M4a - Plant-agnostic dataset for the ML tuners.

For every sampled plant (src/plant_family.py):
    1. bump-test features (counted)                       -> src/tuners/features.py
    2. labels: best of GA (400 simulations) and PSO (400 simulations) on the clean-step ITAE of the N-PID
    3. *every* evaluated (normalised gains, normalised cost) pair is stored (800 per plant): these train the
       surrogate / decision-focused ANN and the Gaussian-process priors.

Splits (fixed by plant index, see split_of): training types {fopdt, sopdt, ipdt, hammerstein} are split 70/15/15
into train/validation/test; the held-out structures {underdamped, third} form the 'ood' set.
A small calibration set (the same pipeline plus a long best-known search) measures how far the labels are from the
optimum.

Resumable (JSONL checkpoint, one record per plant); deterministic per plant index.
Run from the repo root:   python experiments/m4a_dataset.py --n-train 1200 --n-ood 150 --workers 12
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plant_family import OOD_TYPES, TRAIN_TYPES, problem_from_spec, sample_specs, spec_to_dict
from src.tuners import NotApplicable
from src.tuners.features import extract_features, normalise_cost, normalise_gains
from src.tuners.optimizers import GA, PSO

OUT = "data/m4"
BUDGET = 400


def split_of(index, ptype):
    if ptype in OOD_TYPES:
        return "ood"
    r = (index * 2654435761 % 2 ** 32) / 2 ** 32          # deterministic pseudo-random in [0,1)
    return "train" if r < 0.70 else ("val" if r < 0.85 else "test")


def label_plant(spec_dict, budget=None):
    from src.plant_family import spec_from_dict
    spec = spec_from_dict(spec_dict)
    t0 = time.perf_counter()
    problem = problem_from_spec(spec)
    try:
        feat = extract_features(problem)
    except NotApplicable as e:
        return dict(ident=spec.ident, ptype=spec.ptype, status="no_features", message=str(e), spec=spec_dict)
    rec = []
    results = []
    for cls, seed in ((GA, 1000 + spec.ident), (PSO, 2000 + spec.ident)):
        p = problem_from_spec(spec)
        results.append(cls(recorder=rec).tune(p, budget=budget or BUDGET, seed=seed))
    best = min(results, key=lambda r: r.cost)
    arr = np.array(rec)
    z = np.stack([normalise_gains(dict(Kp=10 ** a[0], Ki=10 ** a[1], Kd=10 ** a[2]), feat.K_eff, feat.T_n) for a in arr])
    cost_n = normalise_cost(arr[:, 3], spec.y_ref, feat.T_n)
    z_best = normalise_gains(best.gains, feat.K_eff, feat.T_n)
    return dict(
        ident=spec.ident, ptype=spec.ptype, status="ok", split=split_of(spec.ident, spec.ptype), spec=spec_dict,
        features=feat.vec.round(6).tolist(), signature=feat.signature.round(6).tolist(),
        K_eff=feat.K_eff, T_n=feat.T_n, kind=feat.kind, model=feat.model,
        feat_evals=feat.cost_evals, feat_sim_seconds=feat.cost_sim_seconds,
        best_gains=best.gains, best_cost=best.cost, best_cost_norm=float(normalise_cost(best.cost, spec.y_ref, feat.T_n)),
        best_method=best.method, ga_cost=results[0].cost, pso_cost=results[1].cost,
        best_z=z_best.round(5).tolist(),
        eval_z=z.round(4).tolist(), eval_logcost=np.log10(cost_n).round(4).tolist(),
        seconds=time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--n-ood", type=int, default=150)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default=f"{OUT}/dataset.jsonl")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    specs = sample_specs(a.n_train, seed=a.seed, types=TRAIN_TYPES) + \
        sample_specs(a.n_ood, seed=a.seed, types=OOD_TYPES, offset=a.n_train)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["ident"] for l in open(a.out) if l.strip()}
    todo = [spec_to_dict(s) for s in specs if s.ident not in done]
    print(f"{len(done)} done, {len(todo)} to run on {a.workers} workers", flush=True)
    t0 = time.perf_counter()
    with Pool(a.workers) as pool, open(a.out, "a") as f:
        for k, rec in enumerate(pool.imap_unordered(label_plant, todo)):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (k + 1) % 20 == 0 or k + 1 == len(todo):
                el = time.perf_counter() - t0
                print(f"  {k + 1}/{len(todo)}  {el / 60:.1f} min  eta {(el / (k + 1)) * (len(todo) - k - 1) / 60:.1f} min",
                      flush=True)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
