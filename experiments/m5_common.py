"""Shared helpers of the M5 experiments: data protocol (noisy bump-test features) and closed-loop regret evaluation."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plant_family import problem_from_spec, spec_from_dict
from src.tuners.base import GAIN_NAMES
from src.tuners.features import denormalise_gains
from src.tuners.ml import load_bundle, noisy_bundle

DATA = "data/m4/dataset.jsonl"
NOISY = "data/m4/features_noisy.npz"
OUT = "results/methods/M5"
FIG = f"{OUT}/figures"
WORKERS = 12


def load_main():
    """Primary protocol of M4: same plants and labels, features/normalisation from a *noisy* bump test."""
    return noisy_bundle(load_bundle(DATA), NOISY)


def _regret_job(args):
    spec_d, z, K_eff, T_n, label_cost = args
    p = problem_from_spec(spec_from_dict(spec_d))
    g = denormalise_gains(z, K_eff, T_n)
    lb, ub = p.bounds
    g = {k: float(np.clip(g[k], lo, hi)) for k, lo, hi in zip(GAIN_NAMES, lb, ub)}
    return p.itae(p.simulate(g, "npid", "step", count=False)) / label_cost


def regret(bundle, Z, pool):
    """ITAE(gains from z) / ITAE(label) on the clean step, one closed-loop simulation per plant."""
    jobs = [(bundle.specs[i], Z[i], bundle.K_eff[i], bundle.T_n[i], bundle.best_cost[i]) for i in range(len(bundle))]
    return np.array(pool.map(_regret_job, jobs, chunksize=4))


def summarize(ratio):
    r = np.asarray(ratio)
    return dict(median=float(np.median(r)), geo_mean=float(np.exp(np.mean(np.log(np.clip(r, 0.5, 100.0))))),
                p90=float(np.percentile(r, 90)), within1p5=float(np.mean(r <= 1.5)), within1p1=float(np.mean(r <= 1.1)),
                bad=float(np.mean(r >= 10.0)), n=len(r))
