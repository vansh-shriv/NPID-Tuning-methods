"""
M7 - Cross-method benchmark: all 18 methods of the review's taxonomy on the 5 benchmark plants, 4 scenarios.

Scenarios (docs/PROTOCOL.md): step | load (input disturbance) | noise (measurement noise, 10 noise seeds) | change
(the plant's actuator gain x2.5 at 1.95 H of a 5-window run with setpoint steps; static-gain methods keep the gains
they tuned on the nominal plant, the adaptive methods retune online). Every method tunes the N-PID; the same gains are
also applied to an L-PID (RQ5). Optimizers: 400 simulations, 5 seeds; GPO 60 simulations, 5 seeds; RL: 3 PPO models;
hybrid (+20-simulation refinement) and adaptive: 3 seeds; deterministic methods: 1 run.

Stages:  prepare (train the ML models -> models/m7/prepared.pkl) | run [plants...] | analyze

Run from the repo root:  python experiments/m7_benchmark.py prepare | run | analyze   (or python run_methods.py)
"""

import json
import os
import pickle
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.m5_common import DATA, load_main
from src.benchmark_plants import PLANT_NAMES
from src.tuners import TuningProblem
from src.tuners.adaptive import (AdaptiveConfig, AdaptiveMLTuner, AdaptivePolicy, ChangePlant, ClippedModel, FrozenPolicy, N_WINDOWS,
                                 run_loop, sampling_steps)
from src.tuners.base import GAIN_NAMES, Tuner
from src.tuners.benchmark import evaluate_gains
from src.tuners.classical import CohenCoon, ContinuousCycling, TimeDomainTuning, TrialAndError, ZieglerNicholsReactionCurve
from src.tuners.features import denormalise_gains, extract_features
from src.tuners.hybrid import ClassicalMLTuner, HybridModelTuner, ModelConfig, base_matrix, build_model
from src.tuners.ml import ANNPolicy, BayesOptTuner, load_bundle, make_model
from src.tuners.modelmatching import ModelMatching
from src.tuners.optimizers import ABC, GA, PSO, SA, RandomSearch
from src.tuners.rl import RLTuner
from src.tuners.selftuning import SelfTuningPolePlacement

warnings.filterwarnings("ignore")
OUT = "results/methods/M7"
FIG = f"{OUT}/figures"
PREP = "models/m7/prepared.pkl"
WORKERS = 12
CHANGE_GAIN, T_SWITCH = 2.5, 1.95

# id, name, family, seeds, budget
METHODS = [
    ("01", "Trial and error", "classical", (0,), None),
    ("02", "Ziegler-Nichols (reaction curve)", "classical", (0,), None),
    ("03", "Cohen-Coon", "classical", (0,), None),
    ("04", "Continuous cycling", "classical", (0,), None),
    ("05", "Time-domain (SIMC)", "classical", (0,), None),
    ("06", "Self-tuning (pole placement)", "intelligent", (0,), None),
    ("07", "Model matching", "intelligent", (0,), None),
    ("08", "PSO", "optimization", tuple(range(5)), 400),
    ("09", "GA", "optimization", tuple(range(5)), 400),
    ("10", "ABC", "optimization", tuple(range(5)), 400),
    ("11", "SA", "optimization", tuple(range(5)), 400),
    ("12", "Regression (SVR)", "ml", (0,), None),
    ("13", "Reinforcement learning (PPO)", "ml", (0, 1, 2), None),
    ("14", "Gaussian-process optimisation", "ml", tuple(range(5)), 60),
    ("15", "ANN (decision-focused)", "ml", (0,), None),
    ("16", "GA-enhanced ML", "hybrid", (0,), None),
    ("17", "Hybrid ML + classical (+20 sims)", "hybrid", (0, 1, 2), None),
    ("18", "Adaptive ML", "adaptive", (0, 1, 2), None),
    ("R1", "Random search (reference)", "reference", tuple(range(5)), 400),
    ("R2", "Constant normalised gains (reference)", "reference", (0,), None),
]
INFO = {m[0]: m for m in METHODS}


class LazyRL(Tuner):
    """PPO session tuner that loads its model in the worker (SB3 models are loaded from disk, not pickled)."""
    name, family, stochastic = "Reinforcement learning (PPO)", "ml", True

    def __init__(self, z0, lo, hi):
        self.z0, self.lo, self.hi = z0, lo, hi

    def _tune(self, problem, budget, seed):
        from stable_baselines3 import PPO
        t = RLTuner(PPO.load(f"models/m4/ppo_seed{seed}"), self.z0, self.lo, self.hi, n_sims=8)
        return t._tune(problem, None, 0)


# ----------------------------------------------------------------------------- model preparation
def prepare():
    os.makedirs(os.path.dirname(PREP), exist_ok=True)
    b = load_main()                                         # noisy-feature protocol (primary)
    tr = b.select(b.split == "train")
    bc = load_bundle(DATA)
    trc = bc.select(bc.split == "train")
    X, Y = tr.X("engineered"), tr.best_z
    svr = make_model("svr").fit(X, Y)
    svr_res = make_model("svr").fit(X, Y - base_matrix(X, "cohen-coon"))
    import torch
    torch.set_num_threads(4)
    ann = ANNPolicy(features="signature", loss="combined", seed=0).fit(tr)
    cfgs = pd.read_csv("results/methods/M5/M5a_configs.csv")
    row = cfgs[cfgs.method == "GA"].sort_values("cv_fitness").iloc[0]
    ga_cfg = ModelConfig(row.model, json.loads(row.params), tuple(json.loads(row["mask"])))
    ga_model = build_model(ga_cfg).fit(X, Y)
    prep = dict(svr=svr, svr_res=svr_res, ann=ann, ga_model=ga_model, ga_desc=ga_cfg.describe(),
                svr_clean=make_model("svr").fit(trc.X("engineered"), trc.best_z), X_train_clean=trc.X_eng,
                z0=np.median(trc.best_z, axis=0), z_lo_hi=np.percentile(np.vstack(trc.eval_z), [0.5, 99.5], axis=0),
                z_const=np.median(Y, axis=0))
    with open(PREP, "wb") as f:
        pickle.dump(prep, f)
    print("prepared; GA-selected model:", prep["ga_desc"], flush=True)


_PREP = {}


def _init(prep):
    _PREP.update(prep)


def make_tuner(mid):
    P = _PREP
    if mid == "01": return TrialAndError()
    if mid == "02": return ZieglerNicholsReactionCurve()
    if mid == "03": return CohenCoon()
    if mid == "04": return ContinuousCycling("zn", "oscillation", fallback=False)
    if mid == "05": return TimeDomainTuning("simc")
    if mid == "06": return SelfTuningPolePlacement()
    if mid == "07": return ModelMatching("frequency")
    if mid == "08": return PSO()
    if mid == "09": return GA()
    if mid == "10": return ABC()
    if mid == "11": return SA()
    if mid == "12": return HybridModelTuner(P["svr"], "Regression (SVR)", family="ml")
    if mid == "13": return LazyRL(P["z0"], *P["z_lo_hi"])
    if mid == "14": return BayesOptTuner()
    if mid == "15": return HybridModelTuner(P["ann"], "ANN (decision-focused)", features="signature", family="ml")
    if mid == "16": return HybridModelTuner(P["ga_model"], "GA-enhanced ML")
    if mid == "17": return ClassicalMLTuner(P["svr_res"], "cohen-coon", 20)
    if mid == "18": return AdaptiveMLTuner(P["svr"], ClippedModel(P["svr_clean"], P["X_train_clean"]))
    if mid == "R1": return RandomSearch()
    if mid == "R2": return ClassicalMLTuner(None, "constant:" + ",".join(f"{v:.6f}" for v in P["z_const"]), 0)
    raise KeyError(mid)


# ----------------------------------------------------------------------------- scenario 4 (plant change)
def change_windows(problem, gains, policy_kind, seed, svr, svr_est_model):
    """5-window run with the actuator gain x2.5 at 1.95 H. policy_kind: 'frozen' | 'ml' | 'pp'. Returns window metrics or None."""
    H = problem.t_end
    f = extract_features(problem, noise_sigma=problem.noise_sigma, seed=seed)
    m = sampling_steps(f.T_n, f.model["theta"], problem.dt, 10.0)
    factory = lambda: ChangePlant(problem.plant_factory(), T_SWITCH * H, gain=CHANGE_GAIN)
    cfg = AdaptiveConfig()
    if policy_kind == "frozen":
        pol = FrozenPolicy()
    elif policy_kind == "ml":
        pol = AdaptivePolicy(cfg, svr_est_model)
    else:
        pol = AdaptivePolicy(replace(cfg, mode="pp"))
    res = run_loop(problem, factory, H, pol, gains, m, True, seed, f.kind, f.T_n)
    return None if res.failed else res.windows


def _initial_svr_gains(problem, seed):
    t = HybridModelTuner(_PREP["svr"], "svr").tune(problem, seed=seed)
    return t.gains if t.success else None


def _job(args):
    mid, plant, seed = args
    _, name, family, _, budget = INFO[mid]
    tuner = make_tuner(mid)
    problem = TuningProblem.from_scenario(plant)
    res = tuner.tune(problem, budget=budget, seed=seed)
    base = dict(id=mid, method=name, family=family, plant=plant, seed=seed, evals=res.evals, sim_seconds=res.sim_seconds,
                wall_time=res.wall_time, success=res.success, message=res.message,
                **{f"gain_{k}": v for k, v in res.gains.items()})
    rows = []
    ref = TuningProblem.from_scenario(plant)
    if res.success:
        for kind in ("npid", "lpid"):
            ev = evaluate_gains(ref, res.gains, kind)
            for scen in ("step", "load", "noise"):
                if scen in ev:
                    rows.append(dict(base, kind=kind, scenario=scen, **ev[scen]))
        variants = [("frozen", res.gains)]
    else:
        rows.append(dict(base, kind="npid", scenario="step", ITAE=np.nan))
        variants = []
    # plant change: static gains frozen; method 18 and the extra row 06b run online from the SVR bump-test gains
    est_model = ClippedModel(_PREP["svr_clean"], _PREP["X_train_clean"])
    if mid == "18":
        g0 = _initial_svr_gains(ref, seed)
        variants = [("ml", g0)] if g0 else []
    for pk, g in variants:
        w = change_windows(TuningProblem.from_scenario(plant), g, pk, seed, _PREP["svr"], est_model)
        if w is None:
            rows.append(dict(base, kind="npid", scenario="change", ITAE=np.nan, diverged=True))
        else:
            rows.append(dict(base, kind="npid", scenario="change", ITAE=float(np.mean([w[3]["itae"], w[4]["itae"]])), itae_first=w[2]["itae"],
                             itae_w3=w[3]["itae"], itae_w4=w[4]["itae"], overshoot_pct=w[4]["overshoot_pct"], diverged=False))
    if mid == "06":                                                    # extra row: the online variant of method 6
        g0 = _initial_svr_gains(ref, seed)
        if g0:
            w = change_windows(TuningProblem.from_scenario(plant), g0, "pp", seed, _PREP["svr"], est_model)
            rows.append(dict(base, id="06b", method="Self-tuning online (pole placement)", family="adaptive", kind="npid", scenario="change",
                             ITAE=np.nan if w is None else float(np.mean([w[3]["itae"], w[4]["itae"]])),
                             itae_first=np.nan if w is None else w[2]["itae"], diverged=w is None))
    return rows


def run(plants=None):
    os.makedirs(OUT, exist_ok=True)
    plants = plants or PLANT_NAMES
    with open(PREP, "rb") as f:
        prep = pickle.load(f)
    jobs = [(m[0], p, s) for p in plants for m in METHODS for s in m[3]]
    # long jobs first
    jobs.sort(key=lambda j: -(INFO[j[0]][4] or 20))
    print(len(jobs), "jobs", flush=True)
    rows = []
    with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(prep,)) as ex:
        for i, r in enumerate(ex.map(_job, jobs, chunksize=1)):
            rows += r
            if (i + 1) % 25 == 0:
                print(i + 1, "done", flush=True)
                pd.DataFrame(rows).to_csv(f"{OUT}/M7_benchmark_partial.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/M7_benchmark.csv", index=False)
    if os.path.exists(f"{OUT}/M7_benchmark_partial.csv"):
        os.remove(f"{OUT}/M7_benchmark_partial.csv")          # checkpoint of an unfinished run, superseded
    print("saved", len(df), flush=True)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "run"
    if stage == "prepare":
        prepare()
    elif stage == "run":
        run(sys.argv[2:] or None)
    else:
        from experiments.m7_analyze import analyze
        analyze()
