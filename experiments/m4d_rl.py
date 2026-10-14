"""
M4d - Reinforcement learning as a learned local search (method 13).

train    PPO (3 seeds) on tuning sessions over the training plants (src/tuners/rl.py): 8 simulations per session
eval     on val / test / ood plants: RL vs random local search vs the constant start (same simulation budgets), and
         the budget (4 / 8 / 16 simulations) and step-size ablations of the random baseline
analyze  tables and figures

Run from the repo root:  python experiments/m4d_rl.py train [timesteps] | eval | analyze
"""

import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plant_family import problem_from_spec, spec_from_dict
from src.tuners.benchmark import evaluate_gains
from src.tuners.ml import load_bundle
from src.tuners.rl import ConstantStart, RandomLocalSearch, RLTuner, make_env

warnings.filterwarnings("ignore")
DATA = "data/m4/dataset.jsonl"
OUT = "results/methods/M4"
FIG = f"{OUT}/figures"
MODELS = "models/m4"
SEEDS = (0, 1, 2)
N_SIMS = 8


def _train_seed(args):
    seed, timesteps = args
    # NOTE: the models in models/m4 were trained WITHOUT the next two lines (multi-threaded torch, 83 min per seed);
    # single-threaded learners give the same algorithm much faster and are recommended for re-training.
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    envs = SubprocVecEnv([make_env(DATA, "train", seed=1000 * seed + i, n_sims=N_SIMS) for i in range(4)])
    model = PPO("MlpPolicy", envs, n_steps=64, batch_size=128, n_epochs=6, learning_rate=3e-4, gamma=0.99, ent_coef=0.003,
                seed=seed, verbose=0, policy_kwargs=dict(net_arch=[128, 128]))
    t0 = time.time()
    model.learn(total_timesteps=timesteps, progress_bar=False, log_interval=None)
    print(f"seed {seed} finished", flush=True)
    os.makedirs(MODELS, exist_ok=True)
    model.save(f"{MODELS}/ppo_seed{seed}")
    envs.close()
    return seed, time.time() - t0


def stage_train(timesteps=120_000):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=3) as ex:
        for seed, sec in ex.map(_train_seed, [(s, timesteps) for s in SEEDS]):
            print(f"seed {seed}: {sec / 60:.1f} min", flush=True)


def _eval_job(args):
    spec_d, ident, ptype, split, label_gains, label_cost, z0, lo, hi, which, n_sims, seed = args
    spec = spec_from_dict(spec_d)
    if which == "const":
        t = ConstantStart(z0, lo, hi)
    elif which == "random":
        t = RandomLocalSearch(z0, lo, hi, n_sims=n_sims)
    else:
        from stable_baselines3 import PPO
        t = RLTuner(PPO.load(f"{MODELS}/ppo_seed{which}"), z0, lo, hi, n_sims=n_sims)
    p = problem_from_spec(spec)
    r = t.tune(p, seed=seed)
    ref = problem_from_spec(spec)
    ev = evaluate_gains(ref, r.gains, "npid", noise_seeds=tuple(range(4)))
    return dict(method=t.name if which in ("const", "random") else f"RL seed {which}", which=str(which), n_sims=n_sims,
                ident=ident, ptype=ptype, split=split, step=ev["step"]["ITAE"], noise=ev["noise"]["ITAE"],
                load=ev["load"]["ITAE"], label=label_cost, ratio=ev["step"]["ITAE"] / label_cost, evals=r.evals)


def stage_eval():
    import json
    from concurrent.futures import ProcessPoolExecutor
    b = load_bundle(DATA)
    labels = {int(r["ident"]): r["best_gains"] for r in (json.loads(l) for l in open(DATA)) if r.get("status") == "ok"}
    tr = b.select(b.split == "train")
    z0 = np.median(tr.best_z, axis=0)
    lo, hi = np.percentile(np.vstack(tr.eval_z), [0.5, 99.5], axis=0)
    ev = b.select(np.isin(b.split, ["val", "test", "ood"]))
    jobs = []
    for i in range(len(ev)):
        base = (ev.specs[i], int(ev.ident[i]), str(ev.ptype[i]), str(ev.split[i]), labels[int(ev.ident[i])], float(ev.best_cost[i]), z0, lo, hi)
        jobs.append(base + ("const", 1, 0))
        for n in (4, 8, 16):
            jobs.append(base + ("random", n, 11))
        for s in SEEDS:
            jobs.append(base + (s, 8, 0))
    with ProcessPoolExecutor(max_workers=12) as ex:
        rows = list(ex.map(_eval_job, jobs, chunksize=6))
    pd.DataFrame(rows).to_csv(f"{OUT}/M4_rl_eval.csv", index=False)
    print("rl eval done", len(rows), flush=True)


def stage_analyze():
    import matplotlib.pyplot as plt

    from src.tuners.plots import _save
    d = pd.read_csv(f"{OUT}/M4_rl_eval.csv")
    d["variant"] = np.where(d.which.isin(["0", "1", "2"]), "RL (PPO) n=8", np.where(d.which == "const", "Constant start (1 sim)",
                            "Random local search n=" + d.n_sims.astype(str)))
    t = d.groupby(["variant", "split"]).ratio.agg(["median", lambda x: np.percentile(x, 90), lambda x: np.mean(x <= 1.5)]).unstack("split")
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    t.to_csv(f"{OUT}/M4_rl_summary.csv")
    seeds = d[d.variant == "RL (PPO) n=8"].groupby(["which", "split"]).ratio.median().unstack("split")
    seeds.to_csv(f"{OUT}/M4_rl_per_seed.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    order = ["Constant start (1 sim)", "Random local search n=4", "Random local search n=8", "Random local search n=16", "RL (PPO) n=8"]
    data = [d[(d.variant == v) & (d.split != "ood")].ratio.values for v in order]
    ax.boxplot(data, tick_labels=[o.replace(" ", "\n", 1) for o in order], showfliers=False)
    ax.set_yscale("log"); ax.axhline(1, color="k", ls="--", lw=0.8); ax.grid(alpha=0.3, which="both", axis="y")
    ax.set_ylabel("ITAE / label (val+test plants)")
    _save(fig, f"{FIG}/M4_rl_regret.png")
    print("analysis written")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "train"
    if stage == "train":
        stage_train(int(sys.argv[2]) if len(sys.argv) > 2 else 120_000)
    elif stage == "eval":
        stage_eval()
    elif stage == "analyze":
        stage_analyze()
