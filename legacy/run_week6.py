import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # _legacy_root: run from the repo root
"""
Week 6 Runner — Phase 5 (Phase5_RL_Spec.md §5 Week 6)

Orchestrates:
  1. TD3 training on the static case (drift=False, 300k timesteps)
  2. Training reward curve plot
  3. Static-case evaluation vs. spec §1.4 reference table
  4. Action trajectory plot
  5. Static step-response comparison plot (RL vs PSO)

Run: python legacy/run_week6.py [--timesteps N] [--skip-training]
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.legacy.train_rl import train
from src.legacy.rl_diagnostics import (
    plot_training_reward_curve,
    run_static_evaluation,
    plot_static_comparison,
    plot_action_trajectory,
)

MODEL_PATH = "models/legacy/td3_npid_static"
LOG_PATH   = "results/legacy/tables/td3_static_training_log"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000,
                        help="Training timesteps (default 300000)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training and go straight to evaluation (if model exists)")
    args = parser.parse_args()

    # ── Step 1: Train ────────────────────────────────────────────────────────
    if not args.skip_training:
        print("\n[WEEK 6] Step 1/5 — Training TD3 (static case) ...\n")
        train(
            drift=False,
            episode_length=4.0,
            total_timesteps=args.timesteps,
        )
    else:
        if not os.path.exists(MODEL_PATH + ".zip"):
            print(f"[ERROR] --skip-training specified but model not found: {MODEL_PATH}.zip")
            sys.exit(1)
        print(f"[WEEK 6] --skip-training: using existing model {MODEL_PATH}.zip")

    # ── Step 2: Reward curve ─────────────────────────────────────────────────
    print("\n[WEEK 6] Step 2/5 — Plotting training reward curve ...")
    if os.path.exists(LOG_PATH + ".npz"):
        plot_training_reward_curve(
            LOG_PATH,
            save_path="results/legacy/figures/rl_training_reward_curve.png",
            tag="static",
        )
    else:
        print(f"  [skip] Log not found: {LOG_PATH}.npz")

    # ── Step 3: Evaluation table ─────────────────────────────────────────────
    print("\n[WEEK 6] Step 3/5 — Evaluating at standard benchmark scenario ...")
    run_static_evaluation(
        MODEL_PATH,
        omega_ref=20.0,
        test_speeds=[10.0, 20.0, 35.0],
    )

    # ── Step 4: Action trajectory ────────────────────────────────────────────
    print("\n[WEEK 6] Step 4/5 — Plotting action trajectory ...")
    plot_action_trajectory(
        MODEL_PATH,
        save_path="results/legacy/figures/rl_action_trajectory.png",
        omega_ref=20.0,
        drift=False,
        episode_length=4.0,
        tag="static",
    )

    # ── Step 5: Step-response comparison ────────────────────────────────────
    print("\n[WEEK 6] Step 5/5 — Plotting RL vs PSO step response ...")
    plot_static_comparison(
        MODEL_PATH,
        save_path="results/legacy/figures/rl_static_vs_pso_comparison.png",
        omega_ref=20.0,
    )

    print("\n" + "=" * 65)
    print("  WEEK 6 COMPLETE")
    print("  Trained model:   models/legacy/td3_npid_static.zip")
    print("  Figures:         results/legacy/figures/rl_*.png")
    print("  Training log:    results/legacy/tables/td3_static_training_log.npz")
    print("=" * 65)
    print()
