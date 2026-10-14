import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # _legacy_root: run from the repo root
"""
Week 7 Runner — Phase 5 (Phase5_RL_Spec.md §5 Week 7)

Orchestrates:
  1. Drift-case fine-tuning from the static-case agent (300k steps, episode=20s)
  2. Centerpiece comparison: frozen-ML vs adaptive-RL under drift (k_heat=0.05)
  3. Windowed ITAE table (t in [0,5), [5,15), [15,20])
  4. Honest assessment of whether RL's adaptivity advantage showed up

Run: python legacy/run_week7.py [--skip-training] [--timesteps N]
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.legacy.train_rl import train
from src.legacy.rl_drift_experiment import run_centerpiece_comparison

DRIFT_MODEL  = "models/legacy/td3_npid_drift"
STATIC_MODEL = "models/legacy/td3_npid_static"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip drift training, go straight to comparison")
    parser.add_argument("--timesteps", type=int, default=300_000)
    args = parser.parse_args()

    # ── Step 1: Fine-tune for drift ──────────────────────────────────────────
    if not args.skip_training:
        print("\n[WEEK 7] Step 1/2 — Fine-tuning TD3 for drift case ...\n")
        print("  Starting from: models/legacy/td3_npid_static.zip")
        print("  drift=True, k_heat=0.05, episode_length=20.0s")
        print(f"  total_timesteps={args.timesteps:,}\n")

        train(
            drift=True,
            finetune_from=STATIC_MODEL,
            total_timesteps=args.timesteps,
            episode_length=20.0,
        )
    else:
        if not os.path.exists(DRIFT_MODEL + ".zip"):
            print(f"[ERROR] --skip-training set but drift model not found: {DRIFT_MODEL}.zip")
            sys.exit(1)
        print(f"[WEEK 7] --skip-training: using {DRIFT_MODEL}.zip")

    # ── Step 2: Centerpiece comparison ──────────────────────────────────────
    print("\n[WEEK 7] Step 2/2 — Running centerpiece drift comparison ...\n")
    results = run_centerpiece_comparison(
        drift_model_path=DRIFT_MODEL,
        omega_ref=20.0,
        k_heat=0.05,
        t_end=20.0,
    )

    # ── Honest assessment ────────────────────────────────────────────────────
    ml_itae = results["ml"]["total_itae"]
    rl_itae = results["rl"]["total_itae"]
    advantage = (ml_itae - rl_itae) / ml_itae * 100

    print("\n" + "="*65)
    print("  WEEK 7 HONEST ASSESSMENT")
    print("="*65)
    print(f"\n  Total ITAE (20s episode):")
    print(f"    Frozen ML:   {ml_itae:.4f}")
    print(f"    Adaptive RL: {rl_itae:.4f}")

    if advantage > 10:
        verdict = "ADVANTAGE CONFIRMED"
        explanation = (
            f"RL outperforms frozen ML by {advantage:.1f}% total ITAE.\n"
            f"  The agent successfully detected and compensated for R drift."
        )
    elif advantage > 0:
        verdict = "MARGINAL ADVANTAGE"
        explanation = (
            f"RL is {advantage:.1f}% better than frozen ML — modest improvement.\n"
            f"  Drift adaptation is present but small relative to baseline error."
        )
    else:
        verdict = "NO CLEAR ADVANTAGE"
        explanation = (
            f"RL ({rl_itae:.4f}) is {-advantage:.1f}% WORSE than frozen ML ({ml_itae:.4f}).\n"
            f"  Likely causes: insufficient drift training, or drift too slow\n"
            f"  relative to 20s episode length for the agent to adapt meaningfully."
        )

    print(f"\n  Verdict: {verdict}")
    print(f"  {explanation}")

    print(f"\n  Figures: results/legacy/figures/drift_error_over_time_comparison.png")
    print(f"  Table:   results/legacy/tables/drift_windowed_itae.csv")
    print("\n  [WEEK 7 COMPLETE]\n")
