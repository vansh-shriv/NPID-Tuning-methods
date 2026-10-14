import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # _legacy_root: run from the repo root
"""
Week 8 Runner — Phase 5 (Phase5_RL_Spec.md §5 Week 8)

Orchestrates:
  1. RL static evaluation (for master table's static-case RL row)
  2. Generalization sweep at k_heat ∈ {0.02, 0.05, 0.10}
       k_heat=0.02 → slower drift (R: 1.0 → 1.4 Ω)
       k_heat=0.05 → training condition (R: 1.0 → 2.0 Ω)
       k_heat=0.10 → faster drift  (R: 1.0 → 3.0 Ω)
  3. Master 5-controller comparison table (static + drift, all controllers)

Run: python legacy/run_week8.py [--skip-static] [--k-heats 0.02 0.05 0.10]
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.legacy.week8_generalization import run_week8

DRIFT_MODEL  = "models/legacy/td3_npid_drift"
STATIC_MODEL = "models/legacy/td3_npid_static"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--k-heats", nargs="+", type=float, default=[0.02, 0.05, 0.10],
        help="Drift rate values to test (default: 0.02 0.05 0.10)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("  WEEK 8 — Generalization, Master Table, Final Assessment")
    print("=" * 65)

    results = run_week8(
        drift_model_path=DRIFT_MODEL,
        static_model_path=STATIC_MODEL,
        k_heat_values=args.k_heats,
        omega_ref=20.0,
        t_end=20.0,
    )

    # ── Final concluding assessment ──────────────────────────────────────────
    drift_results = results["drift_results"]
    rl_static     = results["rl_static"]

    print("\n" + "=" * 65)
    print("  WEEK 8 FINAL CONCLUDING ASSESSMENT")
    print("=" * 65)

    print("""
  Question: When is online RL adaptation worth its complexity vs offline ML?

  FINDINGS:
""")

    k_heat_values = sorted(drift_results.keys())
    for k in k_heat_values:
        adv = drift_results[k]["advantage_pct"]
        ml_itae = drift_results[k]["ml"]["total_itae"]
        rl_itae = drift_results[k]["rl"]["total_itae"]
        drift_label = (
            "slower (unseen)"  if k < 0.05 else
            "training rate"    if k == 0.05 else
            "faster (unseen)"
        )
        verdict = (
            "RL ADVANTAGE"          if adv > 10 else
            "MARGINAL RL ADVANTAGE" if adv > 0  else
            "NO RL ADVANTAGE"
        )
        print(f"  k_heat={k:.2f} ({drift_label}):  "
              f"ML ITAE={ml_itae:.4f}, RL ITAE={rl_itae:.4f} -> {verdict} ({adv:+.1f}%)")

    print("""
  CONCLUSION:
  -----------
  The offline ML (Random Forest) approach is strongly recommended when:
    - Plant parameters are known or measurable a priori.
    - The operating environment is stable (no significant thermal drift).
    - Inference latency requirements are strict (ML is ~1900× faster than PSO).

  Online RL (TD3) becomes worthwhile when:
    - Parameters drift significantly during operation (motor heating, wear).
    - No parameter estimation sensor is available.
    - The extra training and deployment complexity is acceptable.

  A negative or mixed result here is itself a valid finding: RL adaptation
  advantage depends heavily on drift rate relative to episode length and on
  sufficient exposure to the drift distribution during training.
  See phase5_rl_report.md for the full written assessment.
""")

    print("=" * 65)
    print("  WEEK 8 COMPLETE")
    print("  Generalization figure: results/legacy/figures/rl_generalization_sweep.png")
    print("  Master table:          results/legacy/tables/master_5way_comparison.csv")
    print("  Written report:        report/legacy/phase5_rl_report.md")
    print("=" * 65)
    print()
