"""
Week 8 Generalization & Master Table — Phase 5 (Phase5_RL_Spec.md §5 Week 8)

Tests:
  1. Drift-trained RL agent at k_heat values not used in training:
       k_heat = 0.02 (slower drift) — R goes from 1.0 to 1.4 Ω over 20s
       k_heat = 0.10 (faster drift) — R goes from 1.0 to 3.0 Ω over 20s
     (Training used k_heat = 0.05)

  2. Assembles the master 5-controller comparison table (static + drift conditions).

  3. Writes results/legacy/tables/master_5way_comparison.csv
"""

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import TD3
from src.legacy.rl_env import NPIDTuningEnv, DT, ACTION_HOLD_N
from src.legacy.rl_drift import compute_R
from src.legacy.rl_drift_experiment import (
    run_frozen_ml_under_drift,
    run_rl_under_drift,
    windowed_itae,
    OMEGA_REF,
)
from src.controllers import NPID
from src.plant import DCMotorParams
from src.simulate import run_simulation
from src.metrics import compute_metrics

SEED = 42

# ---------------------------------------------------------------------------
# Static-case reference (from spec §1.4 and master_baseline_comparison.csv)
# ---------------------------------------------------------------------------
STATIC_REF = {
    "L-PID (ZN)":   {"rise": 0.1350, "over": 0.22,  "settle": 0.1990, "itae": 0.1631},
    "N-PID (ZN)":   {"rise": 0.1340, "over": 8.07,  "settle": 0.7510, "itae": 2.2720},
    "N-PID (PSO)":  {"rise": 0.1350, "over": 7.17,  "settle": 0.6480, "itae": 0.3466},
    "N-PID (ML)":   {"rise": 0.1350, "over": 7.20,  "settle": 0.6460, "itae": 0.3894},
}

PSO_GAINS = {"Kp": 127.9988, "Ki": 681.7335, "Kd": 1.8942}


# ---------------------------------------------------------------------------
# Helper: evaluate RL agent at a specific k_heat under drift
# ---------------------------------------------------------------------------

def evaluate_rl_at_kheat(drift_model_path: str,
                          k_heat: float,
                          omega_ref: float = OMEGA_REF,
                          t_end: float = 20.0) -> dict:
    """
    Run drift-trained RL agent at a given k_heat.
    Returns dict with time array, error array, total ITAE, and windowed ITAE.
    """
    t_rl, w_rl, e_rl, kps, kis, kds = run_rl_under_drift(
        drift_model_path, omega_ref=omega_ref, k_heat=k_heat, t_end=t_end
    )
    total_itae = np.trapezoid(t_rl * np.abs(e_rl), t_rl)
    w_itae = windowed_itae(t_rl, e_rl, windows=((0, 5), (5, 15), (15, 20)))
    return {
        "time": t_rl,
        "omega": w_rl,
        "error": e_rl,
        "kps": kps, "kis": kis, "kds": kds,
        "total_itae": total_itae,
        "windowed_itae": w_itae,
    }


def evaluate_frozen_at_kheat(k_heat: float,
                              omega_ref: float = OMEGA_REF,
                              t_end: float = 20.0) -> dict:
    """
    Run frozen-ML (PSO gains as fallback) at a given k_heat.
    Returns dict with time array, error array, total ITAE.
    """
    t_ml, w_ml, e_ml, gains = run_frozen_ml_under_drift(
        omega_ref=omega_ref, k_heat=k_heat, t_end=t_end
    )
    total_itae = np.trapezoid(t_ml * np.abs(e_ml), t_ml)
    w_itae = windowed_itae(t_ml, e_ml, windows=((0, 5), (5, 15), (15, 20)))
    return {
        "time": t_ml,
        "omega": w_ml,
        "error": e_ml,
        "gains": gains,
        "total_itae": total_itae,
        "windowed_itae": w_itae,
    }


# ---------------------------------------------------------------------------
# Static-case RL evaluation (from trained static model)
# ---------------------------------------------------------------------------

def evaluate_rl_static(static_model_path: str,
                        omega_ref: float = OMEGA_REF) -> dict:
    """
    Evaluate the static-case RL agent at omega_ref=20 rad/s (no drift).
    Returns metrics dict.
    """
    model = TD3.load(static_model_path)
    env = NPIDTuningEnv(
        drift=False,
        episode_length=4.0,
        omega_ref_range=(omega_ref, omega_ref),
        seed=SEED,
    )
    obs, _ = env.reset(seed=SEED)
    done = False
    times, omegas = [0.0], [0.0]
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        times.append(info["t"])
        omegas.append(info["omega"])

    metrics = compute_metrics(np.array(times), omega_ref, np.array(omegas))
    return metrics


# ---------------------------------------------------------------------------
# Generalization sweep: plot error over time for all k_heat values
# ---------------------------------------------------------------------------

def run_generalization_sweep(drift_model_path: str,
                              k_heat_values: list = None,
                              omega_ref: float = OMEGA_REF,
                              t_end: float = 20.0,
                              save_dir: str = "results/legacy/figures") -> dict:
    """
    Run RL agent + frozen ML at three k_heat values (0.02, 0.05, 0.10).
    Produces a 3-panel comparison figure.
    Returns all results keyed by k_heat.
    """
    if k_heat_values is None:
        k_heat_values = [0.02, 0.05, 0.10]

    os.makedirs(save_dir, exist_ok=True)

    all_results = {}
    print("\n" + "=" * 65)
    print("  WEEK 8 GENERALIZATION SWEEP")
    print(f"  k_heat values tested: {k_heat_values}")
    print(f"  (Training used k_heat=0.05)")
    print("=" * 65)

    fig, axes = plt.subplots(len(k_heat_values), 1,
                              figsize=(12, 4 * len(k_heat_values)),
                              sharex=False)
    if len(k_heat_values) == 1:
        axes = [axes]

    for idx, k in enumerate(k_heat_values):
        R_final = compute_R(1.0, k, t_end)
        print(f"\n  k_heat={k:.2f}: R goes {compute_R(1.0,k,0):.1f} -> {R_final:.2f} Ohm")

        print(f"    Running frozen ML ...")
        ml_res = evaluate_frozen_at_kheat(k, omega_ref, t_end)

        print(f"    Running adaptive RL ...")
        rl_res = evaluate_rl_at_kheat(drift_model_path, k, omega_ref, t_end)

        advantage = (ml_res["total_itae"] - rl_res["total_itae"]) / ml_res["total_itae"] * 100
        print(f"    Total ITAE — ML: {ml_res['total_itae']:.4f}, "
              f"RL: {rl_res['total_itae']:.4f}, "
              f"RL advantage: {advantage:+.1f}%")

        # Windowed ITAE
        print(f"    Windowed ITAE:")
        for (wml, ml_v), (wrl, rl_v) in zip(ml_res["windowed_itae"], rl_res["windowed_itae"]):
            lo, hi = wml
            adv = (ml_v - rl_v) / ml_v * 100 if ml_v > 0 else 0.0
            print(f"      t=[{lo:.0f},{hi:.0f}): ML={ml_v:.4f}, RL={rl_v:.4f} ({adv:+.1f}%)")

        all_results[k] = {"ml": ml_res, "rl": rl_res, "advantage_pct": advantage}

        # Plot this panel
        ax = axes[idx]
        ax.plot(ml_res["time"], np.abs(ml_res["error"]),
                color="tab:red", linewidth=1.2, alpha=0.85,
                label="Frozen ML (gains fixed at t=0)")
        ax.plot(rl_res["time"], np.abs(rl_res["error"]),
                color="tab:blue", linewidth=1.2, alpha=0.85,
                label="Adaptive RL (re-tunes every 50ms)")

        # R(t) on secondary axis
        ax2 = ax.twinx()
        t_ann = np.linspace(0, t_end, 300)
        R_ann = compute_R(1.0, k, t_ann)
        ax2.plot(t_ann, R_ann, color="tab:orange", linewidth=0.9, linestyle=":",
                 alpha=0.7, label="R(t) [Ohm]")
        ax2.set_ylabel("R(t) [Ohm]", color="tab:orange", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        drift_label = ("slower (unseen)"  if k < 0.05 else
                       "training rate"    if k == 0.05 else
                       "faster (unseen)")
        ax.set_title(f"k_heat={k:.2f} ({drift_label} drift) — "
                     f"R: {compute_R(1.0,k,0):.1f}->{R_final:.2f} Ohm | "
                     f"RL advantage: {advantage:+.1f}%")
        ax.set_ylabel("|Tracking error| (rad/s)")
        ax.set_xlabel("Time (s)")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Week 8 Generalization: Frozen ML vs Adaptive RL across Drift Rates\n"
        f"(omega_ref={omega_ref} rad/s, 20s episodes)", fontsize=11
    )
    plt.tight_layout()
    save_path = os.path.join(save_dir, "rl_generalization_sweep.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Generalization sweep figure saved: {save_path}")

    return all_results


# ---------------------------------------------------------------------------
# Master 5-controller comparison table
# ---------------------------------------------------------------------------

def build_master_table(drift_results: dict,
                       rl_static_metrics: dict,
                       k_heat_train: float = 0.05,
                       omega_ref: float = OMEGA_REF,
                       t_end: float = 20.0,
                       save_dir: str = "results/legacy/tables") -> str:
    """
    Assemble the master 5-controller table (static + drift scenario).
    Writes master_5way_comparison.csv.

    Drift scenario: all frozen controllers share the same frozen-ML result
    (they are all static-gains controllers, equally vulnerable to drift).
    Only the RL row shows the adaptive result.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Static-case metrics (from STATIC_REF for the 4 original controllers)
    # RL static metrics come from actual evaluation
    rl_static = rl_static_metrics

    # Drift-case: frozen ML result (used as proxy for ALL frozen controllers)
    ml_drift = drift_results[k_heat_train]["ml"]
    rl_drift  = drift_results[k_heat_train]["rl"]

    # For the "frozen" row under drift: same trajectory for L-PID/ZN, N-PID/ZN,
    # N-PID/PSO, N-PID/ML (they are all equally static — see spec §8.2 note).
    # We use the frozen-ML ITAE (PSO gains) as the representative frozen number.
    frozen_drift_itae = ml_drift["total_itae"]
    rl_drift_itae = rl_drift["total_itae"]

    rows = []

    # Header
    rows.append(
        "Controller,"
        "Static_Rise(s),Static_Over(%),Static_Settle(s),Static_ITAE,"
        "Drift20s_ITAE,Drift_Adapts"
    )

    for name, s in STATIC_REF.items():
        rows.append(
            f"{name},"
            f"{s['rise']:.4f},{s['over']:.2f},{s['settle']:.4f},{s['itae']:.4f},"
            f"{frozen_drift_itae:.4f},No"
        )

    # RL row — static from evaluation, drift from rl_drift
    rl_rise   = rl_static.get("rise_time", float("nan"))
    rl_over   = rl_static.get("overshoot_pct", float("nan"))
    rl_settle = rl_static.get("settling_time", float("nan"))
    rl_itae_s = rl_static.get("ITAE", float("nan"))

    rows.append(
        f"N-PID (RL/TD3),"
        f"{rl_rise:.4f},{rl_over:.2f},{rl_settle:.4f},{rl_itae_s:.4f},"
        f"{rl_drift_itae:.4f},Yes"
    )

    csv_path = os.path.join(save_dir, "master_5way_comparison.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(rows) + "\n")

    # Pretty-print the table
    print("\n" + "=" * 80)
    print("  MASTER 5-CONTROLLER COMPARISON TABLE")
    print("=" * 80)
    print(f"\n  Static benchmark: omega_ref={omega_ref} rad/s, no noise, nominal params")
    print(f"  Drift scenario:   omega_ref={omega_ref} rad/s, k_heat={k_heat_train}, "
          f"20s episode, R: 1.0->{compute_R(1.0,k_heat_train,t_end):.2f} Ohm\n")

    hdr = (f"  {'Controller':<22} {'Rise(s)':>8} {'Over(%)':>8} {'Settle(s)':>10} "
           f"{'ITAE(static)':>13} {'ITAE(drift)':>12} {'Adapts':>7}")
    print(hdr)
    print("  " + "-" * 82)

    for name, s in STATIC_REF.items():
        print(f"  {name:<22} {s['rise']:>8.4f} {s['over']:>8.2f} {s['settle']:>10.4f} "
              f"{s['itae']:>13.4f} {frozen_drift_itae:>12.4f} {'No':>7}")

    print("  " + "-" * 82)
    print(f"  {'N-PID (RL/TD3)':<22} {rl_rise:>8.4f} {rl_over:>8.2f} {rl_settle:>10.4f} "
          f"{rl_itae_s:>13.4f} {rl_drift_itae:>12.4f} {'Yes':>7}  <- NEW")
    print()

    advantage = (frozen_drift_itae - rl_drift_itae) / frozen_drift_itae * 100
    print(f"  Under drift, RL vs frozen-ML ITAE advantage: {advantage:+.1f}%")
    print(f"\n  Saved: {csv_path}")
    print("=" * 80)

    return csv_path


# ---------------------------------------------------------------------------
# Main entry point (called from legacy/run_week8.py)
# ---------------------------------------------------------------------------

def run_week8(drift_model_path: str = "models/legacy/td3_npid_drift",
              static_model_path: str = "models/legacy/td3_npid_static",
              k_heat_values: list = None,
              omega_ref: float = OMEGA_REF,
              t_end: float = 20.0):
    """
    Full Week 8 pipeline:
      1. Evaluate RL static agent (for master table)
      2. Generalization sweep across k_heat values
      3. Build master 5-way comparison table
    """
    if k_heat_values is None:
        k_heat_values = [0.02, 0.05, 0.10]

    os.makedirs("results/legacy/figures", exist_ok=True)
    os.makedirs("results/legacy/tables", exist_ok=True)

    # Step 1: Static-case RL metrics
    print("\n[WEEK 8] Step 1/3 — Evaluating RL agent on static benchmark ...")
    if os.path.exists(static_model_path + ".zip"):
        rl_static = evaluate_rl_static(static_model_path, omega_ref)
        print(f"  RL static: ITAE={rl_static['ITAE']:.4f}, "
              f"rise={rl_static['rise_time']:.4f}s, "
              f"over={rl_static['overshoot_pct']:.2f}%")
    else:
        print(f"  [WARN] Static model not found at {static_model_path}.zip")
        print(f"         Using placeholder zeros for RL static metrics.")
        rl_static = {"rise_time": float("nan"), "overshoot_pct": float("nan"),
                     "settling_time": float("nan"), "ITAE": float("nan")}

    # Step 2: Generalization sweep
    print("\n[WEEK 8] Step 2/3 — Generalization sweep across drift rates ...")
    drift_results = run_generalization_sweep(
        drift_model_path,
        k_heat_values=k_heat_values,
        omega_ref=omega_ref,
        t_end=t_end,
    )

    # Step 3: Master 5-way table
    print("\n[WEEK 8] Step 3/3 — Building master 5-controller comparison table ...")
    csv_path = build_master_table(
        drift_results,
        rl_static,
        k_heat_train=0.05,
        omega_ref=omega_ref,
        t_end=t_end,
    )

    return {
        "rl_static": rl_static,
        "drift_results": drift_results,
        "master_table_path": csv_path,
    }
