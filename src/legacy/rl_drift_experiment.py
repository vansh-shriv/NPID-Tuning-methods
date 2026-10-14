"""
Drift Experiment — Phase 5, Section 5 Week 7 (Phase5_RL_Spec.md)

The centerpiece comparison of Phase 5:
  - Frozen N-PID (ML) controller: Random Forest predicts gains once at t=0
    using nominal (pre-drift) parameters, then holds them fixed for 20 seconds.
  - Adaptive N-PID (RL/TD3) controller: re-tunes gains every N=50 steps
    throughout the episode as R drifts from 1.0 to 2.0 Ohms.

Both run under the same conditions: omega_ref=20 rad/s, k_heat=0.05, 20s episode.

Also produces the windowed ITAE table (t in [0,5), [5,15), [15,20]).
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
from src.controllers import NPID
from src.plant import DCMotorParams, DriftingDCMotor
from src.metrics import compute_metrics

SEED = 42
OMEGA_REF    = 20.0
K_HEAT       = 0.05
EPISODE_LEN  = 20.0

# PSO gains (original project reference)
PSO_GAINS = {"Kp": 127.9988, "Ki": 681.7335, "Kd": 1.8942}


# ---------------------------------------------------------------------------
# Frozen ML controller under drift (baseline)
# ---------------------------------------------------------------------------

def run_frozen_ml_under_drift(omega_ref: float = OMEGA_REF,
                               k_heat: float = K_HEAT,
                               t_end: float = EPISODE_LEN) -> tuple:
    """
    Load the ML-predicted gains for nominal conditions, hold them fixed
    throughout a drift episode. Returns (time_arr, omega_arr, error_arr).

    The spec §7 defines "frozen ML" as: RF predicts gains once at t=0
    using the baseline (pre-drift) parameters, then those gains are held
    constant for the entire 20-second episode.
    """
    # Try to load the trained ML model
    ml_gains = None
    try:
        import joblib
        rf = joblib.load("models/legacy/npid_rf_model.pkl")
        # Feature vector: same as used in Week 3 training
        # [J, B, Tc, T_load, omega_ref] at nominal params
        X = np.array([[0.01, 0.1, 0.05, 0.0, omega_ref]])
        pred = rf.predict(X)[0]
        ml_gains = {"Kp": pred[0], "Ki": pred[1], "Kd": pred[2]}
        print(f"  ML-predicted gains: Kp={ml_gains['Kp']:.2f}, "
              f"Ki={ml_gains['Ki']:.2f}, Kd={ml_gains['Kd']:.2f}")
    except Exception as e:
        print(f"  [INFO] ML model not loaded ({e}), using PSO gains as frozen baseline.")
        ml_gains = PSO_GAINS

    Kp, Ki, Kd = ml_gains["Kp"], ml_gains["Ki"], ml_gains["Kd"]

    params = DCMotorParams()
    plant = DriftingDCMotor(k_heat=k_heat)
    n_steps = int(t_end / DT)
    time_arr  = np.zeros(n_steps)
    omega_arr = np.zeros(n_steps)
    error_arr = np.zeros(n_steps)

    state = [0.0, 0.0]
    ctrl_integral = 0.0
    prev_error = 0.0

    for k in range(n_steps):
        t = k * DT
        time_arr[k] = t
        i, omega = state
        omega_arr[k] = omega

        error = omega_ref - omega
        error_arr[k] = error
        de_dt = (error - prev_error) / DT

        # N-PID (frozen gains)
        alpha_e = NPID.alpha(error, 2)
        beta_e  = NPID.beta(error, 2)
        gamma_e = NPID.gamma(de_dt, 2)

        ctrl_integral += beta_e * error * DT
        u_raw = Kp * alpha_e + Ki * ctrl_integral + Kd * gamma_e * de_dt
        V = max(-params.Vmax, min(u_raw, params.Vmax))
        if V != u_raw:
            ctrl_integral -= beta_e * error * DT

        prev_error = error

        state = plant.step(t, state, V, 0.0, DT)

    return time_arr, omega_arr, error_arr, ml_gains


# ---------------------------------------------------------------------------
# RL adaptive controller under drift
# ---------------------------------------------------------------------------

def run_rl_under_drift(model_path: str,
                       omega_ref: float = OMEGA_REF,
                       k_heat: float = K_HEAT,
                       t_end: float = EPISODE_LEN) -> tuple:
    """
    Run the drift-trained RL agent adaptively, re-tuning gains every 50ms.
    Returns (time_arr, omega_arr, error_arr, kp_arr, ki_arr, kd_arr).
    """
    model = TD3.load(model_path)
    env = NPIDTuningEnv(
        drift=True,
        k_heat=k_heat,
        episode_length=t_end,
        omega_ref_range=(omega_ref, omega_ref),
        seed=SEED,
    )
    obs, _ = env.reset(seed=SEED)
    done = False

    times_rl, omegas_rl, errors_rl = [0.0], [0.0], [omega_ref]
    kps, kis, kds = [], [], []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        times_rl.append(info["t"])
        omegas_rl.append(info["omega"])
        errors_rl.append(omega_ref - info["omega"])
        kps.append(info["Kp"])
        kis.append(info["Ki"])
        kds.append(info["Kd"])

    return (np.array(times_rl), np.array(omegas_rl), np.array(errors_rl),
            np.array(kps), np.array(kis), np.array(kds))


# ---------------------------------------------------------------------------
# Windowed ITAE
# ---------------------------------------------------------------------------

def windowed_itae(time_arr, error_arr, windows=((0, 5), (5, 15), (15, 20))):
    """Compute ITAE over each time window. Returns list of (window, itae)."""
    results = []
    for t_lo, t_hi in windows:
        mask = (time_arr >= t_lo) & (time_arr < t_hi)
        if not np.any(mask):
            results.append(((t_lo, t_hi), np.nan))
            continue
        t_w = time_arr[mask]
        e_w = np.abs(error_arr[mask])
        itae = np.trapezoid(t_w * e_w, t_w)
        results.append(((t_lo, t_hi), itae))
    return results


# ---------------------------------------------------------------------------
# Run full centerpiece comparison
# ---------------------------------------------------------------------------

def run_centerpiece_comparison(
    drift_model_path: str = "models/legacy/td3_npid_drift",
    omega_ref: float = OMEGA_REF,
    k_heat: float = K_HEAT,
    t_end: float = EPISODE_LEN,
    save_dir: str = "results/legacy/figures",
):
    """
    Run and plot the centerpiece Week 7 comparison:
    frozen-ML vs adaptive-RL under drift.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("results/legacy/tables", exist_ok=True)

    print(f"\n{'='*65}")
    print("  WEEK 7 CENTERPIECE: Frozen-ML vs Adaptive-RL under Drift")
    print(f"  k_heat={k_heat}, episode={t_end}s, omega_ref={omega_ref} rad/s")
    print(f"  R(t) goes from {compute_R(1.0, k_heat, 0):.1f} to "
          f"{compute_R(1.0, k_heat, t_end):.2f} Ohms over {t_end}s")
    print(f"{'='*65}\n")

    # ── Frozen ML ────────────────────────────────────────────────────────────
    print("Running FROZEN ML controller (gains fixed at t=0 nominal params)...")
    t_ml, w_ml, e_ml, ml_gains = run_frozen_ml_under_drift(omega_ref, k_heat, t_end)
    itae_ml = windowed_itae(t_ml, e_ml)

    # ── Adaptive RL ───────────────────────────────────────────────────────────
    print("\nRunning ADAPTIVE RL controller (re-tunes every 50ms)...")
    t_rl, w_rl, e_rl, kps, kis, kds = run_rl_under_drift(
        drift_model_path, omega_ref, k_heat, t_end
    )
    itae_rl = windowed_itae(t_rl, e_rl)

    # ── Windowed ITAE table ───────────────────────────────────────────────────
    print(f"\n  Windowed ITAE comparison:")
    print(f"  {'Window':<16} {'Frozen ML':>12} {'Adaptive RL':>12} {'RL advantage':>14}")
    print("  " + "-"*57)
    for (wml, ml_v), (wrl, rl_v) in zip(itae_ml, itae_rl):
        lo, hi = wml
        if not (np.isnan(ml_v) or np.isnan(rl_v)):
            advantage = (ml_v - rl_v) / ml_v * 100
            adv_str = f"{advantage:+.1f}%"
        else:
            adv_str = "N/A"
        print(f"  t=[{lo:.0f},{hi:.0f})s:        {ml_v:>12.4f} {rl_v:>12.4f} {adv_str:>14}")

    total_ml = np.trapezoid(t_ml * np.abs(e_ml), t_ml)
    total_rl = np.trapezoid(t_rl * np.abs(e_rl), t_rl)
    print(f"  {'TOTAL ITAE':<16} {total_ml:>12.4f} {total_rl:>12.4f} "
          f"{(total_ml-total_rl)/total_ml*100:>+13.1f}%")

    # Save windowed ITAE to CSV
    with open("results/legacy/tables/drift_windowed_itae.csv", "w") as f:
        f.write("window_lo,window_hi,frozen_ml_itae,adaptive_rl_itae\n")
        for (wml, ml_v), (wrl, rl_v) in zip(itae_ml, itae_rl):
            f.write(f"{wml[0]},{wml[1]},{ml_v:.6f},{rl_v:.6f}\n")

    # ── Plot 1: |e(t)| over time (the centerpiece plot) ──────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Error plot
    ax = axes[0]
    ax.plot(t_ml, np.abs(e_ml), color="tab:red", linewidth=1.2, alpha=0.9,
            label=f"Frozen ML (gains fixed at t=0)")
    ax.plot(t_rl, np.abs(e_rl), color="tab:blue", linewidth=1.2, alpha=0.9,
            label=f"Adaptive RL (re-tunes every 50ms)")
    ax.axvline(5,  color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(15, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.text(2.5,  ax.get_ylim()[1]*0.95 if ax.get_ylim()[1]>0 else 1, "t∈[0,5)", ha="center", fontsize=9, color="gray")
    ax.text(10.0, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1]>0 else 1, "t∈[5,15)", ha="center", fontsize=9, color="gray")
    ax.text(17.5, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1]>0 else 1, "t∈[15,20]", ha="center", fontsize=9, color="gray")

    # Add R(t) annotation on secondary axis
    ax2 = ax.twinx()
    t_ann = np.linspace(0, t_end, 200)
    R_ann = compute_R(1.0, k_heat, t_ann)
    ax2.plot(t_ann, R_ann, color="tab:orange", linewidth=1.0, linestyle=":",
             alpha=0.7, label="R(t) [Ω]")
    ax2.set_ylabel("R(t) [Ω]", color="tab:orange", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("|Tracking error| (rad/s)")
    ax.set_title(f"Centerpiece: |e(t)| under Drift (k_heat={k_heat}, ω_ref={omega_ref} rad/s)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # RL gain trajectory
    ax3 = axes[1]
    t_gain = t_rl[1:len(kps)+1]  # align with gain decisions
    ax3.plot(t_gain, kps / 500.0, color="tab:blue",   linewidth=1.2, label="Kp/500 (norm)")
    ax3.plot(t_gain, kis / 15000.0, color="tab:orange", linewidth=1.2, label="Ki/15000 (norm)")
    ax3.plot(t_gain, kds / 5.0,   color="tab:green",  linewidth=1.2, label="Kd/5 (norm)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Normalised gain (0–1)")
    ax3.set_title("RL Agent Gain Trajectory under Drift")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    centerpiece_path = os.path.join(save_dir, "drift_error_over_time_comparison.png")
    plt.savefig(centerpiece_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Centerpiece plot saved: {centerpiece_path}")

    return {
        "ml": {"time": t_ml, "omega": w_ml, "error": e_ml, "total_itae": total_ml},
        "rl": {"time": t_rl, "omega": w_rl, "error": e_rl, "total_itae": total_rl,
               "kps": kps, "kis": kis, "kds": kds},
    }
