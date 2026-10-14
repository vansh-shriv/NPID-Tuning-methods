"""
RL Diagnostics — Phase 5, Section 6 (Phase5_RL_Spec.md)

Three plots required by spec:
  1. Training reward curve (ep reward + rolling mean over last 50 episodes).
  2. Action trajectory (Kp, Ki, Kd vs time for one evaluation episode).
  3. Static-case comparison plot (RL agent vs PSO step response).

Also contains run_static_evaluation() — runs the trained agent at omega_ref=20
and returns metrics for the comparison table (Week 6 deliverable).

Usage (standalone):
  python src/legacy/rl_diagnostics.py [--model models/legacy/td3_npid_static]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import TD3
from src.legacy.rl_env import NPIDTuningEnv, gains_to_action, DT, ACTION_HOLD_N
from src.metrics import compute_metrics
from src.simulate import run_simulation
from src.controllers import NPID
from src.plant import DCMotorParams

# Reference PSO gains (spec §1.4 row 3)
PSO_GAINS = {"Kp": 127.9988, "Ki": 681.7335, "Kd": 1.8942}

SEED = 42


# ---------------------------------------------------------------------------
# 1. Training reward curve
# ---------------------------------------------------------------------------

def plot_training_reward_curve(
    log_path: str,
    save_path: str = "results/legacy/figures/rl_training_reward_curve.png",
    rolling_window: int = 50,
    tag: str = "static",
):
    """
    Load training log (npz) saved by RewardLoggerCallback and plot reward curve.
    """
    data = np.load(log_path + ".npz")
    rewards    = data["episode_rewards"]
    timesteps  = data["episode_timesteps"]

    rolling_mean = np.convolve(rewards, np.ones(rolling_window) / rolling_window,
                               mode="valid")
    rm_ts = timesteps[rolling_window - 1:]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timesteps, rewards, alpha=0.25, color="steelblue", linewidth=0.8,
            label="Episode reward")
    ax.plot(rm_ts, rolling_mean, color="steelblue", linewidth=2.0,
            label=f"Rolling mean ({rolling_window} ep)")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Episode reward")
    ax.set_title(f"TD3 Training Reward Curve ({tag} case)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# 2. Action trajectory plot
# ---------------------------------------------------------------------------

def plot_action_trajectory(
    model_path: str,
    save_path: str = "results/legacy/figures/rl_action_trajectory.png",
    omega_ref: float = 20.0,
    drift: bool = False,
    episode_length: float = 4.0,
    tag: str = "static",
):
    """
    Run one deterministic evaluation episode and plot Kp(t), Ki(t), Kd(t).
    Spec §6.2: "directly shows whether the agent is doing something sensible."
    """
    model = TD3.load(model_path)
    env = NPIDTuningEnv(
        drift=drift,
        episode_length=episode_length,
        omega_ref_range=(omega_ref, omega_ref),
        seed=SEED,
    )
    obs, _ = env.reset(seed=SEED)

    times, Kps, Kis, Kds = [], [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        times.append(info["t"])
        Kps.append(info["Kp"])
        Kis.append(info["Ki"])
        Kds.append(info["Kd"])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f"RL Agent Action Trajectory ({tag} case, omega_ref={omega_ref} rad/s)")

    ax1.plot(times, Kps, color="tab:blue", linewidth=1.5)
    ax1.set_ylabel("Kp")
    ax1.axhline(PSO_GAINS["Kp"], color="tab:blue", linestyle="--", alpha=0.5,
                label=f"PSO Kp={PSO_GAINS['Kp']:.1f}")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    ax2.plot(times, Kis, color="tab:orange", linewidth=1.5)
    ax2.set_ylabel("Ki")
    ax2.axhline(PSO_GAINS["Ki"], color="tab:orange", linestyle="--", alpha=0.5,
                label=f"PSO Ki={PSO_GAINS['Ki']:.1f}")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    ax3.plot(times, Kds, color="tab:green", linewidth=1.5)
    ax3.set_ylabel("Kd")
    ax3.set_xlabel("Time (s)")
    ax3.axhline(PSO_GAINS["Kd"], color="tab:green", linestyle="--", alpha=0.5,
                label=f"PSO Kd={PSO_GAINS['Kd']:.2f}")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# 3. Static-case comparison plot (RL vs PSO step response)
# ---------------------------------------------------------------------------

def plot_static_comparison(
    model_path: str,
    save_path: str = "results/legacy/figures/rl_static_vs_pso_comparison.png",
    omega_ref: float = 20.0,
):
    """
    Compare RL agent and PSO step response for the standard benchmark scenario.
    """
    # -- RL agent step response (high-res replay) -----------------------------
    model = TD3.load(model_path)
    env = NPIDTuningEnv(
        drift=False,
        episode_length=4.0,
        omega_ref_range=(omega_ref, omega_ref),
        seed=SEED,
    )
    obs, _ = env.reset(seed=SEED)
    done = False

    # Collect RL ω at each RL step
    rl_times, rl_omega = [0.0], [0.0]
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rl_times.append(info["t"])
        rl_omega.append(info["omega"])

    # -- PSO reference (high-res 1ms) ----------------------------------------
    ctrl = NPID(Kp=PSO_GAINS["Kp"], Ki=PSO_GAINS["Ki"], Kd=PSO_GAINS["Kd"],
                dt=DT, n=2, m=2, p=2, u_min=-24.0, u_max=24.0)
    result = run_simulation(ctrl, DCMotorParams(), omega_ref, t_end=4.0, dt=DT)

    # -- Plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rl_times, rl_omega, color="tab:blue", linewidth=2.0,
            label="N-PID (RL/TD3)")
    ax.plot(result.time, result.omega, color="tab:orange", linewidth=1.5,
            linestyle="--", label="N-PID (PSO, reference)")
    ax.axhline(omega_ref, color="k", linestyle=":", linewidth=1, label=f"omega_ref={omega_ref}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angular velocity (rad/s)")
    ax.set_title(f"Static Benchmark: RL Agent vs PSO (omega_ref={omega_ref} rad/s)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# 4. Evaluation metrics (for Week 6 comparison table)
# ---------------------------------------------------------------------------

def run_static_evaluation(
    model_path: str,
    omega_ref: float = 20.0,
    episode_length: float = 4.0,
    test_speeds: list | None = None,
):
    """
    Run deterministic evaluation at omega_ref=20 and optionally at extra speeds.
    Returns dict of metrics and prints a comparison table.
    """
    if test_speeds is None:
        test_speeds = [10.0, 20.0, 35.0]

    model = TD3.load(model_path)

    print("\n" + "=" * 65)
    print("  WEEK 6 EVALUATION — Static Case")
    print("=" * 65)

    # Reference §1.4 table
    REF = {
        "L-PID (ZN)":             (0.1350, 0.22,  0.1990, 0.1631),
        "N-PID (ZN)":             (0.1340, 8.07,  0.7510, 2.2720),
        "N-PID (PSO)":            (0.1350, 7.17,  0.6480, 0.3466),
        "N-PID (ML, RF)":         (0.1350, 7.20,  0.6460, 0.3894),
    }

    all_results = {}

    for speed in test_speeds:
        env = NPIDTuningEnv(
            drift=False,
            episode_length=episode_length,
            omega_ref_range=(speed, speed),
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

        metrics = compute_metrics(
            np.array(times), speed, np.array(omegas)
        )
        all_results[speed] = metrics

    # Print table for omega_ref=20 (primary benchmark)
    m20 = all_results[20.0] if 20.0 in all_results else all_results[test_speeds[0]]
    main_speed = 20.0 if 20.0 in all_results else test_speeds[0]

    print(f"\n  Primary benchmark (omega_ref={main_speed} rad/s):\n")
    header = f"  {'Controller':<24} {'Rise(s)':>10} {'Over(%)':>10} {'Settle(s)':>11} {'ITAE':>10}"
    print(header)
    print("  " + "-" * 67)
    for name, vals in REF.items():
        print(f"  {name:<24} {vals[0]:>10.4f} {vals[1]:>10.2f} {vals[2]:>11.4f} {vals[3]:>10.4f}")
    print("  " + "-" * 67)
    print(f"  {'N-PID (RL/TD3)':<24} "
          f"{m20['rise_time']:>10.4f} "
          f"{m20['overshoot_pct']:>10.2f} "
          f"{m20['settling_time']:>11.4f} "
          f"{m20['ITAE']:>10.4f}  <-- NEW")
    print()

    # Success check
    itae_ok   = 0.3 <= m20["ITAE"] <= 0.6
    over_ok   = m20["overshoot_pct"] <= 15.0
    stable_ok = not np.isnan(m20["rise_time"])

    print(f"  Success criteria:")
    print(f"    ITAE in [0.3, 0.6]:    {'PASS' if itae_ok else 'FAIL'} (ITAE={m20['ITAE']:.4f})")
    print(f"    Overshoot <= 15%:       {'PASS' if over_ok else 'FAIL'} (over={m20['overshoot_pct']:.2f}%)")
    print(f"    No instability:         {'PASS' if stable_ok else 'FAIL'}")

    # Generalization across speeds
    if len(test_speeds) > 1:
        print(f"\n  Generalization across speeds:")
        for spd, met in all_results.items():
            print(f"    omega_ref={spd:5.1f}: ITAE={met['ITAE']:.4f}, "
                  f"over={met['overshoot_pct']:.2f}%, "
                  f"rise={met['rise_time']:.4f}s")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL diagnostics plots and evaluation")
    parser.add_argument("--model", default="models/legacy/td3_npid_static",
                        help="Path to trained TD3 model (without .zip)")
    parser.add_argument("--log", default="results/legacy/tables/td3_static_training_log",
                        help="Path to training log .npz (without extension)")
    parser.add_argument("--drift", action="store_true",
                        help="Use drift-case model and settings")
    args = parser.parse_args()

    tag = "drift" if args.drift else "static"

    print(f"\n[RL Diagnostics] Model: {args.model}")

    if os.path.exists(args.log + ".npz"):
        plot_training_reward_curve(args.log, tag=tag)
    else:
        print(f"  [skip] Training log not found: {args.log}.npz")

    if os.path.exists(args.model + ".zip"):
        run_static_evaluation(args.model)
        plot_static_comparison(args.model)
        plot_action_trajectory(
            args.model,
            drift=args.drift,
            episode_length=20.0 if args.drift else 4.0,
            tag=tag,
        )
    else:
        print(f"  [skip] Model not found: {args.model}.zip")
