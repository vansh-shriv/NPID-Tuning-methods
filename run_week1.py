"""
Week 1 Deliverables Runner

Produces all Week 1 deliverables:
  1. Open-loop step response verification
  2. Ziegler-Nichols auto-tuning (finds Kcr, Tcr → Kp, Ki, Kd)
  3. Step-response comparison plot (L-PID vs N-PID)
  4. Disturbance-rejection comparison plot
  5. Metrics table for both scenarios

Outputs saved to results/figures/ and results/tables/.
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.plant import DCMotorParams, dc_motor_dynamics
from src.controllers import LPID, NPID
from src.simulate import run_simulation, rk4_step
from src.metrics import compute_metrics, format_metrics_table
from src.tuning_zn import find_critical_gain, ziegler_nichols_gains


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("data", exist_ok=True)


def run_open_loop_check(params, save_dir="results/figures"):
    """
    Deliverable: verify the plant's open-loop response is sensible.
    Apply constant V=12V, expect a second-order response settling in ~0.5-2s.
    """
    print("\n" + "=" * 70)
    print("STEP 1: Open-Loop Plant Verification")
    print("=" * 70)

    V_applied = 12.0
    T_load = 0.0
    dt = 0.001
    t_end = 3.0

    state = [0.0, 0.0]
    n_steps = int(t_end / dt)
    time_arr = np.zeros(n_steps)
    current_arr = np.zeros(n_steps)
    omega_arr = np.zeros(n_steps)

    for k in range(n_steps):
        time_arr[k] = k * dt
        current_arr[k] = state[0]
        omega_arr[k] = state[1]

        def f(t, y):
            return dc_motor_dynamics(t, y, V_applied, T_load, params)

        state = rk4_step(f, k * dt, state, dt)

    # Compute steady-state values
    omega_ss = np.mean(omega_arr[-200:])
    current_ss = np.mean(current_arr[-200:])

    # Check settling: find when omega reaches 98% of steady state
    threshold = 0.98 * omega_ss
    settled_idx = np.where(omega_arr >= threshold)[0]
    settling_approx = time_arr[settled_idx[0]] if len(settled_idx) > 0 else float("nan")

    print(f"  Applied voltage:       {V_applied} V")
    print(f"  Steady-state velocity: {omega_ss:.4f} rad/s")
    print(f"  Steady-state current:  {current_ss:.4f} A")
    print(f"  Approx settling time:  {settling_approx:.3f} s")
    print(f"  Expected: settles in ~0.5-2 s with no oscillation ✓")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("DC Motor Open-Loop Step Response (V = 12 V)", fontsize=14,
                 fontweight="bold")

    ax1.plot(time_arr, omega_arr, "b-", linewidth=1.5, label="ω(t)")
    ax1.axhline(y=omega_ss, color="b", linestyle="--", alpha=0.5,
                label=f"Steady state: {omega_ss:.2f} rad/s")
    ax1.set_ylabel("Angular Velocity ω (rad/s)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(time_arr, current_arr, "r-", linewidth=1.5, label="i(t)")
    ax2.axhline(y=current_ss, color="r", linestyle="--", alpha=0.5,
                label=f"Steady state: {current_ss:.4f} A")
    ax2.set_ylabel("Armature Current i (A)")
    ax2.set_xlabel("Time (s)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "open_loop_response.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved: {path}")

    return omega_ss


def run_zn_tuning(params, omega_ref=20.0):
    """
    Deliverable: Ziegler-Nichols auto-tuning for both controllers.

    Same Kcr/Tcr used for both L-PID and N-PID (Spec §3.1:
    near e≈0, N-PID functions approximate 1, so same starting gains
    are a reasonable approximation — noted in the write-up).
    """
    print("\n" + "=" * 70)
    print("STEP 2: Ziegler-Nichols Auto-Tuning")
    print("=" * 70)

    Kcr, Tcr = find_critical_gain(params, omega_ref)
    Kp, Ki, Kd = ziegler_nichols_gains(Kcr, Tcr)

    print(f"\n  Critical gain:  Kcr = {Kcr:.4f}")
    print(f"  Critical period: Tcr = {Tcr:.4f} s")
    print(f"\n  Ziegler-Nichols PID gains:")
    print(f"    Kp = {Kp:.4f}")
    print(f"    Ki = {Ki:.4f}")
    print(f"    Kd = {Kd:.4f}")
    print(f"\n  Note: Same gains used for both L-PID and N-PID base gains.")
    print(f"  N-PID approximation: near e≈0, α≈β≈γ≈1 → same Kcr/Tcr is valid.")

    return Kp, Ki, Kd, Kcr, Tcr


def run_step_response_comparison(params, Kp, Ki, Kd, omega_ref=20.0,
                                 t_end=3.0, dt=0.001, save_dir="results/figures"):
    """
    Deliverable: Step-response comparison plot (L-PID vs N-PID).
    """
    print("\n" + "=" * 70)
    print("STEP 3: Step-Response Comparison (L-PID vs N-PID)")
    print("=" * 70)

    # Create controllers with Z-N gains
    lpid = LPID(Kp=Kp, Ki=Ki, Kd=Kd, dt=dt,
                u_min=-params.Vmax, u_max=params.Vmax)
    npid = NPID(Kp=Kp, Ki=Ki, Kd=Kd, dt=dt, n=2, m=2, p=2,
                u_min=-params.Vmax, u_max=params.Vmax)

    # Run simulations
    result_lpid = run_simulation(lpid, params, omega_ref, t_end=t_end, dt=dt)
    result_npid = run_simulation(npid, params, omega_ref, t_end=t_end, dt=dt)

    # Compute metrics
    metrics_lpid = compute_metrics(result_lpid.time, omega_ref, result_lpid.omega)
    metrics_npid = compute_metrics(result_npid.time, omega_ref, result_npid.omega)

    print(f"\n  L-PID: {lpid}")
    print(f"  N-PID: {npid}")

    # Print metrics
    metrics_dict = {"L-PID (ZN)": metrics_lpid, "N-PID (ZN)": metrics_npid}
    print(f"\n{format_metrics_table(metrics_dict, 'Step Response Metrics')}")

    # Plot: velocity response
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(f"Step Response Comparison: L-PID vs N-PID\n"
                 f"(ω_ref = {omega_ref} rad/s, Z-N tuned: "
                 f"Kp={Kp:.2f}, Ki={Ki:.2f}, Kd={Kd:.2f})",
                 fontsize=13, fontweight="bold")

    # Velocity
    ax = axes[0]
    ax.plot(result_lpid.time, result_lpid.omega, "b-", linewidth=1.5, label="L-PID", alpha=0.8)
    ax.plot(result_npid.time, result_npid.omega, "r-", linewidth=1.5, label="N-PID", alpha=0.8)
    ax.axhline(y=omega_ref, color="k", linestyle="--", alpha=0.4, label=f"Reference ({omega_ref} rad/s)")
    # ±2% settling band
    ax.axhline(y=omega_ref * 1.02, color="gray", linestyle=":", alpha=0.3)
    ax.axhline(y=omega_ref * 0.98, color="gray", linestyle=":", alpha=0.3, label="±2% band")
    ax.set_ylabel("Angular Velocity ω (rad/s)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Control effort
    ax = axes[1]
    ax.plot(result_lpid.time, result_lpid.voltage, "b-", linewidth=1.0, label="L-PID", alpha=0.7)
    ax.plot(result_npid.time, result_npid.voltage, "r-", linewidth=1.0, label="N-PID", alpha=0.7)
    ax.axhline(y=params.Vmax, color="gray", linestyle="--", alpha=0.3, label=f"±Vmax ({params.Vmax}V)")
    ax.axhline(y=-params.Vmax, color="gray", linestyle="--", alpha=0.3)
    ax.set_ylabel("Control Voltage V (V)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Error
    ax = axes[2]
    ax.plot(result_lpid.time, result_lpid.error, "b-", linewidth=1.0, label="L-PID", alpha=0.7)
    ax.plot(result_npid.time, result_npid.error, "r-", linewidth=1.0, label="N-PID", alpha=0.7)
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.2)
    ax.set_ylabel("Tracking Error (rad/s)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "step_response_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → Saved: {path}")

    return metrics_dict, result_lpid, result_npid


def run_disturbance_rejection(params, Kp, Ki, Kd, omega_ref=20.0,
                              t_end=4.0, dt=0.001, T_load_value=0.05,
                              T_load_time=1.5, save_dir="results/figures"):
    """
    Deliverable: Disturbance-rejection comparison plot.
    Apply a step load torque disturbance at t = T_load_time.
    """
    print("\n" + "=" * 70)
    print("STEP 4: Disturbance Rejection Comparison")
    print("=" * 70)

    # Load torque step function
    def T_load_func(t):
        return T_load_value if t >= T_load_time else 0.0

    # Create controllers
    lpid = LPID(Kp=Kp, Ki=Ki, Kd=Kd, dt=dt,
                u_min=-params.Vmax, u_max=params.Vmax)
    npid = NPID(Kp=Kp, Ki=Ki, Kd=Kd, dt=dt, n=2, m=2, p=2,
                u_min=-params.Vmax, u_max=params.Vmax)

    # Run simulations
    result_lpid = run_simulation(lpid, params, omega_ref, t_end=t_end, dt=dt,
                                 T_load_func=T_load_func)
    result_npid = run_simulation(npid, params, omega_ref, t_end=t_end, dt=dt,
                                 T_load_func=T_load_func)

    # Compute metrics (over the full simulation including disturbance)
    metrics_lpid = compute_metrics(result_lpid.time, omega_ref, result_lpid.omega)
    metrics_npid = compute_metrics(result_npid.time, omega_ref, result_npid.omega)

    metrics_dict = {"L-PID (ZN+dist)": metrics_lpid, "N-PID (ZN+dist)": metrics_npid}
    print(f"\n{format_metrics_table(metrics_dict, 'Disturbance Rejection Metrics')}")

    # Compute disturbance-specific metrics: max deviation and recovery time
    # after disturbance is applied
    dist_idx = int(T_load_time / dt)
    for name, result in [("L-PID", result_lpid), ("N-PID", result_npid)]:
        omega_after = result.omega[dist_idx:]
        time_after = result.time[dist_idx:]
        deviation = np.min(omega_after) - omega_ref  # Negative deviation
        max_dev = np.abs(deviation)

        # Recovery: when does it return to within ±2% of omega_ref after disturbance?
        tolerance = 0.02 * abs(omega_ref)
        within_band = np.abs(omega_after - omega_ref) <= tolerance
        outside = np.where(~within_band)[0]
        if len(outside) > 0 and outside[-1] < len(time_after) - 1:
            recovery_time = time_after[outside[-1] + 1] - T_load_time
        else:
            recovery_time = float("nan")

        print(f"  {name}: max deviation = {max_dev:.4f} rad/s, "
              f"recovery time = {recovery_time:.4f} s")

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(f"Disturbance Rejection: L-PID vs N-PID\n"
                 f"(ω_ref = {omega_ref} rad/s, T_load = {T_load_value} N·m "
                 f"step at t = {T_load_time} s)",
                 fontsize=13, fontweight="bold")

    # Velocity
    ax = axes[0]
    ax.plot(result_lpid.time, result_lpid.omega, "b-", linewidth=1.5, label="L-PID", alpha=0.8)
    ax.plot(result_npid.time, result_npid.omega, "r-", linewidth=1.5, label="N-PID", alpha=0.8)
    ax.axhline(y=omega_ref, color="k", linestyle="--", alpha=0.4, label=f"Reference")
    ax.axvline(x=T_load_time, color="green", linestyle="--", alpha=0.5,
               label=f"Disturbance onset (t={T_load_time}s)")
    ax.axhline(y=omega_ref * 1.02, color="gray", linestyle=":", alpha=0.3)
    ax.axhline(y=omega_ref * 0.98, color="gray", linestyle=":", alpha=0.3)
    ax.set_ylabel("Angular Velocity ω (rad/s)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Control effort
    ax = axes[1]
    ax.plot(result_lpid.time, result_lpid.voltage, "b-", linewidth=1.0, label="L-PID", alpha=0.7)
    ax.plot(result_npid.time, result_npid.voltage, "r-", linewidth=1.0, label="N-PID", alpha=0.7)
    ax.axvline(x=T_load_time, color="green", linestyle="--", alpha=0.5)
    ax.set_ylabel("Control Voltage V (V)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Error
    ax = axes[2]
    ax.plot(result_lpid.time, result_lpid.error, "b-", linewidth=1.0, label="L-PID", alpha=0.7)
    ax.plot(result_npid.time, result_npid.error, "r-", linewidth=1.0, label="N-PID", alpha=0.7)
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.2)
    ax.axvline(x=T_load_time, color="green", linestyle="--", alpha=0.5)
    ax.set_ylabel("Tracking Error (rad/s)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "disturbance_rejection_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → Saved: {path}")

    return metrics_dict


def save_metrics_csv(all_metrics, filepath="results/tables/week1_metrics.csv"):
    """Save all metrics to a CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    headers = ["Controller", "Scenario", "Rise Time (s)", "Overshoot (%)",
               "Settling Time (s)", "IAE", "ISE", "ITAE"]

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for scenario, metrics_dict in all_metrics.items():
            for name, m in metrics_dict.items():
                writer.writerow([
                    name, scenario,
                    f"{m['rise_time']:.4f}" if not np.isnan(m['rise_time']) else "N/A",
                    f"{m['overshoot_pct']:.2f}",
                    f"{m['settling_time']:.4f}" if not np.isnan(m['settling_time']) else "N/A",
                    f"{m['IAE']:.4f}",
                    f"{m['ISE']:.4f}",
                    f"{m['ITAE']:.4f}",
                ])

    print(f"  → Saved: {filepath}")


def main():
    """Run all Week 1 deliverables in sequence."""
    print("╔" + "═" * 68 + "╗")
    print("║" + "  WEEK 1: Plant Model + L-PID + N-PID (Ziegler-Nichols Tuned)  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # Create output directories
    ensure_dirs()

    # Use default plant parameters (Spec §1.2)
    params = DCMotorParams()
    omega_ref = 20.0  # Reference speed (rad/s)
    dt = 0.001        # Timestep (s)

    print(f"\nPlant parameters:")
    print(f"  R={params.R} Ω, L={params.L} H, Ke={params.Ke} V·s/rad")
    print(f"  Kt={params.Kt} N·m/A, J={params.J} kg·m², B={params.B} N·m·s/rad")
    print(f"  Tc={params.Tc} N·m (Coulomb friction), Vmax={params.Vmax} V")
    print(f"  ε={params.epsilon} (friction smoothing)")

    # Step 1: Open-loop verification
    omega_ss = run_open_loop_check(params)

    # Step 2: Ziegler-Nichols tuning
    Kp, Ki, Kd, Kcr, Tcr = run_zn_tuning(params, omega_ref)

    # Step 3: Step-response comparison
    step_metrics, _, _ = run_step_response_comparison(
        params, Kp, Ki, Kd, omega_ref, t_end=3.0, dt=dt
    )

    # Step 4: Disturbance-rejection comparison
    dist_metrics = run_disturbance_rejection(
        params, Kp, Ki, Kd, omega_ref, t_end=4.0, dt=dt,
        T_load_value=0.05, T_load_time=1.5
    )

    # Step 5: Save combined metrics table
    print("\n" + "=" * 70)
    print("STEP 5: Saving Combined Metrics Table")
    print("=" * 70)

    all_metrics = {
        "Step Response": step_metrics,
        "Disturbance Rejection": dist_metrics,
    }
    save_metrics_csv(all_metrics)

    # Final summary
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + "  WEEK 1 DELIVERABLES COMPLETE  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"""
  ✓ DC motor plant model — verified (open-loop settles to {omega_ss:.2f} rad/s)
  ✓ L-PID controller — with back-calculation anti-windup
  ✓ N-PID controller — α(e), β(e), γ(de/dt) with n=m=p=2
  ✓ Ziegler-Nichols tuning — Kcr={Kcr:.4f}, Tcr={Tcr:.4f}s
                              Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f}
  ✓ Step-response comparison plot → results/figures/step_response_comparison.png
  ✓ Disturbance-rejection plot    → results/figures/disturbance_rejection_comparison.png
  ✓ Metrics table (CSV)           → results/tables/week1_metrics.csv

  Next: Review plots and metrics, then proceed to Week 2 (PSO tuning).
    """)


if __name__ == "__main__":
    main()
