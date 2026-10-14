import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # _legacy_root: run from the repo root
"""
Week 4: Robustness Testing, Extrapolation, and Final Benchmarking

Performs:
  1. Noisy feedback test (omega_measured = omega_true + noise) for 4 controllers.
  2. Extrapolation test on unseen out-of-bounds parameter (J = 0.03).
  3. Saves master comparison tables (CSV & Markdown) to results/legacy/tables/.
  4. Saves step response plots under noise and extrapolation to results/legacy/figures/.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.plant import DCMotorParams
from src.controllers import LPID, NPID
from src.simulate import run_simulation
from src.metrics import compute_metrics, format_metrics_table, metrics_to_csv_row
from src.tuning_zn import find_critical_gain, ziegler_nichols_gains
from src.legacy.tuning_pso import optimize_pso


def get_trained_rf_model(dataset_path="data/legacy/tuning_dataset.csv", seed=42):
    """Train the Random Forest model on the full dataset and return it along with its scaler."""
    df = pd.read_csv(dataset_path)
    features = ["J", "B", "Tc", "T_load", "omega_ref"]
    targets = ["Kp", "Ki", "Kd"]
    
    X = df[features]
    y = df[targets]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=seed)
    model.fit(X_scaled, y)
    
    return model, scaler


def run_week4_deliverables():
    print("╔" + "═" * 68 + "╗")
    print("║" + "   WEEK 4: ROBUSTNESS, EXTRAPOLATION & MASTER BENCHMARKING    ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Load and train RF model
    model, scaler = get_trained_rf_model()
    
    # -------------------------------------------------------------------------
    # Setup baseline and noisy test scenarios on the default plant
    # -------------------------------------------------------------------------
    params = DCMotorParams()  # Default: J=0.01, B=0.1, Tc=0.05, Vmax=24V
    omega_ref = 20.0
    dt = 0.001
    t_end = 4.0
    noise_sigma = 0.3  # ~1.5% of reference speed (20 rad/s)
    
    print("\n" + "=" * 70)
    print("STEP 1: Tuning All 4 Controller Variants for Default Plant")
    print("=" * 70)
    
    # 1. Z-N gains
    Kcr, Tcr = find_critical_gain(params, omega_ref, verbose=False)
    Kp_zn, Ki_zn, Kd_zn = ziegler_nichols_gains(Kcr, Tcr)
    print(f"  Z-N Gains: Kp={Kp_zn:.4f}, Ki={Ki_zn:.4f}, Kd={Kd_zn:.4f}")
    
    # 2. PSO gains (optimized on default plant)
    print("  Running PSO tuning on default plant...")
    pso_gains, _ = optimize_pso(
        params, omega_ref, init_gains=[Kp_zn, Ki_zn, Kd_zn],
        n_particles=15, iters=20, verbose=False, t_end=t_end
    )
    Kp_pso, Ki_pso, Kd_pso = pso_gains
    print(f"  PSO Gains: Kp={Kp_pso:.4f}, Ki={Ki_pso:.4f}, Kd={Kd_pso:.4f}")
    
    # 3. ML-predicted gains
    # Default parameters: J=0.01, B=0.1, Tc=0.05, T_load=0.0, omega_ref=20.0
    x_scaled = scaler.transform([[params.J, params.B, params.Tc, 0.0, omega_ref]])
    ml_gains = model.predict(x_scaled)[0]
    Kp_ml = max(0.1, ml_gains[0])
    Ki_ml = max(0.1, ml_gains[1])
    Kd_ml = max(0.0, ml_gains[2])
    print(f"  ML Gains:  Kp={Kp_ml:.4f}, Ki={Ki_ml:.4f}, Kd={Kd_ml:.4f}")

    # Define the 4 controllers
    controllers = {
        "L-PID (ZN)": LPID(Kp=Kp_zn, Ki=Ki_zn, Kd=Kd_zn, dt=dt, u_min=-params.Vmax, u_max=params.Vmax),
        "N-PID (ZN)": NPID(Kp=Kp_zn, Ki=Ki_zn, Kd=Kd_zn, dt=dt, n=2, m=2, p=2, u_min=-params.Vmax, u_max=params.Vmax),
        "N-PID (PSO)": NPID(Kp=Kp_pso, Ki=Ki_pso, Kd=Kd_pso, dt=dt, n=2, m=2, p=2, u_min=-params.Vmax, u_max=params.Vmax),
        "N-PID (ML)": NPID(Kp=Kp_ml, Ki=Ki_ml, Kd=Kd_ml, dt=dt, n=2, m=2, p=2, u_min=-params.Vmax, u_max=params.Vmax)
    }

    # -------------------------------------------------------------------------
    # Scenario A: Baseline (No Noise)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Simulating Baseline Scenario (No Noise)")
    print("=" * 70)
    
    baseline_metrics = {}
    baseline_results = {}
    
    for name, ctrl in controllers.items():
        res = run_simulation(ctrl, params, omega_ref, t_end=t_end, dt=dt)
        m = compute_metrics(res.time, omega_ref, res.omega)
        baseline_metrics[name] = m
        baseline_results[name] = res

    print(format_metrics_table(baseline_metrics, "Baseline Step Response (No Noise)"))

    # -------------------------------------------------------------------------
    # Scenario B: Noisy Feedback (noise_sigma = 0.3)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Simulating Noisy Feedback Scenario (noise_sigma = 0.3)")
    print("=" * 70)
    
    noisy_metrics = {}
    noisy_results = {}
    
    # We use a fixed generator seed for reproducible noise across controllers
    rng = np.random.default_rng(42)
    
    for name, ctrl in controllers.items():
        # Re-initialize noise generator seed for fair comparison
        rng_comp = np.random.default_rng(42)
        res = run_simulation(ctrl, params, omega_ref, t_end=t_end, dt=dt,
                             noise_sigma=noise_sigma, rng=rng_comp)
        m = compute_metrics(res.time, omega_ref, res.omega)
        noisy_metrics[name] = m
        noisy_results[name] = res

    print(format_metrics_table(noisy_metrics, f"Noisy Feedback (sigma = {noise_sigma})"))

    # Plot Noisy Feedback Comparison
    plt.figure(figsize=(11, 7))
    for name, res in noisy_results.items():
        plt.plot(res.time, res.omega, label=f"{name} (ITAE: {noisy_metrics[name]['ITAE']:.2f})", alpha=0.8)
    plt.axhline(y=omega_ref, color="k", linestyle="--", alpha=0.5, label="Reference")
    plt.ylabel("Velocity ω (rad/s)")
    plt.xlabel("Time (s)")
    plt.title(f"Noise Robustness Comparison (Feedback Noise σ = {noise_sigma})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    noise_plot_path = "results/legacy/figures/noise_robustness_comparison.png"
    plt.savefig(noise_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved noise robustness plot: {noise_plot_path}")

    # -------------------------------------------------------------------------
    # Scenario C: Extrapolation Test (J = 0.03 - Outside Training Range [0.005, 0.02])
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Extrapolation Test (J = 0.03, Training Limit J = 0.02)")
    print("=" * 70)
    
    params_ext = DCMotorParams(J=0.03)  # Large inertia extrapolation
    T_load_val = 0.02
    
    def T_load_ext_func(t):
        return T_load_val if t >= 1.5 else 0.0
        
    # 1. Z-N Tuning on J=0.03 plant
    Kcr_ext, Tcr_ext = find_critical_gain(params_ext, omega_ref, verbose=False)
    Kp_zn_ext, Ki_zn_ext, Kd_zn_ext = ziegler_nichols_gains(Kcr_ext, Tcr_ext)
    
    # 2. PSO Tuning on J=0.03 plant
    print("  Running PSO on extrapolation plant (J=0.03)...")
    pso_gains_ext, pso_cost_ext = optimize_pso(
        params_ext, omega_ref, init_gains=[Kp_zn_ext, Ki_zn_ext, Kd_zn_ext],
        n_particles=15, iters=20, T_load_func=T_load_ext_func,
        verbose=False, t_end=t_end
    )
    Kp_pso_ext, Ki_pso_ext, Kd_pso_ext = pso_gains_ext
    
    # 3. ML-Predicted Gains (Extrapolating outside training boundaries)
    x_scaled_ext = scaler.transform([[params_ext.J, params_ext.B, params_ext.Tc, T_load_val, omega_ref]])
    ml_gains_ext = model.predict(x_scaled_ext)[0]
    Kp_ml_ext = max(0.1, ml_gains_ext[0])
    Ki_ml_ext = max(0.1, ml_gains_ext[1])
    Kd_ml_ext = max(0.0, ml_gains_ext[2])
    
    print(f"  Extrapolated gains (J=0.03):")
    print(f"    Z-N gains:  Kp={Kp_zn_ext:.2f}, Ki={Ki_zn_ext:.2f}, Kd={Kd_zn_ext:.2f}")
    print(f"    PSO gains:  Kp={Kp_pso_ext:.2f}, Ki={Ki_pso_ext:.2f}, Kd={Kd_pso_ext:.2f}")
    print(f"    ML gains:   Kp={Kp_ml_ext:.2f}, Ki={Ki_ml_ext:.2f}, Kd={Kd_ml_ext:.2f}")

    # Instantiate controllers for extrapolation
    ctrls_ext = {
        "L-PID (ZN)": LPID(Kp=Kp_zn_ext, Ki=Ki_zn_ext, Kd=Kd_zn_ext, dt=dt, u_min=-params_ext.Vmax, u_max=params_ext.Vmax),
        "N-PID (ZN)": NPID(Kp=Kp_zn_ext, Ki=Ki_zn_ext, Kd=Kd_zn_ext, dt=dt, n=2, m=2, p=2, u_min=-params_ext.Vmax, u_max=params_ext.Vmax),
        "N-PID (PSO)": NPID(Kp=Kp_pso_ext, Ki=Ki_pso_ext, Kd=Kd_pso_ext, dt=dt, n=2, m=2, p=2, u_min=-params_ext.Vmax, u_max=params_ext.Vmax),
        "N-PID (ML)": NPID(Kp=Kp_ml_ext, Ki=Ki_ml_ext, Kd=Kd_ml_ext, dt=dt, n=2, m=2, p=2, u_min=-params_ext.Vmax, u_max=params_ext.Vmax)
    }

    ext_metrics = {}
    ext_results = {}
    for name, ctrl in ctrls_ext.items():
        res = run_simulation(ctrl, params_ext, omega_ref, t_end=t_end, dt=dt, T_load_func=T_load_ext_func)
        m = compute_metrics(res.time, omega_ref, res.omega)
        ext_metrics[name] = m
        ext_results[name] = res

    print(format_metrics_table(ext_metrics, "Extrapolation Test (J = 0.03, T_load = 0.02 step at 1.5s)"))

    # Plot Extrapolation
    plt.figure(figsize=(11, 7))
    for name, res in ext_results.items():
        plt.plot(res.time, res.omega, label=f"{name} (ITAE: {ext_metrics[name]['ITAE']:.2f})", alpha=0.8)
    plt.axhline(y=omega_ref, color="k", linestyle="--", alpha=0.5, label="Reference")
    plt.axvline(x=1.5, color="g", linestyle=":", label="Disturbance step")
    plt.ylabel("Velocity ω (rad/s)")
    plt.xlabel("Time (s)")
    plt.title("Extrapolation Test Comparison (J = 0.03 outside training bound [0.005, 0.02])")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    ext_plot_path = "results/legacy/figures/extrapolation_comparison.png"
    plt.savefig(ext_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved extrapolation plot: {ext_plot_path}")

    # -------------------------------------------------------------------------
    # STEP 5: Save Master Tables
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: Saving Master Benchmark Tables")
    print("=" * 70)
    
    # Baseline Scenario CSV
    os.makedirs("results/legacy/tables", exist_ok=True)
    baseline_rows = []
    for name, m in baseline_metrics.items():
        baseline_rows.append(metrics_to_csv_row(name, m))
    pd.DataFrame(baseline_rows).to_csv("results/legacy/tables/master_baseline_comparison.csv", index=False)
    print("  → Saved: results/legacy/tables/master_baseline_comparison.csv")
    
    # Noisy Scenario CSV
    noisy_rows = []
    for name, m in noisy_metrics.items():
        noisy_rows.append(metrics_to_csv_row(name, m))
    pd.DataFrame(noisy_rows).to_csv("results/legacy/tables/master_noisy_comparison.csv", index=False)
    print("  → Saved: results/legacy/tables/master_noisy_comparison.csv")
    
    # Extrapolation Scenario CSV
    ext_rows = []
    for name, m in ext_metrics.items():
        ext_rows.append(metrics_to_csv_row(name, m))
    pd.DataFrame(ext_rows).to_csv("results/legacy/tables/master_extrapolation_comparison.csv", index=False)
    print("  → Saved: results/legacy/tables/master_extrapolation_comparison.csv")

    print("\n╔" + "═" * 68 + "╗")
    print("║" + "   WEEK 4 DELIVERABLES COMPLETE  ".center(68) + "║")
    print("╚" + "═" * 68 + "╝\n")


if __name__ == "__main__":
    run_week4_deliverables()
