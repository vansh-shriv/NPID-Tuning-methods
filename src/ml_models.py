"""
Machine Learning Gain Predictor (Spec Section 4)

Trains and evaluates Random Forest and MLP regressors to predict N-PID gains
from DC motor system parameters. Benchmarks inference speed against PSO.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plant import DCMotorParams
from src.controllers import NPID
from src.simulate import run_simulation
from src.metrics import compute_metrics
from src.tuning_pso import optimize_pso


def train_and_evaluate(dataset_path="data/tuning_dataset.csv", seed=42):
    """
    Load dataset, train Random Forest and MLP models, evaluate on test set,
    run downstream closed-loop simulation validation, and run speed benchmarks.
    """
    print("=" * 70)
    print("Machine Learning Model Training & Evaluation")
    print("=" * 70)

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please run src/generate_dataset.py first.")
        return

    # Load data
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset with {len(df)} samples.")

    # Features and targets
    features = ["J", "B", "Tc", "T_load", "omega_ref"]
    targets = ["Kp", "Ki", "Kd"]

    X = df[features]
    y = df[targets]

    # Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Standardize input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Model 1: Random Forest
    print("\nTraining Multi-Output Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=200, max_depth=None, random_state=seed
    )
    t0 = time.perf_counter()
    rf_model.fit(X_train_scaled, y_train)
    t_train_rf = time.perf_counter() - t0
    print(f"  Random Forest trained in {t_train_rf:.3f}s")

    # Train Model 2: MLP
    print("\nTraining Multi-Output MLP Regressor...")
    mlp_model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        max_iter=2000,
        random_state=seed
    )
    t0 = time.perf_counter()
    mlp_model.fit(X_train_scaled, y_train)
    t_train_mlp = time.perf_counter() - t0
    print(f"  MLP trained in {t_train_mlp:.3f}s")

    # Evaluate models on test set
    rf_preds = rf_model.predict(X_test_scaled)
    mlp_preds = mlp_model.predict(X_test_scaled)

    print("\n" + "=" * 50)
    print("Test Set Metrics (R² and MAE)")
    print("=" * 50)

    for i, target in enumerate(targets):
        rf_r2 = r2_score(y_test[target], rf_preds[:, i])
        rf_mae = mean_absolute_error(y_test[target], rf_preds[:, i])
        mlp_r2 = r2_score(y_test[target], mlp_preds[:, i])
        mlp_mae = mean_absolute_error(y_test[target], mlp_preds[:, i])

        print(f"{target:3s}:")
        print(f"  Random Forest -> R² = {rf_r2:.4f}, MAE = {rf_mae:.4f}")
        print(f"  MLP           -> R² = {mlp_r2:.4f}, MAE = {mlp_mae:.4f}")

    # Save Random Forest feature importances
    os.makedirs("results/figures", exist_ok=True)
    importances = rf_model.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.bar(features, importances, color="skyblue", edgecolor="grey")
    plt.ylabel("Relative Importance")
    plt.title("Random Forest Regressor: Feature Importance")
    plt.grid(True, alpha=0.3, axis="y")
    fig_path = "results/figures/feature_importances.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved feature importance plot to {fig_path}")

    # Downstream Closed-Loop Simulation Performance
    print("\nRunning downstream closed-loop simulation validation...")
    print("Simulating test set plant configurations with ML-predicted gains...")
    
    pso_itaes = []
    ml_itaes = []
    stable_count = 0

    # Test all configurations in the test set
    test_indices = X_test.index
    for idx_num, idx in enumerate(test_indices):
        row = df.loc[idx]
        J, B, Tc, T_load_val, omega_ref = row[features]
        true_pso_itae = row["achieved_ITAE"]

        # Predict gains with RF and clip to safe minimums to prevent instabilites
        x_scaled = scaler.transform([[J, B, Tc, T_load_val, omega_ref]])
        gains_pred = rf_model.predict(x_scaled)[0]
        Kp_pred = max(0.1, gains_pred[0])
        Ki_pred = max(0.1, gains_pred[1])
        Kd_pred = max(0.0, gains_pred[2])

        # Configure plant
        params = DCMotorParams(
            R=1.0, L=0.5, Ke=0.5, Kt=0.5, J=J, B=B, Tc=Tc, Vmax=24.0, epsilon=0.01
        )
        def T_load_func(t):
            return T_load_val if t >= 1.5 else 0.0

        # Simulate
        controller = NPID(Kp=Kp_pred, Ki=Ki_pred, Kd=Kd_pred, dt=0.001,
                          n=2, m=2, p=2, u_min=-params.Vmax, u_max=params.Vmax)
        try:
            res = run_simulation(controller, params, omega_ref, t_end=4.0, dt=0.001,
                                 T_load_func=T_load_func)
            
            # Check stability
            if np.any(np.isnan(res.omega)) or np.any(np.isinf(res.omega)) or np.max(np.abs(res.omega)) > 10.0 * omega_ref:
                achieved_itae = 1e6
            else:
                m = compute_metrics(res.time, omega_ref, res.omega)
                achieved_itae = m["ITAE"]
                if achieved_itae < 1e5:
                    stable_count += 1
        except Exception:
            achieved_itae = 1e6

        pso_itaes.append(true_pso_itae)
        ml_itaes.append(achieved_itae)

    pso_itaes = np.array(pso_itaes)
    ml_itaes = np.array(ml_itaes)

    # Filter out unstable runs for ratio calculations
    valid_mask = ml_itaes < 1e5
    stable_pct = (stable_count / len(X_test)) * 100.0
    
    print(f"\nDownstream results summary:")
    print(f"  Stable controllers: {stable_count}/{len(X_test)} ({stable_pct:.1f}%)")
    
    if stable_count > 0:
        ratios = ml_itaes[valid_mask] / pso_itaes[valid_mask]
        print(f"  Mean ITAE Ratio (ML/PSO):   {np.mean(ratios):.3f} (closer to 1.0 is better)")
        print(f"  Median ITAE Ratio (ML/PSO): {np.median(ratios):.3f}")
        print(f"  Max ITAE Ratio (ML/PSO):    {np.max(ratios):.3f}")
    else:
        print("  WARNING: No stable controllers using ML predicted gains.")

    # Speed Benchmark (PSO Search vs ML Inference)
    print("\nRunning timing benchmark (PSO Search vs ML Inference)...")
    
    # Pick a random sample configuration
    J_bench = 0.012
    B_bench = 0.12
    Tc_bench = 0.06
    T_load_bench = 0.02
    omega_bench = 22.0
    
    params_bench = DCMotorParams(
        R=1.0, L=0.5, Ke=0.5, Kt=0.5, J=J_bench, B=B_bench, Tc=Tc_bench,
        Vmax=24.0, epsilon=0.01
    )
    def T_load_bench_func(t):
        return T_load_bench if t >= 1.5 else 0.0

    # Benchmark PSO
    print("  Benchmarking PSO search (this takes a few seconds)...")
    t_start = time.perf_counter()
    best_gains, _ = optimize_pso(
        params_bench, omega_bench, init_gains=[139.2, 1139.4, 0.7],
        n_particles=20, iters=30, T_load_func=T_load_bench_func,
        overshoot_penalty_weight=5.0, verbose=False
    )
    t_pso = time.perf_counter() - t_start
    print(f"    PSO optimization time: {t_pso:.4f} seconds")

    # Benchmark ML inference
    bench_scaled = scaler.transform([[J_bench, B_bench, Tc_bench, T_load_bench, omega_bench]])
    
    # Run 1000 inferences to get a precise average
    t_start = time.perf_counter()
    for _ in range(1000):
        _ = rf_model.predict(bench_scaled)
    t_ml = (time.perf_counter() - t_start) / 1000.0
    print(f"    ML inference time: {t_ml*1000.0:.4f} milliseconds ({t_ml:.8f} seconds)")

    speedup = t_pso / t_ml
    print(f"\n  → ML-predicted gains are {speedup:.1f}x FASTER than iterative PSO search!")


if __name__ == "__main__":
    train_and_evaluate()
