"""
Dataset Generation Script (Spec Section 3.4)

Randomizes DC motor plant parameters (J, B, Tc, T_load, omega_ref) across many trials
and runs PSO-tuning to find optimal N-PID gains [Kp, Ki, Kd] for each configuration.
Saves the resulting dataset to a CSV file.
Uses multiprocessing to parallelize PSO searches across CPU cores.
"""

import os
import sys
import csv
import json
import argparse
import time
import numpy as np
from multiprocessing import Pool, cpu_count

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plant import DCMotorParams
from src.controllers import NPID
from src.simulate import run_simulation
from src.metrics import compute_metrics
from src.tuning_zn import find_critical_gain, ziegler_nichols_gains
from src.legacy.tuning_pso import optimize_pso


def worker_task(task_args):
    """
    Worker task to perform a single trial optimization.
    Runs PSO search with fixed wide bounds to find optimal gains.
    """
    trial_idx, params_dict, omega_ref = task_args
    
    # Reconstruct plant parameters
    params = DCMotorParams(
        R=params_dict["R"],
        L=params_dict["L"],
        Ke=params_dict["Ke"],
        Kt=params_dict["Kt"],
        J=params_dict["J"],
        B=params_dict["B"],
        Tc=params_dict["Tc"],
        Vmax=params_dict["Vmax"],
        epsilon=params_dict["epsilon"]
    )
    
    T_load_val = params_dict["T_load_val"]
    
    # Load torque step disturbance applied at t = 1.5s in a 3s simulation
    def T_load_func(t):
        return T_load_val if t >= 1.5 else 0.0

    # Define fixed search bounds centered around default plant's Z-N gains
    min_bounds = np.array([10.0, 100.0, 0.05])
    max_bounds = np.array([500.0, 15000.0, 5.0])
    bounds = (min_bounds, max_bounds)

    # Run PSO optimization (iters=8, particles=8, t_end=3.0)
    try:
        best_gains, best_cost = optimize_pso(
            params, omega_ref, bounds=bounds,
            n_particles=8, iters=8,
            T_load_func=T_load_func,
            overshoot_penalty_weight=5.0,
            verbose=False,
            t_end=3.0
        )
        
        return {
            "status": "success",
            "trial": trial_idx,
            "J": params.J,
            "B": params.B,
            "Tc": params.Tc,
            "T_load": T_load_val,
            "omega_ref": omega_ref,
            "Kp": best_gains[0],
            "Ki": best_gains[1],
            "Kd": best_gains[2],
            "achieved_ITAE": best_cost
        }
    except Exception as e:
        return {
            "status": "fail",
            "error": str(e),
            "J": params.J,
            "B": params.B,
            "Tc": params.Tc,
            "T_load": T_load_val,
            "omega_ref": omega_ref
        }


def generate_dataset(num_samples=500, output_path="data/legacy/tuning_dataset.csv", seed=42, num_workers=1):
    """
    Generate the dataset by randomizing plant parameters and running optimization.
    """
    print(f"Generating dataset of {num_samples} samples...")
    print(f"Output path: {output_path}")
    print(f"Random seed: {seed}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Setup random generator for reproducibility
    rng = np.random.default_rng(seed)
    
    # Generate randomized plant settings
    tasks = []
    for i in range(num_samples):
        # Sample plant parameter ranges from Spec §3.4
        J = rng.uniform(0.005, 0.02)
        B = rng.uniform(0.05, 0.2)
        Tc = rng.uniform(0.02, 0.1)
        T_load_val = rng.uniform(0.0, 0.05)
        omega_ref = rng.uniform(5.0, 50.0)
        
        # Store as dict to pass to worker
        params_dict = {
            "R": 1.0,
            "L": 0.5,
            "Ke": 0.5,
            "Kt": 0.5,
            "J": J,
            "B": B,
            "Tc": Tc,
            "Vmax": 24.0,
            "epsilon": 0.01,
            "T_load_val": T_load_val
        }
        
        tasks.append((i, params_dict, omega_ref))
        
    # Resume support: every finished trial is appended to a JSONL checkpoint, so an
    # interrupted run (hours long) restarts where it stopped. Trial i is fully
    # determined by (seed, i), so already-done trials are simply skipped.
    ckpt_path = output_path + ".partial.jsonl"
    results = []
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            results = [json.loads(line) for line in f if line.strip()]
        done = {r["trial"] for r in results}
        tasks = [t for t in tasks if t[0] not in done]
        print(f"Resuming: {len(done)} trials already in {ckpt_path}, {len(tasks)} left.")

    # Run parallel processing
    print(f"Using {num_workers} worker processes in parallel.")
    
    start_time = time.perf_counter()
    
    success_count = len(results)
    n_todo = len(tasks)
    
    with Pool(processes=num_workers) as pool, open(ckpt_path, "a") as ckpt:
        # We use imap_unordered to process tasks dynamically and print progress
        for idx, res in enumerate(pool.imap_unordered(worker_task, tasks)):
            if res["status"] == "success":
                results.append(res)
                success_count += 1
                ckpt.write(json.dumps(res) + "\n")
                ckpt.flush()
            else:
                print(f"  [WARNING] Trial failed: {res['error']}")
                
            if (idx + 1) % 10 == 0 or (idx + 1) == n_todo:
                elapsed = time.perf_counter() - start_time
                avg_time = elapsed / (idx + 1)
                est_remaining = avg_time * (n_todo - (idx + 1))
                print(f"  Progress: {idx+1}/{n_todo} completed ({success_count} success) | "
                      f"Elapsed: {elapsed:.1f}s | Est. Remaining: {est_remaining:.1f}s")
                sys.stdout.flush()

    # Save to CSV
    headers = ["J", "B", "Tc", "T_load", "omega_ref", "Kp", "Ki", "Kd", "achieved_ITAE"]
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in sorted(results, key=lambda r: r["trial"]):
            writer.writerow([
                f"{r['J']:.6f}",
                f"{r['B']:.6f}",
                f"{r['Tc']:.6f}",
                f"{r['T_load']:.6f}",
                f"{r['omega_ref']:.4f}",
                f"{r['Kp']:.6f}",
                f"{r['Ki']:.6f}",
                f"{r['Kd']:.6f}",
                f"{r['achieved_ITAE']:.6f}"
            ])

    os.remove(ckpt_path)  # CSV is complete; the checkpoint is no longer needed
    total_time = time.perf_counter() - start_time
    print(f"\nDataset generation complete!")
    print(f"  Saved {len(results)} successful samples to {output_path}")
    print(f"  Total time elapsed: {total_time:.1f} seconds ({total_time/60.0:.2f} minutes)")


if __name__ == "__main__":
    # Setup argument parser to easily support short tests
    parser = argparse.ArgumentParser(description="Generate PID tuning dataset.")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="data/legacy/tuning_dataset.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default=1)")
    args = parser.parse_args()
    
    generate_dataset(num_samples=args.samples, output_path=args.output, seed=args.seed, num_workers=args.workers)
