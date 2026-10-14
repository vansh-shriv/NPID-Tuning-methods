"""
PSO-Based N-PID Gain Tuning (Spec Section 3.3)

Uses Particle Swarm Optimization to search for [Kp, Ki, Kd] that minimize
an ITAE-based fitness function (optionally with an overshoot penalty) for N-PID.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyswarms as ps

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plant import DCMotorParams
from src.controllers import NPID, LPID
from src.simulate import run_simulation
from src.metrics import compute_metrics, format_metrics_table
from src.tuning_zn import find_critical_gain, ziegler_nichols_gains


def run_single_sim(Kp, Ki, Kd, params, omega_ref, t_end=4.0, dt=0.001,
                    T_load_func=None, overshoot_penalty_weight=0.0):
    """
    Run a single closed-loop N-PID simulation and compute its fitness.

    Args:
        Kp, Ki, Kd: PID gains.
        params: DCMotorParams instance.
        omega_ref: Reference velocity (rad/s).
        t_end: Simulation duration (s).
        dt: Timestep (s).
        T_load_func: Disturbance torque function.
        overshoot_penalty_weight: Weight for penalizing overshoot > 10%.

    Returns:
        fitness: ITAE (+ optional overshoot penalty), or a large penalty (1e6) if unstable.
    """
    # Reject negative gains
    if Kp < 0 or Ki < 0 or Kd < 0:
        return 1e6

    # Create controller with candidate gains and fixed shape parameters n=m=p=2
    controller = NPID(Kp=Kp, Ki=Ki, Kd=Kd, dt=dt, n=2, m=2, p=2,
                      u_min=-params.Vmax, u_max=params.Vmax)

    try:
        res = run_simulation(controller, params, omega_ref, t_end=t_end, dt=dt,
                             T_load_func=T_load_func)
        
        # Check for numerical instability (NaN/inf or extreme velocities)
        if np.any(np.isnan(res.omega)) or np.any(np.isinf(res.omega)) or np.max(np.abs(res.omega)) > 10.0 * omega_ref:
            return 1e6

        metrics = compute_metrics(res.time, omega_ref, res.omega)
        
        # Base fitness is ITAE
        fitness = metrics["ITAE"]

        # Add optional penalty for overshoot exceeding 10%
        overshoot = metrics["overshoot_pct"]
        if overshoot_penalty_weight > 0.0 and overshoot > 10.0:
            fitness += overshoot_penalty_weight * (overshoot - 10.0)

        return fitness
    except Exception:
        return 1e6


def pso_objective(X, params, omega_ref, t_end=4.0, dt=0.001,
                  T_load_func=None, overshoot_penalty_weight=0.0):
    """
    Objective function for pyswarms GlobalBestPSO.

    Args:
        X: (N, 3) array of particle positions. Each row represents [Kp, Ki, Kd].
        params: DCMotorParams.
        omega_ref: Reference velocity (rad/s).
        t_end: Sim duration.
        dt: Timestep.
        T_load_func: Disturbance function.
        overshoot_penalty_weight: Overshoot penalty weight.

    Returns:
        costs: (N,) array of fitness values.
    """
    n_particles = X.shape[0]
    costs = np.zeros(n_particles)
    for idx in range(n_particles):
        Kp, Ki, Kd = X[idx]
        costs[idx] = run_single_sim(Kp, Ki, Kd, params, omega_ref, t_end, dt,
                                    T_load_func, overshoot_penalty_weight)
    return costs


def optimize_pso(params, omega_ref, init_gains=None, bounds=None,
                 n_particles=20, iters=40, T_load_func=None,
                 overshoot_penalty_weight=0.0, verbose=True, t_end=4.0):
    """
    Run PSO to find optimal gains for N-PID.

    Args:
        params: DCMotorParams.
        omega_ref: Reference velocity (rad/s).
        init_gains: [Kp, Ki, Kd] to center search bounds around (0.1x to 3x).
        bounds: Explicit (min_bounds, max_bounds) tuple.
        n_particles: Number of PSO particles.
        iters: Number of iterations.
        T_load_func: Disturbance function.
        overshoot_penalty_weight: Overshoot penalty weight.
        verbose: Print progress.

    Returns:
        (best_gains, best_cost) where best_gains is list [Kp, Ki, Kd].
    """
    if bounds is None:
        if init_gains is not None:
            Kp_zn, Ki_zn, Kd_zn = init_gains
            min_bounds = np.array([0.1 * Kp_zn, 0.1 * Ki_zn, 0.1 * Kd_zn])
            max_bounds = np.array([3.0 * Kp_zn, 3.0 * Ki_zn, 3.0 * Kd_zn])
        else:
            # Wide default bounds if no starting point
            min_bounds = np.array([1.0, 10.0, 0.01])
            max_bounds = np.array([500.0, 5000.0, 50.0])
        bounds = (min_bounds, max_bounds)

    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    
    # Configure logging level based on verbose
    import logging
    if not verbose:
        logging.getLogger("pyswarms.single.global_best").setLevel(logging.WARNING)
    else:
        logging.getLogger("pyswarms.single.global_best").setLevel(logging.INFO)

    optimizer = ps.single.GlobalBestPSO(n_particles=n_particles, dimensions=3,
                                         options=options, bounds=bounds)

    cost, pos = optimizer.optimize(
        pso_objective,
        iters=iters,
        verbose=verbose,
        params=params,
        omega_ref=omega_ref,
        t_end=t_end,
        dt=0.001,
        T_load_func=T_load_func,
        overshoot_penalty_weight=overshoot_penalty_weight
    )

    return pos.tolist(), cost


if __name__ == "__main__":
    """
    Run a comparison between Ziegler-Nichols tuned N-PID and PSO optimized N-PID.
    Verify that PSO achieves lower ITAE.
    """
    print("=" * 70)
    print("PSO Gain Tuning and Validation")
    print("=" * 70)

    params = DCMotorParams()
    omega_ref = 20.0
    dt = 0.001
    t_end = 4.0

    # Step 1: Find Z-N baseline
    print("Finding Z-N ultimate gains...")
    Kcr, Tcr = find_critical_gain(params, omega_ref, verbose=False)
    Kp_zn, Ki_zn, Kd_zn = ziegler_nichols_gains(Kcr, Tcr)
    print(f"  Z-N Gains: Kp={Kp_zn:.4f}, Ki={Ki_zn:.4f}, Kd={Kd_zn:.4f}")

    # Step 2: Run PSO centered around Z-N gains
    print("\nRunning PSO Tuning (minimizing ITAE)...")
    best_gains, best_cost = optimize_pso(
        params, omega_ref, init_gains=[Kp_zn, Ki_zn, Kd_zn],
        n_particles=20, iters=40, overshoot_penalty_weight=5.0
    )
    Kp_pso, Ki_pso, Kd_pso = best_gains
    print(f"\n  PSO Optimal Gains: Kp={Kp_pso:.4f}, Ki={Ki_pso:.4f}, Kd={Kd_pso:.4f}")
    print(f"  PSO Min Cost (ITAE + penalty): {best_cost:.4f}")

    # Step 3: Run simulations for comparison
    npid_zn = NPID(Kp=Kp_zn, Ki=Ki_zn, Kd=Kd_zn, dt=dt, n=2, m=2, p=2,
                   u_min=-params.Vmax, u_max=params.Vmax)
    npid_pso = NPID(Kp=Kp_pso, Ki=Ki_pso, Kd=Kd_pso, dt=dt, n=2, m=2, p=2,
                    u_min=-params.Vmax, u_max=params.Vmax)

    res_zn = run_simulation(npid_zn, params, omega_ref, t_end=t_end, dt=dt)
    res_pso = run_simulation(npid_pso, params, omega_ref, t_end=t_end, dt=dt)

    metrics_zn = compute_metrics(res_zn.time, omega_ref, res_zn.omega)
    metrics_pso = compute_metrics(res_pso.time, omega_ref, res_pso.omega)

    metrics_dict = {
        "N-PID (ZN)": metrics_zn,
        "N-PID (PSO)": metrics_pso,
    }
    print(f"\n{format_metrics_table(metrics_dict, 'ZN vs PSO Tuning comparison')}")

    # Verify that PSO-tuned controller outperforms ZN-tuned (lower ITAE)
    if metrics_pso["ITAE"] < metrics_zn["ITAE"]:
        print("  [SUCCESS] PSO-tuned controller achieves lower ITAE than Z-N! [SUCCESS]")
    else:
        print("  WARNING: PSO-tuned controller did not achieve a lower ITAE than Z-N.")

    # Plot step responses
    os.makedirs("results/legacy/figures", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(res_zn.time, res_zn.omega, "r--", label=f"N-PID (ZN) - ITAE: {metrics_zn['ITAE']:.3f}")
    plt.plot(res_pso.time, res_pso.omega, "b-", label=f"N-PID (PSO) - ITAE: {metrics_pso['ITAE']:.3f}")
    plt.axhline(y=omega_ref, color="k", linestyle=":", label="Reference")
    plt.ylabel("Velocity ω (rad/s)")
    plt.xlabel("Time (s)")
    plt.title("Tuning Comparison: Ziegler-Nichols vs Particle Swarm Optimization")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    fig_path = "results/legacy/figures/pso_vs_zn_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved comparison plot: {fig_path}")
