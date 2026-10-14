import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # _legacy_root: run from the repo root
"""
Week 6 Diagnostics — evaluate best_model.zip with high-resolution (1ms) replay.

The initial legacy/run_week6.py evaluation used 50ms (RL-step) resolution, making
ITAE incomparable to the 1ms reference. This script:
  1. Loads best_model.zip (EvalCallback's best, not the final model).
  2. Runs one evaluation episode collecting the gain trajectory [Kp(t), Ki(t), Kd(t)].
  3. Replays those same gains at 1ms using run_simulation, giving a fair ITAE.
  4. Also evaluates final td3_npid_static.zip for comparison.
  5. Reports pass/fail against spec §6 criteria.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import TD3
from src.legacy.rl_env import NPIDTuningEnv, DT, ACTION_HOLD_N
from src.metrics import compute_metrics
from src.simulate import run_simulation
from src.controllers import NPID
from src.plant import DCMotorParams
import math

SEED = 42
OMEGA_REF = 20.0

PSO_GAINS = {"Kp": 127.9988, "Ki": 681.7335, "Kd": 1.8942}
REF = {
    "L-PID (ZN)":    (0.1350, 0.22,  0.1990, 0.1631),
    "N-PID (ZN)":    (0.1340, 8.07,  0.7510, 2.2720),
    "N-PID (PSO)":   (0.1350, 7.17,  0.6480, 0.3466),
    "N-PID (ML,RF)": (0.1350, 7.20,  0.6460, 0.3894),
}


def evaluate_model_hires(model_path: str, omega_ref: float = OMEGA_REF, label: str = ""):
    """
    Run agent to collect gain trajectory, then replay at 1ms for fair ITAE.
    Returns metrics dict.
    """
    model = TD3.load(model_path)
    env = NPIDTuningEnv(
        drift=False,
        episode_length=4.0,
        omega_ref_range=(omega_ref, omega_ref),
        seed=SEED,
    )
    obs, _ = env.reset(seed=SEED)
    done = False

    # Collect (t_start, t_end, Kp, Ki, Kd) for each RL step
    gain_schedule = []
    t_prev = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        gain_schedule.append((t_prev, info["t"], info["Kp"], info["Ki"], info["Kd"]))
        t_prev = info["t"]

    # Replay at 1ms with time-varying gains (piecewise constant)
    params = DCMotorParams()
    n_steps = int(4.0 / DT)
    time_arr  = np.zeros(n_steps)
    omega_arr = np.zeros(n_steps)

    state = [0.0, 0.0]
    ctrl_integral = 0.0
    prev_error = 0.0
    gain_idx = 0

    for k in range(n_steps):
        t = k * DT
        time_arr[k] = t

        # Advance gain_schedule pointer
        while gain_idx < len(gain_schedule) - 1 and t >= gain_schedule[gain_idx][1]:
            gain_idx += 1
        _, _, Kp, Ki, Kd = gain_schedule[gain_idx]

        i, omega = state
        omega_arr[k] = omega

        error = omega_ref - omega
        de_dt = (error - prev_error) / DT

        # N-PID
        alpha_e = NPID.alpha(error, 2)
        beta_e  = NPID.beta(error, 2)
        gamma_e = NPID.gamma(de_dt, 2)

        ctrl_integral += beta_e * error * DT
        u_raw = Kp * alpha_e + Ki * ctrl_integral + Kd * gamma_e * de_dt
        V = max(-params.Vmax, min(u_raw, params.Vmax))
        if V != u_raw:
            ctrl_integral -= beta_e * error * DT

        prev_error = error

        # RK4
        R = params.R
        di1 = (V - R*i       - params.Ke*omega ) / params.L
        dw1 = (params.Kt*i   - params.B*omega   - params.Tc*math.tanh(omega/params.epsilon)) / params.J
        i2  = i + 0.5*DT*di1;  w2 = omega + 0.5*DT*dw1
        di2 = (V - R*i2      - params.Ke*w2    ) / params.L
        dw2 = (params.Kt*i2  - params.B*w2     - params.Tc*math.tanh(w2/params.epsilon)) / params.J
        i3  = i + 0.5*DT*di2;  w3 = omega + 0.5*DT*dw2
        di3 = (V - R*i3      - params.Ke*w3    ) / params.L
        dw3 = (params.Kt*i3  - params.B*w3     - params.Tc*math.tanh(w3/params.epsilon)) / params.J
        i4  = i + DT*di3;      w4 = omega + DT*dw3
        di4 = (V - R*i4      - params.Ke*w4    ) / params.L
        dw4 = (params.Kt*i4  - params.B*w4     - params.Tc*math.tanh(w4/params.epsilon)) / params.J
        i_new     = i     + (DT/6.0)*(di1 + 2*di2 + 2*di3 + di4)
        omega_new = omega + (DT/6.0)*(dw1 + 2*dw2 + 2*dw3 + dw4)
        state = [i_new, omega_new]

    metrics = compute_metrics(time_arr, omega_ref, omega_arr)
    itae_ok   = 0.3 <= metrics["ITAE"] <= 0.6
    over_ok   = metrics["overshoot_pct"] <= 15.0
    print(f"\n  [{label}] High-res metrics (omega_ref={omega_ref}):")
    print(f"    Rise:     {metrics['rise_time']:.4f}s  (ref 0.1350)")
    print(f"    Overshoot:{metrics['overshoot_pct']:.2f}%  (ref 7.17%)")
    print(f"    Settling: {metrics['settling_time']:.4f}s  (ref 0.6480)")
    print(f"    ITAE:     {metrics['ITAE']:.4f}  (target 0.3-0.6)")
    print(f"    ITAE pass:{itae_ok}  Overshoot pass:{over_ok}")
    return metrics


if __name__ == "__main__":
    print("\n" + "="*65)
    print("  WEEK 6 HIGH-RES EVALUATION")
    print("="*65)

    # 1. Reference PSO hi-res (ground truth)
    ctrl = NPID(Kp=PSO_GAINS["Kp"], Ki=PSO_GAINS["Ki"], Kd=PSO_GAINS["Kd"],
                dt=DT, n=2, m=2, p=2, u_min=-24.0, u_max=24.0)
    r = run_simulation(ctrl, DCMotorParams(), OMEGA_REF, t_end=4.0, dt=DT)
    pso_m = compute_metrics(r.time, OMEGA_REF, r.omega)
    print(f"\n  [PSO reference] ITAE={pso_m['ITAE']:.4f}, over={pso_m['overshoot_pct']:.2f}%")

    # 2. Best model (EvalCallback checkpoint)
    best_m = evaluate_model_hires("models/legacy/best_model", label="best_model.zip")

    # 3. Final model
    final_m = evaluate_model_hires("models/legacy/td3_npid_static", label="td3_npid_static.zip")

    # 4. Comparison table
    print("\n" + "="*65)
    print(f"  {'Controller':<24} {'Rise(s)':>8} {'Over(%)':>9} {'Settle(s)':>10} {'ITAE':>8}")
    print("  " + "-"*62)
    for name, v in REF.items():
        print(f"  {name:<24} {v[0]:>8.4f} {v[1]:>9.2f} {v[2]:>10.4f} {v[3]:>8.4f}")
    print("  " + "-"*62)
    for label, m in [("N-PID (RL best_model)", best_m), ("N-PID (RL final)", final_m)]:
        print(f"  {label:<24} {m['rise_time']:>8.4f} {m['overshoot_pct']:>9.2f} "
              f"{m['settling_time']:>10.4f} {m['ITAE']:>8.4f}")
    print()
