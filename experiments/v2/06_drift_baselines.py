"""
Experiment 06 - Does adaptive control under parameter drift need RL? (ROADMAP Phase 6)

Armature resistance drifts, R(t) = R0 (1 + k_heat t), over a 20 s run at w_ref = 20 rad/s.
Controllers (all N-PID with the dataset's scale-free shaping):

    frozen-PSO        PSO gains for the nominal (t=0) plant, never updated
    frozen-ML         v2 model queried once at t=0 with nominal parameters
    ML+oracle-R       v2 model re-queried every 50 ms with the true R(t)   (upper bound for scheduling)
    ML+estimated-R    same, but R is estimated online from V = R i + Ke w (needs current measurement)
    RL (TD3, drift)   the Phase-5 agent, if models/legacy/td3_npid_drift.zip is available

The v2 model has armature resistance R as an input feature, which the legacy 5-feature model lacked, so
gain scheduling over R is possible without any learning loop. Everything runs on the same
DriftingDCMotor plant that the RL environment now uses.

Run from repo root (after 03/04):  python experiments/v2/06_drift_baselines.py
"""

import importlib.util
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.controllers import NPID
from src.plant import DCMotorParams, DriftingDCMotor
from src.tuning_generic import tune_pso

_spec = importlib.util.spec_from_file_location(
    "gen_v2", os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_generate_dataset_v2.py"))
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

W_REF, T_END, DT = 20.0, 20.0, 0.001
UPDATE_EVERY = 50                    # steps (50 ms), same cadence as the RL agent
WINDOWS = ((0, 5), (5, 15), (15, 20))
OUT = "results/v2"


def ml_gains(bundle, R, w_ref=W_REF, plant=None):
    plant = plant or DCMotorParams()
    x = np.array([[plant.J, plant.B, plant.Tc, 0.0, R, w_ref]])
    return 10 ** bundle["model"].predict(x)[0]


def run(plant, schedule, gains0):
    """Closed loop on a drifting plant. `schedule(t, state, V_prev, est) -> gains or None` every 50 ms."""
    n = int(T_END / DT)
    ctrl = NPID(*gains0, DT, u_min=-plant.Vmax, u_max=plant.Vmax, e_scale=W_REF, de_scale=10 * W_REF)
    x = plant.initial_state()
    err = np.zeros(n)
    V = 0.0
    R_hat, filt = plant.R, 0.0
    for k in range(n):
        t = k * DT
        i, w = x
        err[k] = W_REF - w
        if schedule is not None:
            # online resistance estimate from the electrical equation, steady-state-ish: R = (V - Ke w)/i
            if abs(i) > 0.5 and k > 0:
                r = (V - plant.Ke * w) / i
                if 0.2 < r < 10.0:
                    R_hat = 0.995 * R_hat + 0.005 * r
            if k % UPDATE_EVERY == 0 and k > 0:
                g = schedule(t, R_hat)
                ctrl.Kp, ctrl.Ki, ctrl.Kd = g
        V = ctrl.compute(W_REF - w)
        x = plant.step(t, x, V, 0.0, DT)
    return err


def windowed(err):
    t = np.arange(len(err)) * DT
    out = {}
    for lo, hi in WINDOWS:
        m = (t >= lo) & (t < hi)
        out[f"ITAE[{lo},{hi})"] = float(np.trapezoid((t[m] - lo) * np.abs(err[m]), t[m]))
    out["ITAE_total"] = float(np.trapezoid(t * np.abs(err), t))
    out["IAE_total"] = float(np.trapezoid(np.abs(err), t))
    return out


def rl_error(model_path, k_heat):
    from src.legacy.rl_drift_experiment import run_rl_under_drift
    t, w, e, *_ = run_rl_under_drift(model_path, W_REF, k_heat, T_END)
    tt = np.arange(0, T_END, DT)
    return np.interp(tt, t, e)


def main():
    os.makedirs(OUT, exist_ok=True)
    bundle = joblib.load("models/v2/gain_predictor_v2.joblib")
    nominal = DCMotorParams()
    g_pso, _ = tune_pso("npid", nominal, W_REF, None, n_particles=16, iters=25, t_end=3.0, seed=3,
                        bounds=gen.BOUNDS, log_space=True, e_scale=W_REF, de_scale=10 * W_REF)
    g_pso = [g_pso["Kp"], g_pso["Ki"], g_pso["Kd"]]
    g_ml = ml_gains(bundle, nominal.R)
    print("frozen PSO gains", np.round(g_pso, 3), "| frozen ML gains", np.round(g_ml, 3))

    rows = []
    for k_heat in (0.02, 0.05, 0.10, 0.30):
        plant = lambda: DriftingDCMotor(k_heat=k_heat)
        cases = {
            "frozen-PSO": (None, g_pso),
            "frozen-ML": (None, g_ml),
            "ML+oracle-R": (lambda t, Rh, kh=k_heat: ml_gains(bundle, nominal.R * (1 + kh * t)), g_ml),
            "ML+estimated-R": (lambda t, Rh: ml_gains(bundle, Rh), g_ml),
        }
        errs = {name: run(plant(), sch, g0) for name, (sch, g0) in cases.items()}
        if os.path.exists("models/legacy/td3_npid_drift.zip"):
            try:
                errs["RL (TD3, drift)"] = rl_error("models/legacy/td3_npid_drift", k_heat)
            except Exception as e:                                  # keep the comparison alive
                print("RL evaluation failed:", e)
        for name, e in errs.items():
            rows.append(dict(k_heat=k_heat, R_end=1 + k_heat * T_END, controller=name, **windowed(e)))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/06_drift_baselines.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
