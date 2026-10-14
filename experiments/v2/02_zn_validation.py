"""
Experiment 02 - Validate the Ziegler-Nichols critical-gain detectors (ROADMAP Phase 2)

Compares, on plants with analytic Kcr/Tcr: the original detector's behaviour (see
progress.md), the new half-cycle-ratio detector, and relay feedback.

Run from repo root:  python experiments/v2/02_zn_validation.py
"""

import math
import os
import sys
import warnings

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.plant import DCMotorParams, TransferFunctionPlant
from src.tuning_relay import relay_autotune
from src.tuning_zn import find_critical_gain

# name, num, den, true Kcr, true Tcr, sim length (s)
CASES = [
    ("1/((s+1)(s+2)(s+3))", [1], [1, 6, 11, 6], 60.0, 2 * math.pi / math.sqrt(11), 30),
    ("1/(s(s+1)(s+5))", [1], [1, 6, 5, 0], 30.0, 2 * math.pi / math.sqrt(5), 30),
    ("1/(s+1)^4", [1], [1, 4, 6, 4, 1], 4.0, 2 * math.pi, 40),
]


def main():
    rows = []
    for name, num, den, Kt, Tt, te in CASES:
        plant = TransferFunctionPlant(num, den)
        Kz, Tz = find_critical_gain(plant, 1.0, t_end=te, kp_start=0.5, kp_max=500,
                                    verbose=False, mode="linear")
        gdc = num[-1] / den[-1] if den[-1] else 0.0
        # type-0 plant: relay about the trim input holding y=1; integrator plant: relay about y=0
        ref, bias = (1.0, 1.0 / gdc) if gdc else (0.0, 0.0)
        Kr, Tr = relay_autotune(plant, ref, d=1.0, bias=bias, t_end=3 * te)
        rows.append(dict(plant=name, Kcr_true=Kt, Kcr_ratio=Kz, Kcr_relay=Kr,
                         err_ratio_pct=100 * (Kz - Kt) / Kt, err_relay_pct=100 * (Kr - Kt) / Kt,
                         Tcr_true=Tt, Tcr_ratio=Tz, Tcr_relay=Tr))
    df = pd.DataFrame(rows).round(3)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(df.to_string(index=False))
    os.makedirs("results/v2", exist_ok=True)
    df.to_csv("results/v2/02_zn_validation.csv", index=False)

    print("\nDC motor (no linear ultimate gain exists):")
    motor = DCMotorParams()
    try:
        find_critical_gain(motor, 20.0, verbose=False, mode="linear")
    except RuntimeError as e:
        print("  linear mode  ->", str(e).split(".")[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print("  auto mode    -> (Kcr, Tcr) =", find_critical_gain(motor, 20.0, verbose=False))
    bias = (motor.B * 20 + motor.Tc) / motor.Kt * motor.R + motor.Ke * 20   # trim voltage at 20 rad/s
    try:
        print("  relay        -> (Ku, Tu)   =", relay_autotune(motor, 20.0, d=5.0, bias=bias, hysteresis=0.2))
    except RuntimeError as e:
        print("  relay        ->", e)


if __name__ == "__main__":
    main()
