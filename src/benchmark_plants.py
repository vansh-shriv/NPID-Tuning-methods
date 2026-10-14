"""
Benchmark plant registry (docs/PROTOCOL.md section 3).

Each entry is a *scenario*: a plant factory plus the closed-loop settings used to tune and evaluate it
(setpoint, sampling time, horizon, disturbance, noise level). Time scales and gain boxes are NOT stored
here - they are derived automatically by `TuningProblem`.

`make_plant` always returns a fresh instance: some plants (DelayedPlant) carry state.
"""

import math

from src.plant import (DCMotorParams, DelayedPlant, FunctionPlant, TransferFunctionPlant,
                       dc_motor_dynamics)


def make_plant(name):
    if name == "dc_speed":
        return DCMotorParams()
    if name == "dc_position":
        m = DCMotorParams()
        f = lambda t, x, u, d: [*dc_motor_dynamics(t, x[:2], u, d, m), x[1]]
        return FunctionPlant(f, [0.0, 0.0, 0.0], output_fn=lambda x: x[2], u_limits=(-m.Vmax, m.Vmax), name=name)
    if name == "link":
        J, b, mgl, Tc = 0.05, 0.1, 1.47, 0.02

        def f(t, x, u, d):
            q, w = x
            return [w, (u - d - b * w - mgl * math.sin(q) - Tc * math.tanh(w / 0.01)) / J]
        return FunctionPlant(f, [0.0, 0.0], output_fn=lambda x: x[0], u_limits=(-5.0, 5.0), name=name)
    if name == "fopdt":
        return DelayedPlant(TransferFunctionPlant([2.0], [5.0, 1.0], u_limits=(-10.0, 10.0), name="fopdt"), 1.0)
    if name == "tank":
        f = lambda t, x, u, d: [-0.5 * math.sqrt(max(x[0], 0.0)) + 0.4 * (u + d)]
        return FunctionPlant(f, [0.0], u_limits=(0.0, 10.0), name=name)
    raise KeyError(name)


# setpoint, sampling time, horizon, (disturbance magnitude, onset time), measurement-noise sigma
SCENARIOS = {
    "dc_speed":    dict(y_ref=20.0, dt=0.002, t_end=3.0,  disturbance=(0.05, 1.5), noise_sigma=0.3),
    "dc_position": dict(y_ref=1.0,  dt=0.002, t_end=2.0,  disturbance=(0.05, 1.0), noise_sigma=0.01),
    "link":        dict(y_ref=1.0,  dt=0.002, t_end=4.0,  disturbance=(0.3, 2.0),  noise_sigma=0.01),
    "fopdt":       dict(y_ref=1.0,  dt=0.02,  t_end=60.0, disturbance=(0.5, 30.0), noise_sigma=0.02),
    "tank":        dict(y_ref=4.0,  dt=0.02,  t_end=60.0, disturbance=(0.5, 30.0), noise_sigma=0.04),
}
PLANT_NAMES = list(SCENARIOS)
