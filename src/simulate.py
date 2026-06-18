"""
Closed-Loop Simulation Engine (Spec Section 1.5)

Discrete-time stepping approach: at each timestep, compute error, compute
controller output, integrate plant forward one step using RK4.

Uses manual fixed-step RK4 (not scipy.solve_ivp per timestep) for speed —
this is critical for PSO in Week 2 which runs ~800 simulations (Spec §3.3, Gotcha #4).
"""

import numpy as np
import math
from collections import namedtuple
from src.plant import DCMotorParams, dc_motor_dynamics


# Simulation result container
SimResult = namedtuple("SimResult", [
    "time",      # (N,) array of timestamps (s)
    "omega",     # (N,) angular velocity (rad/s)
    "current",   # (N,) armature current (A)
    "voltage",   # (N,) applied voltage / controller output (V)
    "error",     # (N,) tracking error = reference - measured (rad/s)
    "reference", # scalar or (N,) reference value
])


def rk4_step(f, t, y, dt):
    """
    Single step of the 4th-order Runge-Kutta method.

    Args:
        f: Callable f(t, y) → dy/dt (list or array).
        t: Current time.
        y: Current state (list or array).
        dt: Timestep.

    Returns:
        y_next: State at t + dt (list).
    """
    y = np.asarray(y, dtype=float)
    k1 = np.asarray(f(t, y), dtype=float)
    k2 = np.asarray(f(t + 0.5 * dt, y + 0.5 * dt * k1), dtype=float)
    k3 = np.asarray(f(t + 0.5 * dt, y + 0.5 * dt * k2), dtype=float)
    k4 = np.asarray(f(t + dt, y + dt * k3), dtype=float)
    y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_next.tolist()


def run_simulation(controller, params, omega_ref, t_end=3.0, dt=0.001,
                   T_load_func=None, noise_sigma=0.0, rng=None):
    """
    Run a closed-loop simulation of the DC motor with a PID controller.

    At each timestep:
      1. Measure ω (with optional noise).
      2. Compute error = ω_ref - ω_measured.
      3. Compute control voltage V = controller.compute(error).
      4. Clip V to [-Vmax, Vmax].
      5. Integrate plant forward one RK4 step.
      6. Log everything.

    Args:
        controller: LPID or NPID instance (must have .compute(error) and .reset()).
        params: DCMotorParams instance.
        omega_ref: Reference angular velocity (rad/s), scalar.
        t_end: Simulation duration (s).
        dt: Timestep (s), default 0.001.
        T_load_func: Callable T_load(t) → load torque (N·m). Defaults to zero.
        noise_sigma: Std dev of Gaussian measurement noise on ω (rad/s).
        rng: numpy random Generator for reproducible noise. Created if None.

    Returns:
        SimResult namedtuple with time, omega, current, voltage, error arrays.
    """
    if T_load_func is None:
        T_load_func = lambda t: 0.0

    if rng is None:
        rng = np.random.default_rng(42)

    # Reset controller state
    controller.reset()

    # Preallocate arrays
    n_steps = int(t_end / dt)
    time_arr = np.zeros(n_steps)
    omega_arr = np.zeros(n_steps)
    current_arr = np.zeros(n_steps)
    voltage_arr = np.zeros(n_steps)
    error_arr = np.zeros(n_steps)

    # Initial state: [i=0, ω=0]
    state = [0.0, 0.0]

    for k in range(n_steps):
        t = k * dt
        i, omega = state

        # Log state before update
        time_arr[k] = t
        current_arr[k] = i
        omega_arr[k] = omega

        # Measurement (with optional noise for Week 4 robustness tests)
        if noise_sigma > 0:
            omega_measured = omega + rng.normal(0, noise_sigma)
        else:
            omega_measured = omega

        # Compute error and control action
        error = omega_ref - omega_measured
        error_arr[k] = omega_ref - omega  # Log true error (not noisy) for metrics
        V = controller.compute(error)

        # Clip voltage to actuator limits
        V = np.clip(V, -params.Vmax, params.Vmax)
        voltage_arr[k] = V

        # Get load torque at current time
        T_load = T_load_func(t)

        # Integrate plant forward one step using optimized inline RK4
        # k1
        di1 = (V - params.R * i - params.Ke * omega) / params.L
        domega1 = (params.Kt * i - params.B * omega - T_load - params.Tc * math.tanh(omega / params.epsilon)) / params.J
        
        # k2
        i2 = i + 0.5 * dt * di1
        omega2 = omega + 0.5 * dt * domega1
        di2 = (V - params.R * i2 - params.Ke * omega2) / params.L
        domega2 = (params.Kt * i2 - params.B * omega2 - T_load - params.Tc * math.tanh(omega2 / params.epsilon)) / params.J
        
        # k3
        i3 = i + 0.5 * dt * di2
        omega3 = omega + 0.5 * dt * domega2
        di3 = (V - params.R * i3 - params.Ke * omega3) / params.L
        domega3 = (params.Kt * i3 - params.B * omega3 - T_load - params.Tc * math.tanh(omega3 / params.epsilon)) / params.J
        
        # k4
        i4 = i + dt * di3
        omega4 = omega + dt * domega3
        di4 = (V - params.R * i4 - params.Ke * omega4) / params.L
        domega4 = (params.Kt * i4 - params.B * omega4 - T_load - params.Tc * math.tanh(omega4 / params.epsilon)) / params.J
        
        # Update state
        state = [
            i + (dt / 6.0) * (di1 + 2 * di2 + 2 * di3 + di4),
            omega + (dt / 6.0) * (domega1 + 2 * domega2 + 2 * domega3 + domega4)
        ]

    return SimResult(
        time=time_arr,
        omega=omega_arr,
        current=current_arr,
        voltage=voltage_arr,
        error=error_arr,
        reference=omega_ref,
    )
