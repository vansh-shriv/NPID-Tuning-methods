"""
Closed-Loop Simulation Engine (Spec Section 1.5)

Plant-agnostic: works with any `src.plant.Plant`. The plant owns its integrator
(`plant.step`), so DCMotorParams keeps its hand-inlined fast RK4 while generic
plants use a generic RK4.

Discrete-time stepping approach: at each timestep, compute error, compute
controller output, integrate plant forward one step using RK4.

Uses manual fixed-step RK4 (not scipy.solve_ivp per timestep) for speed —
this is critical for PSO in Week 2 which runs ~800 simulations (Spec §3.3, Gotcha #4).
"""

import numpy as np
import math
from collections import namedtuple
from src.plant import Plant, DCMotorParams, dc_motor_dynamics  # noqa: F401 (re-exported)


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


def run_simulation(controller, plant, omega_ref, t_end=3.0, dt=0.001,
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
        plant: Any `Plant` (DCMotorParams, TransferFunctionPlant, StateSpacePlant, FunctionPlant, ...).
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

    # Initial state (plant-defined; [i=0, ω=0] for the DC motor)
    state = plant.initial_state()
    u_min, u_max = plant.u_min, plant.u_max

    for k in range(n_steps):
        t = k * dt
        y = plant.output(state)

        # Log state before update
        time_arr[k] = t
        current_arr[k] = state[0]
        omega_arr[k] = y

        # Measurement (with optional noise for Week 4 robustness tests)
        if noise_sigma > 0:
            y_measured = y + rng.normal(0, noise_sigma)
        else:
            y_measured = y

        # Compute error and control action
        error = omega_ref - y_measured
        error_arr[k] = omega_ref - y  # Log true error (not noisy) for metrics
        u = controller.compute(error)

        # Clip control signal to actuator limits
        u = min(max(u, u_min), u_max)
        voltage_arr[k] = u

        # Integrate plant forward one step (plant-specific fast RK4 or generic RK4)
        state = plant.step(t, state, u, T_load_func(t), dt)

    return SimResult(
        time=time_arr,
        omega=omega_arr,
        current=current_arr,
        voltage=voltage_arr,
        error=error_arr,
        reference=omega_ref,
    )
