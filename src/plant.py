"""
DC Motor Plant Model (Spec Section 1)

Armature-controlled DC motor with electrical and mechanical dynamics,
including Coulomb (static) friction as a nonlinearity.

State vector: x = [i, ω]
    i  = armature current (A)
    ω  = angular velocity (rad/s)

Control input: V = applied armature voltage (V)
Disturbance:   T_load = external load torque (N·m)
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class DCMotorParams:
    """Parameters for the DC motor plant model (Spec §1.2)."""
    R: float = 1.0       # Armature resistance (Ω)
    L: float = 0.5       # Armature inductance (H)
    Ke: float = 0.5      # Back-EMF constant (V·s/rad)
    Kt: float = 0.5      # Torque constant (N·m/A); Kt = Ke in SI
    J: float = 0.01      # Rotor + load inertia (kg·m²)
    B: float = 0.1       # Viscous friction coefficient (N·m·s/rad)
    Tc: float = 0.05     # Coulomb friction torque (N·m)
    Vmax: float = 24.0   # Maximum applied voltage (V)
    epsilon: float = 0.01 # Smoothing constant for tanh friction


def dc_motor_dynamics(t, state, V, T_load, params):
    """
    Compute state derivatives for the DC motor.

    Uses tanh-smoothed Coulomb friction (Spec §1.3) to avoid numerical
    instability from the discontinuous sign() function.

    Args:
        t: Current time (s) — unused but required by ODE solver interface.
        state: [i, omega] — armature current (A) and angular velocity (rad/s).
        V: Applied armature voltage (V), already clipped to [-Vmax, Vmax].
        T_load: External load torque disturbance (N·m).
        params: DCMotorParams instance.

    Returns:
        [di_dt, domega_dt]: State derivatives.
    """
    i, omega = state

    # Electrical equation: V = R*i + L*di/dt + Ke*ω
    # → di/dt = (V - R*i - Ke*ω) / L
    di_dt = (V - params.R * i - params.Ke * omega) / params.L

    # Coulomb friction: Tc * tanh(ω/ε) — smoothed approximation (Spec §1.3)
    T_friction = params.Tc * np.tanh(omega / params.epsilon)

    # Mechanical equation: J*dω/dt = Kt*i - B*ω - T_load - T_friction
    # → dω/dt = (Kt*i - B*ω - T_load - T_friction) / J
    domega_dt = (params.Kt * i - params.B * omega - T_load - T_friction) / params.J

    return [di_dt, domega_dt]


if __name__ == "__main__":
    """
    Open-loop sanity check: apply constant voltage, verify sensible
    second-order step response that settles in ~0.5-2s (Spec §1.2).
    """
    import matplotlib.pyplot as plt

    params = DCMotorParams()
    V_applied = 12.0  # Constant applied voltage (V)
    T_load = 0.0      # No load disturbance
    dt = 0.001         # Timestep (s)
    t_end = 3.0        # Simulation duration (s)

    # Manual RK4 integration (Spec §3.3 / Gotcha #4 — faster than solve_ivp)
    from src.simulate import rk4_step

    state = [0.0, 0.0]  # [i=0, ω=0]
    n_steps = int(t_end / dt)
    time_arr = np.zeros(n_steps)
    current_arr = np.zeros(n_steps)
    omega_arr = np.zeros(n_steps)

    for k in range(n_steps):
        time_arr[k] = k * dt
        current_arr[k] = state[0]
        omega_arr[k] = state[1]

        # RK4 step
        def f(t, y):
            return dc_motor_dynamics(t, y, V_applied, T_load, params)

        state = rk4_step(f, k * dt, state, dt)

    # Plot open-loop response
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("DC Motor Open-Loop Step Response (V = 12 V)", fontsize=14)

    ax1.plot(time_arr, omega_arr, "b-", linewidth=1.5)
    ax1.set_ylabel("Angular Velocity ω (rad/s)")
    ax1.set_title("Velocity Response")
    ax1.grid(True, alpha=0.3)

    ax2.plot(time_arr, current_arr, "r-", linewidth=1.5)
    ax2.set_ylabel("Armature Current i (A)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Current Response")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/figures/open_loop_response.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print summary
    omega_ss = omega_arr[-1]
    print(f"\nOpen-loop sanity check:")
    print(f"  Applied voltage:       {V_applied} V")
    print(f"  Steady-state velocity: {omega_ss:.4f} rad/s")
    print(f"  Steady-state current:  {current_arr[-1]:.4f} A")
    print(f"  Parameters: R={params.R}, L={params.L}, Ke={params.Ke}, "
          f"Kt={params.Kt}, J={params.J}, B={params.B}, Tc={params.Tc}")
