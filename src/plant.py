"""
Plant Models (Spec Section 1)

Generic plant interface so the same controllers, tuners (ZN/PSO), metrics and
simulation loop work on *any* single-input single-output plant, linear or
nonlinear:

    Plant                    abstract interface (state_dim, derivatives, output, step)
    DCMotorParams            the project's anchor plant (Spec §1): armature DC motor + Coulomb friction
    DriftingDCMotor          same motor with thermally drifting armature resistance R(t)
    TransferFunctionPlant    G(s) = num(s)/den(s), any proper transfer function
    StateSpacePlant          x' = Ax + B(u + d), y = Cx + Du
    FunctionPlant            arbitrary user ODE x' = f(t, x, u, d) (nonlinear, time-varying, ...)
    DelayedPlant             wraps any plant with input dead time

Conventions shared by every plant:
    u  - scalar control input, saturated to [u_min, u_max] by the simulator
    d  - scalar disturbance (load torque for the DC motor, input-side
         disturbance for the other plants)
    y  - scalar measured/controlled output = plant.output(x)

Adding a new plant only requires implementing `derivatives` and `output`
(plus `state_dim`/`initial_state`); `step` falls back to generic RK4.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import numpy as np


class Plant(ABC):
    """Abstract SISO plant with saturating input."""

    name = "plant"
    u_min = -np.inf
    u_max = np.inf

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Number of state variables."""

    def initial_state(self):
        """State at t=0 (defaults to rest)."""
        return [0.0] * self.state_dim

    @abstractmethod
    def derivatives(self, t, x, u, d=0.0):
        """Return dx/dt for state x, control input u, disturbance d."""

    @abstractmethod
    def output(self, x) -> float:
        """Controlled/measured output y for state x."""

    def step(self, t, x, u, d, dt):
        """Advance the state one fixed step of RK4. Subclasses may override with a faster version."""
        x = np.asarray(x, dtype=float)
        f = self.derivatives
        k1 = np.asarray(f(t, x, u, d), dtype=float)
        k2 = np.asarray(f(t + 0.5 * dt, x + 0.5 * dt * k1, u, d), dtype=float)
        k3 = np.asarray(f(t + 0.5 * dt, x + 0.5 * dt * k2, u, d), dtype=float)
        k4 = np.asarray(f(t + dt, x + dt * k3, u, d), dtype=float)
        return (x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)).tolist()

    @property
    def Vmax(self):
        """Symmetric actuator limit (kept as `Vmax` for backwards compatibility)."""
        return self.u_max

    def saturate(self, u):
        """Clip a control signal to the actuator limits."""
        return min(max(u, self.u_min), self.u_max)


# ----------------------------------------------------------------------------
# DC motor (anchor plant)
# ----------------------------------------------------------------------------

@dataclass
class DCMotorParams(Plant):
    """Armature-controlled DC motor with tanh-smoothed Coulomb friction (Spec §1.2-1.4).

    State x = [i, ω]; output y = ω; disturbance d = T_load (N·m).
    """
    R: float = 1.0       # Armature resistance (Ω)
    L: float = 0.5       # Armature inductance (H)
    Ke: float = 0.5      # Back-EMF constant (V·s/rad)
    Kt: float = 0.5      # Torque constant (N·m/A); Kt = Ke in SI
    J: float = 0.01      # Rotor + load inertia (kg·m²)
    B: float = 0.1       # Viscous friction coefficient (N·m·s/rad)
    Tc: float = 0.05     # Coulomb friction torque (N·m)
    Vmax: float = 24.0   # Maximum applied voltage (V)
    epsilon: float = 0.01  # Smoothing constant for tanh friction

    name = "dc_motor"

    @property
    def state_dim(self):
        return 2

    @property
    def u_max(self):
        return self.Vmax

    @property
    def u_min(self):
        return -self.Vmax

    def derivatives(self, t, x, u, d=0.0):
        return dc_motor_dynamics(t, x, u, d, self)

    def output(self, x):
        return x[1]

    def step(self, t, x, u, d, dt):
        return self._rk4(self.R, x, u, d, dt)

    def _rk4(self, R, x, u, d, dt):
        """Hand-inlined RK4 (plain floats, no numpy temporaries) - ~63x faster than solve_ivp.

        `R` is passed explicitly so subclasses can make it time-varying. Must stay
        numerically identical to `dc_motor_dynamics`.
        """
        L, Ke, Kt, J, B, Tc, eps = (self.L, self.Ke, self.Kt,
                                    self.J, self.B, self.Tc, self.epsilon)
        tanh = math.tanh
        i, w = x

        di1 = (u - R * i - Ke * w) / L
        dw1 = (Kt * i - B * w - d - Tc * tanh(w / eps)) / J

        i2 = i + 0.5 * dt * di1
        w2 = w + 0.5 * dt * dw1
        di2 = (u - R * i2 - Ke * w2) / L
        dw2 = (Kt * i2 - B * w2 - d - Tc * tanh(w2 / eps)) / J

        i3 = i + 0.5 * dt * di2
        w3 = w + 0.5 * dt * dw2
        di3 = (u - R * i3 - Ke * w3) / L
        dw3 = (Kt * i3 - B * w3 - d - Tc * tanh(w3 / eps)) / J

        i4 = i + dt * di3
        w4 = w + dt * dw3
        di4 = (u - R * i4 - Ke * w4) / L
        dw4 = (Kt * i4 - B * w4 - d - Tc * tanh(w4 / eps)) / J

        return [i + (dt / 6.0) * (di1 + 2 * di2 + 2 * di3 + di4),
                w + (dt / 6.0) * (dw1 + 2 * dw2 + 2 * dw3 + dw4)]


@dataclass
class DriftingDCMotor(DCMotorParams):
    """DC motor whose armature resistance heats up:  R(t) = R * (1 + k_heat * t).

    Copper resistivity rises with winding temperature, so R grows during a long
    run (Phase 5 drift model). R is held constant over each integration step
    (evaluated at the step's start), matching the original RL environment.
    `R` is the cold (t=0) resistance.
    """
    k_heat: float = 0.0   # 1/s; 0.05 doubles R over 20 s

    name = "dc_motor_drift"

    def resistance(self, t):
        return self.R * (1.0 + self.k_heat * t)

    def step(self, t, x, u, d, dt):
        return self._rk4(self.resistance(t), x, u, d, dt)

    def derivatives(self, t, x, u, d=0.0):
        cold = self.R
        self.R = self.resistance(t)
        try:
            return dc_motor_dynamics(t, x, u, d, self)
        finally:
            self.R = cold


def dc_motor_dynamics(t, state, V, T_load, params):
    """
    State derivatives of the DC motor, `scipy.integrate.solve_ivp`-compatible
    via `lambda t, y: dc_motor_dynamics(t, y, V, T_load, params)`.

    Uses tanh-smoothed Coulomb friction (Spec §1.3) to avoid the discontinuous sign().

    Returns:
        [di_dt, domega_dt]
    """
    i, omega = state

    # V = R*i + L*di/dt + Ke*ω
    di_dt = (V - params.R * i - params.Ke * omega) / params.L

    # J*dω/dt = Kt*i - B*ω - T_load - Tc*tanh(ω/ε)
    T_friction = params.Tc * np.tanh(omega / params.epsilon)
    domega_dt = (params.Kt * i - params.B * omega - T_load - T_friction) / params.J

    return [di_dt, domega_dt]


# ----------------------------------------------------------------------------
# Generic plants
# ----------------------------------------------------------------------------

class StateSpacePlant(Plant):
    """Linear SISO plant  x' = A x + B (u + d),  y = C x + D u.

    Optional `nonlinearity(t, x, u, d) -> array` is *added* to x' so that a
    linear model can be augmented with e.g. friction or saturation terms.
    """

    name = "state_space"

    def __init__(self, A, B, C, D=0.0, u_limits=(-np.inf, np.inf),
                 x0=None, nonlinearity=None, name=None):
        self.A = np.atleast_2d(np.asarray(A, dtype=float))
        self.B = np.asarray(B, dtype=float).reshape(-1)
        self.C = np.asarray(C, dtype=float).reshape(-1)
        self.D = float(D)
        n = self.A.shape[0]
        if self.A.shape != (n, n) or self.B.size != n or self.C.size != n:
            raise ValueError("A must be n×n, B and C must have n entries")
        self.u_min, self.u_max = u_limits
        self._x0 = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float)
        self.nonlinearity = nonlinearity
        if name:
            self.name = name
        self._u_last = 0.0  # direct feed-through term needs the last input

    @property
    def state_dim(self):
        return self.A.shape[0]

    def initial_state(self):
        return self._x0.tolist()

    def derivatives(self, t, x, u, d=0.0):
        x = np.asarray(x, dtype=float)
        dx = self.A @ x + self.B * (u + d)
        if self.nonlinearity is not None:
            dx = dx + np.asarray(self.nonlinearity(t, x, u, d), dtype=float)
        return dx

    def step(self, t, x, u, d, dt):
        self._u_last = u
        return super().step(t, x, u, d, dt)

    def output(self, x):
        return float(self.C @ np.asarray(x, dtype=float) + self.D * self._u_last)


class TransferFunctionPlant(StateSpacePlant):
    """SISO plant from a proper transfer function G(s) = num(s)/den(s).

    Coefficients are in descending powers of s, e.g. 1/(s²+3s+2) -> num=[1], den=[1, 3, 2].
    """

    name = "transfer_function"

    def __init__(self, num, den, u_limits=(-np.inf, np.inf), nonlinearity=None, name=None):
        from scipy.signal import tf2ss
        num = np.atleast_1d(np.asarray(num, dtype=float))
        den = np.atleast_1d(np.asarray(den, dtype=float))
        if len(num) > len(den):
            raise ValueError("Transfer function must be proper (deg num <= deg den)")
        A, B, C, D = tf2ss(num, den)
        super().__init__(A, B, C, float(np.squeeze(D)), u_limits=u_limits,
                         nonlinearity=nonlinearity, name=name)
        self.num, self.den = num, den


class FunctionPlant(Plant):
    """Wrap any ODE x' = f(t, x, u, d) (nonlinear, time-varying, ...) as a plant.

    Example - inverted-pendulum-like or tank-level dynamics:
        FunctionPlant(lambda t, x, u, d: [-x[0] + math.sqrt(abs(u)) + d], x0=[0.0])
    """

    name = "function"

    def __init__(self, f, x0, output_fn=lambda x: x[0], u_limits=(-np.inf, np.inf), name=None):
        self._f = f
        self._x0 = list(np.atleast_1d(np.asarray(x0, dtype=float)))
        self._output_fn = output_fn
        self.u_min, self.u_max = u_limits
        if name:
            self.name = name

    @property
    def state_dim(self):
        return len(self._x0)

    def initial_state(self):
        return list(self._x0)

    def derivatives(self, t, x, u, d=0.0):
        return self._f(t, x, u, d)

    def output(self, x):
        return float(self._output_fn(x))


class DelayedPlant(Plant):
    """Pure input dead time: the plant sees u(t - delay) (process control, networked loops).

    Implemented as a FIFO of `round(delay/dt)` held inputs, so it is exact for the fixed
    step used by `run_simulation`. The FIFO is stateful: `initial_state()` clears it, so
    every simulation must start from `initial_state()` (run_simulation does) and a
    DelayedPlant instance must not be shared between concurrent simulations.
    """

    def __init__(self, plant, delay):
        self.plant, self.delay = plant, float(delay)
        self.name = f"{plant.name}+delay"
        self._fifo = None

    @property
    def state_dim(self):
        return self.plant.state_dim

    @property
    def u_min(self):
        return self.plant.u_min

    @property
    def u_max(self):
        return self.plant.u_max

    def initial_state(self):
        self._fifo = None
        return self.plant.initial_state()

    def derivatives(self, t, x, u, d=0.0):
        return self.plant.derivatives(t, x, u, d)      # undelayed input (only meaningful via step())

    def output(self, x):
        return self.plant.output(x)

    def step(self, t, x, u, d, dt):
        if self._fifo is None:
            from collections import deque
            self._fifo = deque([0.0] * int(round(self.delay / dt)))
        self._fifo.append(u)
        return self.plant.step(t, x, self._fifo.popleft(), d, dt)


if __name__ == "__main__":
    """Sanity checks: DC motor open-loop plus a generic transfer-function plant."""
    import os
    import matplotlib.pyplot as plt

    def open_loop(plant, u, t_end, dt=0.001):
        x = plant.initial_state()
        n = int(t_end / dt)
        t_arr, y_arr = np.arange(n) * dt, np.zeros(n)
        for k in range(n):
            y_arr[k] = plant.output(x)
            x = plant.step(k * dt, x, u, 0.0, dt)
        return t_arr, y_arr

    motor = DCMotorParams()
    t, w = open_loop(motor, 12.0, 3.0)
    print(f"DC motor @12V: steady-state w = {w[-1]:.4f} rad/s")

    tf = TransferFunctionPlant([2.0], [1.0, 3.0, 2.0])   # DC gain 1, poles -1, -2
    t2, y2 = open_loop(tf, 1.0, 8.0)
    print(f"G(s)=2/(s^2+3s+2) step: y(inf) = {y2[-1]:.4f} (expected 1.0000)")

    os.makedirs("results/legacy/figures", exist_ok=True)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6))
    a1.plot(t, w); a1.set_title("DC motor open loop (12 V)"); a1.set_ylabel("ω (rad/s)")
    a2.plot(t2, y2); a2.set_title("Transfer function 2/(s²+3s+2) step"); a2.set_xlabel("t (s)")
    plt.tight_layout()
    plt.savefig("results/legacy/figures/open_loop_response.png", dpi=150)
