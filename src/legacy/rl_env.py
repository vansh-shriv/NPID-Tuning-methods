"""
Gymnasium Environment for N-PID Gain Tuning — Phase 5, Section 2 (Phase5_RL_Spec.md)

NPIDTuningEnv wraps the DC motor closed-loop simulation as a Gymnasium environment
so a TD3 agent can learn to tune [Kp, Ki, Kd] gains online, step-by-step.

Key design decisions (from spec):
  - Observation: 6-dim [e, de/dt, leaky_integral, ω/ω_ref, R_proxy, t/T_ep]
  - Action:      3-dim, normalised [-1,1], rescaled to physical gain ranges
  - Action-holding: N=50 physics steps per RL step (agent acts every 0.05 s)
  - Drift:       R(t) = R0*(1+k_heat*t) when drift=True; identity when drift=False
  - Plant ODEs:  Inline RK4 (same structure as simulate.py) so R can be overridden
                 per-step WITHOUT touching plant.py.

Plant files imported (never modified):
  src/plant.py       -> DCMotorParams (baseline parameters)
  src/controllers.py -> NPID.alpha / beta / gamma (static gain-shaping functions)
  src/legacy/rl_drift.py    -> compute_R (drift equation)
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.plant import DCMotorParams, DriftingDCMotor
from src.controllers import NPID


# ---------------------------------------------------------------------------
# Gain rescaling bounds — match PSO search bounds (original project Week 2)
# ---------------------------------------------------------------------------
KP_MIN, KP_MAX = 10.0,   500.0
KI_MIN, KI_MAX = 100.0,  15000.0
KD_MIN, KD_MAX = 0.05,   5.0

# Action-holding period: physics steps per RL decision (spec §2.2: N=50)
ACTION_HOLD_N = 50

# Physics timestep (same as original project throughout)
DT = 0.001  # seconds

# Leaky integrator decay factors
LEAKY_INTEGRAL_ALPHA = 0.99     # ~100-step half-life for error integral obs
LEAKY_PROXY_ALPHA    = 0.995    # ~200-step half-life for R_proxy (~0.5 s window)

# Nonlinearity shape parameters — fixed at n=m=p=2 per spec §1.2
_N = _M = _P = 2


# ---------------------------------------------------------------------------
# Action rescaling helpers
# ---------------------------------------------------------------------------

def _rescale(val: float, src_lo: float, src_hi: float,
             dst_lo: float, dst_hi: float) -> float:
    """Linearly map val from [src_lo, src_hi] to [dst_lo, dst_hi]."""
    t = (val - src_lo) / (src_hi - src_lo)
    return dst_lo + t * (dst_hi - dst_lo)


def action_to_gains(action: np.ndarray):
    """Convert normalised action in [-1,1]^3 to physical (Kp, Ki, Kd)."""
    Kp = _rescale(float(np.clip(action[0], -1, 1)), -1, 1, KP_MIN, KP_MAX)
    Ki = _rescale(float(np.clip(action[1], -1, 1)), -1, 1, KI_MIN, KI_MAX)
    Kd = _rescale(float(np.clip(action[2], -1, 1)), -1, 1, KD_MIN, KD_MAX)
    return Kp, Ki, Kd


def gains_to_action(Kp: float, Ki: float, Kd: float) -> np.ndarray:
    """Inverse: physical gains → normalised action (useful for sanity-check scripts)."""
    a0 = _rescale(Kp, KP_MIN, KP_MAX, -1, 1)
    a1 = _rescale(Ki, KI_MIN, KI_MAX, -1, 1)
    a2 = _rescale(Kd, KD_MIN, KD_MAX, -1, 1)
    return np.array([a0, a1, a2], dtype=np.float32)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class NPIDTuningEnv(gym.Env):
    """
    Gymnasium environment for online N-PID gain tuning on a DC motor.

    The agent selects new (Kp, Ki, Kd) gains every N=50 simulation steps.
    Gains are held constant over those N steps, then the agent observes
    the accumulated system state and receives a reward.

    Args:
        drift (bool):            Enable thermal resistance drift. Default False.
        k_heat (float):          Heating coefficient (1/s) — only used when drift=True.
        episode_length (float):  Episode wall-clock duration (s).
                                 Recommended: 4.0 (static), 20.0 (drift).
        seed (int | None):       RNG seed.
        omega_ref_range (tuple): (lo, hi) for randomised reference speed (rad/s).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        drift: bool = False,
        k_heat: float = 0.05,
        episode_length: float = 4.0,
        seed: int | None = None,
        omega_ref_range: tuple = (5.0, 50.0),
    ):
        super().__init__()

        self.drift = drift
        self.k_heat = k_heat
        self.episode_length = episode_length
        self.omega_ref_range = omega_ref_range

        # Base plant — parameters read but never mutated (drift handled locally)
        self._base_params = DCMotorParams()
        self._plant = (DriftingDCMotor(k_heat=k_heat) if drift else self._base_params)

        # ── Observation space ───────────────────────────────────────────────
        # [ e, de/dt, integral_obs, ω/ω_ref, R_proxy, t/T_ep ]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, 0.0, 0.0],
                         dtype=np.float32),
            high=np.array([ np.inf,  np.inf,  np.inf,  np.inf, np.inf, 1.0],
                          dtype=np.float32),
            dtype=np.float32,
        )

        # ── Action space ────────────────────────────────────────────────────
        # Normalised to [-1, 1]; rescaled to physical ranges inside step()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Seed the internal RNG
        self._rng = np.random.default_rng(seed)

        # All mutable state initialised in reset()
        self._omega_ref: float = 20.0
        self._state: list = [0.0, 0.0]   # [i, ω]
        self._t: float = 0.0
        self._ctrl_integral: float = 0.0
        self._integral_obs: float = 0.0
        self._leaky_u: float = 0.0
        self._leaky_e: float = 0.0
        self._prev_error: float = 0.0
        self._prev_action_norm: np.ndarray = np.zeros(3, dtype=np.float32)
        self._Kp: float = (KP_MIN + KP_MAX) / 2.0
        self._Ki: float = (KI_MIN + KI_MAX) / 2.0
        self._Kd: float = (KD_MIN + KD_MAX) / 2.0

    # ── Gymnasium API ───────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Randomise reference speed over training distribution
        self._omega_ref = float(self._rng.uniform(*self.omega_ref_range))

        # Reset physical + bookkeeping state
        self._state = [0.0, 0.0]
        self._t = 0.0
        self._ctrl_integral = 0.0
        self._integral_obs = 0.0
        self._leaky_u = 0.0
        self._leaky_e = 0.0
        self._prev_error = 0.0
        self._prev_action_norm = np.zeros(3, dtype=np.float32)

        # Mid-range initial gains (agent replaces on first step)
        self._Kp = (KP_MIN + KP_MAX) / 2.0
        self._Ki = (KI_MIN + KI_MAX) / 2.0
        self._Kd = (KD_MIN + KD_MAX) / 2.0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Decode action to physical gains
        Kp, Ki, Kd = action_to_gains(action)
        self._Kp, self._Ki, self._Kd = Kp, Ki, Kd

        # Smoothness penalty on normalised action delta
        # Weight reduced 0.01 -> 0.002 (spec §8.4: smoothness was discouraging
        # the agent from applying meaningful control effort near setpoint)
        delta_norm = action - self._prev_action_norm
        smoothness_penalty = 0.002 * float(np.sum(np.abs(delta_norm)))
        self._prev_action_norm = action.copy()

        params = self._base_params
        plant = self._plant
        itae_contribution = 0.0
        overshoot_penalty = 0.0
        terminated = False

        # ── Inner physics loop: N=50 steps held at current gains ───────────
        for _ in range(ACTION_HOLD_N):
            i, omega = self._state
            t = self._t

            # Tracking error & derivative
            error = self._omega_ref - omega
            de_dt = (error - self._prev_error) / DT

            # ── N-PID control law (uses NPID static shaping functions) ──────
            alpha_e = NPID.alpha(error, _N)
            beta_e  = NPID.beta(error, _M)
            gamma_e = NPID.gamma(de_dt, _P)

            # Controller integral with anti-windup (identical to controllers.py logic)
            self._ctrl_integral += beta_e * error * DT

            P     = Kp * alpha_e
            I     = Ki * self._ctrl_integral
            D     = Kd * gamma_e * de_dt
            u_raw = P + I + D
            V     = max(-params.Vmax, min(u_raw, params.Vmax))

            if V != u_raw:  # anti-windup
                self._ctrl_integral -= beta_e * error * DT

            # ── Update leaky averages (for R_proxy observation) ─────────────
            self._leaky_u = (LEAKY_PROXY_ALPHA * self._leaky_u
                             + (1 - LEAKY_PROXY_ALPHA) * abs(V))
            self._leaky_e = (LEAKY_PROXY_ALPHA * self._leaky_e
                             + (1 - LEAKY_PROXY_ALPHA) * abs(error))

            # ── Update bounded integral observation ─────────────────────────
            self._integral_obs = (LEAKY_INTEGRAL_ALPHA * self._integral_obs
                                  + error * DT)

            self._prev_error = error

            # ── Plant step (DriftingDCMotor applies R(t) when drift is on) ──
            i_new, omega_new = plant.step(t, self._state, V, 0.0, DT)

            self._state = [i_new, omega_new]
            self._t += DT

            # ITAE contribution (use current t after increment)
            itae_contribution += self._t * abs(error) * DT

            # Overshoot penalty — hard cost for exceeding 2.0× reference
            # Threshold raised 1.5 -> 2.0: the 7% PSO overshoot (omega_peak ~21.4
            # at ref=20) was being clipped by 1.5× (=30) but not triggering;
            # however raising it gives cleaner gradient for mild overshoot cases.
            if abs(omega_new) > 2.0 * self._omega_ref:
                overshoot_penalty += 50.0 * (abs(omega_new) - 2.0 * self._omega_ref)

            # Runaway early termination
            if abs(omega_new) > 5.0 * self._omega_ref:
                terminated = True
                break

        if self._t >= self.episode_length:
            terminated = True

        reward = float(-itae_contribution - smoothness_penalty - overshoot_penalty)

        obs = self._get_obs()
        info = {
            "t": self._t,
            "omega": self._state[1],
            "omega_ref": self._omega_ref,
            "Kp": self._Kp,
            "Ki": self._Ki,
            "Kd": self._Kd,
        }
        return obs, reward, terminated, False, info

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build the 6-dim observation vector."""
        _, omega = self._state
        error = self._omega_ref - omega
        de_dt = (error - self._prev_error) / DT if self._t > 0.0 else 0.0
        R_proxy = self._leaky_u / (self._leaky_e + 1e-6)
        return np.array([
            error,
            de_dt,
            self._integral_obs,
            omega / (self._omega_ref + 1e-6),
            R_proxy,
            self._t / self.episode_length,
        ], dtype=np.float32)

    # Public alias so validation scripts can call env.gains_to_action(...)
    @staticmethod
    def gains_to_action(Kp: float, Ki: float, Kd: float) -> np.ndarray:
        return gains_to_action(Kp, Ki, Kd)
