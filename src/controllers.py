"""
PID Controllers (Spec Section 2)

Implements:
  - LPID: Standard Linear PID with anti-windup (Spec §2.1, Eq.1)
  - NPID: Nonlinear PID with error-dependent gain shaping (Spec §2.2, Eq.9-12)
"""

import numpy as np
import math


class LPID:
    """
    Standard Linear PID controller with back-calculation anti-windup.

    u(t) = Kp*e(t) + Ki*∫e(τ)dτ + Kd*de(t)/dt   (Spec Eq. 1)

    Anti-windup (Spec §2.1, Gotcha #2): when the output saturates at
    [u_min, u_max], the integral term is adjusted to prevent windup.
    """

    def __init__(self, Kp, Ki, Kd, dt, u_min=-24.0, u_max=24.0):
        """
        Args:
            Kp, Ki, Kd: PID gains.
            dt: Controller timestep (s).
            u_min, u_max: Output saturation limits (V).
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max

        self._integral = 0.0
        self._prev_error = 0.0
        self._first_call = True

    def compute(self, error):
        """
        Compute control output from current error.

        Args:
            error: e(t) = reference - measured.

        Returns:
            u: Control signal, clipped to [u_min, u_max].
        """
        # Proportional term
        P = self.Kp * error

        # Integral term: accumulate before computing
        self._integral += error * self.dt
        I = self.Ki * self._integral

        # Derivative term: finite difference
        if self._first_call:
            D = 0.0
            self._first_call = False
        else:
            D = self.Kd * (error - self._prev_error) / self.dt

        # Raw output
        u_raw = P + I + D
 
        # Clip to saturation limits (using fast native min/max)
        u = max(self.u_min, min(u_raw, self.u_max))
 
        # Anti-windup: back-calculate integral if saturated
        if u != u_raw and self.Ki != 0:
            self._integral = (u - P - D) / self.Ki
 
        self._prev_error = error
        return u

    def reset(self):
        """Reset controller state for a fresh simulation run."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._first_call = True

    def get_gains(self):
        """Return current gains as a dict."""
        return {"Kp": self.Kp, "Ki": self.Ki, "Kd": self.Kd}

    def __repr__(self):
        return f"LPID(Kp={self.Kp:.4f}, Ki={self.Ki:.4f}, Kd={self.Kd:.4f})"


class NPID:
    """
    Nonlinear PID controller with error-dependent gain shaping functions.

    u(t) = Kp*α(e)*e + Ki*β(e)*∫e(τ)dτ + Kd*γ(de/dt)*de/dt   (Spec Eq. 9)

    Nonlinear functions (Spec Eq. 10-12):
      α(e) = e / (1 + |e|^n)^(1/n)         — proportional gain shaping
      β(e) = 1 / (1 + |e|^m)               — integral gain shaping (anti-windup)
      γ(de/dt) = 1 / (1 + |de/dt|^p)       — derivative gain shaping (noise suppression)

    Note on β(e): provides inherent anti-windup behavior (Spec §2.2, Gotcha #3)
    in addition to the explicit clamping anti-windup applied on top.
    """

    def __init__(self, Kp, Ki, Kd, dt, n=2, m=2, p=2, u_min=-24.0, u_max=24.0):
        """
        Args:
            Kp, Ki, Kd: Base PID gains (same as LPID).
            dt: Controller timestep (s).
            n, m, p: Nonlinearity shape parameters (Spec §2.2).
            u_min, u_max: Output saturation limits (V).
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.n = n
        self.m = m
        self.p = p
        self.u_min = u_min
        self.u_max = u_max

        self._integral = 0.0
        self._prev_error = 0.0
        self._first_call = True

    @staticmethod
    def alpha(e, n):
        """Nonlinear proportional gain function (Spec Eq. 10)."""
        if n == 2:
            return e / math.sqrt(1.0 + e*e)
        abs_e = abs(e)
        return e / (1.0 + abs_e**n) ** (1.0 / n)
 
    @staticmethod
    def beta(e, m):
        """Nonlinear integral gain function (Spec Eq. 11)."""
        if m == 2:
            return 1.0 / (1.0 + e*e)
        return 1.0 / (1.0 + abs(e)**m)
 
    @staticmethod
    def gamma(de_dt, p):
        """Nonlinear derivative gain function (Spec Eq. 12)."""
        if p == 2:
            return 1.0 / (1.0 + de_dt*de_dt)
        return 1.0 / (1.0 + abs(de_dt)**p)

    def compute(self, error):
        """
        Compute control output from current error using nonlinear gain shaping.

        Args:
            error: e(t) = reference - measured.

        Returns:
            u: Control signal, clipped to [u_min, u_max].
        """
        # Nonlinear proportional term (Spec Eq. 9-10):
        # α(e) = e / (1 + |e|^n)^(1/n) is itself the shaped error signal.
        # P = Kp * α(e), which gives:
        #   small e: α(e) ≈ e → P ≈ Kp*e (standard linear proportional)
        #   large e: α(e) → sign(e) → P ≈ Kp*sign(e) (bounded, prevents overshoot)
        # Note: the spec's Eq. 9 notation "Kp*α(e)*e" would produce e²/(...)
        # which is always positive and breaks the feedback loop. The behavioral
        # description ("≈ e for small e, saturates for large e") only holds
        # when α(e) is used directly as the proportional signal.
        alpha_e = self.alpha(error, self.n)
        P = self.Kp * alpha_e

        # Nonlinear integral term (Spec Eq. 9, 11):
        # β(e) = 1/(1+|e|^m) acts as a gain on the integral accumulation.
        # During large transients (β → 0), integral barely grows.
        # Near setpoint (β → 1), standard integral behavior is restored.
        beta_e = self.beta(error, self.m)
        self._integral += beta_e * error * self.dt
        I = self.Ki * self._integral

        # Nonlinear derivative term (Spec Eq. 9, 12):
        # γ(de/dt) = 1/(1+|de/dt|^p) suppresses derivative for large rate spikes.
        if self._first_call:
            de_dt = 0.0
            self._first_call = False
        else:
            de_dt = (error - self._prev_error) / self.dt

        gamma_de = self.gamma(de_dt, self.p)
        D = self.Kd * gamma_de * de_dt

        # Raw output
        u_raw = P + I + D

        # Clip to saturation limits
        u = max(self.u_min, min(u_raw, self.u_max))
 
        # Anti-windup: prevent integral from accumulating when output saturates.
        if u != u_raw:
            self._integral -= beta_e * error * self.dt
 
        self._prev_error = error
        return u

    def reset(self):
        """Reset controller state for a fresh simulation run."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._first_call = True

    def get_gains(self):
        """Return current gains and shape parameters as a dict."""
        return {
            "Kp": self.Kp, "Ki": self.Ki, "Kd": self.Kd,
            "n": self.n, "m": self.m, "p": self.p,
        }

    def __repr__(self):
        return (f"NPID(Kp={self.Kp:.4f}, Ki={self.Ki:.4f}, Kd={self.Kd:.4f}, "
                f"n={self.n}, m={self.m}, p={self.p})")


if __name__ == "__main__":
    """
    Unit tests for the nonlinear gain functions α, β, γ.
    Verify behavior at known input values (Spec §7 — independently testable).
    """
    print("=" * 60)
    print("Controller Unit Tests: Nonlinear Gain Functions")
    print("=" * 60)

    n, m, p = 2, 2, 2
    passed = 0
    total = 0

    def check(name, got, expected, tol=1e-4):
        global passed, total
        total += 1
        ok = abs(got - expected) < tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: got {got:.6f}, expected {expected:.6f}")
        if ok:
            passed += 1

    # alpha(e) tests -- Eq. 10: alpha(e) = e / (1 + |e|^n)^(1/n)
    print("\nalpha(e) -- Nonlinear proportional gain (Eq. 10):")
    check("alpha(0)", NPID.alpha(0.0, n), 0.0)
    check("alpha(1)", NPID.alpha(1.0, n), 1.0 / (1 + 1)**0.5)   # 1/sqrt(2) ~ 0.7071
    check("alpha(-1)", NPID.alpha(-1.0, n), -1.0 / (1 + 1)**0.5) # -1/sqrt(2)
    check("alpha(0.01) ~ 0.01 (linear for small e)", NPID.alpha(0.01, n), 0.01, tol=1e-3)
    check("alpha(100) ~ 1.0 (saturates for large e)", NPID.alpha(100.0, n), 1.0, tol=0.01)

    # alpha(e)*e tests -- the actual term used in the controller
    print("\nalpha(e)*e -- Proportional term behavior:")
    ae_small = NPID.alpha(0.01, n) * 0.01
    check("alpha(0.01)*0.01 ~ 0.0001 (~ e^2)", ae_small, 0.0001, tol=1e-4)
    ae_large = NPID.alpha(100.0, n) * 100.0
    check("alpha(100)*100 ~ 100 (sublinear saturation)", ae_large, 100.0, tol=1.0)

    # beta(e) tests -- Eq. 11: beta(e) = 1 / (1 + |e|^m)
    print("\nbeta(e) -- Nonlinear integral gain (Eq. 11):")
    check("beta(0) = 1", NPID.beta(0.0, m), 1.0)
    check("beta(1) = 0.5", NPID.beta(1.0, m), 0.5)
    check("beta(-1) = 0.5", NPID.beta(-1.0, m), 0.5)
    check("beta(10) ~ 0 (suppresses for large e)", NPID.beta(10.0, m), 0.0, tol=0.01)

    # gamma(de/dt) tests -- Eq. 12: gamma(de/dt) = 1 / (1 + |de/dt|^p)
    print("\ngamma(de/dt) -- Nonlinear derivative gain (Eq. 12):")
    check("gamma(0) = 1", NPID.gamma(0.0, p), 1.0)
    check("gamma(1) = 0.5", NPID.gamma(1.0, p), 0.5)
    check("gamma(-1) = 0.5", NPID.gamma(-1.0, p), 0.5)
    check("gamma(10) ~ 0 (suppresses noise spikes)", NPID.gamma(10.0, p), 0.0, tol=0.01)

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"{'=' * 60}")
