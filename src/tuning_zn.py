"""
Ziegler-Nichols Auto-Tuning (Spec Section 3.1)

Automated critical gain (Kcr) and critical period (Tcr) detection via
proportional-only closed-loop sweeps, followed by the standard Z-N formulas.

For this DC motor (2nd-order system with actuator saturation), the "sustained
oscillation" is a limit cycle induced by the saturation nonlinearity at high Kp,
not linear instability. The detection algorithm measures peak amplitude decay
ratios in the tail of the response to classify oscillation type.

This is the trickiest piece of control logic in Week 1 (Gotcha #7) —
kept isolated and well-tested.
"""

import numpy as np
from scipy.signal import find_peaks
from src.plant import DCMotorParams
from src.controllers import LPID
from src.simulate import run_simulation


def oscillation_growth_ratio(y, time, max_halfcycles=30):
    """
    Per-half-cycle amplitude ratio of the oscillation in a P-only step response.

    Extrema of y (peaks and troughs merged) give half-cycle amplitudes
    a_k = |y_{k+1} - y_k|, which are independent of the unknown steady-state
    offset. For a linear loop a_{k+1}/a_k = r is constant: r < 1 damped,
    r = 1 sustained (Kp = Kcr), r > 1 growing. The median of the first
    `max_halfcycles` ratios is returned, so a loop that grows and then locks
    into a saturation limit cycle still reads as r >= 1.

    Returns:
        (ratio, period, amplitude) or (None, None, None) if fewer than 4 extrema
        (monotone / overdamped response).
    """
    y = np.asarray(y, dtype=float)
    span = np.max(y) - np.min(y)
    if not np.isfinite(span) or span < 1e-12:
        return None, None, None
    prom = 1e-5 * span
    pk, _ = find_peaks(y, prominence=prom)
    tr, _ = find_peaks(-y, prominence=prom)
    ext = np.sort(np.concatenate([pk, tr]))
    if len(ext) < 4:
        return None, None, None
    ext = ext[: max_halfcycles + 2]
    amps = np.abs(np.diff(y[ext]))
    amps = amps[amps > 0]
    if len(amps) < 3:
        return None, None, None
    ratio = float(np.median(amps[1:] / amps[:-1]))
    period = 2.0 * float(np.mean(np.diff(time[ext])))
    return ratio, period, float(np.mean(amps))


def detect_oscillation_type(omega, time, tol=1e-3, **_ignored):
    """
    Classify a P-only step response as "none", "damped", "sustained" or "growing"
    from its half-cycle amplitude ratio (see `oscillation_growth_ratio`).

    Returns:
        (osc_type, period, amplitude)
    """
    ratio, period, amp = oscillation_growth_ratio(omega, time)
    if ratio is None:
        return "none", None, None
    if ratio < 1.0 - tol:
        return "damped", period, amp
    if ratio > 1.0 + tol:
        return "growing", period, amp
    return "sustained", period, amp


def detect_limit_cycle(omega, time, y_ref=None, tail_fraction=0.25, rel_threshold=1e-3):
    """
    Practical "sustained oscillation" test for plants with NO linear ultimate gain.

    Plants like the DC motor (relative degree 2, minimum phase) never go unstable
    in continuous time; at high Kp the sampled, saturated, friction-stiff loop
    instead settles into a small chattering limit cycle. This flags a limit
    cycle when the tail of the response keeps swinging by more than
    `rel_threshold * |y_ref|`, regardless of the decay ratio. It is a numerical /
    saturation artifact rather than a true Kcr, so results found this way should
    be reported as such.
    """
    n = len(omega)
    tail = np.asarray(omega[int(n * (1 - tail_fraction)):], dtype=float)
    ref = abs(y_ref) if y_ref else max(abs(np.mean(tail)), 1e-12)
    if np.ptp(tail) <= rel_threshold * ref:
        return "damped", None, None
    pk, _ = find_peaks(tail, prominence=0.1 * np.ptp(tail))
    if len(pk) < 3:
        return "none", None, None
    t_tail = np.asarray(time[int(n * (1 - tail_fraction)):], dtype=float)
    return "sustained", float(np.mean(np.diff(t_tail[pk]))), float(np.ptp(tail))


def find_critical_gain(params=None, omega_ref=20.0, t_end=8.0, dt=0.001,
                       kp_start=1.0, kp_max=2000.0, verbose=True, mode="auto"):
    """
    Critical gain / period for Ziegler-Nichols.

    mode:
      "linear"      - true marginal stability: half-cycle amplitude ratio crosses 1
                      (accurate to <0.5% on plants with an analytic Kcr).
      "limit_cycle" - onset of a persistent saturation/sampling limit cycle, for
                      plants with no linear Kcr (e.g. this DC motor).
      "auto"        - try "linear"; if no ultimate gain exists below kp_max, warn and
                      fall back to "limit_cycle".
    """
    import warnings
    if mode in ("auto", "linear"):
        try:
            return _find_critical_gain(params, omega_ref, t_end, dt, kp_start, kp_max, verbose,
                                       lambda w, t: detect_oscillation_type(w, t))
        except RuntimeError:
            if mode == "linear":
                raise
            warnings.warn("No linear ultimate gain found; falling back to limit-cycle "
                          "detection (Kcr is a saturation/sampling artifact, not marginal stability).")
    return _find_critical_gain(params, omega_ref, t_end, dt, kp_start, kp_max, verbose,
                               lambda w, t: detect_limit_cycle(w, t, omega_ref))


def _find_critical_gain(params, omega_ref, t_end, dt, kp_start, kp_max, verbose, classify):
    """
    Find the critical gain Kcr and critical period Tcr using the
    Ziegler-Nichols method (Spec S3.1).

    Strategy:
      Phase 1: Exponential coarse sweep (1, 2, 4, 8, ...) to quickly find
               the region where oscillation transitions from damped to sustained.
      Phase 2: Binary search within the bracketed region for precision.

    For a 2nd-order DC motor with actuator saturation, "sustained oscillation"
    is a saturation-induced limit cycle. The detection uses peak amplitude
    decay ratios in the response tail.

    Args:
        params: DCMotorParams (defaults to standard params).
        omega_ref: Reference speed for the tuning test (rad/s).
        t_end: Simulation duration per trial (s).
        dt: Timestep (s).
        kp_start: Starting Kp for the coarse sweep.
        kp_max: Maximum Kp to try before giving up.
        verbose: Print progress.

    Returns:
        (Kcr, Tcr): Critical gain and critical period.

    Raises:
        RuntimeError: If sustained oscillation is not found.
    """
    if params is None:
        params = DCMotorParams()

    if verbose:
        print("=" * 60)
        print("Ziegler-Nichols Critical Gain Search")
        print(f"  omega_ref = {omega_ref} rad/s, t_end = {t_end} s")
        print("=" * 60)

    def test_kp(kp):
        """Run P-only sim and classify oscillation."""
        controller = LPID(Kp=kp, Ki=0.0, Kd=0.0, dt=dt,
                          u_min=params.u_min, u_max=params.u_max)
        result = run_simulation(controller, params, omega_ref, t_end=t_end, dt=dt)
        return classify(result.omega, result.time)

    # Phase 1: Exponential coarse sweep to bracket the critical region
    kp_damped = None
    kp_sustained_or_growing = None

    kp = kp_start
    while kp <= kp_max:
        osc_type, period, amplitude = test_kp(kp)

        if verbose:
            amp_str = f"{amplitude:.6f}" if amplitude is not None else "N/A"
            per_str = f"{period:.4f}" if period is not None else "N/A"
            print(f"  Kp = {kp:8.1f} -> {osc_type:10s} "
                  f"(period = {per_str}, amp = {amp_str})")

        if osc_type == "sustained":
            if kp_damped is None:
                # First Kp already sustained — use it directly
                if verbose:
                    print(f"\n  -> Found sustained oscillation at Kp = {kp:.2f}")
                return kp, period
            kp_sustained_or_growing = kp
            break
        elif osc_type == "growing":
            kp_sustained_or_growing = kp
            break
        else:
            kp_damped = kp

        # Exponential growth for fast coverage
        kp *= 2.0

    if kp_sustained_or_growing is None:
        raise RuntimeError(
            f"Could not find sustained/growing oscillation up to Kp = {kp_max}. "
            f"Try increasing kp_max or adjusting plant parameters."
        )

    if kp_damped is None:
        kp_damped = kp_start / 2.0

    # Phase 2: Binary search between kp_damped and kp_sustained_or_growing
    if verbose:
        print(f"\n  Binary search: [{kp_damped:.1f}, {kp_sustained_or_growing:.1f}]")

    best_sustained_kp = kp_sustained_or_growing
    best_sustained_period = None

    for iteration in range(40):
        kp_mid = (kp_damped + kp_sustained_or_growing) / 2.0
        osc_type, period, amplitude = test_kp(kp_mid)

        if verbose and (iteration < 5 or iteration % 5 == 0):
            per_val = f"{period:.4f}" if period else "N/A"
            print(f"    Iter {iteration:2d}: Kp = {kp_mid:8.2f} -> {osc_type} "
                  f"(period={per_val})")

        if osc_type in ("sustained", "growing"):
            kp_sustained_or_growing = kp_mid
            if period is not None:
                best_sustained_kp = kp_mid
                best_sustained_period = period
        else:
            kp_damped = kp_mid

        # Convergence check
        if (kp_sustained_or_growing - kp_damped) / kp_sustained_or_growing < 0.003:
            break

    # Use the best sustained point found
    Kcr = best_sustained_kp
    if best_sustained_period is not None:
        Tcr = best_sustained_period
    else:
        # Fallback: measure period at the sustained point
        _, Tcr, _ = test_kp(Kcr)

    if Tcr is None:
        raise RuntimeError("Could not measure oscillation period at critical gain.")

    if verbose:
        print(f"\n  -> Converged: Kcr = {Kcr:.4f}, Tcr = {Tcr:.4f} s")

    return Kcr, Tcr


def ziegler_nichols_gains(Kcr, Tcr):
    """
    Compute PID gains from the standard Ziegler-Nichols formulas (Spec S3.1).

    Kp = 0.6 * Kcr
    Ki = 1.2 * Kcr / Tcr   (Ti = Tcr/2, Ki = Kp/Ti)
    Kd = 0.075 * Kcr * Tcr  (Td = Tcr/8, Kd = Kp*Td)

    Args:
        Kcr: Critical (ultimate) gain.
        Tcr: Critical (ultimate) period (s).

    Returns:
        (Kp, Ki, Kd): Tuned PID gains.
    """
    Kp = 0.6 * Kcr
    Ki = 1.2 * Kcr / Tcr
    Kd = 0.075 * Kcr * Tcr
    return Kp, Ki, Kd


if __name__ == "__main__":
    """
    Run the Z-N auto-tuning procedure and print the resulting gains.
    """
    params = DCMotorParams()
    omega_ref = 20.0

    print("Finding critical gain for the DC motor plant...")
    print(f"Plant parameters: R={params.R}, L={params.L}, Ke={params.Ke}, "
          f"Kt={params.Kt}, J={params.J}, B={params.B}, Tc={params.Tc}\n")

    try:
        Kcr, Tcr = find_critical_gain(params, omega_ref)
        Kp, Ki, Kd = ziegler_nichols_gains(Kcr, Tcr)

        print(f"\nZiegler-Nichols Results:")
        print(f"  Kcr = {Kcr:.4f}")
        print(f"  Tcr = {Tcr:.4f} s")
        print(f"  Kp  = {Kp:.4f}")
        print(f"  Ki  = {Ki:.4f}")
        print(f"  Kd  = {Kd:.4f}")
    except RuntimeError as e:
        print(f"\nError: {e}")
