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


def detect_oscillation_type(omega, time, t_start_fraction=0.4):
    """
    Analyze the tail of a P-only closed-loop response to classify oscillation.

    Uses peak amplitude decay ratio to distinguish:
      - damped: later peaks are significantly smaller than earlier peaks
      - sustained: peak amplitudes are roughly constant (limit cycle)
      - growing: later peaks are larger than earlier peaks

    Args:
        omega: (N,) angular velocity array.
        time: (N,) time array.
        t_start_fraction: Fraction of simulation to skip (analyze tail only).

    Returns:
        (osc_type, period, amplitude) where:
          osc_type: "damped", "sustained", "growing", or "none"
          period: Average oscillation period (s), or None.
          amplitude: Average peak-to-peak amplitude, or None.
    """
    # Focus on the tail portion of the signal
    t_start = time[-1] * t_start_fraction
    mask = time >= t_start
    omega_tail = omega[mask]
    time_tail = time[mask]

    if len(omega_tail) < 50:
        return "none", None, None

    # Remove DC offset for peak detection
    omega_centered = omega_tail - np.mean(omega_tail)

    # Overall range check
    amp_range = np.max(omega_centered) - np.min(omega_centered)
    if amp_range < 1e-8:
        return "none", None, None

    # Find peaks with adaptive thresholding
    dt = time_tail[1] - time_tail[0]
    min_distance = max(3, int(0.005 / dt))  # At least 5ms between peaks

    # Use a very low height threshold to catch small oscillations
    min_height = 0.001 * amp_range
    peaks, _ = find_peaks(omega_centered, height=min_height, distance=min_distance)

    if len(peaks) < 4:
        return "none", None, None

    # Compute peak amplitudes and periods
    peak_amplitudes = np.abs(omega_centered[peaks])
    peak_times = time_tail[peaks]
    periods = np.diff(peak_times)
    avg_period = np.mean(periods)

    # Classify by comparing first-third and last-third peak amplitudes
    n_peaks = len(peak_amplitudes)
    third = max(1, n_peaks // 3)

    first_third_amp = np.mean(peak_amplitudes[:third])
    last_third_amp = np.mean(peak_amplitudes[-third:])

    if first_third_amp < 1e-12:
        return "none", None, None

    amp_ratio = last_third_amp / first_third_amp
    avg_amplitude = np.mean(peak_amplitudes)

    # Classification thresholds
    if amp_ratio < 0.5:
        return "damped", avg_period, avg_amplitude
    elif amp_ratio > 2.0:
        return "growing", avg_period, avg_amplitude
    else:
        # Between 0.5 and 2.0 — approximately sustained
        return "sustained", avg_period, avg_amplitude


def find_critical_gain(params=None, omega_ref=20.0, t_end=8.0, dt=0.001,
                       kp_start=1.0, kp_max=2000.0, verbose=True):
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
                          u_min=-params.Vmax, u_max=params.Vmax)
        result = run_simulation(controller, params, omega_ref, t_end=t_end, dt=dt)
        return detect_oscillation_type(result.omega, result.time)

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

    for iteration in range(25):
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
        if (kp_sustained_or_growing - kp_damped) / kp_sustained_or_growing < 0.02:
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
