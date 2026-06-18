"""
Performance Metrics (Spec Section 3.2)

Single reusable function for all six metrics used throughout the project.
Every controller/tuning-method comparison goes through this same function
so results are directly comparable.
"""

import numpy as np


def compute_metrics(time, reference, output):
    """
    Compute standard step-response performance metrics.

    Args:
        time: (N,) array of timestamps (s).
        reference: Scalar reference value (e.g., ω_ref).
        output: (N,) array of measured output values (e.g., ω(t)).

    Returns:
        dict with keys:
          - rise_time: Time from 10% to 90% of final value (s).
          - overshoot_pct: Percent overshoot above final value (%).
          - settling_time: Time after which output stays within ±2% of final (s).
          - IAE: Integral of Absolute Error.
          - ISE: Integral of Squared Error.
          - ITAE: Integral of Time-weighted Absolute Error.
    """
    # Steady-state estimate: average of last 100 samples
    final_value = np.mean(output[-100:])

    # Error signal
    error = reference - output

    # --- Rise Time (10% → 90% of final value) ---
    threshold_10 = 0.1 * final_value
    threshold_90 = 0.9 * final_value

    rise_time = np.nan
    if final_value > 0:
        idx_10 = np.where(output >= threshold_10)[0]
        idx_90 = np.where(output >= threshold_90)[0]
        if len(idx_10) > 0 and len(idx_90) > 0:
            t_10 = time[idx_10[0]]
            t_90 = time[idx_90[0]]
            rise_time = t_90 - t_10

    # --- Overshoot (%) ---
    peak_value = np.max(output)
    if final_value > 0 and peak_value > final_value:
        overshoot_pct = 100.0 * (peak_value - final_value) / final_value
    else:
        overshoot_pct = 0.0

    # --- Settling Time (±2% band around final value) ---
    tolerance = 0.02 * abs(final_value)
    settling_time = np.nan
    if final_value != 0:
        within_band = np.abs(output - final_value) <= tolerance
        # Find last index where output is outside the band
        outside_indices = np.where(~within_band)[0]
        if len(outside_indices) > 0:
            last_outside = outside_indices[-1]
            if last_outside < len(time) - 1:
                settling_time = time[last_outside + 1]
            else:
                settling_time = time[-1]  # Never settled
        else:
            settling_time = time[0]  # Always within band

    # --- Integral metrics (numerical integration via trapezoidal rule) ---
    abs_error = np.abs(error)
    # Use np.trapezoid (NumPy 2.x+) with fallback to np.trapz (NumPy 1.x)
    _trapz = getattr(np, 'trapezoid', None) or np.trapz
    IAE = _trapz(abs_error, time)
    ISE = _trapz(error**2, time)
    ITAE = _trapz(time * abs_error, time)

    return {
        "rise_time": rise_time,
        "overshoot_pct": overshoot_pct,
        "settling_time": settling_time,
        "IAE": IAE,
        "ISE": ISE,
        "ITAE": ITAE,
    }


def format_metrics_table(metrics_dict, title="Performance Metrics"):
    """
    Format a dictionary of {controller_name: metrics} into a readable table.

    Args:
        metrics_dict: dict of {str: dict} — e.g., {"L-PID": {...}, "N-PID": {...}}.
        title: Table title.

    Returns:
        Formatted string.
    """
    headers = ["Controller", "Rise Time (s)", "Overshoot (%)", "Settling Time (s)",
               "IAE", "ISE", "ITAE"]
    header_line = " | ".join(f"{h:>16s}" for h in headers)
    separator = "-" * len(header_line)

    lines = [title, separator, header_line, separator]

    for name, m in metrics_dict.items():
        row = [
            f"{name:>16s}",
            f"{m['rise_time']:>16.4f}" if not np.isnan(m['rise_time']) else f"{'N/A':>16s}",
            f"{m['overshoot_pct']:>16.2f}",
            f"{m['settling_time']:>16.4f}" if not np.isnan(m['settling_time']) else f"{'N/A':>16s}",
            f"{m['IAE']:>16.4f}",
            f"{m['ISE']:>16.4f}",
            f"{m['ITAE']:>16.4f}",
        ]
        lines.append(" | ".join(row))

    lines.append(separator)
    return "\n".join(lines)


def metrics_to_csv_row(name, metrics):
    """Convert a single controller's metrics to a CSV-compatible dict."""
    return {
        "Controller": name,
        "Rise Time (s)": f"{metrics['rise_time']:.4f}",
        "Overshoot (%)": f"{metrics['overshoot_pct']:.2f}",
        "Settling Time (s)": f"{metrics['settling_time']:.4f}",
        "IAE": f"{metrics['IAE']:.4f}",
        "ISE": f"{metrics['ISE']:.4f}",
        "ITAE": f"{metrics['ITAE']:.4f}",
    }
