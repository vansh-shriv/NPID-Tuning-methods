"""
Relay-feedback auto-tuning (Astrom-Hagglund), ROADMAP Phase 2.

A relay in the loop drives the plant into a small, self-limiting oscillation, from
which the ultimate gain/period follow via the describing function:

    Ku = 4 d / (pi * sqrt(a^2 - h^2)),   Tu = oscillation period

(d = relay amplitude, a = output oscillation amplitude, h = hysteresis).
Unlike the increasing-Kp search it never drives the plant unstable, and it
needs only one short experiment. The relay swings about `bias`, the steady
input that holds the output at y_ref (pass the plant's trim input; 0 is fine
for plants with zero DC gain offset).
"""

import numpy as np


def relay_autotune(plant, y_ref, d, bias=0.0, hysteresis=0.0, t_end=20.0, dt=0.001,
                   n_cycles=6):
    """
    Returns:
        (Ku, Tu): ultimate gain and period estimated from the last `n_cycles` cycles.

    Raises:
        RuntimeError if no steady oscillation is established.
    """
    n = int(t_end / dt)
    x = plant.initial_state()
    ys = np.zeros(n)
    u_prev = bias + d
    for k in range(n):
        y = plant.output(x)
        ys[k] = y
        e = y_ref - y
        if e > hysteresis:
            u_prev = bias + d
        elif e < -hysteresis:
            u_prev = bias - d
        u = plant.saturate(u_prev)
        x = plant.step(k * dt, x, u, 0.0, dt)

    # upward crossings of the setpoint mark cycle boundaries
    above = ys > y_ref
    up = np.where(~above[:-1] & above[1:])[0]
    if len(up) < n_cycles + 2:
        raise RuntimeError("Relay did not produce a sustained oscillation; "
                           "increase t_end or check relay amplitude/bias.")
    up = up[-(n_cycles + 1):]
    Tu = float(np.mean(np.diff(up)) * dt)
    seg = ys[up[0]:up[-1]]
    a = 0.5 * (np.max(seg) - np.min(seg))
    if a <= hysteresis:
        raise RuntimeError("Oscillation amplitude not above hysteresis.")
    Ku = 4.0 * d / (np.pi * np.sqrt(a * a - hysteresis * hysteresis))
    return float(Ku), Tu
