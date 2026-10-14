"""
Process identification from plant experiments (M0), used by the classical and model-based tuners.

A *bump test* is the standard industrial experiment: hold the plant at the operating point (the constant
input that brings the output to the setpoint), apply a small step, record the response. From the unit-step
response s(t) = (y(t) - y0)/du we fit

    FOPDT   G(s) = K e^{-theta s} / (tau s + 1)               self-regulating plants
    IPDT    G(s) = Kv e^{-theta s} / (s (tau s + 1))           integrating plants (no steady state)

Plants that are unstable or have a negative steady-state gain are outside the scope of the study
(docs/PROTOCOL.md section 2) and raise `NotApplicable`.
Every experiment is run through the `TuningProblem`, which counts it.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from src.tuners.base import NotApplicable, OpenLoopRun


@dataclass
class ProcessModel:
    kind: str                 # "fopdt", "sopdt" or "ipdt"
    K: float                  # steady-state gain (fopdt) or velocity gain Kv (ipdt), output units per input unit [per s]
    tau: float                # lag time constant (s)
    theta: float              # dead time (s)
    nrmse: float              # model-vs-data RMSE / |response range|: identification quality
    method: str = "lsq"
    u_op: float = float("nan")
    y_op: float = float("nan")
    du: float = float("nan")
    data: dict = field(default_factory=dict, repr=False)   # t, s (unit-step response), s_fit
    tau2: float = 0.0         # second lag (sopdt only); tau >= tau2

    @property
    def theta_eff(self):
        """Apparent dead time including the lag of an integrating process (IPDT approximation)."""
        return self.theta + (self.tau if self.kind == "ipdt" else 0.0)

    @property
    def poor_fit(self):
        return self.nrmse > 0.1


# ----------------------------------------------------------------------------- model responses
def fopdt_step(t, K, tau, theta):
    """Unit-step response of K e^{-theta s}/(tau s + 1)."""
    t = np.asarray(t, dtype=float)
    return np.where(t > theta, K * (1.0 - np.exp(-(np.maximum(t - theta, 0.0)) / tau)), 0.0)


def ipdt_step(t, Kv, tau, theta):
    """Unit-step response of Kv e^{-theta s}/(s (tau s + 1))."""
    t = np.asarray(t, dtype=float)
    x = np.maximum(t - theta, 0.0)
    return np.where(t > theta, Kv * (x - tau * (1.0 - np.exp(-x / tau))), 0.0)


def sopdt_step(t, K, tau1, tau2, theta):
    """Unit-step response of K e^{-theta s}/((tau1 s+1)(tau2 s+1)), tau1 >= tau2 >= 0."""
    t = np.asarray(t, dtype=float)
    x = np.maximum(t - theta, 0.0)
    if tau2 < 1e-9 * max(tau1, 1e-12):
        y = 1.0 - np.exp(-x / tau1)
    elif abs(tau1 - tau2) < 1e-6 * tau1:
        y = 1.0 - (1.0 + x / tau1) * np.exp(-x / tau1)
    else:
        y = 1.0 - (tau1 * np.exp(-x / tau1) - tau2 * np.exp(-x / tau2)) / (tau1 - tau2)
    return np.where(t > theta, K * y, 0.0)


# ----------------------------------------------------------------------------- fitting
def _subsample(t, s, n=500):
    if len(t) <= n:
        return t, s
    idx = np.linspace(0, len(t) - 1, n).astype(int)
    return t[idx], s[idx]


def fit_fopdt_two_point(t, s):
    """Smith's two-point method (t28.3, t63.2). Requires a settled response. Returns (K, tau, theta)."""
    K = float(s[-1])
    if K <= 0:
        raise NotApplicable("negative or zero steady-state gain")
    t283 = t[np.argmax(s >= 0.283 * K)]
    t632 = t[np.argmax(s >= 0.632 * K)]
    tau = 1.5 * (t632 - t283)
    theta = max(t632 - tau, 0.0)
    return K, max(tau, t[1] - t[0]), theta


def fit_fopdt_tangent(t, s):
    """Ziegler's graphical tangent method: tangent at the point of maximum slope. Returns (K, tau, theta)."""
    K = float(s[-1])
    if K <= 0:
        raise NotApplicable("negative or zero steady-state gain")
    from scipy.signal import savgol_filter
    w = max(5, (len(t) // 60) | 1)                      # odd window, ~1.7% of the record: derivative smoothing
    dt = t[1] - t[0]
    sm = savgol_filter(s, w, 2)
    ds = savgol_filter(s, w, 2, deriv=1, delta=dt)
    i = int(np.argmax(ds))
    slope = ds[i]
    t_cross = t[i] - sm[i] / slope                      # tangent crosses zero
    theta = max(t_cross, 0.0)
    tau = max(K / slope - theta, t[1] - t[0]) if slope > 0 else t[-1] / 5
    return K, tau, theta


def fit_fopdt(t, s, method="lsq"):
    """
    Fit a FOPDT model to a unit-step response (starting at t=0, s(0)=0).
    method: "lsq" (nonlinear least squares, default), "two_point", "tangent".
    Returns (K, tau, theta, nrmse).
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    span = max(abs(s[-1]), np.ptp(s), 1e-12)
    dt = t[1] - t[0]
    if method == "two_point":
        K, tau, theta = fit_fopdt_two_point(t, s)
    elif method == "tangent":
        K, tau, theta = fit_fopdt_tangent(t, s)
    elif method == "lsq":
        ts, ss = _subsample(t, s)
        starts = []
        for fn in (fit_fopdt_two_point, fit_fopdt_tangent):
            try:
                starts.append(fn(t, s))
            except NotApplicable:
                pass
        T = t[-1]
        starts += [(float(s[-1]), T / 5, T / 20), (float(s[-1]), T / 10, T / 50)]
        lo, hi = [-np.inf, dt, 0.0], [np.inf, 10 * T, T]
        best = None
        for x0 in starts:
            x0 = np.clip(x0, np.array(lo) + 1e-12, np.array(hi) - 1e-12)
            r = least_squares(lambda x: fopdt_step(ts, *x) - ss, x0, bounds=(lo, hi), x_scale=[abs(x0[0]) + 1e-9, x0[1], x0[2] + dt])
            if best is None or r.cost < best.cost:
                best = r
        K, tau, theta = best.x
    else:
        raise ValueError(method)
    nrmse = float(np.sqrt(np.mean((fopdt_step(t, K, tau, theta) - s) ** 2)) / span)
    return float(K), float(tau), float(theta), nrmse


def fit_sopdt(t, s):
    """
    Fit K e^{-theta s}/((tau1 s+1)(tau2 s+1)) (overdamped, tau1 >= tau2) by least squares.
    Returns (K, tau1, tau2, theta, nrmse). Lightly damped plants are only approximated (check nrmse).
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    K0, tau0, th0, _ = fit_fopdt(t, s, "lsq")
    dt = t[1] - t[0]
    T = t[-1]
    span = max(abs(s[-1]), np.ptp(s), 1e-12)
    ts, ss = _subsample(t, s)
    # parametrize tau2 = r * tau1, r in [0,1]
    f = lambda x: sopdt_step(ts, x[0], x[1], x[1] * x[2], x[3]) - ss
    lo, hi = [-np.inf, dt, 0.0, 0.0], [np.inf, 10 * T, 1.0, T]
    best = None
    for r0, split in ((0.2, 0.5), (0.5, 0.3), (0.05, 0.7)):
        x0 = np.clip([K0, tau0 * (1 + r0), r0, th0 * split], np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        r = least_squares(f, x0, bounds=(lo, hi), x_scale=[abs(K0) + 1e-9, tau0, 0.3, th0 + dt])
        if best is None or r.cost < best.cost:
            best = r
    K, tau1, r, theta = best.x
    nrmse = float(np.sqrt(np.mean((sopdt_step(t, K, tau1, tau1 * r, theta) - s) ** 2)) / span)
    return float(K), float(tau1), float(tau1 * r), float(theta), nrmse


def fit_ipdt(t, s):
    """Fit Kv e^{-theta s}/(s(tau s+1)) to a unit-step response from rest. Returns (Kv, tau, theta, nrmse)."""
    t = np.asarray(t, dtype=float)
    s = np.asarray(s, dtype=float)
    dt = t[1] - t[0]
    T = t[-1]
    span = max(np.ptp(s), 1e-12)
    # initial guess: slope of the last third gives Kv; intercept gives theta+tau
    k0 = int(len(t) * 2 / 3)
    Kv0, c0 = np.polyfit(t[k0:], s[k0:], 1)
    if Kv0 <= 0:
        raise NotApplicable("integrating plant with non-positive velocity gain")
    theta_eff0 = max(-c0 / Kv0, dt)
    ts, ss = _subsample(t, s)
    lo, hi = [0.0, dt, 0.0], [np.inf, T, T]
    best = None
    for frac in (0.1, 0.5, 0.9):
        x0 = [Kv0, max(frac * theta_eff0, 2 * dt), max((1 - frac) * theta_eff0, dt)]
        x0 = np.clip(x0, np.array(lo) + 1e-12, np.array(hi) - 1e-12)
        r = least_squares(lambda x: ipdt_step(ts, *x) - ss, x0, bounds=(lo, hi), x_scale=[Kv0, x0[1], x0[2] + dt])
        if best is None or r.cost < best.cost:
            best = r
    Kv, tau, theta = best.x
    nrmse = float(np.sqrt(np.mean((ipdt_step(t, Kv, tau, theta) - s) ** 2)) / span)
    return float(Kv), float(tau), float(theta), nrmse


# ----------------------------------------------------------------------------- experiments
class NotSelfRegulating(Exception):
    """Internal: the plant has no steady state under a constant input (integrating plant)."""


def _classify_unsettled(t, y, y_ref):
    """Decide what an output that never settled is: integrating (ramp) -> NotSelfRegulating, else NotApplicable."""
    if (not np.all(np.isfinite(y))) or np.max(np.abs(y)) > 1e6 * max(abs(y_ref), 1e-9):
        raise NotApplicable("unstable plant (output diverges under constant input)")
    k = int(len(y) * 0.6)
    tt, yy = t[k:], y[k:]
    slope, icpt = np.polyfit(tt, yy, 1)
    ss_res = float(np.sum((yy - (slope * tt + icpt)) ** 2))
    ss_tot = float(np.sum((yy - yy.mean()) ** 2)) + 1e-30
    r2 = 1.0 - ss_res / ss_tot
    if r2 >= 0.98 and abs(slope) * (tt[-1] - tt[0]) >= 0.02 * abs(y_ref):
        raise NotSelfRegulating
    raise NotApplicable("no steady state and not integrating (sustained oscillation / marginal stability)")


def _steady_output(problem, u, t_max, noise_sigma=0.0, seed=0):
    """Steady-state output under constant input u (waits for settling). Counted as one experiment."""
    run = OpenLoopRun(problem, noise_sigma, seed)
    t, y, settled = run.hold(u, t_max=t_max, tol=0.002 if noise_sigma == 0 else 0.002)
    if settled:
        return float(np.mean(y[-max(5, int(2.0 * problem.Ts / problem.dt) // 4):]))
    _classify_unsettled(t, y, problem.y_ref)


def find_trim(problem, t_max=None, noise_sigma=0.0, seed=0, tol=0.005, max_iter=12):
    """
    Constant input that holds the plant at the setpoint (open-loop operating point), by a damped secant search
    on the steady-state output starting from a modest probe (20% of the actuator range). No actuator extremes
    are visited unless required. Raises NotSelfRegulating for integrating plants, NotApplicable if the setpoint
    is unreachable, the plant is unstable, or its gain is not positive.
    """
    t_max = t_max or 4.0 * max(problem.t_end, 10.0 * problem.Ts)
    ref = problem.y_ref
    lim = max(abs(problem.u_min), abs(problem.u_max))
    u = 0.2 * lim if np.isfinite(lim) else abs(ref)
    u = u if problem.u_max >= abs(problem.u_min) else -u

    def clip(v):
        return min(max(v, problem.u_min), problem.u_max)

    u = clip(u)
    y = _steady_output(problem, u, t_max, noise_sigma, seed)
    for _ in range(8):                                      # dead zone (e.g. friction): need a bigger input
        if abs(y) >= 1e-3 * abs(ref) or abs(u) >= lim:
            break
        u = clip(2 * u)
        y = _steady_output(problem, u, t_max, noise_sigma, seed)
    prev = None
    u_fail = None                                  # smallest input at which the plant stopped being self-regulating
    u_ok = u                                       # largest settled input below the setpoint
    for _ in range(max_iter):
        if abs(y - ref) <= tol * abs(ref):
            return float(u)
        if y < ref:
            u_ok = max(u_ok, u) if u_ok is not None else u
        if prev is not None and abs(u - prev[0]) > 1e-12 and (y - prev[1]) * (u - prev[0]) > 0:
            slope = (y - prev[1]) / (u - prev[0])
        else:
            slope = y / u if (u != 0 and y / u > 0) else None
        if slope is None or slope <= 0:
            raise NotApplicable("non-positive steady-state gain at the operating point")
        u_new = clip(u + (ref - y) / slope)
        if u_fail is not None and not (u_ok < u_new < u_fail):
            u_new = 0.5 * (u_ok + u_fail)          # secant left the bracket (strong nonlinearity): bisect
        if u_new == u:
            raise NotApplicable("setpoint not reachable within the actuator limits")
        try:
            y_new = _steady_output(problem, u_new, t_max, noise_sigma, seed)
        except NotSelfRegulating:
            # self-regulating at lower inputs but running away here: the setpoint lies between u_ok and u_new
            if prev is None and y >= ref:
                raise
            u_fail = u_new if u_fail is None else min(u_fail, u_new)
            u_new = 0.5 * (u_ok + u_fail)
            y_new = _steady_output(problem, u_new, t_max, noise_sigma, seed)
        prev = (u, y)
        u, y = u_new, y_new
    if abs(y - ref) <= 3 * tol * abs(ref):
        return float(u)
    raise NotApplicable("operating-point search did not converge (non-monotone plant?)")


def bump_test(problem, u_trim, t_max=None, rel_step=0.1, noise_sigma=0.0, seed=0):
    """
    Settle at `u_trim`, then apply a step du (10% of the trim input, or 5% of the range if the trim is ~0) in
    the direction with more actuator room, and record until settled. Returns the unit-step response
    (t from the step, s = (y - y0)/du), du and y0.
    """
    t_max = t_max or 4.0 * max(problem.t_end, 10.0 * problem.Ts)
    span = problem.u_max - problem.u_min
    span = span if np.isfinite(span) else 10.0 * abs(u_trim) + 1.0
    du = rel_step * abs(u_trim) if abs(u_trim) > 1e-3 * span else 0.05 * span
    if u_trim + du > problem.u_max:
        du = -du
    run = OpenLoopRun(problem, noise_sigma, seed)
    t1, y1, ok = run.hold(u_trim, t_max=t_max)
    if not ok:
        raise NotApplicable("plant did not settle at the operating point")
    win = max(5, int(2.0 * problem.Ts / problem.dt) // 4)
    y0 = float(np.mean(y1[-win:]))
    t2, y2, _ = run.hold(u_trim + du, t_max=t_max, scale=max(abs(problem.y_ref), abs(du)))
    return t2 - t2[0], (y2 - y0) / du, du, y0


def integrator_test(problem, noise_sigma=0.0, seed=0):
    """
    Response of an integrating plant from rest to a constant input just large enough to move the output by about
    the setpoint during the horizon (input doubled from 2% of the range until it does). Returns (t, s, u, y0).
    """
    lim = max(abs(problem.u_min), abs(problem.u_max))
    lim = lim if np.isfinite(lim) else 10.0 * abs(problem.y_ref)
    u = 0.02 * lim
    for _ in range(8):
        run = OpenLoopRun(problem, noise_sigma, seed)
        t, y, _ = run.hold(u, duration=problem.t_end)
        if abs(y[-1] - y[0]) >= abs(problem.y_ref) or u >= lim:
            break
        u = min(2 * u, lim)
    return t, (y - y[0]) / u, u, float(y[0])


def identify_process(problem, method="lsq", noise_sigma=0.0, seed=0, order=1):
    """
    Identify a FOPDT (self-regulating) or IPDT (integrating) model around the setpoint operating point using
    counted open-loop experiments. Returns a `ProcessModel`; `noise_sigma` adds measurement noise to the tests.
    """
    try:
        u_trim = find_trim(problem, noise_sigma=noise_sigma, seed=seed)
        t, s, du, y0 = bump_test(problem, u_trim, noise_sigma=noise_sigma, seed=seed)
        if order == 2:
            K, tau, tau2, theta, nrmse = fit_sopdt(t, s)
            if K <= 0:
                raise NotApplicable("negative steady-state gain")
            s_fit = sopdt_step(t, K, tau, tau2, theta)
            return ProcessModel("sopdt", K, tau, theta, nrmse, "lsq", u_trim, y0, du,
                                dict(t=t, s=s, s_fit=s_fit), tau2=tau2)
        K, tau, theta, nrmse = fit_fopdt(t, s, method)
        if K <= 0:
            raise NotApplicable("negative steady-state gain")
        s_fit = fopdt_step(t, K, tau, theta)
        return ProcessModel("fopdt", K, tau, theta, nrmse, method, u_trim, y0, du, dict(t=t, s=s, s_fit=s_fit))
    except NotSelfRegulating:
        t, s, u, y0 = integrator_test(problem, noise_sigma=noise_sigma, seed=seed)
        Kv, tau, theta, nrmse = fit_ipdt(t, s)
        s_fit = ipdt_step(t, Kv, tau, theta)
        return ProcessModel("ipdt", Kv, tau, theta, nrmse, "lsq", 0.0, y0, u, dict(t=t, s=s, s_fit=s_fit))
