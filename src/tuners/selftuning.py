"""
Self-tuning pole-placement PID (docs/METHODS_ROADMAP.md, phase M2; reused online by M6).

Pipeline (Astrom & Wittenmark, "Adaptive Control", self-tuning regulators):
  1. closed-loop excitation: the setpoint is toggled +-`amp` around its nominal value while the loop runs with
     the current gains (conservative start, then the gains designed so far);
  2. delay/structure selection (batch): ARX(2, 2, nk) fitted for nk = 1..nk_max on the first part of the data, the
     smallest nk within 5% of the best loss is kept; the identification sampling period is then
     Delta = nk * Delta0, so that the delay is one sample (the design below assumes nk = 1);
  3. recursive least squares with forgetting on the *differenced* ARIMAX model
        A(q) dy(k) = B(q) du(k - 1),  A = 1 + a1 q^-1 + a2 q^-2,  B = b0 + b1 q^-1
     (differencing removes the operating-point offset and constant disturbances);
  4. pole placement of the velocity-form PID  du = (s0 + s1 q^-1 + s2 q^-2) e  on the closed-loop polynomial
        c(z^-1) = (1 - q^-1) A + q^-1 B S = 1 + c1 q^-1 + ... + c4 q^-4,
     c1..c3 are assigned to a desired pair of poles (wn, zeta) plus one real auxiliary pole, c4 = b1 s2 is whatever
     the plant gives (checked for stability, design detuned if unstable);
  5. conversion to the parallel gains Kp = -s1 - 2 s2, Ki = (s0 + s1 + s2)/Delta, Kd = s2 Delta.

Honest limitations (documented in the M2 report): a 2nd-order model with a one-sample delay, so plants whose
dynamics need more (lightly damped modes, dead time longer than ~0.3 Ts after resampling) are approximated; the
estimate quality depends on the excitation.
"""

from dataclasses import dataclass, field

import numpy as np

from src.tuners.base import GAIN_NAMES, NotApplicable, Tuner


# ----------------------------------------------------------------------------- estimator
class RLS:
    """Recursive least squares with exponential forgetting and covariance-trace limiting."""

    def __init__(self, n, lam=0.995, p0=1e6, p_max=1e8):
        self.theta = np.zeros(n)
        self.P = p0 * np.eye(n)
        self.lam, self.p_max = lam, p_max

    def fit_batch(self, rows, targets, ridge=1e-9):
        """Initialise from a batch least-squares solution: theta = LS estimate, P = (Phi'Phi + ridge I)^-1."""
        rows = np.asarray(rows, dtype=float)
        info = rows.T @ rows + ridge * np.eye(rows.shape[1])
        self.theta = np.linalg.solve(info, rows.T @ np.asarray(targets, dtype=float))
        self.P = np.linalg.inv(info)

    def update(self, phi, y):
        phi = np.asarray(phi, dtype=float)
        err = y - phi @ self.theta
        Pphi = self.P @ phi
        k = Pphi / (self.lam + phi @ Pphi)
        self.theta = self.theta + k * err
        self.P = (self.P - np.outer(k, Pphi)) / self.lam
        tr = np.trace(self.P)
        if tr > self.p_max:                           # covariance windup guard (poor excitation)
            self.P *= self.p_max / tr
        return err


def arx_regressor(dy, du, k, nk, u=None, integrating=False):
    """
    Regulating:   dy(k) = -a1 dy(k-1) - a2 dy(k-2) + b0 du(k-nk) + b1 du(k-nk-1)        (A dy = B du)
    Integrating:  dy(k) = -a1' dy(k-1) + b0 u(k-nk) + b1 u(k-nk-1) + c                  (A' dy = B u + c)
    The second form is the correct one for plants with an integrator: A = (1 - q^-1)(1 + a1' q^-1), so
    A y = B u  <=>  (1 + a1' q^-1) dy = B u  (the input *level*, not its difference, drives the velocity). The
    constant c absorbs an input offset (Coulomb friction, constant load) that the differenced form removes
    automatically but the level form does not.
    """
    if integrating:
        return np.array([-dy[k - 1], u[k - nk], u[k - nk - 1], 1.0])
    return np.array([-dy[k - 1], -dy[k - 2], du[k - nk], du[k - nk - 1]])


def fit_arx_batch(dy, du, nk, u=None, integrating=False):
    """Least-squares fit of the chosen structure. Returns (theta, mean squared residual)."""
    start = nk + 2
    rows = np.array([arx_regressor(dy, du, k, nk, u, integrating) for k in range(start, len(dy))])
    target = dy[start:]
    theta, *_ = np.linalg.lstsq(rows, target, rcond=None)
    res = target - rows @ theta
    return theta, float(np.mean(res ** 2))


def select_delay(dy, du, nk_max=6, tol=0.05, u=None, integrating=False):
    """Smallest nk whose loss is within `tol` of the best loss over nk = 1..nk_max."""
    losses = []
    for nk in range(1, nk_max + 1):
        if len(dy) < nk + 12:
            break
        losses.append(fit_arx_batch(dy, du, nk, u, integrating)[1])
    best = min(losses)
    for nk, l in enumerate(losses, start=1):
        if l <= (1 + tol) * best:
            return nk, losses
    return len(losses), losses


def choose_structure(dy, du, u, nk_max=6, tol=0.05):
    """
    Regulating or integrating model (4 parameters each), by Akaike's criterion N log(loss) + 2 p at the selected
    delay (equal parameter counts: the lower loss wins). Returns (integrating, nk, losses, aic_regulating, aic_integrating).
    """
    nk_r, loss_r = select_delay(dy, du, nk_max, tol)
    nk_i, loss_i = select_delay(dy, du, nk_max, tol, u=u, integrating=True)
    n = len(dy)
    aic_r = n * np.log(max(loss_r[nk_r - 1], 1e-30)) + 2 * 4
    aic_i = n * np.log(max(loss_i[nk_i - 1], 1e-30)) + 2 * 4
    if aic_i < aic_r:
        return True, nk_i, loss_i, aic_r, aic_i
    return False, nk_r, loss_r, aic_r, aic_i


def to_arx(theta, integrating):
    """(a1, a2, b0, b1) of A = 1 + a1 q^-1 + a2 q^-2, B = b0 + b1 q^-1 from the estimated parameter vector."""
    if integrating:
        a1p, b0, b1 = theta[:3]
        return float(a1p - 1.0), float(-a1p), float(b0), float(b1)      # A = (1 - q^-1)(1 + a1' q^-1)
    return tuple(float(x) for x in theta)


def arx_step_response(a1, a2, b0, b1, n=2000, nk=1):
    """Unit-step response of A y = B u (undifferenced) for the identified ARX; returns y(0..n-1)."""
    y = np.zeros(n)
    for k in range(n):
        y1 = y[k - 1] if k >= 1 else 0.0
        y2 = y[k - 2] if k >= 2 else 0.0
        u1 = 1.0 if k - nk >= 0 else 0.0
        u2 = 1.0 if k - nk - 1 >= 0 else 0.0
        y[k] = -a1 * y1 - a2 * y2 + b0 * u1 + b1 * u2
    return y


# ----------------------------------------------------------------------------- design
@dataclass
class PolePlacementDesign:
    Kp: float
    Ki: float
    Kd: float
    wn: float
    zeta: float
    poles: np.ndarray = field(default_factory=lambda: np.array([]))
    gamma: float = float("nan")          # auxiliary double pole at z = -gamma
    ok: bool = True
    note: str = ""

    @property
    def gains(self):
        return dict(Kp=self.Kp, Ki=self.Ki, Kd=self.Kd)


def dominant_pair(wn, zeta, delta):
    """(d1, d2) of 1 + d1 z^-1 + d2 z^-2 for the sampled continuous poles of (wn, zeta)."""
    r = np.exp(-zeta * wn * delta)
    wd = wn * np.sqrt(max(1.0 - zeta ** 2, 1e-12))
    return -2.0 * r * np.cos(wd * delta), r * r


def _assign(a1, a2, b0, b1, rho, targets, delta):
    """
    Solve for (s0, s1, s2) so that the closed-loop polynomial R A + z^-1 B S has coefficients c1..c3 equal to the
    targets; returns (s, c4_residual) where the residual is c4 - target4. R = (1 - z^-1)(1 - rho z^-1).
    """
    R = np.convolve([1.0, -1.0], [1.0, -rho])
    RA = np.convolve(R, [1.0, a1, a2])
    t1, t2, t3, t4 = targets
    s0 = (t1 - RA[1]) / b0
    s1 = (t2 - RA[2] - b1 * s0) / b0
    s2 = (t3 - RA[3] - b1 * s1) / b0
    c4 = RA[4] + b1 * s2
    return np.array([s0, s1, s2]), c4 - t4


def _to_parallel(s, rho, delta):
    """(s0,s1,s2) of C = S/((1 - z^-1)(1 - rho z^-1)) -> (Kp, Ki, Kd, tau_f) for a PID with a first-order derivative filter."""
    M = np.array([[1.0, 1.0, 1.0], [-(1.0 + rho), -rho, -2.0], [rho, 0.0, 1.0]])
    kp, kid, kappa = np.linalg.solve(M, s)
    tau_f = rho * delta / (1.0 - rho) if rho < 1.0 else np.inf
    return float(kp), float(kid / delta), float(kappa * (tau_f + delta))


def pole_placement_pid(a1, a2, b0, b1, delta, wn, zeta=0.8, tau_d=0.0):
    """
    PID with derivative filter tau_d by (exact) pole placement for A = 1 + a1 q^-1 + a2 q^-2, B = b0 + b1 q^-1 with a
    one-sample delay, sampling period delta: the four closed-loop poles are the pair (wn, zeta) plus a real double
    pole at z = -gamma, gamma found by a root search so that all four coefficients can be matched with the three PID
    parameters. `ok=False` if no admissible solution (stable, Kp > 0, Ki >= 0, Kd >= 0) exists.
    """
    if abs(b0) < 1e-12:
        return PolePlacementDesign(0, 0, 0, wn, zeta, ok=False, note="b0 ~ 0 (no instantaneous gain)")
    rho = tau_d / (tau_d + delta) if tau_d > 0 else 0.0
    d1, d2 = dominant_pair(wn, zeta, delta)

    def targets(g):
        return (d1 + 2 * g, d2 + 2 * g * d1 + g * g, 2 * g * d2 + g * g * d1, g * g * d2)

    f = lambda g: _assign(a1, a2, b0, b1, rho, targets(g), delta)[1]
    grid = np.linspace(-0.97, 0.97, 389)
    vals = np.array([f(g) for g in grid])
    cands = []
    for i in range(len(grid) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
            from scipy.optimize import brentq
            cands.append(brentq(f, grid[i], grid[i + 1]))
    best = None
    for g in cands or [float(grid[int(np.nanargmin(np.abs(vals)))])]:
        s, resid = _assign(a1, a2, b0, b1, rho, targets(g), delta)
        kp, ki, kd = _to_parallel(s, rho, delta)
        R = np.convolve([1.0, -1.0], [1.0, -rho])
        cl = np.convolve(R, [1.0, a1, a2]) + np.convolve([0.0, b0, b1], s)       # closed-loop polynomial in z^-1
        poles = np.roots(cl)
        stable = bool(np.all(np.abs(poles) < 0.995))
        ok = stable and kp > 0 and ki >= 0 and kd >= -1e-12 and np.all(np.isfinite([kp, ki, kd]))
        effort = float(np.linalg.norm(s))
        slow_aux = abs(g) > np.sqrt(d2) + 1e-9                  # auxiliary poles slower than the dominant pair: avoid
        key = (not ok, slow_aux, effort)
        if best is None or key < best[0]:
            best = (key, g, kp, ki, kd, poles, stable)
    key, g, kp, ki, kd, poles, stable = best
    slow_aux = key[1]
    ok = (not key[0]) and not slow_aux                 # the placed pair must really be the dominant one
    note = "" if ok else ("unstable closed loop" if not stable else
                          "auxiliary poles slower than the dominant pair" if slow_aux and not key[0] else "non-positive gain")
    if not cands:
        note = (note + " no exact assignment (least-residual aux pole)").strip()
    return PolePlacementDesign(float(kp), float(ki), float(max(kd, 0.0)), wn, zeta, poles, float(g), ok, note)


def pole_placement_pi(a1, a2, b0, b1, delta, wn, zeta=0.8):
    """
    PI (Kd = 0) by exact pole placement: the closed loop (1 - q^-1) A + q^-1 B S has degree 3, three coefficients are
    matched to (pair (wn, zeta)) x (real auxiliary pole at z = alpha), alpha found by a root search (cf. the PID
    design). Suited to lag-dominant plants where a PID with positive gains cannot place the poles.
    """
    from scipy.optimize import brentq
    if abs(b0) < 1e-12:
        return PolePlacementDesign(0, 0, 0, wn, zeta, ok=False, note="b0 ~ 0 (no instantaneous gain)")
    d1, d2 = dominant_pair(wn, zeta, delta)

    def solve(alpha):
        t1, t2, t3 = d1 - alpha, d2 - alpha * d1, -alpha * d2
        s0 = (t1 - (a1 - 1.0)) / b0
        s1 = (t2 - (a2 - a1) - b1 * s0) / b0
        return s0, s1, (-a2 + b1 * s1) - t3                    # residual of the third coefficient

    grid = np.linspace(-0.97, 0.97, 389)
    vals = np.array([solve(al)[2] for al in grid])
    roots = [brentq(lambda al: solve(al)[2], grid[i], grid[i + 1]) for i in range(len(grid) - 1)
             if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0]
    best = None
    for alpha in roots:
        s0, s1, _ = solve(alpha)
        kp, ki = -s1, (s0 + s1) / delta
        poles = np.roots([1.0, d1 - alpha, d2 - alpha * d1, -alpha * d2])
        ok = bool(np.all(np.abs(poles) < 0.995)) and kp > 0 and ki >= 0
        slow = alpha > np.sqrt(d2) + 1e-9
        key = (not ok, slow, abs(s0) + abs(s1))
        if best is None or key < best[0]:
            best = (key, alpha, kp, ki, poles)
    if best is None:
        return PolePlacementDesign(0, 0, 0, wn, zeta, ok=False, note="PI: no exact assignment")
    key, alpha, kp, ki, poles = best
    ok = (not key[0]) and not key[1]
    note = "PI" if ok else "PI: " + ("unstable or non-positive gain" if key[0] else "auxiliary pole slower than the dominant pair")
    return PolePlacementDesign(float(kp), float(ki), 0.0, wn, zeta, poles, float(-alpha), ok, note)


def design_with_fallback(a1, a2, b0, b1, delta, wn, zeta=0.8, tau_d=0.0, max_detune=12, detune=0.75):
    """
    For the requested wn try PID placement, then PI placement; if neither is admissible detune (wn x 0.75) and repeat.
    The note records PI use and the number of detunings.
    """
    k = 0
    while True:
        d = pole_placement_pid(a1, a2, b0, b1, delta, wn, zeta, tau_d)
        if not d.ok:
            pi = pole_placement_pi(a1, a2, b0, b1, delta, wn, zeta)
            if pi.ok:
                d = pi
        if d.ok or k >= max_detune:
            break
        wn *= detune
        k += 1
    d.note = (d.note + (f" (detuned {k}x)" if k else "")).strip()
    return d


def desired_wn(a1, a2, b0, b1, delta, speedup=3.0, zeta=0.8, nk=1, integrating=False):
    """
    Desired natural frequency from the identified model: closed-loop settling time = open-loop (2%) settling time
    divided by `speedup`;  wn = 4 / (zeta T_set). For an integrating model the open-loop step response does not
    settle; the time scale is then 10 (tau + Delta) with the lag tau of the remaining pole, as in model matching.
    Returns (wn, T_open) or None if the model gives no usable time scale (non-finite, unstable, zero gain).
    """
    if not np.all(np.isfinite([a1, a2, b0, b1])):
        return None
    if integrating:
        alpha = a2                                      # A = (1 - q^-1)(1 - alpha q^-1)
        if not (-0.5 < alpha < 0.999):
            return None
        tau = -delta / np.log(alpha) if alpha > 1e-6 else 0.0
        t_open = 10.0 * (tau + delta)
        return 4.0 / (zeta * max(t_open / speedup, 2 * delta)), t_open
    roots = np.roots([1.0, a1, a2])
    near_one = np.abs(roots - 1.0) < 0.03               # an identified model with an integrator (root at z = 1)
    if np.any(near_one) and not np.all(near_one):
        alpha = float(np.max(np.abs(roots[~near_one])))
        if alpha >= 0.999:
            return None
        tau = -delta / np.log(alpha) if alpha > 1e-6 else 0.0
        t_open = 10.0 * (tau + delta)
        return 4.0 / (zeta * max(t_open / speedup, 2 * delta)), t_open
    if np.any(np.abs(roots) >= 0.999):
        return None                                     # not (yet) a stable model: no time scale to derive
    y = arx_step_response(a1, a2, b0, b1, n=4000, nk=nk)
    final = y[-1]
    if not np.isfinite(final) or abs(final) < 1e-12:
        return None
    idx = np.where(np.abs(y - final) > 0.02 * abs(final))[0]
    t_open = (idx[-1] + 1) * delta if len(idx) else delta
    return 4.0 / (zeta * max(t_open / speedup, 2 * delta)), t_open
# ----------------------------------------------------------------------------- tuner
class SelfTuningPolePlacement(Tuner):
    """
    Self-tuning pole placement used as an auto-tuner: run the excitation experiment in closed loop, adapt the
    gains online, return the converged gains (median of the valid designs in the last 25% of the experiment) as
    static N-PID base gains. The same machinery used online is evaluated in M6.

    Parameters: amp (setpoint toggle amplitude, fraction of |y_ref|), speedup (closed-loop settling = open-loop
    settling / speedup), zeta, lam (forgetting factor), periods (excitation periods: 2 for the delay-selection
    batch, the rest for the recursive phase), noisy (corrupt the measured output with the problem's noise).
    """
    name = "Self-tuning (pole placement)"
    family = "intelligent"
    stochastic = False

    def __init__(self, amp=0.1, speedup=3.0, zeta=0.8, lam=0.995, periods=8, period_Ts=8.0, update_every=5,
                 noisy=False, nk_max=6, kind="npid", allow_integrating=True, sampled=True, name=None):
        self.amp, self.speedup, self.zeta, self.lam = amp, speedup, zeta, lam
        self.periods, self.period_Ts, self.update_every = periods, period_Ts, update_every
        self.noisy, self.nk_max, self.kind = noisy, nk_max, kind
        self.allow_integrating = allow_integrating
        self.sampled = sampled        # sampled-data experiment: controller runs at the identification rate (ZOH input)
        self.trace = {}
        if name:
            self.name = name

    # initial gains: cautious P-I that any stable benchmark plant tolerates
    @staticmethod
    def initial_gains(problem):
        lim = max(abs(problem.u_min), abs(problem.u_max))
        lim = lim if np.isfinite(lim) else 10 * abs(problem.y_ref)
        kp = 0.05 * lim / abs(problem.y_ref)
        return dict(Kp=kp, Ki=kp / (5.0 * problem.Ts), Kd=0.0)

    def _controller(self, problem, gains, dt_c, like=None):
        """Controller with sample time dt_c (state copied from `like`)."""
        from src.controllers import LPID, NPID
        kw = problem.controller_kwargs(self.kind)
        g = {k: float(gains[k]) for k in GAIN_NAMES}
        if self.kind == "lpid":
            c = LPID(g["Kp"], g["Ki"], g["Kd"], dt_c, **kw)
        else:
            c = NPID(g["Kp"], g["Ki"], g["Kd"], dt_c, **problem.shape, **kw)
        if like is not None:
            c._integral, c._prev_error, c._d_filt, c._first_call = like._integral, like._prev_error, like._d_filt, False
        return c

    def _tune(self, problem, budget, seed):
        dt, Ts = problem.dt, problem.Ts
        m0 = max(1, int(round(Ts / 20.0 / dt)))                 # batch sampling interval Delta0 = m0 dt
        T_ex = self.period_Ts * Ts
        n_steps = int(self.periods * T_ex / dt)
        n_A = int(2 * T_ex / dt)                               # end of the batch (delay-selection) phase
        plant = problem.plant_factory()
        x = plant.initial_state()
        m_c = m0 if self.sampled else 1                         # controller update period (steps)
        ctrl = self._controller(problem, self.initial_gains(problem), m_c * dt)
        u_hold = 0.0
        rng = np.random.default_rng(seed)
        problem._count(0.0)
        y_full, u_full = np.zeros(n_steps), np.zeros(n_steps)
        lb, ub = problem.bounds
        designs = []
        trace = dict(t=[], theta=[], gains=[], wn=[], nk=None, delta=None, losses=None)
        rls, m, sy, su, since, integ = None, None, None, None, 0, False

        for k in range(n_steps):
            t = k * dt
            y = plant.output(x)
            y_full[k] = y
            ym = y + (rng.normal(0, problem.noise_sigma) if self.noisy else 0.0)
            r = problem.y_ref * (1.0 + self.amp * (1.0 if (t % T_ex) < T_ex / 2 else -1.0))
            if k % m_c == 0:                                    # zero-order hold between controller updates
                u_hold = min(max(ctrl.compute(r - ym), problem.u_min), problem.u_max)
            u = u_hold
            u_full[k] = u
            x = plant.step(t, x, u, 0.0, dt)

            if k == n_A:                                        # structure selection, RLS warm start
                yA, uA = y_full[:n_A:m0], u_full[:n_A:m0]
                dyA, duA = np.diff(yA, prepend=yA[0]), np.diff(uA, prepend=uA[0])
                if self.allow_integrating:
                    integ, nk, losses, aic_r, aic_i = choose_structure(dyA, duA, uA, self.nk_max)
                else:
                    nk, losses = select_delay(dyA, duA, self.nk_max)
                    integ, aic_r, aic_i = False, np.nan, np.nan
                m = m0 * nk
                if self.sampled:                                # from now on the controller runs at the sampling period
                    m_c = m
                    cur = {n: getattr(ctrl, n) for n in GAIN_NAMES}
                    ctrl = self._controller(problem, cur, m * dt, like=ctrl)
                n_al = (n_A // m) * m                           # last aligned sampling instant of the batch
                sy, su = list(y_full[:n_al + 1:m]), list(u_full[:n_al + 1:m])
                dy, du = np.diff(sy, prepend=sy[0]), np.diff(su, prepend=su[0])
                su_arr = np.array(su)
                rls = RLS(4, self.lam)
                rls.fit_batch([arx_regressor(dy, du, j, 1, su_arr, integ) for j in range(3, len(dy))], dy[3:])
                trace.update(nk=nk, delta=m * dt, losses=losses, integrating=integ, aic=(aic_r, aic_i))
                continue
            if m is not None and k % m == 0 and k > n_al:       # regular sampling instant of the recursive phase
                sy.append(y_full[k])
                su.append(u_full[k])
                dy, du = np.diff(sy[-6:]), np.diff(su[-6:])
                rls.update(arx_regressor(dy, du, len(dy) - 1, 1, np.array(su[-5:]), integ), dy[-1])
                since += 1
                if since >= self.update_every:
                    since = 0
                    a1, a2, b0, b1 = to_arx(rls.theta, integ)
                    delta = m * dt
                    w = desired_wn(a1, a2, b0, b1, delta, self.speedup, self.zeta, integrating=integ)
                    if w is None:
                        continue
                    d = design_with_fallback(a1, a2, b0, b1, delta, w[0], self.zeta, problem.tau_d)
                    if d.ok:
                        g = {n: float(np.clip(v, lo, hi)) for n, v, lo, hi in zip(GAIN_NAMES, (d.Kp, d.Ki, d.Kd), lb, ub)}
                        _set_gains(ctrl, g)
                        designs.append((t, g, d.wn, float(np.max(np.abs(d.poles)))))
                        trace["t"].append(t)
                        trace["theta"].append((a1, a2, b0, b1))
                        trace["gains"].append(g)
                        trace["wn"].append(d.wn)
        problem.sim_seconds += n_steps * dt
        self.trace = trace
        if not designs:
            raise NotApplicable("no stable pole placement could be designed from the identified model")
        tail = [dsg for dsg in designs if dsg[0] >= 0.75 * n_steps * dt] or designs[-3:]
        final = {n: float(np.median([g[n] for _, g, _, _ in tail])) for n in GAIN_NAMES}
        a1, a2, b0, b1 = trace["theta"][-1]
        meta = dict(nk=int(trace["nk"]), delta=float(trace["delta"]), integrating=bool(trace["integrating"]),
                    a1=a1, a2=a2, b0=b0, b1=b1,
                    wn=float(np.median([w for _, _, w, _ in tail])), n_designs=len(designs),
                    max_pole=float(np.median([p for _, _, _, p in tail])))
        return final, meta


def _set_gains(ctrl, g):
    """Bumpless gain update for LPID/NPID: keep Ki * integral continuous."""
    if ctrl.Ki > 0 and g["Ki"] > 0:
        ctrl._integral *= ctrl.Ki / g["Ki"]
    ctrl.Kp, ctrl.Ki, ctrl.Kd = g["Kp"], g["Ki"], g["Kd"]
