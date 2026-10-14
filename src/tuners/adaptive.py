"""
Adaptive ML-based tuning (docs/METHODS_ROADMAP.md, phase M6, method 18).

Online loop: a recursive plant estimator (RLS on the ARX model of M2) feeds the ML predictor of M4 (bump-test features ->
dimensionless log-gains), which retunes the N-PID on a schedule with safety limits (gain box, per-update rate limit,
bumpless transfer, estimator updated only while the loop is excited). Compared in `experiments/m6_adaptive.py`
with frozen gains, classical self-tuning pole placement (the same estimator, a model-based design) and a retune by
plant experiment that knows when the plant changed.

The controller runs at the simulation rate (like every tuner of M3-M5: gains tuned in that environment stay valid); the
estimator samples at Delta = max(theta, T_n/10) with the *mean* input of each interval (the zero-order-hold equivalent;
M2 found that feeding a fast controller's instantaneous input to a slower estimator biases the model).

    SwitchingPlant / change_scenario   a plant whose parameters change at a given time (gain, time scale), with
                                       output continuity at the switch
    OnlineEstimator                    gated RLS of A dy = B du (or the integrating form) + ARX -> FOPDT/IPDT model
    FrozenPolicy, ScheduledPolicy, AdaptivePolicy('ml' | 'pp')
    run_scenario                       the closed loop with reference steps; returns signals, gains and window metrics
"""

from dataclasses import dataclass, field

import numpy as np

from src.controllers import NPID
from src.plant import DelayedPlant, Plant
from src.plant_family import LTIPlant, problem_from_spec
from src.tuners.base import GAIN_NAMES
from src.tuners.base import Tuner
from src.tuners.features import denormalise_gains, extract_features, features_from_model
from src.tuners.identify import ProcessModel, fit_fopdt, fit_ipdt
from src.tuners.selftuning import (RLS, arx_regressor, arx_step_response, design_with_fallback, desired_wn, to_arx)

REF_LEVELS = (1.0, 0.8, 1.0, 0.8, 1.0)       # setpoint (fraction of y_ref) in window 0..4
N_WINDOWS = len(REF_LEVELS)


# ----------------------------------------------------------------------------- plant with a regime change
class SwitchingPlant(Plant):
    """
    `before` until `t_switch`, `after` from then on (same state dimension, e.g. two LTI plants of the same structure).
    At the switch the state is moved by the minimum-norm correction that keeps the *output* continuous (a gain or
    time-scale change must not teleport the output). `initial_state()` resets the switch.
    """
    name = "switching"

    def __init__(self, before, after, t_switch):
        self.before, self.after, self.t_switch = before, after, float(t_switch)
        self._switched = False

    @property
    def state_dim(self):
        return self.before.state_dim

    @property
    def u_min(self):
        return self.before.u_min

    @property
    def u_max(self):
        return self.before.u_max

    def initial_state(self):
        self._switched = False
        return self.before.initial_state()

    def _active(self):
        return self.after if self._switched else self.before

    def derivatives(self, t, x, u, d=0.0):
        return self._active().derivatives(t, x, u, d)

    def output(self, x):
        return self._active().output(x)

    def step(self, t, x, u, d, dt):
        if not self._switched and t >= self.t_switch:
            y = self.before.output(x)
            Cb = np.asarray(self.after.C, dtype=float)
            x = (np.asarray(x, dtype=float) + Cb * (y - Cb @ np.asarray(x, dtype=float)) / (Cb @ Cb)).tolist()
            self._switched = True
        return self._active().step(t, x, u, d, dt)


def scaled_spec(spec, change):
    """Spec after a change ('gain', f): K*f (Kv*f for an integrator); ('lag', f): every lag * f. ('none', 1): unchanged."""
    import copy
    s = copy.deepcopy(spec)
    kind, f = change
    if kind == "gain":
        s.K *= f
    elif kind == "lag":
        s.tau1, s.tau2, s.tau3 = s.tau1 * f, s.tau2 * f, s.tau3 * f
    elif kind != "none":
        raise ValueError(kind)
    return s


def _core(spec, limits):
    num, den = spec.num_den()
    return LTIPlant(num, den, limits, input_power=spec.p if spec.ptype == "hammerstein" else None, name=spec.ptype)


def change_scenario(spec, change, t_switch):
    """(plant factory, spec after the change). The actuator limits are those of the original plant."""
    after = scaled_spec(spec, change)
    limits = (0.0 if spec.unipolar else -spec.u_max(), spec.u_max())

    def factory():
        sw = SwitchingPlant(_core(spec, limits), _core(after, limits), t_switch)
        return DelayedPlant(sw, spec.theta) if spec.theta > 0 else sw
    return factory, after


def change_feasible(spec, change, margin=0.7):
    """The setpoint must stay reachable after the change (steady input below `margin` of the actuator limit)."""
    after = scaled_spec(spec, change)
    return spec.integrating or abs(after.u_ss() * max(REF_LEVELS)) <= margin * spec.u_max()


# ----------------------------------------------------------------------------- online estimator
class OnlineEstimator:
    """
    RLS estimate of the sampled ARX model (A y = B u on the signal levels for self-regulating plants: the differenced
    form A dy = B du amplifies measurement noise relative to the signal and biased K by up to 4x; A' dy = B u + c for integrators) from the closed loop,
    updated only while `excited`, with a covariance reset when a new excitation (reference step) starts so that a
    changed plant can be tracked. `model(problem)` converts the estimate into a FOPDT/IPDT `ProcessModel` through
    its step response (the same representation the ML features are defined on).
    """

    def __init__(self, delta, integrating, lam=0.99, p_reset=100.0, n_resp=400, levels=True):
        self.delta, self.integ, self.n_resp, self.levels = delta, integrating, n_resp, levels
        self.rls = RLS(4, lam, p0=p_reset)
        self.p_reset = p_reset
        self.ys, self.us = [], []
        self.n_used = 0

    def new_excitation(self):
        self.rls.P = self.p_reset * np.eye(4)
        self.n_used = 0

    def push(self, y):
        self.ys.append(float(y))

    def push_input(self, u):
        self.us.append(float(u))

    def update(self):
        """
        Use the newest output sample. At that moment the outputs y_0..y_J and the inputs u_0..u_{J-1} are known; the
        regressor of sample J needs dy(J-1..J), du(J-1..J-2) (and the input levels for integrators).
        """
        if not self.integ and self.levels:
            if len(self.ys) < 3 or len(self.us) < 2:
                return False
            ys, us = self.ys, self.us
            self.rls.update(np.array([-ys[-2], -ys[-3], us[-1], us[-2]]), ys[-1])
            self.n_used += 1
            return True
        if len(self.ys) < 5 or len(self.us) < 4:
            return False
        dy = np.diff(self.ys[-5:])
        du = np.append(np.diff(self.us[-4:]), 0.0)            # du[3] (the not yet applied input) is never read
        u_level = np.array(self.us[-3:] + [0.0])
        self.rls.update(arx_regressor(dy, du, 3, 1, u_level, self.integ), dy[3])
        self.n_used += 1
        return True

    def arx(self):
        return to_arx(self.rls.theta, self.integ)

    def process_model(self):
        """FOPDT/IPDT model from the current ARX estimate, or None if it is not (yet) a usable model."""
        a1, a2, b0, b1 = self.arx()
        if not np.all(np.isfinite([a1, a2, b0, b1])):
            return None
        roots = np.roots([1.0, a1, a2])
        if self.integ:
            if np.sum(np.abs(roots - 1.0) < 0.03) < 1 or np.any(np.abs(roots) > 1.0 + 0.03):
                return None
        elif np.any(np.abs(roots) >= 0.999):
            return None
        y = arx_step_response(a1, a2, b0, b1, n=self.n_resp, nk=1)
        t = np.arange(self.n_resp) * self.delta
        if not np.all(np.isfinite(y)):
            return None
        try:
            if self.integ:
                if y[-1] <= y[self.n_resp // 2] or y[-1] <= 0:
                    return None
                K, tau, theta, nrmse = fit_ipdt(t, y)
                kind = "ipdt"
            else:
                if y[-1] <= 1e-12:
                    return None
                K, tau, theta, nrmse = fit_fopdt(t, y)
                kind = "fopdt"
        except Exception:
            return None
        if K <= 0 or nrmse > 0.3:
            return None
        return ProcessModel(kind, K, tau, theta, nrmse, "arx", data=dict(t=t, s=y))


# ----------------------------------------------------------------------------- policies
def rate_limit(current, target, max_decades):
    """Move each gain towards the target by at most `max_decades` (in log10)."""
    out = {}
    for k in GAIN_NAMES:
        c, t = max(current[k], 1e-12), max(target[k], 1e-12)
        out[k] = float(10 ** (np.log10(c) + np.clip(np.log10(t) - np.log10(c), -max_decades, max_decades)))
    return out


def est_us(est):
    return len(est.us) > 0


class ClippedModel:
    """Regressor whose inputs are clipped to the range seen in training (no extrapolation from an online-estimated feature)."""

    def __init__(self, model, X_train, pad=0.0):
        self.model = model
        self.lo, self.hi = np.min(X_train, axis=0) - pad, np.max(X_train, axis=0) + pad

    def predict(self, X):
        return self.model.predict(np.clip(np.asarray(X, dtype=float), self.lo, self.hi))


class FrozenPolicy:
    name = "frozen"

    def begin(self, ctx):
        pass

    def observe(self, j, t, ym, since_step, new_step):
        return None

    def record_input(self, u):
        pass


class ScheduledPolicy(FrozenPolicy):
    """Gains switched at given times (the experiment-based retune that knows when the plant changed)."""
    name = "retune by experiment"

    def __init__(self, schedule):
        self.schedule = sorted(schedule, key=lambda s: s[0])        # [(t, gains)]
        self.i = 0

    def begin(self, ctx):
        self.i = 0

    def observe(self, j, t, ym, since_step, new_step):
        if self.i < len(self.schedule) and t >= self.schedule[self.i][0]:
            g = self.schedule[self.i][1]
            self.i += 1
            return dict(g)
        return None


@dataclass
class AdaptiveConfig:
    mode: str = "ml"                 # 'ml' (estimator -> ML predictor) or 'pp' (estimator -> pole placement)
    lam: float = 0.99
    p_reset: float = 100.0
    levels: bool = True
    delta_div: float = 10.0         # sampling interval = max(theta, T_n / delta_div)
    window_Tn: float = 4.0           # excitation window after a reference step (in T_n of the initial model)
    update_every: int = 5            # redesign every n samples inside the window
    min_samples: int = 8             # samples of a window before the first redesign
    detune: float = 0.5              # ML mode: predicted gains are multiplied by this robustness factor (< 1 = more gain margin)
    max_decades: float = 0.15        # rate limit per redesign (log10 gain)
    skip_saturated: bool = True      # no estimator update / redesign while the actuator is saturated (the data are nonlinear)
    speedup: float = 3.0             # pole-placement closed-loop speed (as in M2)
    zeta: float = 0.8


class AdaptivePolicy(FrozenPolicy):
    """Retunes the gains from the online estimate (see module docstring)."""

    def __init__(self, cfg, model=None, name=None):
        self.cfg, self.model = cfg, model
        self.name = name or {"ml": "adaptive ML", "pp": "self-tuning pole placement"}[cfg.mode]
        self.trace = []

    def begin(self, ctx):
        self.ctx = ctx
        self.est = OnlineEstimator(ctx["delta"], ctx["integrating"], self.cfg.lam, self.cfg.p_reset, levels=self.cfg.levels)
        self.gains = dict(ctx["gains0"])
        self.since_update, self.trace = 0, []
        self.window = max(self.cfg.window_Tn * ctx["T_n0"], 25.0 * ctx["delta"])   # at least 25 samples

    def record_input(self, u):
        self.est.push_input(u)

    def observe(self, j, t, ym, since_step, new_step):
        est = self.est
        est.push(ym)
        if new_step:
            est.new_excitation()
            self.since_update = 0
        if since_step > self.window:
            return None
        if self.cfg.skip_saturated and self._saturated():
            return None
        est.update()
        self.since_update += 1
        if est.n_used < self.cfg.min_samples or self.since_update < self.cfg.update_every:
            return None
        self.since_update = 0
        target = self._design()
        if target is None:
            return None
        lb, ub = self.ctx["bounds"]
        target = {k: float(np.clip(target[k], lo, hi)) for k, lo, hi in zip(GAIN_NAMES, lb, ub)}
        limited = rate_limit(self.gains, target, self.cfg.max_decades)
        self.gains = {k: float(np.clip(limited[k], lo, hi)) for k, lo, hi in zip(GAIN_NAMES, lb, ub)}      # a start below the box (Kd = 0) enters it
        self.trace.append((t, dict(self.gains), dict(target)))
        return dict(self.gains)

    def _saturated(self):
        lo, hi = self.ctx["u_limits"]
        if not est_us(self.est):
            return False
        u = self.est.us[-1]
        span = hi - lo if np.isfinite(hi - lo) else np.inf
        return bool(np.isfinite(span) and (u >= hi - 0.02 * span or u <= lo + 0.02 * span))

    def _design(self):
        m = self.est.process_model()
        if m is None:
            return None
        if self.cfg.mode == "ml":
            f = features_from_model(self.ctx["problem"], m)
            z = np.asarray(self.model.predict(f.vec[None, :]))[0]
            g = denormalise_gains(z, f.K_eff, f.T_n)
            return {k: self.cfg.detune * v for k, v in g.items()}
        a1, a2, b0, b1 = self.est.arx()
        w = desired_wn(a1, a2, b0, b1, self.est.delta, self.cfg.speedup, self.cfg.zeta, integrating=self.est.integ)
        if w is None:
            return None
        d = design_with_fallback(a1, a2, b0, b1, self.est.delta, w[0], self.cfg.zeta, self.ctx["problem"].tau_d)
        return d.gains if d.ok else None


# ----------------------------------------------------------------------------- closed loop with reference steps
@dataclass
class ScenarioResult:
    t: np.ndarray
    y: np.ndarray
    u: np.ndarray
    r: np.ndarray
    windows: list = field(default_factory=list)       # per window: dict(itae, overshoot_pct, iae)
    gain_trace: list = field(default_factory=list)
    failed: bool = False


def _set_gains(ctrl, g):
    if ctrl.Ki > 0 and g["Ki"] > 0:
        ctrl._integral *= ctrl.Ki / g["Ki"]               # bumpless: keep Ki * integral continuous
    ctrl.Kp, ctrl.Ki, ctrl.Kd = g["Kp"], g["Ki"], g["Kd"]


def sampling_steps(T_n0, theta0, dt, div=20.0):
    """Controller/estimator sampling interval in simulation steps: Delta = max(theta, T_n/div), at least one step."""
    return max(1, int(round(max(theta0, T_n0 / div) / dt)))


def window_metrics(t, y, r, H, w):
    """ITAE (time from the window's reference step), IAE and overshoot of window w, from the TRUE output."""
    dt = t[1] - t[0]
    sel = (t >= w * H) & (t < (w + 1) * H)
    tt, e = t[sel] - w * H, r[sel] - y[sel]
    target = r[sel][-1]
    y_ref = target / REF_LEVELS[w]
    step = (REF_LEVELS[w] - (REF_LEVELS[w - 1] if w > 0 else 0.0)) * y_ref
    over = max(0.0, float(np.max(np.sign(step) * (y[sel] - target))))
    return dict(itae=float(np.sum(np.abs(e) * tt) * dt), iae=float(np.sum(np.abs(e)) * dt),
                overshoot_pct=100.0 * over / max(abs(step), 1e-12))


def run_scenario(spec, change, policy, gains0, delta_steps, noise=True, seed=0, t_switch_frac=1.95, kind0="fopdt",
                 T_n0=None, problem=None):
    """
    Closed loop of the N-PID on a family plant `spec` with a `change` at `t_switch_frac` * H, reference steps
    (REF_LEVELS * y_ref) every H = spec.horizon(). `policy` may retune the gains online (see `run_loop`).
    """
    problem = problem or problem_from_spec(spec)
    H = spec.horizon()
    factory, _ = change_scenario(spec, change, t_switch_frac * H)
    return run_loop(problem, factory, H, policy, gains0, delta_steps, noise, seed, kind0, T_n0)


def run_loop(problem, plant_factory, H, policy, gains0, delta_steps, noise=True, seed=0, kind0="fopdt", T_n0=None):
    """Plant-generic loop behind `run_scenario`: N_WINDOWS reference windows of length H, any plant factory."""
    dt = problem.dt
    n_steps = int(N_WINDOWS * H / dt)
    plant = plant_factory()
    x = plant.initial_state()
    kw = problem.controller_kwargs("npid")
    ctrl = NPID(gains0["Kp"], gains0["Ki"], gains0["Kd"], dt, **problem.shape, **kw)      # controller at the simulation rate
    ctrl.reset()
    ctx = dict(u_limits=(problem.u_min, problem.u_max), delta=delta_steps * dt, integrating=(kind0 == "ipdt"), gains0=gains0, T_n0=T_n0 or problem.Ts, bounds=problem.bounds,
               problem=problem)
    policy.begin(ctx)
    rng = np.random.default_rng(seed)
    sigma = problem.noise_sigma if noise else 0.0
    T, Y, U, R = (np.zeros(n_steps) for _ in range(4))
    trace = [(0.0, dict(gains0))]
    u_hold, last_step_t, j, u_sum = 0.0, 0.0, 0, 0.0
    for k in range(n_steps):
        t = k * dt
        w = min(int(t / H), N_WINDOWS - 1)
        r = REF_LEVELS[w] * problem.y_ref
        y = plant.output(x)
        ym = y + (rng.normal(0, sigma) if sigma > 0 else 0.0)
        if k % delta_steps == 0:                                      # estimator sampling instant
            if k > 0:
                policy.record_input(u_sum / delta_steps)             # mean input of the interval just finished (ZOH equivalent)
            u_sum = 0.0
            new_step = (k == 0) or (k * dt >= w * H and (k - delta_steps) * dt < w * H)
            if new_step:
                last_step_t = w * H
            g = policy.observe(j, t, ym, t - last_step_t, new_step)
            if g is not None:
                _set_gains(ctrl, g)
                trace.append((t, dict(g)))
            j += 1
        u_hold = min(max(ctrl.compute(r - ym), problem.u_min), problem.u_max)
        u_sum += u_hold
        T[k], Y[k], U[k], R[k] = t, y, u_hold, r
        x = plant.step(t, x, u_hold, 0.0, dt)
    failed = (not np.all(np.isfinite(Y))) or np.max(np.abs(Y)) > 10.0 * abs(problem.y_ref)
    res = ScenarioResult(T, Y, U, R, gain_trace=trace, failed=bool(failed))
    if not failed:
        res.windows = [window_metrics(T, Y, R, H, w) for w in range(N_WINDOWS)]
    return res


# ----------------------------------------------------------------------------- generic plant change (benchmark plants)
class ChangePlant(Plant):
    """
    Any plant whose actuator gain (`gain`) and/or time rate (`rate`: dynamics run at rate x real time) change at
    `t_switch`. Both keep the state continuous (no jump), unlike a parameter swap, so they work for every `Plant`.
    The actuator limits are those of the wrapped plant (applied by the loop before the gain change).
    """
    name = "change"

    def __init__(self, plant, t_switch, gain=1.0, rate=1.0):
        self.plant, self.t_switch, self.gain, self.rate = plant, float(t_switch), float(gain), float(rate)

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
        return self.plant.initial_state()

    def derivatives(self, t, x, u, d=0.0):
        return self.plant.derivatives(t, x, u, d)

    def output(self, x):
        return self.plant.output(x)

    def step(self, t, x, u, d, dt):
        if t >= self.t_switch:
            return self.plant.step(t, x, u * self.gain, d, dt * self.rate)
        return self.plant.step(t, x, u, d, dt)


class AdaptiveMLTuner(Tuner):
    """
    Method 18 used as an auto-tuner on a nominal plant (so that it can be benchmarked like the others): the
    SVR gains of a bump test start the online loop, the loop runs `n_windows` reference windows without any plant
    change, and the gains it holds at the end are returned as static gains. The *online* behaviour (plant changes)
    is evaluated with `AdaptivePolicy` in `run_loop`. Cost: bump test + n_windows closed-loop horizons.
    """
    name = "Adaptive ML (as auto-tuner)"
    family = "adaptive"
    stochastic = True

    def __init__(self, model_init, model_est, cfg=None, n_windows=3, H=None):
        self.model_init, self.model_est, self.cfg = model_init, model_est, cfg or AdaptiveConfig()
        self.n_windows, self.H = n_windows, H

    def _tune(self, problem, budget, seed):
        feat = extract_features(problem, noise_sigma=problem.noise_sigma, seed=seed)
        z = np.asarray(self.model_init.predict(feat.vec[None, :]))[0]
        g0 = denormalise_gains(z, feat.K_eff, feat.T_n)
        lb, ub = problem.bounds
        g0 = {k: float(np.clip(g0[k], lo, hi)) for k, lo, hi in zip(GAIN_NAMES, lb, ub)}
        m = sampling_steps(feat.T_n, feat.model["theta"], problem.dt, self.cfg.delta_div)
        pol = AdaptivePolicy(self.cfg, self.model_est)
        H = self.H or problem.t_end
        res = run_loop(problem, problem.plant_factory, H, pol, g0, m, True, seed, feat.kind, feat.T_n)
        problem._count(0.0)
        problem.sim_seconds += N_WINDOWS * H
        final = res.gain_trace[-1][1]
        return dict(final), dict(updates=len(res.gain_trace) - 1, loop_failed=bool(res.failed)), []
