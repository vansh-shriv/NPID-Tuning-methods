"""
Tuner framework (docs/METHODS_ROADMAP.md, phase M0; protocol in docs/PROTOCOL.md).

    TuningProblem   what a method is allowed to know: a plant it can experiment on (every experiment is
                    counted), the setpoint, sampling time, horizon, actuator limits, optional
                    disturbance/noise settings. Nothing plant-specific lives in a method.
    Tuner           abstract base class; `tune(problem, budget, seed) -> TuningResult`.
    TuningResult    gains + cost bookkeeping (simulations, simulated seconds, wall time, history).

Controller under test: N-PID with scale-free shaping (see docs/PROTOCOL.md section 5).
Methods return base gains {Kp, Ki, Kd} (optionally also {n, m, p}).
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from src.controllers import LPID, NPID
from src.metrics import compute_metrics
from src.simulate import run_simulation

FAIL_COST = 1e6
GAIN_NAMES = ("Kp", "Ki", "Kd")
SHAPE_NAMES = ("n", "m", "p")


# ----------------------------------------------------------------------------- helpers
def tail_settled(y, frac=0.2, tol=0.02, scale=1.0):
    """True if the last `frac` of the signal moves by less than tol*|scale| (steady state reached)."""
    y = np.asarray(y, dtype=float)
    k = max(2, int(len(y) * frac))
    tail = y[-k:]
    return bool(np.all(np.isfinite(y)) and np.ptp(tail) <= tol * abs(scale))


def estimate_time_scale(plant_factory, y_ref, dt, t_end, u_min, u_max):
    """
    Characteristic time scale Ts of a plant from ONE open-loop step (part of the problem specification).

    Probe input = 20% of the actuator range (unit-gain guess |y_ref| for unbounded actuators). Ts is the time
    to reach 63% of the final value if the response settles; otherwise (integrating, oscillating, still
    moving) it falls back to t_end/5.
    """
    lim = max(abs(u_min), abs(u_max))
    u = 0.2 * lim if np.isfinite(lim) else abs(y_ref)
    u = u if abs(u_max) >= abs(u_min) else -u
    plant = plant_factory()
    x = plant.initial_state()
    n = int(t_end / dt)
    y = np.zeros(n)
    for k in range(n):
        y[k] = plant.output(x)
        x = plant.step(k * dt, x, plant.saturate(u), 0.0, dt)
    if not np.all(np.isfinite(y)):
        return t_end / 5.0
    final = y[-1]
    if abs(final) < 1e-9 or not tail_settled(y, 0.2, 0.02, final):
        return t_end / 5.0
    idx = np.where(np.sign(final) * (y - 0.632 * final) >= 0)[0]
    return float(max(idx[0] * dt, 5 * dt)) if len(idx) else t_end / 5.0


def default_bounds(u_min, u_max, y_ref, Ts):
    """
    Generic gain box (Kp, Ki, Kd) in physical units from actuator range, setpoint and time scale (no per-plant
    tuning). Searched in log-space by the optimizers. Deliberately wide (4-6 decades): a first version spanning
    only 0.05-20 x kp0 excluded the optimum on plants without dead time (see docs/PROTOCOL.md, change P1).
        Kp in [0.01, 100] kp0,   Ki in [1e-3, 1e3] kp0/Ts,   Kd in [1e-3, 1e2] kp0 Ts,   kp0 = |u|max/|y_ref|
    """
    lim = max(abs(u_min), abs(u_max))
    if not np.isfinite(lim):
        lim = 10.0 * abs(y_ref)
    kp0 = lim / abs(y_ref)
    lb = (1e-2 * kp0, 1e-3 * kp0 / Ts, 1e-3 * kp0 * Ts)
    ub = (1e2 * kp0, 1e3 * kp0 / Ts, 1e2 * kp0 * Ts)
    return lb, ub


# ----------------------------------------------------------------------------- problem
class TuningProblem:
    """
    Tuning task for one plant. All plant access goes through this object so that experiments are counted.

    Attributes
        evals        number of simulations run through this problem (closed- and open-loop)
        sim_seconds  total simulated plant time (proxy for experiment time on a real plant)
    """

    def __init__(self, name, plant_factory, y_ref, dt, t_end, disturbance=None, noise_sigma=0.0,
                 Ts=None, tau_d_factor=0.05, scaled_shaping=True, shape=None, bounds=None):
        self.name = name
        self.plant_factory = plant_factory
        self.y_ref = float(y_ref)
        self.dt = float(dt)
        self.t_end = float(t_end)
        self.disturbance = disturbance                 # (magnitude, onset time) or None
        self.noise_sigma = float(noise_sigma)

        template = plant_factory()
        self.u_min, self.u_max = float(template.u_min), float(template.u_max)
        self.Ts = float(Ts) if Ts else estimate_time_scale(plant_factory, self.y_ref, self.dt, self.t_end,
                                                           self.u_min, self.u_max)
        self.shape = dict(n=2, m=2, p=2) if shape is None else dict(shape)
        self.tau_d = tau_d_factor * self.Ts
        self.scaled_shaping = scaled_shaping
        self.bounds = bounds if bounds is not None else default_bounds(self.u_min, self.u_max, self.y_ref, self.Ts)
        self.evals = 0
        self.sim_seconds = 0.0

    @classmethod
    def from_scenario(cls, name, **overrides):
        from src.benchmark_plants import SCENARIOS, make_plant
        kw = dict(SCENARIOS[name])
        kw.update(overrides)
        return cls(name, lambda: make_plant(name), **kw)

    # -- bookkeeping
    def reset_counters(self):
        self.evals, self.sim_seconds = 0, 0.0

    def _count(self, seconds):
        self.evals += 1
        self.sim_seconds += seconds

    # -- controller construction
    def controller_kwargs(self, kind="npid"):
        kw = dict(u_min=self.u_min, u_max=self.u_max, tau_d=self.tau_d)
        if kind == "npid":
            if self.scaled_shaping:
                kw.update(e_scale=abs(self.y_ref), de_scale=abs(self.y_ref) / self.Ts)
        return kw

    def make_controller(self, gains, kind="npid"):
        g = {k: float(gains[k]) for k in GAIN_NAMES}
        kw = self.controller_kwargs(kind)
        if kind == "lpid":
            return LPID(g["Kp"], g["Ki"], g["Kd"], self.dt, **kw)
        shape = dict(self.shape)
        shape.update({k: float(gains[k]) for k in SHAPE_NAMES if k in gains})
        return NPID(g["Kp"], g["Ki"], g["Kd"], self.dt, **shape, **kw)

    # -- experiments (all counted)
    def load_func(self, enabled=True):
        if not enabled or self.disturbance is None:
            return None
        mag, onset = self.disturbance
        return lambda t: mag if t >= onset else 0.0

    def simulate(self, gains, kind="npid", scenario="step", noise_seed=0, t_end=None, count=True):
        """Closed-loop simulation. scenario in {"step", "load", "noise"}."""
        plant = self.plant_factory()
        t_end = self.t_end if t_end is None else t_end
        res = run_simulation(
            self.make_controller(gains, kind), plant, self.y_ref, t_end=t_end, dt=self.dt,
            T_load_func=self.load_func(scenario == "load"),
            noise_sigma=self.noise_sigma if scenario == "noise" else 0.0,
            rng=np.random.default_rng(noise_seed))
        if count:
            self._count(t_end)
        return res

    def itae(self, res):
        """ITAE of a simulation result, FAIL_COST if it diverged."""
        if not np.all(np.isfinite(res.omega)) or np.max(np.abs(res.omega)) > 10.0 * abs(self.y_ref):
            return FAIL_COST
        return float(compute_metrics(res.time, self.y_ref, res.omega)["ITAE"])

    def cost(self, gains, kind="npid", scenario="step", noise_seeds=(0,)):
        """Training objective: ITAE (mean over noise seeds in the noise scenario). Counted."""
        if any((not np.isfinite(gains[k])) or gains[k] < 0 for k in GAIN_NAMES):
            return FAIL_COST
        seeds = noise_seeds if scenario == "noise" else (0,)
        return float(np.mean([self.itae(self.simulate(gains, kind, scenario, s)) for s in seeds]))

    def open_loop(self, u, t_end, count=True):
        """
        Open-loop experiment. `u` is a constant or a callable t -> input (saturated to the actuator limits).
        Returns (time, output, input).
        """
        plant = self.plant_factory()
        x = plant.initial_state()
        n = int(t_end / self.dt)
        t, y, uu = np.arange(n) * self.dt, np.zeros(n), np.zeros(n)
        f = u if callable(u) else (lambda _t, _u=float(u): _u)
        for k in range(n):
            y[k] = plant.output(x)
            uu[k] = min(max(f(t[k]), self.u_min), self.u_max)
            x = plant.step(t[k], x, uu[k], 0.0, self.dt)
        if count:
            self._count(t_end)
        return t, y, uu


class OpenLoopRun:
    """
    One open-loop plant experiment (counts as ONE evaluation; simulated seconds accumulate as it runs).

    `hold(u, ...)` applies a constant input either for a fixed duration or until the output has settled
    (peak-to-peak over the last `window` seconds below `tol` * scale), like an operator who waits for steady
    state. Successive `hold` calls continue the same experiment (e.g. settle at the operating point, then
    step). Measurement noise (sigma > 0) corrupts the recorded output only.
    """

    def __init__(self, problem, noise_sigma=0.0, seed=0):
        self.p = problem
        self.plant = problem.plant_factory()
        self.x = self.plant.initial_state()
        self.k = 0
        self.sigma = noise_sigma
        self.rng = np.random.default_rng(seed)
        problem._count(0.0)

    def hold(self, u, duration=None, t_max=None, window=None, tol=0.002, scale=None):
        p = self.p
        u = min(max(float(u), p.u_min), p.u_max)
        win = max(int((window if window else 2.0 * p.Ts) / p.dt), 20)
        scale = abs(p.y_ref) if scale is None else scale
        n_max = int((duration if duration is not None else t_max) / p.dt)
        ts, ys = [], []
        settled = False
        for i in range(n_max):
            y = self.plant.output(self.x)
            ts.append(self.k * p.dt)
            ys.append(y + (self.rng.normal(0, self.sigma) if self.sigma > 0 else 0.0))
            self.x = self.plant.step(self.k * p.dt, self.x, u, 0.0, p.dt)
            self.k += 1
            if duration is None and i >= 2 * win and (i % 10 == 0):
                tail = ys[-win:]
                if not np.all(np.isfinite(tail)) or abs(tail[-1]) > 1e9:
                    break
                if np.ptp(tail) <= tol * scale + 6.0 * self.sigma:
                    settled = True
                    break
        p.sim_seconds += len(ts) * p.dt
        return np.array(ts), np.array(ys), settled


# ----------------------------------------------------------------------------- tuner API
@dataclass
class TuningResult:
    method: str
    gains: dict                       # {"Kp","Ki","Kd"[, "n","m","p"]}
    cost: float = float("nan")        # training ITAE of the returned gains if known (not required)
    evals: int = 0
    sim_seconds: float = 0.0
    wall_time: float = 0.0
    history: list = field(default_factory=list)   # [(evals_used, best_cost)] for iterative methods
    success: bool = True
    message: str = ""
    meta: dict = field(default_factory=dict)       # identified model, selected rule, ...


class Tuner(ABC):
    """Base class of every tuning method."""

    name = "tuner"
    family = "unspecified"

    @abstractmethod
    def _tune(self, problem, budget, seed):
        """Return (gains_dict, meta_dict[, history]). Raise `NotApplicable` if the method cannot handle the plant."""

    def tune(self, problem, budget=None, seed=0):
        """Run the method on `problem`; experiments it performs are counted by the problem."""
        problem.reset_counters()
        t0 = time.perf_counter()
        try:
            out = self._tune(problem, budget, seed)
        except NotApplicable as e:
            return TuningResult(self.name, {}, success=False, message=f"not applicable: {e}",
                                evals=problem.evals, sim_seconds=problem.sim_seconds,
                                wall_time=time.perf_counter() - t0)
        gains, meta = out[0], out[1]
        history = out[2] if len(out) > 2 else []
        wall = time.perf_counter() - t0
        ok = all(np.isfinite(gains.get(k, np.nan)) and gains[k] >= 0 for k in GAIN_NAMES)
        return TuningResult(self.name, {k: float(v) for k, v in gains.items()},
                            cost=float(meta.get("train_cost", float("nan"))), evals=problem.evals, sim_seconds=problem.sim_seconds, wall_time=wall,
                            history=history, success=ok, meta=meta,
                            message="" if ok else "invalid gains (non-finite or negative)")


class NotApplicable(Exception):
    """Raised by a method that cannot be applied to the given plant (documented, not silently worked around)."""
