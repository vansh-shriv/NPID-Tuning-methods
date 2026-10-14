"""
Randomised plant family for the ML phase (M4): many different SISO plants, each fully described by a picklable
`PlantSpec`, so datasets can be generated in parallel workers and models can be tested on plant *structures* they
never saw.

Structures (training types are marked T, the held-out "unseen structure" types O):
    fopdt          T  K e^{-theta s} / (tau s + 1)
    sopdt          T  K e^{-theta s} / ((tau1 s + 1)(tau2 s + 1))            overdamped
    ipdt           T  Kv e^{-theta s} / (s (tau s + 1))                      integrating
    hammerstein    T  FOPDT whose input passes through sign(u)|u|^p          nonlinear (gain depends on the level)
    underdamped    O  K wn^2 e^{-theta s} / (s^2 + 2 zeta wn s + wn^2)       zeta in [0.25, 0.7]
    third          O  K e^{-theta s} / ((tau1 s+1)(tau2 s+1)(tau3 s+1))      third order, distinct lags

Randomised: gain, time scale (log-uniform over two decades), dead-time ratio, setpoint, actuator authority (how
much more input than the steady state needs), polarity of the actuator, input disturbance, measurement noise.
Every spec is positive-gain, stable or integrating, and has a reachable setpoint by construction.

`LTIPlant` integrates the linear dynamics *exactly* (zero-order-hold discretisation, cached per step size), which
makes a closed-loop simulation about 10x faster than the generic RK4 plants (needed: ~10^6 simulations).
"""

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.signal import cont2discrete, tf2ss

from src.plant import DelayedPlant, Plant

TRAIN_TYPES = ("fopdt", "sopdt", "ipdt", "hammerstein")
OOD_TYPES = ("underdamped", "third")
ALL_TYPES = TRAIN_TYPES + OOD_TYPES


class LTIPlant(Plant):
    """Linear plant given by num/den (descending powers of s), exact ZOH integration, optional input nonlinearity."""

    name = "lti"

    def __init__(self, num, den, u_limits=(-np.inf, np.inf), input_power=None, name=None):
        A, B, C, D = tf2ss(np.atleast_1d(num), np.atleast_1d(den))
        self.A, self.B, self.C, self.D = A, B, np.asarray(C).reshape(-1), float(np.squeeze(D))
        self.u_min, self.u_max = u_limits
        self.input_power = input_power
        self._cache = {}
        if name:
            self.name = name

    @property
    def state_dim(self):
        return self.A.shape[0]

    def _discrete(self, dt):
        if dt not in self._cache:
            Ad, Bd, Cd, Dd, _ = cont2discrete((self.A, self.B, self.C.reshape(1, -1), np.array([[self.D]])), dt, method="zoh")
            self._cache[dt] = (Ad, Bd.reshape(-1))
        return self._cache[dt]

    def _effective_input(self, u):
        if self.input_power is None:
            return u
        return np.sign(u) * abs(u) ** self.input_power

    def derivatives(self, t, x, u, d=0.0):
        return self.A @ np.asarray(x) + self.B.reshape(-1) * (self._effective_input(u) + d)

    def step(self, t, x, u, d, dt):
        Ad, Bd = self._discrete(dt)
        return (Ad @ np.asarray(x, dtype=float) + Bd * (self._effective_input(u) + d)).tolist()

    def output(self, x):
        return float(self.C @ np.asarray(x, dtype=float))


@dataclass
class PlantSpec:
    ptype: str
    K: float                       # DC gain (output/input); velocity gain Kv for ipdt
    tau1: float                    # dominant lag (s); for 'under' unused
    tau2: float = 0.0
    tau3: float = 0.0
    zeta: float = 1.0
    wn: float = 1.0
    theta: float = 0.0
    p: float = 1.0                 # input power (hammerstein)
    y_ref: float = 1.0
    r_auth: float = 4.0            # actuator limit / input needed in steady state
    unipolar: bool = False
    dist_frac: float = 0.1
    noise_rel: float = 0.01
    ident: int = 0
    derived: dict = field(default_factory=dict)

    # ---- transfer function ----------------------------------------------------------------------
    def num_den(self):
        t = self.ptype
        if t in ("fopdt", "hammerstein"):
            return [self.K], [self.tau1, 1.0]
        if t == "sopdt":
            return [self.K], list(np.polymul([self.tau1, 1.0], [self.tau2, 1.0]))
        if t == "third":
            return [self.K], list(np.polymul(np.polymul([self.tau1, 1.0], [self.tau2, 1.0]), [self.tau3, 1.0]))
        if t == "underdamped":
            return [self.K * self.wn ** 2], [1.0, 2 * self.zeta * self.wn, self.wn ** 2]
        if t == "ipdt":
            return [self.K], [self.tau1, 1.0, 0.0]
        raise ValueError(t)

    @property
    def integrating(self):
        return self.ptype == "ipdt"

    def u_ss(self):
        """Steady input holding the setpoint (0 for the integrating plant)."""
        if self.integrating:
            return 0.0
        base = self.y_ref / self.K
        return base ** (1.0 / self.p) if self.ptype == "hammerstein" else base

    def time_scale(self):
        """Dominant time scale used for horizons and sampling: tau + theta (or the oscillation time for 'under')."""
        if self.ptype == "underdamped":
            return 4.0 / (self.zeta * self.wn) + self.theta
        slow = {"third": self.tau1 + self.tau2 + self.tau3}.get(self.ptype, self.tau1 + self.tau2)
        return slow + self.theta

    def u_max(self):
        if self.integrating:
            t_reach = 6.0 * (self.tau1 + self.theta) + 1e-9
            return self.r_auth * abs(self.y_ref) / (self.K * t_reach)
        return self.r_auth * abs(self.u_ss())

    def horizon(self):
        if self.integrating:
            return 12.0 * (self.tau1 + self.theta) + 2.0 * abs(self.y_ref) / (self.K * self.u_max())
        return 10.0 * self.time_scale()

    def fastest_time(self):
        if self.ptype == "underdamped":
            return 1.0 / self.wn
        lags = [x for x in (self.tau1, self.tau2, self.tau3) if x > 0]
        return min(lags)

    def sample_time(self):
        return min(self.horizon() / 1500.0, self.fastest_time() / 10.0)


def build_plant(spec: PlantSpec):
    """Fresh plant instance (DelayedPlant wraps the LTI core when the spec has a dead time)."""
    num, den = spec.num_den()
    umax = spec.u_max()
    limits = (0.0 if spec.unipolar else -umax, umax)
    core = LTIPlant(num, den, limits, input_power=spec.p if spec.ptype == "hammerstein" else None, name=spec.ptype)
    return DelayedPlant(core, spec.theta) if spec.theta > 0 else core


def problem_from_spec(spec: PlantSpec, **overrides):
    """TuningProblem for a spec (scenario settings derived from the spec: horizon, sampling, disturbance, noise)."""
    from src.tuners.base import TuningProblem
    umax_ss = abs(spec.u_ss()) if not spec.integrating else 0.1 * spec.u_max()
    kw = dict(y_ref=spec.y_ref, dt=spec.sample_time(), t_end=spec.horizon(),
              disturbance=(spec.dist_frac * max(umax_ss, 1e-12), 0.5 * spec.horizon()),
              noise_sigma=spec.noise_rel * abs(spec.y_ref))
    kw.update(overrides)
    return TuningProblem(f"{spec.ptype}-{spec.ident}", lambda: build_plant(spec), **kw)


# ----------------------------------------------------------------------------- sampling
def _loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_spec(rng, ptype, ident=0) -> PlantSpec:
    T0 = _loguniform(rng, 0.2, 20.0)
    theta_ratio = 0.0 if rng.random() < 0.1 else _loguniform(rng, 0.03, 1.5)
    s = PlantSpec(ptype=ptype, K=_loguniform(rng, 0.3, 5.0), tau1=T0, y_ref=_loguniform(rng, 0.5, 30.0),
                  r_auth=_loguniform(rng, 1.8, 10.0), unipolar=bool(rng.random() < 0.3),
                  dist_frac=float(rng.uniform(0.05, 0.2)), noise_rel=float(rng.uniform(0.005, 0.02)), ident=ident)
    if ptype in ("fopdt", "hammerstein", "ipdt"):
        s.theta = theta_ratio * T0
    if ptype == "hammerstein":
        s.p = float(rng.uniform(0.5, 0.8))
    if ptype == "sopdt":
        s.tau2 = T0 * float(rng.uniform(0.1, 1.0))
        s.theta = theta_ratio * (s.tau1 + s.tau2)
    if ptype == "third":
        s.tau2 = T0 * float(rng.uniform(0.2, 1.0))
        s.tau3 = T0 * float(rng.uniform(0.05, 0.6))
        s.theta = theta_ratio * (s.tau1 + s.tau2 + s.tau3) * 0.5
    if ptype == "underdamped":
        s.wn = float(rng.uniform(0.8, 3.0)) / T0
        s.zeta = float(rng.uniform(0.25, 0.7))
        s.theta = theta_ratio * 4.0 / (s.zeta * s.wn) * 0.3
    if ptype == "ipdt":
        s.tau1 = T0 * float(rng.uniform(0.1, 1.0))
        s.theta = theta_ratio * s.tau1
        s.unipolar = False
    return s


def sample_specs(n, seed=0, types=TRAIN_TYPES, offset=0):
    """n specs with types cycled so that every type appears equally often; deterministic per (seed, index)."""
    out = []
    for i in range(n):
        rng = np.random.default_rng([seed, offset + i])
        out.append(sample_spec(rng, types[i % len(types)], ident=offset + i))
    return out


def spec_to_dict(spec):
    d = asdict(spec)
    d.pop("derived", None)
    return d


def spec_from_dict(d):
    return PlantSpec(**{k: v for k, v in d.items() if k != "derived"})
