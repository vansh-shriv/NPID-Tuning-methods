"""
Optimization-based tuners (docs/METHODS_ROADMAP.md, phase M3): PSO, GA, ABC, SA (+ random search baseline).

All four are implemented from scratch (numpy only) as pure functions `algo(f, d, budget, rng) -> (u_best, f_best)` on
the unit cube [0, 1]^d with an objective `f(u)`, so that they can be validated on analytic test functions
independently of any plant. `Objective` wraps a `TuningProblem`: it maps the cube to log10(Kp, Ki, Kd) in the
problem's gain box (and optionally to the shape exponents n, m, p), enforces the budget in *simulations* (so that
noise-averaged objectives are charged for every simulation), and records the best-so-far history.

Budget semantics: an algorithm never exceeds `budget` simulations; when the budget runs out the wrapper raises
`BudgetExhausted` and the best point found so far is returned.

Algorithms and default hyper-parameters (the literature defaults; sensitivity studied in experiments/m3_optimizers.py)
    PSO  Kennedy & Eberhart (1995) with linearly decreasing inertia (Shi & Eberhart 1998): n=20, w 0.9 -> 0.4,
         c1 = c2 = 2.0, velocity clamp 20% of the range, global-best topology.
    GA   real-coded: tournament selection (k=3), BLX-alpha crossover (alpha 0.5, pc 0.9), Gaussian mutation
         (pm 0.2, sigma decaying 0.15 -> 0.01), elitism 2, n=20.
    ABC  Karaboga (2005): n_sources=10 employed + 10 onlooker bees, limit = n_sources*d/2 trials, greedy selection,
         scout replaces exhausted sources.
    SA   Kirkpatrick et al. (1983): single chain, Gaussian neighbourhood shrinking with temperature, geometric
         cooling, Metropolis acceptance on log(1 + cost) differences (scale-free), T0 calibrated so that ~50% of
         uphill moves are accepted at the start.
"""

import numpy as np

from src.tuners.base import FAIL_COST, GAIN_NAMES, SHAPE_NAMES, Tuner

SHAPE_RANGE = (0.5, 4.0)          # search range of the shape exponents n, m, p when tuned


class BudgetExhausted(Exception):
    pass


# ----------------------------------------------------------------------------- algorithms
def random_search(f, d, budget, rng):
    best_u, best_f = None, np.inf
    for _ in range(budget):
        u = rng.random(d)
        fu = f(u)
        if fu < best_f:
            best_u, best_f = u, fu
    return best_u, best_f


def pso(f, d, budget, rng, n=20, w0=0.9, w1=0.4, c1=2.0, c2=2.0, vmax=0.2):
    n = min(n, budget)
    x = rng.random((n, d))
    v = rng.uniform(-vmax, vmax, (n, d)) * 0.5
    fx = np.array([f(xi) for xi in x])
    pbest, pf = x.copy(), fx.copy()
    g = int(np.argmin(pf))
    gbest, gf = pbest[g].copy(), pf[g]
    n_iter = max(1, budget // n)
    for it in range(1, n_iter):
        w = w0 - (w0 - w1) * it / max(1, n_iter - 1)
        r1, r2 = rng.random((n, d)), rng.random((n, d))
        v = np.clip(w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x), -vmax, vmax)
        x = x + v
        out = (x < 0) | (x > 1)
        x = np.clip(x, 0.0, 1.0)
        v[out] = 0.0
        for i in range(n):
            fi = f(x[i])
            if fi < pf[i]:
                pbest[i], pf[i] = x[i].copy(), fi
                if fi < gf:
                    gbest, gf = x[i].copy(), fi
    return gbest, gf


def ga(f, d, budget, rng, n=20, k=3, pc=0.9, alpha=0.5, pm=0.2, s0=0.15, s1=0.01, elite=2):
    n = min(n, budget)
    pop = rng.random((n, d))
    fit = np.array([f(p) for p in pop])
    n_gen = max(1, budget // n)
    for gen in range(1, n_gen):
        order = np.argsort(fit)
        new = [pop[i].copy() for i in order[:elite]]
        new_f = [fit[i] for i in order[:elite]]
        sigma = s0 - (s0 - s1) * gen / max(1, n_gen - 1)
        children = []
        while len(children) < n - elite:
            def pick():
                idx = rng.integers(0, n, k)
                return pop[idx[np.argmin(fit[idx])]]
            p1, p2 = pick(), pick()
            if rng.random() < pc:                                    # BLX-alpha
                lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
                span = hi - lo
                c1_ = rng.uniform(lo - alpha * span, hi + alpha * span)
                c2_ = rng.uniform(lo - alpha * span, hi + alpha * span)
            else:
                c1_, c2_ = p1.copy(), p2.copy()
            for c in (c1_, c2_):
                mask = rng.random(d) < pm
                c = np.where(mask, c + rng.normal(0, sigma, d), c)
                children.append(np.clip(c, 0.0, 1.0))
        for c in children[:n - elite]:
            new.append(c)
            new_f.append(f(c))
        pop, fit = np.array(new), np.array(new_f)
    i = int(np.argmin(fit))
    return pop[i], fit[i]


def abc(f, d, budget, rng, n_sources=10, limit=None):
    n_sources = min(n_sources, budget)
    limit = limit if limit is not None else max(3, int(0.5 * n_sources * d))
    x = rng.random((n_sources, d))
    fx = np.array([f(xi) for xi in x])
    trials = np.zeros(n_sources, int)

    def neighbour(i):
        j = int(rng.integers(0, d))
        k = int(rng.choice([m for m in range(n_sources) if m != i])) if n_sources > 1 else i
        v = x[i].copy()
        v[j] = np.clip(x[i, j] + rng.uniform(-1, 1) * (x[i, j] - x[k, j]), 0.0, 1.0)
        return v

    def greedy(i, v):
        fv = f(v)
        if fv < fx[i]:
            x[i], fx[i], trials[i] = v, fv, 0
        else:
            trials[i] += 1

    while True:
        for i in range(n_sources):                                   # employed bees
            greedy(i, neighbour(i))
        fitness = 1.0 / (1.0 + fx)                                  # onlooker bees: roulette on fitness
        prob = fitness / fitness.sum()
        for _ in range(n_sources):
            i = int(rng.choice(n_sources, p=prob))
            greedy(i, neighbour(i))
        worn = int(np.argmax(trials))                                # scout: replace the most exhausted source
        if trials[worn] > limit:
            x[worn] = rng.random(d)
            fx[worn] = f(x[worn])
            trials[worn] = 0


def sa(f, d, budget, rng, sigma0=0.25, t_final_ratio=0.01, accept0=0.5, n_probe=10):
    u = rng.random(d)
    fu = f(u)
    best_u, best_f = u.copy(), fu
    # calibrate T0 from uphill moves of a few probes (these evaluations are part of the budget and can improve the best)
    ups = []
    for _ in range(min(n_probe, max(1, budget // 10))):
        v = np.clip(u + rng.normal(0, sigma0, d), 0.0, 1.0)
        fv = f(v)
        if fv < best_f:
            best_u, best_f = v.copy(), fv
        delta = np.log1p(fv) - np.log1p(fu)
        if delta > 0:
            ups.append(delta)
    t0 = (np.mean(ups) / -np.log(accept0)) if ups else 1.0
    t0 = max(t0, 1e-6)
    remaining = budget - 1 - min(n_probe, max(1, budget // 10))
    for k in range(remaining):
        T = t0 * (t_final_ratio ** (k / max(1, remaining - 1)))
        sigma = max(0.005, sigma0 * np.sqrt(T / t0))
        v = np.clip(u + rng.normal(0, sigma, d), 0.0, 1.0)
        fv = f(v)
        delta = np.log1p(fv) - np.log1p(fu)
        if delta < 0 or rng.random() < np.exp(-delta / T):
            u, fu = v, fv
        if fu < best_f:
            best_u, best_f = u.copy(), fu
    return best_u, best_f


ALGORITHMS = {"PSO": pso, "GA": ga, "ABC": abc, "SA": sa, "Random search": random_search}


# ----------------------------------------------------------------------------- objective wrapper
class Objective:
    """Unit-cube objective over a TuningProblem with simulation-budget enforcement and best-so-far history."""

    def __init__(self, problem, budget, kind="npid", tune_shape=False, scenario="step", noise_seeds=(100, 101, 102),
                 recorder=None):
        self.recorder = recorder          # optional list receiving (log10 Kp, Ki, Kd, cost) for every evaluation
        self.p, self.budget, self.kind, self.scenario = problem, budget, kind, scenario
        self.seeds = noise_seeds
        self.cost_per_call = len(noise_seeds) if scenario == "noise" else 1
        self.start = problem.evals
        lb, ub = np.log10(problem.bounds[0]), np.log10(problem.bounds[1])
        self.lb, self.ub = list(lb), list(ub)
        self.names = list(GAIN_NAMES)
        self.log_dims = [True] * 3
        if tune_shape and kind == "npid":
            self.names += list(SHAPE_NAMES)
            self.lb += [SHAPE_RANGE[0]] * 3
            self.ub += [SHAPE_RANGE[1]] * 3
            self.log_dims += [False] * 3
        self.lb, self.ub = np.array(self.lb), np.array(self.ub)
        self.d = len(self.names)
        self.best_f, self.best_u, self.history = np.inf, None, []

    def decode(self, u):
        x = self.lb + np.asarray(u) * (self.ub - self.lb)
        return {n: float(10 ** xi if lg else xi) for n, xi, lg in zip(self.names, x, self.log_dims)}

    def __call__(self, u):
        if self.p.evals - self.start + self.cost_per_call > self.budget:
            raise BudgetExhausted
        c = self.p.cost(self.decode(u), self.kind, self.scenario, self.seeds)
        if c < self.best_f:
            self.best_f, self.best_u = c, np.array(u, dtype=float)
        self.history.append((self.p.evals - self.start, self.best_f))
        if self.recorder is not None:
            self.recorder.append((*(self.lb[:3] + np.asarray(u)[:3] * (self.ub[:3] - self.lb[:3])), c))
        return c


# ----------------------------------------------------------------------------- tuner classes
class _Optimizer(Tuner):
    family = "optimization"
    stochastic = True
    algo_name = "PSO"

    def __init__(self, kind="npid", tune_shape=False, scenario="step", name=None, recorder=None, **hyper):
        self.kind, self.tune_shape, self.scenario, self.hyper = kind, tune_shape, scenario, hyper
        self.recorder = recorder
        tag = ("" if kind == "npid" else " [L-PID]") + (" +shape" if tune_shape else "") + (" noise-aware" if scenario == "noise" else "")
        self.name = name or f"{self.algo_name}{tag}"

    def _tune(self, problem, budget, seed):
        budget = budget or 400
        obj = Objective(problem, budget, self.kind, self.tune_shape, self.scenario, recorder=self.recorder)
        rng = np.random.default_rng(seed)
        try:
            ALGORITHMS[self.algo_name](obj, obj.d, budget, rng, **self.hyper)
        except BudgetExhausted:
            pass
        if obj.best_u is None:
            raise RuntimeError("optimizer produced no evaluation")
        gains = obj.decode(obj.best_u)
        return gains, {"train_cost": float(obj.best_f), "kind": self.kind}, obj.history


class PSO(_Optimizer):
    algo_name = "PSO"


class GA(_Optimizer):
    algo_name = "GA"


class ABC(_Optimizer):
    algo_name = "ABC"


class SA(_Optimizer):
    algo_name = "SA"


class RandomSearch(_Optimizer):
    algo_name = "Random search"
