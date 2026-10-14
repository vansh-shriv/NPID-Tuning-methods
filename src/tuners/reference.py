"""
Best-known reference optimum per plant, used to measure the gap of every optimizer (M3) and the quality of the
other methods (M4-M7). Not a tuning method: it spends as many simulations as it needs.

best_known(problem, kind, tune_shape):
    * 3-D (Kp, Ki, Kd): log-space grid over the whole box, then Nelder-Mead polish of the best grid points;
    * always: long multi-start PSO (own implementation) followed by Nelder-Mead polish;
    returns the best of everything found. In 6-D (with shape exponents) only the PSO route is used.
"""

import itertools

import numpy as np
from scipy.optimize import minimize

from src.tuners.optimizers import Objective, BudgetExhausted, pso


def _polish(obj, u0, evals, rng_seed=0):
    """Nelder-Mead from u0 inside the cube (cost evaluated through the problem, budget in simulations)."""
    def f(u):
        u = np.clip(u, 0.0, 1.0)
        try:
            return obj(u)
        except BudgetExhausted:
            return np.inf
    minimize(f, u0, method="Nelder-Mead", options=dict(maxfev=evals, xatol=1e-4, fatol=1e-9, initial_simplex=None))


def best_known(problem_factory, kind="npid", tune_shape=False, grid=10, long_budget=800, n_pso=3, nm_evals=200):
    """problem_factory: callable returning a fresh TuningProblem. Returns dict(gains, cost, source, n_evals)."""
    problem = problem_factory()
    total = grid ** 3 + n_pso * long_budget + 4 * nm_evals * 3
    obj = Objective(problem, total * 2, kind, tune_shape)
    cands = []
    if not tune_shape:
        axes = np.linspace(0.0, 1.0, grid)
        pts = np.array(list(itertools.product(axes, repeat=3)))
        vals = np.array([obj(u) for u in pts])
        for i in np.argsort(vals)[:3]:
            cands.append(("grid", pts[i], vals[i]))
    for s in range(n_pso):
        rng = np.random.default_rng(1000 + s)
        try:
            u, fv = pso(obj, obj.d, long_budget, rng)
        except BudgetExhausted:
            break
        cands.append(("pso", u, fv))
    best = min(cands, key=lambda c: c[2])
    for src, u, fv in sorted(cands, key=lambda c: c[2])[:3]:
        _polish(obj, u, nm_evals)
    return dict(gains=obj.decode(obj.best_u), cost=float(obj.best_f),
                source="polished " + best[0], n_evals=problem.evals)
