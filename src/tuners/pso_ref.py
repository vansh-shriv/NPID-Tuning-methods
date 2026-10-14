"""
Reference optimizer used as the performance yardstick in every phase: PSO (pyswarms) on the common interface.

(M3 adds from-scratch PSO/GA/ABC/SA implementations with identical bookkeeping; this wrapper exists so that
M1/M2 can already compare against an optimization-based tuner with counted cost.)
"""

import logging

import numpy as np
import pyswarms as ps

from src.tuners.base import GAIN_NAMES, Tuner


class PSOReference(Tuner):
    name = "PSO (reference)"
    family = "optimization"
    stochastic = True

    def __init__(self, n_particles=16, restarts=1, kind="npid"):
        self.n_particles, self.restarts, self.kind = n_particles, restarts, kind

    def _tune(self, problem, budget, seed):
        budget = budget or 400
        iters = max(1, budget // (self.n_particles * self.restarts))
        lb, ub = np.log10(problem.bounds[0]), np.log10(problem.bounds[1])
        logging.getLogger("pyswarms").setLevel(logging.WARNING)

        def objective(X):
            return np.array([problem.cost(dict(zip(GAIN_NAMES, 10.0 ** x)), self.kind) for x in X])

        best_c, best_x, history = np.inf, None, []
        for r in range(self.restarts):
            np.random.seed(seed * 1000 + r)
            opt = ps.single.GlobalBestPSO(n_particles=self.n_particles, dimensions=3,
                                          options={"c1": 0.5, "c2": 0.3, "w": 0.9}, bounds=(lb, ub))
            cost, pos = opt.optimize(objective, iters=iters, verbose=False)
            for i, c in enumerate(opt.cost_history):
                history.append(((r * iters + i + 1) * self.n_particles, float(min(c, best_c))))
            if cost < best_c:
                best_c, best_x = cost, pos
        return dict(zip(GAIN_NAMES, 10.0 ** best_x)), {"train_cost": float(best_c)}, history
