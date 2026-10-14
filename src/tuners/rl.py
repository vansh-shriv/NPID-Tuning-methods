"""
Reinforcement-learning tuner (docs/METHODS_ROADMAP.md, phase M4): a learned *local search* over normalised log-gains.

Episode = one tuning session on one plant with a budget of `n_sims` closed-loop simulations (default 8):
    sim 1       evaluate the start point z0 (the median label of the training plants: a constant, plant-independent start)
    sims 2..n   the agent proposes a change dz = step_size * a (a in [-1, 1]^3) of the current point, one simulation each
Observation: the plant's dimensionless features (bump test), the current point, how much the cost improved so far, the
shape of the last response (overshoot, final error, rise time) and the remaining budget.
Reward: improvement of the best log10 cost so far (non-negative, exploration is not punished).
At deployment the policy runs the same session on a new plant and returns the best point found. The point of the
design is that the agent can be compared, at an *identical simulation budget*, with a random local search and with
the supervised predictions (which cost no simulations beyond the bump test): does the learned search policy help?

Algorithm: PPO (stable-baselines3, vectorised environments). Training plants come from the dataset of m4a; each reset
draws a plant, builds its closed loop from the spec and re-uses the stored features and normalisation constants.
"""

import numpy as np

from src.metrics import compute_metrics
from src.plant_family import problem_from_spec, spec_from_dict
from src.tuners.base import FAIL_COST, GAIN_NAMES, Tuner
from src.tuners.features import denormalise_gains, extract_features, normalise_cost

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:                                           # pragma: no cover
    gym = None

def response_shape(problem, res, T_n):
    """(overshoot fraction, final error fraction, rise time / T_n) of a step response, each clipped."""
    y = res.omega
    if not np.all(np.isfinite(y)):
        return np.array([1.0, 1.0, 5.0])
    m = compute_metrics(res.time, problem.y_ref, y)
    ess = abs(problem.y_ref - np.mean(y[-max(5, len(y) // 50):])) / abs(problem.y_ref)
    rise = m["rise_time"] / T_n if np.isfinite(m["rise_time"]) else 5.0
    return np.array([min(m["overshoot_pct"] / 100.0, 1.0), min(ess, 1.0), min(rise, 5.0)])


def log_cost(problem, res, T_n, cap):
    """log10 normalised ITAE of a result, capped at `cap` (diverging runs)."""
    itae = problem.itae(res)
    if itae >= FAIL_COST:
        return cap
    return float(min(np.log10(max(normalise_cost(itae, problem.y_ref, T_n), 1e-12)), cap))


def make_observation(feat_vec, z, z_lo, z_hi, start_lc, last_lc, best_lc, shape, step, n_sims):
    """Observation vector: features (8), normalised point (3), improvement of the last/best point (2), response shape (3), progress (1)."""
    zn = 2.0 * (np.asarray(z) - z_lo) / (z_hi - z_lo) - 1.0
    return np.concatenate([feat_vec, zn, [np.clip(start_lc - last_lc, -3.0, 6.0), np.clip(start_lc - best_lc, 0.0, 6.0)],
                           shape, [step / n_sims]]).astype(np.float32)


OBS_SIZE = 8 + 3 + 2 + 3 + 1


class TuningEnv(gym.Env if gym is not None else object):
    """Gymnasium environment of a tuning session (see module docstring)."""

    metadata = {"render_modes": []}

    def __init__(self, bundle, n_sims=8, step_size=0.4, z_start=None, seed=0, cap=3.0):
        self.bundle, self.n_sims, self.step_size, self.cap = bundle, n_sims, step_size, cap
        self.z_start = np.median(bundle.best_z, axis=0) if z_start is None else np.asarray(z_start)
        self.z_lo, self.z_hi = np.percentile(np.vstack(bundle.eval_z), [0.5, 99.5], axis=0)
        self.rng = np.random.default_rng(seed)
        self.observation_space = spaces.Box(-20.0, 20.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def _simulate(self, z, cap):
        g = denormalise_gains(z, self.K_eff, self.T_n)
        res = self.problem.simulate(g)
        return log_cost(self.problem, res, self.T_n, cap), response_shape(self.problem, res, self.T_n), g

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        i = int(self.rng.integers(len(self.bundle)))
        spec = spec_from_dict(self.bundle.specs[i])
        self.problem = problem_from_spec(spec)
        self.feat = self.bundle.X_eng[i]
        self.K_eff, self.T_n = float(self.bundle.K_eff[i]), float(self.bundle.T_n[i])
        self.index = i
        self.z = self.z_start.copy()
        lc, shape, g = self._simulate(self.z, 6.0)
        self.start_lc, self.last_lc, self.best_lc, self.shape = lc, lc, lc, shape
        self.best_z, self.step_count = self.z.copy(), 1
        return make_observation(self.feat, self.z, self.z_lo, self.z_hi, self.start_lc, self.last_lc, self.best_lc,
                                self.shape, self.step_count, self.n_sims), {}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.z = np.clip(self.z + self.step_size * a, self.z_lo - 0.5, self.z_hi + 0.5)
        lc, shape, g = self._simulate(self.z, self.start_lc + self.cap)
        prev_best = self.best_lc
        self.last_lc, self.shape = lc, shape
        if lc < self.best_lc:
            self.best_lc, self.best_z = lc, self.z.copy()
        self.step_count += 1
        reward = float(prev_best - self.best_lc)
        done = self.step_count >= self.n_sims
        obs = make_observation(self.feat, self.z, self.z_lo, self.z_hi, self.start_lc, self.last_lc, self.best_lc,
                               self.shape, self.step_count, self.n_sims)
        return obs, reward, done, False, {"best_lc": self.best_lc, "gap_to_label": self.best_lc - self._label_lc()}

    def _label_lc(self):
        return float(np.log10(normalise_cost(self.bundle.best_cost[self.index], self.problem.y_ref, self.T_n)))


def make_env(bundle_path, split="train", seed=0, **kw):
    """Picklable environment factory for SubprocVecEnv (the bundle is loaded inside the worker)."""
    from src.tuners.ml import load_bundle

    def _init():
        b = load_bundle(bundle_path)
        b = b.select(b.split == split)
        return TuningEnv(b, seed=seed, **kw)
    return _init


# ----------------------------------------------------------------------------- tuners
class _SessionTuner(Tuner):
    """Shared session logic: start point, simulation budget, best-so-far."""
    family = "ml"
    stochastic = True

    def __init__(self, z_start, z_lo, z_hi, n_sims=8, step_size=0.4, name=None):
        self.z_start, self.z_lo, self.z_hi = np.asarray(z_start), np.asarray(z_lo), np.asarray(z_hi)
        self.n_sims, self.step_size = n_sims, step_size
        if name:
            self.name = name

    def _propose(self, obs, rng, z, best_z):
        raise NotImplementedError

    def _tune(self, problem, budget, seed):
        n_sims = budget or self.n_sims
        feat = extract_features(problem)                                      # counted bump test
        rng = np.random.default_rng(seed)
        z = self.z_start.copy()
        res = problem.simulate(denormalise_gains(z, feat.K_eff, feat.T_n))
        cap_ref = log_cost(problem, res, feat.T_n, 6.0)
        start_lc = last_lc = best_lc = cap_ref
        shape, best_z = response_shape(problem, res, feat.T_n), z.copy()
        history = [(problem.evals, best_lc)]
        for step in range(1, n_sims):
            obs = make_observation(feat.vec, z, self.z_lo, self.z_hi, start_lc, last_lc, best_lc, shape, step, n_sims)
            a = np.clip(self._propose(obs, rng, z, best_z), -1.0, 1.0)
            z = np.clip(z + self.step_size * a, self.z_lo - 0.5, self.z_hi + 0.5)
            res = problem.simulate(denormalise_gains(z, feat.K_eff, feat.T_n))
            last_lc = log_cost(problem, res, feat.T_n, start_lc + 3.0)
            shape = response_shape(problem, res, feat.T_n)
            if last_lc < best_lc:
                best_lc, best_z = last_lc, z.copy()
            history.append((problem.evals, best_lc))
        gains = denormalise_gains(best_z, feat.K_eff, feat.T_n)
        lb, ub = problem.bounds
        gains = {k: float(np.clip(gains[k], lo, hi)) for k, lo, hi in zip(GAIN_NAMES, lb, ub)}
        return gains, dict(best_logcost=best_lc, start_logcost=start_lc), history


class RLTuner(_SessionTuner):
    """Runs a trained PPO policy as the proposal rule (deterministic actions)."""

    def __init__(self, model, z_start, z_lo, z_hi, n_sims=8, step_size=0.4, name="Reinforcement learning (PPO)"):
        super().__init__(z_start, z_lo, z_hi, n_sims, step_size, name)
        self.model = model

    def _propose(self, obs, rng, z, best_z):
        a, _ = self.model.predict(obs, deterministic=True)
        return a


class RandomLocalSearch(_SessionTuner):
    """Baseline with the same start and budget: Gaussian steps from the best point so far (greedy hill climbing)."""

    def __init__(self, z_start, z_lo, z_hi, n_sims=8, step_size=0.4, sigma=0.5, name="Random local search"):
        super().__init__(z_start, z_lo, z_hi, n_sims, step_size, name)
        self.sigma = sigma
        self._anchor = None

    def _propose(self, obs, rng, z, best_z):
        return (best_z - z) / self.step_size + rng.normal(0, self.sigma, 3)       # perturb around the best point


class ConstantStart(_SessionTuner):
    """No search: the start point (a constant normalised gain set), after the bump test; uses 1 simulation (the start)."""

    def __init__(self, z_start, z_lo, z_hi, name="Constant gains (no learning)"):
        super().__init__(z_start, z_lo, z_hi, 1, 0.0, name)

    def _propose(self, obs, rng, z, best_z):
        return np.zeros(3)
