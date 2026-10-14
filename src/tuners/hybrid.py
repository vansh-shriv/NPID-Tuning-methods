"""
Hybrid tuners (docs/METHODS_ROADMAP.md, phase M5).

    GA-enhanced ML (method 16)   a genetic algorithm chooses the model family, its hyper-parameters and the feature
                                 subset of the supervised tuner of M4, with the cross-validated *closed-loop* regret
                                 as fitness (`ModelGenome`, `cv_regret_fitness`, `HybridModelTuner` deploys the result).
    ML + classical (method 17)   a classical rule (Cohen-Coon) gives the starting gains, a regressor predicts a
                                 multiplicative (additive in log space) correction of them, and an optional short local
                                 search refines the result (`ClassicalMLTuner`, `LocalRefiner`).

Both work in the dimensionless log-gain space z = log10(Kp K_eff, Ki K_eff T_n, Kd K_eff / T_n) of M4, so the same
trained models apply to plants of any scale.
"""

import time
from dataclasses import dataclass

import numpy as np

from src.tuners.base import GAIN_NAMES, Tuner
from src.tuners.features import denormalise_gains, extract_features

N_FEATURES = 8
THETA_FLOOR = 0.05               # dead-time fraction floor of the classical rule (rules diverge as theta -> 0)
Z_RANGE = (-3.0, 4.0)            # clip for classical base gains in z


# ----------------------------------------------------------------------------- classical base in z-space
def cohen_coon_z(vec, theta_floor=THETA_FLOOR):
    """
    Cohen-Coon PID gains in normalised log space from the engineered features of the bump test.

    With T_n = tau + theta and K_eff = K (self-regulating) or Kv T_n (integrating) the identified FOPDT/IPDT is fully
    described by the dead-time fraction f = theta/T_n (feature 1), so the rule needs nothing but the feature vector:
        FOPDT: r = theta/tau,  Kp K = (tau/theta)(4/3 + r/4),  Ti = theta (32 + 6r)/(13 + 8r),  Td = 4 theta/(11 + 2r)
        IPDT : Kp Kv = 4/(3 theta), Ti = 32 theta/13, Td = 4 theta/11
    and, in units of T_n,  z = log10(Kp K_eff, Kp K_eff/Ti_n, Kp K_eff Td_n).
    """
    integrating = vec[0] > 0.5
    th = max(float(vec[1]), theta_floor)
    if integrating:
        kpk, ti, td = 4.0 / (3.0 * th), 32.0 * th / 13.0, 4.0 * th / 11.0
    else:
        tau = max(1.0 - float(vec[1]), 1e-3)
        r = th / tau
        kpk = (tau / th) * (4.0 / 3.0 + r / 4.0)
        ti = th * (32.0 + 6.0 * r) / (13.0 + 8.0 * r)
        td = 4.0 * th / (11.0 + 2.0 * r)
    return np.clip(np.log10([kpk, kpk / ti, kpk * td]), *Z_RANGE)


def classical_base(rule):
    """Base-point function vec -> z for a rule name ('cohen-coon' or 'constant:<z0>,<z1>,<z2>')."""
    if rule == "cohen-coon":
        return cohen_coon_z
    if rule.startswith("constant:"):
        z = np.array([float(v) for v in rule[len("constant:"):].split(",")])
        return lambda vec: z
    raise ValueError(rule)


def base_matrix(X_eng, rule):
    f = classical_base(rule)
    return np.array([f(x) for x in X_eng])


# ----------------------------------------------------------------------------- GA-enhanced ML: genome
MODEL_TYPES = ("svr", "mlp", "extra-trees")
GENOME_DIM = 9 + N_FEATURES


@dataclass
class ModelConfig:
    model: str
    params: dict
    mask: tuple                              # indices of the engineered features used

    def describe(self):
        p = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.params.items())
        return f"{self.model}({p}) feats={list(self.mask)}"


def decode_genome(u):
    """
    Unit-cube genome -> ModelConfig. Genes: 0 model family; 1-3 SVR (log C, log gamma, log epsilon); 4-6 MLP (hidden
    layers, width, log alpha); 7-8 extra-trees (min leaf, max-features fraction); 9.. feature on/off (> 0.5), the two
    strongest genes are forced on if fewer than two features are selected. Genes of unused families are neutral.
    """
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    model = MODEL_TYPES[min(int(u[0] * len(MODEL_TYPES)), len(MODEL_TYPES) - 1)]
    if model == "svr":
        params = dict(C=float(10 ** (-1 + 4 * u[1])), gamma=float(10 ** (-3 + 3.5 * u[2])), epsilon=float(10 ** (-2.5 + 2.2 * u[3])))
    elif model == "mlp":
        params = dict(layers=1 + min(int(u[4] * 3), 2), width=int(round(8 * 16 ** u[5])), alpha=float(10 ** (-5 + 4 * u[6])))
    else:
        params = dict(min_leaf=1 + min(int(u[7] * 8), 7), max_features=float(0.4 + 0.6 * u[8]))
    sel = u[9:9 + N_FEATURES]
    mask = np.where(sel > 0.5)[0]
    if len(mask) < 2:
        mask = np.sort(np.argsort(-sel)[:2])
    return ModelConfig(model, params, tuple(int(i) for i in mask))


class _Masked:
    """Regressor restricted to a feature subset (so it can be fed the full feature vector)."""

    def __init__(self, model, mask):
        self.model, self.mask = model, list(mask)

    def fit(self, X, Y):
        self.model.fit(np.asarray(X)[:, self.mask], Y)
        return self

    def predict(self, X):
        return self.model.predict(np.asarray(X)[:, self.mask])


def build_model(cfg, seed=0):
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.multioutput import MultiOutputRegressor
    p = cfg.params
    if cfg.model == "svr":
        core = make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(kernel="rbf", C=p["C"], gamma=p["gamma"], epsilon=p["epsilon"])))
    elif cfg.model == "mlp":
        core = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(p["width"],) * p["layers"], alpha=p["alpha"], max_iter=800,
                                                             early_stopping=True, n_iter_no_change=20, random_state=seed))
    else:
        core = make_pipeline(StandardScaler(), ExtraTreesRegressor(n_estimators=150, min_samples_leaf=p["min_leaf"],
                                                                    max_features=p["max_features"], random_state=seed, n_jobs=1))
    return _Masked(core, cfg.mask)


def geometric_regret(ratio, cap=100.0):
    """Fitness summary of regrets: geometric mean with each regret clipped to [0.5, cap] (rewards median and tail)."""
    r = np.clip(np.asarray(ratio, dtype=float), 0.5, cap)
    return float(np.exp(np.mean(np.log(r))))


def kfold_indices(n, k, seed):
    perm = np.random.default_rng(seed).permutation(n)
    return np.array_split(perm, k)


def cv_regret_fitness(cfg, X, Y, regret_fn, k=3, seed=0, targets_base=None):
    """
    Cross-validated closed-loop regret of a model configuration: for each fold fit on the other folds, predict z, and
    score by `regret_fn(indices, Z) -> ratios` (the caller simulates the predicted gains). Returns the geometric mean.
    `targets_base` (optional) = base points added to the predicted corrections (hybrid use).
    """
    n = len(X)
    Z = np.zeros((n, 3))
    for fold in kfold_indices(n, k, seed):
        tr = np.setdiff1d(np.arange(n), fold)
        try:
            m = build_model(cfg, seed).fit(X[tr], Y[tr])
            pred = np.asarray(m.predict(X[fold]))
        except Exception:                                    # numerical failure of a candidate: worst fitness
            return float("inf")
        Z[fold] = pred if targets_base is None else pred + targets_base[fold]
    if not np.all(np.isfinite(Z)):
        return float("inf")
    return geometric_regret(regret_fn(np.arange(n), Z))


# ----------------------------------------------------------------------------- deployment of a trained model
class HybridModelTuner(Tuner):
    """Bump test -> features -> (GA-selected) regressor -> gains. Like `SupervisedTuner` but with feature noise handling."""
    family = "hybrid"
    stochastic = False

    def __init__(self, model, name="GA-enhanced ML", clip=True, features="engineered", family="hybrid"):
        self.model, self.name, self.clip, self.features, self.family = model, name, clip, features, family

    def _tune(self, problem, budget, seed):
        feat = extract_features(problem, noise_sigma=problem.noise_sigma, seed=seed)
        t0 = time.perf_counter()
        z = np.asarray(self.model.predict(feat.x(self.features)[None, :]))[0]
        infer = time.perf_counter() - t0
        gains = _clip_gains(denormalise_gains(z, feat.K_eff, feat.T_n), problem)
        return gains, dict(inference_seconds=infer, K_eff=feat.K_eff, T_n=feat.T_n, plant_kind=feat.kind)


def _clip_gains(gains, problem):
    lb, ub = problem.bounds
    return {k: float(np.clip(gains[k], lo, hi)) for k, lo, hi in zip(GAIN_NAMES, lb, ub)}


# ----------------------------------------------------------------------------- local refinement
class LocalRefiner:
    """
    (1+1) evolution strategy with the 1/5-success rule in the normalised log-gain space, started at z0.
    Uses `n_sims` closed-loop simulations of the problem (the first one evaluates the start point), keeps the best.
    """

    def __init__(self, n_sims=20, step=0.3, grow=1.4, shrink=0.8, step_min=0.02, step_max=0.8):
        self.n_sims, self.step, self.grow, self.shrink = n_sims, step, grow, shrink
        self.step_min, self.step_max = step_min, step_max

    def refine(self, problem, z0, K_eff, T_n, seed=0, kind="npid"):
        rng = np.random.default_rng(seed)
        cost = lambda z: problem.cost(_clip_gains(denormalise_gains(z, K_eff, T_n), problem), kind)
        z_best = np.array(z0, dtype=float)
        c_best = cost(z_best)
        history = [(problem.evals, c_best)]
        step = self.step
        for _ in range(self.n_sims - 1):
            z = z_best + step * rng.normal(0.0, 1.0, 3)
            c = cost(z)
            if c < c_best:
                z_best, c_best = z, c
                step = min(step * self.grow, self.step_max)
            else:
                step = max(step * self.shrink, self.step_min)
            history.append((problem.evals, c_best))
        return z_best, c_best, history


# ----------------------------------------------------------------------------- method 17
class ClassicalMLTuner(Tuner):
    """
    Hybrid ML + classical. Components (each can be switched off for the ablation):
      base   classical rule from the bump-test features (`rule`: 'cohen-coon', or 'constant:..' for no classical rule),
      ML     z = base + model(features), the model having been trained on the residual label - base;
             `model=None` gives the base alone,
      refine `refine_sims` closed-loop simulations of local search around the result (0 = none).
    Tuning cost: bump test + refine_sims simulations (+ 1 if refining, for the start point, included in refine_sims).
    """
    family = "hybrid"
    stochastic = True

    def __init__(self, model=None, rule="cohen-coon", refine_sims=0, name=None, step=0.3):
        self.model, self.rule, self.refine_sims, self.step = model, rule, refine_sims, step
        parts = [("Cohen-Coon" if rule == "cohen-coon" else "constant start")] + (["+ ML correction"] if model is not None else []) \
            + ([f"+ {refine_sims}-sim refinement"] if refine_sims else [])
        self.name = name or "Hybrid: " + " ".join(parts)

    @property
    def base(self):                              # not stored: keeps the tuner picklable (multiprocessing)
        return classical_base(self.rule)

    def _tune(self, problem, budget, seed):
        feat = extract_features(problem, noise_sigma=problem.noise_sigma, seed=seed)
        z0 = self.base(feat.vec)
        if self.model is not None:
            z0 = z0 + np.asarray(self.model.predict(feat.vec[None, :]))[0]
        meta = dict(K_eff=feat.K_eff, T_n=feat.T_n, plant_kind=feat.kind, z_start=z0.tolist())
        history = []
        z = z0
        n_sims = budget if budget is not None else self.refine_sims
        if n_sims:
            z, c, history = LocalRefiner(n_sims=n_sims, step=self.step).refine(problem, z0, feat.K_eff, feat.T_n, seed=seed)
            meta["train_cost"] = float(c)
        return _clip_gains(denormalise_gains(z, feat.K_eff, feat.T_n), problem), meta, history
