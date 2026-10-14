"""
ML-based tuners (docs/METHODS_ROADMAP.md, phase M4): regression (SVR / NN / ...), ANN with a task-level loss, and
Gaussian-process (Bayesian) optimisation. Reinforcement learning is in `src/tuners/rl.py`.

Common idea of the supervised tuners: a counted bump test gives plant-agnostic features (`features.py`); a trained
model maps them to the *dimensionless* log-gains z = log10(Kp K_eff, Ki K_eff T_n, Kd K_eff / T_n); the gains are
recovered with the plant's own K_eff, T_n. Tuning cost = the bump test (3-6 experiments) + a model evaluation.

Data: `Bundle` (loaded from the JSONL of experiments/m4a_dataset.py), per plant: features, signature, the label
(best of GA and PSO), and every evaluated (z, log cost) pair of the labelling runs.
"""

import json
import time
from dataclasses import dataclass, field

import numpy as np

from src.tuners.base import FAIL_COST, GAIN_NAMES, NotApplicable, Tuner, default_bounds
from src.tuners.features import denormalise_gains, extract_features


# ----------------------------------------------------------------------------- data
@dataclass
class Bundle:
    ident: np.ndarray
    ptype: np.ndarray
    split: np.ndarray
    X_eng: np.ndarray                  # (N, 8) engineered dimensionless features
    X_sig: np.ndarray                  # (N, 24) normalised step-response signature
    K_eff: np.ndarray
    T_n: np.ndarray
    y_ref: np.ndarray
    best_z: np.ndarray                 # (N, 3) normalised log-gains of the label
    best_cost: np.ndarray              # physical ITAE of the label
    eval_z: list = field(default_factory=list)          # per plant (n_i, 3)
    eval_logcost: list = field(default_factory=list)    # per plant (n_i,) log10 normalised ITAE
    specs: list = field(default_factory=list)
    feat_sim_seconds: np.ndarray = None

    def __len__(self):
        return len(self.ident)

    def X(self, which="engineered"):
        if which == "engineered":
            return self.X_eng
        if which == "signature":
            return np.hstack([self.X_sig, self.X_eng[:, [0, 3, 4]]])
        if which == "both":
            return np.hstack([self.X_eng, self.X_sig])
        if which == "raw":                                  # no normalisation: absolute scale features instead
            return np.hstack([self.X_eng, np.log10(self.K_eff)[:, None], np.log10(self.T_n)[:, None],
                              np.log10(np.abs(self.y_ref))[:, None]])
        raise ValueError(which)

    def select(self, mask):
        idx = np.where(mask)[0]
        return Bundle(self.ident[idx], self.ptype[idx], self.split[idx], self.X_eng[idx], self.X_sig[idx],
                      self.K_eff[idx], self.T_n[idx], self.y_ref[idx], self.best_z[idx], self.best_cost[idx],
                      [self.eval_z[i] for i in idx], [self.eval_logcost[i] for i in idx], [self.specs[i] for i in idx],
                      self.feat_sim_seconds[idx])

    def targets(self, variant="best"):
        """Regression targets z: the label ('best') or the mean of the best k% evaluations ('top5', 'top1')."""
        if variant == "best":
            return self.best_z
        frac = {"top5": 0.05, "top1": 0.01}[variant]
        out = []
        for z, lc in zip(self.eval_z, self.eval_logcost):
            k = max(3, int(frac * len(lc)))
            out.append(z[np.argsort(lc)[:k]].mean(axis=0))
        return np.array(out)


def load_bundle(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r.get("status") == "ok"]
    rows.sort(key=lambda r: r["ident"])
    return Bundle(
        ident=np.array([r["ident"] for r in rows]), ptype=np.array([r["ptype"] for r in rows]),
        split=np.array([r["split"] for r in rows]), X_eng=np.array([r["features"] for r in rows]),
        X_sig=np.array([r["signature"] for r in rows]), K_eff=np.array([r["K_eff"] for r in rows]),
        T_n=np.array([r["T_n"] for r in rows]), y_ref=np.array([r["spec"]["y_ref"] for r in rows]),
        best_z=np.array([r["best_z"] for r in rows]), best_cost=np.array([r["best_cost"] for r in rows]),
        eval_z=[np.array(r["eval_z"]) for r in rows], eval_logcost=[np.array(r["eval_logcost"]) for r in rows],
        specs=[r["spec"] for r in rows], feat_sim_seconds=np.array([r["feat_sim_seconds"] for r in rows]))


def noisy_bundle(bundle, npz_path):
    """
    Same plants and labels, but with the features / normalisation constants of a *noisy* bump test (the realistic
    deployment condition). The labels are re-expressed in the noisy constants (exact shift in log space):
        z0 = log10(Kp K), z1 = log10(Ki K T), z2 = log10(Kd K / T)  ->  + [lK, lK + lT, lK - lT]
    with lK = log10(K_noisy / K_clean), lT = log10(T_noisy / T_clean).
    """
    z = np.load(npz_path)
    assert np.array_equal(z["ident"], bundle.ident), "feature file does not match the dataset"
    c = bundle.select(np.ones(len(bundle), dtype=bool))
    lK, lT = np.log10(z["K_eff"] / bundle.K_eff), np.log10(z["T_n"] / bundle.T_n)
    shift = np.stack([lK, lK + lT, lK - lT], axis=1)
    c.X_eng, c.X_sig, c.K_eff, c.T_n = z["X_eng"], z["X_sig"], z["K_eff"], z["T_n"]
    c.best_z = bundle.best_z + shift
    c.eval_z = [ez + shift[i] for i, ez in enumerate(bundle.eval_z)]
    return c


# ----------------------------------------------------------------------------- supervised regression
class ConstantModel:
    """The 'no learning' baseline: the median normalised label of the training plants."""

    def fit(self, X, Y):
        self.value = np.median(Y, axis=0)
        return self

    def predict(self, X):
        return np.tile(self.value, (len(X), 1))


def make_model(name, seed=0):
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.svm import SVR
    if name == "constant":
        return ConstantModel()
    if name == "ridge-poly2":
        return make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1.0))
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7, weights="distance"))
    if name == "svr":                                      # RBF SVR per output, C / gamma chosen by inner cross-validation
        grid = {"estimator__C": [1.0, 10.0, 100.0], "estimator__gamma": ["scale", 0.03, 0.1], "estimator__epsilon": [0.02, 0.1]}
        base = MultiOutputRegressor(SVR(kernel="rbf"))
        return make_pipeline(StandardScaler(), GridSearchCV(base, grid, cv=4, scoring="neg_mean_absolute_error", n_jobs=1))
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000, early_stopping=True,
                                                             n_iter_no_change=30, random_state=seed))
    if name == "extra-trees":
        return make_pipeline(StandardScaler(), ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1))
    raise ValueError(name)


class SupervisedTuner(Tuner):
    """Bump test -> features -> trained regressor -> gains. `model.predict` maps feature rows to normalised log-gains."""
    family = "ml"
    stochastic = False

    def __init__(self, model, features="engineered", name="Regression", clip=True):
        self.model, self.features, self.name, self.clip = model, features, name, clip

    def _tune(self, problem, budget, seed):
        feat = extract_features(problem)
        x = feat.x(self.features if self.features in ("engineered", "signature", "both") else "engineered")[None, :]
        t0 = time.perf_counter()
        z = np.asarray(self.model.predict(x))[0]
        infer = time.perf_counter() - t0
        gains = denormalise_gains(z, feat.K_eff, feat.T_n)
        if self.clip:
            lb, ub = problem.bounds
            gains = {k: float(np.clip(gains[k], lo, hi)) for k, lo, hi in zip(GAIN_NAMES, lb, ub)}
        return gains, dict(inference_seconds=infer, K_eff=feat.K_eff, T_n=feat.T_n, plant_kind=feat.kind)


# ----------------------------------------------------------------------------- ANN with a task-level loss
class ANNPolicy:
    """
    Neural policy z = pi(x) trained *through a learned cost surrogate* (decision-focused learning):
      1. surrogate S(x, z) ~ log10 excess cost, trained on all evaluated (plant, gains, cost) triples of the labelling runs;
      2. policy pi(x) trained to minimise S(x, pi(x)) plus a small supervised anchor to the smoothed label,
         with the output confined to the range of the data (a sigmoid onto [z_lo, z_hi]) so the surrogate is not
         queried far from where it was trained.
    An ensemble of policies is averaged. `loss` in {"supervised", "decision", "combined"} selects the training signal.
    """

    def __init__(self, features="signature", hidden=(64, 64), ensemble=5, loss="combined", lam_sup=0.3,
                 epochs_surrogate=40, epochs_policy=80, seed=0, label_variant="top5"):
        self.features, self.hidden, self.ensemble, self.loss, self.lam = features, hidden, ensemble, loss, lam_sup
        self.epochs_s, self.epochs_p, self.seed, self.label_variant = epochs_surrogate, epochs_policy, seed, label_variant
        self.policies, self.surrogate = [], None

    @staticmethod
    def _mlp(n_in, n_out, hidden):
        import torch.nn as nn
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_out))
        return nn.Sequential(*layers)

    def fit(self, bundle):
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = bundle.X(self.features)
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-8
        Xn = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
        Zs = bundle.targets(self.label_variant)
        self.z_lo, self.z_hi = np.percentile(np.vstack(bundle.eval_z), [0.5, 99.5], axis=0)
        z_lo, z_hi = torch.tensor(self.z_lo, dtype=torch.float32), torch.tensor(self.z_hi, dtype=torch.float32)
        Zt = torch.tensor(Zs, dtype=torch.float32)

        if self.loss in ("decision", "combined"):
            # ---- surrogate on (x, z) -> excess log cost
            rows_x, rows_z, rows_y = [], [], []
            for i in range(len(bundle)):
                lc, ez = bundle.eval_logcost[i], bundle.eval_z[i]
                keep = rng.choice(len(lc), size=min(300, len(lc)), replace=False)
                rows_x.append(np.repeat(Xn[i:i + 1].numpy(), len(keep), axis=0))
                rows_z.append(ez[keep])
                rows_y.append(np.clip(lc[keep] - lc.min(), 0.0, 6.0))
            SX = torch.tensor(np.vstack(rows_x), dtype=torch.float32)
            SZ = torch.tensor((np.vstack(rows_z) - (self.z_lo + self.z_hi) / 2) / ((self.z_hi - self.z_lo) / 2), dtype=torch.float32)
            SY = torch.tensor(np.concatenate(rows_y), dtype=torch.float32)
            self.surrogate = self._mlp(SX.shape[1] + 3, 1, (128, 128, 64))
            opt = torch.optim.Adam(self.surrogate.parameters(), lr=2e-3)
            n = len(SX)
            for ep in range(self.epochs_s):
                perm = torch.randperm(n)
                for b in range(0, n, 1024):
                    idx = perm[b:b + 1024]
                    loss = torch.nn.functional.smooth_l1_loss(self.surrogate(torch.cat([SX[idx], SZ[idx]], 1)).squeeze(1), SY[idx])
                    opt.zero_grad(); loss.backward(); opt.step()
            for p_ in self.surrogate.parameters():
                p_.requires_grad_(False)

        # ---- policy ensemble
        self.policies = []
        for m in range(self.ensemble):
            torch.manual_seed(self.seed * 100 + m)
            pol = self._mlp(Xn.shape[1], 3, self.hidden)
            opt = torch.optim.Adam(pol.parameters(), lr=2e-3)
            n = len(Xn)
            idx_boot = torch.tensor(rng.integers(0, n, n))                      # bootstrap for ensemble diversity
            for ep in range(self.epochs_p):
                perm = idx_boot[torch.randperm(n)]
                for b in range(0, n, 64):
                    idx = perm[b:b + 64]
                    z = z_lo + (z_hi - z_lo) * torch.sigmoid(pol(Xn[idx]))
                    loss = torch.zeros(())
                    if self.loss in ("supervised", "combined"):
                        sup = ((z - Zt[idx]) ** 2).mean()
                        loss = loss + (sup if self.loss == "supervised" else self.lam * sup)
                    if self.loss in ("decision", "combined"):
                        zn = (z - (z_lo + z_hi) / 2) / ((z_hi - z_lo) / 2)
                        loss = loss + self.surrogate(torch.cat([Xn[idx], zn], 1)).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            self.policies.append(pol)
        return self

    def predict(self, X):
        import torch
        Xn = torch.tensor((np.asarray(X) - self.mu) / self.sd, dtype=torch.float32)
        z_lo, z_hi = torch.tensor(self.z_lo, dtype=torch.float32), torch.tensor(self.z_hi, dtype=torch.float32)
        with torch.no_grad():
            zs = [(z_lo + (z_hi - z_lo) * torch.sigmoid(p(Xn))).numpy() for p in self.policies]
        return np.mean(zs, axis=0)

    def predict_with_spread(self, X):
        import torch
        Xn = torch.tensor((np.asarray(X) - self.mu) / self.sd, dtype=torch.float32)
        z_lo, z_hi = torch.tensor(self.z_lo, dtype=torch.float32), torch.tensor(self.z_hi, dtype=torch.float32)
        with torch.no_grad():
            zs = np.stack([(z_lo + (z_hi - z_lo) * torch.sigmoid(p(Xn))).numpy() for p in self.policies])
        return zs.mean(axis=0), zs.std(axis=0)


# ----------------------------------------------------------------------------- Gaussian-process optimisation
class BayesOptTuner(Tuner):
    """
    Bayesian optimisation of log10(Kp, Ki, Kd) with a Gaussian-process surrogate of log10 ITAE (Matern-5/2 ARD kernel
    + noise), expected improvement maximised over random candidates + local perturbations of the best points.
    n_init Latin-hypercube points (optionally preceded by a warm-start point), then one evaluation per iteration;
    kernel hyper-parameters are re-fitted every `refit_every` evaluations and kept in between.
    """
    family = "ml"
    stochastic = True

    def __init__(self, n_init=8, refit_every=5, n_candidates=2000, warm_start=None, kind="npid", name=None,
                 xi=0.01):
        self.n_init, self.refit_every, self.n_cand, self.warm, self.kind, self.xi = n_init, refit_every, n_candidates, warm_start, kind, xi
        self.name = name or "Gaussian-process optimisation"
        self.trace = []

    @staticmethod
    def _lhs(rng, n, d):
        cut = (np.arange(n)[:, None] + rng.random((n, d))) / n
        for j in range(d):
            cut[:, j] = cut[rng.permutation(n), j]
        return cut

    def _tune(self, problem, budget, seed):
        from scipy.stats import norm
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        budget = budget or 60
        rng = np.random.default_rng(seed)
        lb, ub = np.log10(problem.bounds[0]), np.log10(problem.bounds[1])
        to_gains = lambda u: dict(zip(GAIN_NAMES, 10.0 ** (lb + u * (ub - lb))))
        U, Y = [], []
        history = []

        def evaluate(u):
            c = problem.cost(to_gains(u), self.kind)
            U.append(np.array(u, dtype=float))
            Y.append(c)
            history.append((problem.evals, float(min(Y))))
            self.trace.append((np.array(u), c))

        init = list(self._lhs(rng, self.n_init, 3))
        if self.warm is not None:
            w = np.log10([max(self.warm[k], 1e-12) for k in GAIN_NAMES])
            init.insert(0, np.clip((w - lb) / (ub - lb), 0.0, 1.0))
        for u in init[:budget]:
            evaluate(u)
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(length_scale=[0.3] * 3, length_scale_bounds=(0.03, 3.0), nu=2.5) \
            + WhiteKernel(1e-3, (1e-6, 1e-1))
        gp = None
        while problem.evals < budget:
            y = np.log10(np.minimum(np.array(Y), 1e3 * np.min(Y)))               # damp divergent runs
            y_n = (y - y.mean()) / (y.std() + 1e-9)
            refit = gp is None or (len(Y) % self.refit_every == 0)
            if refit:
                gp = GaussianProcessRegressor(kernel=kernel if gp is None else gp.kernel_, normalize_y=False,
                                              n_restarts_optimizer=1 if gp is None else 0,
                                              random_state=int(rng.integers(1 << 30)))
            else:
                gp = GaussianProcessRegressor(kernel=gp.kernel_, optimizer=None, normalize_y=False)
            gp.fit(np.array(U), y_n)
            best = y_n.min()
            cand = rng.random((self.n_cand, 3))
            top = np.array(U)[np.argsort(y_n)[:3]]
            local = np.clip(top[rng.integers(0, len(top), 400)] + rng.normal(0, 0.05, (400, 3)), 0.0, 1.0)
            cand = np.vstack([cand, local])
            mu, sd = gp.predict(cand, return_std=True)
            imp = best - mu - self.xi
            z = imp / np.maximum(sd, 1e-12)
            ei = imp * norm.cdf(z) + sd * norm.pdf(z)
            evaluate(cand[int(np.argmax(ei))])
        i = int(np.argmin(Y))
        self.last_gp, self.last_U, self.last_Y = gp, np.array(U), np.array(Y)      # for in-process diagnostics
        return to_gains(U[i]), dict(train_cost=float(Y[i]), n_gp_fits=len(Y)), history
