"""Validation of the ML tuners: data handling, regressors, decision-focused ANN, Bayesian optimisation, RL session logic."""
import importlib.util
import json
import os

import numpy as np
import pytest

from src.plant_family import PlantSpec, problem_from_spec, sample_specs, spec_to_dict
from src.tuners import TuningProblem
from src.tuners.features import extract_features, normalise_gains
from src.tuners.ml import ANNPolicy, BayesOptTuner, Bundle, ConstantModel, SupervisedTuner, load_bundle, make_model

_spec = importlib.util.spec_from_file_location(
    "m4a", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "m4a_dataset.py"))
m4a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m4a)


@pytest.fixture(scope="module")
def mini_jsonl(tmp_path_factory):
    """A real (tiny) labelled dataset: 4 plants, 30-simulation labels."""
    path = tmp_path_factory.mktemp("m4") / "mini.jsonl"
    with open(path, "w") as f:
        for s in sample_specs(4, seed=5):
            f.write(json.dumps(m4a.label_plant(spec_to_dict(s), budget=30)) + "\n")
    return str(path)


def synthetic_bundle(n=60, seed=0):
    """Plants whose cost is an exact bowl around a known optimum that depends linearly on the features."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    best_z = np.stack([0.5 + 0.3 * X[:, 1], 0.2 - 0.4 * X[:, 2], -0.5 + 0.2 * X[:, 3]], axis=1)
    ez, elc = [], []
    for i in range(n):
        z = best_z[i] + rng.normal(0, 0.6, size=(200, 3))
        ez.append(z)
        elc.append(np.log10(1.0 + ((z - best_z[i]) ** 2).sum(axis=1)))
    return Bundle(ident=np.arange(n), ptype=np.array(["fopdt"] * n), split=np.array(["train"] * n), X_eng=X,
                  X_sig=rng.normal(size=(n, 24)), K_eff=np.ones(n), T_n=np.ones(n), y_ref=np.ones(n), best_z=best_z,
                  best_cost=np.ones(n), eval_z=ez, eval_logcost=elc, specs=[{}] * n, feat_sim_seconds=np.zeros(n))


# ---- data ---------------------------------------------------------------------------------------------
def test_bundle_loading_selection_and_target_variants(mini_jsonl):
    b = load_bundle(mini_jsonl)
    assert len(b) == 4 and b.X_eng.shape == (4, 8) and b.X_sig.shape == (4, 24) and b.best_z.shape == (4, 3)
    assert all(len(z) >= 30 for z in b.eval_z)
    sub = b.select(b.ptype == b.ptype[0])
    assert len(sub) >= 1 and sub.ptype[0] == b.ptype[0]
    for variant in ("best", "top5", "top1"):
        assert b.targets(variant).shape == (4, 3)
    # the smoothed label lies inside the cloud of evaluated points
    t = b.targets("top5")
    for i in range(4):
        assert np.all(t[i] >= b.eval_z[i].min(axis=0) - 1e-9) and np.all(t[i] <= b.eval_z[i].max(axis=0) + 1e-9)
    assert b.X("signature").shape == (4, 27) and b.X("both").shape == (4, 32) and b.X("raw").shape[1] == 11


def test_labels_are_normalised_consistently(mini_jsonl):
    rows = [json.loads(l) for l in open(mini_jsonl)]
    r = rows[0]
    z = normalise_gains(r["best_gains"], r["K_eff"], r["T_n"])
    assert np.allclose(z, r["best_z"], atol=1e-4)


# ---- regressors -------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["constant", "ridge-poly2", "knn", "mlp", "extra-trees"])
def test_regressors_fit_a_smooth_mapping_better_than_the_constant_baseline(name):
    b = synthetic_bundle(120)
    X, Y = b.X("engineered"), b.best_z
    tr, te = slice(0, 90), slice(90, 120)
    m = make_model(name).fit(X[tr], Y[tr])
    err = np.abs(m.predict(X[te]) - Y[te]).mean()
    base = np.abs(ConstantModel().fit(X[tr], Y[tr]).predict(X[te]) - Y[te]).mean()
    assert m.predict(X[te]).shape == (30, 3)
    if name != "constant":
        assert err < 0.7 * base, (name, err, base)


def test_svr_with_inner_cross_validation_learns_the_mapping():
    b = synthetic_bundle(100)
    m = make_model("svr").fit(b.X("engineered")[:80], b.best_z[:80])
    err = np.abs(m.predict(b.X("engineered")[80:]) - b.best_z[80:]).mean()
    assert err < 0.12


def test_supervised_tuner_end_to_end_counts_the_bump_test_and_respects_bounds(mini_jsonl):
    b = load_bundle(mini_jsonl)
    tuner = SupervisedTuner(ConstantModel().fit(b.X(), b.best_z), name="constant")
    p = problem_from_spec(PlantSpec("fopdt", K=2.0, tau1=5.0, theta=1.0, y_ref=1.0, r_auth=5.0))
    r = tuner.tune(p)
    assert r.success and 2 <= r.evals <= 12 and "inference_seconds" in r.meta
    lb, ub = p.bounds
    for k, lo, hi in zip(("Kp", "Ki", "Kd"), lb, ub):
        assert lo <= r.gains[k] <= hi


# ---- ANN with a task-level loss --------------------------------------------------------------------------
def test_ann_supervised_training_recovers_the_mapping():
    b = synthetic_bundle(80)
    ann = ANNPolicy(features="engineered", loss="supervised", ensemble=2, epochs_policy=60, hidden=(32, 32), label_variant="best").fit(b)
    z = ann.predict(b.X("engineered"))
    assert np.abs(z - b.best_z).mean() < 0.2
    assert np.all(z >= ann.z_lo - 1e-6) and np.all(z <= ann.z_hi + 1e-6)             # confined to the data range


def test_decision_focused_training_finds_the_cost_minimum_without_labels_in_the_loss():
    """loss='decision' never sees the label: the policy must find the optimum through the learned surrogate alone."""
    b = synthetic_bundle(80)
    dec = ANNPolicy(features="engineered", loss="decision", ensemble=2, epochs_surrogate=25, epochs_policy=40,
                    hidden=(32, 32)).fit(b)
    rand = np.random.default_rng(0).uniform(dec.z_lo, dec.z_hi, size=(80, 3))
    err_dec = np.abs(dec.predict(b.X("engineered")) - b.best_z).mean()
    err_rand = np.abs(rand - b.best_z).mean()
    assert err_dec < 0.6 * err_rand, (err_dec, err_rand)


def test_ann_ensemble_spread_and_determinism():
    b = synthetic_bundle(40)
    a1 = ANNPolicy(features="engineered", loss="supervised", ensemble=3, epochs_policy=10, hidden=(16,), seed=3).fit(b)
    a2 = ANNPolicy(features="engineered", loss="supervised", ensemble=3, epochs_policy=10, hidden=(16,), seed=3).fit(b)
    assert np.allclose(a1.predict(b.X("engineered")), a2.predict(b.X("engineered")))
    mean, spread = a1.predict_with_spread(b.X("engineered"))
    assert mean.shape == spread.shape == (40, 3) and np.all(spread >= 0) and spread.mean() > 0


# ---- Bayesian optimisation -----------------------------------------------------------------------------
class BowlProblem(TuningProblem):
    """Analytic cost: ITAE = 1 + |log10 gains - g*|^2 (counted like a simulation)."""
    optimum = np.array([0.5, 0.0, -1.0])

    def cost(self, gains, kind="npid", scenario="step", noise_seeds=(0,)):
        self._count(1.0)
        z = np.log10([gains["Kp"], gains["Ki"], gains["Kd"]])
        return float(1.0 + np.sum((z - self.optimum) ** 2))


def bowl():
    return BowlProblem("bowl", lambda: None.__class__ and __import__("src.plant", fromlist=["x"]).TransferFunctionPlant([1.0], [1.0, 1.0]),
                       y_ref=1.0, dt=0.05, t_end=5.0, bounds=((1e-2, 1e-2, 1e-3), (1e2, 1e2, 1e1)))


def test_bayesian_optimisation_is_sample_efficient_on_an_analytic_cost():
    from src.tuners.optimizers import GA, RandomSearch
    bo = [BayesOptTuner().tune(bowl(), budget=30, seed=s).cost for s in range(3)]
    rs = [RandomSearch().tune(bowl(), budget=30, seed=s).cost for s in range(3)]
    assert np.median(bo) < 1.1 and np.median(bo) < np.median(rs)
    r = BayesOptTuner().tune(bowl(), budget=30, seed=0)
    assert r.evals == 30 and r.history[-1][1] <= r.history[0][1] and r.history[-1][0] == 30


def test_bayesian_optimisation_is_deterministic_and_honours_a_warm_start():
    a = BayesOptTuner().tune(bowl(), budget=14, seed=4).gains
    b = BayesOptTuner().tune(bowl(), budget=14, seed=4).gains
    assert a == b
    warm = dict(Kp=10 ** 0.5, Ki=1.0, Kd=10 ** -1.0)                                  # exactly the optimum
    r = BayesOptTuner(warm_start=warm).tune(bowl(), budget=10, seed=1)
    assert r.cost == pytest.approx(1.0, abs=1e-6)


def test_noisy_bundle_rebases_labels_without_changing_the_physical_gains(tmp_path):
    from src.tuners.features import denormalise_gains
    from src.tuners.ml import noisy_bundle
    b = synthetic_bundle(6)
    rng = np.random.default_rng(2)
    b.K_eff, b.T_n = rng.uniform(0.5, 3.0, 6), rng.uniform(0.5, 10.0, 6)
    np.savez(tmp_path / "f.npz", X_eng=b.X_eng + 0.1, X_sig=b.X_sig, K_eff=b.K_eff * 1.2, T_n=b.T_n * 0.85, ident=b.ident)
    c = noisy_bundle(b, str(tmp_path / "f.npz"))
    for i in range(6):
        g0 = denormalise_gains(b.best_z[i], b.K_eff[i], b.T_n[i])
        g1 = denormalise_gains(c.best_z[i], c.K_eff[i], c.T_n[i])
        assert (g1["Kp"], g1["Ki"], g1["Kd"]) == pytest.approx((g0["Kp"], g0["Ki"], g0["Kd"]), rel=1e-9)
        e0 = denormalise_gains(b.eval_z[i][3], b.K_eff[i], b.T_n[i])
        e1 = denormalise_gains(c.eval_z[i][3], c.K_eff[i], c.T_n[i])
        assert e1["Ki"] == pytest.approx(e0["Ki"], rel=1e-9)
    assert np.allclose(c.X_eng, b.X_eng + 0.1)
