"""Validation of the hybrid tuners (M5): classical base in z-space, GA genome, GA vs random search, local refinement."""
import numpy as np
import pytest

from src.plant_family import PlantSpec, problem_from_spec, sample_specs
from src.tuners.classical import CohenCoon
from src.tuners.features import extract_features, normalise_gains
from src.tuners.hybrid import (GENOME_DIM, ClassicalMLTuner, LocalRefiner, ModelConfig, base_matrix, build_model,
                               cohen_coon_z, cv_regret_fitness, decode_genome, geometric_regret, kfold_indices)
from src.tuners.optimizers import ga, random_search


# ---- classical base ----------------------------------------------------------------------------------
@pytest.mark.parametrize("K,tau,theta", [(2.0, 5.0, 1.0), (0.5, 3.0, 2.0), (4.0, 10.0, 3.0)])
def test_cohen_coon_z_matches_the_m1_rule_on_a_fopdt(K, tau, theta):
    T_n = tau + theta
    vec = np.array([0.0, theta / T_n, 0, 0, 0, 0, 1.0, 0.6])
    Kp, Ti, Td = CohenCoon()._rule("fopdt", K, tau, theta)
    ref = normalise_gains(dict(Kp=Kp, Ki=Kp / Ti, Kd=Kp * Td), K, T_n)
    assert np.allclose(cohen_coon_z(vec), ref, atol=1e-9)


def test_cohen_coon_z_matches_the_m1_rule_on_an_integrator():
    Kv, theta, tau = 0.4, 1.5, 4.0
    T_n = tau + theta
    vec = np.array([1.0, theta / T_n, 0, 0, 0, 0, 1.0, 0.0])
    Kp, Ti, Td = CohenCoon()._rule("ipdt", Kv, None, theta)
    ref = normalise_gains(dict(Kp=Kp, Ki=Kp / Ti, Kd=Kp * Td), Kv * T_n, T_n)
    assert np.allclose(cohen_coon_z(vec), ref, atol=1e-9)


def test_cohen_coon_z_is_finite_and_bounded_for_vanishing_dead_time():
    z = cohen_coon_z(np.array([0.0, 0.0, 0, 0, 0, 0, 1.0, 0.6]))
    assert np.all(np.isfinite(z)) and np.all(np.abs(z) <= 4.0)
    assert np.allclose(base_matrix(np.tile(np.array([[0.0, 0.3, 0, 0, 0, 0, 1.0, 0.6]]), (3, 1)), "cohen-coon").std(axis=0), 0)


def test_constant_base_returns_the_constant():
    z = base_matrix(np.zeros((2, 8)), "constant:0.5,-0.2,0.1")
    assert np.allclose(z, [[0.5, -0.2, 0.1]] * 2)


# ---- genome --------------------------------------------------------------------------------------------
def test_decode_genome_is_valid_over_the_whole_cube():
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(300):
        cfg = decode_genome(rng.random(GENOME_DIM))
        seen.add(cfg.model)
        assert len(cfg.mask) >= 2 and all(0 <= i < 8 for i in cfg.mask) and list(cfg.mask) == sorted(set(cfg.mask))
        p = cfg.params
        if cfg.model == "svr":
            assert 0.1 <= p["C"] <= 1e3 and 1e-3 <= p["gamma"] <= 10 ** 0.5
        elif cfg.model == "mlp":
            assert p["layers"] in (1, 2, 3) and 8 <= p["width"] <= 128
        else:
            assert 1 <= p["min_leaf"] <= 8 and 0.4 <= p["max_features"] <= 1.0
    assert seen == {"svr", "mlp", "extra-trees"}
    for u in (np.zeros(GENOME_DIM), np.ones(GENOME_DIM)):                 # boundary genomes decode too
        assert decode_genome(u).model in ("svr", "mlp", "extra-trees")


def test_decode_genome_forces_two_features_when_none_selected():
    u = np.full(GENOME_DIM, 0.5)
    u[9:] = 0.1
    u[9 + 3], u[9 + 6] = 0.3, 0.4
    assert decode_genome(u).mask == (3, 6)


def test_every_model_family_trains_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 8))
    Y = np.stack([X[:, 0], X[:, 1] ** 2, X[:, 2]], axis=1)
    for fam, val in (("svr", 0.1), ("mlp", 0.4), ("extra-trees", 0.9)):
        u = np.full(GENOME_DIM, 0.5)
        u[0] = {"svr": 0.1, "mlp": 0.5, "extra-trees": 0.9}[fam]
        u[9:] = 1.0
        m = build_model(decode_genome(u)).fit(X, Y)
        assert m.predict(X[:5]).shape == (5, 3)


def test_selected_features_restrict_the_model_input():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 8))
    Y = np.stack([X[:, 3], X[:, 3], X[:, 3]], axis=1)
    cfg = ModelConfig("extra-trees", dict(min_leaf=1, max_features=1.0), (3, 5))
    m = build_model(cfg).fit(X, Y)
    Xt = X.copy()
    Xt[:, [0, 1, 2, 4, 6, 7]] = rng.normal(size=(120, 6))                   # unselected columns are ignored
    assert np.allclose(m.predict(X), m.predict(Xt))


def test_geometric_regret_clips_and_kfold_partitions():
    assert geometric_regret([1.0, 1.0]) == pytest.approx(1.0)
    assert geometric_regret([1.0, 1e9]) == pytest.approx(10.0)             # cap 100 -> sqrt(100)
    folds = kfold_indices(10, 3, seed=0)
    assert sorted(np.concatenate(folds).tolist()) == list(range(10)) and kfold_indices(10, 3, 0)[0].tolist() == folds[0].tolist()


def test_cv_fitness_prefers_a_good_model_on_an_exact_problem():
    """Regret = exp(distance to the exact optimum): a model that sees the informative feature must score better."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 8))
    best = np.stack([X[:, 1], -X[:, 1], 0.5 * X[:, 1]], axis=1)
    regret = lambda idx, Z: np.exp(np.abs(Z - best[idx]).sum(axis=1))
    good = ModelConfig("extra-trees", dict(min_leaf=1, max_features=1.0), (1, 2))
    bad = ModelConfig("extra-trees", dict(min_leaf=1, max_features=1.0), (4, 5))
    assert cv_regret_fitness(good, X, best, regret) < 0.8 * cv_regret_fitness(bad, X, best, regret)


def test_ga_beats_random_search_on_the_genome_cost():
    """The GA on a rugged genome-shaped function reaches a lower cost than random search with the same evaluations."""
    target = np.random.default_rng(3).random(GENOME_DIM)

    def f(u):
        return float(np.sum((np.asarray(u) - target) ** 2) + 0.2 * np.sum(np.sin(12 * np.asarray(u)) ** 2))

    wins = 0
    for s in range(5):
        _, fg = ga(f, GENOME_DIM, 160, np.random.default_rng(s), n=16)
        _, fr = random_search(f, GENOME_DIM, 160, np.random.default_rng(s))
        wins += fg < fr
    assert wins >= 4


# ---- local refinement ---------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def plant_problem():
    return problem_from_spec(sample_specs(1, seed=11)[0])


def test_refiner_budget_and_monotone_history(plant_problem):
    p = plant_problem
    f = extract_features(p)
    p.reset_counters()
    z0 = cohen_coon_z(f.vec)
    z, c, hist = LocalRefiner(n_sims=8).refine(p, z0, f.K_eff, f.T_n, seed=0)
    assert p.evals == 8 and len(hist) == 8
    assert all(hist[i + 1][1] <= hist[i][1] + 1e-12 for i in range(7))
    assert c <= hist[0][1] + 1e-12                                          # never worse than the start


def test_refiner_is_deterministic_per_seed(plant_problem):
    p = plant_problem
    f = extract_features(p)
    z0 = cohen_coon_z(f.vec)
    a = LocalRefiner(n_sims=6).refine(p, z0, f.K_eff, f.T_n, seed=3)[0]
    b = LocalRefiner(n_sims=6).refine(p, z0, f.K_eff, f.T_n, seed=3)[0]
    c = LocalRefiner(n_sims=6).refine(p, z0, f.K_eff, f.T_n, seed=4)[0]
    assert np.allclose(a, b) and not np.allclose(a, c)


def test_classical_ml_tuner_costs_and_components(plant_problem):
    p = plant_problem
    base = ClassicalMLTuner().tune(p)                                       # Cohen-Coon alone: bump test only
    refined = ClassicalMLTuner(refine_sims=7).tune(p, seed=0)
    assert base.success and refined.success
    assert refined.evals == base.evals + 7                                  # refinement adds exactly its simulations
    assert refined.cost <= p.cost(base.gains) + 1e-9                        # refined start is the base (no model)

    class Shift:                                                            # model that always adds +0.3 to every z
        def predict(self, X):
            return np.full((len(X), 3), 0.3)

    shifted = ClassicalMLTuner(model=Shift()).tune(p)
    for k, lo, hi in zip(("Kp", "Ki", "Kd"), *p.bounds):
        assert shifted.gains[k] == pytest.approx(float(np.clip(base.gains[k] * 10 ** 0.3, lo, hi)), rel=1e-6)
    assert "ML" in ClassicalMLTuner(model=Shift()).name and "refinement" in ClassicalMLTuner(refine_sims=5).name
