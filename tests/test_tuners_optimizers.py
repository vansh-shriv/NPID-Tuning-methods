"""Validation of the optimizers on analytic functions (independent of any plant) and of the plant wrapper."""
import numpy as np
import pytest

from src.tuners import TuningProblem
from src.tuners.optimizers import (ABC, ALGORITHMS, GA, PSO, SA, BudgetExhausted, Objective, RandomSearch)

OPT = np.array([0.3, 0.7, 0.45])


def sphere(u):
    return float(np.sum((np.asarray(u) - OPT) ** 2))


def rosenbrock(u):
    x = np.asarray(u) * 4 - 2                           # [-2, 2]^d, optimum at 1 -> u = 0.75
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def rastrigin(u):
    x = (np.asarray(u) - 0.5) * 10.24                    # [-5.12, 5.12]^d, optimum at 0 -> u = 0.5
    return float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def run(algo, f, d, budget, seed):
    calls = {"n": 0, "hist": []}
    best = {"f": np.inf}

    def wrapped(u):
        assert np.all(np.asarray(u) >= -1e-12) and np.all(np.asarray(u) <= 1 + 1e-12), "left the unit cube"
        if calls["n"] >= budget:
            raise BudgetExhausted
        calls["n"] += 1
        v = f(u)
        best["f"] = min(best["f"], v)
        calls["hist"].append(best["f"])
        return v

    try:
        ALGORITHMS[algo](wrapped, d, budget, np.random.default_rng(seed))
    except BudgetExhausted:
        pass
    return calls, best["f"]


@pytest.mark.parametrize("algo", ["PSO", "GA", "ABC", "SA"])
def test_converges_on_sphere(algo):
    res = [run(algo, sphere, 3, 400, s)[1] for s in range(5)]
    assert np.median(res) < 1e-3, (algo, res)


@pytest.mark.parametrize("algo", ["PSO", "GA", "ABC", "SA"])
def test_beats_random_search_on_rosenbrock(algo):
    f = [run(algo, rosenbrock, 3, 600, s)[1] for s in range(7)]
    r = [run("Random search", rosenbrock, 3, 600, s)[1] for s in range(7)]
    assert np.median(f) < np.median(r), (algo, np.median(f), np.median(r))


@pytest.mark.parametrize("algo", ["PSO", "GA", "ABC", "SA"])
def test_budget_is_respected_history_monotone_and_seeded(algo):
    c1, f1 = run(algo, rastrigin, 3, 137, seed=3)
    c2, f2 = run(algo, rastrigin, 3, 137, seed=3)
    c3, f3 = run(algo, rastrigin, 3, 137, seed=4)
    assert c1["n"] <= 137 and c1["n"] >= 100            # uses (almost) all of the budget
    assert f1 == f2 and c1["hist"] == c2["hist"]        # reproducible
    assert f1 != f3                                      # but seed-dependent
    h = np.array(c1["hist"])
    assert np.all(np.diff(h) <= 1e-15)                  # best-so-far never increases


@pytest.mark.parametrize("algo", ["PSO", "GA", "ABC", "SA"])
def test_handles_tiny_budgets_and_failure_plateaus(algo):
    assert run(algo, sphere, 3, 5, 0)[0]["n"] <= 5
    plateau = lambda u: 1e6 if u[0] < 0.5 else float(np.sum((np.asarray(u) - OPT) ** 2))     # FAIL_COST region
    assert run(algo, plateau, 3, 300, 1)[1] < 1.0


def test_finds_global_basin_better_than_start_on_multimodal():
    for algo in ("PSO", "GA", "ABC", "SA"):
        res = [run(algo, rastrigin, 3, 600, s)[1] for s in range(5)]
        assert np.median(res) < 15, (algo, res)           # random points average ~ 70 in 3-D


# ---- plant wrapper ---------------------------------------------------------------------------------
def test_objective_maps_cube_to_gain_box_and_counts_simulations():
    p = TuningProblem.from_scenario("dc_speed")
    obj = Objective(p, budget=5)
    g = obj.decode(np.zeros(3))
    lb, ub = p.bounds
    assert g["Kp"] == pytest.approx(lb[0]) and g["Ki"] == pytest.approx(lb[1]) and g["Kd"] == pytest.approx(lb[2])
    g = obj.decode(np.ones(3))
    assert g["Kp"] == pytest.approx(ub[0])
    for _ in range(5):
        obj(np.random.default_rng(0).random(3))
    assert p.evals == 5
    with pytest.raises(BudgetExhausted):
        obj(np.zeros(3))


def test_objective_with_shape_and_noise_budget_accounting():
    p = TuningProblem.from_scenario("dc_speed")
    obj = Objective(p, budget=7, tune_shape=True, scenario="noise")         # 3 simulations per call
    assert obj.d == 6 and set(obj.decode(np.full(6, 0.5))) == {"Kp", "Ki", "Kd", "n", "m", "p"}
    obj(np.full(6, 0.5))
    obj(np.full(6, 0.5))
    assert p.evals == 6
    with pytest.raises(BudgetExhausted):                                      # 6 + 3 > 7
        obj(np.full(6, 0.5))


@pytest.mark.parametrize("cls", [PSO, GA, ABC, SA, RandomSearch])
def test_tuner_interface_on_a_plant(cls):
    p = TuningProblem.from_scenario("dc_speed")
    r = cls().tune(p, budget=120, seed=0)
    assert r.success and 100 <= r.evals <= 120 and r.sim_seconds == pytest.approx(r.evals * p.t_end)
    assert r.history and r.history[-1][1] <= r.history[0][1]
    lb, ub = p.bounds
    for k, a, b in zip(("Kp", "Ki", "Kd"), lb, ub):
        assert a * 0.999 <= r.gains[k] <= b * 1.001
    assert r.cost == pytest.approx(r.meta["train_cost"])                      # reported training cost
    assert cls().tune(TuningProblem.from_scenario("dc_speed"), budget=120, seed=0).gains == r.gains


def test_lpid_and_shape_variants():
    p = TuningProblem.from_scenario("dc_speed")
    r = PSO(kind="lpid").tune(p, budget=80, seed=0)
    assert r.success and set(r.gains) == {"Kp", "Ki", "Kd"} and r.method.endswith("[L-PID]")
    r = GA(tune_shape=True).tune(TuningProblem.from_scenario("dc_speed"), budget=80, seed=0)
    assert {"n", "m", "p"} <= set(r.gains) and all(0.5 <= r.gains[k] <= 4.0 for k in "nmp")
