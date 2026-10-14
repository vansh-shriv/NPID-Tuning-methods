import numpy as np
import pandas as pd
import pytest

from src.tuners import NotApplicable, Tuner, TuningProblem
from src.tuners.benchmark import annotate_failures, evaluate_gains, run_benchmark
from src.tuners.pso_ref import PSOReference


class Fixed(Tuner):
    """Returns given gains without experimenting (deterministic)."""
    family = "test"

    def __init__(self, name, gains):
        self.name, self.gains = name, gains

    def _tune(self, problem, budget, seed):
        return dict(self.gains), {"note": 1.0}


class Refuse(Tuner):
    name, family = "refuse", "test"

    def _tune(self, problem, budget, seed):
        raise NotApplicable("by design")


def test_evaluate_gains_is_not_counted_and_has_all_scenarios():
    p = TuningProblem.from_scenario("dc_speed")
    ev = evaluate_gains(p, dict(Kp=100.0, Ki=2000.0, Kd=1.0))
    assert p.evals == 0
    assert set(ev) == {"step", "load", "noise", "noise_std"}
    assert ev["step"]["ITAE"] > 0 and ev["noise"]["ITAE"] >= ev["step"]["ITAE"]
    assert ev["noise_std"]["ITAE"] >= 0


def test_evaluate_gains_flags_divergence():
    p = TuningProblem.from_scenario("fopdt")
    ev = evaluate_gains(p, dict(Kp=1e6, Ki=1e6, Kd=1e6))
    assert ev["step"]["ITAE"] >= 1.0   # saturated actuator: poor but may stay finite; never crashes


def test_run_benchmark_rows_and_failure_annotation():
    df = run_benchmark([Fixed("good", dict(Kp=100.0, Ki=2000.0, Kd=1.0)),
                        Fixed("bad", dict(Kp=0.01, Ki=0.0, Kd=0.0)), Refuse()], ["dc_speed"])
    assert set(df.method) == {"good", "bad", "refuse"}
    assert set(df[df.method == "good"].kind) == {"npid", "lpid"}
    assert not df[(df.method == "good") & (df.scenario == "step")].failed.any()
    assert df[(df.method == "bad") & (df.scenario == "step") & (df.kind == "npid")].failed.all()   # >100x worse
    r = df[df.method == "refuse"]
    assert (~r.success).all() and r.failed.all() and "not applicable" in r.message.iloc[0]


def test_annotate_failures_threshold():
    df = pd.DataFrame(dict(plant="p", scenario="step", kind="npid", success=True, diverged=False,
                           ITAE=[1.0, 50.0, 500.0], method=list("abc")))
    assert list(annotate_failures(df).failed) == [False, False, True]


def test_pso_reference_respects_budget_and_seed():
    p = TuningProblem.from_scenario("dc_speed")
    r1 = PSOReference().tune(p, budget=96, seed=1)
    assert r1.evals == 96 and r1.success and r1.sim_seconds == pytest.approx(96 * p.t_end)
    r2 = PSOReference().tune(TuningProblem.from_scenario("dc_speed"), budget=96, seed=1)
    assert r1.gains == r2.gains                                      # seeded -> reproducible
    r3 = PSOReference().tune(TuningProblem.from_scenario("dc_speed"), budget=96, seed=2)
    assert r1.gains != r3.gains
    assert r1.history[-1][0] == 96 and r1.history[0][1] >= r1.history[-1][1]   # best-so-far is non-increasing
    lb, ub = p.bounds
    for k, a, b in zip(("Kp", "Ki", "Kd"), lb, ub):
        assert a <= r1.gains[k] <= b * 1.0000001
