"""Validation of model matching on exact plants (analytic ideal controller, reference scaling, time-domain refinement)."""
import numpy as np
import pytest

from src.plant import DelayedPlant, TransferFunctionPlant
from src.tuners import NotApplicable, TuningProblem
from src.tuners.identify import ProcessModel
from src.tuners.modelmatching import (ModelMatching, fit_pid_to_ideal, open_loop_settling, reference_frequency_response,
                                      reference_parameters, reference_response)

K, TAU, THETA = 2.0, 5.0, 1.0


def fopdt_problem(theta=THETA, gain=K, dt=0.02, t_end=60.0):
    def factory():
        p = TransferFunctionPlant([gain], [TAU, 1.0], u_limits=(-50, 50))
        return DelayedPlant(p, theta) if theta else p
    return TuningProblem("fopdt-mm", factory, y_ref=1.0, dt=dt, t_end=t_end)


def model(kind="fopdt", K_=K, tau=TAU, theta=THETA):
    return ProcessModel(kind, K_, tau, theta, 0.0)


# ---- reference model -------------------------------------------------------------------------------
def test_open_loop_settling_and_reference_scaling():
    assert open_loop_settling(model()) == pytest.approx(4 * TAU + THETA)
    assert open_loop_settling(model("ipdt", 2.0, 0.1, 0.3)) == pytest.approx(10 * 0.4)
    r3 = reference_parameters(model(), speedup=3.0, zeta=0.9, order=2, dt=0.02)
    r6 = reference_parameters(model(), speedup=6.0, zeta=0.9, order=2, dt=0.02)
    assert r3["theta"] == THETA and r3["T_resp"] == pytest.approx((4 * TAU) / 3.0)
    assert r6["wn"] == pytest.approx(2 * r3["wn"])                       # twice the speed
    r1 = reference_parameters(model(), speedup=3.0, zeta=0.9, order=1, dt=0.02)
    assert r1["Tc"] == pytest.approx(r3["T_resp"] / 4)


def test_reference_response_has_the_plants_dead_time_and_unit_gain():
    p = fopdt_problem()
    ref = reference_parameters(model(), 3.0, 0.9, 2, p.dt)
    t, y = reference_response(p, ref, 2)
    assert np.max(np.abs(y[t < THETA - 0.05])) < 1e-9                    # nothing before the dead time
    assert y[-1] == pytest.approx(p.y_ref, rel=1e-3)
    zeta = 0.9
    assert 100 * (y.max() / p.y_ref - 1) < 0.5 * 100 * np.exp(-np.pi * zeta / np.sqrt(1 - zeta ** 2)) + 1.0   # ~0.2% overshoot


# ---- frequency-domain fit ---------------------------------------------------------------------------
def test_exactly_realisable_case_recovers_the_analytic_pi_controller():
    """G = K/(tau s + 1), M = 1/(Tc s + 1)  =>  C* = (tau s + 1)/(K Tc s) = Kp + Ki/s with Kp = tau/(K Tc), Ki = 1/(K Tc)."""
    Tc = 1.5
    m = model(theta=0.0)
    ref = dict(theta=0.0, Tc=Tc, T_resp=4 * Tc)
    gains, resid = fit_pid_to_ideal(m, ref, order=1, tau_d=0.01, dt=0.001)
    assert resid < 1e-3
    assert gains["Kp"] == pytest.approx(TAU / (K * Tc), rel=0.01)
    assert gains["Ki"] == pytest.approx(1.0 / (K * Tc), rel=0.01)
    assert gains["Kd"] == pytest.approx(0.0, abs=1e-3)


def test_ideal_controller_fit_is_worse_with_dead_time_but_still_stable_loop():
    p = fopdt_problem()
    m = model()
    ref = reference_parameters(m, 3.0, 0.9, 2, p.dt)
    _, resid_delay = fit_pid_to_ideal(m, ref, 2, p.tau_d, p.dt)
    ref0 = reference_parameters(model(theta=0.0), 3.0, 0.9, 2, p.dt)
    _, resid_nodelay = fit_pid_to_ideal(model(theta=0.0), ref0, 2, p.tau_d, p.dt)
    assert resid_delay > 0.05 and resid_delay > resid_nodelay * 0.5      # a PID cannot realise C* for a delayed plant exactly


def test_reference_frequency_response_limits():
    ref = dict(theta=0.0, wn=1.0, zeta=0.9, T_resp=4.0)
    assert abs(reference_frequency_response(ref, 2, np.array([1e-6]))[0]) == pytest.approx(1.0, rel=1e-4)
    assert abs(reference_frequency_response(ref, 2, np.array([100.0]))[0]) < 1e-3


# ---- tuner ---------------------------------------------------------------------------------------
def test_frequency_mode_closed_loop_is_stable_follows_the_reference_and_costs_few_experiments():
    p = fopdt_problem()
    r = ModelMatching("frequency").tune(p)
    assert r.success and r.evals <= 8 and r.meta["freq_residual"] < 0.5
    res = fopdt_problem().simulate(r.gains, "lpid")
    ref = reference_parameters(model(), 3.0, 0.9, 2, p.dt)
    t, ym = reference_response(p, ref, 2)
    err = np.trapezoid((res.omega - ym) ** 2, t) / (p.y_ref ** 2 * p.t_end)
    assert err < 0.05 and res.omega[-1] == pytest.approx(1.0, abs=0.02)


def test_time_mode_improves_matching_within_budget():
    pf, pt = fopdt_problem(), fopdt_problem()
    rf = ModelMatching("frequency").tune(pf)
    rt = ModelMatching("time", max_sims=60).tune(pt)
    assert rt.success and rt.evals <= 60 + 8
    p = fopdt_problem()
    ref = reference_parameters(model(), 3.0, 0.9, 2, p.dt)
    t, ym = reference_response(p, ref, 2)

    def match(g):
        res = fopdt_problem().simulate(g)
        return np.trapezoid((res.omega - ym) ** 2, t) / (p.y_ref ** 2 * p.t_end)

    assert match(rt.gains) <= match(rf.gains) * 1.05
    assert rt.history and rt.history[-1][1] <= rt.history[0][1]


def test_gains_scale_inversely_with_plant_gain_and_runs_are_deterministic():
    g1 = ModelMatching("frequency").tune(fopdt_problem(gain=1.0)).gains
    g2 = ModelMatching("frequency").tune(fopdt_problem(gain=2.0)).gains
    assert g2["Kp"] == pytest.approx(g1["Kp"] / 2.0, rel=0.1) and g2["Ki"] == pytest.approx(g1["Ki"] / 2.0, rel=0.1)
    assert ModelMatching("frequency").tune(fopdt_problem()).gains == ModelMatching("frequency").tune(fopdt_problem()).gains


def test_faster_reference_gives_larger_gains():
    slow = ModelMatching("frequency", speedup=2.0).tune(fopdt_problem()).gains["Kp"]
    fast = ModelMatching("frequency", speedup=6.0).tune(fopdt_problem()).gains["Kp"]
    assert fast > slow


def test_not_applicable_for_unstable_plant():
    p = TuningProblem("unstable", lambda: TransferFunctionPlant([1.0], [1.0, -1.0], u_limits=(-5, 5)),
                      y_ref=1.0, dt=0.01, t_end=10.0)
    r = ModelMatching("frequency").tune(p)
    assert not r.success and "not applicable" in r.message


def test_works_on_integrating_plant():
    p = TuningProblem("ipdt", lambda: DelayedPlant(TransferFunctionPlant([2.0], [0.2, 1.0, 0.0], u_limits=(-20, 20)), 0.2),
                      y_ref=1.0, dt=0.005, t_end=8.0)
    r = ModelMatching("frequency").tune(p)
    assert r.success and r.meta["model"] == "ipdt"
    res = p.simulate(r.gains, "lpid", count=False)
    assert res.omega[-1] == pytest.approx(1.0, abs=0.03)
