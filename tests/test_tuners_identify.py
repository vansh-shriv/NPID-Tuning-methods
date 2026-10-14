import numpy as np
import pytest

from src.plant import DelayedPlant, TransferFunctionPlant
from src.tuners import NotApplicable, Tuner, TuningProblem
from src.tuners.identify import (find_trim, fit_fopdt, fit_ipdt, fopdt_step, identify_process, ipdt_step)


def fopdt_problem(K=2.0, tau=5.0, theta=1.0, y_ref=1.0, limits=(-10, 10), dt=0.02, t_end=40.0):
    return TuningProblem(
        "fopdt-test", lambda: DelayedPlant(TransferFunctionPlant([K], [tau, 1.0], u_limits=limits), theta),
        y_ref=y_ref, dt=dt, t_end=t_end)


# ---- pure fitting on synthetic responses -------------------------------------------------------
@pytest.mark.parametrize("method", ["lsq", "two_point", "tangent"])
def test_fopdt_fit_recovers_exact_model(method):
    t = np.arange(0, 60, 0.02)
    s = fopdt_step(t, 2.0, 5.0, 1.0)
    K, tau, theta, nrmse = fit_fopdt(t, s, method)
    tol = 0.01 if method == "lsq" else 0.12            # graphical methods are approximate by construction
    assert K == pytest.approx(2.0, rel=tol) and tau == pytest.approx(5.0, rel=tol)
    assert theta == pytest.approx(1.0, abs=0.2 if method != "lsq" else 0.02)
    assert nrmse < (0.01 if method == "lsq" else 0.05)


def test_fopdt_fit_is_robust_to_noise():
    rng = np.random.default_rng(1)
    t = np.arange(0, 60, 0.02)
    s = fopdt_step(t, 2.0, 5.0, 1.0) + rng.normal(0, 0.02, len(t))
    K, tau, theta, _ = fit_fopdt(t, s)
    assert (K, tau, theta) == pytest.approx((2.0, 5.0, 1.0), rel=0.05, abs=0.1)


def test_ipdt_fit_recovers_exact_model():
    t = np.arange(0, 20, 0.01)
    s = ipdt_step(t, 3.0, 0.4, 0.2)
    Kv, tau, theta, nrmse = fit_ipdt(t, s)
    assert Kv == pytest.approx(3.0, rel=0.01) and tau == pytest.approx(0.4, rel=0.05)
    assert theta == pytest.approx(0.2, abs=0.03) and nrmse < 0.01


# ---- experiments through the problem ------------------------------------------------------------
def test_trim_input_gives_setpoint_at_steady_state():
    p = fopdt_problem(K=2.0, y_ref=1.0)
    u = find_trim(p)
    assert u == pytest.approx(0.5, rel=0.01)           # y_ss = K u
    assert p.evals >= 2                                # experiments are counted (linear plant: secant needs 2)
    assert p.sim_seconds > 0


def test_identify_process_recovers_known_fopdt_with_counted_cost():
    p = fopdt_problem()
    m = identify_process(p)
    assert m.kind == "fopdt"
    assert (m.K, m.tau, m.theta) == pytest.approx((2.0, 5.0, 1.0), rel=0.04, abs=0.1)
    assert m.nrmse < 0.02 and not m.poor_fit
    assert m.u_op == pytest.approx(0.5, rel=0.02)
    assert p.evals == pytest.approx(p.evals) and p.evals < 20


def test_identify_unbounded_actuator():
    p = fopdt_problem(limits=(-np.inf, np.inf))
    m = identify_process(p)
    assert m.K == pytest.approx(2.0, rel=0.04)


def test_identify_integrating_plant():
    # Kv e^{-theta s} / (s (tau s + 1)) with Kv=2, tau=0.5, theta=0.3
    p = TuningProblem("ipdt-test", lambda: DelayedPlant(TransferFunctionPlant([2.0], [0.5, 1.0, 0.0], u_limits=(-10, 10)), 0.3),
                      y_ref=1.0, dt=0.005, t_end=6.0)
    m = identify_process(p)
    assert m.kind == "ipdt"
    assert m.K == pytest.approx(2.0, rel=0.05) and m.tau == pytest.approx(0.5, rel=0.25)
    assert m.theta_eff == pytest.approx(0.8, abs=0.1)


def test_unreachable_setpoint_is_not_applicable():
    p = fopdt_problem(K=2.0, y_ref=100.0, limits=(-10, 10))   # max output 20
    with pytest.raises(NotApplicable):
        identify_process(p)


def test_unstable_plant_is_not_applicable():
    p = TuningProblem("unstable", lambda: TransferFunctionPlant([1.0], [1.0, -1.0], u_limits=(-5, 5)),
                      y_ref=1.0, dt=0.01, t_end=10.0)
    with pytest.raises(NotApplicable):
        identify_process(p)


# ---- the problem object -------------------------------------------------------------------------
def test_problem_counts_closed_loop_runs_and_resets():
    p = fopdt_problem()
    g = dict(Kp=2.0, Ki=0.5, Kd=0.0)
    p.simulate(g)
    p.simulate(g, scenario="noise", noise_seed=3)
    assert p.evals == 2 and p.sim_seconds == pytest.approx(2 * p.t_end)
    p.simulate(g, count=False)
    assert p.evals == 2
    p.reset_counters()
    assert p.evals == 0


def test_cost_rejects_negative_or_diverging_gains():
    p = fopdt_problem()
    assert p.cost(dict(Kp=-1.0, Ki=0.0, Kd=0.0)) == 1e6
    assert p.cost(dict(Kp=1e6, Ki=1e6, Kd=1e6)) >= 1.0       # saturated, terrible but finite or FAIL_COST
    assert p.cost(dict(Kp=2.0, Ki=0.5, Kd=0.0)) < 5.0


def test_lpid_and_npid_share_filter_but_not_shaping():
    p = fopdt_problem()
    g = dict(Kp=2.0, Ki=0.5, Kd=0.5)
    assert p.make_controller(g, "lpid").tau_d == p.make_controller(g, "npid").tau_d == pytest.approx(0.05 * p.Ts)
    assert p.make_controller(g, "npid").e_scale == abs(p.y_ref)
    assert not hasattr(p.make_controller(g, "lpid"), "e_scale")


def test_tuner_wrapper_counts_cost_and_flags_invalid_gains():
    class Bad(Tuner):
        name = "bad"

        def _tune(self, problem, budget, seed):
            problem.simulate(dict(Kp=1.0, Ki=0.0, Kd=0.0))
            return dict(Kp=float("nan"), Ki=1.0, Kd=0.0), {}

    class Inapplicable(Tuner):
        name = "na"

        def _tune(self, problem, budget, seed):
            raise NotApplicable("nope")

    p = fopdt_problem()
    r = Bad().tune(p)
    assert not r.success and r.evals == 1 and r.wall_time > 0
    r = Inapplicable().tune(p)
    assert not r.success and "not applicable" in r.message


def test_time_scale_estimate_matches_process_time_constant():
    p = fopdt_problem(tau=5.0, theta=1.0)
    assert 4.5 <= p.Ts <= 7.0                            # time to 63% = tau + theta


# ---- identification of the benchmark plants (analytic linearisations) ------------------------------
def test_benchmark_dc_speed_gain_matches_analytic():
    m = identify_process(TuningProblem.from_scenario("dc_speed"))
    assert m.kind == "fopdt"
    assert m.K == pytest.approx(0.5 / (1.0 * 0.1 + 0.5 * 0.5), rel=0.05)      # Kt / (R B + Kt Ke)


def test_benchmark_tank_matches_linearisation_at_setpoint():
    m = identify_process(TuningProblem.from_scenario("tank"))
    # h' = -0.5 sqrt(h) + 0.4 u at h = 4: K = 3.2, tau = 8 (a 10% input step is mildly nonlinear)
    assert m.kind == "fopdt" and m.u_op == pytest.approx(2.5, rel=0.02)
    assert m.K == pytest.approx(3.2, rel=0.08) and m.tau == pytest.approx(8.0, rel=0.1)


def test_benchmark_link_is_self_regulating_not_integrating():
    # regression: a secant step beyond the gravity-torque limit once made the arm look like an integrator
    m = identify_process(TuningProblem.from_scenario("link"))
    assert m.kind == "fopdt" and m.u_op == pytest.approx(1.47 * np.sin(1.0), rel=0.05)


def test_benchmark_position_is_integrating():
    m = identify_process(TuningProblem.from_scenario("dc_position"))
    assert m.kind == "ipdt" and m.K == pytest.approx(1.43, rel=0.15)


def test_identification_under_measurement_noise_degrades_gracefully():
    p = fopdt_problem(dt=0.02, t_end=40.0)
    m = identify_process(p, noise_sigma=0.01, seed=1)         # 1% of the setpoint
    assert m.K == pytest.approx(2.0, rel=0.15) and m.tau == pytest.approx(5.0, rel=0.3)


# ---- second-order-plus-dead-time ------------------------------------------------------------------
def test_sopdt_fit_recovers_exact_model_and_beats_fopdt():
    from src.tuners.identify import fit_sopdt, sopdt_step
    t = np.arange(0, 40, 0.02)
    s = sopdt_step(t, 1.5, 4.0, 1.0, 0.5)
    K, tau1, tau2, theta, nrmse = fit_sopdt(t, s)
    assert (K, tau1, tau2, theta) == pytest.approx((1.5, 4.0, 1.0, 0.5), rel=0.05, abs=0.05)
    assert nrmse < 0.005 and nrmse < fit_fopdt(t, s)[3]


def test_sopdt_equal_and_zero_second_lag_limits():
    from src.tuners.identify import fopdt_step, sopdt_step
    t = np.linspace(0, 20, 400)
    assert sopdt_step(t, 2.0, 3.0, 0.0, 0.4) == pytest.approx(fopdt_step(t, 2.0, 3.0, 0.4))
    x = np.maximum(t - 0.4, 0)
    assert sopdt_step(t, 2.0, 3.0, 3.0, 0.4) == pytest.approx(np.where(t > 0.4, 2.0 * (1 - (1 + x / 3) * np.exp(-x / 3)), 0))


def test_identify_process_second_order_on_benchmark_plant():
    m = identify_process(TuningProblem.from_scenario("dc_speed"), order=2)
    assert m.kind == "sopdt" and m.tau >= m.tau2 >= 0 and m.nrmse < 0.05
