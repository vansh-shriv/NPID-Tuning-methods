"""Validation of the adaptive tuning machinery (M6): switching plant, online estimator, rate limit, scenario loop."""
import numpy as np
import pytest

from src.plant_family import PlantSpec, build_plant, problem_from_spec
from src.tuners.adaptive import (AdaptiveConfig, AdaptivePolicy, FrozenPolicy, OnlineEstimator, REF_LEVELS, ScheduledPolicy,
                                 change_feasible, change_scenario, rate_limit, run_scenario, sampling_steps, scaled_spec,
                                 window_metrics)


def fopdt_spec(K=2.0, tau=5.0, theta=1.0, y_ref=1.0, r_auth=6.0):
    return PlantSpec(ptype="fopdt", K=K, tau1=tau, theta=theta, y_ref=y_ref, r_auth=r_auth, noise_rel=0.0)


# ---- plant with a change ------------------------------------------------------------------------------
def test_scaled_spec_and_feasibility():
    s = fopdt_spec()
    assert scaled_spec(s, ("gain", 0.5)).K == pytest.approx(1.0)
    a = scaled_spec(s, ("lag", 3.0))
    assert (a.tau1, a.K) == (pytest.approx(15.0), 2.0)
    assert scaled_spec(s, ("none", 1.0)) == s and s.K == 2.0                          # the original is untouched
    assert change_feasible(fopdt_spec(r_auth=6.0), ("gain", 0.4)) and not change_feasible(fopdt_spec(r_auth=2.0), ("gain", 0.4))
    with pytest.raises(ValueError):
        scaled_spec(s, ("bogus", 2.0))


def _open_loop(factory, u, n, dt):
    p = factory()
    x = p.initial_state()
    y = []
    for k in range(n):
        y.append(p.output(x))
        x = p.step(k * dt, x, u, 0.0, dt)
    return np.array(y)


@pytest.mark.parametrize("change", [("gain", 0.5), ("lag", 3.0)])
def test_switching_plant_is_output_continuous_and_reaches_the_new_steady_state(change):
    spec, dt = fopdt_spec(theta=0.0), 0.02
    factory, after = change_scenario(spec, change, t_switch=40.0)
    y = _open_loop(factory, 0.5, int(400 / dt), dt)
    k = int(40.0 / dt)
    assert abs(y[k + 1] - y[k - 1]) < 5e-3 * abs(y[k - 1])                         # no jump at the switch (settled output)
    assert y[-1] == pytest.approx(after.K * 0.5, rel=1e-3)                          # new DC gain
    # before the switch it is the original plant
    ref = _open_loop(lambda: build_plant(spec), 0.5, k, dt)
    assert np.allclose(y[:k], ref, atol=1e-9)


def test_switching_plant_resets_between_runs():
    spec, dt = fopdt_spec(theta=0.0), 0.05
    factory, _ = change_scenario(spec, ("gain", 2.0), t_switch=10.0)
    plant = factory()
    runs = []
    for _ in range(2):
        x = plant.initial_state()
        for k in range(400):
            x = plant.step(k * dt, x, 0.5, 0.0, dt)
        runs.append(plant.output(x))
    assert runs[0] == pytest.approx(runs[1])


def test_delayed_switching_plant_keeps_the_dead_time():
    spec, dt = fopdt_spec(theta=2.0), 0.05
    factory, _ = change_scenario(spec, ("none", 1.0), 1e9)
    y = _open_loop(factory, 1.0, int(10 / dt), dt)
    assert np.all(y[: int(2.0 / dt)] == 0.0) and y[int(2.5 / dt)] > 0


# ---- estimator -----------------------------------------------------------------------------------------
def _identify_closed_form(K=2.0, tau=5.0, theta=1.0, noise=0.0, seed=0):
    """Drive a FOPDT with a PRBS-like input sampled at Delta and let the estimator identify it."""
    from src.plant import DelayedPlant
    from src.plant_family import LTIPlant
    dt, delta_steps = 0.05, 20                                                       # Delta = 1 s = theta
    plant = DelayedPlant(LTIPlant([K], [tau, 1.0]), theta)
    x = plant.initial_state()
    est = OnlineEstimator(delta_steps * dt, integrating=False, lam=1.0, p_reset=1e3)
    rng = np.random.default_rng(seed)
    u = 0.0
    for k in range(int(300 / dt)):
        y = plant.output(x)
        if k % delta_steps == 0:
            est.push(y + rng.normal(0, noise))
            est.update()
            if (k // delta_steps) % 15 == 0:
                u = rng.choice([-1.0, 1.0])
            est.push_input(u)
        x = plant.step(k * dt, x, u, 0.0, dt)
    return est


def test_estimator_recovers_a_fopdt_from_excited_data():
    m = _identify_closed_form().process_model()
    assert m is not None and m.kind == "fopdt"
    assert m.K == pytest.approx(2.0, rel=0.03) and m.tau == pytest.approx(5.0, rel=0.1) and m.theta == pytest.approx(1.0, abs=0.35)


def test_estimator_is_accurate_under_measurement_noise_in_level_form():
    """Regression for the differenced-form noise bias (K was off by up to 4x): levels keep K within a few percent."""
    m = _identify_closed_form(noise=0.02).process_model()
    assert m is not None and m.K == pytest.approx(2.0, rel=0.06)


def test_estimator_declines_an_unstable_or_empty_model():
    est = OnlineEstimator(1.0, False)
    assert est.process_model() is None                                               # theta = 0: no gain
    est.rls.theta = np.array([-2.5, 1.6, 0.1, 0.1])                                  # roots outside the unit circle
    assert est.process_model() is None


# ---- policies and loop ----------------------------------------------------------------------------------
def test_rate_limit_bounds_the_step_in_decades():
    cur = dict(Kp=1.0, Ki=1.0, Kd=1.0)
    out = rate_limit(cur, dict(Kp=100.0, Ki=0.001, Kd=1.1), 0.15)
    assert out["Kp"] == pytest.approx(10 ** 0.15) and out["Ki"] == pytest.approx(10 ** -0.15) and out["Kd"] == pytest.approx(1.1)


def test_window_metrics_on_an_exact_signal():
    t = np.arange(0, 100, 0.1)
    r = np.where(t < 50, 1.0, 0.8)
    y = np.where(t < 50, 1.0, 0.8) * (1 - np.exp(-((t % 50) / 5.0)))
    y = np.where(t >= 50, 1.0 + (0.8 - 1.0) * (1 - np.exp(-((t - 50) / 5.0))), y)
    m0 = window_metrics(t, y, r, 50.0, 0)
    assert m0["overshoot_pct"] == pytest.approx(0.0, abs=1e-9)
    assert m0["iae"] == pytest.approx(5.0, rel=0.02) and m0["itae"] == pytest.approx(25.0, rel=0.03)   # int e^{-t/5} = 5, int t e^{-t/5} = 25


def _loop(policy, change=("none", 1.0), noise=False):
    spec = fopdt_spec(K=1.5, tau=3.0, theta=0.4, y_ref=2.0, r_auth=8.0)
    p = problem_from_spec(spec, noise_sigma=0.0 if not noise else 0.02)
    kp = 1.0 / 1.5
    g0 = dict(Kp=kp * 1.5, Ki=kp * 1.5 / 3.0, Kd=0.0)
    m = sampling_steps(3.4, 0.4, p.dt, 10.0)
    return run_scenario(spec, change, policy, g0, m, noise=noise, kind0="fopdt", T_n0=3.4, problem=p), p


def test_loop_frozen_tracks_all_windows_and_is_deterministic():
    r1, p = _loop(FrozenPolicy())
    r2, _ = _loop(FrozenPolicy())
    assert not r1.failed and len(r1.windows) == len(REF_LEVELS) and np.allclose(r1.y, r2.y)
    for w in range(len(REF_LEVELS)):
        assert abs(r1.y[int((w + 1) * len(r1.y) / 5) - 1] - REF_LEVELS[w] * p.y_ref) < 0.03 * p.y_ref
    assert len(r1.gain_trace) == 1


def test_scheduled_policy_switches_exactly_once_at_its_time():
    spec = fopdt_spec(K=1.5, tau=3.0, theta=0.4, y_ref=2.0, r_auth=8.0)
    g_new = dict(Kp=2.0, Ki=1.0, Kd=0.1)
    r, p = _loop(ScheduledPolicy([(spec.horizon() * 2.0, g_new)]))
    assert len(r.gain_trace) == 2 and r.gain_trace[1][1] == g_new
    assert r.gain_trace[1][0] >= spec.horizon() * 2.0 - 1e-9


def test_adaptive_policy_updates_only_in_excitation_windows_and_respects_limits():
    from src.tuners.ml import ConstantModel
    spec = fopdt_spec(K=1.5, tau=3.0, theta=0.4, y_ref=2.0, r_auth=8.0)
    model = ConstantModel().fit(np.zeros((3, 8)), np.array([[0.4, 0.1, -0.5]] * 3))      # always predicts the same z
    pol = AdaptivePolicy(AdaptiveConfig("ml"), model)
    r, p = _loop(pol)
    H = spec.horizon()
    assert len(pol.trace) > 5
    lb, ub = p.bounds
    for t, g, target in pol.trace:
        # inside a window right after a reference step (window length = 4 T_n at least)
        assert (t % H) <= max(4.0 * 3.4, 25 * 0.34) + 1e-6
        assert all(lo <= g[k] <= hi for k, lo, hi in zip(("Kp", "Ki", "Kd"), lb, ub))
    for (_, g0, _), (_, g1, _) in zip(pol.trace[:-1], pol.trace[1:]):
        for k in ("Kp", "Ki", "Kd"):
            assert abs(np.log10(g1[k] / g0[k])) <= 0.15 + 1e-9                          # rate limit


def test_adaptive_pole_placement_rescales_the_gain_after_a_plant_gain_increase():
    """Self-tuning must move Kp in proportion to 1/K: after a 3x gain increase it ends near a third of the no-change gain."""
    r0, _ = _loop(AdaptivePolicy(AdaptiveConfig("pp")), ("none", 1.0))
    r3, _ = _loop(AdaptivePolicy(AdaptiveConfig("pp")), ("gain", 3.0))
    assert not r3.failed
    ratio = r3.gain_trace[-1][1]["Kp"] / r0.gain_trace[-1][1]["Kp"]
    assert 0.2 < ratio < 0.5
    # while frozen gains keep their value
    rf, _ = _loop(FrozenPolicy(), ("gain", 3.0))
    assert len(rf.gain_trace) == 1
