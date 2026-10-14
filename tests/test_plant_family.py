import numpy as np
import pytest

from src.plant_family import (ALL_TYPES, OOD_TYPES, TRAIN_TYPES, LTIPlant, PlantSpec, build_plant, problem_from_spec,
                              sample_spec, sample_specs, spec_from_dict, spec_to_dict)
from src.plant import TransferFunctionPlant


def run_open_loop(plant, u, t_end, dt):
    x = plant.initial_state()
    ys = []
    for k in range(int(t_end / dt)):
        ys.append(plant.output(x))
        x = plant.step(k * dt, x, u, 0.0, dt)
    return np.array(ys)


def test_lti_plant_matches_generic_rk4_plant():
    num, den = [2.0], [3.0, 4.0, 1.0]
    fast = LTIPlant(num, den)
    ref = TransferFunctionPlant(num, den)
    a, b = run_open_loop(fast, 1.0, 50.0, 0.01), run_open_loop(ref, 1.0, 50.0, 0.01)
    assert np.max(np.abs(a - b)) < 1e-6                                   # exact vs 4th-order RK4 at small dt
    assert a[-1] == pytest.approx(2.0, rel=1e-3)


def test_lti_plant_is_faster_than_generic_plant():
    import time
    fast, slow = LTIPlant([1.0], [2.0, 3.0, 1.0]), TransferFunctionPlant([1.0], [2.0, 3.0, 1.0])
    t0 = time.perf_counter(); run_open_loop(fast, 1.0, 20.0, 0.005); t_fast = time.perf_counter() - t0
    t0 = time.perf_counter(); run_open_loop(slow, 1.0, 20.0, 0.005); t_slow = time.perf_counter() - t0
    assert t_fast < t_slow / 2


def test_input_power_nonlinearity_and_disturbance_enter_as_specified():
    p = LTIPlant([1.0], [1.0, 1.0], input_power=0.5)
    y = run_open_loop(p, 4.0, 30.0, 0.01)
    assert y[-1] == pytest.approx(2.0, rel=1e-3)                          # sqrt(4)
    x = p.initial_state()
    x1 = p.step(0.0, x, 0.0, 1.0, 0.01)                                   # disturbance adds to the (nonlinear) input
    assert x1[0] > 0


@pytest.mark.parametrize("ptype", ALL_TYPES)
def test_every_type_is_well_posed_reachable_and_resolved(ptype):
    for i in range(12):
        spec = sample_spec(np.random.default_rng([7, i]), ptype, i)
        plant = build_plant(spec)
        dt, t_end = spec.sample_time(), spec.horizon()
        assert t_end / dt <= 20000 and dt <= spec.fastest_time() / 10.0 + 1e-12      # sampling resolves the dynamics
        assert spec.u_max() > abs(spec.u_ss()) or spec.integrating                    # actuator has headroom
        if not spec.integrating:
            y = run_open_loop(plant, spec.u_ss(), t_end, dt)
            assert y[-1] == pytest.approx(spec.y_ref, rel=0.02), (ptype, i)           # steady input reaches the setpoint
        else:
            y = run_open_loop(plant, spec.u_max(), t_end, dt)
            assert y[-1] > spec.y_ref                                                 # the integrator can move far enough


def test_dead_time_and_underdamped_ringing_are_present():
    spec = PlantSpec("fopdt", K=2.0, tau1=2.0, theta=1.0, y_ref=1.0)
    y = run_open_loop(build_plant(spec), 0.5, 10.0, 0.01)
    assert np.max(np.abs(y[:99])) < 1e-12 and y[300] > 0.1
    under = PlantSpec("underdamped", K=1.0, tau1=1.0, wn=2.0, zeta=0.3, y_ref=1.0)
    y = run_open_loop(build_plant(under), 1.0, 15.0, 0.005)
    assert y.max() > 1.3 * y[-1]                                          # ~37% overshoot at zeta = 0.3


def test_specs_are_deterministic_cycle_types_and_roundtrip():
    a, b = sample_specs(10, seed=3), sample_specs(10, seed=3)
    assert [spec_to_dict(s) for s in a] == [spec_to_dict(s) for s in b]
    assert [s.ptype for s in a[:8]] == list(TRAIN_TYPES) * 2
    assert sample_specs(5, seed=3, offset=5)[0].ident == 5
    d = spec_to_dict(a[0])
    assert spec_from_dict(d) == a[0].__class__(**{**d})
    assert set(OOD_TYPES).isdisjoint(TRAIN_TYPES)


def test_problem_from_spec_closed_loop_runs_fast_and_is_controllable():
    import time
    spec = sample_spec(np.random.default_rng(1), "fopdt")
    p = problem_from_spec(spec)
    assert p.u_max == pytest.approx(spec.u_max())
    # a sensible PI (SIMC-like) drives the output to the setpoint
    K, tau, th = spec.K, spec.tau1, spec.theta
    tau_c = max(th, 0.1 * tau)
    g = dict(Kp=tau / (K * (tau_c + th)), Ki=tau / (K * (tau_c + th)) / min(tau, 4 * (tau_c + th)), Kd=0.0)
    t0 = time.perf_counter()
    res = p.simulate(g)
    assert time.perf_counter() - t0 < 1.0
    assert res.omega[-1] == pytest.approx(spec.y_ref, rel=0.05)


def test_scale_invariance_of_the_family_construction():
    """Stretching all times by 3 and scaling the sampling and horizon accordingly keeps the dimensionless loop identical."""
    s1 = PlantSpec("sopdt", K=2.0, tau1=2.0, tau2=0.5, theta=0.4, y_ref=3.0)
    s2 = PlantSpec("sopdt", K=2.0, tau1=6.0, tau2=1.5, theta=1.2, y_ref=3.0)
    assert s2.horizon() == pytest.approx(3 * s1.horizon()) and s2.sample_time() == pytest.approx(3 * s1.sample_time())
    g1 = dict(Kp=0.8, Ki=0.4, Kd=0.1)
    g2 = dict(Kp=0.8, Ki=0.4 / 3, Kd=0.1 * 3)
    r1, r2 = problem_from_spec(s1).simulate(g1), problem_from_spec(s2).simulate(g2)
    assert np.max(np.abs(r1.omega - r2.omega)) < 5e-2 * s1.y_ref          # same response up to the filter/shaping scale
