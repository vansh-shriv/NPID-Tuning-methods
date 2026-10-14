import numpy as np
import pytest

from src.controllers import LPID, NPID
from src.metrics import compute_metrics
from src.plant import DCMotorParams, TransferFunctionPlant
from src.simulate import run_simulation


def test_metrics_on_first_order_response():
    t = np.linspace(0, 10, 10001)
    tau = 1.0
    y = 1 - np.exp(-t / tau)
    m = compute_metrics(t, 1.0, y)
    assert m["rise_time"] == pytest.approx(tau * np.log(9), rel=1e-2)       # 10%-90% = 2.197 tau
    assert m["overshoot_pct"] == pytest.approx(0.0, abs=1e-2)   # "final" is the mean of the last 100 samples
    assert m["settling_time"] == pytest.approx(tau * np.log(50), rel=1e-2)  # 2% band = 3.91 tau
    assert m["IAE"] == pytest.approx(tau, rel=1e-2)
    assert m["ISE"] == pytest.approx(tau / 2, rel=1e-2)
    assert m["ITAE"] == pytest.approx(tau ** 2, rel=2e-2)


def test_metrics_overshoot():
    t = np.linspace(0, 1, 1001)
    y = np.ones_like(t)
    y[100:200] = 1.25
    assert compute_metrics(t, 1.0, y)["overshoot_pct"] == pytest.approx(25.0, rel=1e-6)


# Pinned results of the original (pre-refactor) code: guards the fast RK4 path.
@pytest.mark.parametrize("ctrl_cls, itae, overshoot", [(LPID, 0.1631, 0.2188), (NPID, 1.4052, 8.5964)])
def test_dc_motor_closed_loop_regression(ctrl_cls, itae, overshoot):
    p = DCMotorParams()
    r = run_simulation(ctrl_cls(139.2, 6747.3, 0.718, 0.001), p, 20.0, t_end=3.0)
    m = compute_metrics(r.time, 20.0, r.omega)
    assert m["ITAE"] == pytest.approx(itae, abs=1e-3)
    assert m["overshoot_pct"] == pytest.approx(overshoot, abs=1e-3)


def test_closed_loop_on_generic_plant_tracks_reference():
    plant = TransferFunctionPlant([1.0], [1.0, 3.0, 2.0], u_limits=(-50, 50))
    r = run_simulation(LPID(5.0, 8.0, 0.0, 0.001, -50, 50), plant, 1.0, t_end=10.0)
    assert r.omega[-1] == pytest.approx(1.0, abs=1e-2)
    assert np.max(np.abs(r.voltage)) <= 50.0


def test_noise_is_reproducible_and_does_not_enter_logged_error():
    p = DCMotorParams()
    kw = dict(t_end=0.5, noise_sigma=0.3)
    a = run_simulation(LPID(10, 50, 0.1, 0.001), p, 20.0, rng=np.random.default_rng(3), **kw)
    b = run_simulation(LPID(10, 50, 0.1, 0.001), p, 20.0, rng=np.random.default_rng(3), **kw)
    assert np.array_equal(a.omega, b.omega)
    assert np.allclose(a.error, 20.0 - a.omega)          # logged error is the true error


def test_load_disturbance_reduces_speed_under_p_control():
    p = DCMotorParams()
    base = run_simulation(LPID(20, 0, 0, 0.001), p, 20.0, t_end=2.0)
    loaded = run_simulation(LPID(20, 0, 0, 0.001), p, 20.0, t_end=2.0, T_load_func=lambda t: 0.05)
    assert loaded.omega[-1] < base.omega[-1]
