import numpy as np
import pytest

from src.controllers import LPID, NPID


def test_shaping_functions_known_values():
    assert NPID.alpha(0.0, 2) == 0.0
    assert NPID.alpha(1.0, 2) == pytest.approx(1 / np.sqrt(2))
    assert NPID.alpha(-1.0, 2) == pytest.approx(-1 / np.sqrt(2))
    assert NPID.alpha(100.0, 2) == pytest.approx(1.0, abs=1e-3)       # saturates
    assert NPID.alpha(0.01, 2) == pytest.approx(0.01, rel=1e-3)        # linear near 0
    assert NPID.beta(0.0, 2) == 1.0 and NPID.beta(1.0, 2) == 0.5
    assert NPID.gamma(0.0, 2) == 1.0 and NPID.gamma(1.0, 2) == 0.5
    assert NPID.alpha(2.0, 3) == pytest.approx(2.0 / 9.0 ** (1 / 3))   # general-exponent branch


def test_pid_terms_and_saturation():
    c = LPID(Kp=2.0, Ki=0.0, Kd=0.0, dt=0.01, u_min=-5, u_max=5)
    assert c.compute(1.0) == 2.0
    assert c.compute(100.0) == 5.0 and c.compute(-100.0) == -5.0


def test_lpid_anti_windup_bounds_integral():
    c = LPID(Kp=1.0, Ki=10.0, Kd=0.0, dt=0.01, u_min=-5, u_max=5)
    for _ in range(100):
        assert c.compute(50.0) == 5.0
    integral_after_100 = c._integral
    for _ in range(900):
        c.compute(50.0)
    assert c._integral == pytest.approx(integral_after_100)   # saturation: integral does not keep growing
    assert c.compute(-50.0) == -5.0                           # and reverses at once when the error flips


def test_npid_integral_is_suppressed_for_large_error():
    big, small = NPID(0, 1.0, 0, 0.01), NPID(0, 1.0, 0, 0.01)
    for _ in range(100):
        big.compute(10.0)
        small.compute(0.1)
    assert big._integral / 10.0 < small._integral / 0.1


def test_npid_derivative_is_suppressed_for_large_rate():
    c = NPID(0, 0, 1.0, 0.001)
    c.compute(0.0)
    assert abs(c.compute(0.001)) < 1.0      # de/dt = 1 -> gamma = 0.5, D = 0.5
    assert abs(c.compute(1.0)) < 5.0        # huge de/dt: linear D would be ~1000


def test_default_scales_are_paper_literal():
    a = NPID(10, 5, 0.1, 0.001)
    b = NPID(10, 5, 0.1, 0.001, e_scale=1.0, de_scale=1.0)
    for e in (0.0, 0.3, -2.0, 7.0):
        assert a.compute(e) == b.compute(e)


def test_scales_make_shaping_scale_free():
    n1 = NPID(10, 0, 0, 0.001, u_min=-1e9, u_max=1e9)
    n2 = NPID(10, 0, 0, 0.001, u_min=-1e9, u_max=1e9, e_scale=20.0)
    # P term at unit scale for e=1 equals P term at e_scale=20 for e=20, up to the scale factor
    assert n2.compute(20.0) == pytest.approx(20.0 * n1.compute(1.0))


def test_derivative_filter_smooths_noise():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 0.3, 500)
    big = dict(u_min=-1e9, u_max=1e9)
    raw, filt = LPID(0, 0, 1.0, 0.001, **big), LPID(0, 0, 1.0, 0.001, tau_d=0.01, **big)
    u_raw = [raw.compute(x) for x in e]
    u_filt = [filt.compute(x) for x in e]
    assert np.std(u_filt) < 0.3 * np.std(u_raw)


def test_reset_clears_state_including_filter():
    c = LPID(1, 1, 1, 0.01, tau_d=0.05)
    for e in (1, 2, 3):
        c.compute(e)
    c.reset()
    assert (c._integral, c._prev_error, c._d_filt) == (0.0, 0.0, 0.0)
