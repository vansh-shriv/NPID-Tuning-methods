import math

import pytest

from src.controllers import LPID
from src.plant import DCMotorParams, TransferFunctionPlant
from src.simulate import run_simulation
from src.tuning_generic import make_controller, step_itae, tune_pso
from src.tuning_relay import relay_autotune
from src.tuning_zn import (detect_oscillation_type, find_critical_gain,
                           oscillation_growth_ratio, ziegler_nichols_gains)

# 1/(s+1)^4 has an exact ultimate point: Kcr = 4, Tcr = 2*pi.
QUARTIC = TransferFunctionPlant([1.0], [1, 4, 6, 4, 1])


def test_zn_formulas():
    assert ziegler_nichols_gains(10.0, 2.0) == pytest.approx((6.0, 6.0, 1.5))


def test_zn_detector_finds_analytic_ultimate_point():
    Kcr, Tcr = find_critical_gain(QUARTIC, 1.0, t_end=40, kp_start=0.5, kp_max=100,
                                  verbose=False, mode="linear")
    assert Kcr == pytest.approx(4.0, rel=0.01)
    assert Tcr == pytest.approx(2 * math.pi, rel=0.01)


def test_growth_ratio_classification():
    for kp, expected in [(1.0, "damped"), (8.0, "growing")]:
        r = run_simulation(LPID(kp, 0, 0, 0.001, -1e9, 1e9), QUARTIC, 1.0, t_end=60.0)
        assert detect_oscillation_type(r.omega, r.time)[0] == expected
    ratio, _, _ = oscillation_growth_ratio([0, 1, 2, 3], [0, 1, 2, 3])
    assert ratio is None                                      # monotone response: no oscillation


def test_dc_motor_has_no_linear_ultimate_gain():
    with pytest.raises(RuntimeError):
        find_critical_gain(DCMotorParams(), 20.0, verbose=False, mode="linear", kp_max=300)


def test_dc_motor_limit_cycle_fallback_close_to_legacy_value():
    with pytest.warns(UserWarning):
        Kcr, Tcr = find_critical_gain(DCMotorParams(), 20.0, verbose=False)
    assert 200 < Kcr < 260 and 0.03 < Tcr < 0.06


def test_relay_matches_ultimate_point_within_describing_function_error():
    Ku, Tu = relay_autotune(QUARTIC, 1.0, d=1.0, bias=1.0, t_end=120.0)
    assert Ku == pytest.approx(4.0, rel=0.08)
    assert Tu == pytest.approx(2 * math.pi, rel=0.05)


def test_step_itae_penalizes_negative_gains():
    p = DCMotorParams()
    assert step_itae("lpid", dict(Kp=-1, Ki=0, Kd=0), p, 20.0) == 1e6
    assert step_itae("lpid", dict(Kp=100, Ki=1000, Kd=1), p, 20.0) < 10.0


def test_tune_pso_respects_bounds_and_is_seeded():
    p = DCMotorParams()
    center = (100.0, 1000.0, 1.0)
    kw = dict(n_particles=6, iters=4, t_end=1.0, seed=7)
    g1, c1 = tune_pso("lpid", p, 20.0, center, **kw)
    g2, c2 = tune_pso("lpid", p, 20.0, center, **kw)
    assert g1 == g2 and c1 == c2
    for name, c in zip(("Kp", "Ki", "Kd"), center):
        assert 0.1 * c <= g1[name] <= 3.0 * c


def test_make_controller_kinds():
    p = DCMotorParams()
    assert type(make_controller("lpid", dict(Kp=1, Ki=1, Kd=1), p)).__name__ == "LPID"
    n = make_controller("npid", dict(Kp=1, Ki=1, Kd=1, n=3), p, e_scale=5.0)
    assert (n.n, n.e_scale, n.u_max) == (3, 5.0, p.Vmax)
    with pytest.raises(ValueError):
        make_controller("pid", dict(Kp=1, Ki=1, Kd=1), p)
