import numpy as np
import pytest

from src.plant import (DCMotorParams, DriftingDCMotor, FunctionPlant, Plant,
                       StateSpacePlant, TransferFunctionPlant, dc_motor_dynamics)
from src.legacy.rl_drift import compute_R


def open_loop(plant, u, t_end, dt=0.001, d=0.0):
    x = plant.initial_state()
    for k in range(int(t_end / dt)):
        x = plant.step(k * dt, x, u, d, dt)
    return x


def test_dc_motor_steady_state_matches_analytic():
    # Ke*w + R*i = V, Kt*i = B*w + Tc  ->  w = (Kt*V/R - Tc) / (B + Kt*Ke/R)
    p = DCMotorParams()
    w = open_loop(p, 12.0, 4.0)[1]
    expected = (p.Kt * 12.0 / p.R - p.Tc) / (p.B + p.Kt * p.Ke / p.R)
    assert w == pytest.approx(expected, rel=1e-3)


def test_dc_motor_fast_step_equals_generic_rk4():
    p = DCMotorParams()
    x = [0.3, 4.0]
    fast = p.step(0.0, x, 10.0, 0.02, 0.001)
    generic = Plant.step(p, 0.0, x, 10.0, 0.02, 0.001)
    assert fast == pytest.approx(generic, abs=1e-12)


def test_dc_motor_is_a_plant_with_symmetric_limits():
    p = DCMotorParams(Vmax=12.0)
    assert (p.u_min, p.u_max, p.Vmax) == (-12.0, 12.0, 12.0)
    assert p.saturate(100.0) == 12.0 and p.saturate(-100.0) == -12.0
    assert dc_motor_dynamics(0.0, [0.0, 0.0], 1.0, 0.0, p) == p.derivatives(0.0, [0.0, 0.0], 1.0)


def test_drifting_motor_resistance_and_no_drift_limit():
    d = DriftingDCMotor(k_heat=0.05)
    assert d.resistance(20.0) == pytest.approx(compute_R(d.R, 0.05, 20.0))
    assert d.resistance(20.0) == pytest.approx(2.0 * d.R)
    still = DriftingDCMotor(k_heat=0.0)
    assert still.step(0.0, [0, 0], 5.0, 0.0, 0.001) == DCMotorParams().step(0.0, [0, 0], 5.0, 0.0, 0.001)


def test_drift_slows_the_motor():
    cold = open_loop(DCMotorParams(), 12.0, 3.0)[1]
    hot = open_loop(DriftingDCMotor(k_heat=0.5), 12.0, 3.0)[1]
    assert hot < cold


def test_transfer_function_dc_gain():
    tf = TransferFunctionPlant([2.0], [1.0, 3.0, 2.0])      # DC gain 1
    assert tf.output(open_loop(tf, 1.0, 10.0)) == pytest.approx(1.0, abs=2e-3)


def test_transfer_function_with_direct_feedthrough():
    tf = TransferFunctionPlant([1.0, 1.0], [1.0, 1.0])       # (s+1)/(s+1) = 1
    x = tf.initial_state()
    x = tf.step(0.0, x, 2.0, 0.0, 0.001)
    assert tf.output(x) == pytest.approx(2.0, rel=1e-6)


def test_improper_transfer_function_rejected():
    with pytest.raises(ValueError):
        TransferFunctionPlant([1, 0, 0], [1, 1])


def test_state_space_matches_transfer_function():
    a = TransferFunctionPlant([1.0], [1.0, 3.0, 2.0])
    b = StateSpacePlant(a.A, a.B, a.C, a.D)
    assert a.output(open_loop(a, 1.0, 2.0)) == pytest.approx(b.output(open_loop(b, 1.0, 2.0)))


def test_function_plant_nonlinear_tank():
    tank = FunctionPlant(lambda t, x, u, d: [-np.sqrt(max(x[0], 0.0)) + u + d], x0=[0.0])
    # steady state: sqrt(h) = u  ->  h = u^2
    assert tank.output(open_loop(tank, 2.0, 30.0)) == pytest.approx(4.0, rel=2e-2)


def test_delayed_plant_output_is_exactly_delayed():
    from src.plant import DelayedPlant
    base = TransferFunctionPlant([1.0], [1.0, 1.0])
    delayed = DelayedPlant(base, 0.5)
    ref = open_loop(base, 1.0, 1.0)                 # state after 1 s without delay
    x = delayed.initial_state()
    traj = []
    for k in range(1500):
        x = delayed.step(k * 0.001, x, 1.0, 0.0, 0.001)
        traj.append(delayed.output(x))
    assert max(abs(v) for v in traj[:499]) == 0.0   # nothing happens during the dead time
    assert traj[1499] == pytest.approx(base.output(ref), rel=1e-3)   # then it follows the 1 s response


def test_delayed_plant_resets_between_runs():
    from src.plant import DelayedPlant
    d = DelayedPlant(TransferFunctionPlant([1.0], [1.0, 1.0]), 0.1)
    a = open_loop(d, 1.0, 0.5)
    b = open_loop(d, 1.0, 0.5)                      # initial_state() clears the FIFO
    assert a == pytest.approx(b)
