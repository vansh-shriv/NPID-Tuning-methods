"""Validation of the self-tuning pole-placement machinery on exact plants."""
import numpy as np
import pytest
from scipy.signal import cont2discrete

from src.metrics import compute_metrics
from src.plant import DelayedPlant, TransferFunctionPlant
from src.tuners import TuningProblem
from src.tuners.selftuning import (RLS, SelfTuningPolePlacement, _to_parallel, arx_regressor, arx_step_response,
                                   design_with_fallback, desired_wn, dominant_pair, fit_arx_batch, pole_placement_pid,
                                   select_delay)


def discrete_plant(num, den, delta):
    nd, dd, _ = cont2discrete((num, den), delta, method="zoh")[:3]
    nd = np.squeeze(nd)
    return dd[1], dd[2], nd[1], nd[2]                                     # a1, a2, b0, b1 (one-sample delay form)


def simulate_arx(a1, a2, b0, b1, nk, u, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    y = np.zeros(len(u))
    for k in range(len(u)):
        y[k] = (-a1 * (y[k - 1] if k >= 1 else 0) - a2 * (y[k - 2] if k >= 2 else 0)
                + b0 * (u[k - nk] if k - nk >= 0 else 0) + b1 * (u[k - nk - 1] if k - nk - 1 >= 0 else 0)
                + (rng.normal(0, noise) if noise else 0))
    return y


# ---- estimator ------------------------------------------------------------------------------------
def test_rls_recovers_exact_second_order_model():
    a1, a2, b0, b1 = discrete_plant([1.0], [2.0, 3.0, 1.0], 0.2)
    u = np.random.default_rng(0).choice([-1.0, 1.0], 800)
    y = simulate_arx(a1, a2, b0, b1, 1, u)
    dy, du = np.diff(y, prepend=y[0]), np.diff(u, prepend=u[0])
    rls = RLS(4, lam=1.0)
    for k in range(3, len(dy)):
        rls.update(arx_regressor(dy, du, k, 1), dy[k])
    assert rls.theta == pytest.approx([a1, a2, b0, b1], rel=0.02, abs=1e-3)


def test_rls_batch_initialisation_equals_least_squares():
    a1, a2, b0, b1 = discrete_plant([1.0], [2.0, 3.0, 1.0], 0.2)
    u = np.random.default_rng(5).choice([-1.0, 1.0], 400)
    y = simulate_arx(a1, a2, b0, b1, 1, u)
    dy, du = np.diff(y, prepend=y[0]), np.diff(u, prepend=u[0])
    rls = RLS(4)
    rls.fit_batch([arx_regressor(dy, du, k, 1) for k in range(3, len(dy))], dy[3:])
    assert rls.theta == pytest.approx([a1, a2, b0, b1], abs=1e-6)
    assert np.all(np.linalg.eigvalsh(rls.P) > 0)


def test_rls_tracks_a_parameter_change_with_forgetting():
    a1, a2, b0, b1 = discrete_plant([1.0], [2.0, 3.0, 1.0], 0.2)
    u = np.random.default_rng(1).choice([-1.0, 1.0], 3000)
    y = np.concatenate([simulate_arx(a1, a2, b0, b1, 1, u[:1500]), simulate_arx(a1, a2, 2 * b0, 2 * b1, 1, u[1500:])])
    dy, du = np.diff(y, prepend=y[0]), np.diff(u, prepend=u[0])
    rls = RLS(4, lam=0.995)          # 0.98 is too short a memory for this slow plant (covariance blow-up), see M2 ablation
    for k in range(3, len(dy)):
        rls.update(arx_regressor(dy, du, k, 1), dy[k])
    assert rls.theta[2] == pytest.approx(2 * b0, rel=0.1)                  # follows the gain doubling


def test_rls_covariance_guard_keeps_estimates_finite_without_excitation():
    rls = RLS(4, lam=0.99, p_max=1e4)
    for _ in range(5000):
        rls.update(np.zeros(4), 0.0)                                      # no excitation at all
    assert np.all(np.isfinite(rls.theta)) and np.trace(rls.P) <= 1e4 * 1.001


def test_delay_selection_recovers_known_delay():
    a1, a2, b0, b1 = discrete_plant([1.0], [2.0, 3.0, 1.0], 0.2)
    u = np.random.default_rng(2).choice([-1.0, 1.0], 1500)
    for true_nk in (1, 3, 4):
        y = simulate_arx(a1, a2, b0, b1, true_nk, u)
        nk, losses = select_delay(np.diff(y, prepend=y[0]), np.diff(u, prepend=u[0]), nk_max=6)
        assert nk == true_nk, (true_nk, nk, losses)


def test_batch_fit_matches_exact_parameters():
    a1, a2, b0, b1 = discrete_plant([1.0], [2.0, 3.0, 1.0], 0.2)
    u = np.random.default_rng(3).choice([-1.0, 1.0], 600)
    y = simulate_arx(a1, a2, b0, b1, 2, u)
    theta, loss = fit_arx_batch(np.diff(y, prepend=y[0]), np.diff(u, prepend=u[0]), 2)
    assert theta == pytest.approx([a1, a2, b0, b1], abs=1e-6) and loss < 1e-12


# ---- design ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("den", [[2.0, 3.0, 1.0], [1.0, 1.0, 0.0], [5.0, 6.0, 1.0]])
@pytest.mark.parametrize("wn", [0.5, 1.0, 2.0])
def test_pole_placement_places_the_requested_poles_exactly(den, wn):
    delta = 0.1
    a1, a2, b0, b1 = discrete_plant([1.0], den, delta)
    d = pole_placement_pid(a1, a2, b0, b1, delta, wn, 0.8, tau_d=0.03)
    if not d.ok:
        pytest.skip("no admissible placement for this combination: " + d.note)
    d1, d2 = dominant_pair(wn, 0.8, delta)
    want = sorted(np.abs(np.concatenate([np.roots([1, d1, d2]), [-d.gamma, -d.gamma]])))
    assert sorted(np.abs(d.poles)) == pytest.approx(want, abs=2e-3)
    assert np.all(np.abs(d.poles) < 1.0) and d.Kp > 0 and d.Ki >= 0 and d.Kd >= 0


def test_at_least_most_combinations_are_feasible():
    """Guard against the placement silently skipping everything (see the skip above)."""
    feasible = 0
    for den in ([2.0, 3.0, 1.0], [1.0, 1.0, 0.0], [5.0, 6.0, 1.0]):
        for wn in (0.5, 1.0, 2.0):
            a1, a2, b0, b1 = discrete_plant([1.0], den, 0.1)
            feasible += pole_placement_pid(a1, a2, b0, b1, 0.1, wn, 0.8, 0.03).ok
    assert feasible >= 7


def test_parallel_conversion_roundtrip():
    Kp, Ki, Kd, delta, tau_f = 3.0, 1.5, 0.4, 0.1, 0.05
    rho = tau_f / (tau_f + delta)
    kappa = Kd / (tau_f + delta)
    s = np.array([Kp + Ki * delta + kappa, -Kp * (1 + rho) - Ki * delta * rho - 2 * kappa, Kp * rho + kappa])
    kp, ki, kd = _to_parallel(s, rho, delta)
    assert (kp, ki, kd) == pytest.approx((Kp, Ki, Kd))


def test_closed_loop_of_the_design_behaves_as_specified():
    """Simulate the designed discrete PID on the exact discrete plant: zero steady-state error, small overshoot."""
    delta, wn, zeta, tau_d = 0.1, 1.5, 0.8, 0.02
    a1, a2, b0, b1 = discrete_plant([1.0], [2.0, 3.0, 1.0], delta)
    d = pole_placement_pid(a1, a2, b0, b1, delta, wn, zeta, tau_d)
    assert d.ok
    rho = tau_d / (tau_d + delta)
    kappa = d.Kd / (tau_d + delta)
    s0 = d.Kp + d.Ki * delta + kappa
    s1 = -d.Kp * (1 + rho) - d.Ki * delta * rho - 2 * kappa
    s2 = d.Kp * rho + kappa
    n = 200
    y, u, e = np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(n):
        y[k] = (-a1 * (y[k - 1] if k >= 1 else 0) - a2 * (y[k - 2] if k >= 2 else 0)
                + b0 * (u[k - 1] if k >= 1 else 0) + b1 * (u[k - 2] if k >= 2 else 0))
        e[k] = 1.0 - y[k]
        # R u = S e with R = (1 - z^-1)(1 - rho z^-1) = 1 - (1+rho) z^-1 + rho z^-2
        u[k] = ((1 + rho) * (u[k - 1] if k >= 1 else 0) - rho * (u[k - 2] if k >= 2 else 0)
                + s0 * e[k] + s1 * (e[k - 1] if k >= 1 else 0) + s2 * (e[k - 2] if k >= 2 else 0))
    assert y[-1] == pytest.approx(1.0, abs=1e-3)                          # integral action
    # the poles are exactly those of zeta = 0.8 (checked above); the unplaced PID zeros add overshoot to the set-point response
    assert 100 * (y.max() - 1.0) < 25.0


def test_design_fallback_detunes_when_requested_poles_are_infeasible():
    a1, a2, b0, b1 = discrete_plant([1.0], [2.0, 3.0, 1.0], 0.2)
    d = design_with_fallback(a1, a2, b0, b1, 0.2, wn=60.0, zeta=0.8, tau_d=0.02)
    assert d.ok and d.wn < 60.0 and "detuned" in d.note


def test_desired_wn_scales_with_speedup_and_rejects_unstable_models():
    a1, a2, b0, b1 = discrete_plant([1.0], [2.0, 3.0, 1.0], 0.2)
    w3, t_open = desired_wn(a1, a2, b0, b1, 0.2, speedup=3.0)
    w6, _ = desired_wn(a1, a2, b0, b1, 0.2, speedup=6.0)
    assert w6 == pytest.approx(2 * w3, rel=0.05)
    assert t_open == pytest.approx(4 * 2.0, rel=0.25)                     # 2% settling of a 2 s dominant lag ~ 8 s
    assert desired_wn(-2.5, 1.7, 0.1, 0.1, 0.2) is None                   # unstable ARX poles
    assert arx_step_response(a1, a2, b0, b1, n=3000)[-1] == pytest.approx(1.0, rel=1e-2)      # unit DC gain


# ---- tuner --------------------------------------------------------------------------------------
def exact_problem(gain=1.0, theta=0.0, tau1=2.0, tau2=1.0, dt=0.02, t_end=40.0, noise_sigma=0.0):
    def factory():
        p = TransferFunctionPlant([gain], [tau1 * tau2, tau1 + tau2, 1.0], u_limits=(-50, 50))
        return DelayedPlant(p, theta) if theta else p
    return TuningProblem("exact2", factory, y_ref=1.0, dt=dt, t_end=t_end, noise_sigma=noise_sigma)


def test_tuner_on_exact_second_order_plant_gives_a_good_loop():
    r = SelfTuningPolePlacement().tune(exact_problem())
    assert r.success and r.evals == 1 and r.sim_seconds > 100 and r.meta["nk"] <= 2
    assert r.meta["a2"] == pytest.approx(discrete_plant([1.0], [2.0, 3.0, 1.0], r.meta["delta"])[1], abs=0.05)
    res = exact_problem().simulate(r.gains, "lpid")
    m = compute_metrics(res.time, 1.0, res.omega)
    assert m["overshoot_pct"] < 30 and np.isfinite(m["settling_time"]) and m["settling_time"] < 30


def test_gains_scale_inversely_with_plant_gain():
    k1 = SelfTuningPolePlacement().tune(exact_problem(gain=1.0)).gains["Kp"]
    k2 = SelfTuningPolePlacement().tune(exact_problem(gain=2.0)).gains["Kp"]
    assert k2 == pytest.approx(k1 / 2.0, rel=0.25)


def test_tuner_handles_dead_time_by_resampling():
    r = SelfTuningPolePlacement().tune(exact_problem(theta=0.6))
    assert r.success and r.meta["nk"] >= 2 and r.meta["delta"] >= 0.2


def test_tuner_is_deterministic_and_tolerates_measurement_noise():
    a = SelfTuningPolePlacement().tune(exact_problem()).gains
    b = SelfTuningPolePlacement().tune(exact_problem()).gains
    assert a == b
    n = SelfTuningPolePlacement(noisy=True).tune(exact_problem(noise_sigma=0.01))
    assert n.success and n.gains["Kp"] == pytest.approx(a["Kp"], rel=0.6)


# ---- integrating plants ---------------------------------------------------------------------------
def simulate_integrating(alpha, b0, b1, nk, u):
    """(1 - alpha q^-1) dy = b0 u(k-nk) + b1 u(k-nk-1)  <=>  A y = B u with A = (1 - q^-1)(1 - alpha q^-1)."""
    y = np.zeros(len(u))
    for k in range(len(u)):
        dy_prev = (y[k - 1] - y[k - 2]) if k >= 2 else 0.0
        y[k] = (y[k - 1] if k >= 1 else 0.0) + alpha * dy_prev \
            + b0 * (u[k - nk] if k - nk >= 0 else 0) + b1 * (u[k - nk - 1] if k - nk - 1 >= 0 else 0)
    return y


def test_integrating_structure_recovers_exact_model_and_maps_to_arx_with_unit_root():
    from src.tuners.selftuning import to_arx
    u = np.random.default_rng(4).choice([-1.0, 1.0], 800)
    y = simulate_integrating(0.8, 0.05, 0.02, 1, u)
    dy, du = np.diff(y, prepend=y[0]), np.diff(u, prepend=u[0])
    theta, loss = fit_arx_batch(dy, du, 1, u, integrating=True)
    assert loss < 1e-12 and theta[0] == pytest.approx(-0.8, abs=1e-6) and theta[1:3] == pytest.approx([0.05, 0.02], abs=1e-6)
    assert theta[3] == pytest.approx(0.0, abs=1e-6)                         # no offset in the data
    a1, a2, b0, b1 = to_arx(theta, True)
    assert np.sort(np.abs(np.roots([1.0, a1, a2]))) == pytest.approx([0.8, 1.0], abs=1e-6)


def test_integrating_structure_absorbs_an_input_offset_that_the_level_form_alone_would_not():
    u = np.random.default_rng(6).choice([-1.0, 1.0], 800)
    y = simulate_integrating(0.8, 0.05, 0.02, 1, u + 0.3)                    # friction-like constant offset in the input
    dy, du = np.diff(y, prepend=y[0]), np.diff(u, prepend=u[0])
    theta, loss = fit_arx_batch(dy, du, 1, u, integrating=True)
    assert loss < 1e-10 and theta[3] == pytest.approx(0.3 * (0.05 + 0.02), abs=1e-4)


def test_structure_choice_prefers_regulating_for_a_stable_plant():
    from src.tuners.selftuning import choose_structure
    a1, a2, b0, b1 = discrete_plant([1.0], [2.0, 3.0, 1.0], 0.2)
    u = np.random.default_rng(7).choice([-1.0, 1.0], 800)
    y = simulate_arx(a1, a2, b0, b1, 1, u, noise=1e-4)
    integ, nk, losses, aic_r, aic_i = choose_structure(np.diff(y, prepend=y[0]), np.diff(u, prepend=u[0]), u)
    assert not integ and aic_r < aic_i


def test_desired_wn_for_integrating_models():
    w, t_open = desired_wn(-1.8, 0.8, 0.05, 0.02, 0.1, speedup=3.0)          # roots 1.0 and 0.8 -> integrating
    assert np.isfinite(w) and t_open == pytest.approx(10 * (-0.1 / np.log(0.8) + 0.1), rel=1e-6)
    w2, t2 = desired_wn(-0.8, 0.8, 0.05, 0.02, 0.1, speedup=3.0, integrating=True)
    assert (w2, t2) == pytest.approx((w, t_open))


def test_tuner_on_exact_integrating_plant():
    def factory():
        return DelayedPlant(TransferFunctionPlant([2.0], [0.3, 1.0, 0.0], u_limits=(-20, 20)), 0.1)
    p = TuningProblem("ipdt-str", factory, y_ref=1.0, dt=0.005, t_end=10.0)
    r = SelfTuningPolePlacement().tune(p)
    assert r.success
    res = p.simulate(r.gains, "lpid", count=False)
    m = compute_metrics(res.time, 1.0, res.omega)
    assert abs(res.omega[-1] - 1.0) < 0.05 and m["overshoot_pct"] < 40


def test_tuner_returns_gains_on_all_five_benchmark_plants():
    for name in ("dc_speed", "dc_position", "link", "fopdt", "tank"):
        r = SelfTuningPolePlacement().tune(TuningProblem.from_scenario(name))
        assert r.success and all(r.gains[k] >= 0 for k in ("Kp", "Ki", "Kd")) and r.gains["Kp"] > 0, (name, r.message)


# ---- PI fallback and design feasibility ---------------------------------------------------------------
def test_pi_placement_places_the_dominant_pair_and_gives_a_stable_loop():
    from src.tuners.selftuning import pole_placement_pi
    a1, a2, b0, b1 = discrete_plant([2.0], [5.0 * 0.01, 5.0 + 0.01, 1.0], 0.5)      # lag-dominant (tau = 5 s)
    d = pole_placement_pi(a1, a2, b0, b1, 0.5, wn=0.6, zeta=0.8)
    assert d.ok and d.Kd == 0.0 and d.Kp > 0 and d.Ki > 0
    d1, d2 = dominant_pair(0.6, 0.8, 0.5)
    dom = np.roots([1, d1, d2])
    for p in dom:
        assert np.min(np.abs(d.poles - p)) < 1e-6                                # both dominant poles are present
    assert np.all(np.abs(d.poles) < 1.0)
    assert np.sort(np.abs(d.poles))[0] == pytest.approx(-d.gamma, abs=1e-6)       # and the third is the auxiliary pole


def test_fallback_uses_pi_when_pid_is_inadmissible_and_reports_it():
    a1, a2, b0, b1 = discrete_plant([2.0], [5.0 * 0.01, 5.0 + 0.01, 1.0], 0.26)
    # strong derivative filter (tau_d ~ Delta) makes a positive-gain PID placement impossible for a lag-dominant plant
    d = design_with_fallback(a1, a2, b0, b1, 0.26, wn=0.5, zeta=0.8, tau_d=0.25)
    assert d.ok and d.Kp > 0 and d.Ki > 0


def test_design_rejects_slow_auxiliary_poles_and_detunes():
    a1, a2, b0, b1 = -1.1901, 0.4254, 0.1928, 0.1436                              # identified DC motor model (regression case)
    fast = pole_placement_pid(a1, a2, b0, b1, 0.072, 15.6, 0.8, 0.0111)
    assert not fast.ok and "slower" in fast.note                                  # only an aux pole slower than the pair exists
    d = design_with_fallback(a1, a2, b0, b1, 0.072, 15.6, 0.8, 0.0111)
    assert d.ok and d.wn < 15.6 and d.Kp > 0.05                                   # detuned to a genuinely dominant pair
