"""Validation of the classical methods on exact plants with analytically known answers (docs/PROTOCOL.md section 9)."""
import numpy as np
import pytest

from src.plant import DelayedPlant, TransferFunctionPlant
from src.tuners import NotApplicable, TuningProblem
from src.tuners.classical import (CYCLING_RULES, CohenCoon, ContinuousCycling, TimeDomainTuning, TrialAndError,
                                  ZieglerNicholsReactionCurve, to_parallel)

K, TAU, THETA = 2.0, 5.0, 1.0            # exact FOPDT: theta/tau = 0.2, in the rules' validity range


def fopdt(theta=THETA, tau=TAU, y_ref=1.0, dt=0.02, t_end=60.0):
    return TuningProblem("fopdt-test", lambda: DelayedPlant(TransferFunctionPlant([K], [tau, 1.0], u_limits=(-10, 10)), theta),
                         y_ref=y_ref, dt=dt, t_end=t_end)


def test_to_parallel_conversion():
    assert to_parallel(3.0, 2.0, 0.5) == dict(Kp=3.0, Ki=1.5, Kd=1.5)
    assert to_parallel(3.0, np.inf, 0.0)["Ki"] == 0.0


# ---- formulas: identification on an exact plant must reproduce the textbook gains -----------------
def test_ziegler_nichols_reaction_curve_formulas():
    g = ZieglerNicholsReactionCurve(id_method="lsq").tune(fopdt()).gains      # exact (K,tau,theta) -> textbook values
    assert g["Kp"] == pytest.approx(1.2 * TAU / (K * THETA), rel=0.03)         # 3.0
    assert g["Ki"] == pytest.approx(g["Kp"] / (2 * THETA), rel=0.05)            # Ti = 2 theta
    assert g["Kd"] == pytest.approx(g["Kp"] * 0.5 * THETA, rel=0.05)            # Td = theta/2


def test_ziegler_nichols_default_uses_tangent_and_is_close_but_more_conservative_than_exact():
    g_tan = ZieglerNicholsReactionCurve().tune(fopdt()).gains
    g_lsq = ZieglerNicholsReactionCurve(id_method="lsq").tune(fopdt()).gains
    assert g_tan["Kp"] == pytest.approx(g_lsq["Kp"], rel=0.15)


def test_cohen_coon_formulas():
    r = THETA / TAU
    kp = (TAU / (K * THETA)) * (4 / 3 + r / 4)
    ti = THETA * (32 + 6 * r) / (13 + 8 * r)
    td = 4 * THETA / (11 + 2 * r)
    g = CohenCoon(id_method="lsq").tune(fopdt()).gains
    assert (g["Kp"], g["Ki"], g["Kd"]) == pytest.approx((kp, kp / ti, kp * td), rel=0.04)


@pytest.mark.parametrize("rule,kp_c,ti_c,td_c", [("chr0", 0.6, 1.0, 0.5), ("chr20", 0.95, 1.357, 0.473)])
def test_chr_formulas(rule, kp_c, ti_c, td_c):
    g = TimeDomainTuning(rule).tune(fopdt()).gains
    kp = kp_c * TAU / (K * THETA)
    assert (g["Kp"], g["Ki"], g["Kd"]) == pytest.approx((kp, kp / (ti_c * TAU), kp * td_c * THETA), rel=0.04)


def test_simc_formulas_fopdt_and_sopdt():
    # exact FOPDT: tau_c = theta -> Kp = tau/(K (2 theta)), Ti = min(tau, 8 theta)
    g = TimeDomainTuning("simc").tune(fopdt()).gains
    kp = TAU / (K * 2 * THETA)
    assert g["Kp"] == pytest.approx(kp, rel=0.1)
    assert g["Ki"] == pytest.approx(kp / min(TAU, 8 * THETA), rel=0.12)
    # exact SOPDT 1/((4s+1)(1s+1)) with delay 0.5: Td = tau2, Kp = tau1/(K (tau_c+theta)), tau_c = theta
    p = TuningProblem("sopdt", lambda: DelayedPlant(TransferFunctionPlant([1.5], [4.0, 5.0, 1.0], u_limits=(-10, 10)), 0.5),
                      y_ref=1.0, dt=0.02, t_end=60.0)
    r = TimeDomainTuning("simc").tune(p)
    assert r.meta["tau2"] == pytest.approx(1.0, rel=0.2) and r.meta["Td"] == pytest.approx(r.meta["tau2"])
    assert r.gains["Kp"] == pytest.approx(4.0 / (1.5 * (0.5 + 0.5)), rel=0.2)


def test_simc_tau_c_controls_aggressiveness():
    fast = TimeDomainTuning("simc", tau_c_factor=0.5).tune(fopdt()).gains["Kp"]
    slow = TimeDomainTuning("simc", tau_c_factor=3.0).tune(fopdt()).gains["Kp"]
    assert fast > slow


def test_integrating_plant_rules():
    # Kv e^{-theta s}/s with Kv=2, theta=0.5 (no lag)
    p = TuningProblem("ipdt", lambda: DelayedPlant(TransferFunctionPlant([2.0], [1.0, 0.0], u_limits=(-10, 10)), 0.5),
                      y_ref=1.0, dt=0.005, t_end=8.0)
    zn = ZieglerNicholsReactionCurve().tune(p)
    assert zn.meta["model"] == "ipdt"
    assert zn.gains["Kp"] == pytest.approx(1.2 / (2.0 * 0.5), rel=0.15)             # 1.2/(Kv L)
    cc = CohenCoon().tune(p).gains["Kp"]
    assert cc == pytest.approx(4.0 / (3.0 * 2.0 * 0.5), rel=0.15)


# ---- validity and strictness -----------------------------------------------------------------------
def test_open_loop_rules_refuse_out_of_range_dead_time_in_strict_mode_only():
    p = TuningProblem("tiny-delay", lambda: DelayedPlant(TransferFunctionPlant([K], [TAU, 1.0], u_limits=(-10, 10)), 0.02),
                      y_ref=1.0, dt=0.02, t_end=60.0)
    for tuner in (ZieglerNicholsReactionCurve(), CohenCoon(), TimeDomainTuning("chr0")):
        r = tuner.tune(p)
        assert not r.success and "not applicable" in r.message
    relaxed = ZieglerNicholsReactionCurve(strict=False).tune(p)
    assert relaxed.success and "outside" in relaxed.meta["validity"] and relaxed.gains["Kp"] > 10   # absurdly aggressive
    assert TimeDomainTuning("simc").tune(p).success                                   # SIMC has no ratio restriction


def test_experiment_cost_is_counted_and_small_for_open_loop_methods():
    r = CohenCoon().tune(fopdt())
    assert 2 <= r.evals <= 12 and r.sim_seconds > 0 and r.wall_time > 0


# ---- continuous cycling ---------------------------------------------------------------------------
def analytic_ultimate_point():
    from scipy.optimize import brentq
    w = brentq(lambda w: np.arctan(TAU * w) + THETA * w - np.pi, 0.5, 3.0)
    return np.sqrt(1 + (TAU * w) ** 2) / K, 2 * np.pi / w


def test_continuous_cycling_oscillation_matches_analytic_ultimate_point():
    Ku, Tu = analytic_ultimate_point()
    r = ContinuousCycling("zn").tune(fopdt())
    assert r.success and not r.meta["limit_cycle_artefact"]
    assert r.meta["Ku"] == pytest.approx(Ku, rel=0.03) and r.meta["Tu"] == pytest.approx(Tu, rel=0.03)
    kp_c, ti_c, td_c = CYCLING_RULES["zn"]
    assert r.gains["Kp"] == pytest.approx(kp_c * Ku, rel=0.04)
    assert r.gains["Ki"] == pytest.approx(kp_c * Ku / (ti_c * Tu), rel=0.06)


def test_continuous_cycling_relay_within_describing_function_error():
    Ku, Tu = analytic_ultimate_point()
    r = ContinuousCycling("zn", source="relay").tune(fopdt())
    assert r.success and r.meta["Ku"] == pytest.approx(Ku, rel=0.25) and r.meta["Tu"] == pytest.approx(Tu, rel=0.08)


def test_rule_sets_are_ordered_by_aggressiveness():
    kp = {rule: ContinuousCycling(rule).tune(fopdt()).gains["Kp"] for rule in ("zn", "tyreus-luyben", "no-overshoot")}
    assert kp["zn"] > kp["tyreus-luyben"] > kp["no-overshoot"]


def test_cycling_not_applicable_without_phase_crossover_and_limit_cycle_fallback_flagged():
    first_order = TuningProblem("1st-order", lambda: TransferFunctionPlant([2.0], [5.0, 1.0], u_limits=(-10, 10)),
                                y_ref=1.0, dt=0.02, t_end=60.0)
    r = ContinuousCycling("zn").tune(first_order)
    assert not r.success and "no ultimate gain" in r.message or "sampling limit" in r.message
    dc = TuningProblem.from_scenario("dc_speed")
    strict = ContinuousCycling("zn").tune(dc)
    assert not strict.success                                      # the motor has no true ultimate gain
    fb = ContinuousCycling("zn", fallback=True).tune(TuningProblem.from_scenario("dc_speed"))
    # the limit-cycle "Ku" is an artefact that depends on horizon and threshold (quantified in experiment m1): wide band only
    assert fb.success and fb.meta["limit_cycle_artefact"] and 50 < fb.meta["Ku"] < 400


# ---- trial and error -------------------------------------------------------------------------------
def test_trial_and_error_is_deterministic_budgeted_and_improves_on_start():
    p1, p2 = fopdt(), fopdt()
    r1, r2 = TrialAndError().tune(p1), TrialAndError().tune(p2)
    assert r1.gains == r2.gains and r1.evals == r2.evals <= 40
    assert r1.success and r1.history[-1][1] <= r1.history[0][1]
    assert TrialAndError(max_trials=5).tune(fopdt()).evals <= 5
    start = fopdt().cost(dict(Kp=0.1 * 10 / 1.0, Ki=0.0, Kd=0.0))
    assert fopdt().cost(r1.gains) < start


def test_trial_and_error_features_detect_overshoot_and_steady_state_error():
    p = fopdt()
    f = TrialAndError.features(p, p.simulate(dict(Kp=0.3, Ki=0.0, Kd=0.0)))
    assert f["ess"] > 20 and not f["diverged"]                      # P-only: large offset
    f = TrialAndError.features(p, p.simulate(dict(Kp=6.0, Ki=2.0, Kd=0.0)))
    assert f["mp"] > 20 or f["oscillating"]
