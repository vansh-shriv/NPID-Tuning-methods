import numpy as np
import pytest

from src.plant_family import PlantSpec, problem_from_spec
from src.tuners import TuningProblem
from src.tuners.features import (FEATURE_NAMES, N_SIGNATURE, denormalise_gains, extract_features, normalise_cost,
                                 normalise_gains)
from src.tuners.optimizers import GA, PSO, Objective


def spec(**kw):
    base = dict(ptype="fopdt", K=2.0, tau1=5.0, theta=1.0, y_ref=1.0, r_auth=5.0, dist_frac=0.1, noise_rel=0.01)
    base.update(kw)
    return PlantSpec(**base)


def test_features_of_an_exact_fopdt_are_the_textbook_values_and_cost_is_counted():
    p = problem_from_spec(spec())
    f = extract_features(p)
    assert f.kind == "fopdt" and f.K_eff == pytest.approx(2.0, rel=0.02) and f.T_n == pytest.approx(6.0, rel=0.03)
    names = dict(zip(FEATURE_NAMES, f.vec))
    assert names["integrating"] == 0 and names["delay_fraction"] == pytest.approx(1.0 / 6.0, rel=0.05)
    assert names["peak_ratio"] == pytest.approx(1.0, abs=0.01) and names["t63_over_Tn"] == pytest.approx(1.0, abs=0.05)
    assert names["log_authority"] == pytest.approx(np.log10(5.0), abs=0.05)               # r_auth = u_max K / y_ref
    assert f.signature.shape == (N_SIGNATURE,) and f.signature[0] == pytest.approx(0.0, abs=0.01) and f.signature[-1] == pytest.approx(1.0, abs=0.01)
    assert 2 <= f.cost_evals <= 12 and f.cost_sim_seconds > 0 and p.evals == f.cost_evals


def test_features_are_invariant_to_plant_gain_and_time_scale():
    a = extract_features(problem_from_spec(spec()))
    b = extract_features(problem_from_spec(spec(K=0.5, tau1=15.0, theta=3.0, y_ref=7.0)))      # K/4, time x3, other setpoint
    assert b.vec == pytest.approx(a.vec, rel=0.05, abs=0.03)
    assert b.signature == pytest.approx(a.signature, abs=0.03)
    assert b.K_eff == pytest.approx(a.K_eff / 4, rel=0.03) and b.T_n == pytest.approx(3 * a.T_n, rel=0.03)


def test_underdamped_plant_shows_in_the_peak_ratio_and_fit_error():
    f = extract_features(problem_from_spec(PlantSpec("underdamped", K=1.0, tau1=1.0, wn=2.0, zeta=0.3, y_ref=1.0, r_auth=5.0)))
    d = dict(zip(FEATURE_NAMES, f.vec))
    assert d["peak_ratio"] > 1.2 and d["log_fit_error"] > np.log10(0.01)


def test_integrating_plant_features_and_ramp_signature():
    f = extract_features(problem_from_spec(PlantSpec("ipdt", K=1.5, tau1=0.5, theta=0.2, y_ref=1.0, r_auth=3.0)))
    assert f.kind == "ipdt" and f.vec[0] == 1.0
    assert f.signature[-1] > 3.0 and np.all(np.diff(f.signature) >= -1e-3)                 # increasing ramp, slope ~ 1 per T_n


def test_unipolar_flag_and_noisy_identification_stay_close():
    s = spec(unipolar=True)
    clean = extract_features(problem_from_spec(s))
    noisy = extract_features(problem_from_spec(s), noise_sigma=0.01, seed=1)
    assert clean.vec[4] == 1.0 and noisy.vec[4] == 1.0
    assert noisy.vec[[1, 3, 6]] == pytest.approx(clean.vec[[1, 3, 6]], rel=0.35, abs=0.15)


def test_gain_normalisation_roundtrip_and_physical_scaling():
    g = dict(Kp=3.0, Ki=0.7, Kd=0.4)
    z = normalise_gains(g, K_eff=2.0, T_n=6.0)
    back = denormalise_gains(z, 2.0, 6.0)
    assert (back["Kp"], back["Ki"], back["Kd"]) == pytest.approx((3.0, 0.7, 0.4))
    # plant gain x4, time x3: the *same* normalised gains correspond to Kp/4, Ki/(4*3), Kd*3/4
    scaled = denormalise_gains(z, 8.0, 18.0)
    assert (scaled["Kp"], scaled["Ki"], scaled["Kd"]) == pytest.approx((3.0 / 4, 0.7 / 12, 0.4 * 3 / 4))
    assert normalise_gains(dict(Kp=1.0, Ki=1.0, Kd=0.0), 1.0, 1.0)[2] == pytest.approx(-9.0)   # Kd = 0 is floored
    assert normalise_cost(12.0, 2.0, 3.0) == pytest.approx(12.0 / (2.0 * 9.0))


def test_normalised_optimum_transfers_between_scaled_plants():
    """The loop (plant, gains) scaled consistently has the same response: the basis of the whole normalisation."""
    s1, s2 = spec(), spec(K=0.5, tau1=15.0, theta=3.0)
    f1, f2 = extract_features(problem_from_spec(s1)), extract_features(problem_from_spec(s2))
    z = normalise_gains(dict(Kp=1.1, Ki=0.15, Kd=0.5), f1.K_eff, f1.T_n)
    g2 = denormalise_gains(z, f2.K_eff, f2.T_n)
    r1 = problem_from_spec(s1).simulate(dict(Kp=1.1, Ki=0.15, Kd=0.5), "lpid")
    r2 = problem_from_spec(s2).simulate(g2, "lpid")
    t1 = np.linspace(0, 1, 200)
    y1 = np.interp(t1 * r1.time[-1], r1.time, r1.omega) / s1.y_ref
    y2 = np.interp(t1 * r2.time[-1], r2.time, r2.omega) / s2.y_ref
    assert np.max(np.abs(y1 - y2)) < 0.06


def test_optimizer_recorder_logs_every_evaluation():
    p = TuningProblem.from_scenario("dc_speed")
    rec = []
    res = GA(recorder=rec).tune(p, budget=60, seed=0)
    assert len(rec) == res.evals >= 50 and all(len(r) == 4 for r in rec)       # one record per simulation actually used
    lb, ub = np.log10(p.bounds[0]), np.log10(p.bounds[1])
    arr = np.array(rec)
    assert np.all(arr[:, :3] >= lb - 1e-9) and np.all(arr[:, :3] <= ub + 1e-9)
    assert arr[:, 3].min() > 0
    rec2 = []
    res2 = PSO(recorder=rec2).tune(TuningProblem.from_scenario("dc_speed"), budget=60, seed=0)
    assert len(rec2) == res2.evals >= 50
