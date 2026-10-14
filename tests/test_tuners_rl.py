"""Validation of the RL tuning environment, the session tuners and the PPO integration."""
import importlib.util
import json
import os

import numpy as np
import pytest

from src.plant_family import PlantSpec, problem_from_spec, sample_specs, spec_to_dict
from src.tuners.features import extract_features
from src.tuners.ml import load_bundle
from src.tuners.rl import (OBS_SIZE, ConstantStart, RandomLocalSearch, RLTuner, TuningEnv, make_observation,
                           response_shape)

_spec = importlib.util.spec_from_file_location(
    "m4a", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "m4a_dataset.py"))
m4a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m4a)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    path = tmp_path_factory.mktemp("rl") / "mini.jsonl"
    with open(path, "w") as f:
        for s in sample_specs(4, seed=9):
            f.write(json.dumps(m4a.label_plant(spec_to_dict(s), budget=30)) + "\n")
    return load_bundle(str(path))


def zlims(b):
    return np.percentile(np.vstack(b.eval_z), [0.5, 99.5], axis=0)


def test_observation_vector_layout_and_normalisation():
    lo, hi = np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    o = make_observation(np.arange(8.0), np.array([0.0, 1.0, -1.0]), lo, hi, 2.0, 1.5, 1.0, np.array([0.1, 0.2, 0.3]), 3, 8)
    assert o.shape == (OBS_SIZE,) and o.dtype == np.float32
    assert list(o[8:11]) == [0.0, 1.0, -1.0]                     # normalised point in [-1, 1]
    assert o[11] == pytest.approx(0.5) and o[12] == pytest.approx(1.0)                 # improvements start-last, start-best
    assert o[-1] == pytest.approx(3 / 8)


def test_response_shape_of_a_clean_and_a_diverging_response():
    p = problem_from_spec(PlantSpec("fopdt", K=2.0, tau1=5.0, theta=1.0, y_ref=1.0, r_auth=5.0))
    res = p.simulate(dict(Kp=0.6, Ki=0.15, Kd=0.0))
    ov, ess, rise = response_shape(p, res, 6.0)
    assert 0 <= ov < 0.5 and ess < 0.2 and 0 < rise < 5
    bad = p.simulate(dict(Kp=1e5, Ki=1e5, Kd=1e5))
    assert response_shape(p, bad, 6.0).shape == (3,)


def test_environment_api_budget_reward_and_determinism(bundle):
    env = TuningEnv(bundle, n_sims=5, seed=3)
    obs, _ = env.reset()
    assert obs.shape == (OBS_SIZE,) and env.observation_space.contains(obs)
    best, done, steps, rewards = env.best_lc, False, 0, []
    while not done:
        obs, r, done, trunc, info = env.step(env.action_space.sample())
        rewards.append(r)
        steps += 1
        assert env.best_lc <= best + 1e-12
        best = env.best_lc
        assert env.observation_space.contains(obs)
    assert steps == 4 and env.problem.evals == 5                   # start evaluation + 4 actions = 5 simulations
    assert min(rewards) >= 0.0 and np.isfinite(info["gap_to_label"])
    a = TuningEnv(bundle, n_sims=3, seed=7); b = TuningEnv(bundle, n_sims=3, seed=7)
    oa, _ = a.reset(); ob, _ = b.reset()
    assert np.array_equal(oa, ob)
    act = np.array([0.3, -0.2, 0.1], dtype=np.float32)
    assert np.array_equal(a.step(act)[0], b.step(act)[0])


def test_actions_move_the_point_by_step_size_and_zero_action_repeats_the_evaluation(bundle):
    env = TuningEnv(bundle, n_sims=4, step_size=0.4, seed=1)
    env.reset()
    z0 = env.z.copy()
    env.step(np.zeros(3, dtype=np.float32))
    assert np.allclose(env.z, z0)
    lc_before = env.last_lc
    env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert env.z[0] == pytest.approx(min(z0[0] + 0.4, env.z_hi[0] + 0.5)) and env.last_lc != lc_before


def test_session_tuners_count_the_bump_test_plus_exactly_the_budget(bundle):
    lo, hi = zlims(bundle)
    z0 = np.median(bundle.best_z, axis=0)
    p = problem_from_spec(PlantSpec("fopdt", K=2.0, tau1=5.0, theta=1.0, y_ref=1.0, r_auth=5.0))
    r = RandomLocalSearch(z0, lo, hi, n_sims=8).tune(p, seed=0)
    bump = extract_features(problem_from_spec(PlantSpec("fopdt", K=2.0, tau1=5.0, theta=1.0, y_ref=1.0, r_auth=5.0))).cost_evals
    assert r.success and r.evals == bump + 8 and len(r.history) == 8
    assert all(r.history[i + 1][1] <= r.history[i][1] for i in range(7))              # best-so-far non-increasing
    r2 = RandomLocalSearch(z0, lo, hi, n_sims=8).tune(problem_from_spec(PlantSpec("fopdt", K=2.0, tau1=5.0, theta=1.0, y_ref=1.0, r_auth=5.0)), seed=0)
    assert r.gains == r2.gains
    c = ConstantStart(z0, lo, hi).tune(problem_from_spec(PlantSpec("fopdt", K=2.0, tau1=5.0, theta=1.0, y_ref=1.0, r_auth=5.0)))
    assert c.evals == bump + 1


def test_rl_tuner_with_a_stub_policy_and_with_a_real_ppo_model(bundle):
    from stable_baselines3 import PPO
    lo, hi = zlims(bundle)
    z0 = np.median(bundle.best_z, axis=0)

    class Zero:
        def predict(self, obs, deterministic=True):
            return np.zeros(3), None

    p = problem_from_spec(PlantSpec("fopdt", K=2.0, tau1=5.0, theta=1.0, y_ref=1.0, r_auth=5.0))
    stub = RLTuner(Zero(), z0, lo, hi, n_sims=5).tune(p)
    bump = extract_features(problem_from_spec(PlantSpec("fopdt", K=2.0, tau1=5.0, theta=1.0, y_ref=1.0, r_auth=5.0))).cost_evals
    assert stub.success and stub.evals == bump + 5

    env = TuningEnv(bundle, n_sims=4, seed=0)
    model = PPO("MlpPolicy", env, n_steps=32, batch_size=16, n_epochs=2, seed=0, verbose=0)
    model.learn(total_timesteps=64)
    real = RLTuner(model, z0, lo, hi, n_sims=4).tune(problem_from_spec(PlantSpec("sopdt", K=1.5, tau1=4.0, tau2=1.0, theta=0.5, y_ref=2.0, r_auth=5.0)))
    assert real.success and real.evals >= 4 + 2
