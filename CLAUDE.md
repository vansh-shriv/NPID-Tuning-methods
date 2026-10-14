# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Critical re-implementation of Çelik et al. (2026), "Advancements in nonlinear PID controllers: A comprehensive review": L-PID vs N-PID (paper Eq. 9-12), ZN/PSO tuning, ML gain prediction, RL adaptation. Original specs: `../NPID_Project_Spec.md`, `../Phase5_RL_Spec.md` (git-ignored). **Current work is the v2 re-evaluation** tracked in `docs/ROADMAP_v2.md` (plan), `progress.md` (engineering log with every finding) and `report/final_report_v2.md` (conclusions). The original Weeks 1-8 pipeline (`legacy/`, `src/legacy/`, `report/legacy/`, `results/legacy/`, `data/legacy/`, `models/legacy/`) is kept as a record and contains known errors listed in `progress.md`; do not cite its numbers. **`docs/DIRECTORY_GUIDE.md` maps every folder and says what is current, legacy or generated.**

The parent `E:\College_Intern` also holds unrelated sibling projects (`ml-pid-tuning/`, `Algorithm_Simulation/`, other `*_Spec.md`); this repo is only `npid-tuning-methods/`. (Formerly named `npid-ml-tuning`; the rename changed no code or result.)

## Commands

Run from the repo root with the bundled venv (`.venv\Scripts\python.exe`). Scripts use relative paths (`results/`, `data/`, `models/`).

```bash
pytest                              # 225 tests, 2-5 min (tests/ has a conftest path shim; no install needed)
pytest tests/test_tuning.py -k zn   # single test selection
python run_v2.py [--with-dataset] [--only 01 05]   # regenerate v2 results; dataset generation (~1 h) is opt-in
python experiments/v2/0X_*.py       # run one v2 experiment (01 fair comparison, 02 ZN validation, 03a PSO budget,
                                    #   03 dataset v2, 04 ML models, 05 multi-plant, 06 drift baselines, 07 figures)
python legacy/run_week1.py ... legacy/run_week8.py   # legacy pipeline (run from the repo root)
python run_methods.py [--all | --only m3 m7]       # methods study (docs/METHODS_ROADMAP.md): default = M7 benchmark
```

Windows notes: console is cp1252 (no non-ASCII in `print`); `multiprocessing` needs a real script file with `if __name__ == "__main__"` (does not work from `python -` stdin); very long multi-heredoc shell commands can fail, write files with the editor tool instead. Dependencies: `requirements.txt` (+ `requirements-dev.txt` for pytest).

## Architecture

- **`src/plant.py`** - `Plant` ABC (`state_dim`, `initial_state`, `derivatives`, `output`, `step`, `u_min/u_max`, `Vmax` alias). Implementations: `DCMotorParams` (anchor plant; hand-inlined fast RK4 in `step`/`_rk4(R, ...)`), `DriftingDCMotor` (R(t) = R(1 + k_heat t)), `TransferFunctionPlant`, `StateSpacePlant`, `FunctionPlant` (any ODE), `DelayedPlant` (input dead time; stateful FIFO reset by `initial_state()`). New plants only need `derivatives` + `output`.
- **`src/simulate.py`** - `run_simulation(controller, plant, y_ref, t_end, dt, T_load_func, noise_sigma, rng)`: the single closed-loop engine for every tuner, metric and experiment; the plant owns its integrator (`plant.step`). Noise corrupts only the controller's measurement; logged error and metrics use the true output. The first positional field names (`omega`, `current`, `voltage`) are generic: output, `state[0]`, control input.
- **`src/controllers.py`** - `LPID`, `NPID` with `.compute(error)` / `.reset()`. Both: back-calculation anti-windup, optional derivative low-pass `tau_d`. `NPID` uses `P = Kp*alpha(e)` (NOT the literal `Kp*alpha(e)*e` of paper Eq. 9, which loses the error sign), plus optional `e_scale`/`de_scale` for scale-free shaping (1.0 = paper-literal).
- **`src/metrics.py`** - `compute_metrics`: rise time, overshoot, settling time, IAE, ISE, ITAE. All comparisons go through it.
- **Tuning:** `tuning_zn.py` (`find_critical_gain(mode="auto|linear|limit_cycle")`, half-cycle amplitude-ratio detector; `auto` warns and falls back to a limit-cycle detector when no linear ultimate gain exists, as for the DC motor), `tuning_relay.py` (relay feedback), `tuning_generic.py` (`tune_pso(kind, plant, ...)` for LPID/NPID on any plant: explicit bounds, `log_space`, `restarts` (best of N seeds), noise-aware fitness, optional n,m,p). Legacy: `src/legacy/tuning_pso.py` (N-PID, DC motor only).
- **ML/dataset:** v2 in `experiments/v2/03_generate_dataset_v2.py` (setpoint = fraction of the plant's w_max, resumable JSONL checkpoint, seeded per trial) and `experiments/v2/04_ml_models.py` (saves `models/v2/gain_predictor_v2.joblib`, ridge poly-2, 6 features incl. R, predicts log10 gains). Legacy: `src/legacy/generate_dataset.py`, `src/legacy/ml_models.py`.
- **RL (legacy, Phase 5)**, all in `src/legacy/`: `rl_env.py` (`NPIDTuningEnv`; keeps its own copy of the control law deliberately, so trained models in `models/legacy/` stay valid; plant stepping now goes through `plant.step`), `train_rl.py`, `rl_diagnostics.py`, `rl_drift.py`, `rl_drift_experiment.py` (also used by `experiments/v2/06_drift_baselines.py`), `week8_generalization.py`.

## Methods study (second round) - read `docs/METHODS_ROADMAP.md`, `docs/PROTOCOL.md`

- `src/tuners/base.py`: `TuningProblem` (the only plant handle a method gets; counts experiments: `evals`, `sim_seconds`), `Tuner.tune(problem, budget, seed) -> TuningResult`, `NotApplicable` (a method must raise it instead of returning unreliable gains), `default_bounds` (wide log-space gain box; protocol change P1), `OpenLoopRun`.
- `src/tuners/identify.py`: `identify_process` (operating-point search, bump test, FOPDT/SOPDT/IPDT fits). `src/tuners/classical.py`: M1 methods. `src/tuners/benchmark.py`: `evaluate_gains` (uncounted referee) and `run_benchmark`. `src/tuners/plots.py`: shared figures. `src/benchmark_plants.py`: the 5 benchmark scenarios.
- Per-phase experiment scripts `experiments/m<N>_*.py` write to `results/methods/M<N>/`; per-phase reports in `report/methods/`.
- **M2** (`src/tuners/selftuning.py`: RLS + exact PID/PI pole placement + sampled-data auto-tuner; `modelmatching.py`), **M3** (`optimizers.py`: PSO/GA/ABC/SA/random from scratch on the unit cube with a simulation-budget `Objective`; `reference.py`: best-known optimum), **M4**: `src/plant_family.py` (randomised plant structures, exact LTI integration), `src/tuners/features.py` (bump-test features; gains normalised by K_eff, T_n), `ml.py` (Bundle, `noisy_bundle`, regressors, `ANNPolicy`, `BayesOptTuner`), `rl.py` (`TuningEnv`, `RLTuner`, baselines). Data/model artefacts: `data/m4/dataset.jsonl` (68 min to regenerate), `models/m4/ppo_seed*.zip` (83 min/seed); results in `results/methods/M<N>/`, reports `report/methods/M<N>_*.md`.
- **M5** (`src/tuners/hybrid.py`): GA-selected ML (genome -> model family/hyper-parameters/feature mask, fitness = CV closed-loop regret) and `ClassicalMLTuner` (Cohen-Coon base in z-space + SVR residual + optional 20-sim `LocalRefiner`). Experiments `m5a_ga_ml.py run|analyze`, `m5b_hybrid.py run|bench|analyze` (share `m5_common.py`); results `results/methods/M5/`, report `report/methods/M5_hybrid.md`. Tuners passed to `multiprocessing` must be picklable (no lambdas as attributes).
- **M6** (`src/tuners/adaptive.py`): online loop with `SwitchingPlant` (plant change mid-run), `OnlineEstimator` (gated RLS, level form), policies (`FrozenPolicy`, `ScheduledPolicy`, `AdaptivePolicy('ml'|'pp')`), `run_scenario`. Experiments `m6_adaptive.py run <split> <n> <tag> [offset] | analyze <tag>` and `m6_pool.py`; report `report/methods/M6_adaptive.md`. Estimator-side ML must use the clean-feature SVR (`ClippedModel`), the bump-test side the noisy one.
- **M7** (`experiments/m7_benchmark.py prepare|run|analyze`, `m7_analyze.py`, `run_methods.py`): the cross-method benchmark (20 tuners x 5 plants x 4 scenarios incl. an actuator-gain change); final report `report/methods_report.md`, results `results/methods/M7/`. `prepare` trains the ML models once (`models/m7/prepared.pkl`); worker processes load them via the pool initializer.
- **Always evaluate ML tuners with features from a NOISY bump test** (`M4_FEATURES=noisy`, the default in `m4b_supervised.py`): models trained on noise-free features collapse (median regret 80) on noisy ones.
- Experiment scripts for M2-M4 take a stage argument (`run`/`analyze`, `compare`/`ablate`/`final`/`noisefeat`/`noiserobust`, `train`/`eval`/`analyze`); long jobs are resumable only for the dataset generators. Use single-threaded torch (`torch.set_num_threads(1)`) when several learners run in parallel.
- Rules for new methods: no plant constants, all plant access through the problem object, validate against known answers in `tests/` before benchmarking, document pros/cons/errors in the phase report.

## Gotchas

- `DCMotorParams` defaults use **Ke = Kt = 0.5**, not the spec's 0.01. With Vmax = 24 V this limits the maximum speed to ~26-40 rad/s; setpoints must stay below it (the legacy dataset did not, v2 does).
- ZN is not a principled baseline for the DC motor (no true ultimate gain); treat it as a PSO box centre / "limit-cycle heuristic".
- Single-seed PSO can converge to a bad solution; evaluation tables use `restarts=3`.
- Controller `dt` must equal the simulation `dt`. Use the same `u_min/u_max` as the plant (`make_controller` does).
- `models/legacy/npid_rf_model.pkl` (referenced by legacy drift code) does not exist; the legacy "frozen ML" drift baseline silently fell back to PSO gains. Use the v2 model.
- `git`: only the original (legacy) pipeline and a few docs were ever committed; `src/tuners/`, `experiments/`, `tests/`, `docs/`, `report/methods*`, `results/methods|v2`, `data/`, `models/` (except legacy files) are untracked. Generated/large artefacts: `data/m4/` (68 min to regenerate), `models/m4/` (PPO, 83 min per seed), `models/m7/` (`m7_benchmark.py prepare`).
