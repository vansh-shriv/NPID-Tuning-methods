# Directory guide

Every folder and file of `npid-tuning-methods/`, with its status. **Current** = part of the finished work (the methods study, M0-M7, and the v2 re-evaluation it builds on); **Legacy** = the original Weeks 1-8 pipeline, kept as a record (known errors are listed in `progress.md`; do not cite its numbers); **Generated** = produced by a script (regenerable; see the command).

Run everything from this folder (`npid-tuning-methods/`) with `.venv\Scripts\python.exe`; scripts use paths relative to it.

*The folder was renamed from `npid-ml-tuning` to `npid-tuning-methods` after the study was finished (no code, data or result refers to the folder name; the git remote `origin` is unchanged). Older notes in `progress.md` or the legacy reports may use the old name.*

## Where to start
| You want | Read / run |
|---|---|
| The results of the 18-method study | `report/methods_report.md` |
| Details, ablations, validation of one phase | `report/methods/M<N>_*.md` (M1-M6) |
| The experimental protocol every method was held to | `docs/PROTOCOL.md` |
| What was planned and why | `docs/METHODS_ROADMAP.md`, `docs/ROADMAP_v2.md` |
| What happened in what order, with every bug and finding | `progress.md` |
| To re-run the whole benchmark of the 18 methods | `python run_methods.py` (about 1 h) |
| To re-run the v2 comparison (L-PID vs N-PID, ZN, ML, drift) | `python run_v2.py` |
| To check that everything works | `pytest` (225 tests, 2-5 min) |

## Top level
| Path | Status | Content |
|---|---|---|
| `README.md` | current | project overview, headline findings, setup |
| `CLAUDE.md` | current | guidance for Claude Code: commands, architecture, gotchas |
| `progress.md` | current | chronological engineering log (v2 phases 0-7, methods phases M0-M7) |
| `requirements.txt`, `requirements-dev.txt`, `pytest.ini` | current | dependencies, test config |
| `run_v2.py` | current | regenerates the v2 results (`--with-dataset`, `--only 01 05`) |
| `run_methods.py` | current | regenerates the methods study (default M7 benchmark; `--all`; `--only m3 m7-run`) |
| `.gitignore` | current | ignores `.venv`, caches, spec files, generated logs |
| `.venv/`, `.git/` | environment, history | the bundled virtual environment; git history of the original pipeline only (see `CLAUDE.md`) |

## `src/` - library
| Path | Content |
|---|---|
| `plant.py` | `Plant` interface; DC motor, drifting motor, transfer-function, state-space, ODE, dead-time plants |
| `simulate.py`, `controllers.py`, `metrics.py` | closed-loop engine; `LPID` / `NPID`; rise time, overshoot, settling, IAE, ISE, ITAE |
| `tuning_zn.py`, `tuning_relay.py`, `tuning_generic.py` | v2 helpers: ultimate-gain detector, relay auto-tuner, PSO for any plant (used by the v2 experiments) |
| `plant_family.py` | M4: randomized plant family (6 structures) and `problem_from_spec` |
| `benchmark_plants.py` | the 5 benchmark plants and their scenarios |
| `tuners/` | **the methods study**: `base` (interface, counting), `identify`, `classical` (methods 1-5), `selftuning`, `modelmatching` (6-7), `optimizers` (8-11, plus random search and the simulation-budget `Objective` wrapper), `ml` (12, 14, 15), `rl` (13), `features`, `hybrid` (16-17), `adaptive` (18), `benchmark`, `plots`, `reference` (best-known optimum), `pso_ref` |
| `legacy/` | original modules: `generate_dataset`, `ml_models`, `tuning_pso`, `rl_env`, `rl_drift`, `rl_drift_experiment`, `rl_diagnostics`, `train_rl`, `week8_generalization` (still imported by `experiments/v2/06_drift_baselines.py` and two tests) |

## `experiments/`
| Path | Content (output) |
|---|---|
| `v2/01..07_*.py` | v2 re-evaluation: fair comparison, ZN validation, PSO budget, dataset v2, ML models, multi-plant, drift baselines, figures (`results/v2/`) |
| `m1_classical.py` ... `m7_benchmark.py` | one script per phase M1-M7 (`results/methods/M<N>/`); stages as arguments (`run`, `analyze`, ...) - see each file's docstring |
| `m5_common.py` | helpers shared by M5-M7 (data loading, closed-loop regret) |
| `m6_pool.py`, `m7_analyze.py` | pooled M6 analysis and example trajectories; M7 tables and figures |

## `tests/`
`test_*.py` (225 tests): plants, controllers, metrics, simulation, ZN/relay/PSO, dataset resume (legacy generator), and one file per method family (`test_tuners_*.py`), plant family, features, adaptive machinery. `conftest.py` adds the repo root to `sys.path`.

## `docs/`
`PROTOCOL.md` (protocol and its change log), `METHODS_ROADMAP.md` (plan M0-M7), `ROADMAP_v2.md` (plan of the v2 re-evaluation), this guide.

## `report/`
| Path | Status | Content |
|---|---|---|
| `methods_report.md` | **final report** | 18 methods, benchmark, recommendations, failure modes |
| `methods/M1..M6_*.md` | current | per-phase reports (M7 is `methods_report.md`) |
| `final_report_v2.md` | current | conclusions of the v2 re-evaluation |
| `legacy/` | legacy | the original reports (`final_report.md`, `phase5_rl_report.md`), superseded by the v2 report |

## `results/`
| Path | Status | Content |
|---|---|---|
| `methods/M1..M7/` | current | CSVs and `figures/` of each phase (`M7/` = the cross-method benchmark: `M7_benchmark.csv` raw rows, `M7_aggregate.csv`, ratio/rank/applicability tables, heat map, Pareto, robustness) |
| `methods/M6/dev/` | current (provenance) | M6 development runs: detune sweep on validation plants and the first test run before the safeguards (quoted in the M6 report, section 4.3) |
| `v2/` | current | CSVs and figures of the v2 experiments |
| `logs/` | generated | console logs of the long runs (`methods_m<N>*.log`) |
| `legacy/figures`, `legacy/tables`, `legacy/tb_logs` | legacy | original figures, tables, TD3 training logs |

## `data/` and `models/`
| Path | Status | Regenerate with |
|---|---|---|
| `data/m4/dataset.jsonl`, `features_noisy.npz` | generated, **needed** by M4-M7 | `python experiments/m4a_dataset.py` (68 min); noisy features: `m4b_supervised.py noiserobust` |
| `data/v2/` (`dataset_npid.csv`, `generation.log`) | generated | `python run_v2.py --with-dataset` (about 1 h) |
| `data/legacy/` | legacy | `python -m src.legacy.generate_dataset` (the rows are known to be partly invalid) |
| `models/m4/ppo_seed{0,1,2}.zip` | generated, **needed** by method 13 | `python experiments/m4d_rl.py train` (83 min per seed) |
| `models/m7/prepared.pkl` | generated | `python experiments/m7_benchmark.py prepare` (2 min) |
| `models/v2/gain_predictor_v2.joblib` | generated | `python experiments/v2/04_ml_models.py` |
| `models/legacy/` | legacy | original TD3 agents (`td3_npid_*.zip`, `best_model.zip`); only the drift agent is used by `experiments/v2/06_drift_baselines.py` |

## `legacy/`
Original runners `run_week1.py, run_week4.py ... run_week8.py`, `eval_week6_hires.py` (there is no week 2-3 runner; those stages live in `src/legacy/`). Run from the repo root: `python legacy/run_week1.py`.

## What the clean-up of this folder changed
(One-off reorganisation; code and results are unchanged, only locations and the paths inside scripts and docs.)

| Old location | New location |
|---|---|
| `run_week1..8.py`, `eval_week6_hires.py` | `legacy/` |
| `src/{generate_dataset,ml_models,tuning_pso,rl_env,rl_drift,rl_drift_experiment,rl_diagnostics,train_rl,week8_generalization}.py` | `src/legacy/` (imports: `src.legacy.<name>`) |
| `experiments/01..07_*.py` | `experiments/v2/` |
| `ROADMAP.md`, `METHODS_ROADMAP.md` | `docs/ROADMAP_v2.md`, `docs/METHODS_ROADMAP.md` |
| `report/final_report.md`, `report/phase5_rl_report.md` | `report/legacy/` |
| `results/figures`, `results/tables`, `tb_logs/` | `results/legacy/figures`, `results/legacy/tables`, `results/legacy/tb_logs` |
| `results/methods_*.log` | `results/logs/` (an empty log and two `.done` markers were deleted) |
| `data/tuning_dataset.csv`, `data/test_tuning_dataset.csv` | `data/legacy/` |
| `data/v2_gen.log` | `data/v2/generation.log` |
| `models/td3_npid_*.zip`, `models/best_model.zip` | `models/legacy/` |
| `models/gain_predictor_v2.joblib` | `models/v2/` |
| `results/methods/M6/M6_*val_d*.csv`, `*_val_dev.csv`, `M6_{runs,ratios}_test.csv` | `results/methods/M6/dev/` |
| deleted (regenerable): `report.log` (pyswarms log written to the working directory), `__pycache__/`, `.pytest_cache/`, `results/methods/M7/M7_benchmark_partial.csv` | - |
