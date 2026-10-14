# Roadmap: Improving the N-PID ML-Tuning Project

Goal: scientific rigor first, then polish, then performance. Re-running the dataset generation and RL training is approved wherever a phase changes results.

**Structure decision:** `src/` stays as the core library (it is reasonably clean and now plant-agnostic). The legacy `legacy/run_week1.py`–`legacy/run_week8.py` scripts stay untouched as a record of the original work. New experiments go in `experiments/` (one script per question, writing to `results/v2/`) so old and new results never get mixed.

Status of each phase is tracked in `progress.md`. **All phases 0-7 are complete** (see the notes marked *Done* below for deviations from the original plan).

---

## Phase 0: Plant-agnostic foundation (DONE)
- `Plant` interface; `DCMotorParams`, `TransferFunctionPlant`, `StateSpacePlant`, `FunctionPlant`.
- `run_simulation` takes any plant; DC-motor results are bit-identical to before.

## Phase 1: Fair comparison (rigor, highest priority) - DONE
The headline claim, "N-PID beats L-PID", is not currently tested like for like.
1. Add a PSO-tuned **L-PID** baseline (same bounds, budget and ITAE objective as N-PID).
2. Add a **derivative low-pass filter** option to both controllers, so L-PID's noise collapse is not an artifact of an unfiltered derivative. Report results with and without it.
3. Investigate why N-PID (ZN) has ITAE 2.27 vs L-PID (ZN) 0.16 without noise. Check whether it is the α(e) shaping (which attenuates the proportional term for large e) rather than the tuning.
4. Include n, m, p in the PSO search space (spec stretch goal) and report the effect.
5. Multi-seed noise runs with mean ± std, not one noise realization.
- Output: `experiments/v2/01_fair_comparison.py`, a 2x2 table (L-PID/N-PID × ZN/PSO) per scenario.

## Phase 2: Tuning-method accuracy - DONE
1. Rework the Ziegler-Nichols detector. It returned Kcr=52 vs a theoretical 60 on a known third-order plant. Validate against plants with analytic Kcr/Tcr (e.g. 1/((s+1)(s+2)(s+3)): Kcr=60, Tcr=2π/√11≈1.9).
2. Add a relay-feedback (Åström-Hägglund) auto-tuner as a robust alternative.
3. Convergence checks for PSO (several seeds, report variance of the optimum).

## Phase 3: Engineering hygiene - DONE
1. `tests/` with pytest: plant steady-states vs analytics, α/β/γ values, metrics on known signals, ZN on known plants, simulate regression (DC-motor reference arrays).
2. Complete `requirements.txt` (gymnasium, stable-baselines3, torch, tensorboard) and pin versions.
3. Fold the duplicated motor equations in `rl_env.py` and `rl_drift_experiment.py` into the plant interface (time-varying parameters, e.g. R(t), via a `DriftingDCMotor` plant).
4. Central config for seeds and paths; remove `sys.path.insert` hacks by making `src` an installable package.
5. Make `generate_dataset.py` resumable (checkpoint rows to disk).

## Phase 4: Multi-plant evaluation (ties to the source review paper) - DONE
The paper surveys process control, motor drives and robotics. Show the method is not motor-specific:
- Process: first/second-order plus dead time (needs a delay element, to add to the plant interface).
- Nonlinear tank level (`FunctionPlant`, sqrt outflow).
- Electromechanical: DC motor position control, or a single-link manipulator.
For each: ZN, PSO, and N-PID vs L-PID comparison using the same metrics.

## Phase 5: Better ML (re-run approved) - DONE
1. Larger dataset (2000 rows; the spec's upper bound) and a train/validation/test split by plant configuration.
2. Cross-validation, hyperparameter search for RF/MLP; try gradient boosting; predict log-gains.
3. Label quality: multi-start PSO and keep the best, drop labels with high overshoot or non-convergence.
4. Extrapolation study: sweep J beyond the training range and report where the model breaks, not just J=0.03.
5. Add R (resistance) as a feature so the model can address drift offline.

## Phase 6: RL review - DONE
1. Re-validate the Phase 5 report claims against the numbers in `results/legacy/tables/` (frozen-ML vs adaptive-RL under drift).
2. Add a stronger non-learning baseline (e.g. gain scheduling, or an adaptive/self-tuning rule) so RL is compared against a fair adaptive alternative.
3. Multiple training seeds with confidence intervals.

## Phase 7: Final deliverable - DONE
1. Single reproducible entry point (`make`/script) that regenerates all v2 figures and tables.
2. Updated README with honest results (including negative findings) and a short methods write-up in `report/`.
3. Clean up git: commit the untracked work (`models/`, reports) deliberately, with large artifacts reviewed first.

---

## Suggested order
Phase 1 → 2 → 3 → 5 → 4 → 6 → 7. Phase 1 can change the project's headline conclusion, so it comes before anything that depends on the controller comparison.

## Deviations from the plan as executed
- Phase 2: PSO multi-seed variance was done in Phase 5 (`03a`); single-seed PSO was found unreliable, so `tune_pso(restarts=...)` was added.
- Phase 3: `src` was not made pip-installable (pytest uses a conftest path shim).
- Phase 4: manipulator = single link with gravity; process = FOPDT and nonlinear tank (via the new `DelayedPlant`).
- Phase 5: model features include armature R; labels filtered for box-edge solutions; 'constant gains' baseline added.
- Phase 6: RL was **not retrained** (the drift scenario has no headroom; see progress.md). A simple estimator-based gain-scheduling baseline replaced it as the adaptive reference.
- Phase 7: nothing was committed to git; that is left for the owner to review (large untracked artifacts: `models/`, `data/`).
