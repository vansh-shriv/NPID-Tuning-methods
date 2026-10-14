# Progress Log

Plan: see `docs/ROADMAP_v2.md`. Newest entries at the bottom of each section. Legacy `run_week*.py` scripts and their results are the original work and are not modified.

## Status

| Phase | Description | Status |
|---|---|---|
| 0 | Plant-agnostic foundation | Done |
| 1 | Fair comparison (PSO L-PID, derivative filter, multi-seed) | Done (see findings; shape-scale design open) |
| 2 | ZN accuracy + relay tuner | Done |
| 3 | Tests, requirements, dedupe dynamics | Done |
| 4 | Multi-plant evaluation | Done |
| 5 | Better ML (re-run) | Done |
| 6 | RL review | Done (RL not retrained) |
| 7 | Final deliverable | Done |

## Log

### Phase 0: Plant-agnostic foundation
**Why:** `plant.py` was hard-wired to the DC motor, and the motor equations were pasted into `simulate.py`, so no other plant could be simulated, ZN-tuned or PSO-tuned.

**What changed**
- `src/plant.py`: new abstract `Plant` (`state_dim`, `initial_state`, `derivatives`, `output`, `step`, `u_min/u_max`). `DCMotorParams` now subclasses it (fields and `dc_motor_dynamics` unchanged, `Vmax` still works) and keeps its fast hand-inlined RK4 in `step`. New plants: `TransferFunctionPlant(num, den)`, `StateSpacePlant(A, B, C, D)`, `FunctionPlant(f, x0)`.
- `src/simulate.py`: `run_simulation(controller, plant, ...)` is plant-agnostic and calls `plant.step()`; the inlined motor equations moved into `DCMotorParams.step`.

**Verification**
- Noisy closed-loop run with a load step, L-PID and N-PID: max abs difference vs pre-change code = 0.0.
- `python -m src.plant`: motor @12 V settles at 17.0 rad/s; 2/(s^2+3s+2) step reaches 0.9993.
- ZN + L-PID on a third-order TF plant and a PID on a nonlinear tank plant both run end to end.
- `tuning_pso`, `ml_models`, `rl_env`, `generate_dataset`, `rl_drift_experiment` still import.
- Not re-run: `run_week*.py` runners.

**Findings to follow up**
- ZN found Kcr=52 vs analytic 60 for 1/((s+1)(s+2)(s+3)): detector is coarse (Phase 2).
- `rl_env.py` and `rl_drift_experiment.py` still duplicate the motor equations (Phase 3).
- No dead-time support in the plant interface yet (Phase 4).
- Console is cp1252: avoid non-ASCII characters in `print` output (some legacy scripts print `ω`, which fails on this console).

### Phase 1: Fair comparison
**Why:** The README claimed N-PID beats L-PID, but only N-PID was PSO-tuned, L-PID's derivative was unfiltered, and noise was a single realization.

**What changed**
- `src/controllers.py`: optional derivative low-pass `tau_d` on `LPID`/`NPID`; `NPID` gets `e_scale`/`de_scale` so α, β, γ act at a chosen error/rate scale (defaults 1.0 = paper-literal; old results reproduce to 1e-12).
- `src/tuning_generic.py` (new): one PSO tuner for either controller on any plant (same box 0.1x-3x of ZN, 20 particles x 40 iters, seeded), optional shape tuning (n, m, p) and noise-aware fitness (mean ITAE over noise seeds, disjoint from evaluation seeds).
- `experiments/v2/01_fair_comparison.py` (new): 21 variants x {clean step, load step, noise sigma=0.3 averaged over 10 seeds}; output `results/v2/01_fair_comparison.csv`. Runtime ~5 min (parallel).

**Results (DC motor, w_ref=20; ITAE, lower is better)**

| Controller / tuning | clean step | noisy (10 seeds) |
|---|---|---|
| L-PID ZN | 0.163 | 121.5 |
| L-PID PSO (clean-trained) | **0.160** | 126.3 |
| N-PID(paper-literal) ZN | 2.272 | 2.58 |
| N-PID(paper-literal) PSO | 0.353 | 2.59 |
| N-PID(paper-literal) PSO + n,m,p | 0.238 | 2.16 |
| N-PID(scaled) PSO | 0.163 | 2.42 |
| L-PID PSO, noise-aware, 10 ms filter | 10.61 | 3.38 |
| N-PID(paper-literal) PSO, noise-aware, 10 ms filter | 1.55 | 1.76 |
| N-PID(scaled) PSO, noise-aware, 10 ms filter | 1.87 | **1.71** |

*(Table updated after re-running with PSO best-of-3 restarts, see Phase 4 finding 4: all clean-trained rows moved by <= 3%; the noise-aware rows moved by up to 40% (N-PID literal clean 2.66 -> 1.55), which shows that single-seed noise-aware PSO is noisy. The single-seed results are kept in `results/v2/01_fair_comparison_singleseed.csv`. Every qualitative conclusion below is unchanged.)*

**Findings**
1. **Without noise, N-PID does not beat L-PID.** A PSO-tuned L-PID (0.160) matches the best N-PID (scaled, 0.163) and beats the paper-literal N-PID (0.24-0.35). The earlier "N-PID (ZN) 2.27 vs L-PID 0.16" gap is a tuning/scaling artifact, not a property of nonlinear control.
2. **Paper-literal shaping is not scale-free.** α, β, γ take raw e (rad/s) and de/dt (rad/s^2). At w_ref=20 with 1 ms steps, β(e)~0 for most of the transient (integral almost off) and γ(de/dt)~0 whenever noise exists (derivative almost off). The same N-PID would behave differently at w_ref=5 vs 50, which matters for the Week 3 dataset. Normalizing by e_scale/de_scale recovers L-PID-level clean performance.
3. **N-PID's noise advantage is mostly "γ switches the derivative off".** An unfiltered L-PID collapses under noise (ITAE ~121), a 10 ms filter helps (44), and a noise-aware-tuned filtered L-PID gets 3.3. N-PID still wins under noise (1.7-1.8) but the gap shrinks from ~50x to ~2x once L-PID is treated fairly. That is a real but modest advantage.
4. **Trade-off for L-PID:** noise-aware tuning ruins its clean step (ITAE 10.6), while N-PID tuned for noise keeps 1.5-1.9 clean / 1.7 noisy. So N-PID gives a better clean/noisy compromise, which is the defensible claim.
5. **Spec deviation noted:** Eq. 9 literally gives Kp*alpha(e)*e = e^2/..., which loses the sign and breaks feedback. The code uses P = Kp*alpha(e) (equivalent to alpha being the gain 1/(1+|e|^n)^(1/n) times e). Keep and document this in the report.
6. Shape tuning (n, m, p) helped paper-literal N-PID (ITAE 0.353 -> 0.238 clean).

**Caveats / next:** single plant and operating point; PSO budget small (20x40, one seed per variant, so differences of a few percent are not significant); load-step scenario not used for training. The e_scale/de_scale heuristic (e_scale=w_ref, de_scale=10*w_ref) is untuned. Legacy README tables should not be cited as-is.

### Phase 2: Tuning-method accuracy
**Why:** The ZN detector was coarse (Kcr 52 vs analytic 60) and its "sustained" label accepted slowly decaying responses.

**What changed**
- `src/tuning_zn.py`: new `oscillation_growth_ratio` (median half-cycle amplitude ratio of the extrema, independent of the unknown steady-state offset); `detect_oscillation_type` now classifies by that ratio (damped < 0.999 < sustained < 1.001 < growing). Bisection tolerance 2% -> 0.3%. New `find_critical_gain(mode="auto" | "linear" | "limit_cycle")` and `detect_limit_cycle`.
- `src/tuning_relay.py` (new): Astrom-Hagglund relay-feedback auto-tuner (`relay_autotune`).
- `experiments/v2/02_zn_validation.py` (new) -> `results/v2/02_zn_validation.csv`.

**Results (plants with analytic Kcr/Tcr)**

| plant | Kcr true | ratio detector | relay | Tcr true | ratio | relay |
|---|---|---|---|---|---|---|
| 1/((s+1)(s+2)(s+3)) | 60 | 59.75 (-0.4%) | 58.2 (-3.0%) | 1.894 | 1.898 | 1.928 |
| 1/(s(s+1)(s+5)) | 30 | 29.88 (-0.4%) | 28.5 (-4.9%) | 2.810 | 2.816 | 2.904 |
| 1/(s+1)^4 | 4 | 4.000 (0.0%) | 3.93 (-1.9%) | 6.283 | 6.284 | 6.330 |

The ratio detector is now accurate to <0.5% (was 13% off). Relay is within 5% (expected: describing-function approximation) in a single short run.

**Key finding: the DC motor has no true ultimate gain.** Its P-only response is damped at every Kp (ratio 0.81-0.98, never >= 1), because the plant is minimum-phase with relative degree 2 and cannot go unstable in continuous time. The Kcr = 232, Tcr = 0.041 s used in all legacy results (and the ZN gains Kp=139, Ki=6747, Kd=0.72) came from a chattering limit cycle caused by 1 ms sampling + stiff tanh friction + 24 V saturation (V swings +/-24 in the tail), not marginal stability. Consequences:
- ZN is not a principled baseline for this plant. It is a reasonable *search-box centre* for PSO, which is how the project uses it, but "ZN-tuned" results should be described as "limit-cycle heuristic".
- `find_critical_gain(mode="auto")` now says so with a warning and falls back to the limit-cycle detector (Kcr = 226.5, Tcr = 0.0418 s, 2.4% from legacy). Phase 1 results used the legacy detector (232); they remain valid, the difference is within the PSO box.
- Relay on the motor gives Ku ~ 11.9, Tu ~ 0.19 s with 0.2 rad/s hysteresis: a different (hysteresis-dependent) answer, so on this plant the relay result is a design choice, not a measurement of a physical property.

**Not done:** PSO multi-seed convergence variance (roadmap Phase 2 item 3) is deferred to Phase 5, where it is needed for label quality.

### Phase 3: Engineering hygiene
**What changed**
- `src/plant.py`: new `DriftingDCMotor(DCMotorParams)` with `R(t) = R*(1 + k_heat*t)`; `DCMotorParams.step` now delegates to `_rk4(R, ...)` so the drift variant reuses the same fast integrator.
- `src/legacy/rl_env.py` and `src/legacy/rl_drift_experiment.py`: the pasted RK4 motor equations are replaced by `plant.step(...)` (DriftingDCMotor when drift is on). **Verified bit-identical** (max abs difference 0.0) against seeded rollouts of the old env (drift on/off, 162 steps) and the frozen-gain drift run, so the trained TD3 models in `models/` remain valid. The controller law inside the env was deliberately left alone for the same reason.
- `src/legacy/generate_dataset.py`: results are checkpointed per trial to `<csv>.partial.jsonl`; an interrupted run resumes and skips finished trials; the checkpoint is removed once the CSV is written; rows are written in trial order.
- `tests/` (36 pytest tests, ~75 s, `pytest` from repo root): plant (analytic steady state, fast-vs-generic RK4, drift = `compute_R`, TF/state-space/nonlinear plants), controllers (alpha/beta/gamma values, anti-windup, shaping scale, derivative filter), metrics (first-order analytics), simulate (pinned DC-motor regression, noise reproducibility), tuning (ZN on analytic plants, relay, PSO seeding/bounds), dataset resume.
- `requirements.txt` now lists the RL stack; `requirements-dev.txt` adds pytest; `pytest.ini` added.

**Test-writing lessons (kept honest)**
- Four initial failures were test errors (saturated outputs hiding the behaviour under test, an anti-windup assertion that misread back-calculation); one was a real bug in my own checkpoint patch (checkpoint never deleted), now fixed and covered by the resume test.
- Note on `LPID` back-calculation anti-windup: while saturated it sets the integral so that P+I equals the limit, so the integral term goes strongly *negative* when the error is large and positive (e.g. -45 V at Kp=1, e=50). It works (reproduces the legacy results) but is more aggressive than a conditional-integration scheme; worth revisiting if L-PID transient behaviour matters.

**New findings for later phases**
- `models/legacy/npid_rf_model.pkl` does not exist, so the "frozen ML" baseline in `rl_drift_experiment.py` silently falls back to **PSO gains**. The Phase 5 report's "frozen ML vs RL" comparison therefore compared RL with frozen PSO gains, not an RF model (Phase 6).
- `generate_dataset.py` labels come from PSO with only **8 particles x 8 iterations**, fixed wide bounds (not ZN-centred), overshoot penalty weight 5, and no PSO seed: labels are noisy and not reproducible (Phase 5). Also all dataset plants have T_load applied at t=1.5 s in a 3 s run while ml_models later evaluates with t_end=4 s.
- Not done: making `src` pip-installable (the `conftest.py` path shim and the legacy `sys.path.insert` lines remain). Low value for now.

### Phase 5: Better ML (dataset regenerated)
**Audit of the legacy dataset (500 rows) - why it was regenerated**
- 211 of 500 rows (42%) request a setpoint above 95% of the motor's maximum speed at 24 V (w_max = 26-40 rad/s with R=1, Ke=0.5, versus a sampled range of 5-50 rad/s). The setpoint is unreachable, ITAE does not depend on the gains, and PSO returns arbitrary gains. Legacy R^2/regret results are built on these labels.
- Labels came from PSO with 8 particles x 8 iterations, no seed, a wide linear box. Rows were written only at the end.

**Dataset v2** (`experiments/v2/03_generate_dataset_v2.py`, `data/v2/dataset_npid.csv`, 1200 rows, 57 min on 14 workers)
- Setpoint sampled as 15-85% of each plant's own w_max (feasible by construction); armature R randomized (0.5-2.0 Ohm) as a sixth feature; PSO 16x25 on log10 gains in a fixed box, seeded per trial, scale-free N-PID shaping, load step at 1.5 s; resumable.
- PSO budget from `03a_pso_convergence.py`: ITAE is repeatable across seeds (CV <= 1.3% at 16x25) but the *gains* are not (CV 12-150%): many gain triples are equally good, so the labels are weakly identifiable. Single seeds can also fail outright (see Phase 4); the dataset labels use one seed per trial, so a small fraction of labels may be poor (the quality filter removes the worst).
- Quality filter used for ML: drop rows where a gain sits at the search-box edge (16.8%) or overshoot > 15% -> 985 rows.

**Results** (`experiments/v2/04_ml_models.py`, 788 train / 197 held-out plants; regret = ITAE with predicted gains / ITAE with PSO gains)

| model | R2 Kp / Ki / Kd (log gains) | median regret | p90 regret | within 1.5x |
|---|---|---|---|---|
| constant (median gains) | 0.00 / -0.02 / -0.02 | 1.026 | 1.190 | 99.0% |
| ridge poly-2 | 0.78 / 0.76 / 0.23 | **1.000** | **1.017** | 99.5% |
| kNN | 0.70 / 0.68 / 0.16 | 1.006 | 1.092 | 99.5% |
| random forest | 0.73 / 0.74 / 0.25 | 1.003 | 1.073 | 100% |
| extra trees | 0.75 / 0.75 / 0.22 | 1.001 | 1.061 | 100% |
| hist gradient boosting | 0.72 / 0.71 / -0.02 | 1.002 | 1.054 | 100% |
| MLP (64,32) | 0.73 / 0.73 / 0.07 | 1.003 | 1.081 | 99.5% |

**Findings**
1. **A single fixed gain set is already within 3% (median) of per-plant PSO** on this plant family. ML improves the worst cases (p90 1.19 -> 1.02) but the problem is easy: the cost landscape is flat. The legacy claim "ML generalizes extremely well" is true but uninformative; the constant baseline was missing.
2. A simple polynomial ridge regression matches or beats RF/GB/MLP (smooth, low-dimensional mapping, ~1000 rows).
3. Kd is poorly predictable (R2 0.0-0.25); Kp/Ki ~0.75. Consistent with weak identifiability of the labels.
4. Permutation importance (RF): w_ref 0.91, B 0.53, J 0.22, R 0.07; Tc and T_load ~0. Plausible for the speed loop; the integral action rejects the load disturbance whatever the gains.
5. Speed: fresh PSO 25.5 s per plant vs 0.99 ms for `predict` (about 25,000x). The legacy "1887x" used a 30 s PSO and a 16 ms prediction.
6. Extrapolation (plant at 50% of w_max, regret vs fresh PSO): J up to 0.03 (1.5x the training maximum) 1.05; J = 0.05 and 0.08: 2.6-2.8x. B up to 0.5, Tc up to 0.2 and R up to 4 (2x training max): <= 1.08x. Only inertia breaks the model; the legacy report tested only J = 0.03, the benign case.
7. The saved model is `models/v2/gain_predictor_v2.joblib` (ridge poly-2, 6 features). The legacy `models/legacy/npid_rf_model.pkl` still does not exist.

### Phase 4: Multi-plant evaluation
`experiments/v2/05_multi_plant.py` -> `results/v2/05_multi_plant.csv`. Five plants (DC speed, DC position, single-link arm with gravity, first-order-plus-dead-time, nonlinear tank); gain boxes derived generically from actuator range, setpoint and time scale; PSO = best of 3 restarts; filtered derivative (tau_d = Ts/20) vs raw; ITAE on clean step / disturbance step / noisy feedback (10 seeds). New `DelayedPlant` for dead time (tests added); `tuning_zn` now respects the plant's own `u_min/u_max`.

ITAE on the clean step (overshoot %):

| plant | L-PID ZN | L-PID PSO | N-PID literal PSO | N-PID scaled PSO |
|---|---|---|---|---|
| dc_speed | 0.172 | 0.187-0.209 | 0.424 | 0.169-0.173 |
| dc_position | 0.280 (63%) | 0.035-0.051 | 0.054-0.060 | **0.031-0.038** |
| link (gravity) | 0.085 (26%) | **0.012-0.016** | 0.069-0.074 (43%) | 0.035-0.051 (40%) |
| fopdt (dead time) | 6.26 (54%) | **1.85-2.69** | 2.17-2.45 | 3.05-3.10 |
| tank (nonlinear) | 18.9 | 2.57-2.94 | **1.21-1.40** | 1.32-1.35 |

Noisy feedback: L-PID with a raw derivative collapses on 4 of 5 plants (dc_speed 84, tank 69, link 7.4, dc_position 1.9). With a derivative filter it is competitive: link 0.023 (best of all), fopdt 6.9 (tie with N-PID literal filtered), dc_position 0.052 (N-PID scaled 0.041), but still poor on dc_speed (46.7 vs 0.9-1.2) and tied on the tank (12.1 vs 12.4-14.4).

**Findings**
1. **No universal winner.** N-PID is clearly better on the tank (state-dependent gain: about half the ITAE) and under noise on the DC speed loop; L-PID is better on the gravity-loaded link (N-PID shows ~40% overshoot there; a plausible but untested explanation is that beta(e) ~ 0 during the large initial error keeps the integral from pre-loading the gravity torque) and on the dead-time plant.
2. Scale-free shaping beats paper-literal shaping on 3 of 5 plants (dc_speed, dc_position, link) and loses on the dead-time plant; the shaping scales are a design variable the review does not discuss.
3. ZN exists as a linear ultimate gain for dc_position, fopdt and tank; the DC speed loop and the link only produce a limit-cycle artefact (Phase 2). ZN is a poor baseline on most plants (63% overshoot on dc_position); PSO is far better on every plant except dc_speed, where all are close.
4. **Single-seed PSO is unreliable**: L-PID on the dead-time plant converged to ITAE 954 on seed 0 and 1.7-4.4 on seeds 1-5. Experiments 01 and 05 therefore use best-of-3 restarts. With one seed the fopdt row looked like a huge N-PID win (2.8 vs 954); with restarts it is a loss (2.2-3.1 vs 1.85).

### Phase 6: RL review
`experiments/v2/06_drift_baselines.py` -> `results/v2/06_drift_baselines.csv`. Drift R(t) = R0(1 + k_heat t) over 20 s at w_ref = 20, all on `DriftingDCMotor`. Controllers: frozen PSO, frozen ML (the v2 model, which actually exists), ML re-queried every 50 ms with the true R (oracle) and with an online estimate of R from V = R i + Ke w, and the Phase-5 TD3 agent.

Total ITAE over 20 s:

| k_heat (R_end) | frozen PSO | frozen ML | ML + oracle R | ML + estimated R | RL (TD3, drift) |
|---|---|---|---|---|---|
| 0.02 (1.4 Ohm) | 0.164 | 0.166 | 0.167 | 0.185 | 1179 |
| 0.05 (2.0) | 0.169 | 0.173 | 0.174 | 0.192 | 730 |
| 0.10 (3.0) | 0.176 | 0.186 | 0.181 | 0.200 | 574 |
| 0.30 (7.0) | 827 | 827 | 827 | 827 | 893 |

**Findings**
1. **The drift scenario cannot discriminate between controllers.** Tripling R changes the frozen controller's ITAE by 7% (0.164 -> 0.176); the integral action absorbs it, so no adaptive scheme (not even an oracle that knows R) has anything to gain. At k_heat = 0.3, R reaches 7 Ohm and w_max < 20 rad/s: the setpoint is physically unreachable and all controllers fail identically.
2. The Phase-5 TD3 agent has ITAE 574-1179 vs 0.17: three to four orders of magnitude worse than doing nothing, matching (and strengthening) the old report's negative result. The RL-vs-baseline error comparison interpolates the agent's 50 ms log onto the 1 ms grid, so it is not exactly like-for-like with the 1 ms controllers; the gap is far too large for that to matter.
3. The old master table (`master_5way_comparison.csv`) reports the identical drift ITAE (0.3999) for L-PID ZN, N-PID ZN, N-PID PSO and N-PID ML: it never evaluated each controller under drift, and the "frozen ML" in the drift experiment was PSO gains because the RF pickle does not exist. Both are corrected in experiment 06 and the old numbers should not be cited.
4. RL was **not retrained**: in this plant there is no regime where adaptation has headroom, so more training would not answer a meaningful question. A fair RL study needs a scenario where frozen gains degrade (e.g. a step change in inertia or friction, or operation near the voltage limit) and must be compared against the cheap estimator-based baseline from experiment 06.

### Phase 7: Final deliverable
- `run_v2.py`: one entry point (all experiments in dependency order; dataset generation opt-in via `--with-dataset`).
- `experiments/v2/07_make_figures.py` -> `results/v2/figures/fig1..fig4` (fair comparison, multi-plant heat map, ML regret + extrapolation, drift).
- `report/final_report_v2.md`: conclusions, limitations, reproduction, next steps. `README.md` rewritten around the v2 findings (legacy tables removed); `CLAUDE.md` updated.
- Re-run of experiment 01 with best-of-3 PSO restarts: conclusions unchanged (table in Phase 1 updated; single-seed CSV kept).
- Tests: 38 passing (`pytest`).
- Not done on purpose: committing to git (untracked: `models/`, `data/`, `results/v2/`, `experiments/`, `tests/`, docs), pip-installable package, retraining RL.

**Open items for the next round (kept general, per the plan to extend the review's methods later)**
1. Add the other N-PID structures surveyed in the paper behind the `compute(error)` interface; experiments 01/05 rerun unchanged.
2. Plant-agnostic ML features (step-response signatures instead of motor parameters).
3. Actuator/sensor non-idealities as plant wrappers; a harder drift/regime-change scenario before revisiting RL.
4. Dataset labels use one PSO seed per trial; best-of-3 would cost ~3 h and may raise label quality.

---

# Methods study (second round; plan in `docs/METHODS_ROADMAP.md`, protocol in `docs/PROTOCOL.md`)

| Phase | Content | Status |
|---|---|---|
| M0 | Tuner framework, identification, benchmark runner, protocol | Done |
| M1 | Classical: trial and error, ZN (reaction curve), Cohen-Coon, continuous cycling, time-domain (SIMC, CHR) | Done, report `report/methods/M1_classical.md` |
| M2 | Intelligent: self-tuning (pole placement), model matching | Done, report `report/methods/M2_intelligent.md` |
| M3 | Optimization: PSO, GA, ABC, SA | Done, report `report/methods/M3_optimization.md` |
| M4 | ML: regression (SVR/NN), RL, Gaussian-process optimization, ANN | Done, report `report/methods/M4_ml_based.md` |
| M5 | Hybrid: GA-enhanced ML, ML + classical | Done, report `report/methods/M5_hybrid.md` |
| M6 | Adaptive ML-based tuning | Done, report `report/methods/M6_adaptive.md` |
| M7 | Cross-method benchmark and final report | Done, report `report/methods_report.md` |

## M0 + M1 (summary; details in `report/methods/M1_classical.md`)
- New code: `src/tuners/{base,identify,benchmark,plots,classical,pso_ref}.py`, `src/benchmark_plants.py`, `experiments/m1_classical.py`, tests `tests/test_tuners_{identify,benchmark,classical}.py` (46 tests; total suite 84 passing).
- Everything a method knows about a plant is learned through counted experiments on a `TuningProblem`; methods raise `NotApplicable` rather than return unreliable gains; classical open-loop rules are guarded by their validity range (`strict=True`).
- Headline findings: best classical rule per plant is 1.5-29x worse than a 400-evaluation PSO but costs under 2% of its experiment time; clean-trained PSO is 2.7-51x worse under noise than on the clean step and loses to classical rules on the tank and DC speed under noise; 9 of 45 (variant, plant) pairs are not applicable; the DC speed loop's limit-cycle "ultimate gain" varies 64-129 with the detection settings (legacy 232); with identical gains the N-PID is not better than the L-PID on average (9 better / 10 similar / 17 worse of 36); trial and error varies up to 12x with its acceptance thresholds.
- **Protocol change P1**: the generic gain box was too narrow (Ki <= 10 kp0/Ts) and excluded the optimum on the tank; widened to 4-6 decades and M1 re-run.
- Bugs found and fixed on the way (all covered by regression tests): tangent-method smoothing, `link` mis-identified as an integrator (secant overshoot beyond the gravity limit), `tank` mis-identified (probing the actuator limit), limit-cycle horizon.
- Not yet covered in M1: identification under noise only on two plants; trial-and-error is an emulation.

## M3 (optimization) - summary; details in `report/methods/M3_optimization.md`
- New code: `src/tuners/optimizers.py` (PSO, GA, ABC, SA, random search from scratch + `Objective` budget wrapper), `src/tuners/reference.py` (best-known optimum), `experiments/m3a_best_known.py`, `experiments/m3_optimizers.py`, `tests/test_tuners_optimizers.py` (25 tests).
- Results: GA ~ PSO > SA ~ ABC > random search; 28/50 pairwise comparisons significant (PSO 10 wins, GA 9, SA 5, ABC 4, random 0); no optimizer within 5% of best-known on the link at 400 simulations; cost is repeatable but gains lie on ridges (gain CV 40-120%); with ideal tuning the N-PID is not better than the L-PID on the clean step (<= 0.5% anywhere, L-PID better by 15-31% on dc_position and dead-time); N-PID's noise advantage vanishes when the L-PID is also tuned for noise (within 10% on 4/5 plants; noise-aware L-PID unstable in 5/10 runs on the dead-time plant, N-PID 0/10); shape exponents help only on the link (2.1x); noise-aware training trades clean for noise performance.
- Caveats: 5 seeds (noisy medians), best-known is a lower bound on the optimum, default hyper-parameters (link ranking depends on them), noise-aware runs have 133 candidates.

## M2 (intelligent: self-tuning pole placement, model matching) - summary; details in `report/methods/M2_intelligent.md`
- New code: `src/tuners/selftuning.py` (RLS, delay/structure selection, exact PID/PI pole placement, sampled-data auto-tuner), `src/tuners/modelmatching.py` (frequency-domain NNLS fit, time-domain refinement), `experiments/m2_intelligent.py`, `tests/test_tuners_selftuning.py` (33), `tests/test_tuners_modelmatching.py` (11).
- Results: self-tuning works on all 5 plants from one experiment, best non-optimizing method on the link (ITAE 0.051) but conservative elsewhere (13.3 dead-time, 18.2 tank); model matching reproduces its reference (time variant: matching error 4e-4..1.6e-2) but ITAE is set by the chosen reference speed; neither approaches PSO (3.4-20x); N-PID worse than L-PID with the same gains in 9/15 cells.
- Important development finding: running the identification experiment with a fast controller and sampling slowly (inter-sample inconsistency) biased the identified DC gains by 47-78% and made ablations vary 10-50x; the sampled-data experiment fixed it. Other fixes: exact four-pole placement with derivative filter, auxiliary-pole feasibility, PI fallback, integrating structure, RLS batch initialisation.
- Limits: ARX(2,2) cannot represent the third-order integrating dc_position plant; catastrophic without delay selection on the dead-time plant (ITAE 3320) and at 2% excitation on the tank (2701).

## M4 (ML-based: regression SVR/NN, ANN, Gaussian-process optimisation, reinforcement learning) - done; details in `report/methods/M4_ml_based.md`
- New code: `src/plant_family.py` (6 plant structures, exact LTI integration), `src/tuners/features.py` (bump-test features, dimensionless gain normalisation), `src/tuners/ml.py` (bundle, noisy-feature rebasing, regressors, ANN with decision-focused loss, GP optimiser), `src/tuners/rl.py` (tuning environment, PPO session tuner, random-search and constant baselines), experiments `m4a_dataset.py` (68 min), `m4b_supervised.py`, `m4c_bayesopt.py`, `m4d_rl.py` (83 min/seed), `m4e_label_quality.py`, `m4_analyze.py`; tests +42 (plant family 13, features 8, ml 15, rl 6).
- Dataset: 1350 plants (840 train / 180 val / 180 test + 150 held-out structures: underdamped, third order), 762 labelled evaluations each; labels within 0.7% (median), 5% (p90) of a 3000-simulation search. Not versioned (data/m4, models/m4).
- Results (regret = ITAE / label): constant normalised gains median 6.1-6.7 (the v2 DC-motor family: 3%) so plant structure matters; ideal noise-free bump test: SVR/extra-trees/ANN 1.01-1.04; **clean-trained models collapse (median 80) when features come from a noisy bump test**, training on noisy features gives 1.2-1.3 (SVR p90 3.0-3.5) = primary protocol; unseen structures median 1.9-3, underdamped plants 13-21; decision-focused ANN helps only with clean features; GPO most sample-efficient without training data (median 1.20 at 60 simulations vs PSO 1.71, GA 2.36) but stalls on the link (gap 2.35 at 400); **RL as a learned local search with 8 simulations: median 1.24-1.27, p90 ~2, better than random search with 16 simulations; OOD tail p90 5.0 vs SVR 333**.
- Process notes: the RL job was left to finish (83 min/seed) after the harness did not allow stopping it; the script now documents that the shipped models used multi-threaded learners. A first full evaluation on noise-free features had to be redone with a realistic noisy protocol after the collapse was found.
- Not done: RL budget/step-size ablation, RL with noisy features, noise-aware labels.

## M5 (hybrid: GA-enhanced ML, ML + classical) - done; details in `report/methods/M5_hybrid.md`
- New code: `src/tuners/hybrid.py` (genome decoder, CV closed-loop-regret fitness, Cohen-Coon in normalised z-space, `LocalRefiner`, `ClassicalMLTuner`, `HybridModelTuner`), `experiments/m5_common.py`, `m5a_ga_ml.py`, `m5b_hybrid.py`, `tests/test_tuners_hybrid.py` (16).
- Results (noisy protocol, median regret val/test): GA-selected ML 1.38/1.47 vs default SVR 1.23/1.30 vs random-search-selected 1.25/1.27 -> **GA-enhanced ML gives no gain** (also not on its own CV fitness). Hybrid: Cohen-Coon alone 5.1/5.6; SVR 1.32/1.36; CC + SVR correction 1.31/1.39 (no better than SVR); **+20-simulation refinement: 1.03/1.04** (p90 1.5-1.7, 87-90% within 1.5x). The value is in the refinement, not the classical rule; the rule is only useful as a cold start (CC + refine 1.29 vs constant + refine 1.66). On the benchmark plants the link stays >= 12x from the best-known optimum for every variant.
- Bugs: unpicklable lambda in the tuner (multiprocessing), wrong metric column name in the benchmark stage; both fixed.

## M6 (adaptive ML-based tuning) - done; details in `report/methods/M6_adaptive.md`
- New code: `src/tuners/adaptive.py` (switching plant, gated RLS estimator on signal levels, ML / pole-placement / scheduled / frozen policies, scenario loop), `features_from_model` (refactor of `extract_features`), `experiments/m6_adaptive.py`, `m6_pool.py`, `tests/test_tuners_adaptive.py` (14).
- Scenarios (80 held-out plants, noisy and noise-free): gain x0.4 / x2.5, lags x3 / x0.4, no change. Frozen ML gains are 17-25x (noisy) / 45-100x (noise-free) worse than a retune after gain x2.5 / lag x0.4; adaptive ML brings that to 1.1-1.5x after the first post-change step (first window not recovered: 5.2x), but is 1.4-3x worse than frozen gains when nothing changes. Classical self-tuning pole placement (same estimator) is as good or better in most noisy cases: the estimator, not the ML predictor, is the valuable part.
- Findings that shaped the method: differenced ARX estimate biased K up to 4x under noise (-> level form); noisy-trained SVR is out of distribution on estimator features (-> clean-trained, clipped); aggressive ML gains + saturation + estimator drift diverged (-> saturation gating, `detune` 0.5 chosen on validation plants); controller must run at the simulation rate or the fine-rate-tuned gains are unusable. Test subset A informed the safeguards; subset B (40 fresh plants) confirms.
- Not done: RL comparison, benchmark plants with mid-run parameter change, friction/load scenarios.

## M7 (cross-method benchmark and final report) - done; `report/methods_report.md`
- New code: `experiments/m7_benchmark.py` (prepare / run / analyze), `m7_analyze.py`, `run_methods.py`; `adaptive.py` gained the plant-generic `run_loop`, `ChangePlant` (actuator gain / time-rate change at a time) and `AdaptiveMLTuner`; `HybridModelTuner` got a `features` option.
- Run: 20 tuners (18 methods, 2 references, plus the online variant 06b) x 5 plants x seeds = 250 jobs, 1 725 rows, about 1 h; results in `results/methods/M7/` (CSVs, heat map, Pareto, robustness, family figures).
- Findings: GA/PSO/ABC/SA 1.0-1.3x on the clean step; Pareto front self-tuning - GA-enhanced ML - PPO - hybrid (24 sims, 2.4x) - GPO (60, 1.6x) - GA; noise: hybrid best (1.26x); plant change (actuator gain x2.5): aggressive static gains unstable on the dead-time process, conservative/adaptive survive, online self-tuning best overall; adaptive ML fails on DC position (short horizon, coarse sampling: diagnosed, deliberately not fixed so the benchmark stays free of tuning); N-PID vs L-PID with equal gains: better in 44/36/51 of 95 cells (step/load/noise).
