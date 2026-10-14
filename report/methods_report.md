# Tuning Methods for the N-PID Controller: Implementation and Benchmark of 18 Methods

*Final report of the methods study (docs/METHODS_ROADMAP.md, phases M0-M7; protocol `docs/PROTOCOL.md`). Per-phase detail, ablations and validation: `report/methods/M1_classical.md`, `M2_intelligent.md`, `M3_optimization.md`, `M4_ml_based.md`, `M5_hybrid.md`, `M6_adaptive.md`. Engineering log: `progress.md`. Reproduction: `python run_methods.py` (M7) or `--all`. All numbers below come from `results/methods/M7/` (cross-benchmark) unless a phase report is cited.*

## 1. Summary

**What was done.** All 18 tuning methods of the taxonomy (5 classical, 2 intelligent, 4 optimization, 4 ML, 2 hybrid, 1 adaptive) were implemented from scratch behind one interface (`Tuner.tune(problem, budget, seed)`, every plant experiment counted), validated against analytic answers (225 tests in the suite), and benchmarked in one run on 5 plants (DC speed, DC position, gravity link, dead-time process, nonlinear tank) and 4 scenarios (clean step, input disturbance, measurement noise with 10 noise seeds, and a mid-run 2.5x actuator-gain change), 250 tuning jobs, 1 725 evaluated rows, each method's gains applied to an N-PID and (RQ5) to an L-PID. Two reference methods were added (random search with the optimizers' budget; one constant normalised gain set).

**Findings.**

1. **With a budget of hundreds of simulations, optimization wins on a clean step, and it is not close.** GA, PSO, ABC and SA are within 1.0-1.3x (geometric mean over the 5 plants, relative to the best method on each plant) and within 1.04-1.4x of the best-known optimum of M3a; GA ranks first on every plant it can be separated on. Everything else is 1.6-33x worse. The price is 400 simulations (about 1 600 s of plant time), 100-1 000x the cost of a rule.
2. **The Pareto front of tuning cost vs accuracy (step, geometric mean) consists of six methods**: self-tuning pole placement (1 closed-loop experiment, 13.4x), GA-enhanced ML (4 experiments, 3.4x), the PPO search (12 simulations, 3.2x), the hybrid ML + classical with 20 refinement simulations (24, **2.4x**), Gaussian-process optimisation (60, **1.6x**) and GA (362, 1.0x); the SVR (5.2x) and ANN (7.1x) at the same 4 experiments are dominated by GA-enhanced ML. Classical rules and model matching are dominated (8-33x for 4-40 experiments), the nearest to the front being SIMC (8.4x for 4 experiments; the only classical rule applicable on all five plants).
3. **Nothing but the iterative methods is reliable on the hard plants.** On the gravity link every method that is not a search is 7-100x off the best (hybrid with refinement 12.9x, PPO 28x, SVR 38x); on the DC-position plant only the searchers (GPO 3.8x), PPO, the hybrid and the SVR (2.5x) are within 2.5x of the best. Five classical/intelligent methods are **not applicable or fail** on at least one plant (section 5.3).
4. **Under measurement noise the ordering compresses and changes**: the best geometric-mean ratios are the hybrid (1.26x), GA-enhanced ML (1.85x), PPO (1.87x), the optimizers (1.8-2.0x) and the SVR (2.0x); clean-trained searchers lose 2x because ITAE-optimal gains amplify noise; SIMC (2.9x) is the best classical rule. Model-based methods that are conservative (self-tuning 3.9x) are not better than noise-aware ML.
5. **When the plant changes, aggressive tuning is fragile.** After a 2.5x actuator-gain increase the four optimizers (8.7-19x geometric mean), GPO (12x) and random search (9x) have the worst mean ranks of the main families; on the dead-time process every method that tuned statically to the nominal plant and aggressively (optimizers 470-870, ML and hybrids 630-1 510, ZN/Cohen-Coon/cycling 1 900-2 700) leaves the loop oscillating or unstable, against 4-28 for the conservative or adaptive methods (self-tuning 4.8, online 4.2, model matching 4.1, SIMC 14, trial and error 28, adaptive ML 9.2). The best method overall in this scenario is the **online self-tuning controller (1.7x, mean rank 3.4)** and the conservative **self-tuning auto-tuner (mean rank 2.5)**; adaptive ML recovers the dead-time process (9 vs 1 512 for frozen SVR gains) but destroys the DC-position loop (3.7 vs 0.007): the estimator cannot resolve that fast integrating plant at the sampling interval and window length available (section 6.2). M6 (80 random plants, 4 structures) gave a more favourable and more detailed picture of adaptive ML; the two should be read together.
6. **N-PID vs L-PID with identical gains (RQ5)**: across 95 (method, plant) cells per scenario the N-PID is better by more than 10% in 44 cells (step), 36 (load) and 51 (noise); the L-PID in 28, 22 and 27; the rest are within 10%. The N-PID's advantage is mostly noise-related, consistent with M1-M3 (with ideal tuning of both controllers it is not better on the clean step).
7. **Hybrid and ML methods help most when the cost budget is small.** The ML regressors are almost free once trained (3-4 experiments) but only 3-7x from optimal and fragile under noisy identification and unseen structures (M4); the hybrid (ML start + 20 simulations) is the most cost-efficient method between "a rule" and "an optimizer"; GA-selected hyper-parameters (method 16) were no better than the defaults on the 1 350-plant family (M5) but gave the best of the three static regressors on the benchmark (3.4x vs 5.1x SVR, 7.1x ANN), a difference we do not claim to be causal (two different tests, one model of each kind).

**Recommendations.** (a) Plant simulable or cheap to experiment on, no tight budget: **GA or PSO** (400 simulations), possibly noise-aware if the sensor is noisy (M3). (b) Budget of about 60 simulations: **Gaussian-process optimisation**, except on rugged landscapes (the link). (c) A handful of experiments and a trained model for the plant class: **ML start + 20 simulations (hybrid)**; without a trained model **SIMC** (rule, 4 experiments, applicable everywhere). (d) Plant parameters will change and aggressive gains cannot be afforded: a **conservative model-based design with online estimation (self-tuning)**; the adaptive ML variant is not recommended without a better estimator for integrating/fast plants. (e) Do not use trial-and-error, reaction-curve rules or cycling as the *only* method on plants with strong nonlinearity, integrators or no dead time (their rules assume a FOPDT).

## 2. Study design and honesty notes

* **Controller under test**: N-PID with scale-free shaping (n=m=p=2, e_scale=|y_ref|, de_scale=|y_ref|/Ts), filtered derivative, back-calculation anti-windup, saturation. Linear-gain designs are used as the N-PID base gains (the approximation of the specification); the same gains on an L-PID measure its effect (RQ5).
* **Plants and scenarios** fixed in `docs/PROTOCOL.md` before any method was run. The fourth scenario (plant change) was defined in M7: the actuator gain is multiplied by 2.5 at 1.95 H of a five-window run with setpoint steps (1.0, 0.8, 1.0, 0.8, 1.0 x y_ref), H = the scenario horizon; the reported metric is the ITAE of windows 3-4 (after the first post-change step), static methods keep their nominal gains, method 18 and the extra row 06b retune online.
* **Budgets and seeds**: optimizers 400 simulations x 5 seeds; GPO 60 x 5 seeds; RL: the three PPO models (8 simulations per session); hybrid and adaptive: 3 seeds; deterministic methods one run. Medians over seeds are used; "better" in the rank tables means beyond the larger of the two seed IQRs (floor 5%).
* **ML models are trained on the plant family of M4, not on the benchmark plants** (zero-shot transfer; the link, DC motors and tank are outside its structures). RL and the ANN use the bump-test features of the training protocol; the SVR, ANN, GA-ML and hybrid use bump-test features measured with the scenario's noise (the primary M4 protocol).
* **Cost accounting**: `evals` = counted closed-/open-loop simulations (self-tuning counts one long closed-loop experiment, 1 evaluation of 25.6 s median; compare plant time, `median_sim_seconds`, which is the fairer axis). The methods' own training cost (dataset 68 min, RL 83 min per seed, GA search 7-10 min) is **not** included in the per-plant cost.
* **Failure flag**: a cell counts as failed if the tuner is not applicable, the response diverges, or ITAE exceeds 100x the best method on that plant and scenario; failed cells enter aggregates as ratio 100.
* **Process disclosure.** Methods were developed and ablated in their phases (M1-M6) on their own data; hyper-parameters of M1-M5 methods are the defaults documented there. The M6 detune factor, window rules and safeguards were chosen on validation plants and one diagnosed test run (M6 report section 4.3). **M7 introduced no tuning**: it was run once with the phase defaults; the method-18 dc_position failure (section 6.2) was diagnosed but not "fixed" after seeing it, because a fix would use the benchmark as a development set.
* **Limits**: one measurement-noise level per plant, one plant-change type in M7, 5 plants, 5 seeds; ratios are against the best method *in this study* (the best known optimum is reported separately for the step); wall-time numbers depend on the implementation.

## 3. The 18 methods

Notation: ratio = geometric mean over the five plants of ITAE / best ITAE on that plant, N-PID; (step | noise | change). Gap = ITAE / best-known optimum (M3a), step. "Cost" = median simulations (plant seconds).

| # | Method | How it tunes the N-PID | Cost | step | noise | change | Where it fails |
|---|---|---|---|---|---|---|---|
| 1 | **Trial and error** (scripted emulation of a human procedure) | adjusts Kp, Kd, Ki from rise time, overshoot and offset of closed-loop trials | 40 (160 s) | 19.0 | 6.4 | 6.3 | the link (never reaches the acceptance criteria), thresholds set the result (up to 12x, M1) |
| 2 | **Ziegler-Nichols (reaction curve)** | FOPDT from a bump test, tangent method, PID rules | 4 (42 s) | 14.8 | 4.5 | 6.6 | tank (theta/tau = 0.005, out of range), link; aggressive, 25-60% overshoot |
| 3 | **Cohen-Coon** | FOPDT (two-point), Cohen-Coon rules | 4 (42 s) | 19.0 | 5.1 | 7.4 | tank (rule out of range), dead-time-dominant fopdt 38x |
| 4 | **Continuous cycling** (closed-loop ultimate gain, ZN rules) | search for sustained oscillation (half-cycle ratio) | 19 (84 s) | 33.4 | 5.3 | 40.4 | not applicable on 3 of 5 plants (no true ultimate gain: DC speed, link, tank) |
| 5 | **Approximate time-domain** (SIMC, CHR) | identified FOPDT/SOPDT, closed-loop time constant from the dead time | 4 (42 s) | 8.4 | 2.9 | 3.1 | link 55x (step); the only classical rule applicable everywhere |
| 6 | **Self-tuning (pole placement)** | RLS ARX(2,2) + exact PID placement, converged gains returned | 1 run (26 s) | 13.4 | 3.9 | **2.95** | dc_position (step 54x, ARX(2,2) cannot represent the third-order integrating plant); conservative |
| 7 | **Model matching** | frequency-domain fit of the closed loop to a reference model, time-domain refinement | 4 (42 s) | 19.0 | 7.3 | 5.0 | tank (fails), link 100x |
| 8 | **PSO** | 16 particles, log-space, 400 simulations | 400 | 1.07 | 1.8 | 19 | after a plant change (aggressive gains) |
| 9 | **GA** | tournament, BLX-alpha, adaptive mutation, elitism | 362 | **1.00** | 2.0 | 12 | after a plant change |
| 10 | **ABC** | employed/onlooker/scout bees | 400 | 1.26 | 2.0 | 8.7 | slower convergence, link 1.9x |
| 11 | **SA** | geometric cooling, restarts | 400 | 1.31 | 1.8 | 12 | link 3.6x |
| 12 | **Regression (SVR)** | RBF SVR on bump-test features -> dimensionless log-gains | 4 (31 s) | 5.2 | 2.0 | 3.2 | the link (38x), unseen structures, features from noisy bump tests if trained on clean ones (M4: median regret 80) |
| 13 | **Reinforcement learning (PPO)** | learned local search, 8 simulations from a constant start | 12 (74 s) | 3.2 | 1.9 | 6.9 | the link (28x) |
| 14 | **Gaussian-process optimisation** | Matern GP + expected improvement | 60 (240 s) | 1.60 | 2.3 | 12 | rugged landscapes (link 2.7x at 60 simulations, M4: gap 2.35 at 400) |
| 15 | **ANN** (decision-focused loss through a learned cost surrogate, step-response signature) | ensemble of 5 MLPs | 4 (31 s) | 7.1 | 2.4 | 4.0 | link 49x; worse than the SVR with noisy features (M4) |
| 16 | **GA-enhanced ML** | GA selects model/hyper-parameters/features by CV closed-loop regret | 4 (31 s) + 7-10 min offline | 3.4 | 1.85 | 3.2 | no gain over the default SVR on the 1 350-plant family (M5) |
| 17 | **Hybrid ML + classical** | Cohen-Coon start + SVR correction + 20-simulation local search | 24 (118 s) | 2.4 | **1.26** | 3.5 | the link 13x |
| 18 | **Adaptive ML** | online estimator -> ML predictor -> gain update with safeguards | 5 (58 s) as auto-tuner; online in scenario 4 | 14.2 | 7.3 | 6.2 | dc_position (see 6.2), as auto-tuner conservative (detune 0.5) |
| R1 | Random search (reference) | 400 random gain sets | 400 | 1.72 | 2.3 | 9.2 | -- |
| R2 | Constant normalised gains (reference) | one fixed normalised gain set | 4 | 20.9 | 7.9 | 8.1 | tank (fails), dc_position 32x |

(06b, the online pole-placement controller of scenario 4: change 1.70, mean rank 3.4.)

Implementation notes per method (details, equations, ablations and validation in the phase reports):
* **M1** (`src/tuners/classical.py`, `identify.py`): every rule is guarded by its validity range (theta/tau in [0.1, 1]) and raises `NotApplicable` rather than returning unreliable gains; identification uses a counted open-loop bump test (operating-point search, tangent/two-point/least-squares FOPDT, IPDT for integrators). The legacy project's "ZN" was the closed-loop method (method 4).
* **M2** (`selftuning.py`, `modelmatching.py`): the sampled-data experiment fixed a 47-78% bias in identified DC gains found when a fast controller was run with a slower estimator; exact four-pole placement with a derivative filter and PI fallback.
* **M3** (`optimizers.py`): five optimizers from scratch on the unit cube, one `Objective` budget wrapper; validated on analytic test functions and on a plant with a brute-force optimum; M3a computes the best-known optimum (3 000-4 000 simulations, polished) used for the gap.
* **M4** (`plant_family.py`, `features.py`, `ml.py`, `rl.py`): a plant-agnostic dataset of 1 350 randomized plants (1.03 M labelled runs); gains normalised by K_eff and T_n so models transfer across scales.
* **M5** (`hybrid.py`) and **M6** (`adaptive.py`): see the phase reports.

## 4. Results

### 4.1 Per-scenario summary (geometric-mean ratio to the best method on each plant; mean significance rank; lower is better)

| family | step | load | noise | change | mean rank step | mean rank noise |
|---|---|---|---|---|---|---|
| optimization (4) | 1.16 | 1.23 | 1.91 | 12.1 | 1.2 | 4.6 |
| hybrid (2) | 2.89 | 4.07 | 1.56 | 3.33 | 6.3 | 4.6 |
| ml (4) | 4.17 | 4.41 | 2.18 | 5.47 | 8.3 | 7.0 |
| classical (5) | 18.98 | 7.57 | 5.08 | 6.55 | 13.5 | 12.8 |
| intelligent (2) | 16.22 | 18.11 | 5.63 | 3.97 | 14.7 | 12.8 |
| adaptive (1+1) | 14.21 | 15.02 | 7.31 | 3.95 | 12.6 | 13.8 |
| reference (2) | 11.32 | 10.36 | 5.07 | 8.68 | 10.3 | 10.9 |

(Medians over the methods of each family; per-method values in `M7_summary_per_method.csv`.)

Heat map of the whole benchmark: `results/methods/M7/figures/M7_heatmap.png`; cost vs performance: `M7_pareto.png`; clean vs noise: `M7_robustness.png`; per-family boxplots: `M7_families.png`. Tables: `M7_ratio_<scenario>.csv`, `M7_gap_to_best_known.csv`, `M7_ranks.csv`, `M7_applicability.csv`, `M7_rq5_npid_vs_lpid.csv`.

### 4.2 Who wins where (rank-1 methods, ties within the seed spread)
step: DC speed, DC position, tank: optimizers (+GPO on DC speed and tank; random search and adaptive ML tie on the tank); fopdt: optimizers, PPO and the hybrid; link: **GA only** (PSO/ABC/SA 1.4-3.6x). noise: DC speed: hybrid; DC position / fopdt: optimizers (+GPO on fopdt); link: self-tuning, PSO, ABC, SA; tank: SIMC and the ANN. load: the optimizers on four plants (PPO on DC position, GPO on DC speed and tank), ZN / cycling / PPO / hybrid on the dead-time process. change: DC speed: self-tuning, hybrid; fopdt: self-tuning, model matching; link: SVR, ANN, adaptive ML; tank: self-tuning; DC position: GA, ABC, SA, PPO, GA-ML, random search.

### 4.3 Cost vs performance
Pareto front (step, geometric mean over plants vs median simulations): self-tuning (1) -> SVR (4) -> GA-enhanced ML (4) -> PPO (12) -> hybrid (24) -> GPO (60) -> GA (362). Dominated: the SVR and ANN (GA-enhanced ML has the same cost and a lower gap), SIMC and the other rules (same cost, 8-19x), trial and error (40 simulations for 19x), cycling (19 for 33x), model matching and adaptive ML as an auto-tuner. In plant time the order is the same, with the optimizers at 1 450-1 600 s versus 31 s for the ML regressors.

### 4.4 Robustness (clean vs noise)
Clean-trained optimizers are 1.8-2.0x off on the noisy scenario against 1.0-1.3x on the clean step (ITAE-optimal gains with high derivative and proportional action); the ML methods trained on noisy bump-test features degrade much less (SVR 5.2 -> 2.0, hybrid 2.4 -> 1.26, PPO 3.2 -> 1.9) because the *ratio* is taken against the best method on the noisy scenario, where the conservative methods lose less. SIMC (8.4 -> 2.9) shows the same effect for rules. Noise-aware training of the optimizers (M3) removes most of the loss but trades clean performance.

### 4.5 Plant change (scenario 4)
Median window ITAE after the first post-change step (windows 3-4), per plant; lower is better:

| method | dc_speed | dc_position | link | fopdt | tank |
|---|---|---|---|---|---|
| 12 SVR (frozen) | 0.285 | 0.007 | 0.011 | 1 512 | 17.1 |
| 16 GA-enhanced ML (frozen) | 0.251 | 0.006 | 0.013 | 950 | 18.8 |
| 17 hybrid (frozen) | 0.227 | 0.010 | 0.018 | 1 067 | 16.4 |
| 06 self-tuning (static gains) | 0.244 | 7.76 | 0.018 | 4.77 | 8.71 |
| 06b self-tuning (online) | 0.211 | 0.024 | 0.039 | 4.16 | 8.94 |
| 18 adaptive ML (online) | 0.750 | 3.72 | 0.061 | 9.24 | 18.2 |

On the three plants where the nominal gains keep a stability margin (DC speed, DC position, link) the frozen ML gains are the best or tied; on the dead-time process (the plant gain doubles the loop gain of a loop tuned close to its limit) every aggressive static tuning is unstable and the methods that are conservative (self-tuning, SIMC, model matching) or adapt (06b, 18) are 50-300x better (the aggressive ones are all flagged failed); the tank, where the gain change is benign for a unipolar nonlinear plant, shows no difference between conservative and aggressive tuning beyond 2x. The first post-change window is not recovered by the online methods (06b on DC speed: 0.56 in the first window vs 0.21 afterwards; 18 on the dead-time process: 65 vs 9). Where the nominal gains keep a margin (DC position, link) both online methods are *worse* than the frozen ML gains (06b by 3.4x and 3.5x, adaptive ML by 500x and 5.5x), i.e. adaptation is not free; adaptive ML is also worse on DC speed (0.75 vs 0.29).

### 4.6 N-PID vs L-PID with the same gains (RQ5)
Over the (method, plant) cells with valid gains: step N-PID better 44 / similar 23 / L-PID better 28; load 36 / 37 / 22; noise 51 / 17 / 27 (10% band). The N-PID's gain over the L-PID is clearest under noise (the derivative shaping gamma switches the derivative off during fast noise), consistent with M3 where an L-PID tuned for the same noise closes most of the gap.

## 5. Per-family conclusions

### 5.1 Classical (M1)
Cheap (4-40 experiments, 31-160 s) and transparent, but 8-33x from the best on average and limited by their assumptions: open-loop rules need a FOPDT-like process (theta/tau in [0.1, 1]) and are not applicable or poor on the tank (no delay), the link (strongly nonlinear) and the DC speed loop (second order, the "ultimate gain" is a sampling/saturation limit cycle whose value varies 64-129 with detection settings). SIMC is the most robust rule (all five plants, 8.4x step, 2.9x noise, 3.1x after a plant change). Trial and error is a scripted emulation (a human is not simulated): its outcome depends on the acceptance thresholds (up to 12x). Best classical rule per plant is 1.5-29x worse than a 400-simulation PSO but costs under 2% of its experiment time (M1).

### 5.2 Intelligent (M2)
Self-tuning pole placement needs one long closed-loop experiment (26 s median, no bump test), works on all five plants, and is the best non-searching method on the link under noise and the most robust in the plant-change scenario because its gains are conservative; it is poor where its ARX(2,2) model is structurally wrong (DC position) and 13x off on the clean step. Model matching reproduces its reference but the result is set by the chosen reference speed (3.4-20x from PSO) and it fails on the tank.

### 5.3 Optimization (M3) and applicability
GA ~ PSO > SA ~ ABC > random search (28/50 pairwise comparisons significant: PSO 10 wins, GA 9, SA 5, ABC 4, random 0, M3); the best optimizer is within 5% of the best known on four of five plants at 400 simulations, not on the link. Cost is repeatable but the gains lie on ridges (gain CV 40-120%); clean-trained gains are 1.8-2.0x worse under noise and are the least robust to plant change. Applicability (M7_applicability.csv): optimizers, ML, hybrid, adaptive, self-tuning: all five plants; SIMC: all five; Cohen-Coon: 4 (tank n/a); ZN reaction curve: tank n/a, link fails; cycling: 2 (DC position, dead-time process); trial and error: link fails; model matching: tank fails; constant gains: tank fails.

### 5.4 ML (M4)
On the family of randomized plants, ML models with ideal identification are excellent (median regret 1.01-1.04 against 6.1-6.7 for a constant gain set), collapse (median 80) when fed features from a *noisy* bump test unless trained on noisy features, and are limited by unseen structures (underdamped plants median regret 13-21). GPO is the most sample-efficient tuner without training data; the PPO search with 8 simulations is the best tuner at about 11 simulations under realistic conditions. On the benchmark plants (outside the training family) the zero-shot ML tuners are 3.2-7x from the best and 25-49x on the link: the transfer gap is the main practical limit of the learned tuners.

### 5.5 Hybrid (M5)
GA-selected hyper-parameters did not beat defaults or random search at equal cost (M5); ML start + 20-simulation local search is the most cost-efficient method between a rule and an optimizer on the family (median regret 1.03 vs 1.32 for SVR alone) and on the benchmark (2.4x step, 1.26x noise, the best noisy performance of all methods). The classical rule adds nothing once a model exists (it is a cold start).

### 5.6 Adaptive (M6)
On 80 randomized plants with a gain or lag change, frozen ML gains are 17-100x worse than a retune after severe changes; adaptive ML recovers to 1.1-1.5x after the first post-change step but is 1.4-3x worse than frozen gains when nothing changes, and classical self-tuning with the *same estimator* is as good or better in most noisy cases: the estimator, not the ML predictor, is the valuable part. On the benchmark plants (M7) the result is mixed: it rescues the dead-time process (9 vs 1 512) and fails on DC position (section 6.2).

## 6. Observed errors and failure modes (cumulative)

### 6.1 Bugs found by validation and fixed (all covered by tests)
Tangent-method smoothing; `link` identified as an integrator (secant overshoot beyond the gravity limit); `tank` identified by probing the actuator limit; limit-cycle horizon; box too narrow for the optimum (P1); identification bias from running a fast controller with a slower estimator (M2); wrong feature protocol (clean-trained models on noisy features, M4); unpicklable lambda in the hybrid tuner; differenced estimator biasing the gain under noise, estimator features out of the training distribution, saturation-driven estimator drift (M6).

### 6.2 Open failure: adaptive ML on DC position (diagnosed, not fixed)
On the integrating DC-position plant (horizon 2 s, T_n 0.32 s, estimator sampling Delta = 0.148 s = 0.46 T_n, about 13 samples per window) the online estimator returns velocity gains between 0.8 and 15 (true 1.29, then 3.2) and dead-time estimates at the sampling floor; the ML predictor then raises Kp and Kd (Kd 0.65 -> 3.0) on a plant whose gain has just increased 2.5x, and the loop oscillates (window ITAE 3.7 vs 0.007 for the initial gains). The same estimator works on random integrators of M6 (median ratio 0.8) whose windows are 10 time scales long; the benchmark horizon (2 s, 6 T_n) is too short for it. Remedies not tried (would need new development data): minimum window length and sampling-rate rules, a do-no-harm check against the initial gains, an integrator-specific estimator.

### 6.3 Method-specific cautions
Self-tuning counts as 1 evaluation although it is a 26 s closed-loop experiment; the adaptive-as-auto-tuner variant (18, static scenarios) is the detuned (x0.5) gain set after three nominal windows and is therefore not representative of the online method; GPO at 60 simulations is compared with optimizers at 400 (different budgets by design); ratios are relative to the best of the compared methods, so they change if a method is added.

## 7. Reproduction
`python run_methods.py` (M7 benchmark with the existing data and models, about an hour), `python run_methods.py --all` (every phase, several hours; the M4 dataset and PPO models are not versioned and are regenerated by `m4a` / `m4d train`), `python run_methods.py --only m7-analyze` (tables and figures from `results/methods/M7/M7_benchmark.csv`). Tests: `pytest` (2-5 min).
