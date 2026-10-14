# Methods Roadmap: N-PID Tuning Methods (review taxonomy)

**Status:** M0, M1 (`M1_classical.md`), M2 (`M2_intelligent.md`), M3 (`M3_optimization.md`), M4 (`M4_ml_based.md`), M5 (`M5_hybrid.md`), M6 (`M6_adaptive.md`), M7 (`report/methods_report.md`) done: **all phases complete**. Experimental protocol: `docs/PROTOCOL.md`. Log: `progress.md`.

Goal: implement every tuning method in the review's taxonomy for the N-PID controller, benchmark them under identical conditions, and produce results, plots and a per-phase write-up. This is the second round of work; `docs/ROADMAP_v2.md` / `progress.md` cover the first (v2 re-evaluation), and all of that infrastructure (plants, controllers, metrics, PSO, tests) is reused.

| Family | Methods | Phase |
|---|---|---|
| Classical | Trial and error, Ziegler-Nichols, Cohen-Coon, Continuous cycling, Approximate time-domain tuning | M1 |
| Intelligent | Self-tuning (pole placement), Model matching | M2 |
| Optimization-based | PSO, GA, ABC, SA | M3 |
| ML-based | Regression (SVR/NN), Reinforcement learning, Gaussian-process optimization, ANN | M4 |
| Hybrid | GA-enhanced ML, Hybrid (ML + classical) | M5 |
| Adaptive | Adaptive ML-based tuning | M6 |
| Cross-method comparison and report | all of the above | M7 |

---

## Common design (built once, in M0)

**1. One tuner interface.** Every method is a class `Tuner` with `tune(problem, budget, seed) -> TuningResult`.
- `TuningProblem`: plant factory, setpoint, `dt`, `t_end`, disturbance, noise level, controller kind (`npid`), shaping options, gain bounds, objective (ITAE by default).
- `TuningResult`: `gains {Kp, Ki, Kd[, n, m, p]}`, `cost`, `n_evals` (closed-loop simulations used), `wall_time`, `history` (best-so-far vs evaluation count), `meta` (identified model, selected rule, ...).
- `n_evals` and `wall_time` make methods comparable on **tuning cost**, not only on final performance. Methods that need a plant experiment (step test, relay test, oscillation search) are charged for the simulations they run.

**2. How classical and model-based methods tune an N-PID.** They design linear PID gains (Kp, Ki, Kd). These become the N-PID's base gains, with the scale-free shaping of v2 (`n=m=p=2`, `e_scale=|ref|`, `de_scale=|ref|/Ts`). This is exactly the approximation the project spec makes (near e = 0 the shaping functions are about 1) and it is documented in each phase. Every result also reports the same gains on an L-PID, so the effect of the nonlinearity is visible. Optimization and ML methods can additionally tune n, m, p ("+shape" variants).

**3. Fair benchmarking.**
- Plants: the five from experiment 05 (DC speed, DC position, gravity link, dead-time process, nonlinear tank) plus, from M4 on, a randomized plant family.
- Scenarios: clean step, load/input disturbance, noisy feedback (10 seeds), and (M6) parameter drift and regime change.
- Metrics: rise time, overshoot, settling time, IAE, ISE, ITAE (unchanged), plus control effort and tuning cost.
- Stochastic methods: 5 seeds, report mean, std and best. Optimizers get the same evaluation budget (400 simulations, the budget of PSO 16x25) unless the method is meant to be sample-efficient (GPO, 60 evals).
- Statistical honesty: report ranks with spread; no claim from a difference smaller than the seed-to-seed spread.

**4. Every phase delivers**: module(s) in `src/tuners/`, pytest tests (validated on plants with known answers), an experiment script in `experiments/`, CSVs in `results/methods/`, figures in `results/methods/figures/`, and an entry in `progress.md` with findings and caveats.

**Standard figures per phase**: (a) step response of every method of the phase on each plant, (b) a method x plant ITAE table/heat map, (c) tuning cost vs performance, (d) method-specific diagnostic (convergence curves, identified-vs-true model, parameter traces, ...).

---

## M0: Framework (prerequisite)
- `src/tuners/base.py`: `TuningProblem`, `TuningResult`, `Tuner`, common objective (`evaluate(gains)` wrapping `run_simulation` with evaluation counting, instability guard, optional noise and disturbance).
- `src/tuners/identify.py`: shared identification used by several methods: FOPDT from an open-loop step (tangent and two-point/Smith methods), second-order model, ARX by least squares (and recursive LS for M2/M6). Integrating plants (DC position) are handled by an integrator-plus-delay identification.
- `src/tuners/benchmark.py`: runs a list of tuners over plants, scenarios and seeds, writes the standard CSV, and draws the standard figures.
- Port the existing methods onto the interface: PSO (`tune_pso`) and the ZN/relay code.
- Tests: interface contract, evaluation counting, identification accuracy on plants with known parameters.
- Output: `results/methods/M0_*`; a smoke benchmark with PSO and one classical rule.

## M1: Classical methods (5)
All use the identified model or a plant experiment; no optimization loop.
1. **Trial and error**: a scripted emulation of an engineer's procedure (raise Kp until fast but not oscillatory, add Kd to cut overshoot, add Ki to remove the offset, with a fixed trial budget and stopping rule). Each trial is one simulation. It is deterministic, documented as an emulation, and reported as the "manual tuning" reference.
2. **Ziegler-Nichols (open-loop reaction curve)**: FOPDT (K, tau, theta) from a step test, classic ZN reaction-curve rules (PID: Kp = 1.2 tau/(K theta), Ti = 2 theta, Td = 0.5 theta).
3. **Cohen-Coon**: same FOPDT identification, Cohen-Coon formulas (PID: Kp = (1/K)(tau/theta)(4/3 + theta/(4 tau)), Ti = theta (32 + 6 theta/tau)/(13 + 8 theta/tau), Td = 4 theta/(11 + 2 theta/tau)).
4. **Continuous cycling (closed-loop ultimate gain)**: sustained-oscillation search (the v2 half-cycle-ratio detector) and the relay variant, with the ZN, Tyreus-Luyben and Pessen rule sets. Note: the legacy project's "ZN" was this method; M1 separates the two. Plants without a true ultimate gain (the DC speed loop) are reported honestly (limit-cycle fallback).
5. **Approximate time-domain tuning**: tuning from time-domain specifications of the step response. Identify FOPDT/SOPDT, choose a closed-loop time constant from the desired rise/settling time, apply SIMC and Chien-Hrones-Reswick rules (0% and 20% overshoot variants).
- Validation: each rule against hand-computed gains on a known FOPDT; closed-loop sanity on the 5 plants.
- Diagnostics: identified vs true model overlay; rule-set comparison; applicability table (which plants each rule can handle: integrators, no delay, nonlinear).
- Expected finding to test: classical rules are cheap (few evaluations) but unreliable on non-FOPDT plants.

## M2: Intelligent methods (2)
6. **Self-tuning, pole placement**: discrete ARX (na=2, nb=2) identified online by recursive least squares with forgetting; PID gains computed every N samples to place the closed-loop poles at a desired (omega_n, zeta); gains pushed into the N-PID base gains with bumpless transfer. Convergence and tracking are evaluated both from a cold start and under parameter change. Diagnostics: parameter estimates vs truth, pole locations, gain trajectories.
7. **Model matching**: choose the gains so that the closed-loop response matches a reference model (second-order with specified omega_n, zeta, or FOPDT), by minimizing the integral squared model-matching error. Offline version (derivative-free optimizer on the simulated closed loop). Optional online variant (MIT-rule model-reference adaptation) as a stretch goal. Diagnostics: closed-loop vs reference-model overlay, matching error vs gain.
- Validation: pole placement recovers a known plant; model matching reproduces a reference when the plant structure allows it exactly.

## M3: Optimization-based methods (4)
All implemented from scratch in numpy with the shared interface (log-space gains, bounds, best-so-far history, seeds); hyperparameters follow the common literature defaults and the sibling-project implementations are used only as a reference.
8. **PSO**: the existing implementation (restarts, log-space) wrapped in the interface.
9. **Genetic algorithm**: tournament selection, BLX-alpha crossover, adaptive Gaussian mutation, elitism.
10. **Artificial bee colony**: employed, onlooker and scout phases, abandonment limit.
11. **Simulated annealing**: geometric cooling, Gaussian neighbourhood scaled with temperature, restarts.
- Same 400-evaluation budget, 5 seeds, plus an optional "+shape" run that also optimizes n, m, p.
- Diagnostics: convergence curves (median and interquartile range over seeds), success rate (fraction of seeds within 5% of the best known cost), sensitivity to budget (100/200/400/800), gain scatter (to show the flat-valley identifiability found in v2).
- Validation: each optimizer on analytic test functions and on a plant where the optimum is known by brute-force grid.

## M4: ML-based methods (4)
Needs a **plant-agnostic dataset** (open item from v2): randomized plant family (first/second-order plus delay, integrating, with saturation, Coulomb friction) labelled by multi-start PSO (best-of-3), with features from a step-test signature (identified K, tau, theta, dominant time scale, gain/saturation ratio, setpoint scale) instead of motor parameters. About 1500 labelled plants, 2-3 h on 14 workers, resumable. Held-out plants include unseen structures.
12. **Regression models (SVR / NN)**: SVR (RBF, one regressor per log-gain) and scikit-learn MLP on the signature features; baselines constant gains and ridge; metric = closed-loop regret vs PSO, not only R^2.
13. **Reinforcement learning**: a TD3 agent that sets gains in an episodic tuning task over the randomized plant family (observation = signature plus recent response, action = log-gains), trained with several seeds and compared against the simple baselines. This revisits the legacy RL with a fair setup; the honest outcome may again be negative.
14. **Gaussian-process optimization (Bayesian optimization)**: GP surrogate (Matern kernel) of the closed-loop ITAE with expected improvement, 60-evaluation budget, warm start from a classical rule. Diagnostics: surrogate fit, acquisition trace, sample efficiency vs PSO/GA.
15. **ANN**: a neural network on the raw step-response signature (resampled response), trained in PyTorch with a regret-aware loss through a learned ITAE surrogate, ensembles for uncertainty. Differs from method 12 by using the raw response rather than engineered features, and by a task-level loss.
- Diagnostics: regret distributions, extrapolation to unseen plant structures, uncertainty vs error, inference latency vs tuning cost.

## M5: Hybrid methods (2)
16. **GA-enhanced ML**: a GA selects the ML model's hyperparameters, architecture and feature subset using cross-validated closed-loop regret as fitness (the "neuro-evolution" of method 12/15). Compared against default hyperparameters and random search at equal cost.
17. **Hybrid ML + classical**: a classical rule (Cohen-Coon or SIMC) gives initial gains; an ML model predicts a multiplicative correction; optionally a short local refinement (about 20 evaluations). Compared with each part alone, in accuracy and in tuning cost.

## M6: Adaptive method (1)
18. **Adaptive ML-based tuning**: online loop with a recursive plant estimator (from M2) feeding an ML predictor (from M4/M5) that retunes the gains on a schedule, with safety limits (gain bounds, rate limits, bumpless transfer). Evaluated on scenarios designed to **give adaptation headroom** (the v2 drift study showed none for slow resistance drift): step change in inertia, friction growth, load change, setpoint-range change. Compared with frozen gains, classical self-tuning pole placement (M2) and the RL agent. Diagnostics: estimate and gain trajectories, ITAE by time window, recovery time after the change.

## M7: Cross-method benchmark and report
- Single run of all 18 methods over 5 plants x 4 scenarios x seeds with the standard CSV.
- Figures: method x plant ranking heat map, tuning-cost vs performance Pareto front, robustness (clean vs noise) scatter, per-family summary, applicability matrix (which method works on which plant type).
- `report/methods_report.md`: per-method description, implementation notes, results, limitations; per-family conclusions; recommendations. Updates `README.md`, `CLAUDE.md`, `run_v2.py` (or a new `run_methods.py`).

---

## Order, effort and compute
M0 -> M1 -> M3 -> M2 -> M4 -> M5 -> M6 -> M7 (optimizers before the ML phases because ML labels and several hybrids reuse them). Approximate wall time of the heavy items on this machine (16 cores): optimizer benchmark 30-60 min, plant-family dataset 2-3 h, RL training 1-2 h per seed set, final cross-benchmark 1-2 h. Dataset and RL runs are checkpointed and resumable.

## Assumptions to confirm (defaults chosen; tell me to change any)
1. "Trial and error" is implemented as a scripted emulation of the manual procedure (a human cannot be simulated exactly).
2. "Continuous cycling" = closed-loop ultimate-gain method; "Ziegler-Nichols" = open-loop reaction curve. (The legacy project's ZN was the closed-loop one.)
3. "Approximate time-domain tuning" = SIMC / Chien-Hrones-Reswick rules from step-response specifications.
4. "ANN" is distinguished from "Regression (SVR/NN)" by using the raw step response and a task-level loss.
5. All methods tune an N-PID with the scale-free shaping; classical/model-based designs use the linear-gain approximation above.
6. Optimizer budget 400 evaluations, 5 seeds; plants as listed; new plant-agnostic dataset in M4.
