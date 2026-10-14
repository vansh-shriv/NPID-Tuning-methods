# M5 - Hybrid Tuning of the N-PID Controller (GA-enhanced ML, ML + classical)

*Phase M5 of `docs/METHODS_ROADMAP.md`; protocol `docs/PROTOCOL.md`. Code: `src/tuners/hybrid.py`; experiments `experiments/m5_common.py`, `m5a_ga_ml.py`, `m5b_hybrid.py`; tests `tests/test_tuners_hybrid.py` (16); results `results/methods/M5/` (CSVs, `figures/`); logs `results/methods_m5a.log`, `methods_m5b.log`. Data and noisy-feature protocol are those of M4 (1350 plants, features from a noisy bump test).*

## 1. Summary

1. **GA-enhanced ML (method 16) did not help.** A GA that selects model family, hyper-parameters and feature subset by cross-validated closed-loop regret reached held-out median regret 1.38 / 1.47 (val / test, median of 3 seeds) against **1.23 / 1.30 for the default SVR** of M4 and 1.25 / 1.27 for random search over the same genome space with the same budget. The GA is the worst of the three on the held-out plants and has the widest seed spread (val 1.24-1.38). It did not even win on its own fitness (best CV fitness 1.62-1.97 vs 1.60-1.70 for random search). Cause (section 4): the fitness is a noisy, small-sample estimate and the problem is easy for any reasonably regularised SVR, so there is little to select; the search mostly fits CV noise.
2. **Hybrid ML + classical (method 17) works, and the gain comes from the short local search, not from the classical rule.** Median regret (val / test / ood), cost in simulations including the 3-experiment bump test:

| variant | sims | val | test | ood |
|---|---|---|---|---|
| constant start (M4 baseline) | 3 | 7.54 | 6.61 | 39.1 |
| Cohen-Coon alone | 3 | 5.12 | 5.57 | 4.40 |
| SVR alone (M4) | 3 | 1.32 | 1.36 | 2.39 |
| Cohen-Coon + SVR correction | 3 | 1.31 | 1.39 | 2.30 |
| constant start + 20-sim refinement | 23-24 | 1.66 | 1.73 | 1.70 |
| Cohen-Coon + 20-sim refinement | 23-24 | 1.29 | 1.38 | 1.50 |
| **SVR + 20-sim refinement** | 23-24 | 1.035 | 1.048 | 1.38 |
| **Cohen-Coon + SVR + 20-sim refinement** | 23-24 | **1.033** | **1.044** | **1.36** |

   (SVR alone is 1.32/1.36 here, not the M4 1.23/1.30, because the hybrid tuner runs the bump test itself with a new noise seed instead of re-using the stored features; the M4 value is inside the seed noise of that.)
3. **Parts vs whole.** The classical rule alone is poor (median 5.1-5.6; Cohen-Coon is unreliable on dead-time-free and nonlinear plants: Hammerstein median 43.8). Used as the base of an ML correction it is **indistinguishable from ML alone** (1.31 vs 1.32 val, 1.39 vs 1.36 test; 1.033 vs 1.035 with refinement). It does help the search start: Cohen-Coon + refinement (1.29) clearly beats constant start + refinement (1.66). So the classical rule is a valid cold-start when no trained model exists and adds nothing once a model exists.
4. **Refinement is the valuable component:** 20 extra simulations bring SVR from 1.32/1.36 to 1.04 (val/test), cut the 90th percentile from 4.7-5.1 to 1.5-1.7, and put 87-90% of plants within 1.5x of the label (62% without refinement). It also removes most of the damage on unseen structures (ood 90th percentile 439 -> 3.3). Cost: 20 plant experiments (about 6x the bump test).
5. **Benchmark plants (full protocol, gap of the step ITAE to the M3 best-known optimum, median of 3 seeds):** the hybrids are not uniformly good. Best-known gaps: dc_position 1.5 (hybrid + refine) vs 2.5 (SVR); fopdt 1.27 vs 5.9; link 12.9 vs 38.5; tank 1.5 vs 2.2; dc_speed 2.4 vs 3.7. The link remains far from the optimum for every variant (>= 12.6x), as it was for every ML tuner in M4. Plain Cohen-Coon + refinement is best on dc_speed (2.0) and tank (1.1).

## 2. Methods

### 2.1 GA-enhanced ML (method 16) - `hybrid.decode_genome`, `cv_regret_fitness`, `HybridModelTuner`
Genome (17 genes on the unit cube): model family (SVR / MLP / extra-trees); SVR (log C, log gamma, log epsilon); MLP (1-3 layers, width 8-128, log alpha); extra-trees (min leaf, max-features fraction); 8 feature switches (the two strongest are forced on if fewer than two are selected). Fitness = geometric mean of the closed-loop regret (clipped to [0.5, 100]) of 3-fold cross-validated predictions on a fixed subsample of 420 training plants; every prediction is simulated (ITAE with the predicted gains / ITAE of the label). Search: the M3 GA (16 individuals, tournament, BLX-alpha, elitism; 142 evaluations) vs random search over the same genome (160 evaluations), 3 seeds. The selected model is refit on all 840 training plants and evaluated on val/test/ood (noisy protocol). Offline cost: about 160 x 420 = 67 000 simulations and 6.5-10.5 minutes per run (the label dataset cost 1.0 M simulations). Online cost is unchanged from the plain regressor.

### 2.2 Hybrid ML + classical (method 17) - `ClassicalMLTuner`, `cohen_coon_z`, `LocalRefiner`
1. Bump test (noisy, counted) -> identified (K_eff, T_n, dead-time fraction f).
2. **Classical start**: Cohen-Coon from f only (all quantities are dimensionless in the z-space of M4, so the rule needs no extra data); the dead-time fraction is floored at 0.05 because the rule diverges as theta -> 0.
3. **ML correction**: z = z_CC + SVR(features), the SVR trained on the residual (label - z_CC).
4. **Refinement**: (1+1) evolution strategy in z with the 1/5-success step rule (step 0.3, grow 1.4, shrink 0.8), 20 closed-loop simulations (the first evaluates the start), best point returned.
Each component can be disabled; the ablation of section 1 is the full 2x4 grid {no refinement, 20 sims} x {constant start, Cohen-Coon, SVR, Cohen-Coon + SVR}.

## 3. Validation (`tests/test_tuners_hybrid.py`, 16 tests)
Cohen-Coon in z-space equals the M1 rule (`classical.CohenCoon`) on exact FOPDTs and an integrator, and is finite for vanishing dead time; the genome decodes to a valid configuration over the whole cube (all three families reachable, at least two features, boundaries); every model family trains/predicts; feature masks really restrict the input; CV fitness prefers an informative model on an exact synthetic problem; the GA beats random search on a rugged genome-shaped function (4/5 seeds); the refiner respects its budget, is monotone, never worse than its start and deterministic per seed; the hybrid tuner's cost is bump test + exactly the refinement budget, and an ML correction shifts the gains by exactly the predicted factor. Pipeline check: the default SVR/extra-trees/MLP rows of `m5a` reproduce the M4 table to 3 digits.

## 4. Analysis, pros and cons

**GA-enhanced ML.** Pro: automates model selection, and can select by the quantity that matters (closed-loop regret rather than R^2). Cons: (i) 6-10 minutes and 67 000 simulations per search, for no held-out gain here; (ii) the fitness landscape is flat and noisy (SVRs within a wide hyper-parameter range score alike; fitness differences between candidates are comparable to resampling the CV subset), so the best CV candidate is partly a selection artefact: GA seed 2 chose the SVR (fitness 1.62, held-out 1.24/1.27) while seeds 0 and 1 chose extra-trees with 3 features and generalised worse (1.38/1.47; they dropped the actuator-authority feature); (iii) the M4 default already contains a nested grid search for the SVR. Caveats: 3 seeds, 142-160 evaluations, a 420-plant fitness subsample; a larger budget or a repeated-CV fitness might change the picture, but the conclusion at equal cost to random search stands for this setup. Not claimed: that GA hyper-parameter search never helps, only that it did not here.

**Hybrid.** Pros: refinement closes most of the gap to the label for 20 simulations, with a good worst case; the classical start is a free fallback with no training. Cons: the classical rule contributes nothing once a model is trained; Cohen-Coon alone fails badly on nonlinear/dead-time-free plants; refinement needs 20 plant experiments, which is impractical on a slow real plant (the cost axis of `figures/M5b_cost_vs_regret.png`); the local search is greedy and cannot escape on the rugged link.

**Errors / process notes.** The first launch of `m5b` failed: the tuner stored a lambda and could not be sent to worker processes (fixed: the base is now a property, regression-covered by the run itself). The benchmark stage initially read a non-existent `overshoot` column (fixed to `overshoot_pct`). No method returned a failure or "not applicable" on the benchmark plants.

## 5. Limits
Regret is against the M4 label (best of two 400-simulation searches, itself within 0.7% median of a 3000-simulation search). Refinement results are the median of 3 seeds per plant pooled over plants. Benchmark plants are far from the training family (link is lightly damped and nonlinear): numbers there are indicative, 3 seeds. Only Cohen-Coon was tried as the classical part (SIMC would need a derivative convention and was not tested).
