# M6 - Adaptive ML-Based Tuning of the N-PID Controller (method 18)

*Phase M6 of `docs/METHODS_ROADMAP.md`; protocol `docs/PROTOCOL.md`. Code: `src/tuners/adaptive.py` (+ `features_from_model` in `src/tuners/features.py`); experiments `experiments/m6_adaptive.py`, `m6_pool.py`; tests `tests/test_tuners_adaptive.py` (14); results `results/methods/M6/` (CSVs, `figures/`). Reuses the RLS estimator and pole-placement design of M2, the M4 regressors and plant family, the M5 hybrid tuner.*

## 1. Summary

1. **Setting.** The v2 drift study showed no headroom for adaptation. M6 therefore uses explicit regime changes on 80 held-out plants of the M4 family (4 structures: first order, second order, integrating, Hammerstein): plant gain x0.4 / x2.5, all lags x3 / x0.4, and no change (to see whether adaptation does harm), each with measurement noise and noise-free. 5 reference windows; the plant changes just before window 2.
2. **Frozen gains are fragile.** After a gain x2.5 the frozen ML gains are 17-25x worse than a retune (median ITAE ratio, noisy; 45-100x noise-free), after lags x0.4 16-17x; mild changes (gain x0.4, lag x3) cost only 1.0-1.4x. So there is real headroom for adaptation where the plant becomes faster or higher-gain.
3. **Adaptive ML recovers most of that, but not all, and is not better than classical self-tuning.** Noisy protocol, windows 3-4 (after adaptation), median ratio to the retune-by-experiment reference, pooled over the two test subsets (80 plants): gain x2.5: frozen 21.3, **adaptive ML 1.51**, self-tuning pole placement 1.07; lag x0.4: frozen 17.3, **adaptive ML 1.09**, pole placement 0.69. The ML variant is within 2x of the reference on 59-72% of plants in these two cases (frozen 20-34%, pole placement 72-94%).
4. **The first window after the change is not recovered** (the estimator needs the response to a step): ratio 5.2 (ML) / 0.74 (pole placement) for gain x2.5 and 2.2 / 0.80 for lag x0.4 vs frozen 20.7 / 15.9. For a decrease of gain (x0.4) the frozen gains are already good (1.04-1.4x) and the adaptive ML tuner is *worse* in the first window (3.3x).
5. **Adaptation costs something when nothing changes:** with no change the adaptive ML tuner is worse than frozen gains in the adapted windows (ratio 1.38 vs 1.10 noisy, 3.10 vs 1.00 noise-free); it beats frozen gains on only 39% (noisy) / 12% (noise-free) of plants. Gain wander from the noisy estimate is visible in `figures/M6_example_traces.png`.
6. **Self-tuning pole placement (M2) is the stronger adaptive method in this setting** for the severe changes in the noisy protocol (beats frozen on 86-99%, ML on 72% of plants for lag x0.4), while the ML tuner is more consistent in the noise-free protocol (ratio 3.1 vs 3.5 for gain x2.5, 3.1 vs 3.5 for lag x0.4, i.e. similar). It is much better on Hammerstein plants (median 0.68 vs ML 1.28 in the noisy severe cases) and worse on integrators (1.93 vs 0.80).
7. **Honest caveat on the reference.** "Retune by experiment" (the M5 hybrid, tuned for one clean step from rest) is not an oracle for this window metric under noise: pole placement reaches ratios below 1 (0.7-0.95) because its gains are less aggressive and track the noisy small steps better.

## 2. Method

**Loop** (`run_scenario`): N-PID with scale-free shaping at the simulation rate, reference steps every H = 10 time scales (levels 1.0, 0.8, 1.0, 0.8, 1.0 x y_ref), plant change at 1.95 H (`SwitchingPlant`: gain or lag change with output continuity at the switch). The N-PID, filter and saturation are those of the protocol; gains are changed bumplessly (integrator state rescaled so that Ki x integral is continuous).

**Estimator** (`OnlineEstimator`): RLS (forgetting 0.99) of the sampled ARX model with sampling interval Delta = max(theta, T_n/10) from the initial bump test and the *mean* input of each interval (zero-order-hold equivalent). It runs **only while the loop is excited** (a window of max(4 T_n, 25 Delta) after each reference step), with a covariance reset at every step so a changed plant can be tracked, and not while the actuator is saturated. Self-regulating plants are estimated on the signal *levels* (A y = B u); the differenced form of M2 biased the gain by up to 4x under measurement noise (Delta small relative to the noise), the integrating form remains differenced (K estimate still noisy: 10-25% errors on the integrators examined at 1-2% noise). The ARX estimate is turned into a FOPDT/IPDT model through its step response (`process_model`) and then into the M4 features (`features_from_model`).

**ML predictor**: the M4 SVR on the engineered features, predicting dimensionless log-gains that are de-normalised with the *estimated* K_eff and T_n (so a gain change automatically rescales Kp). **Feature protocol (a finding):** the models of M4 trained on noisy bump-test features are *not* the right ones here: the features of an ARX step response are smooth like a noise-free bump test (fit error ~0.001 vs median 0.07 in the noisy training features; peak ratio 1.0 vs 1.26; t63 1.0 vs 0.57). The estimator side therefore uses the SVR trained on *clean* features and clipped to the training feature range (`ClippedModel`). The initial gains (from a bump test with noise) use the noisy-trained SVR, as in M4.

**Safety limits**: gain box (`default_bounds`), rate limit 0.15 decades per redesign (redesign every 5 samples inside the window), the estimator model must be stable, K > 0 and fit error below 0.3, and a **robustness factor `detune` = 0.5** on the predicted gains (chosen on validation plants from {1, 0.5, 0.3}, see section 4).

**Compared methods** (same initial gains, same loop): frozen; adaptive ML; self-tuning pole placement (the *same* estimator and gating with M2's pole-placement design in place of the ML predictor; speed-up 3, zeta 0.8; same bounds and rate limit); retune by experiment (the M5 hybrid, Cohen-Coon + SVR + 20 simulations, run on the plant before and after the change and switched exactly at the change; costs 2 x 23 plant experiments and knows when the change happened). The reinforcement-learning agent of M4 was **not** included: it needs closed-loop plant experiments (8 per session) rather than an online estimate, so the experiment-based reference plays that role; a change-triggered RL session is not tested.

**Metric**: ITAE of each window (from the window's step, true output) divided by the reference's ITAE in the same window, median over plants; ratios clipped to [0, 100] (a failure counts 100; no failures occurred).

## 3. Validation (`tests/test_tuners_adaptive.py`, 14 tests)
Switching plant: output continuity at the switch, new DC gain, restart safety, dead time preserved; scaled specs and feasibility; estimator recovers K/tau/theta of a FOPDT from excited data (K within 3%, tau 10%), stays within 6% in K at 2% noise (regression for the differenced-form bias), declines unstable/empty models; rate limit in decades; window metrics against analytic integrals (IAE 5, ITAE 25 for tau = 5); frozen loop deterministic and tracking; scheduled policy switches once; adaptive policy updates only inside excitation windows, within the gain box and the rate limit; pole-placement policy rescales Kp by about 1/3 after a 3x gain increase. Bugs found by these checks and the dev runs: the first estimator version differenced the signals (noise bias), a start below the gain box (Kd = 0) stayed below it after a rate-limited step (now clipped), and the test of the policy against frozen gains had a wrong premise (an easy PI plant) and was rewritten to test the gain rescaling.

## 4. Results

### 4.1 Median ITAE ratio to the reference, pooled over two test subsets (80 plants, `M6_pooled_median.csv`)

Noisy measurements (pre = windows 0-1, first = window 2, adapted = windows 3-4):

| change | method | pre | first | adapted |
|---|---|---|---|---|
| none | frozen | 1.15 | 1.08 | 1.10 |
| | adaptive ML | 1.20 | 1.46 | 1.38 |
| | pole placement | 0.98 | 1.31 | 1.10 |
| gain x0.4 | frozen | 1.13 | 1.40 | 1.04 |
| | adaptive ML | 1.24 | 3.34 | 1.11 |
| | pole placement | 1.08 | 2.04 | 1.32 |
| gain x2.5 | frozen | 1.06 | 20.7 | 21.3 |
| | adaptive ML | 1.12 | 5.18 | 1.51 |
| | pole placement | 1.03 | 0.74 | 1.07 |
| lag x3 | frozen | 1.14 | 1.25 | 1.10 |
| | adaptive ML | 1.20 | 1.67 | 1.24 |
| | pole placement | 0.98 | 3.64 | 1.88 |
| lag x0.4 | frozen | 1.15 | 15.9 | 17.3 |
| | adaptive ML | 1.21 | 2.16 | 1.09 |
| | pole placement | 0.98 | 0.80 | 0.69 |

Noise-free: frozen is almost optimal without a change (1.00) and fails far worse after severe changes (gain x2.5: 51 then 100 = the cap; lag x0.4: 93, 96); adaptive ML 2.8-3.8 in the first window and 3.1 adapted (gain x2.5), 3.1 (lag x0.4); pole placement 1.5 / 3.5 and 2.5 / 3.5; without a change both adaptive methods are 3.0-3.1x worse than frozen (noise-free adaptation is *harmful*: the identified models are accurate, the redesign is more conservative than the ITAE-optimal initial gains).

### 4.2 Plant-level view (adapted windows, noisy, `M6_winrates.csv`)
Fraction of plants where the adaptive ML tuner is better than frozen gains: gain x2.5 81%, lag x0.4 72%, gain x0.4 45%, lag x3 38%, no change 39%. Pole placement: 86%, 99%, 33%, 40%, 54%. ML better than pole placement: 36%, 28%, 43%, 66%, 34%.
By structure (severe-change cases, noisy, adapted windows, median ratio): frozen first order 1.7, Hammerstein 17.4, integrating 1.3, second order 3.3; adaptive ML 1.5 / 1.3 / 0.8 / 1.5; pole placement 0.66 / 0.68 / 1.93 / 1.5. The nonlinear (Hammerstein) plants gain the most from adapting, with either method.

### 4.3 Development history (what had to be fixed before the numbers above)
| version | outcome (noisy, gain x2.5, adapted windows, ML ratio) |
|---|---|
| controller sampled at Delta (T_n/20), differenced estimator | pole placement beat the reference by 10x; the reference gains (tuned at the fine sampling rate) were unusable at the coarse rate: the environment was wrong, controller moved back to the simulation rate |
| estimator on differences: K biased by up to 4x under noise | ML worse than frozen with no change; switched to level form for regulating plants |
| noisy-trained SVR on estimator features | no-change ratio 21-58 (features out of distribution) -> clean-trained, clipped SVR |
| clean SVR, detune 1.0, no gating | gain x2.5: ratio 14-24 (first test run, 40 plants): high-gain predictions (small dead-time fraction, f ~ 0.02) put the loop near its gain limit; after the plant gain rose the loop saturated and the estimator, fed saturated data, drifted further (Kp -> 500 on one plant). Fixes: skip updates while saturated, clip features, **detune** |
| detune 1.0 / 0.5 / 0.3 on 14 validation plants (adapted windows 3-4, noisy; ML ratio for gain x2.5 / lag x0.4 / none) | 2.3 / 1.5 / 1.45; **1.5 / 0.8 / 1.35**; 1.9 / 1.35 / 1.43 -> 0.5 |

**Disclosure on data use.** The hyper-parameters (rate limit, windows, forgetting) were set during development on 14 validation plants; the safeguards (saturation gating, feature clipping, detune) were motivated by diagnosing failures of the first run on the 40 test plants of subset A, so subset A is not untouched. The numbers above pool subset A with 40 fresh plants (subset B) that were not used for any design decision; the two subsets give the same picture (e.g. noisy gain x2.5 adapted: ML 1.90 / 1.38 on A and 1.40 / 1.57 on B, frozen 21 / 23 vs 19 / 24).

## 5. Analysis, pros and cons

**Adaptive ML.** Pros: it needs no plant experiment and no interruption; it restores near-reference performance (1.1-1.5x) after severe plant changes where frozen ML gains are 17-100x worse; it is safe in the sense of the limits applied (no run diverged: failure rate 0 in 744 runs per method). Cons: (i) it needs a *transient* (a setpoint step) to learn the change, so the first reference step after a change is not recovered (5.2x for a 2.5x higher gain); (ii) it degrades a loop that has not changed (1.3-3x), because the estimate is noisy and the gains follow it; (iii) the ML predictor is a delicate component: it needs features in the training distribution (clean-trained model for estimator features, clipped inputs), it predicts aggressive gains that have little margin for the changes it is meant to handle (hence the 0.5 factor; on the validation plants it was better or equal in all three cases, but it is a hand-chosen safety margin, not a derived one), and its predictions near the edge of the training range vary strongly; (iv) the closed loop is only identified well while excited and unsaturated; integrators are identified worse (differenced form).

**Versus classical self-tuning.** The same estimator with a model-based design is as good or better in most noisy cases and gives the most robust performance on Hammerstein plants; the ML predictor adds value only if the design target (ITAE-optimal for the step) matters more than robustness, e.g. the noise-free case with a lag decrease. There is no evidence in this study that "ML" is the better ingredient of an adaptive scheme; the estimator is.

**Limits.** Scenarios are model changes on randomized linear/Hammerstein plants (no friction growth, load disturbances beyond the nominal, or setpoint-range change on a nonlinear plant); the reference is not an oracle for the metric; single seed per plant and protocol (80 plants per scenario give the spread; medians are reported, the tails are in the CSVs); the detune factor and window rules were chosen by hand on 14 plants; the RL agent is not part of this comparison; the benchmark plants of M0-M5 (DC motor, link, tank) were not used (their parameters are not part of a `Plant` interface that supports a mid-run change; a `SwitchingPlant` of two `DCMotorParams` would be the natural extension).
