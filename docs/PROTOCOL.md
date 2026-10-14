# Experimental Protocol for the Tuning-Method Study

All methods in `docs/METHODS_ROADMAP.md` are evaluated under this protocol. It is fixed before any method is implemented so that results cannot be tuned to favour a method. Changes to the protocol are logged in the "Protocol changes" section with a reason and a date.

## 1. Research questions
- **RQ1 (performance):** how good are the gains each method produces, when applied to the same N-PID controller on the same plant?
- **RQ2 (cost):** what does it take to obtain those gains (plant experiments, simulations, wall time, prior knowledge)?
- **RQ3 (generality):** on which plant types does each method work, fail, or silently produce poor gains?
- **RQ4 (robustness):** how do the gains behave under disturbance and measurement noise?
- **RQ5 (nonlinearity):** what does the N-PID's error shaping add over a linear PID with the same gains?

## 2. Generality requirement
No method implementation may contain plant-specific constants. A method receives only a `TuningProblem` (a plant factory it can simulate, the setpoint, the sampling time, the horizon, actuator limits, and optional disturbance/noise settings) and must work for any single-input single-output plant that is *stable or integrating, positive-gain, and simulable*. Plants outside this class (unstable, negative-gain, multi-variable) are out of scope and a method must say so explicitly instead of returning gains. Anything a method knows about the plant it must learn by experimenting on the plant through the problem object, which counts the experiments.

## 3. Benchmark plants (fixed)
| id | description | what it tests |
|---|---|---|
| `dc_speed` | armature DC motor speed, Coulomb friction, 24 V limit | 2nd-order electromechanical, saturation, friction nonlinearity |
| `dc_position` | same motor, output = shaft angle | integrating plant |
| `link` | single-link arm with gravity torque, Coulomb friction | strongly nonlinear, lightly damped |
| `fopdt` | K=2, tau=5, theta=1 | classical process model with dead time |
| `tank` | nonlinear tank level, sqrt outflow | state-dependent gain |

Additional randomized plant families are introduced in the ML phase (M4) and are documented there. Unit tests use plants with analytically known answers (exact FOPDT, IPDT, second-order).

## 4. Scenarios
For every plant and every method the final gains are evaluated, without further tuning, on:
1. **Step**: setpoint step from rest (no disturbance, no noise). This is also the training scenario of optimization and ML methods.
2. **Disturbance**: the step plus an input-side step disturbance at mid-horizon.
3. **Noise**: the step with white Gaussian measurement noise (sigma set per plant, about 1-2% of the setpoint), 10 independent noise seeds.

Methods that require a different scenario (adaptive methods, M6) additionally use drift and regime-change scenarios defined in that phase.

## 5. Controller under test
N-PID: `P = Kp*alpha(e)`, integral with `beta(e)`, derivative with `gamma(de/dt)` (see `src/controllers.py`), `n = m = p = 2`, scale-free shaping `e_scale = |y_ref|`, `de_scale = |y_ref|/Ts`, derivative low-pass `tau_d = Ts/20`, back-calculation anti-windup, actuator saturation from the plant. The identical gains are also applied to an L-PID (same `tau_d`) to isolate the effect of the nonlinearity (RQ5). Methods that design *linear* PID gains use them as the N-PID base gains; this is the approximation the project specification makes (near e = 0 the shaping functions are about 1) and each phase measures how good it is by comparing the N-PID with the L-PID.

`Ts` is a characteristic time scale used only for shaping, filtering and search-box scaling. It is estimated automatically from an open-loop step response (time to 63% of the final value, or horizon/5 for integrating plants) and is part of the problem specification, not of any method.

## 6. Metrics
Rise time (10-90%), overshoot (%), settling time (2%), IAE, ISE, ITAE (`src/metrics.py`), plus control effort (mean absolute change of the control signal per step) and a **failure flag** (non-finite response, output beyond 10x the setpoint, or ITAE above 100x the best method's value on that plant/scenario). Tuning cost is reported as: number of closed/open-loop simulations, simulated seconds (a proxy for experiment time on a real plant), wall-clock seconds.

## 7. Budget and seeds
- Stochastic methods: 5 seeds (0-4); report median, interquartile range and best. Deterministic methods: one run.
- Optimization methods (M3): 400 closed-loop simulations per run (the budget of PSO with 16 particles x 25 iterations). Sample-efficient methods (Gaussian-process optimization): 60 simulations, and additionally the 400-evaluation budget for comparison.
- Gains are searched in log-space inside a generic box derived from the actuator range, setpoint and `Ts` (`default_bounds`); no method gets a hand-tuned box.

## 8. Statistical treatment
Report medians with interquartile ranges over seeds and noise realizations. A difference between two methods is only called a difference when it exceeds the pooled seed-to-seed spread; otherwise they are "indistinguishable". Rankings use ITAE on the step and noise scenarios and are shown with the spread. No result is discarded; failures are reported as failures.

## 9. Validation before evaluation
Each method is first validated against known answers (analytic formulas on exact plants, optimizer test functions, recovery of a known model) in `tests/` before any benchmark result is produced. A benchmark number from a method that fails its validation tests is not used.

## 10. Documentation per method
Every method gets: algorithm description with equations and references, assumptions and applicability, implementation notes, validation evidence, benchmark results (tables and plots), analysis, pros and cons, and observed errors/failure modes. These are collected in `report/methods/` (one file per phase) and merged into `report/methods_report.md`.

## Protocol changes
- **P1 (M1, widen the search box).** `default_bounds` initially spanned Kp in [0.05, 20] kp0, Ki in [0.01, 10] kp0/Ts, Kd in [0.01, 10] kp0 Ts. In M1 a classical rule applied outside its validity range reached ITAE 0.92 on the tank while the PSO reference found 1.6: the box excluded the optimum (the best controller for a plant without dead time is very aggressive, limited only by actuator saturation and sampling). Changed to Kp in [0.01, 100] kp0, Ki in [1e-3, 1e3] kp0/Ts, Kd in [1e-3, 1e2] kp0 Ts. Reason: a reference optimizer must not be handicapped by the protocol. Effect: only the M0/M1 PSO reference used the old box (the v2 experiments used their own box and are unaffected).
