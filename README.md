# ML-Assisted Tuning of Nonlinear PID (N-PID) Controllers

Implementation and critical re-evaluation of the methods in Çelik et al., *"Advancements in nonlinear PID controllers: A comprehensive review,"* Computers and Electrical Engineering, 2026: linear PID vs error-shaped nonlinear PID (Eq. 9-12), Ziegler-Nichols and PSO tuning, machine-learning gain prediction, and online RL adaptation.

**Start here:** [`report/methods_report.md`](report/methods_report.md) (final report of the 18-method tuning study), [`report/final_report_v2.md`](report/final_report_v2.md) (v2 re-evaluation of the paper's controller comparison), [`progress.md`](progress.md) (engineering log), [`docs/METHODS_ROADMAP.md`](docs/METHODS_ROADMAP.md) / [`docs/ROADMAP_v2.md`](docs/ROADMAP_v2.md) (plans).

## Headline findings of the methods study (18 tuning methods, `report/methods_report.md`)

* All 18 methods of the review's taxonomy (classical, intelligent, optimization, ML, hybrid, adaptive) are implemented behind one `Tuner` interface and benchmarked on 5 plants x 4 scenarios (step, disturbance, noise, plant change).
* **Search wins when 400 simulations are affordable** (GA/PSO/ABC/SA within 1.0-1.3x of the best, 1.04-1.4x of the best-known optimum); everything cheaper is 1.6-33x worse. Cost/accuracy Pareto front: self-tuning (1 experiment) - GA-enhanced ML (4) - PPO search (12) - ML + classical hybrid with 20 simulations (24, 2.4x) - GP optimisation (60, 1.6x) - GA (362).
* **The classical rules are limited by their assumptions** (FOPDT-like plants): 5 of the 18 methods are not applicable or fail on at least one benchmark plant; SIMC is the robust one.
* **Aggressive (ITAE-optimal) gains are fragile** under noise (2x) and plant change (a 2.5x actuator gain makes the dead-time loop unstable for all optimizers and ML methods); conservative or adaptive methods survive. Adaptive ML works on randomized plants (M6) but failed on the short-horizon DC-position loop.
* N-PID vs L-PID with identical gains: better in about half of the cells under noise, similar or worse elsewhere.

## Headline findings (v2)

* **N-PID is not a free win.** With the same PSO tuning, a PSO-tuned L-PID matches the best N-PID on the DC-motor step response. N-PID's real advantage is noise robustness (~2x once L-PID gets a derivative filter and noise-aware tuning, not 50x) and nonlinear plants with state-dependent gain (tank level). It loses on a gravity-loaded link and a dead-time process.
* **The paper's shaping functions are not scale-free**; their behaviour depends on plant units and setpoint. A scale-free variant is provided.
* **Ziegler-Nichols is ill-posed for the DC motor** (no true ultimate gain); the original Kcr was a sampling/saturation limit cycle.
* **ML gain prediction is fast (~25,000x vs PSO) but a single fixed gain set is already within 3% of per-plant PSO** on this plant family. The original dataset was invalid for 42% of its rows (unreachable setpoints) and was regenerated.
* **RL adaptation is not justified for resistance drift**: frozen gains lose 7% at 3x drift; the trained TD3 agent is orders of magnitude worse.

The original Weeks 1-8 scripts and reports (`run_week*.py`, `report/legacy/final_report.md`, `report/legacy/phase5_rl_report.md`) are kept as a record; where they conflict with v2, v2 is correct (see `progress.md` for each correction).

## Layout (full guide with every file: [`docs/DIRECTORY_GUIDE.md`](docs/DIRECTORY_GUIDE.md))

```
README.md  CLAUDE.md  progress.md      this file | guidance for Claude Code | chronological engineering log
requirements*.txt  pytest.ini          dependencies, test config
run_v2.py  run_methods.py              regenerate the v2 study | regenerate the methods study (M1-M7)

src/                 library: plant, simulate, controllers, metrics, tuning_zn/relay/generic, plant_family, benchmark_plants
  tuners/            the 18 tuning methods + framework (base, identify, classical, selftuning, modelmatching, optimizers,
                     ml, rl, hybrid, adaptive, features, benchmark, plots, reference)
  legacy/            original Weeks 1-8 modules (dataset generator, RF models, TD3 RL, PSO) - kept for the record
experiments/
  v2/                01..07: the v2 re-evaluation of the paper's L-PID vs N-PID claims
  m1..m7_*.py        one script per phase of the methods study (m5_common.py is shared)
tests/               pytest suite (225 tests)
docs/                PROTOCOL.md (experimental protocol), METHODS_ROADMAP.md (plan of M0-M7), ROADMAP_v2.md, DIRECTORY_GUIDE.md
report/              methods_report.md (FINAL), methods/M1..M6 (per phase), final_report_v2.md, legacy/ (superseded)
results/             methods/M1..M7 (current), v2/, logs/ (run logs), legacy/ (original figures and tables)
data/                m4/ (plant-family dataset, regenerable), v2/ (v2 dataset), legacy/ (original, invalid rows)
models/              m4/ (PPO), m7/ (trained ML models), v2/ (gain predictor), legacy/ (original TD3 agents)
legacy/              run_week1..8.py, eval_week6_hires.py - original runners, kept for the record
```

## Setup and run

```bash
python -m venv .venv && .venv\Scripts\activate        # Windows (source .venv/bin/activate elsewhere)
pip install -r requirements-dev.txt
pytest                                                # 225 tests, 2-5 min
python run_v2.py                                      # all v2 experiments except dataset generation
python run_v2.py --with-dataset                       # + regenerate data/v2/dataset_npid.csv (~1 h on 14 workers)
python run_v2.py --only 01 07                         # selected experiments
python run_methods.py                                 # methods study: M7 benchmark of all 18 methods (about 1 h)
python run_methods.py --all                           # every phase M1..M7 (hours; see the header of run_methods.py)
```

Using the library on your own plant:

```python
from src.plant import TransferFunctionPlant, DelayedPlant
from src.controllers import NPID
from src.simulate import run_simulation
from src.tuning_generic import tune_pso

plant = DelayedPlant(TransferFunctionPlant([2.0], [5.0, 1.0], u_limits=(-10, 10)), delay=1.0)
gains, cost = tune_pso("npid", plant, y_ref=1.0, center=None, dt=0.02, t_end=60,
                       bounds=((0.5, 0.01, 0.01), (200, 20, 200)), log_space=True, restarts=3)
result = run_simulation(NPID(**gains, dt=0.02, u_min=-10, u_max=10), plant, 1.0, t_end=60, dt=0.02)
```

Notes: use `dt` equal to the controller step for both the tuner and the simulation; avoid non-ASCII characters in `print` output on Windows consoles (cp1252).

## Legacy pipeline

`python legacy/run_week1.py` ... `python legacy/run_week8.py` reproduce the original results (their README tables were replaced by the v2 findings above). Their ZN, dataset and RL-drift numbers have known issues listed in `progress.md` (Phases 2, 3, 5, 6).
