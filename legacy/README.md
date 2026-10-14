# legacy/ - original Weeks 1-8 runners (kept for the record)

`run_week1.py` (plant + L-PID/N-PID baselines), `run_week4.py` (noise/extrapolation comparison, RF model), `run_week5.py`-`run_week7.py` (TD3 environment, training, drift experiment), `run_week8.py` (generalisation), `eval_week6_hires.py` (high-resolution evaluation of the trained agent). Their modules are in `src/legacy/`, outputs in `results/legacy/`, data in `data/legacy/`, models in `models/legacy/`, reports in `report/legacy/`.

Run from the repository root: `python legacy/run_week1.py`.

**Do not cite their numbers**: the original dataset was invalid for 42% of its rows, ZN used a limit-cycle artefact, the "frozen ML" drift baseline silently used PSO gains, and the Week-1 comparison tuned only the N-PID. Each correction is documented in `progress.md`; the corrected study is `report/final_report_v2.md` (v2) and `report/methods_report.md` (methods study).
