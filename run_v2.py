"""
Single entry point that regenerates every v2 result and figure (ROADMAP Phase 7).

    python run_v2.py                 # everything except the long dataset generation (uses existing data/v2)
    python run_v2.py --with-dataset  # also regenerates data/v2/dataset_npid.csv (~1 h on 14 workers)
    python run_v2.py --only 01 05    # run selected experiments

Order matters: 04 needs the dataset, 06 needs the model saved by 04, 07 needs all CSVs.
Approximate runtimes on 16 cores: 01 12 min, 02 1 min, 03a 2 min, 03 60 min, 04 5 min, 05 7 min, 06 1 min, 07 10 s.
"""

import argparse
import subprocess
import sys
import time

STEPS = [
    ("02", "experiments/v2/02_zn_validation.py"),
    ("03a", "experiments/v2/03a_pso_convergence.py"),
    ("01", "experiments/v2/01_fair_comparison.py"),
    ("03", "experiments/v2/03_generate_dataset_v2.py"),      # long; only with --with-dataset
    ("04", "experiments/v2/04_ml_models.py"),
    ("05", "experiments/v2/05_multi_plant.py"),
    ("06", "experiments/v2/06_drift_baselines.py"),
    ("07", "experiments/v2/07_make_figures.py"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-dataset", action="store_true", help="regenerate the 1200-row dataset (long)")
    ap.add_argument("--only", nargs="*", help="experiment ids to run, e.g. 01 05 07")
    a = ap.parse_args()

    for step_id, script in STEPS:
        if a.only and step_id not in a.only:
            continue
        if step_id == "03" and not (a.with_dataset or (a.only and "03" in a.only)):
            print(f"[skip] {script} (pass --with-dataset to regenerate)")
            continue
        print(f"\n=== {script} ===", flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable, script])
        print(f"=== {script} finished in {time.time() - t0:.0f}s (exit {rc}) ===", flush=True)
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main()
