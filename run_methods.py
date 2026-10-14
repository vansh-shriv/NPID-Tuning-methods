"""
Single entry point of the methods study (second round, docs/METHODS_ROADMAP.md): regenerates the per-phase results and the
final cross-method benchmark. Long, resumable-by-stage; each stage is also runnable on its own.

    python run_methods.py                      # M7 only (benchmark of all 18 methods) using the existing data/models
    python run_methods.py --all                # every phase M1..M7 (several hours; needs data/m4, models/m4)
    python run_methods.py --only m3 m7         # selected stages

Approximate runtimes on 16 cores: m1 10 min, m2 20 min, m3a/m3 60 min, m4a (dataset) 68 min, m4b 30 min, m4c 20 min,
m4d (RL training) 83 min per seed, m5a 70 min, m5b 40 min, m6 15 min per 40 plants, m7 prepare 2 min + run 40-60 min.
The dataset (data/m4) and the PPO models (models/m4) are not versioned; `m4a` and `m4d train` regenerate them.
"""

import argparse
import subprocess
import sys
import time

STAGES = [
    ("m1", ["experiments/m1_classical.py"]),
    ("m2", ["experiments/m2_intelligent.py", "run"]),
    ("m3a", ["experiments/m3a_best_known.py"]),
    ("m3", ["experiments/m3_optimizers.py", "run"]),
    ("m4a", ["experiments/m4a_dataset.py"]),
    ("m4b", ["experiments/m4b_supervised.py", "compare"]),
    ("m4c", ["experiments/m4c_bayesopt.py", "run"]),
    ("m4d", ["experiments/m4d_rl.py", "train"]),
    ("m5a", ["experiments/m5a_ga_ml.py", "run", "160"]),
    ("m5b", ["experiments/m5b_hybrid.py", "run"]),
    ("m6", ["experiments/m6_adaptive.py", "run", "test", "40", "test_a", "0"]),
    ("m7-prepare", ["experiments/m7_benchmark.py", "prepare"]),
    ("m7-run", ["experiments/m7_benchmark.py", "run"]),
    ("m7-analyze", ["experiments/m7_benchmark.py", "analyze"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run every phase (hours)")
    ap.add_argument("--only", nargs="*", help="stage ids, e.g. m3 m7-run")
    a = ap.parse_args()
    for sid, cmd in STAGES:
        wanted = (a.only and any(sid == o or sid.startswith(o + "-") for o in a.only)) or \
                 (not a.only and (a.all or sid.startswith("m7")))
        if not wanted:
            continue
        print(f"[{sid}] python {' '.join(cmd)}", flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable] + cmd)
        print(f"[{sid}] done in {time.time() - t0:.0f} s (exit {rc})", flush=True)
        if rc != 0:
            sys.exit(rc)


if __name__ == "__main__":
    main()
