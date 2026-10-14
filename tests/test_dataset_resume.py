import json
import os

from src.legacy.generate_dataset import generate_dataset


def test_generation_resumes_from_checkpoint(tmp_path):
    out = str(tmp_path / "ds.csv")
    # Pretend trials 0 and 1 were finished by an interrupted run.
    with open(out + ".partial.jsonl", "w") as f:
        for i in (0, 1):
            f.write(json.dumps(dict(status="success", trial=i, J=0.01, B=0.1, Tc=0.05, T_load=0.0,
                                    omega_ref=20.0, Kp=1.0, Ki=2.0, Kd=3.0, achieved_ITAE=0.5)) + "\n")
    generate_dataset(num_samples=4, output_path=out, seed=1, num_workers=1)
    rows = open(out).read().strip().splitlines()
    assert len(rows) == 1 + 4                                # header + 4 trials
    assert rows[1].split(",")[5:8] == ["1.000000", "2.000000", "3.000000"]   # checkpointed rows kept, ordered by trial
    assert not os.path.exists(out + ".partial.jsonl")
