"""M6 pooled analysis of the two independent test subsets (test_a, test_b) + the example trajectory figure."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT, FIG = "results/methods/M6", "results/methods/M6/figures"
M = ["frozen", "adaptive ML", "self-tuning pole placement", "retune by experiment"]

def pooled():
    r = pd.concat([pd.read_csv(f"{OUT}/M6_ratios_test_a.csv").assign(subset="a"), pd.read_csv(f"{OUT}/M6_ratios_test_b.csv").assign(subset="b")])
    r["phase"] = np.where(r.window <= 1, "pre", np.where(r.window == 2, "first post", "adapted (w3-4)"))
    r.to_csv(f"{OUT}/M6_ratios_pooled.csv", index=False)
    p = r.groupby(["noise", "change", "method", "phase"]).ratio.median().unstack().loc[:, ["pre", "first post", "adapted (w3-4)"]].round(2)
    p.to_csv(f"{OUT}/M6_pooled_median.csv"); print(p.to_string())
    # win rates in the adapted phase (per plant: mean ratio of w3,w4), and share within 2x of the reference
    a = r[r.window >= 3].groupby(["ident", "subset", "change", "noise", "method"]).ratio.mean().unstack()
    rows = []
    for (ch, no), g in a.groupby(level=["change", "noise"]):
        rows.append(dict(change=ch, noise=no, n=len(g), ml_beats_frozen=(g["adaptive ML"] < g["frozen"]).mean(), pp_beats_frozen=(g["self-tuning pole placement"] < g["frozen"]).mean(),
                         ml_within2x=(g["adaptive ML"] <= 2).mean(), pp_within2x=(g["self-tuning pole placement"] <= 2).mean(), frozen_within2x=(g["frozen"] <= 2).mean(),
                         ml_beats_pp=(g["adaptive ML"] < g["self-tuning pole placement"]).mean()))
    w = pd.DataFrame(rows).round(2); w.to_csv(f"{OUT}/M6_winrates.csv", index=False); print(w.to_string())
    t = r[(r.noise) & (r.window >= 3) & (r.change != "none x1.0")].groupby(["method", "ptype"]).ratio.median().unstack().round(2); print(t.to_string()); t.to_csv(f"{OUT}/M6_by_type.csv")

def example():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from dataclasses import replace
    from experiments.m5_common import DATA, load_main
    from experiments.m5b_hybrid import build_variants
    from experiments.m6_adaptive import CFG, T_SWITCH
    from src.plant_family import problem_from_spec, spec_from_dict
    from src.tuners.adaptive import AdaptivePolicy, ClippedModel, FrozenPolicy, ScheduledPolicy, run_scenario, sampling_steps, scaled_spec
    from src.tuners.features import denormalise_gains, extract_features
    from src.tuners.ml import load_bundle
    b = load_main(); bc = load_bundle(DATA)
    tn = build_variants(b.select(b.split == "train")); tc = build_variants(bc.select(bc.split == "train"))
    svr_n, svr_e = tn["SVR"].model, ClippedModel(tc["SVR"].model, bc.select(bc.split == "train").X_eng)
    ev = b.select(b.split == "test")
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=False)
    for col, (ident, change) in enumerate([(None, ("gain", 2.5)), (None, ("lag", 0.4))]):
        pass
    pick = [i for i in range(len(ev)) if ev.ptype[i] == "fopdt"][:2]
    for col, (i, change) in enumerate(zip(pick, [("gain", 2.5), ("lag", 0.4)])):
        spec = spec_from_dict(ev.specs[i]); p = problem_from_spec(spec); H = spec.horizon()
        f = extract_features(p, noise_sigma=p.noise_sigma)
        g0 = denormalise_gains(np.asarray(svr_n.predict(f.vec[None, :]))[0], f.K_eff, f.T_n)
        lb, ub = p.bounds; g0 = {k: float(np.clip(v, lo, hi)) for (k, v), lo, hi in zip(g0.items(), lb, ub)}
        m = sampling_steps(f.T_n, f.model["theta"], p.dt, CFG.delta_div)
        hyb = tn["Cohen-Coon + SVR + refine"]
        ga_, gb_ = hyb.tune(p).gains, hyb.tune(problem_from_spec(scaled_spec(spec, change))).gains
        pols = {"frozen": FrozenPolicy(), "adaptive ML": AdaptivePolicy(CFG, svr_e), "self-tuning pole placement": AdaptivePolicy(replace(CFG, mode="pp")), "retune by experiment": ScheduledPolicy([(0, ga_), (T_SWITCH * H, gb_)])}
        for name, pol in pols.items():
            r = run_scenario(spec, change, pol, g0, m, True, 0, T_SWITCH, f.kind, f.T_n, p)
            axes[0, 2 * col].plot(r.t / H, r.y, label=name, lw=1)
            tr = np.array([(t, g["Kp"]) for t, g in r.gain_trace]); axes[1, 2 * col].step(tr[:, 0] / H, tr[:, 1], where="post", label=name)
        axes[0, 2 * col].plot(r.t / H, r.r, "k--", lw=0.6); axes[0, 2 * col].axvline(T_SWITCH, color="gray", ls=":")
        axes[0, 2 * col].set_title(f"{spec.ptype} #{spec.ident}, change {change[0]} x{change[1]}: output"); axes[1, 2 * col].set_yscale("log"); axes[1, 2 * col].set_title("Kp"); axes[1, 2 * col].set_xlabel("time / H")
        axes[0, 2 * col].set_ylim(-0.2 * p.y_ref, 2.0 * p.y_ref)
        axes[0, 2 * col].legend(fontsize=6)
        axes[0, 2 * col + 1].axis("off"); axes[1, 2 * col + 1].axis("off")
    fig.tight_layout(); fig.savefig(f"{FIG}/M6_example_traces.png", dpi=120)

if __name__ == "__main__":
    pooled(); example()
