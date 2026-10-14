"""
M4b - Supervised and ANN tuners on the plant-family dataset (methods 12 'Regression (SVR/NN)' and 15 'ANN').

Stages
  compare   all models trained on the train split; z-error and closed-loop regret on val / test / ood
  ablate    learning curve, feature set (engineered / signature / both / raw = no normalisation), label variant,
            ANN loss (supervised / decision / combined), feature noise
  final     the chosen models evaluated on test + ood with the full protocol scenarios (step, load, noise)
  analyze   tables and figures

Regret of a plant = ITAE(model gains) / ITAE(label), both on the clean step with the N-PID. The label is the best
of GA and PSO (400 simulations each), so regret < 1 is possible: the label is not the true optimum.

Run from the repo root:  python experiments/m4b_supervised.py compare|ablate|final|analyze
"""

import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.plant_family import problem_from_spec, spec_from_dict
from src.tuners.base import GAIN_NAMES
from src.tuners.benchmark import evaluate_gains
from src.tuners.features import denormalise_gains
from src.tuners.ml import ANNPolicy, load_bundle, make_model

warnings.filterwarnings("ignore")
DATA = "data/m4/dataset.jsonl"
OUT = "results/methods/M4"
FIG = f"{OUT}/figures"
WORKERS = 12
MODELS = ["constant", "ridge-poly2", "knn", "svr", "mlp", "extra-trees"]
FEATS = os.environ.get("M4_FEATURES", "noisy")          # primary protocol: features from a noisy bump test
SUF = "_noisy" if FEATS == "noisy" else ""


def load_main():
    from src.tuners.ml import noisy_bundle
    b = load_bundle(DATA)
    return noisy_bundle(b, "data/m4/features_noisy.npz") if FEATS == "noisy" else b


# ----------------------------------------------------------------------------- closed-loop evaluation
def _regret_job(args):
    spec_d, z, K_eff, T_n, label_cost = args
    spec = spec_from_dict(spec_d)
    p = problem_from_spec(spec)
    g = denormalise_gains(z, K_eff, T_n)
    lb, ub = p.bounds
    g = {k: float(np.clip(g[k], lo, hi)) for k, lo, hi in zip(GAIN_NAMES, lb, ub)}
    itae = p.itae(p.simulate(g, "npid", "step", count=False))
    return itae, itae / label_cost


def regret(bundle, Z, pool):
    jobs = [(bundle.specs[i], Z[i], bundle.K_eff[i], bundle.T_n[i], bundle.best_cost[i]) for i in range(len(bundle))]
    out = pool.map(_regret_job, jobs, chunksize=4)
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def summarize(ratio):
    r = np.asarray(ratio)
    return dict(median=float(np.median(r)), mean_log=float(np.exp(np.mean(np.log(np.maximum(r, 1e-6))))),
                p90=float(np.percentile(r, 90)), within1p5=float(np.mean(r <= 1.5)), within1p1=float(np.mean(r <= 1.1)),
                bad=float(np.mean(r >= 10.0)), n=len(r))


def fit_predict(name, train, X_eval, features="engineered", variant="best", seed=0):
    if name.startswith("ann"):
        loss = {"ann": "combined", "ann-supervised": "supervised", "ann-decision": "decision"}[name]
        torch_threads()
        m = ANNPolicy(features=features, loss=loss, seed=seed, label_variant="top5" if variant == "best" else variant).fit(train)
        return m.predict(X_eval), m
    m = make_model(name, seed=seed)
    m.fit(train.X(features), train.targets(variant))
    return m.predict(X_eval), m


def torch_threads(n=4):
    import torch
    torch.set_num_threads(n)


def raw_targets(b):
    """log10 of the physical gains of the label (no normalisation)."""
    z = b.best_z
    return np.stack([z[:, 0] - np.log10(b.K_eff), z[:, 1] - np.log10(b.K_eff * b.T_n),
                     z[:, 2] + np.log10(b.T_n) - np.log10(b.K_eff)], axis=1)


def raw_to_z(raw, b):
    return np.stack([raw[:, 0] + np.log10(b.K_eff), raw[:, 1] + np.log10(b.K_eff * b.T_n),
                     raw[:, 2] - np.log10(b.T_n) + np.log10(b.K_eff)], axis=1)


# ----------------------------------------------------------------------------- stages
def stage_compare():
    b = load_main()
    tr = b.select(b.split == "train")
    print("plants:", {s: int((b.split == s).sum()) for s in ("train", "val", "test", "ood")}, flush=True)
    rows, preds = [], {}
    with Pool(WORKERS) as pool:
        for name in MODELS + ["ann-supervised", "ann-decision", "ann"]:
            feats = "signature" if name.startswith("ann") else "engineered"
            for split in ("val", "test", "ood"):
                ev = b.select(b.split == split)
                Z, _ = fit_predict(name, tr, ev.X(feats), features=feats)
                itae, ratio = regret(ev, Z, pool)
                zmae = np.abs(Z - ev.best_z).mean(axis=0)
                rows.append(dict(model=name, features=feats, split=split, z_mae_Kp=zmae[0], z_mae_Ki=zmae[1], z_mae_Kd=zmae[2],
                                 **summarize(ratio)))
                for i, r_ in enumerate(ratio):
                    preds.setdefault(name, []).append(dict(split=split, ident=int(ev.ident[i]), ptype=str(ev.ptype[i]), ratio=float(r_)))
                print(name, split, f"median {rows[-1]['median']:.3f} p90 {rows[-1]['p90']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/M4_compare{SUF}.csv", index=False)
    pd.DataFrame([dict(model=m, **r) for m, rs in preds.items() for r in rs]).to_csv(f"{OUT}/M4_compare_perplant{SUF}.csv", index=False)


def stage_ablate():
    b = load_main()
    tr, va = b.select(b.split == "train"), b.select(b.split == "val")
    rows = []
    with Pool(WORKERS) as pool:
        def ev(tag, value, Z, model, extra=None):
            _, ratio = regret(va, Z, pool)
            rows.append(dict(ablation=tag, value=value, model=model, **summarize(ratio), **(extra or {})))
            print(tag, value, model, f"median {rows[-1]['median']:.3f}", flush=True)

        # learning curve (SVR and ANN)
        rng = np.random.default_rng(0)
        for n in (50, 100, 200, 400, len(tr)):
            idx = rng.permutation(len(tr))[:n]
            sub = tr.select(np.isin(np.arange(len(tr)), idx))
            ev("train size", n, fit_predict("svr", sub, va.X("engineered"))[0], "svr")
            ev("train size", n, fit_predict("ann", sub, va.X("signature"), features="signature")[0], "ann")
        # feature sets (SVR)
        for fs in ("engineered", "signature", "both"):
            ev("feature set", fs, fit_predict("svr", tr, va.X(fs), features=fs)[0], "svr")
        m = make_model("svr").fit(tr.X("raw"), raw_targets(tr))                       # no normalisation
        ev("feature set", "raw (no normalisation)", raw_to_z(m.predict(va.X("raw")), va), "svr")
        # label variants (SVR)
        for variant in ("best", "top5", "top1"):
            ev("label variant", variant, fit_predict("svr", tr, va.X("engineered"), variant=variant)[0], "svr")
        # ANN loss
        for name in ("ann-supervised", "ann-decision", "ann"):
            ev("ANN loss", name, fit_predict(name, tr, va.X("signature"), features="signature")[0], "ann")
        # ANN features
        for fs in ("engineered", "signature", "both"):
            ev("ANN features", fs, fit_predict("ann", tr, va.X(fs), features=fs)[0], "ann")
        # feature noise at inference (relative Gaussian noise on the features of the validation plants)
        svr = make_model("svr").fit(tr.X("engineered"), tr.targets("best"))
        for sigma in (0.0, 0.05, 0.1, 0.2):
            X = va.X("engineered") * (1.0 + np.random.default_rng(1).normal(0, sigma, va.X("engineered").shape))
            ev("feature noise", sigma, svr.predict(X), "svr")
    pd.DataFrame(rows).to_csv(f"{OUT}/M4_ablations{SUF}.csv", index=False)


def _full_eval_job(args):
    spec_d, z, K_eff, T_n, label_gains = args
    p = problem_from_spec(spec_from_dict(spec_d))
    g = denormalise_gains(z, K_eff, T_n)
    lb, ub = p.bounds
    g = {k: float(np.clip(g[k], lo, hi)) for k, lo, hi in zip(GAIN_NAMES, lb, ub)}
    ev = evaluate_gains(p, g, "npid", noise_seeds=tuple(range(4)))
    ref = evaluate_gains(p, label_gains, "npid", noise_seeds=tuple(range(4)))
    return dict(step=ev["step"]["ITAE"], load=ev["load"]["ITAE"], noise=ev["noise"]["ITAE"],
                step_ref=ref["step"]["ITAE"], load_ref=ref["load"]["ITAE"], noise_ref=ref["noise"]["ITAE"])


def stage_final():
    import json
    b = load_main()
    labels = {int(r["ident"]): r["best_gains"] for r in (json.loads(l) for l in open(DATA)) if r.get("status") == "ok"}
    tr = b.select(b.split == "train")
    rows = []
    with Pool(WORKERS) as pool:
        for name in ("constant", "svr", "mlp", "ann"):
            feats = "signature" if name == "ann" else "engineered"
            for split in ("test", "ood"):
                ev = b.select(b.split == split)
                Z, _ = fit_predict(name, tr, ev.X(feats), features=feats)
                jobs = [(ev.specs[i], Z[i], ev.K_eff[i], ev.T_n[i], labels[int(ev.ident[i])]) for i in range(len(ev))]
                for i, r in enumerate(pool.map(_full_eval_job, jobs, chunksize=4)):
                    rows.append(dict(model=name, split=split, ident=int(ev.ident[i]), ptype=str(ev.ptype[i]), **r))
            print("final", name, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/M4_final_eval{SUF}.csv", index=False)


def _noisy_feature_job(args):
    spec_d, sigma_rel, seed = args
    from src.tuners import NotApplicable
    from src.tuners.features import extract_features
    spec = spec_from_dict(spec_d)
    p = problem_from_spec(spec)
    try:
        f = extract_features(p, noise_sigma=sigma_rel * abs(spec.y_ref), seed=seed)
    except (NotApplicable, Exception):
        return None
    return f.vec, f.signature, f.K_eff, f.T_n


def stage_noisefeat():
    """Models fed with features measured by a *noisy* bump test (real identification noise), validation plants."""
    b = load_bundle(DATA)
    tr, va = b.select(b.split == "train"), b.select(b.split == "val")
    models = {"svr": (make_model("svr").fit(tr.X("engineered"), tr.targets("best")), "engineered"),
              "extra-trees": (make_model("extra-trees").fit(tr.X("engineered"), tr.targets("best")), "engineered")}
    torch_threads()
    ann = ANNPolicy(features="signature", loss="combined").fit(tr)
    rows = []
    with Pool(WORKERS) as pool:
        for sigma in (0.0, 0.005, 0.01, 0.02, 0.05):
            feats = pool.map(_noisy_feature_job, [(va.specs[i], sigma, 1) for i in range(len(va))])
            ok = [i for i, f in enumerate(feats) if f is not None]
            sub = va.select(np.isin(np.arange(len(va)), ok))
            sub.X_eng = np.array([feats[i][0] for i in ok]); sub.X_sig = np.array([feats[i][1] for i in ok])
            sub.K_eff = np.array([feats[i][2] for i in ok]); sub.T_n = np.array([feats[i][3] for i in ok])
            for name in ("svr", "extra-trees", "ann"):
                if name == "ann":
                    Z = ann.predict(sub.X("signature"))
                else:
                    m, fs = models[name]
                    Z = m.predict(sub.X(fs))
                _, ratio = regret(sub, Z, pool)
                rows.append(dict(noise_rel=sigma, model=name, identified=len(ok), **summarize(ratio)))
                print("noise", sigma, name, f"median {rows[-1]['median']:.3f} p90 {rows[-1]['p90']:.2f} ({len(ok)}/{len(va)})", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/M4_noisy_features.csv", index=False)


def _plant_noise_feature_job(args):
    spec_d, seed = args
    from src.tuners.features import extract_features
    spec = spec_from_dict(spec_d)
    p = problem_from_spec(spec)
    f = extract_features(p, noise_sigma=spec.noise_rel * abs(spec.y_ref), seed=seed)
    return f.vec, f.signature, f.K_eff, f.T_n


def noisy_copy(b, feats):
    c = b.select(np.ones(len(b), bool))
    c.X_eng = np.array([f[0] for f in feats]); c.X_sig = np.array([f[1] for f in feats])
    c.K_eff = np.array([f[2] for f in feats]); c.T_n = np.array([f[3] for f in feats])
    return c


def stage_noise_robust():
    """Train / test with features measured by a noisy bump test (each plant's own noise level), with and without the
    fit-error feature. Hypothesis: the clean-trained models fail because log_fit_error moves out of distribution."""
    b = load_bundle(DATA)
    with Pool(WORKERS) as pool:
        feats = pool.map(_plant_noise_feature_job, [(b.specs[i], 1000 + i) for i in range(len(b))], chunksize=8)
        bn = noisy_copy(b, feats)
        keep = [0, 1, 2, 3, 4, 6, 7]                                      # engineered features without log_fit_error
        tr_c, va_c = b.select(b.split == "train"), b.select(b.split == "val")
        tr_n, va_n = bn.select(bn.split == "train"), bn.select(bn.split == "val")
        rows = []
        for name in ("svr", "extra-trees"):
            for train_feats, tr, tag in (("clean", tr_c, "trained on clean features"), ("noisy", tr_n, "trained on noisy features")):
                for drop in (False, True):
                    cols = keep if drop else list(range(8))
                    m = make_model(name)
                    m.fit(tr.X("engineered")[:, cols], tr.targets("best"))
                    for test_feats, va in (("clean", va_c), ("noisy", va_n)):
                        Z = m.predict(va.X("engineered")[:, cols])
                        _, ratio = regret(va, Z, pool)
                        rows.append(dict(model=name, train=train_feats, test=test_feats, drop_fit_error=drop, **summarize(ratio)))
                        print(name, "train", train_feats, "test", test_feats, "drop" if drop else "keep", f"median {rows[-1]['median']:.3f} p90 {rows[-1]['p90']:.2f}", flush=True)
        # ANN on signatures (no fit-error feature)
        torch_threads()
        for train_feats, tr in (("clean", tr_c), ("noisy", tr_n)):
            ann = ANNPolicy(features="signature", loss="combined").fit(tr)
            for test_feats, va in (("clean", va_c), ("noisy", va_n)):
                _, ratio = regret(va, ann.predict(va.X("signature")), pool)
                rows.append(dict(model="ann", train=train_feats, test=test_feats, drop_fit_error=False, **summarize(ratio)))
                print("ann train", train_feats, "test", test_feats, f"median {rows[-1]['median']:.3f} p90 {rows[-1]['p90']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/M4_noise_robust.csv", index=False)
    np.savez(f"data/m4/features_noisy.npz", X_eng=bn.X_eng, X_sig=bn.X_sig, K_eff=bn.K_eff, T_n=bn.T_n, ident=bn.ident)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "compare"
    {"compare": stage_compare, "ablate": stage_ablate, "final": stage_final, "noisefeat": stage_noisefeat, "noiserobust": stage_noise_robust}[stage]()
