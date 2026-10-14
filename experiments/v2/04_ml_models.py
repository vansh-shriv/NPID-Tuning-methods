"""
Experiment 04 - ML gain predictors on dataset v2 (ROADMAP Phase 5)

Compares, on the same held-out plants:
    constant   : one fixed gain set (median of training labels) - the "does ML add anything?" baseline
    ridge(poly), knn, random forest, extra trees, hist-gradient-boosting, MLP
Targets are log10(Kp, Ki, Kd) (gains span decades); predictions are converted back to physical units.

Metrics
    * R2 / MAE on log-gains (labels are only weakly identifiable - PSO gains vary a lot between seeds
      at equal cost, see 03a - so these are secondary),
    * downstream regret: closed-loop ITAE with predicted gains / ITAE with PSO gains, on each test plant,
    * speed: fresh PSO vs model.predict,
    * RF permutation feature importance,
    * extrapolation: ML vs fresh PSO on plants outside the training ranges.

Run from repo root (after 03_generate_dataset_v2.py):  python experiments/v2/04_ml_models.py
"""

import importlib.util
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.metrics import compute_metrics
from src.plant import DCMotorParams
from src.simulate import run_simulation
from src.tuning_generic import make_controller, tune_pso

# reuse constants/helpers of the generator (module name starts with a digit)
_spec = importlib.util.spec_from_file_location(
    "gen_v2", os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_generate_dataset_v2.py"))
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

DATA = "data/v2/dataset_npid.csv"
OUT = "results/v2"
SEED = 0
FEATURES, GAINS = gen.FEATURES, ["Kp", "Ki", "Kd"]


def plant_from_row(r):
    return DCMotorParams(J=r["J"], B=r["B"], Tc=r["Tc"], R=r["R"]), r["T_load"], r["omega_ref"]


def sim_itae(row, gains):
    """ITAE of the closed loop under the dataset's scenario for a given gain triple (dict or sequence)."""
    if not isinstance(gains, dict):
        gains = dict(zip(GAINS, gains))
    p, T_load, w = plant_from_row(row)
    ctrl = make_controller("npid", gains, p, 0.001, e_scale=w, de_scale=10 * w)
    res = run_simulation(ctrl, p, w, t_end=gen.T_END, dt=0.001, T_load_func=lambda t: T_load if t >= gen.T_LOAD_ON else 0.0)
    if not np.all(np.isfinite(res.omega)) or np.max(np.abs(res.omega)) > 10 * w:
        return 1e6
    return float(compute_metrics(res.time, w, res.omega)["ITAE"])


def models():
    sc = StandardScaler
    return {
        "ridge-poly2": make_pipeline(sc(), PolynomialFeatures(2), Ridge(alpha=1e-2)),
        "knn": make_pipeline(sc(), KNeighborsRegressor(n_neighbors=7, weights="distance")),
        "random-forest": make_pipeline(sc(), RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)),
        "extra-trees": make_pipeline(sc(), ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1)),
        "hist-gb": make_pipeline(sc(), MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=SEED))),
        "mlp": make_pipeline(sc(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000, early_stopping=True, random_state=SEED)),
    }


def regret_stats(ratios):
    r = np.asarray(ratios)
    return dict(median_ratio=np.median(r), p90_ratio=np.percentile(r, 90), within_1p5x_pct=100 * np.mean(r <= 1.5),
                failed_pct=100 * np.mean(r >= 1e3))


def pso_job(row):
    p, T_load, w = plant_from_row(row)
    t0 = time.perf_counter()
    g, c = tune_pso("npid", p, w, None, n_particles=gen.N_PARTICLES, iters=gen.ITERS, t_end=gen.T_END, seed=1,
                    T_load_func=lambda t: T_load if t >= gen.T_LOAD_ON else 0.0, bounds=gen.BOUNDS, log_space=True,
                    e_scale=w, de_scale=10 * w)
    return c, time.perf_counter() - t0


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    df = pd.read_csv(DATA)
    print(f"{len(df)} rows; at_bound {df.at_bound.mean():.1%}; median overshoot {df.overshoot_pct.median():.2f}%")

    # --- label quality filter: optimizer stuck at the box edge, or poor controller ---
    good = df[(~df.at_bound) & (df.overshoot_pct < 15.0)].reset_index(drop=True)
    print(f"after quality filter: {len(good)} rows")

    X = good[FEATURES].to_numpy()
    Y = np.log10(good[GAINS].to_numpy())
    idx = np.arange(len(good))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED)

    results, preds = [], {}
    # constant-gain baseline
    const = np.median(Y[tr], axis=0)
    preds["constant"] = np.tile(const, (len(te), 1))
    for name, m in models().items():
        cv = cross_val_predict(m, X[tr], Y[tr], cv=KFold(5, shuffle=True, random_state=SEED))
        m.fit(X[tr], Y[tr])
        preds[name] = m.predict(X[te])
        results.append(dict(model=name, cv_r2_logK=r2_score(Y[tr], cv)))

    ref_itae = good.loc[te, "achieved_ITAE"].to_numpy()
    rows = good.loc[te].to_dict("records")
    summary = []
    for name, P in preds.items():
        itae = np.array([sim_itae(r, 10 ** p) for r, p in zip(rows, P)])
        ratio = itae / ref_itae
        row = dict(model=name, **{f"r2_{g}": r2_score(Y[te][:, i], P[:, i]) for i, g in enumerate(GAINS)},
                   **{f"mae_log_{g}": mean_absolute_error(Y[te][:, i], P[:, i]) for i, g in enumerate(GAINS)},
                   **regret_stats(ratio))
        summary.append(row)
    summ = pd.DataFrame(summary)
    summ = summ.merge(pd.DataFrame(results), on="model", how="left")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n=== held-out plants (regret ratio = ITAE(ML gains) / ITAE(PSO gains)) ===")
    print(summ.round(3).to_string(index=False))
    summ.to_csv(f"{OUT}/04_ml_comparison.csv", index=False)

    # --- pick best learned model by median regret, retrain on all rows, save ---
    learned = summ[summ.model != "constant"].sort_values(["median_ratio", "p90_ratio"])
    best = learned.iloc[0]["model"]
    print(f"\nbest learned model by median regret: {best}")
    final = models()[best].fit(X, Y)
    joblib.dump(dict(model=final, features=FEATURES, log_targets=True, gains=GAINS, trained_on=len(good),
                     controller="npid e_scale=w_ref de_scale=10*w_ref"), "models/v2/gain_predictor_v2.joblib")

    # --- feature importance (permutation, RF refit on train) ---
    rf = models()["random-forest"].fit(X[tr], Y[tr])
    imp = permutation_importance(rf, X[te], Y[te], n_repeats=20, random_state=SEED)
    fi = pd.DataFrame(dict(feature=FEATURES, importance=imp.importances_mean, std=imp.importances_std))
    print("\n=== RF permutation importance (R2 drop) ===")
    print(fi.round(4).to_string(index=False))
    fi.to_csv(f"{OUT}/04_feature_importance.csv", index=False)

    # --- speed: fresh PSO vs predict, on 4 unseen held-out plants ---
    sample_rows = rows[:4]
    with ProcessPoolExecutor(max_workers=4) as ex:
        pso_out = list(ex.map(pso_job, sample_rows))
    t_pso = np.mean([t for _, t in pso_out])
    m = models()[best].fit(X[tr], Y[tr])
    t0 = time.perf_counter()
    for _ in range(200):
        m.predict(X[te][:1])
    t_pred = (time.perf_counter() - t0) / 200
    print(f"\nspeed: PSO {t_pso:.1f} s per plant  vs  {best}.predict {t_pred * 1e3:.2f} ms  ->  {t_pso / t_pred:,.0f}x")

    # --- extrapolation: beyond-training-range plants, ML vs fresh PSO ---
    base = dict(J=0.0125, B=0.125, Tc=0.06, T_load=0.025, R=1.25)
    sweeps = dict(J=[0.02, 0.03, 0.05, 0.08], B=[0.2, 0.3, 0.5], Tc=[0.1, 0.15, 0.2], R=[2.0, 3.0, 4.0])
    cases = []
    for k, vals in sweeps.items():
        for v in vals:
            p = {**base, k: v}
            pl = DCMotorParams(J=p["J"], B=p["B"], Tc=p["Tc"], R=p["R"])
            p["omega_ref"] = 0.5 * gen.w_max(pl, p["T_load"])
            p["swept"] = f"{k}={v}"
            p["in_range"] = gen.RANGES[k][0] <= v <= gen.RANGES[k][1]
            cases.append(p)
    with ProcessPoolExecutor(max_workers=14) as ex:
        pso_costs = [c for c, _ in ex.map(pso_job, cases)]
    ext = []
    for c, pc in zip(cases, pso_costs):
        g = 10 ** final.predict(np.array([[c[f] for f in FEATURES]]))[0]
        ml = sim_itae(c, g)
        ext.append(dict(case=c["swept"], in_range=c["in_range"], omega_ref=round(c["omega_ref"], 1),
                        ITAE_pso=pc, ITAE_ml=ml, ratio=ml / pc))
    ext = pd.DataFrame(ext)
    print("\n=== extrapolation (regret ratio = ML / PSO, plant at 50% of w_max) ===")
    print(ext.round(3).to_string(index=False))
    ext.to_csv(f"{OUT}/04_extrapolation.csv", index=False)
    pd.DataFrame([dict(best_model=best, pso_seconds=t_pso, predict_seconds=t_pred, speedup=t_pso / t_pred)]).to_csv(
        f"{OUT}/04_speed.csv", index=False)


if __name__ == "__main__":
    main()
