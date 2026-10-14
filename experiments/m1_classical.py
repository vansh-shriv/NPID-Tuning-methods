"""
M1 - Classical tuning methods (docs/METHODS_ROADMAP.md, docs/PROTOCOL.md).

Main benchmark:  8 classical variants + PSO reference, 5 plants, N-PID and L-PID with the same gains,
                 clean / disturbance / noise scenarios (10 noise seeds), tuning cost counted.
Ablations:       (A) identification method for the open-loop rules, (B) rules applied outside their validity
                 range, (C) rule sets for continuous cycling, (D) sensitivity of the limit-cycle 'ultimate
                 gain' to horizon and threshold, (E) identification under measurement noise.

Outputs: results/methods/M1/*.csv and results/methods/M1/figures/*.png
Run from the repo root:  python experiments/m1_classical.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.benchmark_plants import PLANT_NAMES
from src.tuners import TuningProblem
from src.tuners.benchmark import evaluate_gains, run_benchmark
from src.tuners.classical import (CohenCoon, ContinuousCycling, TimeDomainTuning, TrialAndError,
                                  ZieglerNicholsReactionCurve)
from src.tuners.identify import identify_process
from src.tuners.plots import (plot_cost_vs_performance, plot_heatmap, plot_identification, plot_step_overlays,
                              _save)
from src.tuning_zn import detect_limit_cycle

warnings.filterwarnings("ignore")
OUT = "results/methods/M1"
FIG = f"{OUT}/figures"
SEEDS = tuple(range(5))


def methods():
    return [TrialAndError(), ZieglerNicholsReactionCurve(), CohenCoon(),
            ContinuousCycling("zn"), ContinuousCycling("zn", source="relay"),
            ContinuousCycling("tyreus-luyben", source="relay"),
            TimeDomainTuning("simc"), TimeDomainTuning("chr0"), TimeDomainTuning("chr20")]


def md_table(df, floatfmt="{:.4g}"):
    """Minimal markdown table (no tabulate dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, [df.index.name or ""] + cols)) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = []
        for v in row:
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                         else ("n/a" if isinstance(v, (float, np.floating)) else str(v)))
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def pivot(df, value, kind, scenario, how="median"):
    d = df[(df.kind == kind) & (df.scenario == scenario) & df.success]
    t = d.groupby(["method", "plant"])[value].agg(how).unstack("plant")
    return t.reindex(columns=PLANT_NAMES)


# ----------------------------------------------------------------------------- main benchmark
def main_benchmark():
    from src.tuners.pso_ref import PSOReference
    tuners = methods() + [PSOReference()]
    df = run_benchmark(tuners, PLANT_NAMES, seeds=SEEDS, budget=400, workers=8)
    df.to_csv(f"{OUT}/M1_benchmark.csv", index=False)

    order = [t.name for t in tuners]
    for kind in ("npid", "lpid"):
        for scen in ("step", "load", "noise"):
            t = pivot(df, "ITAE", kind, scen).reindex(order)
            t.to_csv(f"{OUT}/M1_ITAE_{kind}_{scen}.csv")
            plot_heatmap(t, f"{FIG}/M1_heatmap_ITAE_{kind}_{scen}.png",
                         title=f"M1: ITAE, {kind.upper()}, {scen} scenario (median over seeds; colour = x best on the plant)")
    # N-PID / L-PID ratio with identical gains (RQ5)
    a, b = pivot(df, "ITAE", "npid", "step"), pivot(df, "ITAE", "lpid", "step")
    ratio = (a / b).reindex(order)
    ratio.to_csv(f"{OUT}/M1_ratio_npid_over_lpid_step.csv")
    an, bn = pivot(df, "ITAE", "npid", "noise"), pivot(df, "ITAE", "lpid", "noise")
    (an / bn).reindex(order).to_csv(f"{OUT}/M1_ratio_npid_over_lpid_noise.csv")

    # tuning cost
    cost = df[(df.kind == "npid") & (df.scenario == "step")].groupby(["method", "plant"]).agg(
        evals=("evals", "median"), sim_seconds=("sim_seconds", "median"), wall_time=("wall_time", "median"),
        ITAE=("ITAE", "median"), success=("success", "all")).reset_index()
    cost.to_csv(f"{OUT}/M1_cost.csv", index=False)
    plot_cost_vs_performance(cost[cost.success], f"{FIG}/M1_cost_vs_performance.png",
                             title="M1: tuning cost vs. result quality (N-PID, step)")

    # applicability matrix
    app = df[(df.kind == "npid") & (df.scenario == "step")].groupby(["method", "plant"]).apply(
        lambda g: "n/a: " + g.message.iloc[0].replace("not applicable: ", "")[:40] if not g.success.all() else
        ("limit-cycle artefact" if str(g.get("meta_limit_cycle_artefact", pd.Series([False])).iloc[0]) == "True"
         else ("out of validity" if "outside" in str(g.get("meta_validity", pd.Series([""])).iloc[0]) else "ok")),
        include_groups=False).unstack("plant").reindex(order)[PLANT_NAMES]
    app.to_csv(f"{OUT}/M1_applicability.csv")
    return df, order


def step_traces(df, order):
    """Re-simulate the returned gains (seed 0) for the overlay figures."""
    out = {}
    for plant in PLANT_NAMES:
        p = TuningProblem.from_scenario(plant)
        entry = {"ref": p.y_ref, "methods": {}}
        for m in order:
            r = df[(df.plant == plant) & (df.method == m) & (df.scenario == "step") & (df.kind == "npid")]
            r = r[r.seed == r.seed.min()] if len(r) else r
            if not len(r) or not bool(r.success.iloc[0]):
                entry["methods"][m] = None
                continue
            g = {k: float(r[f"gain_{k}"].iloc[0]) for k in ("Kp", "Ki", "Kd")}
            res = p.simulate(g, "npid", count=False)
            entry["methods"][m] = (res.time, res.omega)
        out[plant] = entry
    return out


# ----------------------------------------------------------------------------- ablations
def ablation_identification():
    """(A) FOPDT identification method (tangent / two-point / least squares) feeding the open-loop rules."""
    rows = []
    for plant in PLANT_NAMES:
        for rule_cls, name in ((ZieglerNicholsReactionCurve, "ZN"), (CohenCoon, "Cohen-Coon")):
            for idm in ("tangent", "two_point", "lsq"):
                p = TuningProblem.from_scenario(plant)
                r = rule_cls(id_method=idm, strict=False).tune(p)
                if not r.success:
                    rows.append(dict(plant=plant, rule=name, id_method=idm, ITAE=np.nan, overshoot=np.nan))
                    continue
                ev = evaluate_gains(p, r.gains)
                rows.append(dict(plant=plant, rule=name, id_method=idm, K=r.meta["K"], tau=r.meta["tau"],
                                 theta=r.meta["theta"], **r.gains, ITAE=ev["step"]["ITAE"],
                                 overshoot=ev["step"]["overshoot_pct"]))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/M1_ablation_A_identification.csv", index=False)
    return df


def ablation_validity():
    """(B) Open-loop rules applied outside their validity range (strict=False): how bad does it get?"""
    rows = []
    for plant in PLANT_NAMES:
        for cls, kw, name in ((ZieglerNicholsReactionCurve, {}, "ZN"), (CohenCoon, {}, "Cohen-Coon"),
                              (TimeDomainTuning, dict(rule="chr0"), "CHR 0%")):
            p = TuningProblem.from_scenario(plant)
            r = cls(strict=False, **kw).tune(p)
            ev = evaluate_gains(p, r.gains)
            rows.append(dict(plant=plant, rule=name, validity=r.meta.get("validity", ""),
                             theta_over_tau=(r.meta["theta"] / r.meta["tau"]) if r.meta.get("model") != "ipdt" else np.nan,
                             **r.gains, ITAE=ev["step"]["ITAE"], overshoot=ev["step"]["overshoot_pct"],
                             diverged=ev["step"]["diverged"]))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/M1_ablation_B_validity.csv", index=False)
    return df


def ablation_rulesets():
    """(C) Rule sets for continuous cycling, oscillation source where an ultimate point exists, relay elsewhere."""
    rows = []
    for plant in PLANT_NAMES:
        for rule in ("zn", "pessen", "some-overshoot", "tyreus-luyben", "no-overshoot"):
            for source in ("oscillation", "relay"):
                p = TuningProblem.from_scenario(plant)
                r = ContinuousCycling(rule, source=source).tune(p)
                if not r.success:
                    rows.append(dict(plant=plant, rule=rule, source=source, ITAE=np.nan, overshoot=np.nan, note=r.message[:50]))
                    continue
                ev = evaluate_gains(p, r.gains)
                rows.append(dict(plant=plant, rule=rule, source=source, Ku=r.meta["Ku"], Tu=r.meta["Tu"], **r.gains,
                                 ITAE=ev["step"]["ITAE"], overshoot=ev["step"]["overshoot_pct"],
                                 ITAE_noise=ev["noise"]["ITAE"], evals=r.evals, sim_seconds=r.sim_seconds))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/M1_ablation_C_rulesets.csv", index=False)
    return df


def ablation_limit_cycle():
    """(D) The 'ultimate gain' of the DC speed loop is a limit-cycle artefact: how much does it depend on how it is detected?"""
    rows = []
    for hf in (10.0, 20.0, 40.0, 80.0):
        for thr in (3e-4, 1e-3, 3e-3, 1e-2):
            class CC(ContinuousCycling):
                def _classify(self, problem, kp, t_cc, mode, _thr=thr):
                    res = self._p_only(problem, kp, t_cc)
                    kind, period, _ = detect_limit_cycle(res.omega, res.time, problem.y_ref, rel_threshold=_thr)
                    return ("sustained" if kind == "sustained" else "damped"), period

                def _search(self, problem, mode, _hf=hf):
                    problem_Ts = problem.Ts
                    problem.Ts = problem_Ts * _hf / 40.0     # horizon = 40*Ts inside _search -> hf*Ts
                    try:
                        return super()._search(problem, "limit_cycle")
                    finally:
                        problem.Ts = problem_Ts
            p = TuningProblem.from_scenario("dc_speed")
            try:
                Ku, Tu = CC("zn")._search(p, "limit_cycle")
            except Exception:
                Ku = Tu = np.nan
            rows.append(dict(horizon_Ts=hf, threshold=thr, Ku=Ku, Tu=Tu))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/M1_ablation_D_limit_cycle.csv", index=False)
    return df


def ablation_noise():
    """(E) Identification with noisy measurements: parameter error and the quality of the resulting gains."""
    rows = []
    for plant, sig_rel in (("fopdt", None), ("dc_speed", None)):
        for rel in (0.0, 0.01, 0.02, 0.05):
            for seed in range(10 if rel > 0 else 1):
                p = TuningProblem.from_scenario(plant)
                sigma = rel * abs(p.y_ref)
                try:
                    m = identify_process(p, method="lsq", noise_sigma=sigma, seed=seed)
                except Exception as e:
                    rows.append(dict(plant=plant, noise_rel=rel, seed=seed, failed=True, note=type(e).__name__))
                    continue
                # the gains the Cohen-Coon rule would produce from this noisy model
                from src.tuners.classical import _reduced
                kind, K, tau, theta = _reduced(m, p.dt)
                r = CohenCoon(id_method="lsq", strict=False)
                Kp, Ti, Td = r._rule(kind, K, tau, theta)
                g = dict(Kp=Kp, Ki=Kp / Ti, Kd=Kp * Td)
                ev = evaluate_gains(p, g)
                rows.append(dict(plant=plant, noise_rel=rel, seed=seed, failed=False, K=m.K, tau=m.tau, theta=m.theta,
                                 nrmse=m.nrmse, **g, ITAE=ev["step"]["ITAE"], overshoot=ev["step"]["overshoot_pct"],
                                 sim_seconds=p.sim_seconds))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/M1_ablation_E_noise.csv", index=False)
    return df


def ablation_trial_and_error():
    """(F) 'Human' variability: how much does the scripted trial-and-error result depend on its acceptance thresholds?"""
    rows = []
    for plant in PLANT_NAMES:
        for rise in (0.25, 0.5, 1.0):
            for mp in (10.0, 15.0, 25.0):
                for ess in (1.0, 2.0, 5.0):
                    p = TuningProblem.from_scenario(plant)
                    r = TrialAndError(rise_target=rise, overshoot_target=mp, ess_target=ess).tune(p)
                    ev = evaluate_gains(p, r.gains)
                    rows.append(dict(plant=plant, rise_target=rise, overshoot_target=mp, ess_target=ess, trials=r.evals,
                                     **r.gains, ITAE=ev["step"]["ITAE"], overshoot=ev["step"]["overshoot_pct"]))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/M1_ablation_F_trial_and_error.csv", index=False)
    return df


# ----------------------------------------------------------------------------- figures for ablations
def figures_ablations(A, B, C, D, E, F):
    import matplotlib.pyplot as plt
    # A: identification method
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, rule in zip(axes, ("ZN", "Cohen-Coon")):
        t = A[A.rule == rule].pivot(index="plant", columns="id_method", values="ITAE").reindex(PLANT_NAMES)
        t.plot.bar(ax=ax, logy=True, rot=20)
        ax.set_title(f"{rule}: ITAE by identification method")
        ax.grid(alpha=0.3, axis="y")
    _save(fig, f"{FIG}/M1_ablationA_identification.png")
    # B: validity
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for rule, mk in (("ZN", "o"), ("Cohen-Coon", "s"), ("CHR 0%", "^")):
        d = B[(B.rule == rule) & B.theta_over_tau.notna()]
        ax.scatter(d.theta_over_tau, d.ITAE, marker=mk, label=rule, s=50)
    ax.axvspan(0.1, 1.0, color="#2ca02c", alpha=0.12, label="recommended range")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("identified dead-time ratio theta/tau")
    ax.set_ylabel("ITAE (step, N-PID)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Open-loop rules vs dead-time ratio (strict=False)")
    _save(fig, f"{FIG}/M1_ablationB_validity.png")
    # C: rule sets
    fig, ax = plt.subplots(figsize=(11, 4.2))
    c = C[C.source == "relay"].pivot(index="plant", columns="rule", values="ITAE").reindex(PLANT_NAMES)
    c.plot.bar(ax=ax, logy=True, rot=20)
    ax.set_title("Continuous cycling (relay): ITAE by rule set")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, f"{FIG}/M1_ablationC_rulesets.png")
    # D: limit-cycle sensitivity
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for thr, g in D.groupby("threshold"):
        ax.plot(g.horizon_Ts, g.Ku, marker="o", label=f"threshold {thr:g}")
    ax.set_xlabel("observation horizon (x Ts)")
    ax.set_ylabel("'ultimate gain' Ku of the DC speed loop")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    ax.set_title("The limit-cycle 'Ku' is not a well-defined quantity")
    _save(fig, f"{FIG}/M1_ablationD_limit_cycle.png")
    # E: noise
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for plant, ax in zip(("fopdt", "dc_speed"), axes):
        d = E[(E.plant == plant) & (~E.failed.astype(bool))]
        g = d.groupby("noise_rel")["ITAE"]
        ax.errorbar(g.median().index * 100, g.median(), yerr=[g.median() - g.quantile(0.25), g.quantile(0.75) - g.median()],
                    marker="o", capsize=3)
        ax.set_xlabel("measurement noise (% of setpoint) during identification")
        ax.set_ylabel("ITAE of Cohen-Coon gains (median, IQR)")
        ax.set_title(plant)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    _save(fig, f"{FIG}/M1_ablationE_noise.png")
    # F: trial and error spread
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    data = [F[F.plant == pl].ITAE.values for pl in PLANT_NAMES]
    ax.boxplot(data, tick_labels=PLANT_NAMES, showfliers=True)
    ax.set_yscale("log")
    ax.set_ylabel("ITAE (step, N-PID)")
    ax.set_title("Trial and error: spread over 27 acceptance-threshold settings")
    ax.grid(alpha=0.3, which="both", axis="y")
    _save(fig, f"{FIG}/M1_ablationF_trial_and_error.png")


# ----------------------------------------------------------------------------- main
def main():
    os.makedirs(FIG, exist_ok=True)
    df, order = main_benchmark()
    traces = step_traces(df, order)
    plot_step_overlays(traces, f"{FIG}/M1_step_overlay_npid.png", title="M1: N-PID step responses, all classical methods (+PSO reference)")
    models = {}
    for plant in PLANT_NAMES:
        try:
            models[plant] = identify_process(TuningProblem.from_scenario(plant), method="lsq")
        except Exception:
            models[plant] = None
    plot_identification(models, f"{FIG}/M1_identification.png", title="M1: bump-test identification (measured vs fitted)")
    pd.DataFrame([dict(plant=k, kind=m.kind, K=m.K, tau=m.tau, theta=m.theta, nrmse=m.nrmse, u_op=m.u_op,
                       theta_over_tau=(m.theta / m.tau) if m.kind != "ipdt" else np.nan)
                  for k, m in models.items() if m is not None]).to_csv(f"{OUT}/M1_identified_models.csv", index=False)
    A, B, C, D, E, F = (ablation_identification(), ablation_validity(), ablation_rulesets(), ablation_limit_cycle(),
                        ablation_noise(), ablation_trial_and_error())
    figures_ablations(A, B, C, D, E, F)
    print("M1 done ->", OUT)


if __name__ == "__main__":
    main()
