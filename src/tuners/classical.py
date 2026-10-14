"""
Classical tuning methods (docs/METHODS_ROADMAP.md, phase M1).

All of them design linear PID gains in the standard (ISA) form  C(s) = Kp (1 + 1/(Ti s) + Td s)  and convert to
the parallel form used by the controllers:  Ki = Kp/Ti,  Kd = Kp*Td.  The gains become the base gains of the
N-PID (docs/PROTOCOL.md section 5); only trial-and-error tunes the N-PID itself.

    ZieglerNicholsReactionCurve   open-loop step test, tangent method, ZN reaction-curve rules (1942)
    CohenCoon                     open-loop step test, Cohen-Coon rules (1953)
    ContinuousCycling             closed-loop ultimate gain/period (P-only oscillation search or relay),
                                  ZN / Tyreus-Luyben / Pessen / some-overshoot / no-overshoot rule sets
    TimeDomainTuning              specification-driven rules from the identified step response:
                                  SIMC (Skogestad 2003) and Chien-Hrones-Reswick (1952, set-point, 0% / 20%)
    TrialAndError                 scripted emulation of a human tuning session on the closed-loop step response

Rule validity: the open-loop rules were derived for first-order-plus-dead-time processes with 0.1 <= theta/tau
<= 1. With `strict=True` (default) a method that is outside its validity range raises `NotApplicable`
instead of silently returning unreliable gains; `strict=False` applies the rule anyway (used to measure how bad
it gets).
"""

import numpy as np

from src.metrics import compute_metrics
from src.tuners.base import FAIL_COST, NotApplicable, Tuner
from src.tuners.identify import NotSelfRegulating, find_trim, identify_process
from src.tuning_zn import detect_limit_cycle, oscillation_growth_ratio

RATIO_RANGE = (0.1, 1.0)         # recommended theta/tau range of the open-loop rules


def to_parallel(Kp, Ti, Td):
    """ISA (Kp, Ti, Td) -> parallel gains {Kp, Ki, Kd}."""
    return dict(Kp=float(Kp), Ki=float(Kp / Ti) if np.isfinite(Ti) else 0.0, Kd=float(Kp * Td))


def _reduced(model, dt):
    """(kind, K, tau, theta) with a sampling-limited dead time floor; integrating models use (Kv, theta_eff)."""
    if model.kind == "ipdt":
        return "ipdt", model.K, None, max(model.theta_eff, 2 * dt)
    tau = model.tau + (model.tau2 / 2 if model.kind == "sopdt" else 0.0)      # half-rule
    theta = model.theta + (model.tau2 / 2 if model.kind == "sopdt" else 0.0)
    return "fopdt", model.K, tau, max(theta, 2 * dt)


def _check_validity(kind, tau, theta, strict, rule, lo=RATIO_RANGE[0], hi=RATIO_RANGE[1]):
    """Return a validity note; raise NotApplicable in strict mode when theta/tau is outside [lo, hi]."""
    if kind == "ipdt":
        return "in range"
    r = theta / tau
    if lo <= r <= hi:
        return "in range"
    note = f"theta/tau = {r:.3g} outside [{lo}, {hi}] recommended for {rule}"
    if strict:
        raise NotApplicable(note)
    return note


class _IdentifyThenRule(Tuner):
    family = "classical"
    stochastic = False
    id_method = "lsq"
    order = 1

    def __init__(self, id_method=None, strict=True, name=None):
        if id_method:
            self.id_method = id_method
        self.strict = strict
        if name:
            self.name = name

    def _rule(self, kind, K, tau, theta):
        raise NotImplementedError

    def _tune(self, problem, budget, seed):
        model = identify_process(problem, method=self.id_method, order=self.order)
        kind, K, tau, theta = _reduced(model, problem.dt)
        note = _check_validity(kind, tau, theta, self.strict, self.name,
                               *getattr(self, "ratio_range", RATIO_RANGE))
        Kp, Ti, Td = self._rule(kind, K, tau, theta)
        meta = dict(model=model.kind, K=model.K, tau=model.tau, theta=model.theta, nrmse=model.nrmse,
                    validity=note, Ti=Ti, Td=Td, id_method=self.id_method if model.kind != "ipdt" else "lsq")
        return to_parallel(Kp, Ti, Td), meta


class ZieglerNicholsReactionCurve(_IdentifyThenRule):
    """
    Ziegler-Nichols open-loop (process reaction curve) method. Tangent at the inflection point gives the
    apparent dead time L and slope R = K/tau of the unit step response;
        PID:  Kp = 1.2/(R L),  Ti = 2 L,  Td = 0.5 L        (R L = K L/tau; R = Kv for integrating plants)
    Quarter-amplitude-decay tuning: aggressive, typically 25-60% overshoot.
    """
    name = "Ziegler-Nichols (reaction curve)"
    id_method = "tangent"

    def _rule(self, kind, K, tau, theta):
        a = K * theta / tau if kind == "fopdt" else K * theta
        return 1.2 / a, 2.0 * theta, 0.5 * theta


class CohenCoon(_IdentifyThenRule):
    """
    Cohen-Coon rules (designed for dead-time-dominant processes), FOPDT model, r = theta/tau:
        Kp = (1/K)(tau/theta)(4/3 + r/4),  Ti = theta (32 + 6r)/(13 + 8r),  Td = 4 theta/(11 + 2r)
    For integrating plants the tau -> infinity limit is used (r -> 0): Kp = 4/(3 Kv theta), Ti = 32 theta/13,
    Td = 4 theta/11.
    """
    name = "Cohen-Coon"
    id_method = "two_point"

    def _rule(self, kind, K, tau, theta):
        if kind == "ipdt":
            return 4.0 / (3.0 * K * theta), 32.0 * theta / 13.0, 4.0 * theta / 11.0
        r = theta / tau
        return (tau / (K * theta)) * (4.0 / 3.0 + r / 4.0), theta * (32 + 6 * r) / (13 + 8 * r), 4 * theta / (11 + 2 * r)


class TimeDomainTuning(Tuner):
    """
    Tuning from time-domain specifications of the closed-loop step response, on the identified model.

    rule="simc" (Skogestad 2003, direct synthesis with closed-loop time constant tau_c):
        SOPDT:       Kp = tau1/(K (tau_c + theta)),  Ti = min(tau1, 4 (tau_c + theta)),  Td = tau2
        IPDT+lag:    Kp = 1/(Kv (tau_c + theta)),    Ti = 4 (tau_c + theta),             Td = tau
        tau_c = tau_c_factor * theta, but at least `tau_c_floor` * (dominant lag) so that the bandwidth stays
        within what the actuator can deliver when the dead time is ~0 (SIMC recommends tau_c >= theta).
    rule="chr0" / "chr20" (Chien-Hrones-Reswick, set-point tracking, 0% / 20% overshoot):
        FOPDT: chr0:  Kp = 0.6 tau/(K theta), Ti = tau,        Td = 0.5 theta
               chr20: Kp = 0.95 tau/(K theta), Ti = 1.357 tau,  Td = 0.473 theta
        IPDT:  chr0:  Kp = 0.6/(Kv theta),  Ti = 4 theta,    Td = 0.5 theta
               chr20: Kp = 0.95/(Kv theta), Ti = 2.38 theta, Td = 0.42 theta
    """
    family = "classical"
    stochastic = False

    def __init__(self, rule="simc", tau_c_factor=1.0, tau_c_floor=0.1, strict=True):
        self.rule, self.tau_c_factor, self.tau_c_floor = rule, tau_c_factor, tau_c_floor
        self.name = {"simc": "Time-domain (SIMC)", "chr0": "Time-domain (CHR 0%)",
                     "chr20": "Time-domain (CHR 20%)"}[rule]
        self.order = 2 if rule == "simc" else 1
        self.strict = strict
        self.id_method = "lsq"
        self.ratio_range = (0.0, np.inf) if rule == "simc" else RATIO_RANGE

    def _tune(self, problem, budget, seed):
        model = identify_process(problem, method="lsq", order=self.order)
        if self.rule == "simc":
            return self._simc(model, problem)
        return self._chr(model, problem)

    def _chr(self, model, problem):
        kind, K, tau, theta = _reduced(model, problem.dt)
        note = _check_validity(kind, tau, theta, self.strict, self.name, *self.ratio_range)
        if kind == "ipdt":
            Kp, Ti, Td = ((0.6, 4.0, 0.5) if self.rule == "chr0" else (0.95, 2.38, 0.42))
            Kp, Ti, Td = Kp / (K * theta), Ti * theta, Td * theta
        else:
            c = (0.6, 1.0, 0.5) if self.rule == "chr0" else (0.95, 1.357, 0.473)
            Kp, Ti, Td = c[0] * tau / (K * theta), c[1] * tau, c[2] * theta
        meta = dict(model=model.kind, K=model.K, tau=model.tau, theta=model.theta, nrmse=model.nrmse,
                    validity=note, Ti=Ti, Td=Td)
        return to_parallel(Kp, Ti, Td), meta

    def _simc(self, model, problem):
        dt = problem.dt
        if model.kind == "ipdt":
            theta = max(model.theta, 2 * dt)
            tau_dom = max(model.tau, 2 * dt)
            tau_c = max(self.tau_c_factor * theta, self.tau_c_floor * tau_dom)
            Kp = 1.0 / (model.K * (tau_c + theta))
            Ti, Td = 4.0 * (tau_c + theta), model.tau
        else:
            tau1, tau2 = model.tau, (model.tau2 if model.kind == "sopdt" else 0.0)
            theta = max(model.theta, 2 * dt)
            tau_c = max(self.tau_c_factor * theta, self.tau_c_floor * tau1)
            Kp = tau1 / (model.K * (tau_c + theta))
            Ti, Td = min(tau1, 4.0 * (tau_c + theta)), tau2
        meta = dict(model=model.kind, K=model.K, tau=model.tau, tau2=model.tau2, theta=model.theta,
                    nrmse=model.nrmse, tau_c=tau_c, Ti=Ti, Td=Td, validity="in range")
        return to_parallel(Kp, Ti, Td), meta


# ----------------------------------------------------------------------------- continuous cycling
CYCLING_RULES = {                    # name: (Kp/Ku, Ti/Tu, Td/Tu)
    "zn": (0.6, 0.5, 0.125),
    "tyreus-luyben": (1 / 2.2, 2.2, 1 / 6.3),
    "pessen": (0.7, 0.4, 0.15),
    "some-overshoot": (0.33, 0.5, 1 / 3),
    "no-overshoot": (0.2, 0.5, 1 / 3),
}


class ContinuousCycling(Tuner):
    """
    Closed-loop ultimate-cycle method. Find the ultimate gain Ku (P-only loop at the edge of sustained
    oscillation) and period Tu, then apply a rule set (CYCLING_RULES).

    source="oscillation": raise Kp geometrically until the step response stops decaying (half-cycle amplitude
        ratio >= 1, `src.tuning_zn.oscillation_growth_ratio`), then bisect to 1%. Needs a plant that actually
        reaches marginal stability under P control (a phase crossover: dead time or relative degree >= 3).
        With fallback=True, plants without one fall back to the onset of a saturation/sampling limit cycle (an
        artefact, flagged in meta) - this is what the legacy project's 'ZN' did on the DC speed loop.
    source="relay": Astrom-Hagglund relay-feedback experiment, Ku = 4d / (pi sqrt(a^2 - h^2)), safer than driving
        the loop to its stability limit, needs the operating-point input (bias).
    The experiments are run on the real (saturating) loop and are all counted.
    """
    family = "classical"
    stochastic = False

    def __init__(self, rule="zn", source="oscillation", fallback=False):
        self.rule, self.source, self.fallback = rule, source, fallback
        tag = {"oscillation": "", "relay": " relay"}[source] + (" +limit-cycle" if fallback else "")
        self.name = f"Continuous cycling ({rule}{tag})"

    # -- oscillation search
    def _p_only(self, problem, kp, t_cc):
        res = problem.simulate(dict(Kp=kp, Ki=0.0, Kd=0.0), kind="lpid", t_end=t_cc)
        return res

    def _classify(self, problem, kp, t_cc, mode):
        res = self._p_only(problem, kp, t_cc)
        if not np.all(np.isfinite(res.omega)):
            return "growing", None
        if mode == "ratio":
            ratio, period, _ = oscillation_growth_ratio(res.omega, res.time)
            if ratio is None:
                return "damped", None
            return ("damped" if ratio < 0.999 else "sustained"), period
        kind, period, _ = detect_limit_cycle(res.omega, res.time, problem.y_ref)
        return ("sustained" if kind == "sustained" else "damped"), period

    def _search(self, problem, mode):
        # a limit cycle may only be declared after a long observation, otherwise slowly decaying responses look sustained
        t_cc = max(problem.t_end, (15.0 if mode == "ratio" else 40.0) * problem.Ts)
        kp, kp_hi_limit = problem.bounds[0][0], problem.bounds[1][0] * 50.0
        low, high, period_high = None, None, None
        while kp <= kp_hi_limit:
            kind, period = self._classify(problem, kp, t_cc, mode)
            if kind == "damped":
                low = kp
            else:
                high, period_high = kp, period
                break
            kp *= 2.0
        if high is None:
            raise NotApplicable("no ultimate gain: P-only loop never reaches sustained oscillation "
                                f"up to Kp = {kp_hi_limit:.3g}")
        if low is None:
            low = high / 2.0
        for _ in range(20):
            if (high - low) / high < 0.01:
                break
            mid = float(np.sqrt(low * high))
            kind, period = self._classify(problem, mid, t_cc, mode)
            if kind == "damped":
                low = mid
            else:
                high, period_high = mid, period
        if period_high is None or period_high <= 0:
            raise NotApplicable("could not measure the oscillation period")
        return high, period_high

    # -- relay
    def _relay(self, problem):
        try:
            u_trim = find_trim(problem)
        except NotSelfRegulating:
            u_trim = 0.0                                # integrating plant: relay about zero input
        room = min(problem.u_max - u_trim, u_trim - problem.u_min)
        span = problem.u_max - problem.u_min
        d = min(0.1 * span, 0.5 * room) if np.isfinite(span) else 0.1 * abs(u_trim) + 1.0
        if d <= 0:
            raise NotApplicable("no actuator room for a relay experiment")
        h = 0.02 * abs(problem.y_ref)
        plant = problem.plant_factory()
        x = plant.initial_state()
        n = int(30.0 * max(problem.Ts, 10 * problem.dt) / problem.dt)
        n = max(n, int(problem.t_end / problem.dt))
        ys = np.zeros(n)
        u = u_trim + d
        problem._count(0.0)
        crossings = []
        for k in range(n):
            y = plant.output(x)
            ys[k] = y
            e = problem.y_ref - y
            if e > h:
                u = u_trim + d
            elif e < -h:
                u = u_trim - d
            if k > 0 and ys[k - 1] <= problem.y_ref < y:
                crossings.append(k)
            x = plant.step(k * problem.dt, x, min(max(u, problem.u_min), problem.u_max), 0.0, problem.dt)
            if len(crossings) >= 9 and k == crossings[-1]:                      # enough cycles, check convergence
                per = np.diff(crossings[-6:]) * problem.dt
                if np.std(per) < 0.03 * np.mean(per):
                    n = k + 1
                    break
        problem.sim_seconds += n * problem.dt
        if len(crossings) < 5:
            raise NotApplicable("relay did not produce a sustained oscillation")
        cr = crossings[-6:]
        Tu = float(np.mean(np.diff(cr)) * problem.dt)
        seg = ys[cr[0]:cr[-1]]
        a = 0.5 * (np.max(seg) - np.min(seg))
        if a <= h:
            raise NotApplicable("relay oscillation amplitude is below the hysteresis")
        if a < 1.5 * h:
            raise NotApplicable("relay oscillation is set by the hysteresis (amplitude < 1.5 h): "
                                "the loop has no phase crossover to measure")
        Ku = 4.0 * d / (np.pi * np.sqrt(a * a - h * h))
        self._relay_info = dict(d=d, h=h, a=a)
        return Ku, Tu

    def _tune(self, problem, budget, seed):
        artefact = False
        if self.source == "relay":
            Ku, Tu = self._relay(problem)
        else:
            try:
                Ku, Tu = self._search(problem, "ratio")
            except NotApplicable:
                if not self.fallback:
                    raise
                Ku, Tu = self._search(problem, "limit_cycle")
                artefact = True
        if Tu < 10 * problem.dt:
            raise NotApplicable(f"ultimate period {Tu:.3g}s is at the sampling limit (10 dt): not a real phase crossover")
        kp_c, ti_c, td_c = CYCLING_RULES[self.rule]
        Kp, Ti, Td = kp_c * Ku, ti_c * Tu, td_c * Tu
        meta = dict(Ku=Ku, Tu=Tu, limit_cycle_artefact=artefact, Ti=Ti, Td=Td,
                    validity="limit-cycle artefact" if artefact else "in range")
        if self.source == "relay":
            meta.update(self._relay_info)
        return to_parallel(Kp, Ti, Td), meta


# ----------------------------------------------------------------------------- trial and error
class TrialAndError(Tuner):
    """
    Scripted emulation of an engineer tuning by hand on the closed-loop step response of the actual N-PID.

    Procedure (fixed thresholds, deterministic): start from a small Kp (10% of the gain that would use the full
    actuator range for a setpoint-sized error), P, then I, then D:
      1. diverging, sustained oscillation, or overshoot > 40%       -> Kp x0.6, Ki x0.5, Kd x0.8
      2. rise time > 0.5 Ts (or not reached)                        -> Kp x1.5
      3. steady-state error > 2% of the setpoint                    -> Ki = Kp/(2 Ts) at first, then x1.5
      4. overshoot > 15%                                            -> Kd = Kp Ts/8 at first, then x1.5
      5. otherwise the response is accepted and the session ends.
    Every trial is one closed-loop simulation (counted); the best trial by (ITAE + overshoot penalty) is returned.
    Human performance varies; this is a reproducible stand-in, not a model of a particular engineer.
    """
    name = "Trial and error (scripted)"
    family = "classical"
    stochastic = False

    def __init__(self, max_trials=40, rise_target=0.5, overshoot_target=15.0, ess_target=2.0):
        self.max_trials, self.rise_target = max_trials, rise_target
        self.overshoot_target, self.ess_target = overshoot_target, ess_target

    @staticmethod
    def features(problem, res):
        y, t = res.omega, res.time
        ref = problem.y_ref
        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 10 * abs(ref):
            return dict(diverged=True)
        m = compute_metrics(t, ref, y)
        final = float(np.mean(y[-max(2, len(y) // 20):]))
        ess = 100.0 * abs(ref - final) / abs(ref)
        sgn = np.sign(ref)
        mp = max(0.0, 100.0 * (np.max(sgn * y) - abs(ref)) / abs(ref))
        d = np.diff(y[len(y) // 2:])
        amp = np.ptp(y[len(y) // 2:])
        flips = int(np.sum(np.diff(np.sign(d[np.abs(d) > 1e-9])) != 0))
        oscillating = flips >= 6 and amp > 0.05 * abs(ref)
        rt = m["rise_time"]
        return dict(diverged=False, ess=ess, mp=mp, rise=rt if np.isfinite(rt) else np.inf,
                    oscillating=oscillating, itae=m["ITAE"])

    def _tune(self, problem, budget, seed):
        lim = max(abs(problem.u_min), abs(problem.u_max))
        lim = lim if np.isfinite(lim) else 10 * abs(problem.y_ref)
        Kp, Ki, Kd = 0.1 * lim / abs(problem.y_ref), 0.0, 0.0
        Ts, best, trace = problem.Ts, (np.inf, None), []
        for trial in range(budget or self.max_trials):
            g = dict(Kp=Kp, Ki=Ki, Kd=Kd)
            f = self.features(problem, problem.simulate(g))
            score = FAIL_COST if f["diverged"] else f["itae"] + 5.0 * max(0.0, f["mp"] - 10.0)
            trace.append((trial + 1, score))
            if score < best[0]:
                best = (score, dict(g))
            if f["diverged"] or f["oscillating"] or f["mp"] > 40.0:
                Kp, Ki, Kd = 0.6 * Kp, 0.5 * Ki, 0.8 * Kd
            elif f["rise"] > self.rise_target * Ts and f["mp"] < 25.0:
                Kp *= 1.5
            elif f["ess"] > self.ess_target:
                Ki = Kp / (2.0 * Ts) if Ki == 0 else 1.5 * Ki
            elif f["mp"] > self.overshoot_target:
                Kd = Kp * Ts / 8.0 if Kd == 0 else 1.5 * Kd
            else:
                break
        hist, run = [], np.inf
        for k, s in trace:
            run = min(run, s)
            hist.append((k, run))
        return best[1], dict(trials=len(trace), best_score=best[0], stopped_early=len(trace) < (budget or self.max_trials)), hist
