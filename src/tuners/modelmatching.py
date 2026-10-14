"""
Model matching (docs/METHODS_ROADMAP.md, phase M2): choose PID gains so that the closed loop follows a reference model.

Reference model (the specification):   M(s) = e^{-theta s} wn^2 / (s^2 + 2 zeta wn s + wn^2)      (order 2, default)
                                       M(s) = e^{-theta s} / (Tc s + 1)                              (order 1)
with the plant's own dead time theta (a closed loop cannot respond faster than the plant's delay, so a delay-free
reference would be infeasible) and a speed taken from the identified open-loop settling time:
    T_resp = max((T_open - theta)/speedup, ...),  wn = 4/(zeta T_resp)  (order 2),  Tc = T_resp/4  (order 1),
with T_open = 4 tau + theta for self-regulating plants and 10 theta_eff for integrating plants.

Two variants
  mode="frequency": the controller that would give exactly M is C*(s) = M / (G (1 - M)) for the identified model G
      (FOPDT or IPDT). The PID with the problem's derivative filter,  C(jw) = Kp + Ki/(jw) + Kd jw/(1 + jw tau_d),
      is linear in (Kp, Ki, Kd): relative-error non-negative least squares over a log-frequency grid around wn
      (no further plant experiment).
  mode="time": starting from the frequency solution, Nelder-Mead in log-gain space on the *real* N-PID loop,
      minimising the integral squared difference between the closed-loop step response and the reference-model
      response (counted simulations, `max_sims`).

Applicability: needs the bump-test identification (self-regulating or integrating, positive gain).
Limitations: the PID cannot reproduce C* exactly for dead-time plants (the fit is approximate); the reference speed is
a design choice (`speedup`), not something the method can discover.
"""

import numpy as np
from scipy.optimize import minimize, nnls

from src.plant import DelayedPlant, TransferFunctionPlant
from src.simulate import run_simulation
from src.tuners.base import FAIL_COST, GAIN_NAMES, Tuner
from src.tuners.identify import identify_process


def open_loop_settling(model):
    """Heuristic 2% settling time of the open-loop plant used to scale the reference speed."""
    if model.kind == "ipdt":
        return 10.0 * model.theta_eff
    tau = model.tau + (model.tau2 if model.kind == "sopdt" else 0.0)
    return 4.0 * tau + model.theta


def reference_parameters(model, speedup, zeta, order, dt):
    """Return dict(theta, wn or Tc, T_resp) of the reference model for an identified process model."""
    theta = max(model.theta_eff if model.kind == "ipdt" else model.theta, 0.0)
    t_open = open_loop_settling(model)
    t_resp = max((t_open - theta) / speedup, 10.0 * dt)
    if order == 2:
        return dict(theta=theta, wn=4.0 / (zeta * t_resp), zeta=zeta, T_resp=t_resp)
    return dict(theta=theta, Tc=t_resp / 4.0, T_resp=t_resp)


def reference_response(problem, ref, order, t_end=None):
    """Step response (scaled by the setpoint) of the reference model on the problem's time grid."""
    t_end = t_end or problem.t_end
    if order == 2:
        base = TransferFunctionPlant([ref["wn"] ** 2], [1.0, 2 * ref["zeta"] * ref["wn"], ref["wn"] ** 2])
    else:
        base = TransferFunctionPlant([1.0], [ref["Tc"], 1.0])
    plant = DelayedPlant(base, ref["theta"]) if ref["theta"] > problem.dt else base
    x = plant.initial_state()
    n = int(t_end / problem.dt)
    y = np.zeros(n)
    for k in range(n):
        y[k] = plant.output(x)
        x = plant.step(k * problem.dt, x, problem.y_ref, 0.0, problem.dt)
    return np.arange(n) * problem.dt, y


# ----------------------------------------------------------------------------- frequency-domain fit
def plant_frequency_response(model, w):
    s = 1j * w
    if model.kind == "ipdt":
        return model.K * np.exp(-model.theta * s) / (s * (model.tau * s + 1.0))
    if model.kind == "sopdt":
        return model.K * np.exp(-model.theta * s) / ((model.tau * s + 1.0) * (model.tau2 * s + 1.0))
    return model.K * np.exp(-model.theta * s) / (model.tau * s + 1.0)


def reference_frequency_response(ref, order, w):
    s = 1j * w
    delay = np.exp(-ref["theta"] * s)
    if order == 2:
        return delay * ref["wn"] ** 2 / (s ** 2 + 2 * ref["zeta"] * ref["wn"] * s + ref["wn"] ** 2)
    return delay / (ref["Tc"] * s + 1.0)


def fit_pid_to_ideal(model, ref, order, tau_d, dt, n_grid=60, band=(0.05, 3.0)):
    """Relative-error NNLS fit of (Kp, Ki, Kd) to C*(jw) = M/(G(1-M)). Returns (gains dict, relative residual)."""
    w_ref = ref["wn"] if order == 2 else 1.0 / ref["Tc"]
    w = np.geomspace(band[0] * w_ref, min(band[1] * w_ref, 0.2 * np.pi / dt), n_grid)
    G = plant_frequency_response(model, w)
    M = reference_frequency_response(ref, order, w)
    c_ideal = M / (G * (1.0 - M))
    s = 1j * w
    basis = np.stack([np.ones_like(s), 1.0 / s, s / (1.0 + s * tau_d)], axis=1)       # columns: Kp, Ki, Kd
    A = basis / c_ideal[:, None]                                                      # relative error
    b = np.ones(len(w), dtype=complex)
    x, _ = nnls(np.vstack([A.real, A.imag]), np.concatenate([b.real, b.imag]))
    res = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
    return dict(zip(GAIN_NAMES, map(float, x))), float(res)


# ----------------------------------------------------------------------------- tuner
class ModelMatching(Tuner):
    family = "intelligent"
    stochastic = False

    def __init__(self, mode="frequency", speedup=3.0, zeta=0.9, order=2, max_sims=120, name=None):
        self.mode, self.speedup, self.zeta, self.order, self.max_sims = mode, speedup, zeta, order, max_sims
        self.name = name or {"frequency": "Model matching (frequency)", "time": "Model matching (time domain)"}[mode]
        self.last = {}

    def _tune(self, problem, budget, seed):
        model = identify_process(problem, method="lsq", order=1)
        ref = reference_parameters(model, self.speedup, self.zeta, self.order, problem.dt)
        gains, resid = fit_pid_to_ideal(model, ref, self.order, problem.tau_d, problem.dt)
        meta = dict(model=model.kind, K=model.K, tau=model.tau, theta=model.theta, nrmse=model.nrmse,
                    T_resp=ref["T_resp"], freq_residual=resid, mode=self.mode, order=self.order)
        if "wn" in ref:
            meta["wn"] = ref["wn"]
        if gains["Kp"] <= 0 or gains["Ki"] <= 0:
            # degenerate fit (e.g. all weight on one basis function): keep a tiny positive Ki so the loop has integral action
            gains["Kp"] = max(gains["Kp"], 1e-6)
            gains["Ki"] = max(gains["Ki"], 1e-6)
        history = []
        if self.mode == "time":
            t_ref, y_ref = reference_response(problem, ref, self.order)
            lb, ub = np.log10(problem.bounds[0]), np.log10(problem.bounds[1])
            best = dict(J=np.inf, g=dict(gains))
            norm = (problem.y_ref ** 2) * problem.t_end

            def J(logg):
                g = dict(zip(GAIN_NAMES, 10.0 ** np.clip(logg, lb, ub)))
                res = problem.simulate(g)
                y = res.omega
                if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 10 * abs(problem.y_ref):
                    return FAIL_COST
                val = float(np.trapezoid((y - y_ref) ** 2, t_ref) / norm)
                if val < best["J"]:
                    best.update(J=val, g=g)
                history.append((problem.evals, best["J"]))
                return val

            x0 = np.log10([max(gains["Kp"], 1e-9), max(gains["Ki"], 1e-9), max(gains["Kd"], 1e-9 * gains["Kp"] + 1e-12)])
            minimize(J, x0, method="Nelder-Mead", options=dict(maxfev=self.max_sims, xatol=1e-3, fatol=1e-10,
                                                               initial_simplex=x0 + 0.25 * np.vstack([np.zeros(3), np.eye(3)])))
            gains = best["g"]
            meta["matching_error"] = best["J"]
        self.last = dict(ref=ref, model=model)
        return gains, meta, history
