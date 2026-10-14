"""
Plant-agnostic features for the ML tuners (M4): a counted bump test gives a local model and the unit-step response;
from them a small vector of **dimensionless** features, a **normalised step-response signature**, and the constants
that make gains dimensionless.

Why normalise: for a loop  C(s) G(s)  scaling the plant gain K or the time scale T changes the optimal PID gains in a
known way (Kp ~ 1/K, Ki ~ 1/(K T), Kd ~ T/K). With
        K_eff = K (self-regulating) or Kv T_n (integrating),      T_n = tau + theta (identified, >= 5 dt)
the normalised gains
        Kp~ = Kp K_eff,     Ki~ = Ki K_eff T_n,     Kd~ = Kd K_eff / T_n
depend only on the *shape* of the dynamics, so a model trained on plants of very different scales transfers to new
ones. All features below are built from quantities the bump test measures; nothing about the plant is assumed.
"""

from dataclasses import dataclass, field

import numpy as np

from src.tuners.identify import identify_process

N_SIGNATURE = 24
FEATURE_NAMES = ("integrating", "delay_fraction", "log_delay_fraction", "log_authority", "unipolar", "log_fit_error",
                 "peak_ratio", "t63_over_Tn")


@dataclass
class FeatureSet:
    vec: np.ndarray                 # engineered dimensionless features (FEATURE_NAMES)
    signature: np.ndarray           # normalised unit-step response at N_SIGNATURE points on [0, 5 T_n]
    K_eff: float
    T_n: float
    kind: str
    model: dict = field(default_factory=dict)
    cost_evals: int = 0
    cost_sim_seconds: float = 0.0

    def x(self, which="engineered"):
        if which == "engineered":
            return self.vec
        if which == "signature":
            return np.concatenate([self.signature, self.vec[[0, 3, 4]]])      # + integrating flag, authority, polarity
        if which == "both":
            return np.concatenate([self.vec, self.signature])
        raise ValueError(which)


def extract_features(problem, noise_sigma=0.0, seed=0):
    """Run the (counted) bump test on `problem` and build the feature set. Raises NotApplicable if identification fails."""
    e0, s0 = problem.evals, problem.sim_seconds
    m = identify_process(problem, method="lsq", order=1, noise_sigma=noise_sigma, seed=seed)
    fs = features_from_model(problem, m)
    fs.cost_evals, fs.cost_sim_seconds = problem.evals - e0, problem.sim_seconds - s0
    return fs


def features_from_model(problem, m):
    """
    Feature set from an identified `ProcessModel` (FOPDT or IPDT with its unit-step response in `m.data`). Only the
    actuator limits, setpoint and sampling time are read from `problem`. Used by the bump test (`extract_features`)
    and by the online estimator of M6, which builds the model from a recursively identified ARX model.
    """
    theta = m.theta_eff if m.kind == "ipdt" else m.theta
    T_n = max(m.tau + theta, 5.0 * problem.dt)
    K_eff = m.K * T_n if m.kind == "ipdt" else m.K
    f = theta / T_n
    u_lim = max(abs(problem.u_min), abs(problem.u_max))
    u_lim = u_lim if np.isfinite(u_lim) else 10.0 * abs(problem.y_ref) / K_eff
    authority = u_lim * K_eff / abs(problem.y_ref)

    t, s = m.data["t"], m.data["s"]
    grid = np.linspace(0.0, 5.0 * T_n, N_SIGNATURE)
    # self-regulating: settles at 1; integrating: ramp with unit slope per T_n (K_eff = Kv T_n)
    sig = np.interp(grid, t, s, right=s[-1]) / K_eff
    s_end = s[-1]
    if m.kind == "fopdt" and s_end > 0:
        peak = float(np.clip(np.max(s) / s_end, 1.0, 5.0))
        idx = np.where(s >= 0.632 * s_end)[0]
        t63 = float(t[idx[0]] / T_n) if len(idx) else 5.0
    else:
        peak, t63 = 1.0, 0.0
    vec = np.array([float(m.kind == "ipdt"), f, np.log10(f + 0.01), np.log10(max(authority, 1e-3)),
                    float(problem.u_min == 0.0), np.log10(m.nrmse + 1e-3), peak, t63])
    return FeatureSet(vec=vec, signature=np.asarray(sig, dtype=float), K_eff=float(K_eff), T_n=float(T_n), kind=m.kind,
                      model=dict(K=m.K, tau=m.tau, theta=m.theta, nrmse=m.nrmse, u_op=m.u_op))


# ----------------------------------------------------------------------------- gain normalisation
def normalise_gains(gains, K_eff, T_n):
    """log10 of the dimensionless gains (Kp K_eff, Ki K_eff T_n, Kd K_eff / T_n); Kd = 0 is floored."""
    kp, ki, kd = gains["Kp"], gains["Ki"], gains["Kd"]
    return np.log10([max(kp * K_eff, 1e-9), max(ki * K_eff * T_n, 1e-9), max(kd * K_eff / T_n, 1e-9)])


def denormalise_gains(z, K_eff, T_n):
    """Inverse of `normalise_gains` (z = log10 dimensionless gains) -> physical gains dict."""
    z = np.asarray(z, dtype=float)
    return dict(Kp=float(10 ** z[0] / K_eff), Ki=float(10 ** z[1] / (K_eff * T_n)), Kd=float(10 ** z[2] * T_n / K_eff))


def normalise_cost(itae, y_ref, T_n):
    """Dimensionless ITAE: ITAE / (|y_ref| T_n^2)."""
    return itae / (abs(y_ref) * T_n ** 2)
