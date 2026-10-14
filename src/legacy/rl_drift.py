"""
Drift Model — Phase 5, Section 3 (Phase5_RL_Spec.md)

Models the thermal drift of armature resistance during sustained motor operation:

    R(t) = R0 * (1 + k_heat * t)

Physical motivation: I²R heating raises winding temperature; copper resistivity
increases with temperature, so R rises throughout a long run.

This module is intentionally tiny — a single function imported by rl_env.py.
Do NOT modify plant.py; the drift is applied only inside the RL environment's
step loop where R is passed explicitly to the ODE at each physics timestep.
"""


def compute_R(R0: float, k_heat: float, t: float) -> float:
    """
    Return the time-varying armature resistance at elapsed time t.

    Args:
        R0:     Baseline resistance at t=0 (Ohms). Default plant value: 1.0.
        k_heat: Heating-rate coefficient (1/s).
                - 0.05  → R doubles over a 20-second episode (training default).
                - 0.02  → slower drift (generalization test).
                - 0.10  → faster drift (generalization test).
                - 0.0   → no drift (static case, equivalent to drift=False in env).
        t:      Elapsed time since episode start (s).

    Returns:
        R(t) in Ohms.
    """
    return R0 * (1.0 + k_heat * t)
