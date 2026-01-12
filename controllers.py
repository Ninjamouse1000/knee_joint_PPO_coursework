"""
controllers.py

Classical Proportional Derivative controller for tracking baselines.
"""

from __future__ import annotations

import numpy as np

from knobs import SimKnobs, PDKnobs


def pd_control(
    theta: float,
    dtheta: float,
    theta_star: float,
    dtheta_star: float,
    sim_knobs: SimKnobs,
    pd_knobs: PDKnobs,
) -> float:

    e = theta_star - theta
    de = dtheta_star - dtheta
    tau = pd_knobs.kp * e + pd_knobs.kd * de
    return float(np.clip(tau, -sim_knobs.tau_max, sim_knobs.tau_max))
