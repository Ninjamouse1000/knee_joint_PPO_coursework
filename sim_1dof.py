"""
sim_1dof.py

Minimal 1-DOF torque-driven joint simulator (no gravity, no coupling).
Dynamics:
  theta_ddot = (tau - b*dtheta) / I
Discrete Euler integration.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from knobs import SimKnobs


@dataclass
class SimState:
    theta: float = 0.0
    dtheta: float = 0.0


class OneDOFSim:
    def __init__(self, knobs: SimKnobs):
        self.k = knobs
        self.state = SimState()

    @property
    def dt(self) -> float:
        return self.k.dt

    def reset(self, theta: float = 0.0, dtheta: float = 0.0) -> SimState:
        self.state = SimState(theta=float(theta), dtheta=float(dtheta))
        return self.state

    def step(self, tau: float) -> SimState:
        # Clip torque to actuator limits
        tau = float(np.clip(tau, -self.k.tau_max, self.k.tau_max))

        I = self.k.inertia_I
        b = self.k.damping_b
        dt = self.k.dt

        theta = self.state.theta
        dtheta = self.state.dtheta

        ddtheta = (tau - b * dtheta) / I

        dtheta_new = dtheta + ddtheta * dt
        theta_new = theta + dtheta_new * dt  # semi-implicit Euler is a touch more stable than explicit

        self.state = SimState(theta=theta_new, dtheta=dtheta_new)
        return self.state
