"""
env_ppo.py

Gym-compatible environment for 1-DOF tracking.
Observation (recommended): [e, de, theta*, dtheta*]
Action: scalar in [-1, 1] scaled to torque tau_max
Reward: -(w_e e^2 + w_de de^2 + w_tau tau^2)
"""

from __future__ import annotations

import numpy as np

# Prefer gymnasium if installed (SB3 v2+). Fallback to gym.
import gymnasium as gym
from gymnasium import spaces
_GYMNASIUM = True

from knobs import SimKnobs, TargetKnobs, RewardKnobs
from sim_1dof import OneDOFSim
from target import SineTarget


class OneDOFTrackingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        sim_knobs: SimKnobs,
        target_knobs: TargetKnobs,
        reward_knobs: RewardKnobs,
        seed: int | None = None,
    ):
        super().__init__()
        self.sim_k = sim_knobs
        self.target_k = target_knobs
        self.reward_k = reward_knobs

        self.sim = OneDOFSim(sim_knobs)

        self._rng = np.random.default_rng(seed)
        self.target = SineTarget(target_knobs, rng=self._rng)

        # Spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation: [e, de, theta*, dtheta*]
        high = np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self._t = 0.0
        self._step_i = 0
        self._max_steps = int(np.round(self.sim_k.horizon_s / self.sim_k.dt))

    def _obs(self) -> np.ndarray:
        theta_star, dtheta_star = self.target.state(self._t)
        theta, dtheta = self.sim.state.theta, self.sim.state.dtheta
        e = theta_star - theta
        de = dtheta_star - dtheta
        return np.array([e, de, theta_star, dtheta_star], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.target = SineTarget(self.target_k, rng=self._rng)

        self.sim.reset(theta=0.0, dtheta=0.0)
        self.target.reset()

        self._t = 0.0
        self._step_i = 0

        obs = self._obs()
        info = {"t": self._t}
        if _GYMNASIUM:
            return obs, info
        return obs

    def step(self, action):
        # Action is [-1,1] -> torque [-tau_max, tau_max]
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        tau = self.sim_k.tau_max * a

        theta_star, dtheta_star = self.target.state(self._t)
        theta, dtheta = self.sim.state.theta, self.sim.state.dtheta

        e = theta_star - theta
        de = dtheta_star - dtheta

        # Reward
        r = -(
            self.reward_k.w_e * (e * e)
            + self.reward_k.w_de * (de * de)
            + self.reward_k.w_tau * (tau * tau)
        )

        # Integrate dynamics to next state
        self.sim.step(tau)

        # Advance time
        self._step_i += 1
        self._t += self.sim_k.dt

        # Termination
        terminated = abs(self.sim.state.theta) > self.sim_k.theta_limit
        truncated = self._step_i >= self._max_steps

        obs = self._obs()
        info = {"t": self._t, "tau": tau, "e": e, "de": de}

        if _GYMNASIUM:
            return obs, float(r), bool(terminated), bool(truncated), info
        done = bool(terminated or truncated)
        return obs, float(r), done, info

    def render(self):
        # Minimal env: rendering intentionally omitted.
        # You can add a simple matplotlib animation later if needed.
        pass
