"""
target.py

Reference trajectory generators.
Default: sine wave target.
"""

from __future__ import annotations

import numpy as np

from knobs import TargetKnobs


class SineTarget:
    def __init__(self, knobs: TargetKnobs, rng: np.random.Generator | None = None):
        self.k = knobs
        self.rng = rng if rng is not None else np.random.default_rng()
        self._A = knobs.amplitude_A
        self._phi = 0.0

    def reset(self) -> None:
        # Randomize phase (and optionally amplitude) at episode start
        if self.k.randomize_phase:
            self._phi = float(self.rng.uniform(0.0, 2.0 * np.pi))
        else:
            self._phi = 0.0

        if self.k.randomize_amp:
            self._A = float(self.rng.uniform(self.k.amp_min, self.k.amp_max))
        else:
            self._A = float(self.k.amplitude_A)

    def state(self, t: float) -> tuple[float, float]:
        """
        Returns (theta_star, dtheta_star) at time t.
        """
        A = self._A
        f = self.k.frequency_f
        w = 2.0 * np.pi * f

        theta_star = A * np.sin(w * t + self._phi)
        dtheta_star = A * w * np.cos(w * t + self._phi)
        return float(theta_star), float(dtheta_star)
