import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ActionLoggingCallback(BaseCallback):
    """
    Logs actions during training for heatmap visualisation.
    Stores (timestep, action) pairs.
    """

    def __init__(self, max_entries: int = 2_000_000):
        super().__init__()
        self.timesteps = []
        self.actions = []
        self.max_entries = max_entries

    def _on_step(self) -> bool:
        actions = self.locals.get("actions", None)
        if actions is not None:
            for a in np.atleast_2d(actions):
                if len(self.timesteps) < self.max_entries:
                    self.timesteps.append(self.num_timesteps)
                    self.actions.append(float(a[0]))
        return True

    def save(self, path: str):
        np.savez_compressed(
            path,
            timesteps=np.asarray(self.timesteps, dtype=np.int64),
            actions=np.asarray(self.actions, dtype=np.float32),
        )
