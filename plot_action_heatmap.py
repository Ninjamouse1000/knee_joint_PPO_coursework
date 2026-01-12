"""
plot_action_heatmap.py

Heatmap of action frequency vs training timestep.

Run:
  python plot_action_heatmap.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from knobs import SimKnobs


def main():
    sim_k = SimKnobs()
    log_path = os.path.join("runs_ppo", "action_log.npz")

    data = np.load(log_path)
    timesteps = data["timesteps"]
    actions = data["actions"] * sim_k.tau_max  # convert to torque

    # Bin settings
    n_time_bins = 30
    n_action_bins = 20

    t_bins = np.linspace(timesteps.min(), timesteps.max(), n_time_bins)
    a_bins = np.linspace(-sim_k.tau_max, sim_k.tau_max, n_action_bins)

    H, _, _ = np.histogram2d(
        timesteps,
        actions,
        bins=[t_bins, a_bins],
    )

    # Normalize per time bin (optional but recommended)
    H = H / (H.sum(axis=1, keepdims=True) + 1e-8)

    plt.figure(figsize=(10, 5))
    plt.imshow(
        H.T,
        origin="lower",
        aspect="auto",
        extent=[t_bins[0], t_bins[-1], a_bins[0], a_bins[-1]],
    )

    plt.xlabel("Training timesteps")
    plt.ylabel("Torque τ (N·m)")
    plt.title("Action frequency heatmap over training")
    plt.colorbar(label="Normalized frequency")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
