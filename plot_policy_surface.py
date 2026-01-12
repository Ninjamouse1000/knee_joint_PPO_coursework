"""
plot_policy_surface.py

3D surface plot of the learned PPO policy as a function of (e, de),
with theta*, dtheta* held constant (a policy "slice").
Optionally plots the PD surface too.

Run:
  python plot_policy_surface.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  # colormap is fine
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from knobs import SimKnobs, PDKnobs
from controllers import pd_control

try:
    from stable_baselines3 import PPO
except ImportError:
    raise ImportError("Install stable-baselines3 and gymnasium first.")


def main():
    sim_k = SimKnobs()
    pd_k = PDKnobs()

    # Load your trained policy
    model_path = os.path.join("runs_ppo", "ppo_1dof_tracking.zip")
    model = PPO.load(model_path)

    # ---- Choose slice settings (hold these constant) ----
    theta_star = 0.0
    dtheta_star = 0.0

    # Grid over error space
    e_min, e_max, e_n = -1.0, 1.0, 81
    de_min, de_max, de_n = -4.0, 4.0, 81

    e_vals = np.linspace(e_min, e_max, e_n)
    de_vals = np.linspace(de_min, de_max, de_n)
    E, DE = np.meshgrid(e_vals, de_vals)

    # Evaluate PPO policy
    TAU_PPO = np.zeros_like(E, dtype=float)
    for i in range(DE.shape[0]):
        for j in range(DE.shape[1]):
            e = E[i, j]
            de = DE[i, j]
            obs = np.array([e, de, theta_star, dtheta_star], dtype=np.float32)
            action, _ = model.predict(obs, deterministic=True)
            a = float(np.asarray(action).reshape(-1)[0])
            TAU_PPO[i, j] = sim_k.tau_max * a

    # Evaluate PD surface (optional)
    TAU_PD = np.zeros_like(E, dtype=float)
    for i in range(DE.shape[0]):
        for j in range(DE.shape[1]):
            e = E[i, j]
            de = DE[i, j]
            # Convert (e, de) back into (theta, dtheta) given target held fixed:
            # e = theta* - theta => theta = theta* - e
            # de = dtheta* - dtheta => dtheta = dtheta* - de
            theta = theta_star - e
            dtheta = dtheta_star - de
            tau = pd_control(theta, dtheta, theta_star, dtheta_star, sim_k, pd_k)
            TAU_PD[i, j] = tau

    # ---- Plot PPO surface ----
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(E, DE, TAU_PPO, linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel("e = theta* - theta (rad)")
    ax.set_ylabel("de = dtheta* - dtheta (rad/s)")
    ax.set_zlabel("tau_PPO (N·m)")
    ax.set_title("PPO policy slice: tau(e, de) at theta*=0, dtheta*=0")
    plt.tight_layout()
    plt.show()

    # ---- Plot difference surface (PPO - PD) ----
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(E, DE, TAU_PPO - TAU_PD, linewidth=0, antialiased=True, alpha=0.9)
    ax.set_xlabel("e (rad)")
    ax.set_ylabel("de (rad/s)")
    ax.set_zlabel("tau_PPO - tau_PD (N·m)")
    ax.set_title("Policy difference slice: PPO minus PD")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
