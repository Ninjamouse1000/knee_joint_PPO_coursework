"""
visualise_pd.py

Minimal time-domain visualiser for target vs PD tracker.

Shows:
  - theta* (target) vs theta (PD)
  - tau over time

Run:
  python visualise_pd.py
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from knobs import SimKnobs, TargetKnobs, PDKnobs, VisualiserKnobs
from sim_1dof import OneDOFSim
from target import SineTarget
from controllers import pd_control


def rollout_pd(sim_k: SimKnobs, target_k: TargetKnobs, pd_k: PDKnobs, seed: int):
    rng = np.random.default_rng(seed)
    sim = OneDOFSim(sim_k)
    target = SineTarget(target_k, rng=rng)

    sim.reset(theta=0.0, dtheta=0.0)
    target.reset()

    dt = sim_k.dt
    steps = int(np.round(sim_k.horizon_s / dt))

    t_hist = np.zeros(steps, dtype=float)
    theta_hist = np.zeros(steps, dtype=float)
    theta_star_hist = np.zeros(steps, dtype=float)
    tau_hist = np.zeros(steps, dtype=float)

    t = 0.0
    n = 0
    for i in range(steps):
        theta_star, dtheta_star = target.state(t)
        theta, dtheta = sim.state.theta, sim.state.dtheta

        tau = pd_control(theta, dtheta, theta_star, dtheta_star, sim_k, pd_k)
        sim.step(tau)

        t_hist[i] = t
        theta_hist[i] = sim.state.theta
        theta_star_hist[i] = theta_star
        tau_hist[i] = tau

        t += dt
        n = i + 1

        if abs(sim.state.theta) > sim_k.theta_limit:
            break

    return {
        "t": t_hist[:n],
        "theta": theta_hist[:n],
        "theta_star": theta_star_hist[:n],
        "tau": tau_hist[:n],
    }


def main():
    sim_k = SimKnobs()
    target_k = TargetKnobs()
    pd_k = PDKnobs()
    vis_k = VisualiserKnobs()

    traj = rollout_pd(sim_k, target_k, pd_k, seed=vis_k.seed)

    os.makedirs(vis_k.save_dir, exist_ok=True)

    # Plot tracking
    plt.figure()
    plt.plot(traj["t"], traj["theta_star"], label="target θ*")
    plt.plot(traj["t"], traj["theta"], label="PD θ")
    plt.xlabel("time (s)")
    plt.ylabel("theta (rad)")
    plt.title("1-DOF tracking: target vs PD")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if vis_k.save_png:
        out_path = os.path.join(vis_k.save_dir, "pd_tracking.png")
        plt.savefig(out_path, dpi=vis_k.dpi)

    # Plot torque
    plt.figure()
    plt.plot(traj["t"], traj["tau"], label="PD torque τ")
    plt.xlabel("time (s)")
    plt.ylabel("tau (N·m)")
    plt.title("1-DOF PD torque over time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if vis_k.save_png:
        out_path = os.path.join(vis_k.save_dir, "pd_torque.png")
        plt.savefig(out_path, dpi=vis_k.dpi)

    if vis_k.show:
        plt.show()


if __name__ == "__main__":
    main()
