"""
optimize_pd_control.py

Grid-search PD gains (Kp, Kd) and create a heatmap to help pick good values.

Metric (objective) is set in PDSweepKnobs:
  - "rmse_theta": lower is better
  - "mean_abs_tau": lower is better

Run:
  python optimize_pd_control.py
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

from knobs import SimKnobs, TargetKnobs, PDKnobs, PDSweepKnobs
from sim_1dof import OneDOFSim
from target import SineTarget
from controllers import pd_control


def run_episode(sim_k: SimKnobs, target_k: TargetKnobs, pd_k: PDKnobs, seed: int = 0):
    rng = np.random.default_rng(seed)
    sim = OneDOFSim(sim_k)
    target = SineTarget(target_k, rng=rng)

    sim.reset(theta=0.0, dtheta=0.0)
    target.reset()

    dt = sim_k.dt
    steps = int(np.round(sim_k.horizon_s / dt))

    es = []
    taus = []

    t = 0.0
    for _ in range(steps):
        theta_star, dtheta_star = target.state(t)
        theta, dtheta = sim.state.theta, sim.state.dtheta

        tau = pd_control(theta, dtheta, theta_star, dtheta_star, sim_k, pd_k)
        sim.step(tau)

        e = theta_star - sim.state.theta
        es.append(e)
        taus.append(tau)

        t += dt

        if abs(sim.state.theta) > sim_k.theta_limit:
            break

    es = np.asarray(es, dtype=float)
    taus = np.asarray(taus, dtype=float)
    rmse_theta = float(np.sqrt(np.mean(es ** 2))) if len(es) else float("inf")
    mean_abs_tau = float(np.mean(np.abs(taus))) if len(taus) else float("inf")
    return rmse_theta, mean_abs_tau


def main():
    sim_k = SimKnobs()
    target_k = TargetKnobs()
    sweep_k = PDSweepKnobs()

    kps = np.linspace(sweep_k.kp_min, sweep_k.kp_max, sweep_k.kp_points)
    kds = np.linspace(sweep_k.kd_min, sweep_k.kd_max, sweep_k.kd_points)

    scores = np.zeros((len(kds), len(kps)), dtype=float)  # rows=Kd, cols=Kp

    # Use a couple of seeds to smooth out phase randomization
    seeds = [0, 1, 2]

    for i_kd, kd in enumerate(kds):
        for j_kp, kp in enumerate(kps):
            pd_k = PDKnobs(kp=float(kp), kd=float(kd))
            vals = []
            print(f"Evaluating Kp={kp:.2f}, Kd={kd:.2f}...")
            for s in seeds:
                rmse, mean_abs_tau = run_episode(sim_k, target_k, pd_k, seed=s)
                if sweep_k.objective == "mean_abs_tau":
                    vals.append(mean_abs_tau)
                else:
                    vals.append(rmse)
            scores[i_kd, j_kp] = float(np.mean(vals))

    # Find best
    best_idx = np.unravel_index(np.argmin(scores), scores.shape)
    best_kd = float(kds[best_idx[0]])
    best_kp = float(kps[best_idx[1]])
    best_score = float(scores[best_idx])

    out_dir = "runs_pd"
    os.makedirs(out_dir, exist_ok=True)

    # Heatmap
    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        scores,
        origin="lower",
        aspect="auto",
        extent=[kps[0], kps[-1], kds[0], kds[-1]],
    )
    plt.colorbar(im, label=sweep_k.objective)
    plt.scatter([best_kp], [best_kd], marker="x")
    plt.xlabel("Kp")
    plt.ylabel("Kd")
    plt.title(f"PD gain sweep ({sweep_k.objective}); best Kp={best_kp:.2f}, Kd={best_kd:.2f}")
    plt.grid(False)

    fig_path = os.path.join(out_dir, f"pd_heatmap_{sweep_k.objective}.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)

    # Save best to a JSON
    import json
    best = {"kp": best_kp, "kd": best_kd, "score": best_score, "objective": sweep_k.objective}
    with open(os.path.join(out_dir, "pd_best.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("Best PD gains:")
    print(best)
    print(f"Saved heatmap to: {fig_path}")


if __name__ == "__main__":
    main()
