"""
eval_compare.py

Compare PD vs PPO on identical tracking episodes.
.

"""

from __future__ import annotations

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from knobs import SimKnobs, TargetKnobs, RewardKnobs, PDKnobs, EvalKnobs
from sim_1dof import OneDOFSim
from target import SineTarget
from controllers import pd_control
from stable_baselines3 import PPO



def rollout_pd(sim_k: SimKnobs, target_k: TargetKnobs, pd_k: PDKnobs, seed: int):
    rng = np.random.default_rng(seed)
    sim = OneDOFSim(sim_k)
    target = SineTarget(target_k, rng=rng)

    sim.reset(0.0, 0.0)
    target.reset()

    dt = sim_k.dt
    steps = int(np.round(sim_k.horizon_s / dt))

    t_hist, theta_hist, theta_star_hist, tau_hist, e_hist = [], [], [], [], []
    t = 0.0
    for _ in range(steps):
        theta_star, dtheta_star = target.state(t)
        theta, dtheta = sim.state.theta, sim.state.dtheta

        tau = pd_control(theta, dtheta, theta_star, dtheta_star, sim_k, pd_k)
        sim.step(tau)

        e = theta_star - sim.state.theta

        t_hist.append(t)
        theta_hist.append(sim.state.theta)
        theta_star_hist.append(theta_star)
        tau_hist.append(tau)
        e_hist.append(e)

        t += dt
        if abs(sim.state.theta) > sim_k.theta_limit:
            break

    return {
        "t": np.asarray(t_hist),
        "theta": np.asarray(theta_hist),
        "theta_star": np.asarray(theta_star_hist),
        "tau": np.asarray(tau_hist),
        "e": np.asarray(e_hist),
    }


def rollout_ppo(sim_k: SimKnobs, target_k: TargetKnobs, reward_k: RewardKnobs, model: PPO, seed: int, deterministic: bool):
    # Import env here to avoid circular
    from env_ppo import OneDOFTrackingEnv

    env = OneDOFTrackingEnv(sim_k, target_k, reward_k, seed=seed)
    reset_out = env.reset(seed=seed)
    if isinstance(reset_out, tuple):
        obs, _info = reset_out
    else:
        obs = reset_out

    dt = sim_k.dt
    steps = int(np.round(sim_k.horizon_s / dt))

    t_hist, theta_hist, theta_star_hist, tau_hist, e_hist = [], [], [], [], []
    t = 0.0

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, _r, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            obs, _r, done, info = step_out

        # env info stores tau, but theta is inside sim
        theta = env.sim.state.theta
        theta_star, _dtheta_star = env.target.state(env._t)

        t_hist.append(t)
        theta_hist.append(theta)
        theta_star_hist.append(theta_star)
        tau_hist.append(info.get("tau", 0.0))
        e_hist.append(info.get("e", theta_star - theta))

        t += dt
        if done:
            break

    return {
        "t": np.asarray(t_hist),
        "theta": np.asarray(theta_hist),
        "theta_star": np.asarray(theta_star_hist),
        "tau": np.asarray(tau_hist),
        "e": np.asarray(e_hist),
    }


def metrics(traj: dict):
    e = traj["e"]
    tau = traj["tau"]
    rmse_theta = float(np.sqrt(np.mean(e ** 2))) if len(e) else float("inf")
    mean_abs_tau = float(np.mean(np.abs(tau))) if len(tau) else float("inf")
    smooth = float(np.mean(np.diff(tau) ** 2)) if len(tau) > 1 else float("inf")
    return {"rmse_theta": rmse_theta, "mean_abs_tau": mean_abs_tau, "tau_smoothness": smooth}


def main():
    sim_k = SimKnobs()
    target_k = TargetKnobs()
    reward_k = RewardKnobs()
    pd_k = PDKnobs()
    eval_k = EvalKnobs()

    #load in the PPO model (zip)
    model_path = os.path.join("runs_ppo", "ppo_1dof_tracking.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find trained PPO model at {model_path}. Run train_ppo.py first.")

    model = PPO.load(model_path)

    # do multiple episodes
    pd_ms = []
    ppo_ms = []
    for ep in range(eval_k.n_episodes):
        seed = 1000 + ep
        pd_traj = rollout_pd(sim_k, target_k, pd_k, seed=seed)
        ppo_traj = rollout_ppo(sim_k, target_k, reward_k, model, seed=seed, deterministic=eval_k.deterministic_policy)

        pd_ms.append(metrics(pd_traj))
        ppo_ms.append(metrics(ppo_traj))

    def agg(ms):
        keys = ms[0].keys()
        return {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms]))) for k in keys}

    pd_agg = agg(pd_ms)
    ppo_agg = agg(ppo_ms)

    print("\n=== Metrics (mean ± std over episodes) ===")
    for k in pd_agg:
        print(f"{k:>16} | PD: {pd_agg[k][0]:.4f} ± {pd_agg[k][1]:.4f}   | PPO: {ppo_agg[k][0]:.4f} ± {ppo_agg[k][1]:.4f}")

    # Ploting one rollout
    seed = 999
    pd_traj = rollout_pd(sim_k, target_k, pd_k, seed=seed)
    ppo_traj = rollout_ppo(sim_k, target_k, reward_k, model, seed=seed, deterministic=True)

    out_dir = "runs_eval"
    os.makedirs(out_dir, exist_ok=True)

    plt.figure()
    plt.plot(pd_traj["t"], pd_traj["theta_star"], label="target")
    plt.plot(pd_traj["t"], pd_traj["theta"], label="PD")
    plt.plot(ppo_traj["t"], ppo_traj["theta"], label="PPO")
    plt.xlabel("time (s)")
    plt.ylabel("theta (rad)")
    plt.title("Tracking comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tracking_comparison.png"), dpi=170)

    plt.figure()
    plt.plot(pd_traj["t"], pd_traj["tau"], label="PD tau")
    plt.plot(ppo_traj["t"], ppo_traj["tau"], label="PPO tau")
    plt.xlabel("time (s)")
    plt.ylabel("tau (N·m)")
    plt.title("Torque comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "torque_comparison.png"), dpi=170)

    # Save metrics JSON
    summary = {"pd": pd_agg, "ppo": ppo_agg}
    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved plots + metrics to: {out_dir}/")


if __name__ == "__main__":
    main()
