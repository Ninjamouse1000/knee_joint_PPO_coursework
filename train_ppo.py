"""
train_ppo.py

Train PPO on the OneDOFTrackingEnv using Stable Baselines3.
Saves:
  - trained model zip
  - training reward plot with rolling mean +/- std
  - Monitor logs

Run:
  python train_ppo.py
"""

from __future__ import annotations

import os
from dataclasses import asdict

import numpy as np
import matplotlib.pyplot as plt

from knobs import SimKnobs, TargetKnobs, RewardKnobs, PPOKnobs
from env_ppo import OneDOFTrackingEnv
from callbacks_action_logging import ActionLoggingCallback

# Stable Baselines 3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "stable-baselines3 is required. Install with: pip install stable-baselines3"
    ) from e


class EpisodeRewardCallback(BaseCallback):
    """
    Collects episode rewards from Monitor info and records (episode_idx, ep_reward, ep_len, num_timesteps).
    """

    def __init__(self):
        super().__init__()
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_timesteps = []

    def _on_step(self) -> bool:
        # SB3 puts episode info into infos when using Monitor
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])
                self.ep_timesteps.append(self.num_timesteps)
        return True


def rolling_mean_std(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) == 0:
        return np.array([]), np.array([])
    window = max(1, int(window))
    means = np.empty_like(x, dtype=float)
    stds = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window + 1)
        seg = x[start : i + 1]
        means[i] = float(np.mean(seg))
        stds[i] = float(np.std(seg))
    return means, stds


def make_env(sim_k: SimKnobs, target_k: TargetKnobs, reward_k: RewardKnobs, seed: int):
    def _thunk():
        env = OneDOFTrackingEnv(sim_k, target_k, reward_k, seed=seed)
        return Monitor(env)
    return _thunk


def main():
    sim_k = SimKnobs()
    target_k = TargetKnobs()
    reward_k = RewardKnobs()
    ppo_k = PPOKnobs()

    out_dir = "runs_ppo"
    os.makedirs(out_dir, exist_ok=True)

    # Vectorized environments
    env_fns = [make_env(sim_k, target_k, reward_k, seed=ppo_k.seed + i) for i in range(ppo_k.n_envs)]
    vec_env = DummyVecEnv(env_fns)

    # Policy network settings
    if ppo_k.activation.lower() == "relu":
        import torch as th
        activation_fn = th.nn.ReLU
    else:
        import torch as th
        activation_fn = th.nn.Tanh

    policy_kwargs = dict(
        activation_fn=activation_fn,
        net_arch=dict(pi=[ppo_k.net_arch_pi], vf=[ppo_k.net_arch_vf]),
    )

    callback = EpisodeRewardCallback()
    action_cb = ActionLoggingCallback()


    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        seed=ppo_k.seed,
        learning_rate=ppo_k.learning_rate,
        n_steps=ppo_k.n_steps,
        batch_size=ppo_k.batch_size,
        n_epochs=ppo_k.n_epochs,
        gamma=ppo_k.gamma,
        gae_lambda=ppo_k.gae_lambda,
        clip_range=ppo_k.clip_range,
        ent_coef=ppo_k.ent_coef,
        vf_coef=ppo_k.vf_coef,
        max_grad_norm=ppo_k.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )

    model.learn(total_timesteps=ppo_k.total_timesteps, callback=[callback, action_cb])

    model_path = os.path.join(out_dir, "ppo_1dof_tracking")
    model.save(model_path)
    action_cb.save(os.path.join(out_dir, "action_log.npz"))
    # Save knobs for reproducibility
    with open(os.path.join(out_dir, "knobs_used.json"), "w", encoding="utf-8") as f:
        jsonable = {
            "SimKnobs": asdict(sim_k),
            "TargetKnobs": asdict(target_k),
            "RewardKnobs": asdict(reward_k),
            "PPOKnobs": asdict(ppo_k),
        }
        import json
        json.dump(jsonable, f, indent=2)

    # Plot learning curve: rolling mean +/- std of episode rewards
    ep_r = np.asarray(callback.ep_rewards, dtype=float)
    ep_t = np.asarray(callback.ep_timesteps, dtype=int)

    mean_r, std_r = rolling_mean_std(ep_r, ppo_k.rolling_window_episodes)

    plt.figure()
    plt.plot(ep_t, mean_r, label="Rolling mean episode reward")
    plt.fill_between(ep_t, mean_r - std_r, mean_r + std_r, alpha=0.2, label="±1 std")
    plt.xlabel("Training timesteps")
    plt.ylabel("Episode reward")
    plt.title("PPO learning curve (episode reward)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_path = os.path.join(out_dir, "learning_curve_mean_std.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)

    print(f"Saved model to: {model_path}.zip")
    print(f"Saved learning plot to: {plot_path}")
    print(f"Episodes logged: {len(ep_r)}")


if __name__ == "__main__":
    main()
