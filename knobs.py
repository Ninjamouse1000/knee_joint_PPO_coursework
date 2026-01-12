"""
knobs.py

contains all the parameters which can be tuned for the one degree of freedom tracking problem.
I have defined separate dataclasses for different categories of parameters for ease of use.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SimKnobs:
    # time resolution and episode duration (seconds)
    dt: float = 0.02  
    horizon_s: float = 6.0  # episode length
    # Dynamics of the leg
    inertia_I: float = 0.08
    damping_b: float = 0.02
    # safty limits
    tau_max: float = 2.0
    theta_limit: float = 3.0


@dataclass(frozen=True)
class TargetKnobs: #defines target trajectory
    # Sine reference: theta*(t)=A sin(2π f t + phi)
    amplitude_A: float = 0.6  # radians
    frequency_f: float = 0.25  # Hz

    # Optional randomization at reset (for generalization)
    randomize_phase: bool = True
    randomize_amp: bool = False
    amp_min: float = 0.4
    amp_max: float = 0.8


@dataclass(frozen=True)
class PDKnobs:
    kp: float = 30.0
    kd: float = 3.0


@dataclass(frozen=True)
class RewardKnobs:
    # r = -(w_e e^2 + w_de de^2 + w_tau tau^2)
    w_e: float = 1.0
    w_de: float = 0.1
    w_tau: float = 0.001


@dataclass(frozen=True)
class PPOKnobs:
    seed: int = 0
    total_timesteps: int = 200_000

    # PPO hyperparameters (SB3)
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network
    net_arch_pi: int = 64
    net_arch_vf: int = 64
    activation: str = "tanh"  # "tanh" or "relu"

    # Vectorized envs
    n_envs: int = 8

    # Plotting
    rolling_window_episodes: int = 50


@dataclass(frozen=True)
class EvalKnobs:
    n_episodes: int = 10
    render: bool = False  # keep False for speed
    deterministic_policy: bool = True


@dataclass(frozen=True)
class PDSweepKnobs:
    # Grid search for PD gains
    kp_min: float = 0.1
    kp_max: float = 60.0
    kp_points: int = 61

    kd_min: float = 0.1
    kd_max: float = 20.0
    kd_points: int = 81

    # Metric to optimize: "rmse_theta" or "mean_abs_tau"
    objective: str = "rmse_theta"

@dataclass(frozen=True)
class VisualiserKnobs:
    seed: int = 0
    save_dir: str = "runs_vis"
    save_png: bool = True
    show: bool = True
    dpi: int = 170

