# knee_joint_PPO_coursework
Reinforcement learning coursework aiming to implement a Proximal Policy Optimisation controller for robotic assistive exoskeletons

GPT-5 mini was used extensively as a coding assistant

The only files which really matter ie dont just do plotting are:
knobs.py        -contains all the variables which can be played with
sim_1dof.py     -contains the physics simulation
target.py       -generates the prescribed therapeutic movement
controllers.py  -contains the PD-controller
env_ppo.py      -sets up the gym environment
train_ppo.py    -actually trains ppo, outputs model, learning curves and .npz to be used for action plot
eval_compare.py -time domain plots of angle and torque

other files include
callbacks_action_logging.py -a GPT-5 miracle which retcons in action logging
optimise_pd_control.py      -grid searches for best pd gains
plot_action_heatmap.py      -generates the action frequency plot
visualise_pd.py             -now redundant, check to make sure pd is behaving

to minimally check it does what it says on the tin, run train_ppo.py followed by eval_compare
