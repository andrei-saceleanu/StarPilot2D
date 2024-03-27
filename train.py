import os

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from gym_env import GameEnv

# Create log dir
log_dir = "logs/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = GameEnv("configs/game_config.yml", False)
env = Monitor(env, log_dir)

# Create DQN agent
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=100000, save_path=log_dir, name_prefix="rl_model_v0"
)

# Train the agent
model.learn(
    total_timesteps=10000000,
    callback=[checkpoint_callback]
)
