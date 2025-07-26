from stable_baselines3 import PPO
from mountain_env import MountainPassEnv
import os
import time
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList

# Create the AirSim Gym environment with Monitor for episode stats

def make_env():
    return Monitor(MountainPassEnv())

env = DummyVecEnv([make_env])

model_path = "checkpoints/mountain_pass_model_2000_steps"

# Set up custom TensorBoard logger
new_logger = configure("ppo_tensorboard/PPO_8", ["stdout", "tensorboard"])

class TBCallback(BaseCallback):
    def _on_step(self):
        infos = self.locals["infos"]
        for info in infos:
            ep = info.get("episode")
            if ep:
                self.logger.record("episode/return", ep["r"])
                self.logger.record("episode/length", ep["l"])
                self.logger.dump(self.num_timesteps)
        return True

class SmartCheckpointCallback(BaseCallback):
    def __init__(self, save_freq=10_000, save_path="./checkpoints/", name_prefix="mountain_pass_model"):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.last_save_step = 0
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self):
        current_step = self.num_timesteps
        
        # Check if we've reached or exceeded a multiple of save_freq
        if current_step >= self.last_save_step + self.save_freq:
            save_path = os.path.join(self.save_path, f"{self.name_prefix}_{current_step}_steps")
            self.model.save(save_path)
            self.last_save_step = current_step
            print(f"[INFO] Saved checkpoint at {current_step} total training steps")
        
        return True

# Resume training if a saved model exists, otherwise start fresh
if os.path.exists(model_path + ".zip"):
    print(f"[INFO] Loading existing model from {model_path}.zip and resuming training...")
    model = PPO.load(model_path, env=env)
else:
    print("[INFO] No saved model found. Starting fresh training...")
    model = PPO("CnnPolicy", env, verbose=2, tensorboard_log="./ppo_tensorboard/")

model.set_logger(new_logger)

# Use smart checkpoint callback that saves at multiples of 10,000 steps
checkpoint_callback = SmartCheckpointCallback(save_freq=10_000, save_path='./checkpoints/',
                                             name_prefix='mountain_pass_model')

# Combine TBCallback and SmartCheckpointCallback
callback = CallbackList([TBCallback(), checkpoint_callback])

# When calling model.learn, add the combined callback
model.learn(total_timesteps=100_000, callback=callback)

# Save the trained model
model.save(model_path)
print(f"[INFO] Model saved to {model_path}") 