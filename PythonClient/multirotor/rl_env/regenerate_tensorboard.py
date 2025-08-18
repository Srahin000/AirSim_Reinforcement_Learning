#!/usr/bin/env python3
"""
Quick script to regenerate TensorBoard logs
This will run a very short training session to create valid event files
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import setup_path

from stable_baselines3 import PPO
from mountain_env_clean import MountainPassEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import numpy as np

def make_env():
    """Create a simple environment for quick testing."""
    return Monitor(MountainPassEnv(
        max_steps=100,  # Very short episodes
        step_length=3.0,
        altitude_step=2.0,
        lidar_safety_distance=2.0,
        ground_safety_distance=1.5,
        max_altitude=40.0,
        min_altitude=1.0,
        hard_reset_on_collision=False
    ))

def main():
    print("[INFO] Creating environment...")
    env = DummyVecEnv([make_env])
    
    # Set up logger with tensorboard
    new_logger = configure("ppo_clean_tensorboard/PPO_8", ["tensorboard", "stdout"])
    
    print("[INFO] Creating PPO model...")
    model = PPO(
        "MlpPolicy",  # Use simple MLP policy for quick training
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=64,  # Small batch for quick training
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./ppo_clean_tensorboard/",
    )
    
    model.set_logger(new_logger)
    
    print("[INFO] Starting quick training session to generate TensorBoard logs...")
    print("[INFO] Training for only 1000 timesteps to create valid event files...")
    
    # Train for just 1000 timesteps to generate valid logs
    model.learn(total_timesteps=1000, progress_bar=True)
    
    print("[INFO] Quick training completed!")
    print("[INFO] TensorBoard logs should now be available at ./ppo_clean_tensorboard/")
    print("[INFO] You can now run: python -m tensorboard.main --logdir=ppo_clean_tensorboard --port=6008")
    
    # Test the model briefly
    print("[INFO] Testing model...")
    obs = env.reset()
    total_reward = 0
    
    for i in range(20):  # Very short test
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        total_reward += rewards[0]
        
        if dones[0]:
            print(f"Episode ended after {i+1} steps with total reward: {total_reward:.2f}")
            break
    
    print("[INFO] Done! Check your TensorBoard now.")

if __name__ == "__main__":
    main()
