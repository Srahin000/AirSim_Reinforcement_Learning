#!/usr/bin/env python3
"""
Test script to verify checkpoint loading works correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import setup_path

from stable_baselines3 import PPO
from mountain_env_clean import MountainPassEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def make_env():
    """Create the environment with monitoring."""
    return Monitor(MountainPassEnv(
        max_steps=200,
        step_length=3.0,
        altitude_step=2.0,
        lidar_safety_distance=2.0,
        ground_safety_distance=1.5,
        max_altitude=40.0,
        min_altitude=1.0,
        hard_reset_on_collision=False
    ))

def main():
    """Test checkpoint loading."""
    print("[TEST] Testing checkpoint loading...")
    
    # Create environment
    env = DummyVecEnv([make_env])
    
    # Check paths
    model_path = "checkpoints/mountain_clean_model"
    latest_model_path = "checkpoints/mountain_clean_model_latest"
    
    print(f"[TEST] Current directory: {os.getcwd()}")
    print(f"[TEST] Latest checkpoint exists: {os.path.exists(latest_model_path + '.zip')}")
    print(f"[TEST] Main checkpoint exists: {os.path.exists(model_path + '.zip')}")
    
    # Try to load the latest checkpoint
    if os.path.exists(latest_model_path + ".zip"):
        print(f"[TEST] Attempting to load latest checkpoint...")
        try:
            model = PPO.load(latest_model_path, env=env)
            print(f"[SUCCESS] Latest checkpoint loaded successfully!")
            print(f"[INFO] Model info: {type(model)}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load latest checkpoint: {e}")
            return False
    else:
        print(f"[ERROR] Latest checkpoint not found!")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n[SUCCESS] Checkpoint loading test passed!")
    else:
        print("\n[FAILURE] Checkpoint loading test failed!") 