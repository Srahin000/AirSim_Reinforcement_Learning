#!/usr/bin/env python3
"""
Clean Testing Script for Mountain Pass Environment with Lidar
Test the trained model and visualize performance.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import setup_path

from stable_baselines3 import PPO
from mountain_env_clean import MountainPassEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
import time

def make_env():
    """Create the environment for testing."""
    return Monitor(MountainPassEnv(
        max_steps=200,
        step_length=4.0,
        altitude_step=2.0,
        lidar_safety_distance=2.0,
        ground_safety_distance=1.5,
        max_altitude=30.0,
        min_altitude=1.0
    ))

def test_model(model_path, num_episodes=5):
    """Test the trained model."""
    print(f"[INFO] Testing model from: {model_path}")
    
    # Create environment
    env = DummyVecEnv([make_env])
    
    # Load model
    try:
        model = PPO.load(model_path, env=env)
        print("[SUCCESS] Model loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return
    
    # Test statistics
    total_episodes = 0
    successful_episodes = 0
    total_rewards = []
    episode_lengths = []
    collision_count = 0
    unsafe_count = 0
    
    print(f"\n[INFO] Running {num_episodes} test episodes...")
    print("=" * 60)
    
    for episode in range(num_episodes):
        print(f"\n[EPISODE {episode + 1}/{num_episodes}]")
        print("-" * 40)
        
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_collision = False
        episode_unsafe = False
        episode_goal_reached = False
        
        # Debug: Print observation structure
        if episode == 0:
            print(f"[DEBUG] Observation keys: {obs[0].keys()}")
            print(f"[DEBUG] Depth image shape: {obs[0]['depth_image'].shape}")
            print(f"[DEBUG] Lidar data shape: {obs[0]['lidar_data'].shape}")
            print(f"[DEBUG] Lidar values: {obs[0]['lidar_data']}")
        
        for step in range(200):  # Max 200 steps per episode
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)
            
            # Take step
            obs, rewards, dones, info = env.step(action)
            
            episode_reward += rewards[0]
            episode_length += 1
            
            # Track episode outcomes
            if info[0].get('collision', False):
                episode_collision = True
                collision_count += 1
            
            if info[0].get('unsafe', False):
                episode_unsafe = True
                unsafe_count += 1
            
            if info[0].get('goal_reached', False):
                episode_goal_reached = True
                successful_episodes += 1
            
            # Print step info
            distance = info[0].get('distance_to_goal', 0)
            lidar_ground = info[0].get('lidar_ground_dist', None)
            lidar_horizontal = info[0].get('lidar_horizontal_dist', None)
            
            print(f"Step {step+1:3d}: Action={action[0]}, Reward={rewards[0]:6.2f}, "
                  f"Distance={distance:6.1f}, Ground={lidar_ground:5.1f if lidar_ground else 'None'}, "
                  f"Horizontal={lidar_horizontal:5.1f if lidar_horizontal else 'None'}")
            
            if dones[0]:
                break
        
        # Episode summary
        total_episodes += 1
        total_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        status = []
        if episode_collision:
            status.append("COLLISION")
        if episode_unsafe:
            status.append("UNSAFE")
        if episode_goal_reached:
            status.append("GOAL REACHED")
        if not status:
            status.append("TIMEOUT")
        
        print(f"Episode {episode + 1} Summary:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Episode Length: {episode_length}")
        print(f"  Status: {' | '.join(status)}")
    
    # Overall statistics
    print("\n" + "=" * 60)
    print("OVERALL TEST RESULTS")
    print("=" * 60)
    
    success_rate = (successful_episodes / total_episodes) * 100
    collision_rate = (collision_count / total_episodes) * 100
    unsafe_rate = (unsafe_count / total_episodes) * 100
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)
    
    print(f"Total Episodes: {total_episodes}")
    print(f"Successful Episodes: {successful_episodes} ({success_rate:.1f}%)")
    print(f"Collision Rate: {collision_rate:.1f}%")
    print(f"Unsafe Rate: {unsafe_rate:.1f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.1f}")
    
    if success_rate > 50:
        print("\n[SUCCESS] Model is performing well!")
    elif success_rate > 20:
        print("\n[WARNING] Model needs more training.")
    else:
        print("\n[ERROR] Model needs significant improvement.")
    
    env.close()

def test_environment_only():
    """Test the environment without a trained model."""
    print("[INFO] Testing environment only...")
    
    env = MountainPassEnv()
    
    # Test reset
    print("[INFO] Testing environment reset...")
    obs, info = env.reset()
    print(f"[SUCCESS] Reset completed!")
    print(f"[INFO] Observation keys: {obs.keys()}")
    print(f"[INFO] Depth image shape: {obs['depth_image'].shape}")
    print(f"[INFO] Lidar data shape: {obs['lidar_data'].shape}")
    print(f"[INFO] Lidar values: {obs['lidar_data']}")
    
    # Test random actions
    print("[INFO] Testing random actions...")
    for i in range(10):
        action = np.random.randint(0, 5)  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        distance = info.get('distance_to_goal', 0)
        lidar_ground = info.get('lidar_ground_dist', None)
        lidar_horizontal = info.get('lidar_horizontal_dist', None)
        
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, "
              f"Distance={distance:.1f}, Ground={lidar_ground:.1f if lidar_ground else 'None'}, "
              f"Horizontal={lidar_horizontal:.1f if lidar_horizontal else 'None'}")
        
        if terminated or truncated:
            print(f"Episode ended: {'Goal reached' if info.get('goal_reached') else 'Collision/Unsafe'}")
            break
    
    env.close()
    print("[SUCCESS] Environment test completed!")

def main():
    """Main testing function."""
    print("=" * 60)
    print("MOUNTAIN PASS ENVIRONMENT TESTING")
    print("=" * 60)
    
    # Test environment first
    print("\n[TEST 1] Environment Test")
    print("-" * 30)
    test_environment_only()
    
    # Test model if it exists
    model_path = "checkpoints/mountain_clean_model"
    if os.path.exists(model_path + ".zip"):
        print(f"\n[TEST 2] Model Test")
        print("-" * 30)
        test_model(model_path, num_episodes=3)
    else:
        print(f"\n[INFO] No trained model found at {model_path}")
        print("[INFO] Run train_clean.py first to train a model.")
    
    print("\n[INFO] Testing completed!")

if __name__ == "__main__":
    main() 