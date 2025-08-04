#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import setup_path

from mountain_env_random_goals import MountainPassRandomGoalsEnv
import time

def test_random_goals():
    """Test the randomized goals environment."""
    print("[INFO] Testing Mountain Pass Random Goals Environment...")
    
    # Create environment
    env = MountainPassRandomGoalsEnv()
    
    # Test reset
    print("[INFO] Testing reset...")
    obs, info = env.reset()
    print(f"[SUCCESS] Reset completed!")
    print(f"[INFO] Observation keys: {obs.keys()}")
    print(f"[INFO] Depth image shape: {obs['depth_image'].shape}")
    print(f"[INFO] Lidar data shape: {obs['lidar_data'].shape}")
    print(f"[INFO] Lidar values: {obs['lidar_data']}")
    
    # Test a few steps
    print("[INFO] Testing steps...")
    for i in range(5):
        action = 0  # forward
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Reward={reward:.2f}, Distance={info.get('distance_to_goal', 0):.2f}, "
              f"Lidar: Ground={obs['lidar_data'][0]:.1f}m, Horizontal={obs['lidar_data'][1]:.1f}m")
        
        if terminated or truncated:
            print(f"Episode ended: terminated={terminated}, truncated={truncated}")
            break
    
    # Test reset with new goal
    print("[INFO] Testing reset with new goal...")
    obs, info = env.reset()
    print(f"[SUCCESS] Second reset completed!")
    print(f"[INFO] New goal generated!")
    
    env.close()
    print("[SUCCESS] Environment test completed!")

if __name__ == "__main__":
    test_random_goals() 