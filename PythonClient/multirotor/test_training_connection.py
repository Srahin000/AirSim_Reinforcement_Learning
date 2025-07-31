#!/usr/bin/env python3
"""
Test script to verify training environment connection and basic functionality.
"""

import setup_path
import airsim
import time
import numpy as np

def test_basic_connection():
    """Test basic AirSim connection."""
    try:
        print("[INFO] Testing basic AirSim connection...")
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("[SUCCESS] Basic connection established!")
        return True
    except Exception as e:
        print(f"[ERROR] Basic connection failed: {e}")
        return False

def test_vehicle_control():
    """Test vehicle control and movement."""
    try:
        print("[INFO] Testing vehicle control...")
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        print("[SUCCESS] Vehicle control enabled!")
        
        # Test takeoff
        print("[INFO] Testing takeoff...")
        client.takeoffAsync().join()
        print("[SUCCESS] Takeoff completed!")
        
        # Test hover
        client.hoverAsync().join()
        print("[SUCCESS] Hover test completed!")
        
        # Test movement
        print("[INFO] Testing movement...")
        client.moveToPositionAsync(5, 5, -10, 2).join()
        print("[SUCCESS] Movement test completed!")
        
        # Land and disarm
        client.landAsync().join()
        client.armDisarm(False)
        print("[SUCCESS] Vehicle control test completed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Vehicle control test failed: {e}")
        return False

def test_lidar_sensors():
    """Test lidar sensor functionality."""
    try:
        print("[INFO] Testing lidar sensors...")
        client = airsim.MultirotorClient()
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        
        # Test lidar data retrieval
        lidar1_data = client.getLidarData(lidar_name="Lidar1")
        lidar2_data = client.getLidarData(lidar_name="Lidar2")
        
        print(f"[INFO] Lidar1 points: {len(lidar1_data.point_cloud)}")
        print(f"[INFO] Lidar2 points: {len(lidar2_data.point_cloud)}")
        print("[SUCCESS] Lidar sensors working!")
        
        client.armDisarm(False)
        return True
        
    except Exception as e:
        print(f"[ERROR] Lidar test failed: {e}")
        return False

def test_environment_import():
    """Test if the training environment can be imported."""
    try:
        print("[INFO] Testing environment import...")
        from mountain_env_lidar import MountainPassLidarEnv
        print("[SUCCESS] MountainPassLidarEnv imported successfully!")
        
        # Test environment creation
        print("[INFO] Testing environment creation...")
        env = MountainPassLidarEnv(
            lidar_check_frequency=5,
            lidar_safety_distance=2.0,
            ground_safety_distance=2.0,
            max_altitude=30.0,
            min_altitude=1.0
        )
        print("[SUCCESS] Environment created successfully!")
        
        # Test reset
        print("[INFO] Testing environment reset...")
        obs, info = env.reset()
        print(f"[SUCCESS] Environment reset completed! Observation shape: {obs.shape}")
        
        # Test step
        print("[INFO] Testing environment step...")
        action = 0  # forward
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"[SUCCESS] Environment step completed! Reward: {reward}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"[ERROR] Environment test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("AIRSIM TRAINING ENVIRONMENT TEST")
    print("=" * 50)
    
    tests = [
        ("Basic Connection", test_basic_connection),
        ("Vehicle Control", test_vehicle_control),
        ("Lidar Sensors", test_lidar_sensors),
        ("Environment Import", test_environment_import)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        print("-" * 30)
        if test_func():
            passed += 1
            print(f"[PASS] {test_name}")
        else:
            print(f"[FAIL] {test_name}")
    
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("[SUCCESS] All tests passed! Your training environment is ready.")
        print("[INFO] You can now run your training scripts:")
        print("  - train.py")
        print("  - train_lidar.py")
        print("  - train_improved_lidar.py")
    else:
        print("[WARNING] Some tests failed. Please check the errors above.")
        print("[INFO] Make sure AirSim is running and the settings.json is correct.")

if __name__ == "__main__":
    main() 