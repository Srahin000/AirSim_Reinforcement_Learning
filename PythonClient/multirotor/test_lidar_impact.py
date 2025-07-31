#!/usr/bin/env python3
"""
Test script to check if lidar is causing keyboard control issues.
"""

import setup_path
import airsim
import time
import keyboard

def test_without_lidar():
    """Test drone control without lidar"""
    print("=== TESTING WITHOUT LIDAR ===")
    
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    
    print("Taking off...")
    client.armDisarm(True)
    client.takeoffAsync().join()
    
    print("Testing keyboard control (no lidar)...")
    print("Press W/A/S/D to move, ESC to exit")
    
    start_time = time.time()
    while time.time() - start_time < 10:  # Test for 10 seconds
        if keyboard.is_pressed('w'):
            print("W pressed - moving forward")
            client.moveByVelocityAsync(3, 0, 0, 1.0)
        elif keyboard.is_pressed('s'):
            print("S pressed - moving backward")
            client.moveByVelocityAsync(-3, 0, 0, 1.0)
        elif keyboard.is_pressed('a'):
            print("A pressed - moving left")
            client.moveByVelocityAsync(0, 3, 0, 1.0)
        elif keyboard.is_pressed('d'):
            print("D pressed - moving right")
            client.moveByVelocityAsync(0, -3, 0, 1.0)
        elif keyboard.is_pressed('esc'):
            break
        else:
            client.hoverAsync()
        
        time.sleep(0.05)
    
    client.landAsync().join()
    print("Test completed!")

def test_with_lidar():
    """Test drone control with lidar"""
    print("=== TESTING WITH LIDAR ===")
    
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    
    print("Taking off...")
    client.armDisarm(True)
    client.takeoffAsync().join()
    
    print("Testing keyboard control (with lidar)...")
    print("Press W/A/S/D to move, ESC to exit")
    
    start_time = time.time()
    while time.time() - start_time < 10:  # Test for 10 seconds
        # Get lidar data (this might cause issues)
        try:
            lidar_data = client.getLidarData()
            if lidar_data.point_cloud:
                print(f"Lidar points: {len(lidar_data.point_cloud)}")
        except Exception as e:
            print(f"Lidar error: {e}")
        
        # Test keyboard control
        if keyboard.is_pressed('w'):
            print("W pressed - moving forward")
            client.moveByVelocityAsync(3, 0, 0, 1.0)
        elif keyboard.is_pressed('s'):
            print("S pressed - moving backward")
            client.moveByVelocityAsync(-3, 0, 0, 1.0)
        elif keyboard.is_pressed('a'):
            print("A pressed - moving left")
            client.moveByVelocityAsync(0, 3, 0, 1.0)
        elif keyboard.is_pressed('d'):
            print("D pressed - moving right")
            client.moveByVelocityAsync(0, -3, 0, 1.0)
        elif keyboard.is_pressed('esc'):
            break
        else:
            client.hoverAsync()
        
        time.sleep(0.05)
    
    client.landAsync().join()
    print("Test completed!")

def main():
    """Run both tests"""
    print("Testing drone control with and without lidar...")
    print("This will help identify if lidar is causing issues.")
    
    choice = input("Test without lidar first? (y/n): ")
    if choice.lower() == 'y':
        test_without_lidar()
        input("Press Enter to test with lidar...")
        test_with_lidar()
    else:
        test_with_lidar()
        input("Press Enter to test without lidar...")
        test_without_lidar()

if __name__ == "__main__":
    main() 