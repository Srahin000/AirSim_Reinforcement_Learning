#!/usr/bin/env python3
"""
Simple script to check drone position in AirSim.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import setup_path
import airsim

def check_drone_position():
    """Check and print the drone's current position."""
    try:
        print("[INFO] Connecting to AirSim...")
        
        # Connect to AirSim
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("[SUCCESS] Connected to AirSim!")
        
        # Get drone state and position
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        
        print(f"[INFO] Drone current position: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
        
        # Get additional information
        orientation = state.kinematics_estimated.orientation
        yaw = airsim.to_eularian_angles(orientation)[2] * 180 / 3.14159
        
        print(f"[INFO] Drone orientation - Yaw: {yaw:.2f} degrees")
        
        # Check if drone is armed
        is_armed = state.ready
        print(f"[INFO] Drone armed: {is_armed}")
        
        # Get altitude (positive value)
        altitude = -pos.z_val
        print(f"[INFO] Drone altitude: {altitude:.2f} meters")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to get drone position: {e}")
        return False

def main():
    """Main function."""
    print("=" * 50)
    print("DRONE POSITION CHECKER")
    print("=" * 50)
    
    success = check_drone_position()
    
    if success:
        print("\n[SUCCESS] Position check completed!")
    else:
        print("\n[ERROR] Position check failed!")
        print("[INFO] Make sure AirSim is running and the drone is spawned.")

if __name__ == "__main__":
    main() 