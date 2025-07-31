#!/usr/bin/env python3
"""
Simple connection test for AirSim to verify the drone can be controlled.
"""

import setup_path
import airsim
import time

def test_connection():
    """Test basic AirSim connection and drone control."""
    try:
        print("[INFO] Attempting to connect to AirSim...")
        
        # Connect to AirSim
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("[SUCCESS] Connected to AirSim!")
        
        # Enable API control
        client.enableApiControl(True)
        print("[SUCCESS] API control enabled!")
        
        # Arm the drone
        client.armDisarm(True)
        print("[SUCCESS] Drone armed!")
        
        # Get initial state
        state = client.getMultirotorState()
        print(f"[INFO] Initial position: ({state.kinematics_estimated.position.x_val}, "
              f"{state.kinematics_estimated.position.y_val}, "
              f"{state.kinematics_estimated.position.z_val})")
        
        # Test takeoff
        print("[INFO] Testing takeoff...")
        client.takeoffAsync().join()
        print("[SUCCESS] Takeoff completed!")
        
        # Test hover
        print("[INFO] Testing hover...")
        client.hoverAsync().join()
        print("[SUCCESS] Hover test completed!")
        
        # Test movement
        print("[INFO] Testing movement...")
        client.moveToPositionAsync(5, 5, -10, 2).join()
        print("[SUCCESS] Movement test completed!")
        
        # Land
        print("[INFO] Landing...")
        client.landAsync().join()
        print("[SUCCESS] Landing completed!")
        
        # Disarm
        client.armDisarm(False)
        print("[SUCCESS] Drone disarmed!")
        
        print("[SUCCESS] All connection tests passed! AirSim is working correctly.")
        return True
        
    except Exception as e:
        print(f"[ERROR] Connection test failed: {e}")
        return False

if __name__ == "__main__":
    test_connection() 