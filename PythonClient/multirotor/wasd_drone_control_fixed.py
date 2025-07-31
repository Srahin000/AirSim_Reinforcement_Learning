#!/usr/bin/env python3
"""
Fixed WASD Drone Control Script
Addresses issues with drone being stuck at starting position.
"""

import setup_path
import airsim
import time
import threading
import keyboard
import sys
import cv2
import numpy as np

class DroneController:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        # Ensure API control is properly enabled
        print("[INFO] Requesting API control...")
        self.client.enableApiControl(True)
        time.sleep(1.0)  # Wait for API control to be granted
        
        # Verify API control
        try:
            self.client.armDisarm(False)  # Test API control
            print("[SUCCESS] API control granted!")
        except Exception as e:
            print(f"[ERROR] API control failed: {e}")
            print("[INFO] Please check AirSim settings and restart if needed.")
        
        # Auto takeoff
        self.auto_takeoff()
        
        # Movement state
        self.moving_forward = False
        self.moving_backward = False
        self.moving_left = False
        self.moving_right = False
        self.moving_up = False
        self.moving_down = False
        self.rotating_left = False
        self.rotating_right = False
        
        # Movement parameters
        self.velocity = 5.0  # m/s - increased for more responsive movement
        self.rotation_rate = 45.0  # degrees/s - increased for faster rotation
        self.altitude_change_rate = 3.0  # m/s - increased for faster altitude changes
        
        # Control loop
        self.running = True
        
        # Track if drone is airborne
        self.is_airborne = False
        
    def setup_keyboard_controls(self):
        """Setup keyboard event handlers"""
        # Only setup takeoff/landing/test keys
        keyboard.on_press_key('t', lambda _: self.takeoff())
        keyboard.on_press_key('l', lambda _: self.land())
        keyboard.on_press_key('p', lambda _: self.test_drone())
        
    def set_movement(self, direction, active):
        """Set movement state for a direction"""
        if direction == 'forward':
            self.moving_forward = active
        elif direction == 'backward':
            self.moving_backward = active
        elif direction == 'left':
            self.moving_left = active
        elif direction == 'right':
            self.moving_right = active
        elif direction == 'up':
            self.moving_up = active
        elif direction == 'down':
            self.moving_down = active
        elif direction == 'rotate_left':
            self.rotating_left = active
        elif direction == 'rotate_right':
            self.rotating_right = active
    
    def get_drone_yaw(self):
        """Get the current yaw (in radians) of the drone."""
        state = self.client.getMultirotorState()
        orientation = state.kinematics_estimated.orientation
        # Convert quaternion to yaw
        import math
        w, x, y, z = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
        # Yaw calculation from quaternion
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw
    
    def calculate_velocity(self):
        """Calculate velocity vector based on current movement state, relative to drone's yaw."""
        vx, vy, vz = 0, 0, 0
        
        # Calculate base velocities
        if self.moving_forward:
            vx += self.velocity
        if self.moving_backward:
            vx -= self.velocity
        if self.moving_left:
            vy += self.velocity
        if self.moving_right:
            vy -= self.velocity
        if self.moving_up:
            vz -= self.altitude_change_rate
        if self.moving_down:
            vz += self.altitude_change_rate
            
        # Rotate vx, vy by current yaw
        if vx != 0 or vy != 0:
            import math
            yaw = self.get_drone_yaw()
            # Forward in AirSim is X, left is Y
            # Rotate (vx, vy) by yaw
            vx_new = vx * math.cos(yaw) - vy * math.sin(yaw)
            vy_new = vx * math.sin(yaw) + vy * math.cos(yaw)
            vx, vy = vx_new, vy_new
            
        return vx, vy, vz
    
    def calculate_yaw_rate(self):
        """Calculate yaw rate based on rotation state"""
        yaw_rate = 0
        if self.rotating_left:
            yaw_rate += self.rotation_rate
        if self.rotating_right:
            yaw_rate -= self.rotation_rate
        return yaw_rate
    
    def auto_takeoff(self):
        """Automatically take off the drone"""
        print("[INFO] Auto takeoff starting...")
        try:
            # Arm the drone
            self.client.armDisarm(True)
            time.sleep(0.5)
            
            # Take off
            print("[INFO] Taking off...")
            self.client.takeoffAsync().join()
            self.is_airborne = True
            print("[SUCCESS] Drone is airborne and ready for control!")
            
        except Exception as e:
            print(f"[ERROR] Auto takeoff failed: {e}")
            print("[INFO] Trying alternative takeoff method...")
            try:
                self.client.moveToZAsync(-5, 2).join()
                self.is_airborne = True
                print("[SUCCESS] Alternative takeoff completed!")
            except Exception as alt_error:
                print(f"[ERROR] Alternative takeoff failed: {alt_error}")
                self.is_airborne = False
    
    def takeoff(self):
        """Manual takeoff (for manual control)"""
        if not self.is_airborne:
            self.auto_takeoff()
        else:
            print("[INFO] Drone is already airborne!")
    
    def land(self):
        """Land the drone"""
        if self.is_airborne:
            print("[INFO] Landing...")
            try:
                self.client.landAsync().join()
                self.is_airborne = False
                print("[SUCCESS] Drone landed!")
            except Exception as e:
                print(f"[ERROR] Landing failed: {e}")
        else:
            print("[INFO] Drone is already on ground!")
    
    def get_drone_position(self):
        """Get current drone position"""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return pos.x_val, pos.y_val, pos.z_val
    
    def test_drone(self):
        """Test drone functionality"""
        print("\n[TEST] Testing drone functionality...")
        try:
            # Get current state
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            print(f"[TEST] Current position: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
            print(f"[TEST] Drone ready: {state.ready}")
            print(f"[TEST] Drone armed: {state.armed}")
            print(f"[TEST] Drone airborne: {self.is_airborne}")
            
            # Test simple movement
            if self.is_airborne:
                print("[TEST] Testing hover...")
                self.client.hoverAsync()
                time.sleep(1.0)
                print("[TEST] Hover test completed!")
            else:
                print("[TEST] Drone not airborne, cannot test movement")
                
        except Exception as e:
            print(f"[TEST] Error during test: {e}")
    
    def control_loop(self):
        """Main control loop for continuous movement"""
        print("Starting control loop...")
        print("Controls:")
        print("  L - Land")
        print("  P - Test drone")
        print("  W/S - Forward/Backward")
        print("  A/D - Left/Right")
        print("  Space/Shift - Up/Down")
        print("  Q/E - Rotate Left/Right")
        print("  ESC - Exit")
        print("\nDrone is ready! Use WASD to control...")
        
        while self.running:
            try:
                # Simple keyboard polling
                if keyboard.is_pressed('w'):
                    if self.is_airborne:
                        print("[MOVE] Forward")
                        self.client.moveByVelocityAsync(3, 0, 0, 1.0)
                    else:
                        print("[INFO] Drone not airborne. Auto takeoff failed.")
                
                elif keyboard.is_pressed('s'):
                    if self.is_airborne:
                        print("[MOVE] Backward")
                        self.client.moveByVelocityAsync(-3, 0, 0, 1.0)
                    else:
                        print("[INFO] Drone not airborne. Auto takeoff failed.")
                
                elif keyboard.is_pressed('a'):
                    if self.is_airborne:
                        print("[MOVE] Left")
                        self.client.moveByVelocityAsync(0, 3, 0, 1.0)
                    else:
                        print("[INFO] Drone not airborne. Auto takeoff failed.")
                
                elif keyboard.is_pressed('d'):
                    if self.is_airborne:
                        print("[MOVE] Right")
                        self.client.moveByVelocityAsync(0, -3, 0, 1.0)
                    else:
                        print("[INFO] Drone not airborne. Auto takeoff failed.")
                
                elif keyboard.is_pressed('space'):
                    if self.is_airborne:
                        print("[MOVE] Up")
                        self.client.moveByVelocityAsync(0, 0, -3, 1.0)
                    else:
                        print("[INFO] Drone not airborne. Auto takeoff failed.")
                
                elif keyboard.is_pressed('shift'):
                    if self.is_airborne:
                        print("[MOVE] Down")
                        self.client.moveByVelocityAsync(0, 0, 3, 1.0)
                    else:
                        print("[INFO] Drone not airborne. Auto takeoff failed.")
                
                elif keyboard.is_pressed('q'):
                    if self.is_airborne:
                        print("[MOVE] Rotate Left")
                        self.client.rotateToYawAsync(45, 30, 1)
                    else:
                        print("[INFO] Drone not airborne. Auto takeoff failed.")
                
                elif keyboard.is_pressed('e'):
                    if self.is_airborne:
                        print("[MOVE] Rotate Right")
                        self.client.rotateToYawAsync(-45, 30, 1)
                    else:
                        print("[INFO] Drone not airborne. Auto takeoff failed.")
                
                elif keyboard.is_pressed('esc'):
                    print("[EXIT] Shutting down...")
                    break
                
                else:
                    # Hover if no key is pressed
                    if self.is_airborne:
                        self.client.hoverAsync()
                
                time.sleep(0.05)  # 20 Hz control loop for better responsiveness
                
            except Exception as e:
                print(f"Error in control loop: {e}")
                time.sleep(0.1)
    
    def start(self):
        """Start the drone controller"""
        print("Initializing drone controller...")
        
        # Setup keyboard controls
        self.setup_keyboard_controls()
        
        # Start control loop in main thread to avoid IOLoop issues
        print("Starting control loop in main thread...")
        self.control_loop()
        
        print("Drone controller started!")
        print("Drone should be airborne and ready for control!")
        
        # Control loop is now running in main thread
        pass
    
    def stop(self):
        """Stop the drone controller"""
        print("Stopping drone controller...")
        self.running = False
        
        # Land if airborne
        if self.is_airborne:
            try:
                self.client.landAsync().join()
                print("Drone landed safely.")
            except:
                pass
        
        # Disarm
        try:
            self.client.armDisarm(False)
        except:
            pass
        
        print("Drone controller stopped.")
    
    def cleanup(self):
        """Cleanup resources"""
        keyboard.unhook_all()

def main():
    """Main function"""
    controller = DroneController()
    
    try:
        controller.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        controller.stop()
        controller.cleanup()

if __name__ == "__main__":
    main() 