#!/usr/bin/env python3
"""
Simple Terminal-Controlled Drone Control for AirSim
==================================================

A basic terminal interface for controlling a drone in AirSim.
Make sure AirSim is running before starting this script.

Controls:
- W/S: Forward/Backward
- A/D: Left/Right  
- Q/E: Rotate Left/Right
- Space/Shift: Up/Down
- H: Hover
- T: Takeoff
- L: Land
- X: Exit
"""

import sys
import time
import math
import threading
from typing import Optional

# Add AirSim Python client to path
try:
    import airsim
except ImportError:
    print("Error: AirSim Python client not found!")
    print("Please install it from: https://github.com/microsoft/AirSim/tree/master/PythonClient")
    sys.exit(1)


class TerminalDroneController:
    def __init__(self):
        """Initialize the drone controller"""
        self.client = None
        self.running = False
        self.vehicle_name = "SimpleFlight"
        
        # Movement parameters
        self.velocity = 3.0  # m/s
        self.rotation_rate = 30.0  # degrees/s
        self.altitude_change_rate = 2.0  # m/s
        
        # Current movement state
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_vz = 0.0
        self.current_yaw_rate = 0.0
        
        # Status display
        self.status_thread = None
        self.status_running = False
    
    def connect(self):
        """Connect to AirSim"""
        print("Connecting to AirSim...")
        try:
            self.client = airsim.MultirotorClient()
            self.client.confirmConnection()
            print("✓ Connected to AirSim successfully!")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to AirSim: {e}")
            print("Make sure AirSim is running and the drone is spawned.")
            return False
    
    def enable_api_control(self):
        """Enable API control for the drone"""
        try:
            self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
            print("✓ API control enabled")
            return True
        except Exception as e:
            print(f"✗ Failed to enable API control: {e}")
            return False
    
    def get_drone_state(self):
        """Get current drone state"""
        try:
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            return state
        except Exception as e:
            print(f"Error getting drone state: {e}")
            return None
    
    def get_drone_position(self):
        """Get current drone position"""
        state = self.get_drone_state()
        if state:
            pos = state.kinematics_estimated.position
            return pos.x_val, pos.y_val, pos.z_val
        return None, None, None
    
    def get_drone_yaw(self):
        """Get current yaw in radians"""
        state = self.get_drone_state()
        if state:
            orientation = state.kinematics_estimated.orientation
            w, x, y, z = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return yaw
        return 0.0
    
    def move_drone(self, vx, vy, vz, yaw_rate=0):
        """Move drone with velocity and yaw rate"""
        try:
            self.client.moveByVelocityAsync(
                vx, vy, vz, 0.1,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(True, yaw_rate),
                vehicle_name=self.vehicle_name
            )
            self.current_vx, self.current_vy, self.current_vz = vx, vy, vz
            self.current_yaw_rate = yaw_rate
        except Exception as e:
            print(f"Move error: {e}")
    
    def hover(self):
        """Make drone hover"""
        try:
            self.client.hoverAsync(vehicle_name=self.vehicle_name)
            self.current_vx = self.current_vy = self.current_vz = self.current_yaw_rate = 0.0
            print("Hovering...")
        except Exception as e:
            print(f"Hover error: {e}")
    
    def takeoff(self):
        """Take off the drone"""
        print("Taking off...")
        try:
            self.client.armDisarm(True, vehicle_name=self.vehicle_name)
            self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()
            print("✓ Drone is airborne!")
        except Exception as e:
            print(f"✗ Takeoff failed: {e}")
    
    def land(self):
        """Land the drone"""
        print("Landing...")
        try:
            self.client.landAsync(vehicle_name=self.vehicle_name).join()
            self.client.armDisarm(False, vehicle_name=self.vehicle_name)
            print("✓ Drone landed safely!")
        except Exception as e:
            print(f"✗ Landing failed: {e}")
    
    def display_status(self):
        """Display current drone status"""
        while self.status_running:
            try:
                x, y, z = self.get_drone_position()
                yaw = self.get_drone_yaw()
                
                if x is not None:
                    print(f"\rPosition: ({x:.1f}, {y:.1f}, {z:.1f}) | "
                          f"Yaw: {math.degrees(yaw):.1f}° | "
                          f"Velocity: ({self.current_vx:.1f}, {self.current_vy:.1f}, {self.current_vz:.1f}) | "
                          f"Yaw Rate: {self.current_yaw_rate:.1f}°/s", end="", flush=True)
                
                time.sleep(0.5)
            except Exception as e:
                print(f"\nStatus display error: {e}")
                break
    
    def start_status_display(self):
        """Start the status display thread"""
        self.status_running = True
        self.status_thread = threading.Thread(target=self.display_status, daemon=True)
        self.status_thread.start()
    
    def stop_status_display(self):
        """Stop the status display thread"""
        self.status_running = False
        if self.status_thread:
            self.status_thread.join(timeout=1.0)
    
    def print_controls(self):
        """Print the control instructions"""
        print("\n" + "="*50)
        print("TERMINAL DRONE CONTROL")
        print("="*50)
        print("Controls:")
        print("  W/S     - Forward/Backward")
        print("  A/D     - Left/Right")
        print("  Q/E     - Rotate Left/Right")
        print("  Space   - Up")
        print("  Shift   - Down")
        print("  H       - Hover")
        print("  T       - Takeoff")
        print("  L       - Land")
        print("  X       - Exit")
        print("="*50)
    
    def handle_input(self, key):
        """Handle keyboard input"""
        key = key.lower()
        yaw = self.get_drone_yaw()
        
        # Initialize movement
        vx, vy, vz = 0, 0, 0
        yaw_rate = 0
        
        if key == 'w':  # Forward
            vx = self.velocity * math.cos(yaw)
            vy = self.velocity * math.sin(yaw)
            print(" → Forward")
        elif key == 's':  # Backward
            vx = -self.velocity * math.cos(yaw)
            vy = -self.velocity * math.sin(yaw)
            print(" ← Backward")
        elif key == 'a':  # Left
            vx = -self.velocity * math.sin(yaw)
            vy = self.velocity * math.cos(yaw)
            print(" ← Left")
        elif key == 'd':  # Right
            vx = self.velocity * math.sin(yaw)
            vy = -self.velocity * math.cos(yaw)
            print(" → Right")
        elif key == 'q':  # Rotate left
            yaw_rate = self.rotation_rate
            print(" ↶ Rotate Left")
        elif key == 'e':  # Rotate right
            yaw_rate = -self.rotation_rate
            print(" ↷ Rotate Right")
        elif key == ' ':  # Space - Up
            vz = -self.altitude_change_rate
            print(" ↑ Up")
        elif key == 'shift':  # Shift - Down
            vz = self.altitude_change_rate
            print(" ↓ Down")
        elif key == 'h':  # Hover
            self.hover()
            return
        elif key == 't':  # Takeoff
            self.takeoff()
            return
        elif key == 'l':  # Land
            self.land()
            return
        elif key == 'x':  # Exit
            print("\nExiting...")
            self.running = False
            return
        else:
            print(f"Unknown key: {key}")
            return
        
        # Apply movement
        if any([vx, vy, vz, yaw_rate]):
            self.move_drone(vx, vy, vz, yaw_rate)
    
    def run_control_loop(self):
        """Main control loop"""
        self.print_controls()
        self.start_status_display()
        
        print("\nPress keys to control the drone (X to exit):")
        
        while self.running:
            try:
                # For Windows
                try:
                    import msvcrt
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8')
                        self.handle_input(key)
                except ImportError:
                    # For non-Windows systems
                    key = input("\nEnter command: ").strip()
                    if key:
                        self.handle_input(key)
                
                time.sleep(0.1)
                
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received")
                self.running = False
                break
            except Exception as e:
                print(f"\nError in control loop: {e}")
                break
    
    def start(self):
        """Start the drone controller"""
        if not self.connect():
            return False
        
        if not self.enable_api_control():
            return False
        
        self.running = True
        
        try:
            self.run_control_loop()
        except Exception as e:
            print(f"Error during flight: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        self.stop_status_display()
        
        if self.client:
            try:
                self.client.hoverAsync(vehicle_name=self.vehicle_name)
                time.sleep(1)
                self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
                print("✓ Controller shutdown complete")
            except Exception as e:
                print(f"Cleanup error: {e}")


def main():
    """Main function"""
    print("Terminal Drone Control for AirSim")
    print("Make sure AirSim is running before starting...")
    
    try:
        controller = TerminalDroneController()
        controller.start()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 