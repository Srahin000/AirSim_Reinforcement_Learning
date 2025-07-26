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
        self.client.enableApiControl(True)
        
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
        self.velocity = 3.0  # m/s
        self.rotation_rate = 30.0  # degrees/s
        self.altitude_change_rate = 2.0  # m/s
        
        # Control loop
        self.running = True
        
    def setup_keyboard_controls(self):
        """Setup keyboard event handlers"""
        # Movement keys
        keyboard.on_press_key('w', lambda _: self.set_movement('forward', True))
        keyboard.on_release_key('w', lambda _: self.set_movement('forward', False))
        
        keyboard.on_press_key('s', lambda _: self.set_movement('backward', True))
        keyboard.on_release_key('s', lambda _: self.set_movement('backward', False))
        
        keyboard.on_press_key('a', lambda _: self.set_movement('left', True))
        keyboard.on_release_key('a', lambda _: self.set_movement('left', False))
        
        keyboard.on_press_key('d', lambda _: self.set_movement('right', True))
        keyboard.on_release_key('d', lambda _: self.set_movement('right', False))
        
        # Altitude control
        keyboard.on_press_key('space', lambda _: self.set_movement('up', True))
        keyboard.on_release_key('space', lambda _: self.set_movement('up', False))
        
        keyboard.on_press_key('shift', lambda _: self.set_movement('down', True))
        keyboard.on_release_key('shift', lambda _: self.set_movement('down', False))
        
        # Rotation control
        keyboard.on_press_key('q', lambda _: self.set_movement('rotate_left', True))
        keyboard.on_release_key('q', lambda _: self.set_movement('rotate_left', False))
        
        keyboard.on_press_key('e', lambda _: self.set_movement('rotate_right', True))
        keyboard.on_release_key('e', lambda _: self.set_movement('rotate_right', False))
        
        # Exit key
        keyboard.on_press_key('esc', lambda _: self.stop())
        
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
    
    def update_camera(self):
        """Set the camera to follow the drone's position and orientation."""
        state = self.client.getMultirotorState()
        position = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation
        camera_name = "0"  # default front camera
        vehicle_name = ""
        self.client.simSetCameraPose(camera_name, airsim.Pose(position, orientation), vehicle_name)
    
    def fetch_and_show_images(self):
        """Fetch images from the front camera (no OpenCV display, only for processing or logging if needed)."""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),        # RGB
                airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False), # Segmentation
                airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)      # Depth (float)
            ])
            # You can process responses here if needed, but do not display them
        except Exception as e:
            print(f"Image fetch error: {e}")
    
    def control_loop(self):
        """Main control loop for continuous movement (updated for yaw-relative movement and camera follow)."""
        print("Starting control loop...")
        print("Controls:")
        print("  W/S - Forward/Backward")
        print("  A/D - Left/Right")
        print("  Space/Shift - Up/Down")
        print("  Q/E - Rotate Left/Right")
        print("  ESC - Exit")
        print("\nPress any key to start...")
        while self.running:
            try:
                # Calculate movement
                vx, vy, vz = self.calculate_velocity()
                yaw_rate = self.calculate_yaw_rate()
                # Apply movement if any key is pressed
                if any([vx, vy, vz, yaw_rate]):
                    self.client.moveByVelocityAsync(vx, vy, vz, 0.1, 
                                                   drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                   yaw_mode=airsim.YawMode(True, yaw_rate))
                else:
                    # Hover if no movement keys are pressed
                    self.client.hoverAsync()
                # Update camera to follow drone
                self.update_camera()
                # Fetch and display images
                self.fetch_and_show_images()
                time.sleep(0.1)  # 10 Hz control loop
            except Exception as e:
                print(f"Error in control loop: {e}")
                break
    
    def start(self):
        """Start the drone controller"""
        print("Initializing drone controller...")
        
        # Take off
        print("Taking off...")
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        print("Drone is airborne!")
        
        # Setup keyboard controls
        self.setup_keyboard_controls()
        
        # Start control loop in a separate thread
        control_thread = threading.Thread(target=self.control_loop)
        control_thread.start()
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
        
        # Cleanup
        self.cleanup()
    
    def stop(self):
        """Stop the controller"""
        print("\nStopping drone controller...")
        self.running = False
    
    def cleanup(self):
        """Cleanup and land the drone"""
        print("Landing drone...")
        try:
            self.client.landAsync().join()
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
            print("Drone landed safely!")
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    # Check if keyboard module is available
    try:
        import keyboard
    except ImportError:
        print("Error: 'keyboard' module not found.")
        print("Install it using: pip install keyboard")
        sys.exit(1)
    
    controller = DroneController()
    controller.start() 