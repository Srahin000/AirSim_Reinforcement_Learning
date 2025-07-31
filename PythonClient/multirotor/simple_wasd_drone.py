import setup_path
import airsim
import time
import sys
import numpy as np

class SimpleDroneController:
    def __init__(self):
        print("Initializing Simple Drone Controller...")
        self.client = airsim.MultirotorClient()
        print("Connecting to AirSim...")
        self.client.confirmConnection()
        print("Connected successfully!")
        
        # Enable API control
        self.client.enableApiControl(True, vehicle_name="SimpleFlight")
        print("API control enabled")
        
        # Movement parameters
        self.velocity = 3.0  # m/s
        self.rotation_rate = 30.0  # degrees/s
        self.altitude_change_rate = 2.0  # m/s
        
        # Control state
        self.running = True
        
        # Lidar settings
        self.lidar_enabled = True
        self.lidar_update_interval = 1.0  # seconds
        self.last_lidar_update = 0
        
    def get_lidar_data(self, lidar_name="Lidar1"):
        """Get lidar data and return minimum distance"""
        try:
            lidar_data = self.client.getLidarData(lidar_name=lidar_name, vehicle_name="SimpleFlight")
            if len(lidar_data.point_cloud) < 3:
                return None
            
            # Parse point cloud data
            points = np.array(lidar_data.point_cloud, dtype=np.float32)
            points = points.reshape(int(points.shape[0]/3), 3)
            
            # Calculate distances from origin (0,0,0)
            distances = np.sqrt(np.sum(points**2, axis=1))
            min_distance = np.min(distances)
            
            return min_distance
        except Exception as e:
            print(f"Lidar error ({lidar_name}): {e}")
            return None
    
    def get_all_lidar_data(self):
        """Get data from both lidars"""
        lidar1_dist = self.get_lidar_data("Lidar1")
        lidar2_dist = self.get_lidar_data("Lidar2")
        return lidar1_dist, lidar2_dist
    
    def display_lidar_info(self):
        """Display current lidar readings"""
        if not self.lidar_enabled:
            return
            
        current_time = time.time()
        if current_time - self.last_lidar_update >= self.lidar_update_interval:
            lidar1_dist, lidar2_dist = self.get_all_lidar_data()
            
            print("\n--- Lidar Readings ---")
            if lidar1_dist is not None:
                print(f"Lidar1 (Horizontal): {lidar1_dist:.2f} m")
            else:
                print("Lidar1: No data")
                
            if lidar2_dist is not None:
                print(f"Lidar2 (Ground): {lidar2_dist:.2f} m")
            else:
                print("Lidar2: No data")
            print("---------------------")
            
            self.last_lidar_update = current_time
    
    def get_drone_yaw(self):
        """Get current yaw in radians"""
        state = self.client.getMultirotorState(vehicle_name="SimpleFlight")
        orientation = state.kinematics_estimated.orientation
        import math
        w, x, y, z = orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return yaw
    
    def move_drone(self, vx, vy, vz, yaw_rate=0):
        """Move drone with velocity and yaw rate"""
        try:
            self.client.moveByVelocityAsync(
                vx, vy, vz, 0.1,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(True, yaw_rate),
                vehicle_name="SimpleFlight"
            )
        except Exception as e:
            print(f"Move error: {e}")
    
    def hover(self):
        """Make drone hover"""
        try:
            self.client.hoverAsync(vehicle_name="SimpleFlight")
        except Exception as e:
            print(f"Hover error: {e}")
    
    def takeoff(self):
        """Take off the drone"""
        print("Taking off...")
        self.client.armDisarm(True, vehicle_name="SimpleFlight")
        self.client.takeoffAsync(vehicle_name="SimpleFlight").join()
        print("Drone is airborne!")
    
    def land(self):
        """Land the drone"""
        print("Landing...")
        self.client.landAsync(vehicle_name="SimpleFlight").join()
        self.client.armDisarm(False, vehicle_name="SimpleFlight")
        print("Drone landed safely!")
    
    def run_control_loop(self):
        """Main control loop using simple input polling"""
        print("\n=== Simple WASD Drone Control ===")
        print("Controls:")
        print("  W/S - Forward/Backward")
        print("  A/D - Left/Right")
        print("  Q/E - Rotate Left/Right")
        print("  Space/Shift - Up/Down")
        print("  H - Hover")
        print("  X - Exit")
        print("================================")
        
        while self.running:
            try:
                # Get current yaw for movement calculations
                yaw = self.get_drone_yaw()
                import math
                
                # Initialize movement
                vx, vy, vz = 0, 0, 0
                yaw_rate = 0
                
                # For Windows, we'll use a simple input approach
                try:
                    import msvcrt
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8').lower()
                        print(f"\nKey pressed: {key}")
                        
                        if key == 'w':  # Forward
                            vx = self.velocity * math.cos(yaw)
                            vy = self.velocity * math.sin(yaw)
                        elif key == 's':  # Backward
                            vx = -self.velocity * math.cos(yaw)
                            vy = -self.velocity * math.sin(yaw)
                        elif key == 'a':  # Left
                            vx = -self.velocity * math.sin(yaw)
                            vy = self.velocity * math.cos(yaw)
                        elif key == 'd':  # Right
                            vx = self.velocity * math.sin(yaw)
                            vy = -self.velocity * math.cos(yaw)
                        elif key == 'q':  # Rotate left
                            yaw_rate = self.rotation_rate
                        elif key == 'e':  # Rotate right
                            yaw_rate = -self.rotation_rate
                        elif key == ' ':  # Space - Up
                            vz = -self.altitude_change_rate
                        elif key == 'h':  # Hover
                            self.hover()
                            continue
                        elif key == 'x':  # Exit
                            print("Exiting...")
                            self.running = False
                            break
                        
                        # Apply movement
                        if any([vx, vy, vz, yaw_rate]):
                            self.move_drone(vx, vy, vz, yaw_rate)
                            print(f"Moving: vx={vx:.2f}, vy={vy:.2f}, vz={vz:.2f}, yaw_rate={yaw_rate:.2f}")
                        else:
                            self.hover()
                    
                    time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                    
                except ImportError:
                    # Fallback for non-Windows systems
                    print("msvcrt not available, using input() method")
                    key = input().lower()
                    # Process key as above...
                    
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received")
                self.running = False
                break
            except Exception as e:
                print(f"Error in control loop: {e}")
                break
    
    def start(self):
        """Start the drone controller"""
        try:
            self.takeoff()
            self.run_control_loop()
        except Exception as e:
            print(f"Error during flight: {e}")
        finally:
            self.land()
            self.client.enableApiControl(False, vehicle_name="SimpleFlight")
            print("Controller shutdown complete")

def main():
    print("Simple WASD Drone Control")
    print("Make sure AirSim is running before starting...")
    
    try:
        controller = SimpleDroneController()
        controller.start()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 