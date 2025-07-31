#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import setup_path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import cv2
import time
import math
from typing import Tuple, Optional, Dict, Any

class MountainPassEnv(gym.Env):

    def __init__(self, 
                 vehicle_name: str = "SimpleFlight",
                 max_steps: int = 200,
                 step_length: float = 4.0,
                 altitude_step: float = 2.0,
                 lidar_safety_distance: float = 2.0,
                 ground_safety_distance: float = 1.5,
                 max_altitude: float = 30.0,
                 min_altitude: float = 1.0,
                 hard_reset_on_collision: bool = True):
        super().__init__()
        
        # AirSim client setup
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name=vehicle_name)
        self.client.armDisarm(True, vehicle_name=vehicle_name)
        
        # Environment parameters
        self.vehicle_name = vehicle_name
        self.max_steps = max_steps
        self.step_length = step_length
        self.altitude_step = altitude_step
        self.lidar_safety_distance = lidar_safety_distance
        self.ground_safety_distance = ground_safety_distance
        self.max_altitude = max_altitude
        self.min_altitude = min_altitude
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(5)  # forward, left, right, up, down
        
        # Observation space: depth image (48x48x1) + lidar data (2 values)
        # Total observation: depth image + ground distance + horizontal distance
        self.observation_space = spaces.Dict({
            'depth_image': spaces.Box(low=0, high=255, shape=(48, 48, 1), dtype=np.uint8),
            'lidar_data': spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)  # ground_dist, horizontal_dist
        })
        
        # Episode tracking
        self.current_step = 0
        self.episode_reward = 0
        self.prev_dist = None
        self.collision_count = 0  # Track collision count
        self.hard_reset_on_collision = hard_reset_on_collision
        self.prev_yaw = None  # Track previous yaw for rotation penalty
        self.prev_altitude = None  # Track previous altitude for altitude change penalty
        
        # Goal and start positions
        self.goal_pos = airsim.Vector3r(45.78, 114.50, -19.35)  # Close goal for easier learning
        self.start_pos = airsim.Vector3r(0, 0, -5)  # Start higher to avoid ground collision
        
        # Lidar data tracking
        self.last_lidar_ground_dist = None
        self.last_lidar_horizontal_dist = None
        
        print(f"[INFO] Mountain Pass Environment initialized")
        print(f"[INFO] Goal position: ({self.goal_pos.x_val}, {self.goal_pos.y_val}, {self.goal_pos.z_val})")
    
    def get_lidar_data(self) -> Tuple[Optional[float], Optional[float]]:
        """Get lidar data from both sensors."""
        try:
            lidar_ground = self.client.getLidarData(lidar_name="Lidar1", vehicle_name=self.vehicle_name)
            lidar_horizontal = self.client.getLidarData(lidar_name="Lidar2", vehicle_name=self.vehicle_name)
            
            ground_dist = self._process_lidar_points(lidar_ground.point_cloud)
            horizontal_dist = self._process_lidar_points(lidar_horizontal.point_cloud)
            
            return ground_dist, horizontal_dist
            
        except Exception as e:
            print(f"[LIDAR ERROR] Error getting lidar data: {e}")
            return None, None
    
    def _process_lidar_points(self, points) -> Optional[float]:
        """Process lidar point cloud to find minimum distance."""
        if not points:
            return None
        
        try:
            # Convert points to numpy array - handle different formats
            points_array = np.array(points)
            if len(points_array) == 0:
                return None
            
            # Handle different point cloud formats
            if len(points_array.shape) == 1:
                # If it's a 1D array, it might be flattened (x,y,z,x,y,z,...)
                if len(points_array) % 3 == 0:
                    # Reshape to (N, 3) where N is number of points
                    points_array = points_array.reshape(-1, 3)
                else:
                    # If not divisible by 3, try to interpret as single point
                    if len(points_array) >= 3:
                        points_array = points_array[:3].reshape(1, 3)
                    else:
                        return None
            elif len(points_array.shape) == 2:
                # Already in (N, 3) format
                if points_array.shape[1] != 3:
                    print(f"[LIDAR ERROR] Unexpected shape: {points_array.shape}")
                    return None
            else:
                print(f"[LIDAR ERROR] Unexpected array shape: {points_array.shape}")
                return None
            
            # Calculate distances
            distances = np.sqrt(points_array[:, 0]**2 + points_array[:, 1]**2 + points_array[:, 2]**2)
            min_distance = np.min(distances)
            
            return float(min_distance)
        except Exception as e:
            print(f"[LIDAR ERROR] Error processing points: {e}")
            return None
    
    def check_safety(self) -> Tuple[bool, str]:
        """Check safety conditions using lidar data."""
        ground_dist, horizontal_dist = self.get_lidar_data()
        
        # Update cached values
        self.last_lidar_ground_dist = ground_dist
        self.last_lidar_horizontal_dist = horizontal_dist
        
        # Check ground distance
        if ground_dist is not None and ground_dist < self.ground_safety_distance:
            return False, f"Too close to ground: {ground_dist:.2f}m"
        
        # Check horizontal obstacles
        if horizontal_dist is not None and horizontal_dist < self.lidar_safety_distance:
            return False, f"Obstacle too close: {horizontal_dist:.2f}m"
        
        # Check altitude limits
        state = self.client.getMultirotorState()
        altitude = -state.kinematics_estimated.position.z_val
        
        if altitude > self.max_altitude:
            return False, f"Too high: {altitude:.2f}m"
        elif altitude < self.min_altitude:
            return False, f"Too low: {altitude:.2f}m"
        
        return True, "Safe"
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.client.reset()
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)
        
        # Move to start position
        self.client.moveToPositionAsync(
            self.start_pos.x_val, 
            self.start_pos.y_val, 
            self.start_pos.z_val, 
            5
        ).join()
        time.sleep(1.0)  # Wait for movement to complete
        self.client.hoverAsync().join()
        time.sleep(0.5)  # Wait for hover to stabilize
        
        self.current_step = 0
        self.episode_reward = 0
        self.prev_dist = None
        self.collision_count = 0  # Reset collision count
        self.prev_yaw = None  # Reset yaw tracking
        self.prev_altitude = None  # Reset altitude tracking
        
        time.sleep(0.1)
        obs = self.get_observation()
        info = {}
        return obs, info
    
    def step(self, action):
        """Take a step in the environment."""
        # Apply action
        self.apply_action(action)
        time.sleep(0.05)
        
        # Get observation
        obs = self.get_observation()
        
        # Check safety
        is_safe, safety_reason = self.check_safety()
        
        # Get current yaw and altitude for penalties
        state = self.client.getMultirotorState()
        current_yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2] * 180 / np.pi
        current_altitude = -state.kinematics_estimated.position.z_val
        
        # Debug first step
        if self.current_step == 0:
            pos = state.kinematics_estimated.position
            print(f"[DEBUG] First step - Position: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
            print(f"[DEBUG] First step - Action: {action}, Safe: {is_safe}, Reason: {safety_reason}")
        
        # Compute reward
        reward = self.compute_reward(is_safe, safety_reason, action, current_yaw, current_altitude)
        self.episode_reward += reward
        
        # Update step counter
        self.current_step += 1
        
        # Check termination conditions
        terminated, truncated, info = self._check_termination()
        
        # Add termination penalty for unsafe endings
        if terminated and not info.get('goal_reached', False):
            # Different penalties based on termination reason
            if info.get('collision', False):
                # Heavy penalty for collision termination
                reward -= 25.0
                print(f"[TERMINATION] Episode ended due to collision! Heavy penalty applied.")
            elif info.get('unsafe', False):
                # Penalty for unsafe conditions (too close to obstacles, altitude limits, etc.)
                reward -= 15.0
                print(f"[TERMINATION] Episode ended due to unsafe conditions! Penalty applied.")
            else:
                # General penalty for other unsafe terminations
                reward -= 20.0
                print(f"[TERMINATION] Episode ended unsafely! General penalty applied.")
        
        # Add episode info
        if terminated or truncated:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.current_step
            }
        
        # Add lidar info
        info['lidar_ground_dist'] = self.last_lidar_ground_dist
        info['lidar_horizontal_dist'] = self.last_lidar_horizontal_dist
        info['unsafe_condition'] = not is_safe
        info['safety_reason'] = safety_reason
        info['collision_count'] = self.collision_count
        

        
        return obs, reward, terminated, truncated, info
    
    def apply_action(self, action):
        """Apply the given action to the drone."""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2] * 180 / np.pi
        
        # Check if we're in a collision state and need to recover
        if self.collision_count > 0:
            # If we've had collisions, be more conservative
            step_length = self.step_length * 0.5
            altitude_step = self.altitude_step * 0.5
        else:
            step_length = self.step_length
            altitude_step = self.altitude_step
        
        vx, vy, vz = 0, 0, 0
        
        if action == 0:  # forward
            vx = step_length * np.cos(np.deg2rad(yaw))
            vy = step_length * np.sin(np.deg2rad(yaw))
        elif action == 1:  # left (rotate -30 deg)
            self.rotate_by(-30)
        elif action == 2:  # right (rotate +30 deg)
            self.rotate_by(30)
        elif action == 3:  # up
            vz = -altitude_step
        elif action == 4:  # down
            vz = altitude_step
        
        if vx != 0 or vy != 0 or vz != 0:
            self.client.moveByVelocityAsync(vx, vy, vz, 1).join()
            time.sleep(0.1)  # Give time for movement to start
    
    def rotate_by(self, delta_yaw):
        """Rotate the drone by the given angle."""
        self.client.rotateByYawRateAsync(delta_yaw, 1).join()
    
    def get_observation(self):
        """Get observation including depth camera and lidar data."""
        try:
            # Get depth camera image
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, False, False)
            ])
            
            if responses and len(responses) > 0:
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
                img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                img_resized = cv2.resize(img_gray, (48, 48))
                depth_image = img_resized.reshape(48, 48, 1)
            else:
                depth_image = np.zeros((48, 48, 1), dtype=np.uint8)
            
            # Get lidar data
            ground_dist, horizontal_dist = self.get_lidar_data()
            
            # Normalize lidar data (clamp to reasonable range)
            if ground_dist is None:
                ground_dist = 100.0  # Default to max range if no data
            else:
                ground_dist = min(max(ground_dist, 0.0), 100.0)  # Clamp to 0-100m
                
            if horizontal_dist is None:
                horizontal_dist = 100.0  # Default to max range if no data
            else:
                horizontal_dist = min(max(horizontal_dist, 0.0), 100.0)  # Clamp to 0-100m
            
            # Create lidar data array
            lidar_data = np.array([ground_dist, horizontal_dist], dtype=np.float32)
            
            # Return combined observation
            return {
                'depth_image': depth_image,
                'lidar_data': lidar_data
            }
                
        except Exception as e:
            print(f"[OBSERVATION ERROR] Error getting observation: {e}")
            return {
                'depth_image': np.zeros((48, 48, 1), dtype=np.uint8),
                'lidar_data': np.array([100.0, 100.0], dtype=np.float32)
            }
    
    def compute_reward(self, is_safe: bool, safety_reason: str, action: int, current_yaw: float, current_altitude: float) -> float:
        """Compute reward based on safety and progress."""
        # Get current position and distance to goal
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        current_dist = np.linalg.norm(np.array([pos.x_val, pos.y_val, pos.z_val]) -
                                    np.array([self.goal_pos.x_val, self.goal_pos.y_val, self.goal_pos.z_val]))
        
        # Initialize reward
        reward = 0.0
        
        # Collision penalty based on collision count
        if self.collision_count > 0:
            reward -= self.collision_count * 1.0  # Reduced penalty for each collision
        
        # Safety penalty
        if not is_safe:
            reward -= 10.0  # Large penalty for unsafe conditions
        
        # Progress reward
        if self.prev_dist is not None:
            progress = self.prev_dist - current_dist
            reward += progress * 4.0  # Reward for moving toward goal
        
        # Goal reward
        if current_dist < 5.0:
            reward += 50.0  # Large reward for reaching goal
        
        # Yaw rotation penalty to discourage excessive turning
        if self.prev_yaw is not None:
            yaw_change = abs(current_yaw - self.prev_yaw)
            # Normalize yaw change to 0-180 range
            if yaw_change > 180:
                yaw_change = 360 - yaw_change
            
            # Penalty for rotation actions (left/right)
            if action == 1 or action == 2:  # Left/Right rotation actions
                reward -= 1  # Base penalty for rotation
                if yaw_change > 45:  # Additional penalty for large rotations
                    reward -= 2
                if yaw_change > 90:  # Heavy penalty for very large rotations
                    reward -= 3
        
        # Altitude change penalty to discourage excessive up/down movement
        if self.prev_altitude is not None:
            altitude_change = abs(current_altitude - self.prev_altitude)
            
            # Penalty for altitude actions (up/down)
            if action == 3 or action == 4:  # Up/Down actions
                reward -= 1  # Base penalty for altitude change
                if altitude_change > 2.0:  # Additional penalty for large altitude changes
                    reward -= 2
                if altitude_change > 5.0:  # Heavy penalty for very large altitude changes
                    reward -= 3
        
        # Small penalty for each step to encourage efficiency
        reward -= 0.1
        
        # Update previous values
        self.prev_dist = current_dist
        self.prev_yaw = current_yaw
        self.prev_altitude = current_altitude
        
        return reward
    
    def _check_termination(self) -> Tuple[bool, bool, Dict[str, Any]]:
        """Check if episode should terminate."""
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        current_dist = np.linalg.norm(np.array([pos.x_val, pos.y_val, pos.z_val]) -
                                    np.array([self.goal_pos.x_val, self.goal_pos.y_val, self.goal_pos.z_val]))
        
        # Check collision
        collision_info = self.client.simGetCollisionInfo()
        collision = collision_info.has_collided
        
        # Update collision count if collision occurred
        if collision:
            self.collision_count += 1
            print(f"[COLLISION] Collision #{self.collision_count} detected!")
            print(f"[COLLISION] Position: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
            
            # Try to recover from collision by moving up
            if self.collision_count <= 3:  # Only try recovery for first few collisions
                print(f"[RECOVERY] Attempting collision recovery...")
                try:
                    self.client.moveByVelocityAsync(0, 0, -2, 1).join()  # Move up
                    time.sleep(0.5)
                    self.client.hoverAsync().join()
                    time.sleep(0.5)
                except:
                    pass
        
        # Check if goal reached
        goal_reached = current_dist < 5.0
        
        # Check if unsafe
        is_safe, safety_reason = self.check_safety()
        
        # Termination conditions - only hard reset if enabled
        if collision and self.hard_reset_on_collision:
            terminated = True
        elif self.collision_count >= 10:  # Terminate if too many collisions
            terminated = True
        else:
            terminated = goal_reached or not is_safe
        
        truncated = self.current_step >= self.max_steps
        
        info = {
            'distance_to_goal': current_dist,
            'collision': collision,
            'goal_reached': goal_reached,
            'unsafe': not is_safe,
            'collision_count': self.collision_count
        }
        
        return terminated, truncated, info
    

    
    def close(self):
        """Clean up the environment."""
        try:
            self.client.armDisarm(False, vehicle_name=self.vehicle_name)
            self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
        except:
            pass

def main():
    """Test the environment."""
    print("[INFO] Testing Mountain Pass Environment...")
    
    env = MountainPassEnv()
    
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
    for i in range(10):
        action = 0  # forward
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Reward={reward:.2f}, Distance={info.get('distance_to_goal', 0):.2f}, "
              f"Lidar: Ground={obs['lidar_data'][0]:.1f}m, Horizontal={obs['lidar_data'][1]:.1f}m")
        
        if terminated or truncated:
            break
    
    env.close()
    print("[SUCCESS] Environment test completed!")

if __name__ == "__main__":
    main() 