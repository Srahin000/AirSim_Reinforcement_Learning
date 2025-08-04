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

class MountainPassRandomGoalsEnv(gym.Env):

    def __init__(self, 
                 vehicle_name: str = "SimpleFlight",
                 max_steps: int = 500,  # Increased from 200 to 500
                 step_length: float = 4.0,
                 altitude_step: float = 2.0,
                 lidar_safety_distance: float = 2.0,
                 ground_safety_distance: float = 1.5,
                 max_altitude: float = 30.0,
                 min_altitude: float = 1.0,
                 hard_reset_on_collision: bool = False,
                 verbose: bool = True,  # Added verbose flag for detailed logging
                 verbose_step_interval: int = 1):  # Only log every N steps
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
        
        # Improved safety parameters
        self.min_altitude = max(min_altitude, 3.0)  # Ensure minimum altitude is at least 3m
        self.ground_safety_distance = max(ground_safety_distance, 2.0)  # Increase ground safety distance
        
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
        self.verbose = verbose  # Store verbose flag
        self.verbose_step_interval = verbose_step_interval  # Store step interval for verbose logging
        
        # Goal boundaries (x, y coordinates with fixed z = 5)
        self.goal_boundaries = [
            (3.35, 129.33),    # Boundary 1
            (90.43, -70.28),   # Boundary 2
            (-140.58, -91.36), # Boundary 3
            (80.33, 16.86)     # Boundary 4
        ]
        
        # Start position
        self.start_pos = airsim.Vector3r(0, 0, -5)  # Start higher to avoid ground collision
        
        # Current goal position (will be randomized)
        self.goal_pos = None
        
        # Lidar data tracking
        self.last_lidar_ground_dist = 10.0  # Default safe distance
        self.last_lidar_horizontal_dist = 10.0  # Default safe distance
        
        print(f"[INFO] Mountain Pass Random Goals Environment initialized")
        print(f"[INFO] Goal boundaries: {self.goal_boundaries}")
    
    def _generate_random_goal(self) -> airsim.Vector3r:
        """Generate a random goal position within the specified boundaries."""
        # Create a convex hull of the boundary points
        boundary_points = np.array(self.goal_boundaries)
        
        # Generate random point within the boundary
        # For simplicity, we'll use a bounding box approach
        min_x, max_x = boundary_points[:, 0].min(), boundary_points[:, 0].max()
        min_y, max_y = boundary_points[:, 1].min(), boundary_points[:, 1].max()
        
        # Add some margin to avoid edges
        margin = 5.0
        min_x += margin
        max_x -= margin
        min_y += margin
        max_y -= margin
        
        # Generate random position
        random_x = np.random.uniform(min_x, max_x)
        random_y = np.random.uniform(min_y, max_y)
        fixed_z = -5.0  # Fixed altitude as requested
        
        goal = airsim.Vector3r(random_x, random_y, fixed_z)
        
        print(f"[GOAL] Generated new goal: ({goal.x_val:.2f}, {goal.y_val:.2f}, {goal.z_val:.2f})")
        return goal
    
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
            
            # Handle different point formats
            if len(points_array.shape) == 1:
                # Single dimension array - might be flattened
                if len(points_array) % 3 == 0:
                    # Reshape to (N, 3) format
                    points_array = points_array.reshape(-1, 3)
                else:
                    # Single point or invalid format
                    return None
            elif len(points_array.shape) == 2:
                # Already in (N, 3) format
                if points_array.shape[1] != 3:
                    # Wrong number of dimensions
                    return None
            else:
                # Invalid format
                return None
            
            # Calculate distances from origin
            distances = np.linalg.norm(points_array, axis=1)
            
            # Return minimum distance
            return float(np.min(distances))
            
        except Exception as e:
            print(f"[LIDAR PROCESSING ERROR] Error processing lidar points: {e}")
            return None
    
    def check_safety(self) -> Tuple[bool, str]:
        """Check if the current state is safe."""
        try:
            # Get current state
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            current_altitude = -pos.z_val
            
            # Check altitude limits
            if current_altitude > self.max_altitude:
                return False, f"Altitude too high: {current_altitude:.2f}m"
            if current_altitude < self.min_altitude:
                return False, f"Altitude too low: {current_altitude:.2f}m"
            
            # Get lidar data
            ground_dist, horizontal_dist = self.get_lidar_data()
            
            if ground_dist is not None and ground_dist < self.ground_safety_distance:
                return False, f"Too close to ground: {ground_dist:.2f}m"
            
            if horizontal_dist is not None and horizontal_dist < self.lidar_safety_distance:
                return False, f"Too close to obstacle: {horizontal_dist:.2f}m"
            
            # Store lidar data for observation
            self.last_lidar_ground_dist = ground_dist
            self.last_lidar_horizontal_dist = horizontal_dist
            
            return True, "Safe"
            
        except Exception as e:
            print(f"[SAFETY ERROR] Error checking safety: {e}")
            return False, f"Safety check error: {e}"
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment. Only generate new goal if one doesn't exist yet."""
        super().reset(seed=seed)
        
        self.client.reset()
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)
        
        # Only generate new random goal if one doesn't exist yet
        if self.goal_pos is None:
            self.goal_pos = self._generate_random_goal()
            if self.verbose:
                print(f"[RESET] Generated new goal: ({self.goal_pos.x_val:.2f}, {self.goal_pos.y_val:.2f}, {self.goal_pos.z_val:.2f})")
        else:
            if self.verbose:
                print(f"[RESET] Keeping existing goal: ({self.goal_pos.x_val:.2f}, {self.goal_pos.y_val:.2f}, {self.goal_pos.z_val:.2f})")
        
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
        
        # Initialize lidar data to safe defaults
        self.last_lidar_ground_dist = 10.0
        self.last_lidar_horizontal_dist = 10.0
        
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
        current_pos = state.kinematics_estimated.position
        
        # Verbose logging for detailed step information
        if self.verbose and (self.current_step + 1) % self.verbose_step_interval == 0:
            action_names = ["FORWARD", "LEFT", "RIGHT", "UP", "DOWN"]
            action_name = action_names[action] if 0 <= action < len(action_names) else f"UNKNOWN({action})"
            
            # Calculate distance to goal
            goal_dist = None
            if self.goal_pos:
                goal_dist = math.sqrt(
                    (current_pos.x_val - self.goal_pos.x_val) ** 2 +
                    (current_pos.y_val - self.goal_pos.y_val) ** 2 +
                    (current_pos.z_val - self.goal_pos.z_val) ** 2
                )
            
            print(f"[STEP {self.current_step + 1}] Action: {action_name}")
            print(f"  Position: ({current_pos.x_val:.2f}, {current_pos.y_val:.2f}, {current_pos.z_val:.2f})")
            print(f"  Altitude: {current_altitude:.2f}m, Yaw: {current_yaw:.1f}°")
            print(f"  Safe: {is_safe}, Reason: {safety_reason}")
            if goal_dist is not None:
                print(f"  Distance to Goal: {goal_dist:.2f}m")
            # Safe lidar data display with fallback for None values
            ground_dist_display = f"{self.last_lidar_ground_dist:.2f}" if self.last_lidar_ground_dist is not None else "N/A"
            horizontal_dist_display = f"{self.last_lidar_horizontal_dist:.2f}" if self.last_lidar_horizontal_dist is not None else "N/A"
            print(f"  Lidar - Ground: {ground_dist_display}m, Horizontal: {horizontal_dist_display}m")
            print(f"  Collision Count: {self.collision_count}")
        
        # Debug first step
        if self.current_step == 0:
            pos = state.kinematics_estimated.position
            print(f"[DEBUG] First step - Position: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
            print(f"[DEBUG] First step - Action: {action}, Safe: {is_safe}, Reason: {safety_reason}")
        
        # Check termination conditions first to get collision info
        terminated, truncated, info = self._check_termination()
        
        # Compute reward with collision penalty
        reward = self.compute_reward(is_safe, safety_reason, action, current_yaw, current_altitude, info.get('collision', False))
        self.episode_reward += reward
        
        # Verbose logging for reward and termination
        if self.verbose and (self.current_step + 1) % self.verbose_step_interval == 0:
            print(f"  Reward: {reward:.3f}, Episode Total: {self.episode_reward:.3f}")
            if terminated:
                print(f"  [TERMINATED] Episode ending: {info.get('termination_reason', 'Unknown')}")
            elif truncated:
                print(f"  [TRUNCATED] Episode ending: Max steps reached")
            if info.get('goal_reached', False):
                print(f"  [GOAL REACHED] New goal will be generated!")
        
        # Update step counter
        self.current_step += 1
        
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
        try:
            # Get current state
            state = self.client.getMultirotorState()
            current_pos = state.kinematics_estimated.position
            current_yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2] * 180 / np.pi
            
            # Convert yaw to radians
            yaw_rad = current_yaw * np.pi / 180
            
            # Calculate movement based on action
            if action == 0:  # Forward
                dx = self.step_length * np.cos(yaw_rad)
                dy = self.step_length * np.sin(yaw_rad)
                dz = 0
            elif action == 1:  # Left
                dx = -self.step_length * np.sin(yaw_rad)
                dy = self.step_length * np.cos(yaw_rad)
                dz = 0
            elif action == 2:  # Right
                dx = self.step_length * np.sin(yaw_rad)
                dy = -self.step_length * np.cos(yaw_rad)
                dz = 0
            elif action == 3:  # Up
                dx = 0
                dy = 0
                dz = -self.altitude_step
            elif action == 4:  # Down
                dx = 0
                dy = 0
                dz = self.altitude_step
            else:
                print(f"[ERROR] Unknown action: {action}")
                return
            
            # Calculate new position
            new_x = current_pos.x_val + dx
            new_y = current_pos.y_val + dy
            new_z = current_pos.z_val + dz
            
            # Boundary checks to prevent flying away from environment
            # Define environment boundaries (based on goal boundaries with extra margin)
            boundary_points = np.array(self.goal_boundaries)
            min_x, max_x = boundary_points[:, 0].min(), boundary_points[:, 0].max()
            min_y, max_y = boundary_points[:, 1].min(), boundary_points[:, 1].max()
            
            # Add safety margin
            margin = 20.0
            min_x -= margin
            max_x += margin
            min_y -= margin
            max_y += margin
            
            # Clamp position to boundaries
            new_x = np.clip(new_x, min_x, max_x)
            new_y = np.clip(new_y, min_y, max_y)
            new_z = np.clip(new_z, -self.max_altitude, -self.min_altitude)
            
            # Additional safety check: ensure minimum altitude
            if -new_z < self.min_altitude:
                new_z = -self.min_altitude
                if self.verbose and self.current_step % 10 == 0:  # Log occasionally to avoid spam
                    print(f"[SAFETY] Preventing movement below minimum altitude: {self.min_altitude}m")
            
            # Move to new position
            self.client.moveToPositionAsync(new_x, new_y, new_z, 2).join()
            
        except Exception as e:
            print(f"[ACTION ERROR] Error applying action {action}: {e}")
    
    def rotate_by(self, delta_yaw):
        """Rotate the drone by the given yaw angle."""
        try:
            self.client.rotateByYawRateAsync(delta_yaw, 1).join()
        except Exception as e:
            print(f"[ROTATION ERROR] Error rotating by {delta_yaw}: {e}")
    
    def get_observation(self):
        """Get the current observation."""
        try:
            # Get depth image
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, False, False)
            ])
            
            if responses and len(responses) > 0:
                # Convert depth image - handle different data formats
                if hasattr(responses[0], 'image_data_float') and responses[0].image_data_float:
                    depth_image = airsim.string_to_float_array(responses[0].image_data_float)
                    # Try to determine the correct size
                    total_pixels = len(depth_image)
                    # Common sizes: 256x256=65536, 512x512=262144, 1024x1024=1048576
                    # 110592 is likely 332x332 or similar
                    if total_pixels == 65536:
                        size = 256
                    elif total_pixels == 262144:
                        size = 512
                    elif total_pixels == 1048576:
                        size = 1024
                    elif total_pixels == 110592:
                        # This is likely 332x332 but not a perfect square
                        # Use 332 and handle the extra pixels by truncating
                        size = 332
                        # Truncate to fit 332x332
                        depth_image = depth_image[:size*size]
                    elif total_pixels == 110224:
                        # This is 332x332 exactly
                        size = 332
                    else:
                        # Try to find a reasonable size
                        size = int(np.sqrt(total_pixels))
                        if size * size != total_pixels:
                            # If not a perfect square, use a fallback
                            size = 256
                    
                    depth_image = np.array(depth_image).reshape(size, size)
                    # Resize to 48x48
                    depth_image = cv2.resize(depth_image, (48, 48))
                elif hasattr(responses[0], 'image_data_uint8') and responses[0].image_data_uint8:
                    # Handle uint8 format
                    depth_image = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                    # Try to determine the correct size
                    total_pixels = len(depth_image)
                    # Common sizes: 256x256=65536, 512x512=262144, 1024x1024=1048576
                    # 110592 is likely 332x332 or similar
                    if total_pixels == 65536:
                        size = 256
                    elif total_pixels == 262144:
                        size = 512
                    elif total_pixels == 1048576:
                        size = 1024
                    elif total_pixels == 110592:
                        # This is likely 332x332 but not a perfect square
                        # Use 332 and handle the extra pixels by truncating
                        size = 332
                        # Truncate to fit 332x332
                        depth_image = depth_image[:size*size]
                    elif total_pixels == 110224:
                        # This is 332x332 exactly
                        size = 332
                    else:
                        # Try to find a reasonable size
                        size = int(np.sqrt(total_pixels))
                        if size * size != total_pixels:
                            # If not a perfect square, use a fallback
                            size = 256
                    
                    depth_image = depth_image.reshape(size, size)
                    depth_image = depth_image.astype(np.float32) / 255.0
                    # Resize to 48x48
                    depth_image = cv2.resize(depth_image, (48, 48))
                else:
                    # Fallback: create empty image
                    depth_image = np.zeros((48, 48), dtype=np.float32)
                
                # Normalize to 0-255 range
                depth_image = np.clip(depth_image * 255, 0, 255).astype(np.uint8)
                depth_image = np.expand_dims(depth_image, axis=-1)
            else:
                # Fallback: create empty image
                depth_image = np.zeros((48, 48, 1), dtype=np.uint8)
            
            # Get lidar data
            ground_dist, horizontal_dist = self.get_lidar_data()
            
            # Use fallback values if lidar data is unavailable
            if ground_dist is None:
                ground_dist = 10.0
            if horizontal_dist is None:
                horizontal_dist = 10.0
            
            lidar_data = np.array([ground_dist, horizontal_dist], dtype=np.float32)
            
            return {
                'depth_image': depth_image,
                'lidar_data': lidar_data
            }
            
        except Exception as e:
            print(f"[OBSERVATION ERROR] Error getting observation: {e}")
            # Return fallback observation
            return {
                'depth_image': np.zeros((48, 48, 1), dtype=np.uint8),
                'lidar_data': np.array([10.0, 10.0], dtype=np.float32)
            }
    
    def compute_reward(self, is_safe: bool, safety_reason: str, action: int, current_yaw: float, current_altitude: float, collision: bool = False) -> float:
        """Compute the reward for the current state."""
        reward = 0.0
        
        try:
            # Get current position and distance to goal
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            current_dist = np.linalg.norm(np.array([pos.x_val, pos.y_val, pos.z_val]) -
                                        np.array([self.goal_pos.x_val, self.goal_pos.y_val, self.goal_pos.z_val]))
            
            # Distance-based reward
            if self.prev_dist is not None:
                dist_change = self.prev_dist - current_dist
                reward += dist_change * 2.0  # Reward for getting closer to goal
            self.prev_dist = current_dist
            
            # Goal proximity reward
            if current_dist < 10.0:
                reward += 5.0  # Bonus for being close to goal
            if current_dist < 5.0:
                reward += 10.0  # Big bonus for being very close
            
            # Safety penalties
            if not is_safe:
                reward -= 10.0  # Penalty for unsafe conditions
            
            # Heavy collision penalty
            if collision:
                reward -= 50.0  # Heavy penalty for collisions
                print(f"[COLLISION PENALTY] Heavy penalty applied: -50.0")
            
            # Action penalties to encourage efficient movement
            if action in [3, 4]:  # Up/Down actions
                reward -= 0.5  # Small penalty for altitude changes
            
            # Rotation penalty to encourage straight movement
            if self.prev_yaw is not None:
                yaw_change = abs(current_yaw - self.prev_yaw)
                if yaw_change > 5:  # Penalty for large rotations
                    reward -= 1.0
            self.prev_yaw = current_yaw
            
            # Altitude change penalty
            if self.prev_altitude is not None:
                alt_change = abs(current_altitude - self.prev_altitude)
                if alt_change > 1.0:  # Penalty for large altitude changes
                    reward -= 0.5
            self.prev_altitude = current_altitude
            
            # Small penalty for each step to encourage efficiency
            reward -= 0.1
            
        except Exception as e:
            print(f"[REWARD ERROR] Error computing reward: {e}")
            reward = -1.0  # Penalty for errors
        
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
            if self.verbose:
                print(f"[COLLISION] Collision #{self.collision_count} detected!")
                print(f"[COLLISION] Position: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
            
            # Try to recover from collision by moving up (recovery maneuver)
            if self.verbose:
                print(f"[RECOVERY] Attempting collision recovery maneuver...")
            try:
                # Get current position
                current_pos = state.kinematics_estimated.position
                current_altitude = -current_pos.z_val
                
                # Move up to safe altitude (at least 5m above ground)
                target_altitude = max(current_altitude + 3.0, 5.0)
                target_z = -target_altitude
                
                # Move to safe altitude
                self.client.moveToPositionAsync(current_pos.x_val, current_pos.y_val, target_z, 3).join()
                time.sleep(1.0)
                
                # Hover to stabilize
                self.client.hoverAsync().join()
                time.sleep(0.5)
                
                if self.verbose:
                    print(f"[RECOVERY] Moved to safe altitude: {target_altitude:.2f}m")
                    
            except Exception as e:
                if self.verbose:
                    print(f"[RECOVERY ERROR] Recovery maneuver failed: {e}")
                pass
        
        # Check if goal reached
        goal_reached = current_dist < 5.0
        
        # If goal reached, generate new goal and continue episode
        if goal_reached:
            if self.verbose:
                print(f"[GOAL REACHED] Goal reached! Distance: {current_dist:.2f}m")
                print(f"[GOAL REACHED] Old goal: ({self.goal_pos.x_val:.2f}, {self.goal_pos.y_val:.2f}, {self.goal_pos.z_val:.2f})")
            
            # Generate new random goal
            self.goal_pos = self._generate_random_goal()
            
            # Reset distance tracking for new goal
            self.prev_dist = None
            
            if self.verbose:
                print(f"[GOAL REACHED] New goal: ({self.goal_pos.x_val:.2f}, {self.goal_pos.y_val:.2f}, {self.goal_pos.z_val:.2f})")
            
            # Don't terminate episode, just continue with new goal
            goal_reached = False  # Reset for this step
        
        # Check if unsafe
        is_safe, safety_reason = self.check_safety()
        
        # Termination conditions - only terminate on too many collisions or unsafe conditions
        if self.collision_count >= 10:  # Terminate if too many collisions
            terminated = True
            if self.verbose:
                print(f"[TERMINATION] Too many collisions ({self.collision_count}), terminating episode")
        else:
            terminated = not is_safe  # Only terminate on unsafe conditions, not goal reached
        
        truncated = self.current_step >= self.max_steps
        
        info = {
            'distance_to_goal': current_dist,
            'collision': collision,
            'goal_reached': goal_reached,
            'unsafe': not is_safe,
            'collision_count': self.collision_count
        }
        
        # Add termination reason
        if terminated:
            if self.collision_count >= 10:
                info['termination_reason'] = f"Too many collisions ({self.collision_count})"
            else:
                info['termination_reason'] = f"Unsafe condition: {safety_reason}"
        elif truncated:
            info['termination_reason'] = "Max steps reached"
        
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
    print("[INFO] Testing Mountain Pass Random Goals Environment...")
    
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