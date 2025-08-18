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
import random
from typing import Tuple, Optional, Dict, Any


class MountainPassRandomGoalsEnv(gym.Env):

    def __init__(self, 
                 vehicle_name: str = "SimpleFlight",
                 max_steps: int = 200,
                 step_length: float = 4.0,
                 altitude_step: float = 2.0,
                 lidar_safety_distance: float = 0.25,
                 ground_safety_distance: float = 0.25,
                 max_altitude: float = 50.0,
                 min_altitude: float = 1.0,
                 hard_reset_on_collision: bool = True,
                 verbose: bool = False,
                 ignored_collision_objects: Optional[list] = None,
                 log_steps: bool = False,
                 safety_arm_steps: int = 3,
                 safety_leniency: str = "normal"): # "more", "normal", "less"
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
	    # Curriculum learning variables
        self.episode_count = 0
        self.curriculum_start_dist = 10.0
        self.curriculum_max_dist = 100.0
        self.curriculum_growth_rate = 0.5      # +0.5m every 50 episodes
        self.curriculum_episodes_per_increase = 50  # Increase every 50 episodes

	    # Safe zones for random position generation
        self.safe_zones = [
            # Zone 1: Mountain pass area
            {
                'x_range': (-60.60, -12.57),  # Fixed: min should be less than max
                'y_range': (-62.67, 37.09),
                'z_range': (-25.0, -7.67),    # Fixed: negative Z = above ground in NED system
                'name': 'Mountain Pass'
            },
            # Zone 2: Valley area
            {
                'x_range': (10.90, 21.6),
                'y_range': (-90.0, -75.77),
                'z_range': (-9.0, -3.0),      # Fixed: negative Z = above ground in NED system
                'name': 'Valley'
            },
            # Zone 3: Plateau area
            {
                'x_range': (43.7, 59.4),
                'y_range': (32.65, 46.85),
                'z_range': (-4.0, -3.0),      # Fixed: negative Z = above ground in NED system
                'name': 'Plateau'
            },
            # Zone 4: High mountain area
            {
                'x_range': (33.7, 46.33),
                'y_range': (109.67, 126.46),
                'z_range': (-42.0, -28.0),    # Fixed: negative Z = above ground in NED system
                'name': 'High Mountain'
            }
        ]
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(5)  # forward, left, right, up, down
        
        # Observation space: depth image (48x48x1) normalized to [0,1] + lidar data (2 values) normalized to [0,1]
        # + GPS (lat, lon, alt)
        self.observation_space = spaces.Dict({
            'depth_image': spaces.Box(low=0.0, high=1.0, shape=(48, 48, 1), dtype=np.float32),
            'lidar_data': spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            'gps': spaces.Box(low=np.array([-90.0, -180.0, -1000.0], dtype=np.float32),
                              high=np.array([90.0, 180.0, 10000.0], dtype=np.float32),
                              shape=(3,), dtype=np.float32)
        })
        
        # Episode tracking
        self.current_step = 0
        self.warmup_steps = 2
        self.episode_reward = 0
        self.prev_dist = None
        self.collision_count = 0  # Track collision count
        self.hard_reset_on_collision = hard_reset_on_collision
        self.prev_yaw = None  # Track previous yaw for rotation penalty
        self.prev_altitude = None  # Track previous altitude for altitude change penalty
        
        # Goal and start positions
        self.goal_pos = None
        self.start_pos = None
        self.start_zone = None  # Track which zone was used for start
        
        # Lidar data tracking
        self.last_lidar_ground_dist = None
        self.last_lidar_horizontal_dist = None
        
        # Verbosity
        self.verbose = verbose
        self.ignored_collision_objects = ignored_collision_objects or ["Plane_3"]
        self.log_steps = log_steps
        self._episode_step_trace = []
        self.safety_arm_steps = max(0, int(safety_arm_steps))

        # Safety leniency system
        self.safety_leniency = safety_leniency.lower()
        if self.safety_leniency == "less":
            # Stricter safety - increase safety distances and reduce altitude range
            self.effective_lidar_safety = self.lidar_safety_distance * 1.5
            self.effective_ground_safety = self.ground_safety_distance * 1.5
            self.effective_min_altitude = self.min_altitude * 1.3
            self.effective_max_altitude = self.max_altitude * 0.8
            leniency_description = "STRICT (less lenient)"
        elif self.safety_leniency == "more":
            # More lenient safety - decrease safety distances and increase altitude range
            self.effective_lidar_safety = self.lidar_safety_distance * 0.7
            self.effective_ground_safety = self.ground_safety_distance * 0.7
            self.effective_min_altitude = self.min_altitude * 0.7
            self.effective_max_altitude = self.max_altitude * 1.2
            leniency_description = "LENIENT (more permissive)"
        else:  # "normal"
            # Default safety - use original values
            self.effective_lidar_safety = self.lidar_safety_distance
            self.effective_ground_safety = self.ground_safety_distance
            self.effective_min_altitude = self.min_altitude
            self.effective_max_altitude = self.max_altitude
            leniency_description = "NORMAL (default)"
        
        print(f"[INFO] Mountain Pass Environment initialized with {leniency_description} safety settings")
        print(f"[INFO] Effective safety distances: Lidar={self.effective_lidar_safety:.1f}m, Ground={self.effective_ground_safety:.1f}m")
        print(f"[INFO] Effective altitude range: {self.effective_min_altitude:.1f}m to {self.effective_max_altitude:.1f}m")
        print(f"[INFO] Using 4 predefined safe zones for position generation")
        
        # Print training recommendations
        self._print_training_recommendations()
    
    def _print_training_recommendations(self):
        """Print training recommendations based on current safety leniency setting."""
        print(f"\n[INFO] === TRAINING RECOMMENDATIONS ===")
        
        if self.safety_leniency == "less":
            print(f"[INFO] 🚨 STRICT MODE: Use this for:")
            print(f"     • Final training/testing with real hardware")
            print(f"     • Production deployment")
            print(f"     • When you need maximum safety guarantees")
            print(f"     • Advanced agents that can handle tight constraints")
            print(f"[INFO] ⚠️  Note: Higher failure rates expected during learning")
            
        elif self.safety_leniency == "more":
            print(f"[INFO] 🎯 LENIENT MODE: Use this for:")
            print(f"     • Early training phases")
            print(f"     • Exploration-heavy learning")
            print(f"     • When you want faster learning progress")
            print(f"     • Testing new algorithms or reward functions")
            print(f"[INFO] ⚠️  Note: May allow unsafe behaviors - monitor closely")
            
        else:  # "normal"
            print(f"[INFO] ⚖️  NORMAL MODE: Use this for:")
            print(f"     • Balanced training approach")
            print(f"     • General purpose learning")
            print(f"     • When you want reasonable safety with good learning")
            print(f"     • Most training scenarios")
            print(f"[INFO] ✅ Recommended starting point for most users")
        
        print(f"[INFO] ================================\n")

    def generate_random_position_from_zones(self) -> Tuple[airsim.Vector3r, int]:
        """Generate a random position from one of the 4 predefined safe zones."""
        # Randomly select a zone (1-4)
        zone_idx = random.randint(0, 3)
        zone = self.safe_zones[zone_idx]
        
        # Generate random coordinates within the zone
        x = random.uniform(zone['x_range'][0], zone['x_range'][1])
        y = random.uniform(zone['y_range'][0], zone['y_range'][1])
        z = random.uniform(zone['z_range'][0], zone['z_range'][1])
        
        # Create position vector
        position = airsim.Vector3r(x, y, z)
        
        if self.verbose:
            print(f"[ZONE {zone_idx + 1}] Generated position in {zone['name']}: ({x:.2f}, {y:.2f}, {z:.2f})")
        
        return position, zone_idx + 1

    def generate_random_position(self):
        """Generate a random position from predefined safe zones."""
        max_attempts = 50
        
        for attempt in range(max_attempts):
            # Generate position from random zone
            candidate_pos, zone_num = self.generate_random_position_from_zones()
            
            # Validate the position
            if self._validate_position(candidate_pos, zone_num):
                return candidate_pos
        
        # Fallback: use zone 1 with conservative coordinates
        if self.verbose:
            print("[WARNING] Failed to find valid position in zones, using fallback")
        zone = self.safe_zones[0]
        x = random.uniform(zone['x_range'][0] + 5, zone['x_range'][1] - 5)  # Stay away from edges
        y = random.uniform(zone['y_range'][0] + 5, zone['y_range'][1] - 5)
        z = zone['z_range'][0] + 2  # Conservative height
        return airsim.Vector3r(x, y, z)

    def _validate_position(self, position: airsim.Vector3r, zone_num: int) -> bool:
        """Validate if a position is safe and suitable for spawning."""
        try:
            # Check if position is within zone bounds
            zone = self.safe_zones[zone_num - 1]
            
            # Check X coordinate
            x_ok = zone['x_range'][0] <= position.x_val <= zone['x_range'][1]
            # Check Y coordinate  
            y_ok = zone['y_range'][0] <= position.y_val <= zone['y_range'][1]
            # Check Z coordinate
            z_ok = zone['z_range'][0] <= position.z_val <= zone['z_range'][1]
            
            if not x_ok or not y_ok or not z_ok:
                if self.verbose:
                    print(f"[VALIDATION] Position {position} is OUTSIDE zone {zone_num} bounds:")
                    print(f"  X: {position.x_val:.2f} (range: {zone['x_range'][0]:.2f} to {zone['x_range'][1]:.2f}) - {'OK' if x_ok else 'FAIL'}")
                    print(f"  Y: {position.y_val:.2f} (range: {zone['y_range'][0]:.2f} to {zone['y_range'][1]:.2f}) - {'OK' if y_ok else 'FAIL'}")
                    print(f"  Z: {position.z_val:.2f} (range: {zone['z_range'][0]:.2f} to {zone['z_range'][1]:.2f}) - {'OK' if z_ok else 'FAIL'}")
                return False
            
            # Check if position is in free space
            if not self._is_position_free(position, clearance=3.0):
                return False
            
            # Check open air
            if not self._is_open_air(position, up_clearance_m=3.0):
                return False
            
            # Additional terrain analysis for conservative spawning
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"[DEBUG] Position validation failed: {e}")
            return False

    def _local_ned_offset_to_geopoint(self, origin_geo: airsim.GeoPoint, d_north_m: float, d_east_m: float, d_up_m: float) -> airsim.GeoPoint:
        """Approximate conversion from local NED offset (meters) to GeoPoint around origin_geo."""
        # WGS84 approximate conversion
        earth_radius_m = 6378137.0
        d_lat = (d_north_m / earth_radius_m) * (180.0 / math.pi)
        # avoid cos(lat)=0 at poles
        lat_rad = origin_geo.latitude * math.pi / 180.0
        d_lon = (d_east_m / (earth_radius_m * max(1e-6, math.cos(lat_rad)))) * (180.0 / math.pi)
        lat = origin_geo.latitude + d_lat
        lon = origin_geo.longitude + d_lon
        alt = origin_geo.altitude + d_up_m
        gp = airsim.GeoPoint()
        gp.latitude = lat
        gp.longitude = lon
        gp.altitude = alt
        return gp

    def _geopoint_from_local(self, local: airsim.Vector3r) -> airsim.GeoPoint:
        """Convert a local NED position (x=N, y=E, z=Down) to GeoPoint using current vehicle GPS as origin."""
        state = self.client.getMultirotorState()
        origin = state.gps_location
        # delta from origin local
        # Up is negative of Down
        d_up = -local.z_val
        return self._local_ned_offset_to_geopoint(origin, local.x_val, local.y_val, d_up)

    def _has_line_of_sight(self, start_local: airsim.Vector3r, goal_local: airsim.Vector3r) -> bool:
        """Check line-of-sight between start and goal using GeoPoint approximation."""
        try:
            state = self.client.getMultirotorState()
            start_geo = state.gps_location
            d_north = goal_local.x_val - start_local.x_val
            d_east = goal_local.y_val - start_local.y_val
            d_up = -(goal_local.z_val - start_local.z_val)
            goal_geo = self._local_ned_offset_to_geopoint(start_geo, d_north, d_east, d_up)
            return bool(self.client.simTestLineOfSightBetweenPoints(start_geo, goal_geo))
        except Exception:
            return True  # If API unavailable, assume reachable to avoid deadlock

    def _is_open_air(self, local: airsim.Vector3r, up_clearance_m: float = 5.0) -> bool:
        """Heuristic: ensure there is clear line-of-sight directly upward from a local NED point.

        This helps reject spawn points that are enclosed inside hollow meshes/caves.
        """
        try:
            # Convert local to geopoint and cast a short ray straight up
            start_geo = self._geopoint_from_local(local)
            end_geo = airsim.GeoPoint()
            end_geo.latitude = start_geo.latitude
            end_geo.longitude = start_geo.longitude
            end_geo.altitude = start_geo.altitude + max(1.0, up_clearance_m)
            return bool(self.client.simTestLineOfSightBetweenPoints(start_geo, end_geo))
        except Exception:
            # If LOS API is unavailable, do not block spawning
            return True

    def _is_position_free(self, position: airsim.Vector3r, clearance: float = 3.0) -> bool:
        """Check if a position is in free space with sufficient clearance."""
        try:
            # Test multiple directions to ensure we're not inside solid objects
            test_directions = [
                (0, 0, 1),    # Up
                (0, 0, -1),   # Down  
                (1, 0, 0),    # North
                (-1, 0, 0),   # South
                (0, 1, 0),    # East
                (0, -1, 0),   # West
            ]
            
            # Convert position to geopoint for line-of-sight tests
            start_geo = self._geopoint_from_local(position)
            
            for dx, dy, dz in test_directions:
                # Test line of sight in each direction
                end_geo = airsim.GeoPoint()
                end_geo.latitude = start_geo.latitude + (dx * clearance / 111000.0)  # Rough conversion
                end_geo.longitude = start_geo.longitude + (dy * clearance / (111000.0 * math.cos(math.radians(start_geo.latitude))))
                end_geo.altitude = start_geo.altitude + dz * clearance
                
                # If any direction is blocked, position might be inside solid object
                if not self.client.simTestLineOfSightBetweenPoints(start_geo, end_geo):
                    return False
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"[DEBUG] Position freedom check failed: {e}")
            # If we can't check, assume it's free to avoid blocking
            return True

    def _analyze_terrain_around_position(self, position: airsim.Vector3r, radius: float = 10.0) -> Dict[str, Any]:
        """Analyze terrain around a position to check for steep slopes or dangerous areas."""
        try:
            terrain_info = {
                'max_height': float('-inf'),
                'min_height': float('inf'),
                'height_variance': 0.0,
                'steep_slope': False,
                'safe_landing': True
            }
            
            heights = []
            # Sample terrain in a grid around the position
            for dx in np.arange(-radius, radius + 1, 2.0):
                for dy in np.arange(-radius, radius + 1, 2.0):
                    x = position.x_val + dx
                    y = position.y_val + dy
                    
                    # Keep within bounds
                    if not (self.ENV_X_MIN <= x <= self.ENV_X_MAX and self.ENV_Y_MIN <= y <= self.ENV_Y_MAX):
                        continue
                    
                    try:
                        height = self.client.simGetTerrainHeight(x, y)
                        if height is not None:
                            heights.append(height)
                    except Exception:
                        continue
            
            if heights:
                heights = np.array(heights)
                terrain_info['max_height'] = float(np.max(heights))
                terrain_info['min_height'] = float(np.min(heights))
                terrain_info['height_variance'] = float(np.var(heights))
                
                # Check for steep slopes (high variance indicates rough terrain)
                terrain_info['steep_slope'] = terrain_info['height_variance'] > 25.0
                
                # Check if terrain is suitable for landing (not too rough)
                terrain_info['safe_landing'] = terrain_info['height_variance'] < 50.0
                
            return terrain_info
            
        except Exception as e:
            if self.verbose:
                print(f"[DEBUG] Terrain analysis failed: {e}")
            return {
                'max_height': 0.0,
                'min_height': 0.0,
                'height_variance': 0.0,
                'steep_slope': False,
                'safe_landing': True
            }

    def find_safe_zone(self, center_x: float, center_y: float, radius: float = 50.0) -> Optional[airsim.Vector3r]:
        """Find a safe zone around a center point for spawning."""
        max_attempts = 100
        grid_spacing = 5.0
        
        for attempt in range(max_attempts):
            # Try different distances from center
            distance = random.uniform(5.0, radius)
            angle = random.uniform(0, 2 * math.pi)
            
            x = center_x + distance * math.cos(angle)
            y = center_y + distance * math.sin(angle)
            
            # Keep within bounds
            if not (self.ENV_X_MIN <= x <= self.ENV_X_MAX and self.ENV_Y_MIN <= y <= self.ENV_Y_MAX):
                continue
            
            try:
                terrain_height = self.client.simGetTerrainHeight(x, y)
                if terrain_height is None:
                    continue
                
                # Ensure sufficient clearance above terrain
                z = terrain_height - (self.min_altitude * 2.0)
                z = max(self.ENV_Z_MIN + 5.0, z)
                z = min(self.ENV_Z_MAX - 5.0, z)
                
                candidate_pos = airsim.Vector3r(x, y, z)
                
                # Check if this position is safe
                if self._is_position_free(candidate_pos, clearance=5.0):
                    terrain_analysis = self._analyze_terrain_around_position(candidate_pos)
                    if terrain_analysis['safe_landing'] and not terrain_analysis['steep_slope']:
                        return candidate_pos
                        
            except Exception:
                continue
        
        return None

    def generate_goal_near_start(self, start: airsim.Vector3r, max_distance: float):
        """Generate a goal position near the start, preferably in a different zone."""
        # Try to place goal in a different zone for variety
        other_zones = [i for i in range(4) if i != (self.start_zone - 1)]
        random.shuffle(other_zones)
        
        # First try: goal in different zone
        for zone_idx in other_zones:
            zone = self.safe_zones[zone_idx]
            for attempt in range(20):
                x = random.uniform(zone['x_range'][0], zone['x_range'][1])
                y = random.uniform(zone['y_range'][0], zone['y_range'][1])
                z = random.uniform(zone['z_range'][0], zone['z_range'][1])
                
                candidate = airsim.Vector3r(x, y, z)
                
                # Check if within max distance and has line of sight
                dist = np.linalg.norm(np.array([x, y, z]) - np.array([start.x_val, start.y_val, start.z_val]))
                if dist <= max_distance and self._has_line_of_sight(start, candidate):
                    if self.verbose:
                        print(f"[GOAL] Placed in different zone {zone_idx + 1} ({zone['name']})")
                    return candidate
        
        # Fallback: goal in same zone but different area
        zone = self.safe_zones[self.start_zone - 1]
        for attempt in range(30):
            x = random.uniform(zone['x_range'][0], zone['x_range'][1])
            y = random.uniform(zone['y_range'][0], zone['y_range'][1])
            z = random.uniform(zone['z_range'][0], zone['z_range'][1])
            
            candidate = airsim.Vector3r(x, y, z)
            
            # Ensure minimum distance from start
            dist = np.linalg.norm(np.array([x, y, z]) - np.array([start.x_val, start.y_val, start.z_val]))
            if dist >= 5.0 and dist <= max_distance and self._has_line_of_sight(start, candidate):
                if self.verbose:
                    print(f"[GOAL] Placed in same zone {self.start_zone} ({zone['name']})")
                return candidate
        
        # Final fallback: goal forward from start
        yaw_forward_rad = 0.0
        x = start.x_val + min(max_distance, 10.0) * math.cos(yaw_forward_rad)
        y = start.y_val + min(max_distance, 10.0) * math.sin(yaw_forward_rad)
        z = start.z_val + random.uniform(-2.0, 2.0)
        
        # Keep within zone bounds
        zone = self.safe_zones[self.start_zone - 1]
        x = max(zone['x_range'][0], min(zone['x_range'][1], x))
        y = max(zone['y_range'][0], min(zone['y_range'][1], y))
        z = max(zone['z_range'][0], min(zone['z_range'][1], z))
        
        if self.verbose:
            print(f"[GOAL] Using fallback position in zone {self.start_zone}")
        
        return airsim.Vector3r(x, y, z)
    
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
        
        # Check ground distance (skip for first few armed steps to allow settling)
        if self.current_step >= self.safety_arm_steps:
            if ground_dist is not None and ground_dist < self.effective_ground_safety:
                return False, f"Too close to ground: {ground_dist:.2f}m"
        
        # Check horizontal obstacles (skip for first few armed steps)
        if self.current_step >= self.safety_arm_steps:
            if horizontal_dist is not None and horizontal_dist < self.effective_lidar_safety:
                return False, f"Obstacle too close: {horizontal_dist:.2f}m"
        
        # Check altitude limits relative to ground with small grace margin
        state = self.client.getMultirotorState()
        ned_z = state.kinematics_estimated.position.z_val  # NED Down (positive is down)
        # Prefer lidar ground distance if available, else derive from NED (above ground ~ max(0, -z))
        alt_above_ground = ground_dist if ground_dist is not None else max(0.0, -ned_z)
        grace = 0.3
        # Skip altitude enforcement for the first few steps as well
        if self.current_step >= self.safety_arm_steps:
            if alt_above_ground > self.effective_max_altitude + grace:
                return False, f"Too high: {alt_above_ground:.2f}m"
            elif alt_above_ground < self.effective_min_altitude - grace:
                return False, f"Too low: {alt_above_ground:.2f}m"
        
        return True, "Safe"
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the drone in AirSim
        self.client.reset()
        self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
        self.client.armDisarm(True, vehicle_name=self.vehicle_name)

        self.episode_count += 1
        # Curriculum: base linear growth + step increase every N episodes
        base = self.curriculum_start_dist + self.curriculum_growth_rate * (self.episode_count // self.curriculum_episodes_per_increase)
        max_dist = min(base, self.curriculum_max_dist)

        # Place start at a valid, safe, non-colliding position using zone-based generation
        start_found = False
        for attempt in range(50):
            candidate_start, zone_num = self.generate_random_position_from_zones()
            
            # Additional validation: check if position is reasonable
            if candidate_start is None:
                continue
                
            # Move vehicle to candidate position and check safety
            try:
                self.client.simSetVehiclePose(
                    airsim.Pose(
                        airsim.Vector3r(candidate_start.x_val, candidate_start.y_val, candidate_start.z_val),
                        airsim.Quaternionr()
                    ),
                    True
                )
                time.sleep(0.1)  # Give more time for physics to settle

                # Check for immediate collision
                collided = self.client.simGetCollisionInfo(self.vehicle_name).has_collided
                if collided:
                    if self.verbose:
                        print(f"[DEBUG] Attempt {attempt}: Collision detected at ({candidate_start.x_val:.2f}, {candidate_start.y_val:.2f}, {candidate_start.z_val:.2f})")
                    continue

                # Lidar-based clearance check
                gdist, hdist = self.get_lidar_data()
                ground_ok = (gdist is None) or (gdist >= self.effective_ground_safety)
                horiz_ok = (hdist is None) or (hdist >= self.effective_lidar_safety)
                
                # Enhanced open air check
                open_air = self._is_open_air(candidate_start, up_clearance_m=5.0)
                
                # Additional check: ensure we're not too close to terrain
                try:
                    terrain_height = self.client.simGetTerrainHeight(candidate_start.x_val, candidate_start.y_val)
                    if terrain_height is not None:
                        height_above_terrain = terrain_height - candidate_start.z_val
                        terrain_ok = height_above_terrain >= self.effective_min_altitude
                    else:
                        terrain_ok = True
                except Exception:
                    terrain_ok = True

                if ground_ok and horiz_ok and open_air and terrain_ok:
                    self.start_pos = candidate_start
                    self.start_zone = zone_num
                    start_found = True
                    if self.verbose:
                        print(f"[DEBUG] Valid start position found in zone {zone_num} after {attempt + 1} attempts")
                    break
                else:
                    if self.verbose:
                        print(f"[DEBUG] Attempt {attempt}: ground_ok={ground_ok}, horiz_ok={horiz_ok}, open_air={open_air}, terrain_ok={terrain_ok}")
                        
            except Exception as e:
                if self.verbose:
                    print(f"[DEBUG] Attempt {attempt} failed with error: {e}")
                continue
                
        if not start_found:
            if self.verbose:
                print("[WARNING] Failed to find valid start position, using fallback")
            self.start_pos, self.start_zone = self.generate_random_position_from_zones()

        # Use start to generate a reachable goal
        self.goal_pos = self.generate_goal_near_start(self.start_pos, max_dist)

        # Debug the generated positions using the debug_position function
        if self.verbose:
            print(f"\n[CURRICULUM] Episode {self.episode_count} - Max Distance: {max_dist:.1f}")
            print(f"[START] Zone {self.start_zone} - ({self.start_pos.x_val:.2f}, {self.start_pos.y_val:.2f}, {self.start_pos.z_val:.2f})")
            print(f"[GOAL]  ({self.goal_pos.x_val:.2f}, {self.goal_pos.y_val:.2f}, {self.goal_pos.z_val:.2f})")
            
            # Use debug_position function to verify start position
            self.debug_position(self.start_pos, f"START POSITION (Zone {self.start_zone})")
	
        # Move to start and takeoff to a safe altitude above ground before episode begins
        self.client.moveToPositionAsync(
            self.start_pos.x_val,
            self.start_pos.y_val,
            self.start_pos.z_val,
            5
        ).join()
        # Explicit takeoff to ensure lift-off from ground
        try:
            self.client.takeoffAsync(timeout_sec=10, vehicle_name=self.vehicle_name).join()
        except Exception:
            pass
        # Raise to at least min_altitude + buffer if needed
        try:
            state = self.client.getMultirotorState()
            # Prefer lidar ground estimate if available, else NED z
            gdist, _ = self.get_lidar_data()
            current_alt = gdist if gdist is not None else max(0.0, -state.kinematics_estimated.position.z_val)
            target_alt = max(self.effective_min_altitude + 0.5, current_alt)
            if target_alt > current_alt + 0.1:
                # moveToZ uses NED z (down positive), hence negative target for up
                self.client.moveToZAsync(-target_alt, 2.0, vehicle_name=self.vehicle_name).join()
        except Exception:
            pass
        time.sleep(0.3)
        self.client.hoverAsync().join()
        time.sleep(0.2)

        # Reset episode state
        self.current_step = 0
        self.warmup_steps = 2
        self.episode_reward = 0
        self.prev_dist = None
        self.collision_count = 0
        self.prev_yaw = None
        self.prev_altitude = None
        self._episode_step_trace = []

        time.sleep(0.1)
        obs = self.get_observation()
        info = {}
        return obs, info
    
    def step(self, action):
        """Take a step in the environment."""
        # During initial warmup steps, ignore safety termination to avoid instant end while stabilizing
        ignore_safety = self.current_step < self.warmup_steps
        # Fetch lidar once per step and cache
        ground_dist, horizontal_dist = self.get_lidar_data()
        self.last_lidar_ground_dist = ground_dist
        self.last_lidar_horizontal_dist = horizontal_dist

        # Apply (possibly overridden) action
        self.apply_action(action)
        # Faster control loop: avoid extra sleep here
        
        # Get observation
        obs = self.get_observation()
        
        # Check safety
        is_safe, safety_reason = self.check_safety()
        if ignore_safety and not is_safe:
            # treat as safe during warmup but still record reason for info
            is_safe = True
        
        # Get current yaw and altitude for penalties
        state = self.client.getMultirotorState()
        current_yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2] * 180 / np.pi
        current_altitude = -state.kinematics_estimated.position.z_val
        pos = state.kinematics_estimated.position
        
        # Debug first step
        if self.current_step == 0:
            pos = state.kinematics_estimated.position
            if self.verbose:
                print(f"[DEBUG] First step - Position: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
                print(f"[DEBUG] First step - Action: {action}, Safe: {is_safe}, Reason: {safety_reason}")
        
        # Compute reward
        reward = self.compute_reward(is_safe, safety_reason, action, current_yaw, current_altitude)
        self.episode_reward += reward
        
        # Update step counter
        self.current_step += 1
        
        # Check termination conditions
        terminated, truncated, info = self._check_termination()
        if self.current_step < self.warmup_steps:
            # Do not terminate during warmup steps
            terminated = False
            truncated = False
        
        # Add termination penalty for unsafe endings
        if terminated and not info.get('goal_reached', False):
            # Different penalties based on termination reason
            if info.get('collision', False):
                # Heavy penalty for collision termination
                reward -= 25.0
                if self.verbose:
                    print(f"[TERMINATION] Episode ended due to collision! Heavy penalty applied.")
            elif info.get('unsafe', False):
                # Penalty for unsafe conditions (too close to obstacles, altitude limits, etc.)
                reward -= 15.0
                if self.verbose:
                    print(f"[TERMINATION] Episode ended due to unsafe conditions! Penalty applied.")
            else:
                # General penalty for other unsafe terminations
                reward -= 20.0
                if self.verbose:
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
        
        # Optional trace logging at episode end and minimal step trace keeping
        try:
            step_entry = {
                'step': int(self.current_step),
                'action': int(action),
                'pos': (float(pos.x_val), float(pos.y_val), float(pos.z_val)),
                'yaw_deg': float(current_yaw),
                'alt_m': float(current_altitude),
                'dist_to_goal': float(info.get('distance_to_goal', 0.0)),
                'safe': bool(is_safe),
                'reason': safety_reason,
            }
            self._episode_step_trace.append(step_entry)
        except Exception:
            pass

        if (terminated or truncated) and (self.verbose or self.log_steps):
            to_print = min(10, len(self._episode_step_trace))
            print(f"[TRACE] Last {to_print} steps:")
            for entry in self._episode_step_trace[-to_print:]:
                print(f"  s={entry['step']:>3} a={entry['action']} pos=({entry['pos'][0]:.2f},{entry['pos'][1]:.2f},{entry['pos'][2]:.2f}) yaw={entry['yaw_deg']:.1f} alt={entry['alt_m']:.2f} d={entry['dist_to_goal']:.2f} safe={entry['safe']} reason={entry['reason']}")

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
        elif action == 1:  # left (rotate -15 deg)
            self.rotate_by(-15)
        elif action == 2:  # right (rotate +15 deg)
            self.rotate_by(15)
        elif action == 3:  # up
            # do not command below max altitude
            state = self.client.getMultirotorState()
            cur_alt = -state.kinematics_estimated.position.z_val
            if cur_alt < self.max_altitude:
                vz = -min(altitude_step, self.max_altitude - cur_alt)
        elif action == 4:  # down
            # do not command below min altitude
            state = self.client.getMultirotorState()
            cur_alt = -state.kinematics_estimated.position.z_val
            if cur_alt > self.min_altitude:
                vz = min(altitude_step, cur_alt - self.min_altitude)
        
        if vx != 0 or vy != 0 or vz != 0:
            self.client.moveByVelocityAsync(vx, vy, vz, 0.25).join()
    
    def rotate_by(self, delta_yaw):
        """Rotate the drone by the given angle."""
        self.client.rotateByYawRateAsync(delta_yaw, 0.25).join()
    
    def get_observation(self):
        """Get observation including depth camera and lidar data."""
        try:
            # Get depth camera image
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
            ])
            
            if responses and len(responses) > 0:
                depth_array = np.array(responses[0].image_data_float, dtype=np.float32)
                if depth_array.size == 0:
                    depth_image = np.zeros((48, 48, 1), dtype=np.float32)
                else:
                    depth = depth_array.reshape(responses[0].height, responses[0].width)
                    depth = np.clip(depth, 0.0, 100.0) / 100.0
                    depth_resized = cv2.resize(depth, (48, 48))
                    depth_image = depth_resized.reshape(48, 48, 1).astype(np.float32)
            else:
                depth_image = np.zeros((48, 48, 1), dtype=np.float32)
            
            # Get lidar data
            ground_dist = self.last_lidar_ground_dist
            horizontal_dist = self.last_lidar_horizontal_dist
            if ground_dist is None or horizontal_dist is None:
                ground_dist, horizontal_dist = self.get_lidar_data()
                self.last_lidar_ground_dist = ground_dist
                self.last_lidar_horizontal_dist = horizontal_dist

            # Get GPS data (latitude, longitude, altitude)
            gps_vec = np.zeros((3,), dtype=np.float32)
            try:
                gps_data = self.client.getGpsData(vehicle_name=self.vehicle_name)
                geo_point = None
                # Try common attribute chains
                if hasattr(gps_data, 'gnss') and hasattr(gps_data.gnss, 'geo_point'):
                    geo_point = gps_data.gnss.geo_point
                elif hasattr(gps_data, 'geo_point'):
                    geo_point = gps_data.geo_point
                if geo_point is not None and \
                   hasattr(geo_point, 'latitude') and hasattr(geo_point, 'longitude') and hasattr(geo_point, 'altitude'):
                    gps_vec[0] = float(geo_point.latitude)
                    gps_vec[1] = float(geo_point.longitude)
                    gps_vec[2] = float(geo_point.altitude)
                else:
                    # Fallback: use current local position as pseudo GPS
                    state = self.client.getMultirotorState()
                    pos = state.kinematics_estimated.position
                    gps_vec[0] = float(pos.x_val)
                    gps_vec[1] = float(pos.y_val)
                    gps_vec[2] = float(-pos.z_val)
            except Exception:
                # Fallback on any error: zeros vector is already set
                pass
            
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
            lidar_data = np.array([ground_dist / 100.0, horizontal_dist / 100.0], dtype=np.float32)
            
            # Return combined observation
            return {
                'depth_image': depth_image,
                'lidar_data': lidar_data,
                'gps': gps_vec
            }
                
        except Exception as e:
            print(f"[OBSERVATION ERROR] Error getting observation: {e}")
            return {
                'depth_image': np.zeros((48, 48, 1), dtype=np.float32),
                'lidar_data': np.array([1.0, 1.0], dtype=np.float32),
                'gps': np.zeros((3,), dtype=np.float32)
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
        
        # Soft proximity shaping for horizontal obstacles (prefer margin beyond safety distance)
        if self.last_lidar_horizontal_dist is not None:
            soft_margin = self.effective_lidar_safety + 1.0
            if self.last_lidar_horizontal_dist < soft_margin:
                reward -= 0.5 * (soft_margin - self.last_lidar_horizontal_dist)
        
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
        # Ignore collisions with configured object names (e.g., large ground planes)
        if collision and getattr(collision_info, 'object_name', None) in self.ignored_collision_objects:
            collision = False
        
        # Update collision count if collision occurred
        if collision:
            self.collision_count += 1
            if self.verbose:
                print(f"[COLLISION] Collision #{self.collision_count} detected!")
                print(f"[COLLISION] Position: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
            
            # Try to recover from collision by moving up
            if self.collision_count <= 3:  # Only try recovery for first few collisions
                if self.verbose:
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
            'collision_count': self.collision_count,
            'termination_reason': None,
            'collision_object': getattr(collision_info, 'object_name', None),
            'impact_point': (
                getattr(collision_info.impact_point, 'x_val', None),
                getattr(collision_info.impact_point, 'y_val', None),
                getattr(collision_info.impact_point, 'z_val', None),
            )
        }

        if goal_reached:
            info['termination_reason'] = 'goal_reached'
        elif collision:
            info['termination_reason'] = 'collision'
        elif not is_safe:
            info['termination_reason'] = 'unsafe_condition'
        elif self.current_step >= self.max_steps:
            info['termination_reason'] = 'time_limit'
        
        return terminated, truncated, info
    

    def debug_position(self, position: airsim.Vector3r, label: str = "Position"):
        """Debug helper to check if a position is valid and safe."""
        try:
            print(f"\n[DEBUG] {label}: ({position.x_val:.2f}, {position.y_val:.2f}, {position.z_val:.2f})")
            
            # Check terrain height
            try:
                terrain_height = self.client.simGetTerrainHeight(position.x_val, position.y_val)
                if terrain_height is not None:
                    height_above_terrain = terrain_height - position.z_val
                    print(f"[DEBUG] Terrain height: {terrain_height:.2f}m, Height above terrain: {height_above_terrain:.2f}m")
                    if height_above_terrain < self.min_altitude:
                        print(f"[DEBUG] WARNING: Too close to terrain! Need at least {self.min_altitude}m")
                else:
                    print("[DEBUG] Could not get terrain height")
            except Exception as e:
                print(f"[DEBUG] Terrain height query failed: {e}")
            
            # Analyze terrain around position
            terrain_analysis = self._analyze_terrain_around_position(position)
            print(f"[DEBUG] Terrain analysis: max={terrain_analysis['max_height']:.2f}m, min={terrain_analysis['min_height']:.2f}m, variance={terrain_analysis['height_variance']:.2f}")
            print(f"[DEBUG] Terrain safety: steep_slope={terrain_analysis['steep_slope']}, safe_landing={terrain_analysis['safe_landing']}")
            
            # Check if position is in free space
            is_free = self._is_position_free(position)
            print(f"[DEBUG] Position free: {is_free}")
            
            # Check open air
            open_air = self._is_open_air(position)
            print(f"[DEBUG] Open air: {open_air}")
            
            # Test collision by temporarily moving there
            try:
                original_pose = self.client.simGetVehiclePose(self.vehicle_name)
                self.client.simSetVehiclePose(
                    airsim.Pose(position, airsim.Quaternionr()),
                    True
                )
                time.sleep(0.1)
                
                collision_info = self.client.simGetCollisionInfo(self.vehicle_name)
                print(f"[DEBUG] Collision test: {collision_info.has_collided}")
                if collision_info.has_collided:
                    print(f"[DEBUG] Collision object: {getattr(collision_info, 'object_name', 'Unknown')}")
                
                # Restore original position
                self.client.simSetVehiclePose(original_pose, True)
                
            except Exception as e:
                print(f"[DEBUG] Collision test failed: {e}")
                
        except Exception as e:
            print(f"[DEBUG] Debug position failed: {e}")
    
    def print_zone_info(self):
        """Print information about all available safe zones."""
        print(f"\n[INFO] === SAFE ZONES FOR POSITION GENERATION ===")
        for i, zone in enumerate(self.safe_zones):
            print(f"  Zone {i+1} ({zone['name']}):")
            print(f"    X: {zone['x_range'][0]:.2f} to {zone['x_range'][1]:.2f}")
            print(f"    Y: {zone['y_range'][0]:.2f} to {zone['y_range'][1]:.2f}")
            print(f"    Z: {zone['z_range'][0]:.2f} to {zone['z_range'][1]:.2f}")
        print(f"===============================================\n")
    
    def close(self):
        """Clean up the environment."""
        try:
            self.client.armDisarm(False, vehicle_name=self.vehicle_name)
            self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
        except:
            pass

def main():
    """Test the environment with different safety leniency levels and zone-based positioning."""
    print("[INFO] Testing Mountain Pass Environment with Zone-Based Position Generation...")
    
    # Test different leniency levels
    leniency_levels = ["more", "normal", "less"]
    
    for leniency in leniency_levels:
        print(f"\n{'='*60}")
        print(f"[INFO] TESTING {leniency.upper()} LENIENCY LEVEL")
        print(f"{'='*60}")
        
        # Create environment with specific leniency
        env = MountainPassRandomGoalsEnv(
            verbose=True, 
            safety_leniency=leniency
        )
        
        # Display zone information
        env.print_zone_info()
        
        # Test one episode per leniency level
        print(f"\n[INFO] Testing reset with {leniency} leniency...")
        try:
            obs, info = env.reset()
            print(f"[SUCCESS] Reset completed with {leniency} leniency!")
            
            # Debug the generated positions
            if hasattr(env, 'start_pos') and env.start_pos:
                env.debug_position(env.start_pos, f"START ({leniency.upper()}) - Zone {env.start_zone}")
            if hasattr(env, 'goal_pos') and env.goal_pos:
                env.debug_position(env.goal_pos, f"GOAL ({leniency.upper()})")
            
            # Test a few steps
            print(f"[INFO] Testing steps with {leniency} leniency...")
            for i in range(3):  # Fewer steps for demonstration
                action = 0  # forward
                obs, reward, terminated, truncated, info = env.step(action)
                print(f"Step {i+1}: Reward={reward:.2f}, Distance={info.get('distance_to_goal', 0):.2f}, "
                      f"Lidar: Ground={obs['lidar_data'][0]:.1f}m, Horizontal={obs['lidar_data'][1]:.1f}m")
                
                if terminated or truncated:
                    print(f"Episode ended: terminated={terminated}, truncated={truncated}")
                    break
            
            print(f"[SUCCESS] {leniency.upper()} leniency test completed!")
            
        except Exception as e:
            print(f"[ERROR] Failed to test {leniency} leniency: {e}")
        
        finally:
            env.close()
    
    # Final recommendations
    print(f"\n{'='*60}")
    print(f"[INFO] FINAL TRAINING RECOMMENDATIONS")
    print(f"{'='*60}")
    print(f"[INFO] 🎯 Zone-Based Position Generation:")
    print(f"     • 4 predefined safe zones ensure valid spawn points")
    print(f"     • Random zone selection (1-4) for start positions")
    print(f"     • Goals prefer different zones for training variety")
    print(f"     • All zones validated for safety and accessibility")
    print(f"")
    print(f"[INFO] 🎯 Start with 'more' leniency for:")
    print(f"     • Initial exploration and learning")
    print(f"     • Faster progress in early training")
    print(f"     • Testing new reward functions")
    print(f"")
    print(f"[INFO] ⚖️  Switch to 'normal' leniency when:")
    print(f"     • Agent shows consistent progress")
    print(f"     • You want balanced safety/learning")
    print(f"     • Preparing for production-like conditions")
    print(f"")
    print(f"[INFO] 🚨 Use 'less' leniency for:")
    print(f"     • Final training phases")
    print(f"     • Production deployment")
    print(f"     • Real hardware testing")
    print(f"")
    print(f"[INFO] 💡 Progression strategy:")
    print(f"     • Phase 1: 'more' leniency (exploration)")
    print(f"     • Phase 2: 'normal' leniency (refinement)")
    print(f"     • Phase 3: 'less' leniency (production)")
    print(f"{'='*60}")
    
    print("[SUCCESS] All leniency level tests completed!")

if __name__ == "__main__":
    main() 