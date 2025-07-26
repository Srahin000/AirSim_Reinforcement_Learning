import gymnasium as gym
from gymnasium import spaces
import numpy as np
import airsim
import cv2
import time
import torch

class MountainPassEnv(gym.Env):
    """
    Custom Gym environment for AirSim drone navigation around a mountain using depth images.
    Action space: 0=forward, 1=left, 2=right, 3=up, 4=down
    Observation space: 84x84 depth image (single channel)
    """
    def __init__(self):
        super().__init__()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.action_space = spaces.Discrete(5)  # forward, left, right, up, down
        self.observation_space = spaces.Box(low=0, high=255, shape=(48, 48, 1), dtype=np.uint8)
        self.step_length = 4  # Reduced from 6 for better learning
        self.altitude_step = 2  # Reduced from 3 for better learning
        self.max_steps = 2000  # Increased from 200 to allow meaningful progress
        self.goal_pos = airsim.Vector3r(3230, 17200.0, -10.0)
        self.start_pos = airsim.Vector3r(0, 0, -10)
        self.current_step = 0
        self.episode_reward = 0
        self.prev_dist = None
        self.last_action_was_rotation = False  # Track if last action was a rotation
        print(f"[INFO] PyTorch CUDA available: {torch.cuda.is_available()}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.moveToPositionAsync(self.start_pos.x_val, self.start_pos.y_val, self.start_pos.z_val, 5).join()
        self.client.hoverAsync().join()
        self.current_step = 0
        self.episode_reward = 0
        self.prev_dist = None
        time.sleep(0.05)
        obs = self.get_observation()
        info = {}
        return obs, info

    def step(self, action):
        self.apply_action(action)
        time.sleep(0.05)
        obs = self.get_observation()
        reward = self.compute_reward()
        self.episode_reward += reward
        self.current_step += 1

        # Determine termination and truncation
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        dist = np.linalg.norm(np.array([pos.x_val, pos.y_val, pos.z_val]) -
                              np.array([self.goal_pos.x_val, self.goal_pos.y_val, self.goal_pos.z_val]))
        collision_info = self.client.simGetCollisionInfo()
        ignore_objects = ["Plane", "Plane9", "Ground"]
        collision = (
            collision_info.has_collided and
            collision_info.object_name not in ignore_objects
        )

        terminated = collision or dist < 5
        truncated = self.current_step >= self.max_steps

        info = {}
        if terminated or truncated:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.current_step
            }

        return obs, reward, terminated, truncated, info

    def apply_action(self, action):
        # 0: forward, 1: left, 2: right, 3: up, 4: down
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2] * 180 / np.pi
        vx, vy, vz = 0, 0, 0
        self.last_action_was_rotation = False
        if action == 0:  # forward
            vx = self.step_length * np.cos(np.deg2rad(yaw))
            vy = self.step_length * np.sin(np.deg2rad(yaw))
        elif action == 1:  # left (rotate -30 deg)
            self.rotate_by(-30)
            self.last_action_was_rotation = True
        elif action == 2:  # right (rotate +30 deg)
            self.rotate_by(30)
            self.last_action_was_rotation = True
        elif action == 3:  # up
            vz = -self.altitude_step
        elif action == 4:  # down
            vz = self.altitude_step
        # Move if not a rotation
        if action == 0 or action == 3 or action == 4:
            self.client.moveByVelocityAsync(vx, vy, vz, 1).join()

    def rotate_by(self, delta_yaw):
        state = self.client.getMultirotorState()
        yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2] * 180 / np.pi
        new_yaw = yaw + delta_yaw
        self.client.rotateToYawAsync(new_yaw, 5).join()

    def get_observation(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)
        ])
        img1d = np.array(responses[0].image_data_float, dtype=np.float32)
        if img1d.size == 0:
            # fallback to zeros if image is not received
            img2d = np.zeros((48, 48), dtype=np.float32)
        else:
            img2d = img1d.reshape(responses[0].height, responses[0].width)
            img2d = np.clip(img2d, 0, 100)
            img2d = img2d / 100.0  # normalize to [0,1]
            img2d = cv2.resize(img2d, (48, 48), interpolation=cv2.INTER_AREA)
        img2d = (img2d * 255).astype(np.uint8)
        img2d = np.expand_dims(img2d, axis=-1)  # (48, 48, 1)
        return img2d

    def compute_reward(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        dist = np.linalg.norm(np.array([pos.x_val, pos.y_val, pos.z_val]) -
                              np.array([self.goal_pos.x_val, self.goal_pos.y_val, self.goal_pos.z_val]))
        reward = 0
        if self.prev_dist is not None:
            progress = self.prev_dist - dist
            if progress > 0:
                reward += 5  # reward for getting closer
            else:
                reward -= 2  # penalty for moving away
        self.prev_dist = dist
        # Penalty for collision
        collision_info = self.client.simGetCollisionInfo()
        ignore_objects = ["Plane", "Plane9", "Ground"]
        if collision_info.has_collided and collision_info.object_name not in ignore_objects:
            reward -= 20
        # Penalty for each step
        reward -= 0.1
        # Penalty for rotation
        if self.last_action_was_rotation:
            reward -= 1.0  # Penalize yaw/rotation
        return reward

    def is_done(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        dist = np.linalg.norm(np.array([pos.x_val, pos.y_val, pos.z_val]) -
                              np.array([self.goal_pos.x_val, self.goal_pos.y_val, self.goal_pos.z_val]))
        collision = self.client.simGetCollisionInfo().has_collided
        
        # Always end episode after max_steps to prevent hanging
        if self.current_step >= self.max_steps:
            print(f"[DEBUG] Episode ended after {self.max_steps} steps (timeout)")
            return True
        elif collision:
            print(f"[DEBUG] Episode ended due to collision at step {self.current_step}")
            return True
        elif dist < 5:
            print(f"[DEBUG] Episode ended - goal reached at step {self.current_step}")
            return True
        
        return False 