import gym
from gym import spaces
import numpy as np
import airsim
import cv2
import time

class MountainPassEnv(gym.Env):
    """
    Custom Gym environment for AirSim drone navigation around a mountain using depth images.
    Action space: 0=forward, 1=left, 2=right, 3=up, 4=down
    Observation space: 84x84 depth image (single channel)
    """
    def __init__(self):
        super(MountainPassEnv, self).__init__()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.action_space = spaces.Discrete(5)  # Forward, Left, Right, Up, Down
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)  # Depth image
        self.step_length = 3  # meters
        self.altitude_step = 2  # meters
        self.max_steps = 200
        self.goal_pos = airsim.Vector3r(3230, 17200, -10)  # Example goal
        self.start_pos = airsim.Vector3r(0, 0, -10)
        self.current_step = 0

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.moveToPositionAsync(self.start_pos.x_val, self.start_pos.y_val, self.start_pos.z_val, 5).join()
        self.client.hoverAsync().join()
        self.current_step = 0
        time.sleep(1)
        return self.get_observation()

    def step(self, action):
        self.apply_action(action)
        time.sleep(0.5)
        obs = self.get_observation()
        reward = self.compute_reward()
        done = self.is_done()
        self.current_step += 1
        return obs, reward, done, {}

    def apply_action(self, action):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        if action == 0:  # Forward (X+)
            new_pos = airsim.Vector3r(pos.x_val + self.step_length, pos.y_val, pos.z_val)
        elif action == 1:  # Left (Y+)
            new_pos = airsim.Vector3r(pos.x_val, pos.y_val + self.step_length, pos.z_val)
        elif action == 2:  # Right (Y-)
            new_pos = airsim.Vector3r(pos.x_val, pos.y_val - self.step_length, pos.z_val)
        elif action == 3:  # Up (Z-)
            new_pos = airsim.Vector3r(pos.x_val, pos.y_val, pos.z_val - self.altitude_step)
        elif action == 4:  # Down (Z+)
            new_pos = airsim.Vector3r(pos.x_val, pos.y_val, pos.z_val + self.altitude_step)
        else:
            new_pos = pos
        self.client.moveToPositionAsync(new_pos.x_val, new_pos.y_val, new_pos.z_val, 3).join()
        self.client.hoverAsync().join()

    def get_observation(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)
        ])
        response = responses[0]
        img1d = np.array(response.image_data_float, dtype=np.float32)
        img2d = img1d.reshape(response.height, response.width)
        img2d = np.clip(img2d, 0, 100)
        img2d = (img2d / 100 * 255).astype(np.uint8)
        img2d = cv2.resize(img2d, (84, 84), interpolation=cv2.INTER_AREA)
        img2d = np.expand_dims(img2d, axis=-1)  # Shape: (84, 84, 1)
        return img2d

    def compute_reward(self):
        # Reward: +10 for getting closer to goal, -100 for collision, -1 per step
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        dist = np.linalg.norm(np.array([pos.x_val, pos.y_val, pos.z_val]) -
                              np.array([self.goal_pos.x_val, self.goal_pos.y_val, self.goal_pos.z_val]))
        collision = self.client.simGetCollisionInfo().has_collided
        if collision:
            return -100
        elif dist < 5:
            return 100  # Goal reached
        else:
            return -1 - 0.1 * dist  # Small penalty per step, more for being far

    def is_done(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        dist = np.linalg.norm(np.array([pos.x_val, pos.y_val, pos.z_val]) -
                              np.array([self.goal_pos.x_val, self.goal_pos.y_val, self.goal_pos.z_val]))
        collision = self.client.simGetCollisionInfo().has_collided
        if collision or dist < 5 or self.current_step >= self.max_steps:
            return True
        return False 