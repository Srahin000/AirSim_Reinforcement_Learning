#!/usr/bin/env python3
"""
Clean Training Script for Mountain Pass Environment with Lidar
Simplified training with PPO and proper logging.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import setup_path

from stable_baselines3 import PPO
from mountain_env_clean import MountainPassEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
import os
import time

def make_env():
    """Create the environment with monitoring."""
    return Monitor(MountainPassEnv(
        max_steps=500,
        step_length=3.0,  # Reduced step length for more conservative movement
        altitude_step=2.0,  # Reduced altitude step
        lidar_safety_distance=2.0,
        ground_safety_distance=1.5,
        max_altitude=40.0,
        min_altitude=1.0,
        hard_reset_on_collision=False  # Don't hard reset on collision
    ))

import numpy as np

class TrainingCallback(BaseCallback):
    """Custom callback for training monitoring with detailed episode logging."""
    def __init__(self):
        super().__init__()
        self.episode_count = 0
        self.total_reward = 0
        self.total_length = 0
        self.collision_count = 0
        self.goal_reached_count = 0
        self.unsafe_count = 0
        self.total_collisions_in_episodes = 0  # Track total collisions across episodes
        self.recent_rewards = []
        self.recent_lengths = []
        
    def _on_step(self):
        infos = self.locals["infos"]
        
        for info in infos:
            ep = info.get("episode")
            if ep:
                self.episode_count += 1
                episode_reward = ep["r"]
                episode_length = ep["l"]
                
                self.total_reward += episode_reward
                self.total_length += episode_length
                
                # Keep recent history for better statistics
                self.recent_rewards.append(episode_reward)
                self.recent_lengths.append(episode_length)
                if len(self.recent_rewards) > 50:  # Keep last 50 episodes
                    self.recent_rewards.pop(0)
                    self.recent_lengths.pop(0)
                
                # Track outcomes
                if info.get('collision', False):
                    self.collision_count += 1
                
                # Track collision count from episode
                episode_collisions = info.get('collision_count', 0)
                self.total_collisions_in_episodes += episode_collisions
                
                if info.get('goal_reached', False):
                    self.goal_reached_count += 1
                
                if info.get('unsafe', False):
                    self.unsafe_count += 1
                
                # Log EVERY episode with detailed information
                episode_collisions = info.get('collision_count', 0)
                print(f"[EPISODE {self.episode_count:4d}] "
                      f"Reward: {episode_reward:7.2f}, "
                      f"Length: {episode_length:3d}, "
                      f"Distance: {info.get('distance_to_goal', 0):6.1f}, "
                      f"Collisions: {episode_collisions}, "
                      f"Status: {self._get_episode_status(info)}")
                
                # Log lidar information if available
                lidar_ground = info.get('lidar_ground_dist', None)
                lidar_horizontal = info.get('lidar_horizontal_dist', None)
                if lidar_ground is not None or lidar_horizontal is not None:
                    ground_str = f"{lidar_ground:5.1f}" if lidar_ground is not None else "None "
                    horizontal_str = f"{lidar_horizontal:5.1f}" if lidar_horizontal is not None else "None "
                    print(f"           Lidar - Ground: {ground_str}, Horizontal: {horizontal_str}")
                
                # Log running statistics every 10 episodes
                if self.episode_count % 10 == 0:
                    self._log_running_statistics()
        
        return True
    
    def _get_episode_status(self, info):
        """Get episode status string."""
        status = []
        if info.get('collision', False):
            status.append("COLLISION")
        if info.get('unsafe', False):
            status.append("UNSAFE")
        if info.get('goal_reached', False):
            status.append("GOAL")
        if not status:
            status.append("TIMEOUT")
        return " | ".join(status)
    
    def _log_running_statistics(self):
        """Log running training statistics."""
        avg_reward = self.total_reward / self.episode_count
        avg_length = self.total_length / self.episode_count
        collision_rate = self.collision_count / self.episode_count * 100
        goal_rate = self.goal_reached_count / self.episode_count * 100
        unsafe_rate = self.unsafe_count / self.episode_count * 100
        avg_collisions_per_episode = self.total_collisions_in_episodes / self.episode_count
        
        # Recent statistics (last 50 episodes)
        recent_avg_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0
        recent_avg_length = np.mean(self.recent_lengths) if self.recent_lengths else 0
        
        print("\n" + "=" * 80)
        print(f"RUNNING STATISTICS (Episode {self.episode_count})")
        print("=" * 80)
        print(f"Overall Statistics:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Length: {avg_length:.1f}")
        print(f"  Collision Rate: {collision_rate:.1f}%")
        print(f"  Goal Rate: {goal_rate:.1f}%")
        print(f"  Unsafe Rate: {unsafe_rate:.1f}%")
        print(f"  Avg Collisions per Episode: {avg_collisions_per_episode:.2f}")
        print(f"\nRecent Statistics (Last {len(self.recent_rewards)} episodes):")
        print(f"  Recent Avg Reward: {recent_avg_reward:.2f}")
        print(f"  Recent Avg Length: {recent_avg_length:.1f}")
        print(f"  Recent Goal Rate: {(sum(1 for r in self.recent_rewards if r > 0) / len(self.recent_rewards) * 100):.1f}%")
        print("=" * 80)
        
        # Log to tensorboard
        self.logger.record("episode/avg_reward", avg_reward)
        self.logger.record("episode/avg_length", avg_length)
        self.logger.record("episode/collision_rate", collision_rate)
        self.logger.record("episode/goal_rate", goal_rate)
        self.logger.record("episode/unsafe_rate", unsafe_rate)
        self.logger.record("episode/avg_collisions_per_episode", avg_collisions_per_episode)
        self.logger.record("episode/recent_avg_reward", recent_avg_reward)
        self.logger.record("episode/recent_avg_length", recent_avg_length)
        self.logger.dump(self.num_timesteps)

class CheckpointCallback(BaseCallback):
    """Callback to save model checkpoints."""
    def __init__(self, save_freq=5000, save_path="./checkpoints/", name_prefix="mountain_model"):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.last_save_step = 0
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self):
        current_step = self.num_timesteps
        
        if current_step >= self.last_save_step + self.save_freq:
            save_path = os.path.join(self.save_path, f"{self.name_prefix}_{current_step}_steps")
            self.model.save(save_path)
            self.last_save_step = current_step
            print(f"[CHECKPOINT] Model saved at {current_step} steps: {save_path}")
            
            # Also save a "latest" checkpoint for easy loading
            latest_path = os.path.join(self.save_path, f"{self.name_prefix}_latest")
            self.model.save(latest_path)
            print(f"[CHECKPOINT] Latest checkpoint saved: {latest_path}")
        
        return True

def main():
    """Main training function."""
    print("[INFO] Starting clean training setup...")
    
    # Test environment first
    print("[INFO] Testing environment...")
    test_env = make_env()
    obs, info = test_env.reset()
    print(f"[INFO] Environment reset successful!")
    print(f"[INFO] Observation keys: {obs.keys()}")
    print(f"[INFO] Depth image shape: {obs['depth_image'].shape}")
    print(f"[INFO] Lidar data shape: {obs['lidar_data'].shape}")
    
    # Test a few steps
    for i in range(5):
        action = 0  # forward
        obs, reward, terminated, truncated, info = test_env.step(action)
        print(f"[TEST] Step {i+1}: Reward={reward:.2f}, Distance={info.get('distance_to_goal', 0):.2f}, "
              f"Collisions={info.get('collision_count', 0)}")
        if terminated or truncated:
            break
    
    test_env.close()
    print("[INFO] Environment test completed.")
    
    # Create environment for training
    env = DummyVecEnv([make_env])
    
    # Model path
    model_path = "checkpoints/mountain_clean_model"
    latest_model_path = "checkpoints/mountain_clean_model_latest"
    
    # Set up logger with enhanced PPO logging
    new_logger = configure("ppo_clean_tensorboard/PPO_8", ["tensorboard", "stdout"])
    
    # Resume training if model exists, otherwise start fresh
    print(f"[DEBUG] Checking for checkpoints...")
    print(f"[DEBUG] Latest path: {latest_model_path}.zip")
    print(f"[DEBUG] Main path: {model_path}.zip")
    print(f"[DEBUG] Latest exists: {os.path.exists(latest_model_path + '.zip')}")
    print(f"[DEBUG] Main exists: {os.path.exists(model_path + '.zip')}")
    
    if os.path.exists(latest_model_path + ".zip"):
        print(f"[INFO] Loading latest checkpoint from {latest_model_path}.zip and resuming training...")
        model = PPO.load(latest_model_path, env=env)
        print(f"[INFO] Latest model loaded successfully! Resuming from checkpoint.")
        # Get the timesteps from the loaded model to continue from where we left off
        loaded_timesteps = model.num_timesteps
        print(f"[INFO] Resuming from {loaded_timesteps} timesteps")
    elif os.path.exists(model_path + ".zip"):
        print(f"[INFO] Loading existing model from {model_path}.zip and resuming training...")
        model = PPO.load(model_path, env=env)
        print(f"[INFO] Model loaded successfully! Resuming from checkpoint.")
        loaded_timesteps = model.num_timesteps
        print(f"[INFO] Resuming from {loaded_timesteps} timesteps")
    else:
        print("[INFO] Starting fresh training...")
        loaded_timesteps = 0
        # Create custom policy for dictionary observations
        from stable_baselines3.common.policies import ActorCriticPolicy
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        import torch
        import torch.nn as nn
        
        class LidarFeaturesExtractor(BaseFeaturesExtractor):
            def __init__(self, observation_space, features_dim=128):
                super().__init__(observation_space, features_dim)
                
                # CNN for depth image
                self.cnn = nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Flatten()
                )
                
                # Calculate CNN output size
                cnn_output_size = 64 * 12 * 12  # 48x48 -> 12x12 after 2 stride-2 convs
                
                # MLP for lidar data
                self.lidar_mlp = nn.Sequential(
                    nn.Linear(2, 32),  # 2 lidar values
                    nn.ReLU(),
                    nn.Linear(32, 32),
                    nn.ReLU()
                )
                
                # Combined features
                self.combined_mlp = nn.Sequential(
                    nn.Linear(cnn_output_size + 32, features_dim),
                    nn.ReLU()
                )
            
            def forward(self, observations):
                # Extract depth image and lidar data
                depth_image = observations['depth_image']
                lidar_data = observations['lidar_data']
                
                # Process depth image through CNN
                cnn_features = self.cnn(depth_image)
                
                # Process lidar data through MLP
                lidar_features = self.lidar_mlp(lidar_data)
                
                # Combine features
                combined_features = torch.cat([cnn_features, lidar_features], dim=1)
                
                return self.combined_mlp(combined_features)
        
        model = PPO(
            ActorCriticPolicy, 
            env, 
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="./ppo_clean_tensorboard/",
            policy_kwargs=dict(
                features_extractor_class=LidarFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=128)
            )
        )
    
    model.set_logger(new_logger)
    
    # Create callbacks
    training_callback = TrainingCallback()
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./checkpoints/', name_prefix='mountain_clean_model')
    
    # Combine callbacks
    callback = CallbackList([training_callback, checkpoint_callback])
    
    print("[INFO] Starting training...")
    print("[INFO] Goal: Navigate to (45.78, 114.50, -19.35) with lidar safety")
    print("[INFO] Actions: forward, left, right, up, down")
    print("[INFO] Checkpoints will be saved every 1000 steps in ./checkpoints/")
    print("[INFO] TensorBoard logs available at ./ppo_clean_tensorboard/")
    
    # Calculate remaining timesteps to train
    remaining_timesteps = 100_000 - loaded_timesteps
    print(f"[INFO] Training for {remaining_timesteps} more timesteps (total: {loaded_timesteps + remaining_timesteps})")
    
    # Train the model
    model.learn(total_timesteps=remaining_timesteps, callback=callback, progress_bar=True)
    
    # Save the final model
    model.save(model_path)
    print(f"[INFO] Training completed! Model saved to {model_path}")
    
    # Test the trained model
    print("\n[INFO] Testing the trained model...")
    obs = env.reset()
    total_reward = 0
    
    for i in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        total_reward += rewards[0]
        
        if dones[0]:
            print(f"Episode ended after {i+1} steps with total reward: {total_reward:.2f}")
            break
    
    print("[INFO] Training and testing completed!")

if __name__ == "__main__":
    main() 