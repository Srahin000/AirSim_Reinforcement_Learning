#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import setup_path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from collections import deque
import random
import time
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import os
import argparse

# Import the random goals environment
from mountain_env_random_goals import MountainPassRandomGoalsEnv

class PolicyNetwork(nn.Module):
    def __init__(self, input_channels=1, num_actions=5):
        super(PolicyNetwork, self).__init__()
        
        # CNN for processing depth images
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # Calculate the size after convolutions
        # Input: 48x48 -> 24x24 -> 12x12 -> 6x6
        conv_output_size = 256 * 6 * 6
        
        # Lidar data processing
        self.lidar_fc = nn.Linear(2, 64)
        
        # Combined features
        combined_size = conv_output_size + 64
        self.fc1 = nn.Linear(combined_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_actions)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, depth_image, lidar_data):
        # Process depth image
        x = F.relu(self.conv1(depth_image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(x.size(0), -1)  # Flatten using reshape instead of view
        
        # Process lidar data
        lidar_features = F.relu(self.lidar_fc(lidar_data))
        
        # Combine features
        combined = torch.cat([x, lidar_features], dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.softmax(x, dim=1)

class ValueNetwork(nn.Module):
    def __init__(self, input_channels=1):
        super(ValueNetwork, self).__init__()
        
        # CNN for processing depth images
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        # Calculate the size after convolutions
        conv_output_size = 256 * 6 * 6
        
        # Lidar data processing
        self.lidar_fc = nn.Linear(2, 64)
        
        # Combined features
        combined_size = conv_output_size + 64
        self.fc1 = nn.Linear(combined_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, depth_image, lidar_data):
        # Process depth image
        x = F.relu(self.conv1(depth_image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(x.size(0), -1)  # Flatten using reshape instead of view
        
        # Process lidar data
        lidar_features = F.relu(self.lidar_fc(lidar_data))
        
        # Combine features
        combined = torch.cat([x, lidar_features], dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, c1=1, c2=0.01):
        self.policy_net = PolicyNetwork()
        self.value_net = ValueNetwork()
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.value_net.to(self.device)
        
        print(f"[INFO] Using device: {self.device}")
    
    def preprocess_observation(self, obs):
        """Preprocess observation for neural network input."""
        # Process depth image - correct order: (H, W, C) -> (C, H, W)
        depth_image = torch.FloatTensor(obs['depth_image']).permute(2, 0, 1)  # (C, H, W)
        
        # Process lidar data
        lidar_data = torch.FloatTensor(obs['lidar_data'])
        
        return depth_image.to(self.device), lidar_data.to(self.device)
    
    def select_action(self, obs):
        """Select action using the policy network."""
        depth_image, lidar_data = self.preprocess_observation(obs)
        
        # Add batch dimension for single observation
        depth_image = depth_image.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
        lidar_data = lidar_data.unsqueeze(0)    # (2,) -> (1, 2)
        
        with torch.no_grad():
            action_probs = self.policy_net(depth_image, lidar_data)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def update(self, states, actions, rewards, log_probs, values, dones):
        """Update the policy and value networks using PPO."""
        # Convert to tensors
        states_depth = torch.stack([s[0] for s in states]).to(self.device)  # (B, C, H, W)
        states_lidar = torch.stack([s[1] for s in states]).to(self.device)  # (B, 2)
        
        # Debug shape check
        print(f"[DEBUG] depth shape: {states_depth.shape}")  # Should be [B, 1, 48, 48]
        print(f"[DEBUG] lidar shape: {states_lidar.shape}")  # Should be [B, 2]
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Compute returns
        returns = self.compute_returns(rewards, values, dones)
        
        # Compute advantages
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(10):  # Multiple epochs
            # Get current policy probabilities
            action_probs = self.policy_net(states_depth, states_lidar)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            
            # Policy loss
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            new_values = self.value_net(states_depth, states_lidar).squeeze()
            value_loss = F.mse_loss(new_values, returns)
            
            # Total loss
            loss = policy_loss + self.c1 * value_loss
            
            # Update networks
            self.optimizer_policy.zero_grad()
            self.optimizer_value.zero_grad()
            loss.backward()
            self.optimizer_policy.step()
            self.optimizer_value.step()
    
    def compute_returns(self, rewards, values, dones):
        """Compute returns using GAE."""
        returns = torch.zeros_like(rewards)
        next_value = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                next_value = 0
            returns[t] = rewards[t] + self.gamma * next_value
            next_value = returns[t]
        
        return returns
    
    def get_value(self, obs):
        """Get value estimate for the current observation."""
        depth_image, lidar_data = self.preprocess_observation(obs)
        
        # Add batch dimension for single observation
        depth_image = depth_image.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
        lidar_data = lidar_data.unsqueeze(0)    # (2,) -> (1, 2)
        
        with torch.no_grad():
            value = self.value_net(depth_image, lidar_data)
        
        return value.item()
    
    def save_model(self, path):
        """Save the model."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'optimizer_policy_state_dict': self.optimizer_policy.state_dict(),
            'optimizer_value_state_dict': self.optimizer_value.state_dict(),
        }, path)
    
    def load_model(self, path):
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.optimizer_policy.load_state_dict(checkpoint['optimizer_policy_state_dict'])
        self.optimizer_value.load_state_dict(checkpoint['optimizer_value_state_dict'])

def train_random_goals(verbose=True, verbose_step_interval=10):
    """Train the agent on randomized goals."""
    print("[INFO] Starting training on randomized goals...")
    
    # Create environment with verbose logging
    env = MountainPassRandomGoalsEnv(verbose=verbose, verbose_step_interval=verbose_step_interval)  # Enable detailed step-by-step logging
    
    # Create agent
    agent = PPOAgent(state_dim=None, action_dim=5)  # State dim not needed for this implementation
    
    # Training parameters
    num_episodes = 1000
    max_steps_per_episode = 500  # Increased from 200 to 500 for longer episodes
    save_interval = 100
    log_interval = 10
    
    # Create tensorboard writer
    log_dir = "ppo_random_goals_tensorboard"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    goal_reached_count = 0
    collision_count = 0
    
    print(f"[INFO] Training for {num_episodes} episodes...")
    print(f"[INFO] Logging to: {log_dir}")
    
    for episode in range(num_episodes):
        print(f"\n[EPISODE {episode + 1}/{num_episodes}]")
        
        # Reset environment (this will generate a new random goal)
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Episode data for PPO update
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []
        
        for step in range(max_steps_per_episode):
            # Select action
            action, log_prob = agent.select_action(obs)
            value = agent.get_value(obs)
            
            # Store state and action
            depth_image, lidar_data = agent.preprocess_observation(obs)
            states.append((depth_image, lidar_data))
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Store reward and done flag
            rewards.append(reward)
            dones.append(terminated or truncated)
            
            episode_reward += reward
            episode_length += 1
            
            # Check if goal was reached (but don't break episode)
            if info.get('goal_reached', False):
                goal_reached_count += 1
                print(f"[GOAL REACHED] Episode {episode + 1} reached goal #{goal_reached_count} in {step + 1} steps!")
                # Don't break - continue episode with new goal
            
            # Check for collision
            if info.get('collision', False):
                collision_count += 1
                print(f"[COLLISION] Episode {episode + 1} had collision!")
            
                    # Check if episode should terminate
            if terminated or truncated:
                break
        
        # Episode summary
        if terminated or truncated:
            print(f"\n[EPISODE {episode + 1} SUMMARY]")
            print(f"  Total Steps: {episode_length}")
            print(f"  Final Reward: {episode_reward:.3f}")
            print(f"  Goals Reached: {goal_reached_count}")
            print(f"  Collisions: {collision_count}")
            if terminated:
                print(f"  Termination Reason: {info.get('termination_reason', 'Unknown')}")
            elif truncated:
                print(f"  Termination Reason: Max steps reached")
            print()
        
        # Update agent if we have enough data
        if len(states) > 0:
            agent.update(states, actions, rewards, log_probs, values, dones)
        
        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Log to tensorboard
        writer.add_scalar('Episode/Reward', episode_reward, episode)
        writer.add_scalar('Episode/Length', episode_length, episode)
        writer.add_scalar('Episode/GoalsReached', goal_reached_count, episode)  # Track total goals reached
        writer.add_scalar('Episode/Collision', 1 if info.get('collision', False) else 0, episode)
        
        # Print progress
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_length = np.mean(episode_lengths[-log_interval:])
            goals_per_episode = goal_reached_count / (episode + 1)
            collision_rate = collision_count / (episode + 1)
            
            print(f"[PROGRESS] Episode {episode + 1}")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Average Length: {avg_length:.1f}")
            print(f"  Goals per Episode: {goals_per_episode:.2f}")
            print(f"  Collision Rate: {collision_rate:.2%}")
        
        # Save model periodically
        if (episode + 1) % save_interval == 0:
            model_path = f"ppo_random_goals_model_episode_{episode + 1}.pth"
            agent.save_model(model_path)
            print(f"[SAVE] Model saved to {model_path}")
    
    # Final save
    agent.save_model("ppo_random_goals_model_final.pth")
    print("[SAVE] Final model saved!")
    
    # Close environment and writer
    env.close()
    writer.close()
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 3, 2)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.subplot(1, 3, 3)
    goals_per_episode = [goal_reached_count / (i + 1) for i in range(num_episodes)]
    plt.plot(goals_per_episode)
    plt.title('Goals per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Goals per Episode')
    
    plt.tight_layout()
    plt.savefig('training_curves_random_goals.png')
    plt.show()
    
    print("[SUCCESS] Training completed!")
    print(f"Final Goals per Episode: {goal_reached_count / num_episodes:.2f}")
    print(f"Final Collision Rate: {collision_count / num_episodes:.2%}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train PPO agent on randomized goals')
    parser.add_argument('--verbose', action='store_true', default=True, 
                       help='Enable verbose logging (default: True)')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false',
                       help='Disable verbose logging')
    parser.add_argument('--verbose-step-interval', type=int, default=10,
                       help='Only show detailed logs every N steps (default: 10)')
    
    args = parser.parse_args()
    
    print(f"[CONFIG] Verbose: {args.verbose}")
    print(f"[CONFIG] Verbose Step Interval: {args.verbose_step_interval}")
    
    train_random_goals(verbose=args.verbose, verbose_step_interval=args.verbose_step_interval)