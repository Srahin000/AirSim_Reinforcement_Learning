"""
Training script for Mountain Pass Random Goals environment using Stable-Baselines3 PPO.

SAFETY LENIENCY OPTIONS:
- "more": More lenient safety checks (faster learning, less safe)
- "normal": Balanced safety checks (recommended starting point)
- "less": Stricter safety checks (safer, slower learning)

TRAINING PROGRESSION RECOMMENDATION:
1. Start with "more" leniency for initial exploration
2. Switch to "normal" when agent shows progress
3. Use "less" for final training/production
"""

#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import setup_path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
from typing import Optional
import numpy as np
import glob
import shutil
import argparse

# Import the random goals environment
from mountain_env_random_goals import MountainPassRandomGoalsEnv

def make_env():
    """Create the environment with monitoring."""
    return Monitor(MountainPassRandomGoalsEnv(
        max_steps=500,
        step_length= 4.0,
        altitude_step= 2.0,
        lidar_safety_distance = 0.25,
        ground_safety_distance= 0.25,
        max_altitude= 50.0,
        min_altitude= 1.0,
        hard_reset_on_collision=True,
        safety_arm_steps=8,

        safety_leniency="normal",  # Options: "more" (lenient), "normal" (default), "less" (strict)
        verbose=True  # Enable verbose mode to see zone information
    ))

class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Monitor wrapper stores infos['episode'] when done
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        if infos and dones is not None:
            for i, done in enumerate(dones):
                if done and isinstance(infos[i], dict):
                    self.episode_count += 1
                    reason = infos[i].get('termination_reason')
                    unsafe = infos[i].get('unsafe')
                    collision = infos[i].get('collision')
                    obj = infos[i].get('collision_object')
                    impact = infos[i].get('impact_point')
                    dist = infos[i].get('distance_to_goal')
                    safereason = infos[i].get('safety_reason')
                    lidar_g = infos[i].get('lidar_ground_dist')
                    lidar_h = infos[i].get('lidar_horizontal_dist')
                    ep = infos[i].get('episode', {})
                    r = ep.get('r')
                    l = ep.get('l')
                    print(f"[EP {self.episode_count}] end: steps={l}, return={r:.2f}, reason={reason}, unsafe={unsafe}, safety_reason={safereason}, lidar_g={lidar_g}, lidar_h={lidar_h}, collision={collision}, obj={obj}, impact={impact}, dist={dist:.2f}")
        return True

def main():
    """Train the agent on randomized goals using Stable-Baselines3."""
    print("[INFO] Starting training on randomized goals with Stable-Baselines3...")
    
    # Create environment
    env = DummyVecEnv([lambda: make_env()])
    
    # Training parameters
    total_timesteps = args.timesteps
    save_interval = args.save_interval
    
    # Prepare checkpoint directories
    checkpoint_dirs = ["checkpoints_random", "ppo_random_goals"]
    for d in checkpoint_dirs:
        os.makedirs(d, exist_ok=True)

    # Helpers to locate and manage checkpoints
    def find_existing_checkpoint() -> Optional[str]:
        """Prefer an explicit 'latest' alias; else fall back to most recent *.zip across dirs."""
        base_name = "mountain_random_goals_model_latest"
        # Prefer explicit latest
        for d in checkpoint_dirs:
            candidate = os.path.join(d, base_name)
            if os.path.exists(candidate + ".zip"):
                return candidate
        # Fallback: pick newest zip across both dirs
        newest_path: Optional[str] = None
        newest_mtime: float = -1.0
        for d in checkpoint_dirs:
            for path in glob.glob(os.path.join(d, "*.zip")):
                try:
                    mtime = os.path.getmtime(path)
                    if mtime > newest_mtime:
                        newest_mtime = mtime
                        newest_path = os.path.splitext(path)[0]
                except Exception:
                    continue
        return newest_path

    def copy_as_latest(dir_path: str, source_base: str) -> None:
        """Copy source_base.zip within dir_path to 'mountain_random_goals_model_latest.zip'."""
        try:
            src_zip = source_base + ".zip"
            if not os.path.isabs(src_zip):
                # If given as base with dir embedded, respect it; else join
                if not src_zip.startswith(dir_path):
                    src_zip = os.path.join(dir_path, os.path.basename(src_zip))
            if os.path.exists(src_zip):
                dest_zip = os.path.join(dir_path, "mountain_random_goals_model_latest.zip")
                shutil.copyfile(src_zip, dest_zip)
                print(f"[INFO] Updated latest alias: {dest_zip}")
        except Exception as e:
            print(f"[WARN] Could not set latest alias in {dir_path}: {e}")

    latest_checkpoint_path = find_existing_checkpoint()

    print(f"[INFO] Current directory: {os.getcwd()}")
    print(f"[INFO] Latest checkpoint path: {latest_checkpoint_path if latest_checkpoint_path else 'None found'}")
    
    # Create model
    if latest_checkpoint_path:
        print(f"[INFO] Loading existing checkpoint: {latest_checkpoint_path}")
        try:
            model = PPO.load(latest_checkpoint_path, env=env)
            print(f"[SUCCESS] Checkpoint loaded successfully!")
            print(f"[INFO] Model info: {type(model)}")
            
            # Set up checkpoint callbacks for both directories
            checkpoint_callbacks = CallbackList([
                CheckpointCallback(
                    save_freq=save_interval,
                    save_path=os.path.join(".", checkpoint_dirs[0]),
                    name_prefix="mountain_random_goals_model",
                ),
                CheckpointCallback(
                    save_freq=save_interval,
                    save_path=os.path.join(".", checkpoint_dirs[1]),
                    name_prefix="mountain_random_goals_model",
                ),
            ])
            combined_callback = CallbackList([checkpoint_callbacks, EpisodeLoggerCallback(verbose=0)])
            
            # Continue training
            print(f"[INFO] Continuing training for {total_timesteps} timesteps...")
            model.learn(
                total_timesteps=total_timesteps,
                callback=combined_callback,
                reset_num_timesteps=False  # Don't reset timestep counter
            )
            # After learn, set/update latest alias from the most recent zip in first dir
            for save_dir in checkpoint_dirs:
                # Attempt to find newest zip in this dir and copy as latest alias
                newest = max(glob.glob(os.path.join(save_dir, "*.zip")), key=os.path.getmtime, default=None)
                if newest:
                    copy_as_latest(save_dir, os.path.splitext(newest)[0])
            
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
            print("[INFO] Starting with fresh model...")
            model = PPO("MultiInputPolicy", env, verbose=1)
            
            # Set up checkpoint callbacks for both directories
            checkpoint_callbacks = CallbackList([
                CheckpointCallback(
                    save_freq=save_interval,
                    save_path=os.path.join(".", checkpoint_dirs[0]),
                    name_prefix="mountain_random_goals_model",
                ),
                CheckpointCallback(
                    save_freq=save_interval,
                    save_path=os.path.join(".", checkpoint_dirs[1]),
                    name_prefix="mountain_random_goals_model",
                ),
            ])
            combined_callback = CallbackList([checkpoint_callbacks, EpisodeLoggerCallback(verbose=0)])
            
            # Start fresh training
            print(f"[INFO] Starting fresh training for {total_timesteps} timesteps...")
            model.learn(
                total_timesteps=total_timesteps,
                callback=combined_callback
            )
            for save_dir in checkpoint_dirs:
                newest = max(glob.glob(os.path.join(save_dir, "*.zip")), key=os.path.getmtime, default=None)
                if newest:
                    copy_as_latest(save_dir, os.path.splitext(newest)[0])
    else:
        print(f"[INFO] No existing checkpoint found. Starting fresh training...")
        model = PPO("MultiInputPolicy", env, verbose=1)
        
        # Set up checkpoint callbacks for both directories
        checkpoint_callbacks = CallbackList([
            CheckpointCallback(
                save_freq=save_interval,
                save_path=os.path.join(".", checkpoint_dirs[0]),
                name_prefix="mountain_random_goals_model",
            ),
            CheckpointCallback(
                save_freq=save_interval,
                save_path=os.path.join(".", checkpoint_dirs[1]),
                name_prefix="mountain_random_goals_model",
            ),
        ])
        combined_callback = CallbackList([checkpoint_callbacks, EpisodeLoggerCallback(verbose=0)])
        
        # Start fresh training
        print(f"[INFO] Starting fresh training for {total_timesteps} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=combined_callback
        )
        for save_dir in checkpoint_dirs:
            newest = max(glob.glob(os.path.join(save_dir, "*.zip")), key=os.path.getmtime, default=None)
            if newest:
                copy_as_latest(save_dir, os.path.splitext(newest)[0])
    
    # Save final model to both directories
    for d in checkpoint_dirs:
        final_model_path = os.path.join(d, "mountain_random_goals_model_final")
        model.save(final_model_path)
        print(f"[SUCCESS] Final model saved to {final_model_path}")
    
    # Test the trained model with raw env (terminated/truncated)
    print("[INFO] Testing the trained model (raw env, terminated/truncated)...")
    eval_env = MountainPassRandomGoalsEnv()
    obs, info = eval_env.reset()
    for i in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action if isinstance(action, int) else int(action[0]))
        if terminated or truncated:
            obs, info = eval_env.reset()
    eval_env.close()
    
    env.close()
    print("[SUCCESS] Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train PPO agent on randomized goals using Stable-Baselines3')
    parser.add_argument('--timesteps', type=int, default=1000000, 
                       help='Total timesteps for training (default: 1000000)')
    parser.add_argument('--save-interval', type=int, default=50000,
                       help='Save interval in timesteps (default: 50000)')
    
    args = parser.parse_args()
    
    print(f"[CONFIG] Total timesteps: {args.timesteps}")
    print(f"[CONFIG] Save interval: {args.save_interval}")
    print(f"[CONFIG] Environment: MountainPassRandomGoalsEnv with zone-based curriculum learning")
    print(f"[CONFIG] Safe Zones for Position Generation:")
    print(f"  Zone 1 (Mountain Pass): x: -60.60 to -12.57, y: -62.67 to 37.09, z: -25.0 to -7.67 (NED: above ground)")
    print(f"  Zone 2 (Valley):        x: 10.90 to 21.6,   y: -90.0 to -75.77,  z: -9.0 to -3.0 (NED: above ground)")
    print(f"  Zone 3 (Plateau):       x: 43.7 to 59.4,   y: 32.65 to 46.85,  z: -4.0 to -3.0 (NED: above ground)")
    print(f"  Zone 4 (High Mountain): x: 33.7 to 46.33,  y: 109.67 to 126.46, z: -42.0 to -28.0 (NED: above ground)")
    print(f"[INFO] Note: Z coordinates use NED system (negative = above ground, positive = below ground)")
    print(f"[CONFIG] Curriculum parameters:")
    print(f"  - Start distance: 10.0m")
    print(f"  - Max distance: 100.0m") 
    print(f"  - Growth rate: 0.5m every 50 episodes")
    print(f"  - Zone selection: Random (1-4) for start positions")
    print(f"  - Goal placement: Prefer different zones for variety")
    
    main() 