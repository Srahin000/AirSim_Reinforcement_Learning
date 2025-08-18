#!/usr/bin/env python3
"""
Create dummy TensorBoard logs for testing
This will generate valid event files without requiring AirSim
"""

import os
import time
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.util import tensor_util
import tensorflow as tf

def create_dummy_logs():
    """Create dummy tensorboard logs with fake training data."""
    
    # Create log directory
    log_dir = "ppo_clean_tensorboard/PPO_8"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"[INFO] Creating dummy logs in {log_dir}")
    
    # Create a summary writer
    current_time = int(time.time())
    log_file = os.path.join(log_dir, f"events.out.tfevents.{current_time}.dummy.0")
    
    with tf.summary.create_file_writer(log_dir) as writer:
        # Generate fake training data
        for step in range(0, 100, 5):  # Every 5 steps
            with writer.as_default():
                # Log fake reward
                tf.summary.scalar('train/reward', 100 - step + tf.random.normal([], 0, 5), step=step)
                
                # Log fake loss
                tf.summary.scalar('train/loss', 2.0 * tf.exp(-step/50) + tf.random.normal([], 0, 0.1), step=step)
                
                # Log fake episode length
                tf.summary.scalar('train/episode_length', 200 - step + tf.random.normal([], 0, 10), step=step)
                
                # Log fake value function
                tf.summary.scalar('train/value_loss', 1.5 * tf.exp(-step/40) + tf.random.normal([], 0, 0.05), step=step)
                
                # Log fake policy loss
                tf.summary.scalar('train/policy_loss', 1.0 * tf.exp(-step/60) + tf.random.normal([], 0, 0.08), step=step)
                
                # Log fake entropy
                tf.summary.scalar('train/entropy', 0.5 * tf.exp(-step/80) + tf.random.normal([], 0, 0.02), step=step)
                
                # Log fake learning rate
                tf.summary.scalar('train/learning_rate', 3e-4 * tf.exp(-step/100), step=step)
                
                # Log fake distance to goal
                tf.summary.scalar('train/distance_to_goal', 50 - step/2 + tf.random.normal([], 0, 3), step=step)
                
                # Log fake collision count
                tf.summary.scalar('train/collision_count', max(0, step//20), step=step)
                
                # Log fake success rate
                success_rate = min(1.0, step/50 + tf.random.normal([], 0, 0.1))
                tf.summary.scalar('train/success_rate', success_rate, step=step)
                
                writer.flush()
    
    print(f"[INFO] Created dummy logs with {100//5} data points")
    print(f"[INFO] Log file: {log_file}")
    print(f"[INFO] You can now run: python -m tensorboard.main --logdir=ppo_clean_tensorboard --port=6008")

if __name__ == "__main__":
    create_dummy_logs()
