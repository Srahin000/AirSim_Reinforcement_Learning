#!/usr/bin/env python3
"""
Example script demonstrating different safety leniency levels in the Mountain Pass environment.

This script shows how to create environments with different safety strictness levels
and provides recommendations for when to use each level during training.
"""

import airsim
import time
from mountain_env_random_goals import MountainPassRandomGoalsEnv

def test_leniency_level(leniency: str, num_episodes: int = 2):
    """Test a specific leniency level with multiple episodes."""
    print(f"\n{'='*60}")
    print(f"[INFO] TESTING {leniency.upper()} LENIENCY LEVEL")
    print(f"{'='*60}")
    
    # Create environment with specific leniency
    env = MountainPassRandomGoalsEnv(
        verbose=True,
        conservative_spawning=True,
        safety_leniency=leniency,
        max_steps=100  # Shorter episodes for testing
    )
    
    episode_successes = 0
    episode_collisions = 0
    episode_safety_violations = 0
    
    for episode in range(num_episodes):
        print(f"\n[INFO] Episode {episode + 1}/{num_episodes}")
        
        try:
            # Reset environment
            obs, info = env.reset()
            print(f"[INFO] Reset completed. Start position: ({env.start_pos.x_val:.1f}, {env.start_pos.y_val:.1f}, {env.start_pos.z_val:.1f})")
            
            # Run episode
            step_count = 0
            for step in range(env.max_steps):
                # Simple forward action
                action = 0  # forward
                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                
                # Check for termination reasons
                if terminated:
                    if info.get('collision', False):
                        episode_collisions += 1
                        print(f"[INFO] Episode ended due to collision at step {step_count}")
                    elif info.get('unsafe', False):
                        episode_safety_violations += 1
                        print(f"[INFO] Episode ended due to safety violation: {info.get('safety_reason', 'Unknown')}")
                    else:
                        print(f"[INFO] Episode ended normally at step {step_count}")
                    break
                
                if truncated:
                    print(f"[INFO] Episode truncated at step {step_count}")
                    break
                
                # Print progress every 20 steps
                if step_count % 20 == 0:
                    distance = info.get('distance_to_goal', 0)
                    print(f"[INFO] Step {step_count}: Distance to goal: {distance:.1f}m, Reward: {reward:.2f}")
            
            episode_successes += 1
            
        except Exception as e:
            print(f"[ERROR] Episode {episode + 1} failed: {e}")
        
        finally:
            # Small delay between episodes
            time.sleep(1)
    
    # Print summary for this leniency level
    print(f"\n[INFO] {leniency.upper()} LENIENCY SUMMARY:")
    print(f"  • Episodes completed: {episode_successes}/{num_episodes}")
    print(f"  • Collisions: {episode_collisions}")
    print(f"  • Safety violations: {episode_safety_violations}")
    
    env.close()
    return episode_successes, episode_collisions, episode_safety_violations

def main():
    """Test all leniency levels and provide recommendations."""
    print("[INFO] Safety Leniency Demonstration")
    print("[INFO] This script tests different safety strictness levels")
    
    # Test each leniency level
    leniency_levels = ["more", "normal", "less"]
    results = {}
    
    for leniency in leniency_levels:
        successes, collisions, violations = test_leniency_level(leniency, num_episodes=2)
        results[leniency] = {
            'successes': successes,
            'collisions': collisions,
            'violations': violations
        }
    
    # Print comprehensive recommendations
    print(f"\n{'='*80}")
    print(f"[INFO] COMPREHENSIVE TRAINING RECOMMENDATIONS")
    print(f"{'='*80}")
    
    print(f"\n[INFO] 📊 RESULTS SUMMARY:")
    for leniency, result in results.items():
        print(f"  {leniency.upper()}: {result['successes']} successes, {result['collisions']} collisions, {result['violations']} violations")
    
    print(f"\n[INFO] 🎯 TRAINING STRATEGY:")
    print(f"")
    print(f"[INFO] PHASE 1: EXPLORATION (use 'more' leniency)")
    print(f"  • When: Starting training, testing new algorithms")
    print(f"  • Why: Faster learning, more exploration, less safety constraints")
    print(f"  • Expected: Higher collision rates, faster progress")
    print(f"")
    print(f"[INFO] PHASE 2: REFINEMENT (use 'normal' leniency)")
    print(f"  • When: Agent shows consistent progress, preparing for production")
    print(f"  • Why: Balanced safety and learning, realistic conditions")
    print(f"  • Expected: Moderate safety, steady improvement")
    print(f"")
    print(f"[INFO] PHASE 3: PRODUCTION (use 'less' leniency)")
    print(f"  • When: Final training, real hardware deployment")
    print(f"  • Why: Maximum safety, production-like conditions")
    print(f"  • Expected: Lower failure rates, stricter constraints")
    print(f"")
    print(f"[INFO] 💡 IMPLEMENTATION:")
    print(f"  • In training script: change safety_leniency parameter")
    print(f"  • Start with 'more', progress to 'normal', finish with 'less'")
    print(f"  • Monitor collision rates and adjust accordingly")
    print(f"")
    print(f"[INFO] ⚠️  IMPORTANT NOTES:")
    print(f"  • 'more' leniency may allow unsafe behaviors - monitor closely")
    print(f"  • 'less' leniency may slow learning - use only when ready")
    print(f"  • Always validate agent behavior before real deployment")
    print(f"{'='*80}")
    
    print("[SUCCESS] Leniency demonstration completed!")

if __name__ == "__main__":
    main()

