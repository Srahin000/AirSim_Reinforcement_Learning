from mountain_env import MountainPassEnv
import time

print("[TEST] Creating environment...")
env = MountainPassEnv()

print("[TEST] Testing environment for 5 episodes...")
for episode in range(5):
    print(f"\n[TEST] Episode {episode + 1}")
    
    # Reset environment
    obs = env.reset()
    print(f"[TEST] Reset complete. Observation shape: {obs.shape}")
    print(f"[TEST] Observation min: {obs.min()}, max: {obs.max()}")
    
    # Test a few steps
    total_reward = 0
    for step in range(5):
        # Get current state
        state = env.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        print(f"[TEST] Step {step}: Position=({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
        
        # Take random action
        action = env.action_space.sample()
        print(f"[TEST] Action: {action}")
        
        # Step environment
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"[TEST] Reward: {reward:.2f}, Total: {total_reward:.2f}, Done: {done}")
        
        if done:
            print(f"[TEST] Episode ended after {step + 1} steps")
            break
        
        time.sleep(0.1)

print("\n[TEST] Environment test complete!") 