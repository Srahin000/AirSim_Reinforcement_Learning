from mountain_env import MountainPassEnv
import time

print("[SIMPLE] Testing environment step by step...")

env = MountainPassEnv()
print("[SIMPLE] Environment created")

# Test just one episode with detailed logging
print("[SIMPLE] Starting single episode test...")
obs = env.reset()
print(f"[SIMPLE] Reset complete, obs shape: {obs.shape}")

for step in range(20):  # Test 20 steps
    print(f"[SIMPLE] Step {step}")
    
    # Get current state
    state = env.client.getMultirotorState()
    pos = state.kinematics_estimated.position
    print(f"[SIMPLE] Position: ({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})")
    
    # Take action
    action = env.action_space.sample()
    print(f"[SIMPLE] Action: {action}")
    
    # Step environment
    print(f"[SIMPLE] Calling env.step({action})...")
    obs, reward, done, info = env.step(action)
    print(f"[SIMPLE] Step completed: reward={reward:.2f}, done={done}")
    
    if done:
        print(f"[SIMPLE] Episode ended after {step + 1} steps")
        break
    
    time.sleep(0.1)  # Slow down for debugging

print("[SIMPLE] Test completed!") 