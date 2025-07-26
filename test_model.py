from stable_baselines3 import PPO
from mountain_env import MountainPassEnv

# Create the AirSim Gym environment
env = MountainPassEnv()

# Load the trained PPO model
model = PPO.load("mountain_pass_model")

obs = env.reset()

# Run the trained agent for 500 steps
total_reward = 0
for _ in range(500):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        print(f"Episode done. Total reward: {total_reward}")
        obs = env.reset()
        total_reward = 0 