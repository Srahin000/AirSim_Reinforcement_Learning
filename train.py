from stable_baselines3 import PPO
from mountain_env import MountainPassEnv

# Create the AirSim Gym environment
env = MountainPassEnv()

# Create the PPO model with a convolutional policy
model = PPO("CnnPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100_000)

# Save the trained model
model.save("mountain_pass_model") 