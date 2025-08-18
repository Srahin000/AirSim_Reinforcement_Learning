# Mountain Pass Exploratory Environment

This directory contains an exploratory version of the Mountain Pass environment that generates random goals within an increasing exploration radius around the drone.

## Overview

The exploratory environment (`MountainPassExploratoryEnv`) extends the original mountain environment with the following key features:

### Key Additions

1. **Dynamic Goal Generation**: Instead of a fixed goal, the environment generates random goals within a configurable radius around the drone's current position.

2. **Progressive Exploration Radius**: The exploration radius starts small and increases over time, allowing the drone to explore larger areas as it becomes more proficient.

3. **Episode Tracking**: The environment tracks total episodes and uses this to determine when to increase the exploration radius.

4. **Enhanced Logging**: Detailed logging includes exploration radius, goal positions, and episode statistics.

## Files

- `mountain_env_exploratory.py` - The main exploratory environment class
- `train_exploratory.py` - Training script for the exploratory environment
- `test_exploratory.py` - Testing script for the exploratory environment

## Environment Parameters

### Exploration Parameters

- `initial_exploration_radius` (default: 10.0m): Starting radius for goal generation
- `max_exploration_radius` (default: 50.0m): Maximum radius the environment can reach
- `radius_increase_episodes` (default: 10000): Number of episodes before increasing radius

### Goal Generation Logic

The environment generates goals using the following algorithm:

1. **Random Angle**: Generates a random angle between 0 and 360 degrees
2. **Random Distance**: Generates a random distance between 5m and the current exploration radius
3. **Altitude Variation**: Adds small random altitude variation (±3m) to the current altitude
4. **Safety Bounds**: Ensures the generated goal is within altitude limits

## Usage

### Training

```bash
python train_exploratory.py
```

This will:
- Start with a 10m exploration radius
- Increase the radius by 5m every 10,000 episodes
- Save checkpoints every 1,000 steps
- Log detailed statistics including exploration radius

### Testing

```bash
python test_exploratory.py
```

This will:
- Test the environment without a model
- Test exploration radius progression
- Test trained models if available

### Environment Testing

```bash
python mountain_env_exploratory.py
```

This will test the basic environment functionality.

## Exploration Radius Progression

The exploration radius increases according to this schedule:

- Episodes 1-10,000: 10m radius
- Episodes 10,001-20,000: 15m radius
- Episodes 20,001-30,000: 20m radius
- Episodes 30,001-40,000: 25m radius
- Episodes 40,001-50,000: 30m radius
- And so on, up to the maximum radius

## Key Methods

### `generate_random_goal(current_position)`

Generates a random goal within the current exploration radius:

```python
def generate_random_goal(self, current_position: airsim.Vector3r) -> airsim.Vector3r:
    # Generate random angle and distance
    angle = np.random.uniform(0, 2 * np.pi)
    distance = np.random.uniform(5.0, self.current_exploration_radius)
    
    # Calculate goal position
    goal_x = current_position.x_val + distance * np.cos(angle)
    goal_y = current_position.y_val + distance * np.sin(angle)
    goal_z = current_position.z_val + np.random.uniform(-3.0, 3.0)
    
    return airsim.Vector3r(goal_x, goal_y, goal_z)
```

### `update_exploration_radius()`

Updates the exploration radius based on episode count:

```python
def update_exploration_radius(self):
    radius_increases = self.total_episodes // self.radius_increase_episodes
    new_radius = self.initial_exploration_radius + (radius_increases * 5.0)
    new_radius = min(new_radius, self.max_exploration_radius)
    
    if new_radius != self.current_exploration_radius:
        self.current_exploration_radius = new_radius
        print(f"[EXPLORATION] Exploration radius increased to {self.current_exploration_radius:.1f}m")
```

## Differences from Original Environment

1. **Fixed vs Dynamic Goals**: Original has a fixed goal at (45.78, 114.50, -19.35), exploratory generates random goals
2. **Exploration Tracking**: Tracks total episodes and exploration radius
3. **Enhanced Info**: Additional info fields include `exploration_radius` and `total_episodes`
4. **Progressive Difficulty**: Starts with small exploration area and gradually increases

## Training Considerations

1. **Curriculum Learning**: The increasing exploration radius provides natural curriculum learning
2. **Generalization**: Random goals help the model generalize to different target locations
3. **Exploration vs Exploitation**: Balances exploration of new areas with exploitation of learned skills
4. **Safety**: Maintains all safety features from the original environment

## Monitoring

The training callback (`ExploratoryTrainingCallback`) provides detailed monitoring:

- Episode-by-episode logging with exploration radius
- Running statistics every 10 episodes
- Exploration radius history tracking
- TensorBoard logging with exploration metrics

## Example Output

```
[EPISODE    1] Reward:   45.20, Length:  67, Distance:   4.8, Collisions: 0, Radius: 10.0m, Total Episodes: 1, Status: GOAL
           Lidar - Ground:  15.2, Horizontal:  25.1
[EXPLORATION] Generated goal at (8.45, 12.32, -6.23)
[EXPLORATION] Distance from current: 8.45m, Angle: 45.2°
[EXPLORATION] Current exploration radius: 10.0m
```

This system allows for more realistic exploration scenarios where the drone learns to navigate to arbitrary goals rather than memorizing a single target location. 