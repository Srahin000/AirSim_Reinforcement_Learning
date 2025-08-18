# Safety Leniency System Guide

## Overview

The Mountain Pass environment now includes a configurable safety leniency system that allows you to adjust how strict the safety checks are during training. This system helps balance learning speed with safety requirements.

## Leniency Levels

### 1. "more" - Lenient Mode
- **Safety Distance Multiplier**: 0.7x (more permissive)
- **Altitude Range**: 1.2x (wider range)
- **Use Case**: Early training, exploration, testing new algorithms
- **Benefits**: Faster learning, more exploration, less safety constraints
- **Risks**: May allow unsafe behaviors, higher collision rates

### 2. "normal" - Balanced Mode (Default)
- **Safety Distance Multiplier**: 1.0x (original values)
- **Altitude Range**: 1.0x (original range)
- **Use Case**: General training, balanced approach
- **Benefits**: Reasonable safety with good learning progress
- **Risks**: Moderate safety constraints

### 3. "less" - Strict Mode
- **Safety Distance Multiplier**: 1.5x (stricter)
- **Altitude Range**: 0.8x (narrower range)
- **Use Case**: Final training, production deployment, real hardware
- **Benefits**: Maximum safety, production-like conditions
- **Risks**: Slower learning, stricter constraints

## Implementation Details

### Safety Thresholds Affected

The leniency system adjusts these safety parameters:

```python
# Original values
lidar_safety_distance = 2.0      # Horizontal obstacle safety
ground_safety_distance = 1.5     # Ground proximity safety
min_altitude = 1.0               # Minimum height above ground
max_altitude = 30.0              # Maximum height above ground

# Effective values with leniency
effective_lidar_safety = lidar_safety_distance * leniency_multiplier
effective_ground_safety = ground_safety_distance * leniency_multiplier
effective_min_altitude = min_altitude * leniency_multiplier
effective_max_altitude = max_altitude * leniency_multiplier
```

### Multiplier Values

| Leniency | Lidar Safety | Ground Safety | Min Altitude | Max Altitude |
|----------|--------------|---------------|--------------|--------------|
| "more"   | 0.7x         | 0.7x          | 0.7x         | 1.2x         |
| "normal" | 1.0x         | 1.0x          | 1.0x         | 1.0x         |
| "less"   | 1.5x         | 1.5x          | 1.3x         | 0.8x         |

## Usage Examples

### Basic Usage

```python
from mountain_env_random_goals import MountainPassRandomGoalsEnv

# Lenient mode for early training
env = MountainPassRandomGoalsEnv(safety_leniency="more")

# Normal mode for balanced training
env = MountainPassRandomGoalsEnv(safety_leniency="normal")

# Strict mode for production
env = MountainPassRandomGoalsEnv(safety_leniency="less")
```

### Training Script Integration

```python
# In your training script
def make_env():
    return Monitor(MountainPassRandomGoalsEnv(
        max_steps=500,
        step_length=4.0,
        altitude_step=2.0,
        lidar_safety_distance=2.0,
        ground_safety_distance=1.5,
        max_altitude=30.0,
        min_altitude=1.0,
        hard_reset_on_collision=True,
        safety_arm_steps=8,
        safety_leniency="normal"  # Change this as training progresses
    ))
```

## Training Progression Strategy

### Phase 1: Exploration (Weeks 1-2)
- **Leniency**: "more"
- **Goal**: Rapid exploration and learning
- **Monitor**: Collision rates, learning progress
- **Success Criteria**: Agent shows basic navigation skills

### Phase 2: Refinement (Weeks 3-4)
- **Leniency**: "normal"
- **Goal**: Balanced learning with reasonable safety
- **Monitor**: Safety violations, learning stability
- **Success Criteria**: Consistent performance, few safety violations

### Phase 3: Production (Weeks 5+)
- **Leniency**: "less"
- **Goal**: Production-ready behavior
- **Monitor**: All safety metrics, real-world applicability
- **Success Criteria**: Safe, reliable performance

## Monitoring and Adjustment

### Key Metrics to Watch

1. **Collision Rate**: Should decrease as training progresses
2. **Safety Violations**: Monitor frequency and types
3. **Learning Progress**: Episode completion rates, rewards
4. **Behavior Consistency**: Stable performance across episodes

### When to Adjust Leniency

#### Increase Strictness (move to "less" lenient)
- Agent consistently avoids collisions
- Learning progress has plateaued
- Preparing for real hardware deployment

#### Decrease Strictness (move to "more" lenient)
- Learning is too slow
- Agent is overly cautious
- Testing new algorithms or reward functions

## Safety Considerations

### "more" Leniency Risks
- Higher collision rates
- Potential damage to simulation environment
- Unsafe behaviors may develop

### "less" Leniency Benefits
- Maximum safety guarantees
- Production-ready behavior
- Realistic constraints

### Best Practices
1. **Start Conservative**: Begin with "normal" leniency
2. **Monitor Closely**: Watch safety metrics during training
3. **Gradual Progression**: Move through leniency levels systematically
4. **Validate Behavior**: Test agent thoroughly before deployment

## Troubleshooting

### Common Issues

#### Agent Too Cautious
- **Symptom**: Agent rarely moves, gets stuck
- **Solution**: Try "more" leniency temporarily

#### Too Many Collisions
- **Symptom**: High failure rates, unsafe behavior
- **Solution**: Move to "less" leniency, review reward function

#### Learning Plateau
- **Symptom**: No improvement over many episodes
- **Solution**: Adjust leniency, review hyperparameters

### Debug Tools

The environment includes several debug methods:

```python
# Debug a specific position
env.debug_position(position, "Label")

# Check current safety status
is_safe, reason = env.check_safety()

# Analyze terrain around position
terrain_info = env._analyze_terrain_around_position(position)
```

## Example Scripts

### Basic Testing
```bash
# Test all leniency levels
python leniency_example.py

# Test specific environment
python mountain_env_random_goals.py
```

### Training Integration
```bash
# Train with specific leniency
python train_random_goals_sb3.py
```

## Conclusion

The safety leniency system provides a flexible way to balance learning speed with safety requirements. Use it strategically to:

1. **Accelerate early learning** with "more" leniency
2. **Maintain balanced progress** with "normal" leniency  
3. **Ensure production safety** with "less" leniency

Remember to monitor performance metrics and adjust leniency levels based on your specific training goals and safety requirements.

