#!/usr/bin/env python3
"""
Example 05: Obstacle Avoidance Training

PLACEHOLDER - This example will be completed when Phase 3 is fully implemented.

This example will demonstrate:
- Training with obstacle environments
- Using LIDAR/depth sensors for obstacle detection
- Collision penalty rewards with proximity shaping
- Combining navigation with obstacle avoidance
- Curriculum learning for increasing obstacle density

Prerequisites:
- Phase 1: Sensor pipeline (LIDAR, depth camera)
- Phase 2: Curriculum learning
- Phase 3: Obstacle avoidance task and reward functions
- flightrl_v2.tasks.ObstacleAvoidanceTask must be functional

Current Status:
- ObstacleAvoidanceTask is a stub (raises NotImplementedError)
- Sensor pipeline not yet implemented
- Collision detection requires C++ backend extensions

Usage (when implemented):
    python 05_obstacle_avoidance.py --obstacles 5 --sensor lidar
    python 05_obstacle_avoidance.py --curriculum --timesteps 2000000
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def show_current_status():
    """Show what's currently available vs. what's coming."""
    print("\n" + "="*70)
    print("  Example 05: Obstacle Avoidance Training")
    print("="*70)
    
    print("\n[STATUS] Current Status:")
    print("-"*70)
    
    # Check task
    try:
        from flightrl_v2.tasks import ObstacleAvoidanceTask, ObstacleAvoidanceTaskConfig
        print("  [STUB] ObstacleAvoidanceTask - Stub (raises NotImplementedError)")
        print("  [OK] ObstacleAvoidanceTaskConfig - Available (configuration class)")
    except ImportError as e:
        print(f"  [X] ObstacleAvoidanceTask: {e}")
    
    # Check sensors
    try:
        from flightrl_v2.sensors import LidarSensor
        print("  [STUB] LidarSensor - Stub (required for obstacle detection)")
    except ImportError as e:
        print(f"  [X] LidarSensor: {e}")
    
    # Check rewards
    try:
        from flightrl_v2.rewards import CollisionReward
        print("  [STUB] CollisionReward - Stub (for collision penalties)")
    except ImportError as e:
        print(f"  [X] CollisionReward: {e}")
    
    print("\n[INFO] Obstacle Avoidance Concept:")
    print("-"*70)
    print("""
  The agent must navigate to a target while avoiding obstacles:
  
  Sensors:
  - LIDAR rays detect obstacle distances
  - Depth camera provides spatial awareness
  
  Rewards:
  - Position reward: Approach target
  - Collision penalty: Large negative reward for collisions
  - Proximity penalty: Small negative reward for being close to obstacles
  
  Termination:
  - Success: Reach target and stabilize
  - Failure: Collision with obstacle
  - Timeout: Max episode steps exceeded
""")
    
    print("\n[INFO] Planned Architecture:")
    print("-"*70)
    print("""
  # Future usage (Phases 1-3 complete):
  from flightrl_v2.envs import make_flight_env_for_sb3
  from flightrl_v2.sensors import LidarSensor, LidarConfig
  from flightrl_v2.tasks import ObstacleAvoidanceTask, ObstacleAvoidanceTaskConfig
  from flightrl_v2.rewards import PositionReward, CollisionReward, CompositeReward
  
  # Configure sensors
  lidar = LidarSensor(LidarConfig(
      num_rays=16,
      max_range=10.0,
      fov_horizontal=360.0,
  ))
  
  # Configure task
  task_config = ObstacleAvoidanceTaskConfig(
      target_position=(10.0, 0.0, 5.0),
      collision_penalty=100.0,
      proximity_penalty_weight=1.0,
      min_obstacle_distance=0.5,
  )
  task = ObstacleAvoidanceTask(task_config)
  
  # Configure composite reward
  reward = CompositeReward([
      PositionReward(weight=1.0, target=task_config.target_position),
      CollisionReward(weight=10.0, penalty=100.0),
  ])
  
  # Create environment with obstacles
  env = make_flight_env_for_sb3(
      config_path="obstacle_environment.yaml",
      sensors=[lidar],
      task=task,
      reward=reward,
      seed=42,
  )
  
  # Train
  model = SAC("MlpPolicy", env)
  model.learn(total_timesteps=2000000)
""")
    
    print("\n[INFO] Curriculum for Obstacle Avoidance:")
    print("-"*70)
    print("""
  Progressive training with increasing difficulty:
  
  Level 0: No obstacles (pure navigation)
  Level 1: 3 static obstacles, easy positions
  Level 2: 6 static obstacles, harder positions
  Level 3: 10 static obstacles, random positions
  Level 4: 10+ obstacles, some moving (dynamic)
  
  This curriculum helps the agent:
  1. First learn basic navigation
  2. Then learn to detect and avoid obstacles
  3. Finally handle complex dynamic environments
""")
    
    print("\n[TODO] Implementation Requirements:")
    print("-"*70)
    print("""
  1. C++ Backend Extensions:
     - Obstacle spawning and management
     - Collision detection with obstacles
     - LIDAR ray casting against obstacles
     
  2. Sensor Pipeline (Phase 1):
     - LIDAR sensor functional
     - Depth camera functional
     - Sensor data in observation space
     
  3. Reward System (Phase 3):
     - CollisionReward with proximity shaping
     - CompositeReward for combining signals
     
  4. Task Implementation:
     - ObstacleAvoidanceTask.compute_reward()
     - ObstacleAvoidanceTask.is_terminated() (collision check)
""")
    
    print("\n" + "="*70)
    print("  For a working example (without obstacles), run:")
    print("    python 01_basic_training.py --timesteps 10000")
    print("="*70 + "\n")


def main():
    """Main entry point."""
    show_current_status()


if __name__ == "__main__":
    main()

