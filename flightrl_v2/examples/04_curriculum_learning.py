#!/usr/bin/env python3
"""
Example 04: Curriculum Learning for Progressive Training

PLACEHOLDER - This example will be completed when Phase 2 (Curriculum Learning) is implemented.

This example will demonstrate:
- Setting up curriculum levels with increasing difficulty
- Using CurriculumWrapper to manage environment difficulty
- Using CurriculumCallback to track and advance levels
- Configuring advancement thresholds and success criteria
- Logging curriculum progress to TensorBoard

Prerequisites:
- Phase 2: Curriculum learning wrappers and callbacks must be implemented
- flightrl_v2.envs.wrappers.CurriculumWrapper must be functional
- flightrl_v2.algorithms.callbacks.CurriculumCallback must be functional

Current Status:
- CurriculumWrapper is a stub (raises NotImplementedError)
- CurriculumCallback is a stub (raises NotImplementedError)

Usage (when implemented):
    python 04_curriculum_learning.py --levels 5 --threshold 0.8
    python 04_curriculum_learning.py --config curriculum_config.yaml
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def show_current_status():
    """Show what's currently available vs. what's coming."""
    print("\n" + "="*70)
    print("  Example 04: Curriculum Learning")
    print("="*70)
    
    print("\n[STATUS] Current Status:")
    print("-"*70)
    
    # Check wrapper
    try:
        from flightrl_v2.envs.wrappers import CurriculumWrapper
        print("  [STUB] CurriculumWrapper - Stub (raises NotImplementedError)")
    except ImportError as e:
        print(f"  [X] CurriculumWrapper: {e}")
    
    # Check callback
    try:
        from flightrl_v2.algorithms.callbacks import CurriculumCallback
        print("  [STUB] CurriculumCallback - Stub (raises NotImplementedError)")
    except ImportError as e:
        print(f"  [X] CurriculumCallback: {e}")
    
    print("\n[INFO] Curriculum Learning Concept:")
    print("-"*70)
    print("""
  Curriculum learning trains agents on progressively harder tasks:
  
  Level 1: Easy    - Spawn close to target, no disturbances
  Level 2: Medium  - Spawn farther, small disturbances
  Level 3: Hard    - Random spawn, wind, obstacles
  
  Benefits:
  - Faster initial learning (easy tasks)
  - Better final performance (gradual difficulty increase)
  - More robust policies (exposed to varied conditions)
""")
    
    print("\n[INFO] Planned Architecture:")
    print("-"*70)
    print("""
  # Future usage (Phase 2):
  from flightrl_v2.envs import make_flight_env_for_sb3
  from flightrl_v2.envs.wrappers import CurriculumWrapper
  from flightrl_v2.algorithms.callbacks import CurriculumCallback
  
  # Define curriculum levels
  curriculum_levels = {
      0: {  # Level 0: Easy
          "spawn_radius": 1.0,
          "wind_magnitude": 0.0,
          "max_episode_steps": 200,
      },
      1: {  # Level 1: Medium
          "spawn_radius": 3.0,
          "wind_magnitude": 0.5,
          "max_episode_steps": 250,
      },
      2: {  # Level 2: Hard
          "spawn_radius": 5.0,
          "wind_magnitude": 1.0,
          "max_episode_steps": 300,
      },
  }
  
  # Create base environment
  env = make_flight_env_for_sb3(seed=42)
  
  # Wrap with curriculum
  env = CurriculumWrapper(
      env,
      levels=curriculum_levels,
      advancement_threshold=0.8,  # 80% success to advance
  )
  
  # Create callback for automatic advancement
  curriculum_callback = CurriculumCallback(
      advancement_threshold=0.8,
      eval_freq=10000,
      verbose=1,
  )
  
  # Train with curriculum
  model.learn(
      total_timesteps=1000000,
      callback=curriculum_callback,
  )
""")
    
    print("\n[TODO] Implementation Requirements:")
    print("-"*70)
    print("""
  1. CurriculumWrapper needs to:
     - Store level configurations
     - Modify environment parameters per level
     - Track current level
     - Expose advance_level() method
     
  2. CurriculumCallback needs to:
     - Monitor success rate per level
     - Call wrapper.advance_level() when threshold met
     - Log level changes to TensorBoard
     - Optionally save level-specific checkpoints
     
  3. Environment must support:
     - Dynamic parameter modification
     - Reset with new level parameters
""")
    
    print("\n" + "="*70)
    print("  For a working example (without curriculum), run:")
    print("    python 01_basic_training.py --timesteps 10000")
    print("="*70 + "\n")


def main():
    """Main entry point."""
    show_current_status()


if __name__ == "__main__":
    main()

