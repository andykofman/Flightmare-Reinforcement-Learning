#!/usr/bin/env python3
"""
Example 02: Custom Reward Functions and Task Definitions

PLACEHOLDER - This example will be completed when Phase 3 (Modular Rewards) is implemented.

This example will demonstrate:
- Creating custom Task classes (inheriting from BaseTask)
- Defining modular reward components (inheriting from BaseReward)
- Composing multiple rewards using CompositeReward
- Integrating custom rewards with the training pipeline
- Comparing different reward formulations

Prerequisites:
- Phase 3: Modular reward system must be implemented
- flightrl_v2.rewards module with functional PositionReward, CollisionReward

Current Status:
- BaseTask is implemented (can create custom tasks now)
- BaseReward is a stub (raises NotImplementedError)
- CompositeReward is a stub

Usage (when implemented):
    python 02_custom_reward.py --reward-type shaped
    python 02_custom_reward.py --reward-type sparse --timesteps 1000000
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def show_current_status():
    """Show what's currently available vs. what's coming."""
    print("\n" + "="*70)
    print("  Example 02: Custom Reward Functions")
    print("="*70)
    
    print("\n[STATUS] Current Status:")
    print("-"*70)
    
    # Check what's importable
    try:
        from flightrl_v2.tasks import BaseTask, HoverTask, TargetReachingTask
        print("  [OK] BaseTask - Available (can create custom tasks)")
        print("  [OK] HoverTask - Available (fully implemented)")
        print("  [OK] TargetReachingTask - Available (fully implemented)")
    except ImportError as e:
        print(f"  [X] Tasks module: {e}")
    
    try:
        from flightrl_v2.rewards import BaseReward, PositionReward, CompositeReward
        print("  [STUB] BaseReward - Stub (raises NotImplementedError)")
        print("  [STUB] PositionReward - Stub (raises NotImplementedError)")
        print("  [STUB] CompositeReward - Stub (raises NotImplementedError)")
    except ImportError as e:
        print(f"  [X] Rewards module: {e}")
    
    print("\n[INFO] What you CAN do now:")
    print("-"*70)
    print("""
  1. Create custom Task classes by inheriting from BaseTask:
  
     from flightrl_v2.tasks import BaseTask, TaskConfig
     
     class MyCustomTask(BaseTask):
         @property
         def name(self) -> str:
             return "my_custom_task"
         
         def compute_reward(self, obs, action, next_obs, info):
             # Your custom reward logic here
             distance = np.linalg.norm(next_obs[:3] - self.target)
             reward = -distance  # Simple distance penalty
             return reward, info
         
         def is_terminated(self, obs, info):
             # Your termination condition
             return False
         
         def get_success_info(self, obs, info):
             return {"distance": float(np.linalg.norm(obs[:3]))}

  2. Use existing tasks (HoverTask, TargetReachingTask) as templates
  
  3. See 01_basic_training.py for a complete training example
""")
    
    print("\n[FUTURE] Coming in Phase 3:")
    print("-"*70)
    print("""
  - Modular reward components (PositionReward, VelocityReward, etc.)
  - CompositeReward for combining multiple reward signals
  - Reward normalization and scaling
  - Reward visualization and debugging tools
""")
    
    print("\n" + "="*70)
    print("  For a working example, run:")
    print("    python 01_basic_training.py --timesteps 10000")
    print("="*70 + "\n")


def main():
    """Main entry point."""
    show_current_status()


if __name__ == "__main__":
    main()

