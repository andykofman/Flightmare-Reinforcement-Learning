"""
Hover task definition.

The hover task requires the drone to maintain a stable position at a target height.

"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
from numpy.typing import NDArray

from .base import BaseTask, TaskConfig
from flightrl_v2.core.types import InfoType, ObservationType, ActionType, RewardType

@dataclass
class HoverTaskConfig(TaskConfig):
    """
    Configuration for the hover task.

    """
    target_position: Tuple[float, float, float] = (0.0, 0.0, 5.0)  # Target hover position (x, y, z)
    position_threshold: float = 0.2  # 0.2 meter is the distance threshold to consider as hovering successfully
    velocity_threshold: float = 0.5  # 0.5 m/s is the velocity threshold to consider as stable hover
    success_hold_time: float = 2.0  # 2.0 seconds to hold the hover position for success
    dt: float = 0.02               # Time step duration in seconds

    # Reward weights
    position_weight: float = 1.0
    velocity_weight: float = 0.1 
    action_weight: float = 0.01

class HoverTask(BaseTask):
    """
    Hover task: Stabilize at a fixed position

    Observation space format:

    - [0:3] Position (x, y, z)
    - [3:6] Orientation (Euler angles or quaternion)   
    - [6:9] Linear Velocity (vx, vy, vz)
    - [9:12] Angular Velocity (wx, wy, wz)

    Reward function:
     - Negative distance to target position
     - Negative linear velocity magnitude (for stability)
     - Small action penalty (to encourage minimal control effort) 
    """

    def __init__(self, config: Optional[HoverTaskConfig] = None):
    
        """
        Initialize the hover task with optional configuration.
        """

        self.config: HoverTaskConfig = config or HoverTaskConfig()
        super().__init__(self.config)
        self._success_steps = 0
        self._target = np.array(self.config.target_position, dtype=np.float32)

    @property
    def name(self) -> str:
        return "Hover"
    
    def compute_reward(
        self,
        observation: ObservationType,
        action: ActionType,
        next_observation: ObservationType,
        info: InfoType
    ) -> Tuple[RewardType, InfoType]:
        """Compute the reward for the hover task."""

        position = next_observation[0:3]
        velocity = next_observation[6:9]

        # distance to target position
        distance = np.linalg.norm(position - self._target)
        velocity_mag = np.linalg.norm(velocity)
        action_mag = np.linalg.norm(action)

        # compute reward components
        position_reward = -self.config.position_weight * distance
        velocity_reward = -self.config.velocity_weight * velocity_mag
        action_reward = -self.config.action_weight * action_mag

        total_reward = position_reward + velocity_reward + action_reward

        # update info dictionary
        info ["distance_to_target"] = distance
        info["velocity_magnitude"] = velocity_mag
        info["reward_components"] = {
            "position_reward": position_reward,
            "velocity_reward": velocity_reward,
            "action_reward": action_reward
        }

        return float(total_reward), info
    

    def is_terminated(
            self,
            observation: ObservationType,
            info: InfoType
    ) -> bool:
        """Check if the hover task is terminated."""

        position = observation[0:3]
        velocity = observation[6:9]

        distance = np.linalg.norm(position - self._target)
        velocity_mag = np.linalg.norm(velocity)

        # Check if currently at target with low velocity
        at_target = (
            distance < self.config.position_threshold and
            velocity_mag < self.config.velocity_threshold
        )

        if at_target:
            self._success_steps += 1
        else:
            self._success_steps = 0

        # Success if stable for required time
        required_steps = int(self.config.success_hold_time / self.config.dt)
        return self._success_steps >= required_steps
    
    def reset(self) -> None:
        """Reset the hover task state."""
        super().reset()
        self._success_steps = 0

    
    def get_success_info(self, observation: ObservationType, info: InfoType) -> Dict[str, Any]:
        """Get success information for the hover task."""
        position = observation[0:3]
        velocity = observation[6:9]

        distance = np.linalg.norm(position - self._target)
        velocity_mag = np.linalg.norm(velocity)

        success_info = {
            "distance_to_target": float(distance),
            "velocity_magnitude": float(velocity_mag),
            "at_target": distance < self.config.position_threshold and velocity_mag < self.config.velocity_threshold,
            "stable": velocity_mag < self.config.velocity_threshold,
            "success_steps": self._success_steps
        }
        return success_info