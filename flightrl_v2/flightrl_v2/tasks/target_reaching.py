"""
Target reaching task definition.

The target reaching task requires the quadrotor to navigate to a specified
target position from a random starting point.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .base import BaseTask, TaskConfig
from flightrl_v2.core.types import ActionType, InfoType, ObservationType, RewardType


@dataclass
class TargetReachingTaskConfig(TaskConfig):
    """Configuration for target reaching task."""
    target_position: Tuple[float, float, float] = (0.0, 0.0, 5.0)
    position_threshold: float = 0.5  # meters - success distance
    velocity_threshold: float = 0.5  # m/s - success velocity
    success_hold_time: float = 1.0  # seconds to hold at target
    dt: float = 0.02  # simulation timestep

    # Spawn configuration
    spawn_radius: float = 5.0  # Random spawn distance from target
    spawn_height_range: Tuple[float, float] = (2.0, 8.0)

    # Reward weights
    distance_weight: float = 1.0
    velocity_weight: float = 0.1
    action_weight: float = 0.01
    goal_bonus: float = 100.0


class TargetReachingTask(BaseTask):
    """
    Target reaching task: Navigate to target from random start.

    Similar to hover but with emphasis on reaching the target rather
    than just maintaining position. Used for training navigation.

    Observation space assumed format:
    - [0:3] Position (x, y, z)
    - [3:6] Orientation
    - [6:9] Linear velocity
    - [9:12] Angular velocity
    - [12:15] Target relative position (optional)
    - [15] Target distance (optional)
    """

    def __init__(self, config: Optional[TargetReachingTaskConfig] = None):
        """Initialize target reaching task."""
        self.config: TargetReachingTaskConfig = config or TargetReachingTaskConfig()
        super().__init__(self.config)
        self._success_steps = 0
        self._target = np.array(self.config.target_position, dtype=np.float32)
        self._min_distance = float('inf')
        self._reached_target = False

    @property
    def name(self) -> str:
        return "target_reaching"

    def compute_reward(
        self,
        observation: ObservationType,
        action: ActionType,
        next_observation: ObservationType,
        info: InfoType
    ) -> Tuple[RewardType, InfoType]:
        """Compute target reaching reward."""
        position = next_observation[:3]
        velocity = next_observation[6:9]

        # Distance to target
        distance = np.linalg.norm(position - self._target)
        velocity_mag = np.linalg.norm(velocity)
        action_mag = np.linalg.norm(action)

        # Track minimum distance
        self._min_distance = min(self._min_distance, distance)

        # Reward components
        distance_reward = -self.config.distance_weight * distance
        velocity_reward = -self.config.velocity_weight * velocity_mag
        action_reward = -self.config.action_weight * action_mag

        total_reward = distance_reward + velocity_reward + action_reward

        # Goal bonus for reaching target
        at_target = (
            distance < self.config.position_threshold and
            velocity_mag < self.config.velocity_threshold
        )
        if at_target and not self._reached_target:
            total_reward += self.config.goal_bonus
            self._reached_target = True

        # Update info
        info["distance_to_target"] = distance
        info["min_distance"] = self._min_distance
        info["velocity_magnitude"] = velocity_mag
        info["at_target"] = at_target
        info["reward_components"] = {
            "distance": distance_reward,
            "velocity": velocity_reward,
            "action": action_reward,
        }

        return float(total_reward), info

    def is_terminated(
        self,
        observation: ObservationType,
        info: InfoType
    ) -> bool:
        """Check if target is reached and held."""
        position = observation[:3]
        velocity = observation[6:9]

        distance = np.linalg.norm(position - self._target)
        velocity_mag = np.linalg.norm(velocity)

        at_target = (
            distance < self.config.position_threshold and
            velocity_mag < self.config.velocity_threshold
        )

        if at_target:
            self._success_steps += 1
        else:
            self._success_steps = 0

        required_steps = int(self.config.success_hold_time / self.config.dt)
        return self._success_steps >= required_steps

    def reset(self) -> None:
        """Reset task state for new episode."""
        super().reset()
        self._success_steps = 0
        self._min_distance = float('inf')
        self._reached_target = False

    def get_success_info(self, observation: ObservationType, info: InfoType) -> Dict[str, Any]:
        """Get target reaching success information."""
        position = observation[:3]
        velocity = observation[6:9]

        distance = np.linalg.norm(position - self._target)
        velocity_mag = np.linalg.norm(velocity)

        return {
            "distance_to_target": float(distance),
            "min_distance_achieved": float(self._min_distance),
            "velocity_magnitude": float(velocity_mag),
            "at_target": distance < self.config.position_threshold,
            "stable": velocity_mag < self.config.velocity_threshold,
            "reached_target": self._reached_target,
            "success_steps": self._success_steps,
        }

