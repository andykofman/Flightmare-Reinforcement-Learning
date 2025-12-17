"""
PLACEHOLDER - Phase 2/3 Implementation
TODO: Implement obstacle avoidance task when Phase 2 begins.

Obstacle avoidance task definition.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import BaseTask, TaskConfig
from flightrl_v2.core.types import ActionType, InfoType, ObservationType, RewardType


@dataclass
class ObstacleAvoidanceTaskConfig(TaskConfig):
    """Configuration for obstacle avoidance task."""
    target_position: Tuple[float, float, float] = (0.0, 0.0, 5.0)
    position_threshold: float = 0.5
    collision_penalty: float = 100.0
    proximity_penalty_weight: float = 1.0
    min_obstacle_distance: float = 0.5  # meters
    # TODO: Add obstacle configuration


class ObstacleAvoidanceTask(BaseTask):
    """
    Obstacle avoidance task: Navigate to target while avoiding obstacles.

    NOT YET IMPLEMENTED - Placeholder for Phase 2/3.

    This task will require:
    - Sensor data (LIDAR/depth) for obstacle detection
    - Collision detection from C++ backend
    - Proximity-based reward shaping
    """

    def __init__(self, config: Optional[ObstacleAvoidanceTaskConfig] = None):
        self.config = config or ObstacleAvoidanceTaskConfig()
        super().__init__(self.config)

    @property
    def name(self) -> str:
        return "obstacle_avoidance"

    def compute_reward(
        self,
        observation: ObservationType,
        action: ActionType,
        next_observation: ObservationType,
        info: InfoType
    ) -> Tuple[RewardType, InfoType]:
        """Compute obstacle avoidance reward."""
        raise NotImplementedError(
            "Phase 2/3: Obstacle avoidance task not yet implemented. "
            "This requires sensor pipeline (Phase 1) and obstacle environment (Phase 2)."
        )

    def is_terminated(
        self,
        observation: ObservationType,
        info: InfoType
    ) -> bool:
        """Check for collision or goal reached."""
        raise NotImplementedError(
            "Phase 2/3: Obstacle avoidance task not yet implemented."
        )

    def get_success_info(self, observation: ObservationType, info: InfoType) -> Dict[str, Any]:
        """Get obstacle avoidance success information."""
        raise NotImplementedError(
            "Phase 2/3: Obstacle avoidance task not yet implemented."
        )

