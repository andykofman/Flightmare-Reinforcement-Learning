"""
PLACEHOLDER - Phase 3 Implementation
TODO: Implement collision penalties when obstacle environment is ready.

Collision-based reward functions.
"""
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

from .base import BaseReward, RewardConfig


@dataclass
class CollisionRewardConfig(RewardConfig):
    """Configuration for collision rewards."""
    collision_penalty: float = 100.0
    proximity_penalty_weight: float = 1.0
    min_safe_distance: float = 0.5  # meters


class CollisionReward(BaseReward):
    """
    Collision-based reward: Penalize collisions and proximity to obstacles.

    NOT YET IMPLEMENTED - Placeholder for Phase 3.

    This reward component requires:
    - Collision detection from C++ backend
    - Obstacle distance information (from LIDAR/depth)
    """

    def __init__(self, config: CollisionRewardConfig = None):
        self.config: CollisionRewardConfig = config or CollisionRewardConfig()
        super().__init__(self.config)

    @property
    def name(self) -> str:
        return "CollisionReward"

    def compute(
        self,
        observation: NDArray[np.float32],
        action: NDArray[np.float32],
        next_observation: NDArray[np.float32],
        info: Dict[str, Any]
    ) -> float:
        """Compute collision penalty."""
        raise NotImplementedError(
            "Phase 3: Collision reward not yet implemented. "
            "Requires obstacle environment (Phase 2) and sensor pipeline (Phase 1)."
        )