"""
PLACEHOLDER - Phase 3 Implementation
TODO: Implement position-based rewards when Phase 3 begins.

Position-based reward functions.
"""
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from .base import BaseReward, RewardConfig


@dataclass
class PositionRewardConfig(RewardConfig):
    """Configuration for position rewards."""
    target_position: Tuple[float, float, float] = (0.0, 0.0, 5.0)
    use_exponential: bool = False  # Exponential vs linear distance penalty


class PositionReward(BaseReward):
    """
    Position-based reward: Penalize distance to target.
     NOT YET IMPLEMENTED - Placeholder for Phase 3.

  """
    
    def __init__(self, config: PositionRewardConfig = None):
        self.config: PositionRewardConfig = config or PositionRewardConfig()    
        super().__init__(self.config)
        self._target = np.array(self.config.target_position, dtype=np.float32)


    @property
    def name(self) -> str:
        return "PositionReward"
    
    def compute(
            self,
            observation: NDArray[np.float32],
            action: NDArray[np.float32],
            next_observation: NDArray[np.float32],
            info: Dict[str, Any],
    ) -> float:
        """
        Compute position-based reward.
        Args:
            observation: Current observation.
            action: Action taken.
            next_observation: Next observation after action.
            info: Additional info.
        Returns:
            float: Computed reward.
        """
        raise NotImplementedError("PositionReward is not yet implemented. Placeholder for Phase 3.")