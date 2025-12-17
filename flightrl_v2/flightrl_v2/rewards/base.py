"""
PLACEHOLDER - Phase 3 Implementation
TODO: Implement modular reward system when Phase 3 begins.

Base reward class for modular reward functions.
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

@dataclass
class RewardConfig:
    """
    Base configuration for reward functions.
    """
    weight: float = 1.0  # Weight of the reward component
    enabled: bool = True  # Whether this reward component is enabled

class BaseReward(ABC):
    """
    Abstract base class for modular reward functions.

    NOT IMPLEMENTED - Placeholder for Phase 3.

    
    Reward functions are modular components that can be combined
    to create composite reward signals for training.
    """
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the reward function.
        Returns:
            str: Name of the reward function.
        """
        raise NotImplementedError
    
    @abstractmethod
    def compute(
        self,
        observation: NDArray[np.float32],
        action: NDArray[np.float32],
        next_observation: NDArray[np.float32],
        info: Dict[str, Any],
    ) -> float:
        """
        Compute the reward based on the current and next observations, action, and info.
       
        Args:
            observation: Current observation.
            action: Action taken.
            next_observation: Next observation after action.
            info: Additional info from the environment.
        Returns:
            float: Computed reward value.
          """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset reward function state."""
        pass