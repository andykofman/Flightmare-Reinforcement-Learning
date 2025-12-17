"""
Base environment class for flightrl_v2.

This module provides the abstract base class that all flight environments
must inherit from, ensuring a consistent interface.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from .types import ActionType, InfoType, ObservationType, ResetReturnType, StepReturnType


class BaseFlightEnv(ABC, gym.Env):
    """
    Abstract base class for flight environments.

    All flight environments should inherit from this class to ensure
    a consistent interface across the package.

    Attributes:
        observation_space: Gymnasium observation space
        action_space: Gymnasium action space
        num_envs: Number of parallel environments (for vectorized envs)
    """

    def __init__(self, num_envs: int = 1, seed: Optional[int] = None):
        """
        Initialize the base flight environment.

        Args:
            num_envs: Number of parallel environments
            seed: Random seed for reproducibility
        """
        super().__init__()
        self._num_envs = num_envs
        self._seed = seed

    @property
    def num_envs(self) -> int:
        """Return the number of parallel environments."""
        return self._num_envs
    
    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> ResetReturnType:
        """
        Reset the environment to initial state.

        Args:
            seed: Optional seed for the random number generator
            options: Optional dictionary of reset options

        Returns:
            Tuple of (observation, info)
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, action: ActionType) -> StepReturnType:
        """
        Execute one environment step.

        Args:
            action: Action to take in the environment

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        raise NotImplementedError
    
    @abstractmethod
    def close(self) -> None:
        """Clean up environment resources."""
        raise NotImplementedError

    def seed(self, seed: Optional[int] = None) -> None:
        """Set the random seed for the environment."""
        self._seed = seed