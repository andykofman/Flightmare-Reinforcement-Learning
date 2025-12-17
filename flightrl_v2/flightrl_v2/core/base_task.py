"""
Base task class for flightrl_v2.

Tasks define the objective of the RL agent, including:

- Observation and action spaces
- Reward computation
- Success and failure conditions
- Episode termination logic
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import numpy as np
from numpy.typing import NDArray

from .types import ActionType, InfoType, ObservationType, RewardType



@dataclass
class TaskConfig:
    """
    Base configuration for RL tasks.
    """
    max_episode_steps: int = 300
    success_threshold: float = 0.5  # Example threshold for success condition
    success_hold_time: float = 1.0  # Time to hold success condition to be considered successful


class BaseTask(ABC):
    """
    Abstract base class for RL tasks.

    A task defines what the agent should learn to do. It specifies:
    - How to compute rewards
    - When to terminate an episode
    - What constitutes success or failure

    This separation allows the same environment to be used for different tasks.
    """

    def __init__(self, config: Optional[TaskConfig] = None):

        """
        Initialize the task with optional configuration.
        Args:
            config: Task configuration. Uses defaults if None.
        """
        self.config = config or TaskConfig()
        self._step_count = 0

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the task.
        Returns:
            str: Name of the task.
        """
        raise NotImplementedError
    
    @abstractmethod
    def compute_reward(
        self,
        observation: ObservationType,
        action: ActionType,
        next_observation: ObservationType,
        info: InfoType
    ) -> Tuple[RewardType, InfoType]:
        """
        Compute the reward for a given transition.

        Args:
            observation: State before action
            action: Action taken
            next_observation: State after action
            info: Additional information dictionary

        Returns:
            Tuple[RewardType, InfoType]: A tuple containing:
                - reward: Scalar reward value for the transition.
                - info: Dictionary with additional logging information.

        Example:
            Tasks can provide detailed reward breakdowns for analysis::

                info["reward_components"] = {
                    "position": -1.5,
                    "velocity": -0.2,
                    "action": -0.01
                }
                return total_reward, info
        """
        raise NotImplementedError
    
    @abstractmethod
    def is_terminated(
        self,
        observation: ObservationType,
        info: InfoType,
    ) -> bool:
        """
        Determine if the episode terminated (goal reached or failure).
        Args:
            observation: Current observation state.
            info: Additional info from the environment step.
        Returns:
            bool: True if the episode should terminate, False otherwise.
        """
        raise NotImplementedError
    
    def is_truncated(
        self,
        observation: ObservationType,
        info: InfoType,
    ) -> bool:
        """
        Determine if the episode was truncated (max steps reached).
        Args:
            observation: Current observation state.
            info: Additional info from the environment step.
        Returns:
            bool: True if the episode was truncated, False otherwise.
        """
        return self._step_count >= self.config.max_episode_steps
    
    def reset(self) -> None:
        """Reset task state for new episode."""
        self._step_count = 0

    def step(self) -> None:
        """Called each environment step to update internal state."""
        self._step_count += 1

    @abstractmethod
    def get_success_info(self, observation: ObservationType, info: InfoType) -> Dict[str, Any]:
        """
        Get detailed information about success/progress towards the task goal.
        Args:
            observation: Current observation state.
            info: Additional info from the environment step.
        Returns:
            Dictionary with success metrics
        
        """
        raise NotImplementedError