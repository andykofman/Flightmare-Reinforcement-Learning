"""
PLACEHOLDER - Phase 3 Implementation
TODO: Implement composite reward builder when Phase 3 begins.

Composite reward builder for combining multiple reward components.
"""
from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray

from .base import BaseReward


class CompositeReward:
    """
    Composite reward that combines multiple reward components.

    NOT YET IMPLEMENTED - Placeholder for Phase 3.

    Design Note:
        This class does NOT inherit from BaseReward. It uses the Composite Pattern
        where it acts as a container/aggregator for multiple BaseReward components.
        
        - Does not have a `name` property (aggregates multiple named components)
        - Does not inherit from BaseReward (is a builder, not a component)
        - Contains a list of BaseReward instances that it coordinates
        
        This design allows flexible composition of reward functions without
        requiring the composite itself to be treated as a single reward component.

    Usage (when implemented):
        reward = CompositeReward([
            PositionReward(PositionRewardConfig(weight=1.0)),
            CollisionReward(CollisionRewardConfig(weight=10.0)),
        ])
        total_reward = reward.compute(obs, action, next_obs, info)
    """

    def __init__(self, components: List[BaseReward] = None):
        """
        Initialize composite reward.

        Args:
            components: List of reward components to combine
        """
        self.components = components or []

    def add_component(self, component: BaseReward) -> "CompositeReward":
        """Add a reward component."""
        self.components.append(component)
        return self

    def compute(
        self,
        observation: NDArray[np.float32],
        action: NDArray[np.float32],
        next_observation: NDArray[np.float32],
        info: Dict[str, Any]
    ) -> float:
        """Compute total reward from all components."""
        raise NotImplementedError(
            "Phase 3: Composite reward not yet implemented. "
            "Rewards are currently computed in task classes."
        )

    def reset(self) -> None:
        """Reset all components."""
        for component in self.components:
            component.reset()