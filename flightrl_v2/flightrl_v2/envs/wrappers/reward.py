"""
PLACEHOLDER - Phase 3 Implementation
TODO: Implement reward shaping wrappers when Phase 3 begins.

Reward wrappers will handle:
- Reward shaping
- Reward normalization
- Composite rewards
"""
import gymnasium as gym


class RewardWrapper(gym.RewardWrapper):
    """
    Base class for reward transformation wrappers.

    NOT YET IMPLEMENTED - Placeholder for Phase 3.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reward(self, reward):
        """Transform the reward. Override in subclasses."""
        raise NotImplementedError(
            "Phase 3: Reward wrappers not yet implemented. "
            "This is a placeholder for future reward shaping."
        )