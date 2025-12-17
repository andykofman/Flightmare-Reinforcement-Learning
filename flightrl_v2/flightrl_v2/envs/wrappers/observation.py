"""
PLACEHOLDER - Phase 1 Implementation. Note: check readme for phases.
TODO: Implement observation transformation wrappers when Phase 1 begins.

Observation wrappers will handle:
- Normalization
- Feature extraction
- Sensor data processing
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ObservationWrapper(gym.ObservationWrapper):
    """
    Base class for observation transformation wrappers.

    NOT YET IMPLEMENTED. Placeholder for Phase 1.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Placeholder: Observation stays the same for now

    def observation(self, observation):
        """Transform the observation. Override in subclasses."""
        raise NotImplementedError(
            "Phase 1: Observation wrappers not yet implemented. "
            "This is a placeholder for future sensor processing."
        )