"""
PLACEHOLDER - Phase 1 Implementation
TODO: Implement when depth camera is exposed via pybind.

Depth camera sensor interface.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .base import BaseSensor, SensorConfig


@dataclass
class DepthCameraConfig(SensorConfig):
    """Depth camera configuration."""
    width: int = 64
    height: int = 48
    fov: float = 90.0  # degrees
    min_depth: float = 0.1
    max_depth: float = 10.0


class DepthCameraSensor(BaseSensor):
    """
    Depth camera sensor interface.

    NOT YET IMPLEMENTED - Placeholder for Phase 1.

    Provides depth image observations for obstacle detection
    and navigation.
    """

    def __init__(self, config: Optional[DepthCameraConfig] = None):
        self.config: DepthCameraConfig = config or DepthCameraConfig()
        super().__init__(self.config)

    @property
    def name(self) -> str:
        return "depth_camera"

    @property
    def observation_shape(self) -> tuple:
        return (self.config.height, self.config.width)

    def get_observation(self) -> NDArray[np.float32]:
        """Get depth image."""
        raise NotImplementedError(
            "Phase 1: Depth camera not yet implemented. "
            "Requires extending pybind_wrapper.cpp to expose depth images."
        )

    def reset(self) -> None:
        """Reset depth camera state."""
        pass