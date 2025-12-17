"""
PLACEHOLDER - Phase 1 Implementation
TODO: Implement when IMU interface is formalized.

IMU (Inertial Measurement Unit) sensor interface.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .base import BaseSensor, SensorConfig


@dataclass
class IMUConfig(SensorConfig):
    """IMU sensor configuration."""
    accelerometer_noise: float = 0.01
    gyroscope_noise: float = 0.001


class IMUSensor(BaseSensor):
    """
    IMU sensor interface.

    NOT YET IMPLEMENTED - Placeholder for Phase 1.

    Provides acceleration and angular velocity measurements.
    Note: Basic IMU data is already in the observation space,
    this class will provide a cleaner interface.
    """

    def __init__(self, config: Optional[IMUConfig] = None):
        self.config: IMUConfig = config or IMUConfig()
        super().__init__(self.config)

    @property
    def name(self) -> str:
        return "imu"

    @property
    def observation_shape(self) -> tuple:
        return (6,)  # 3 accel + 3 gyro

    def get_observation(self) -> NDArray[np.float32]:
        """Get IMU measurements (acceleration + angular velocity)."""
        raise NotImplementedError(
            "Phase 1: IMU sensor interface not yet implemented. "
            "Basic IMU data is available in observation space."
        )

    def reset(self) -> None:
        """Reset IMU state."""
        pass