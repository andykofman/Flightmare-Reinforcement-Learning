"""
PLACEHOLDER FILE FOR SENSOR BASE CLASSES - Phase 1 Implementation


TODO: Implement when sensor pipleine (Phase 1) begins.

Base classes for sensors in flightrl_v2.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import numpy as np
from numpy.typing import NDArray

@dataclass
class SensorConfig:
    """
    Base configuration for sensors.
    """
    enabled: bool = True
    update_rate: float = 100.0  # Hz
    noise_std: float = 0.0  # Noise standard deviation

    
class BaseSensor(ABC):
    """
    Abstract base class for sensor interfaces.

    NOT YET IMPLEMENTED - Placeholder for Phase 1.

    Sensors provide observations from the C++ physics engine to the
    Python RL environment. This base class defines the interface.
    """

    def __init__(self, config: Optional[SensorConfig] = None):
        self.config = config or SensorConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return sensor name."""
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_shape(self) -> tuple:
        """Return the shape of sensor observations."""
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> NDArray[np.float32]:
        """Get current sensor observation."""
        raise NotImplementedError(
            "Phase 1: Sensor pipeline not yet implemented. "
            "C++ pybind wrapper must be extended to expose sensor data."
        )

    @abstractmethod
    def reset(self) -> None:
        """Reset sensor state."""
        raise NotImplementedError