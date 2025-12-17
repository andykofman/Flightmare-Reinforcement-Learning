""" 
PLACEHOLDER - Phase 1 Implementation

TODO: Implement when LIDAR is expanded in the pybind wrapper.
LIDAR sensor interface for flightrl_v2.

"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .base import BaseSensor, SensorConfig

@dataclass 
class LidarConfig(SensorConfig):
    """
    Configuration for LIDAR sensor.
    """
    num_rays: int = 10 # Number of LIDAR rays
    max_range: float = 10.0 # Maximum range of LIDAR in meters
    fov_horizontal: float = 360.0 # Horizontal field of view in degrees
    fov_vertical: float = 30.0 # Vertical field of view in degrees


class LidarSensor(BaseSensor):
    """
    LIDAR sensor interface for flightrl_v2.

    NOT IMPLEMENTED YET.
     
    Provides distance measurements in multiple directions for
    obstacle detection.
    """

    def __init__(self, config: Optional[LidarConfig] = None):
        self.config: LidarConfig = config or LidarConfig()
        super().__init__(self.config)


    @property
    def name(self) -> str:
        return "lidar"
    
    @property 
    def observation_shape(self) -> tuple:
        return (self.config.num_rays,)
    
    def get_observation(self) -> NDArray[np.float32]:
        """
        Get LIDAR observation.

        Returns:
            NDArray[np.float32]: Array of distance measurements from LIDAR rays.
        """
        raise NotImplementedError("LidarSensor is not implemented yet.")
    
    def reset(self) -> None:
        """
        Reset the LIDAR sensor state.
        """
        pass