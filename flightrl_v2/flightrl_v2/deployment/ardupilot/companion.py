"""
PLACEHOLDER - Phase 5 Implementation
TODO: Implement companion computer controller when ArduPilot integration begins.

Companion computer controller for RL policy execution.
"""
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray


class CompanionController:
    """
    Companion computer controller for running RL policies.

    NOT YET IMPLEMENTED - Placeholder for Phase 5.

    This class orchestrates:
    - Sensor data collection
    - RL policy inference
    - Command translation to MAVLink
    - Safety monitoring and override
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        connection_string: str = "/dev/ttyACM0",
        control_rate: float = 50.0  # Hz
    ):
        """
        Initialize companion controller.

        Args:
            model_path: Path to exported RL model
            connection_string: MAVLink connection string
            control_rate: Control loop rate in Hz
        """
        raise NotImplementedError(
            "Phase 5: Companion controller not yet implemented."
        )

    def start(self) -> None:
        """Start the control loop."""
        raise NotImplementedError("Phase 5: Not implemented")

    def stop(self) -> None:
        """Stop the control loop."""
        raise NotImplementedError("Phase 5: Not implemented")

    def emergency_stop(self) -> None:
        """Emergency stop - disarm immediately."""
        raise NotImplementedError("Phase 5: Not implemented")