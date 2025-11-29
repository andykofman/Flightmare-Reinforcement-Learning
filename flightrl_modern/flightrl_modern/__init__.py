"""
Flightmare Modern RL

Modern reinforcement learning implementation for Flightmare using:
- PyTorch for neural network backend
- Stable-Baselines3 for RL algorithms
- Gymnasium for environment interface

This replaces the deprecated flightrl (TensorFlow 1.x + stable-baselines 2.x).
"""

__version__ = "0.1.0"
__author__ = "Flightmare Community"

# Import main components for easy access
from flightrl_modern.envs.flight_env_vec import FlightEnvVec
from flightrl_modern.envs.gymnasium_wrapper import make_flight_env

__all__ = [
    "FlightEnvVec",
    "make_flight_env",
]

# Optional imports (may fail if dependencies not installed)
try:
    from flightrl_modern.tools.visualize_model import visualize_model
    from flightrl_modern.tools.rollout_recorder import record_rollouts
    __all__.extend(["visualize_model", "record_rollouts"])
except ImportError:
    pass

