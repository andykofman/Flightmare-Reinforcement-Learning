"""
flightrl_v2 - Modern Reinforcement Learning for Flightmare.

A PyTorch + Stable-Baselines3 + Gymnasium based RL framework for training
quadrotor control policies in the Flightmare simulator.

This package provides a modular, extensible architecture for reinforcement
learning research with quadrotors. It includes:

- Gymnasium-compatible environment wrappers
- Modular task definitions for different control objectives
- Sensor simulation (LIDAR, depth cameras, IMU)
- Reward shaping components
- Training utilities for SAC, PPO, and TD3
- Deployment tools for real hardware (ArduPilot integration)
- Visualization and analysis tools

Quick Start:
    >>> from flightrl_v2 import make_flight_env_for_sb3, train_sac
    >>> 
    >>> # Create environment
    >>> env = make_flight_env_for_sb3(seed=42)
    >>> 
    >>> # Train SAC agent
    >>> model = train_sac(
    ...     total_timesteps=1000000,
    ...     n_envs=16,
    ...     log_dir="./logs",
    ...     save_dir="./models"
    ... )

For more examples, see the `examples/` directory.

Package Structure:
    core/           - Base classes and type definitions
    envs/           - Environment implementations and wrappers
    tasks/          - Task definitions (hover, target reaching, etc.)
    sensors/        - Sensor simulation modules
    rewards/        - Reward function components
    algorithms/     - Training and evaluation utilities
    configs/        - Configuration loading and validation
    deployment/     - Model export and hardware deployment
    tools/          - Rollout recording and analysis
    visualization/  - Trajectory and training visualization
"""
__version__ = "2.0.0"

# Core abstractions
from .core import (
    BaseFlightEnv,
    BaseTask,
    TaskConfig,
)

# Environment creation and utilities
from .envs import (
    FlightEnvVec,
    make_flight_env_for_sb3,
    configure_random_seed,
)

# Task implementations
from .tasks import (
    HoverTask,
    TargetReachingTask,
    ObstacleAvoidanceTask,
)

# Training algorithms
from .algorithms import (
    train_sac,
    evaluate_policy,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "BaseFlightEnv",
    "BaseTask",
    "TaskConfig",
    # Environments
    "FlightEnvVec",
    "make_flight_env_for_sb3",
    "configure_random_seed",
    # Tasks
    "HoverTask",
    "TargetReachingTask",
    "ObstacleAvoidanceTask",
    # Algorithms
    "train_sac",
    "evaluate_policy",
]