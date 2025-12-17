"""
Core abstractions for flightrl_v2.

This module provides base classes and type definitions used throughout
the package.
"""
from .base_env import BaseFlightEnv
from .base_task import BaseTask, TaskConfig
from .types import (
    ActionType,
    DoneType,
    EpisodeStats,
    EvaluationConfig,
    InfoType,
    ObservationType,
    ResetReturnType,
    RewardType,
    StepReturnType,
    TrainingConfig,
)

__all__ = [
    "BaseFlightEnv",
    "BaseTask",
    "TaskConfig",
    "ActionType",
    "DoneType",
    "EpisodeStats",
    "EvaluationConfig",
    "InfoType",
    "ObservationType",
    "ResetReturnType",
    "RewardType",
    "StepReturnType",
    "TrainingConfig",
]