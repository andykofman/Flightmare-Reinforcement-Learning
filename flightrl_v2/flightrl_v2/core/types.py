"""
Core type definitions for flightrl_v2.

This module contains TypedDict definitions, type aliases, and dataclasses
used throughout the package for type safety and documentation.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Type aliases
ObservationType = NDArray[np.float32]
ActionType = NDArray[np.float32]
RewardType = float
DoneType = bool
InfoType = Dict[str, Any]

# Gymnasium step return type
StepReturnType = Tuple[ObservationType, RewardType, DoneType, DoneType, InfoType]
ResetReturnType = Tuple[ObservationType, InfoType]


@dataclass
class EpisodeStats:
    """Statistics for a single episode."""
    total_reward: float = 0.0
    episode_length: int = 0
    min_distance_to_target: float = float('inf')
    final_distance_to_target: float = float('inf')
    success: bool = False

@dataclass
class TrainingConfig:
    """Configuration for training runs."""
    total_timesteps: int = 1_000_000
    n_envs: int = 1
    seed: int = 42
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    save_freq: int = 10_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    save_dir: str = "./models"
    log_dir: Optional[str] = None  # Defaults to {save_dir}/logs if None


@dataclass
class EvaluationConfig:
    """Configuration for policy evaluation."""
    n_episodes: int = 10
    deterministic: bool = True
    render: bool = False
    record_video: bool = False
    video_dir: str = "./videos"