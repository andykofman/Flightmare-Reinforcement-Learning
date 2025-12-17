"""
Tools for model visualization and rollout recording.

This module provides utilities for:
- Recording rollouts from trained models to CSV files
- Visualizing model trajectories in 3D
- Analyzing episode performance metrics

Example:
    Record rollouts from a trained model:
    
    ```python
    from flightrl_v2.tools import record_rollouts
    
    session_dir, episodes = record_rollouts(
        model_path="./models/hover_sac/final_model.zip",
        n_episodes=3,
        output_dir="./visualizations"
    )
    ```
"""

from flightrl_v2.tools.rollout_recorder import (
    RolloutRecorder,
    RolloutConfig,
    EpisodeSummary,
    record_rollouts,
)
from flightrl_v2.tools.visualize_model import visualize_model

__all__ = [
    "RolloutRecorder",
    "RolloutConfig",
    "EpisodeSummary",
    "record_rollouts",
    "visualize_model",
]

