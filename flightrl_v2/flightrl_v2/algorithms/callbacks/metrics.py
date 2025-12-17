"""
PLACEHOLDER - Custom metrics callback
TODO: Implement task-specific metrics logging.

Custom metrics callback for logging task-specific metrics to TensorBoard.
"""
from typing import Any, Callable, Dict, List, Optional

from stable_baselines3.common.callbacks import BaseCallback


class MetricsCallback(BaseCallback):
    """
    Callback for logging custom metrics during training.

    NOT YET IMPLEMENTED - Placeholder for custom metrics.

    This callback will log task-specific metrics such as:
    - Success rate
    - Average distance to target
    - Collision rate (for obstacle avoidance)
    - Custom reward components
    """

    def __init__(
        self,
        metrics_functions: Optional[Dict[str, Callable]] = None,
        log_freq: int = 1000,
        verbose: int = 0
    ):
        """
        Initialize metrics callback.

        Args:
            metrics_functions: Dict mapping metric names to compute functions
            log_freq: Frequency of metric logging
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.metrics_functions = metrics_functions or {}
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        """Called at each step. Returns False to stop training."""
        raise NotImplementedError(
            "Custom metrics callback not yet implemented. "
            "Use standard SB3 logging until custom metrics are needed."
        )