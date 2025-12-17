"""
PLACEHOLDER - Phase 2/3 Implementation
TODO: Implement curriculum callback when curriculum learning is added.

Curriculum learning callback for progressive difficulty adjustment.

ROLE: Decision Maker (WHEN to change difficulty)
- Monitors training performance metrics (success rate, episode rewards)
- Decides WHEN the agent is ready for harder difficulty
- Triggers difficulty changes by calling CurriculumWrapper.advance_level()
- Logs curriculum progression for TensorBoard

This works together with envs/wrappers/curriculum.py:
- Callback (this file) = Decision maker - monitors & decides
- Wrapper (envs/wrappers) = Executor - implements difficulty changes
"""
from typing import Any, Dict, Optional

from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """
    Callback for curriculum learning during training.

    NOT YET IMPLEMENTED - Placeholder for Phase 2/3.

    This callback will:
    - Monitor training progress
    - Adjust environment difficulty based on success rate
    - Log curriculum level changes
    """

    def __init__(
        self,
        advancement_threshold: float = 0.8,
        eval_freq: int = 10000,
        verbose: int = 0
    ):
        """
        Initialize curriculum callback.

        Args:
            advancement_threshold: Success rate required to advance
            eval_freq: Evaluation frequency for advancement check
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.advancement_threshold = advancement_threshold
        self.eval_freq = eval_freq
        self.current_level = 0

    def _on_step(self) -> bool:
        """Called at each step. Returns False to stop training."""
        raise NotImplementedError(
            "Phase 2/3: Curriculum callback not yet implemented. "
            "Use standard SB3 callbacks until curriculum learning is added."
        )