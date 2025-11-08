"""
Algorithms package for flightrl_modern

Provides training utilities and algorithm configurations for Stable-Baselines3.
"""

from flightrl_modern.algorithms.train_sac import train_sac
from flightrl_modern.algorithms.evaluate import evaluate_policy

__all__ = ["train_sac", "evaluate_policy"]
