"""
Training algorithms for flightrl_v2.

This module provides training and evaluation utilities for various
RL algorithms, primarily using Stable-Baselines3.
"""
from .evaluate import evaluate_policy, test_policy
from .train_sac import quick_train_sac, train_sac

# Stubs for future algorithms
from .train_ppo import train_ppo
from .train_td3 import train_td3

__all__ = [
    "train_sac",
    "quick_train_sac",
    "train_ppo",
    "train_td3",
    "evaluate_policy",
    "test_policy",
]





