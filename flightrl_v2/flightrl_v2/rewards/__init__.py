"""
PLACEHOLDER - Phase 3 Implementation
Reward function definitions for flightrl_v2.

This module provides modular reward functions that can be composed
for different training objectives.
"""
from .base import BaseReward, RewardConfig
from .position import PositionReward, PositionRewardConfig
from .collision import CollisionReward, CollisionRewardConfig
from .composite import CompositeReward

__all__ = [
    "BaseReward",
    "RewardConfig",
    "PositionReward",
    "PositionRewardConfig",
    "CollisionReward",
    "CollisionRewardConfig",
    "CompositeReward",
]