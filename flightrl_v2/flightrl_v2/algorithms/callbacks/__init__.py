"""
Custom training callbacks for flightrl_v2.

These callbacks extend Stable-Baselines3's callback system for
task-specific monitoring and curriculum learning.
"""
from .curriculum import CurriculumCallback
from .metrics import MetricsCallback

__all__ = ["CurriculumCallback", "MetricsCallback"]