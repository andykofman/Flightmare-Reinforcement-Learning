""" 
Task definitions for flightrl_v2.

Tasks defines the objective, reward function, 
and success conditions for the RL agent.

"""

from .base import BaseTask, TaskConfig  
from .hover import HoverTask, HoverTaskConfig
from .target_reaching import TargetReachingTask, TargetReachingTaskConfig
from .obstacle_avoidance import ObstacleAvoidanceTask, ObstacleAvoidanceTaskConfig 

__all__ = [
    "BaseTask",
    "TaskConfig",
    "HoverTask",
    "HoverTaskConfig",
    "TargetReachingTask",
    "TargetReachingTaskConfig",
    "ObstacleAvoidanceTask",
    "ObstacleAvoidanceTaskConfig",
]