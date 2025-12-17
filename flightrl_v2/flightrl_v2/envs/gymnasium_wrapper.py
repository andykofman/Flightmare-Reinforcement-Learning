"""
Gymnasium wrapper utilities for Flightmare environments

Purpose: Provides convenience functions to create Gymnasium-compatible environments
from flightgym backends.
"""

import os
from typing import Optional, Dict, Any
import numpy as np
import gymnasium as gym
from ruamel.yaml import YAML
from io import StringIO

from .flight_env_vec import FlightEnvVec


def make_flight_env(
    config_path: Optional[str] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    render: bool = False,
    num_envs: Optional[int] = None,
    num_threads: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs
) -> FlightEnvVec:
    """
    Create a Flightmare environment with Gymnasium interface.
    
    Args:
        config_path: Path to YAML configuration file. If None, uses default config.
        config_dict: Dictionary with configuration. Overrides config_path if provided.
        render: Enable Unity rendering (default: False)
        num_envs: Number of parallel environments (overrides config)
        num_threads: Number of threads (overrides config)
        seed: Random seed
        **kwargs: Additional config overrides
    
    Returns:
        FlightEnvVec: Gymnasium-compatible vectorized environment
    
    Example:
        >>> env = make_flight_env(render=True, num_envs=1)
        >>> obs, info = env.reset()
        >>> obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    """
    # Import flightgym (from flightlib C++ bindings)
    try:
        from flightgym import QuadrotorEnv_v1
    except ImportError as e:
        raise ImportError(
            "flightgym module not found. Make sure flightlib is installed.\n"
            "Install with: cd flightlib && pip install ."
        ) from e
    
    # Load configuration
    if config_dict is not None:
        cfg = config_dict
    elif config_path is not None:
        cfg = YAML().load(open(config_path, 'r'))
    else:
        # Use default config from flightlib
        flightmare_path = os.environ.get("FLIGHTMARE_PATH")
        if flightmare_path is None:
            raise ValueError(
                "FLIGHTMARE_PATH environment variable not set. "
                "Either set it or provide config_path/config_dict."
            )
        default_config = os.path.join(
            flightmare_path,
            "flightlib/configs/vec_env.yaml"
        )
        cfg = YAML().load(open(default_config, 'r'))
        
        # Merge quadrotor_env.yaml to get quadrotor_dynamics section
        quadrotor_config = os.path.join(
            flightmare_path,
            "flightlib/configs/quadrotor_env.yaml"
        )
        if os.path.exists(quadrotor_config):
            quad_cfg = YAML().load(open(quadrotor_config, 'r'))
            # Merge quadrotor sections into main config
            if "quadrotor_env" in quad_cfg:
                cfg["quadrotor_env"] = quad_cfg["quadrotor_env"]
            if "quadrotor_dynamics" in quad_cfg:
                cfg["quadrotor_dynamics"] = quad_cfg["quadrotor_dynamics"]
            if "rl" in quad_cfg:
                cfg["rl"] = quad_cfg["rl"]
    
    # Apply overrides

    """
    Priority: config_dict > config_path > default
    - If config_dict is provided, it overrides config_path
    - If config_path is provided, it overrides default
    - If neither is provided, it uses default
    """
    if num_envs is not None:
        if "env" not in cfg:
            cfg["env"] = {}
        cfg["env"]["num_envs"] = num_envs
    
    if num_threads is not None:
        if "env" not in cfg:
            cfg["env"] = {}
        cfg["env"]["num_threads"] = num_threads
    
    if render:
        if "env" not in cfg:
            cfg["env"] = {}
        cfg["env"]["render"] = "yes"
    else:
        if "env" not in cfg:
            cfg["env"] = {}
        cfg["env"]["render"] = "no"
    
    # Apply additional kwargs
    for key, value in kwargs.items():
        if "env" not in cfg:
            cfg["env"] = {}
        cfg["env"][key] = value
    
    # Convert config to YAML string for C++ backend
    yaml = YAML()
    yaml.default_flow_style = False
    stream = StringIO()
    yaml.dump(cfg, stream)
    config_yaml = stream.getvalue()
    
    # Create flightgym backend
    quad_env = QuadrotorEnv_v1(config_yaml, False)
    
    # Extract max_episode_steps from kwargs if provided, otherwise use default
    max_episode_steps = kwargs.pop("max_episode_steps", 300)
    
    # Wrap in Gymnasium interface
    env = FlightEnvVec(quad_env, max_episode_steps=max_episode_steps)
    
    # Set seed if provided
    if seed is not None:
        env.seed(seed)
    
    return env


def make_flight_env_for_sb3(
    config_path: Optional[str] = None,
    render: bool = False,
    seed: Optional[int] = None,
    **kwargs
) -> gym.Env:
    """
    Create a single Flightmare environment optimized for SB3 VecEnv.
    
    This creates a single environment that can be used with SB3's
    DummyVecEnv or SubprocVecEnv for better multi-processing.
    
    Args:
        config_path: Path to YAML configuration file
        render: Enable Unity rendering
        seed: Random seed
        **kwargs: Additional config overrides
    
    Returns:
        Gymnasium environment suitable for SB3 vectorization
    
    Example:
        >>> from stable_baselines3.common.vec_env import DummyVecEnv
        >>> env = DummyVecEnv([lambda: make_flight_env_for_sb3() for _ in range(4)])
    """
    # Extract max_episode_steps if provided
    max_episode_steps = kwargs.pop("max_episode_steps", 300)
    
    # For SB3, we always want num_envs=1 per process
    env = make_flight_env(
        config_path=config_path,
        render=render,
        num_envs=1,
        num_threads=1,
        seed=seed,
        max_episode_steps=max_episode_steps,
        **kwargs
    )
    
    return env


def configure_random_seed(seed: int, env: Optional[gym.Env] = None):
    """
    Configure random seed for reproducibility.
    
    Args:
        seed: Random seed value
        env: Environment to seed (optional)
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Try to seed PyTorch if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Make CUDA operations deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # Seed environment
    if env is not None:
        env.seed(seed)


class FlightEnvWrapper(gym.Wrapper):
    """
    Additional wrapper for compatibility and extra features.
    
    This can be used to add additional pre-processing, reward shaping,
    or observation filtering.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
    
    def reset(self, **kwargs):
        """Reset with additional processing"""
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step with additional processing"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Add any custom processing here
        # For example, reward shaping, observation normalization, etc.
        
        return obs, reward, terminated, truncated, info
