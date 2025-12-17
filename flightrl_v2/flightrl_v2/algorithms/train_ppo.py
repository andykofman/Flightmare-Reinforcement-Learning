"""
PLACEHOLDER - PPO Training Implementation
TODO: Implement when additional algorithms are needed.

PPO (Proximal Policy Optimization) training utilities.
"""

from typing import Any, Callable, Dict, Optional, Type, Union

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv


def train_ppo(
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    seed: int = 42,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    save_dir: str = "./models/ppo",
    log_dir: Optional[str] = None,  # Defaults to {save_dir}/logs if None
    **kwargs
) -> PPO:
    """
    Train a PPO agent on the flight environment.

    NOT YET IMPLEMENTED - Placeholder for future expansion.

    Args:
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        seed: Random seed
        learning_rate: Learning rate
        n_steps: Steps per environment per update
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clip range
        save_dir: Directory to save models
        log_dir: Directory for TensorBoard logs
        **kwargs: Additional arguments

    Returns:
        Trained PPO model
    """
    raise NotImplementedError(
        "PPO training not yet implemented. Use train_sac() instead, "
        "or implement this function for PPO-specific training."
    )