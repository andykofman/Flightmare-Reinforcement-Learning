"""
SAC Training for Flightmare

Soft Actor-Critic (SAC) training implementation using Stable-Baselines3.
SAC is well-suited for continuous control tasks like quadrotor flight.
"""

import os
from typing import Optional, Dict, Any, Callable
import numpy as np
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from flightrl_modern.envs.gymnasium_wrapper import make_flight_env_for_sb3


def make_env(
    config_path: Optional[str] = None,
    render: bool = False,
    seed: Optional[int] = None,
    rank: int = 0,
) -> Callable: ### Closure to create environment
    """
    Utility function for multiprocessed env.
    
    Args:
        config_path: Path to environment config
        render: Enable rendering
        seed: Random seed
        rank: Index of subprocess (Different seed for each environment)
    
    Returns:
        Callable that creates the environment
    """
    def _init(): ### Closure to create environment
        env = make_flight_env_for_sb3(
            config_path=config_path,
            render=render,
            seed=seed + rank if seed is not None else None,
        )
        env = Monitor(env) ### Track episode statistics (SB3 wrapper)
        return env ### Return the environment   
    
    return _init


def train_sac(
    total_timesteps: int = 1000000,
    config_path: Optional[str] = None,
    log_dir: str = "./logs/sac",
    save_dir: str = "./models/sac",
    render: bool = False,
    n_envs: int = 1,
    use_multiprocessing: bool = False,
    seed: Optional[int] = None,
    eval_freq: int = 10000,
    save_freq: int = 50000,
    learning_rate: float = 3e-4,
    buffer_size: int = 1000000,
    learning_starts: int = 10000,
    batch_size: int = 256,
    tau: float = 0.005,
    gamma: float = 0.99,
    train_freq: int = 1,
    gradient_steps: int = 1,
    policy_kwargs: Optional[Dict[str, Any]] = None,
    tensorboard_log: Optional[str] = None,
    verbose: int = 1,
    **kwargs
) -> SAC:
    """
    Train a SAC agent on Flightmare environment.
    
    Args:
        total_timesteps: Total number of timesteps to train
        config_path: Path to environment configuration file
        log_dir: Directory for logs
        save_dir: Directory to save models
        render: Enable Unity rendering
        n_envs: Number of parallel environments
        use_multiprocessing: Use SubprocVecEnv instead of DummyVecEnv
        seed: Random seed for reproducibility
        eval_freq: Evaluate every N timesteps
        save_freq: Save checkpoint every N timesteps
        learning_rate: Learning rate for optimizer
        buffer_size: Size of replay buffer
        learning_starts: Start training after N steps
        batch_size: Minibatch size
        tau: Soft update coefficient
        gamma: Discount factor
        train_freq: Update the model every N steps
        gradient_steps: How many gradient steps per update
        policy_kwargs: Additional policy arguments
        tensorboard_log: Tensorboard log directory
        verbose: Verbosity level
        **kwargs: Additional SAC arguments
    
    Returns:
        Trained SAC model
    
    Example:
        >>> model = train_sac(total_timesteps=100000, n_envs=4, seed=42)
        >>> model.save("quadrotor_sac.zip")
    """
    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Configure logger
    if tensorboard_log is None:
        tensorboard_log = log_dir
    
    # Create environments
    if n_envs == 1:
        env = make_flight_env_for_sb3(
            config_path=config_path,
            render=render,
            seed=seed,
        )
        env = Monitor(env, log_dir)
    else:   #### For parallel environments (multi-processing)
        # Create vectorized environments
        env_fns = [
            make_env(config_path, render and i == 0, seed, i)
            for i in range(n_envs)
        ]
        
        if use_multiprocessing:   #### For parallel environments (multi-processing)
            env = SubprocVecEnv(env_fns)
        else:   #### For single environment (single-processing)
            env = DummyVecEnv(env_fns)
    
    # Create evaluation environment
    eval_env = make_flight_env_for_sb3(
        config_path=config_path,
        render=False,
        seed=seed + 1000 if seed is not None else None,
    )
    eval_env = Monitor(eval_env)
    
    # Default policy architecture
    if policy_kwargs is None:
        policy_kwargs = dict(
            net_arch=[256, 256],  # Two hidden layers with 256 units each
        )
    
    # Create SAC model
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,    # How fast to learn
        buffer_size=buffer_size,        # Replay buffer size (learn from old experiences)
        learning_starts=learning_starts, # Start training after N steps
        batch_size=batch_size,          # Minibatch size
        tau=tau,                        # Soft update coefficient
        gamma=gamma,                    # Discount factor
        train_freq=train_freq,          # Update the model every N steps
        gradient_steps=gradient_steps,  # How many gradient steps per update
        policy_kwargs=policy_kwargs,    # Additional policy arguments
        tensorboard_log=tensorboard_log, # Tensorboard log directory
        verbose=verbose,                # Verbosity level
        seed=seed,                      # Random seed
        **kwargs                        # Additional SAC arguments
    )
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_dir,
        name_prefix="sac_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback - evaluate and save best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)
    
    callback = CallbackList(callbacks)
    
    # Train the model
    print(f"\n{'='*60}")
    print(f"Training SAC on Flightmare")
    print(f"{'='*60}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Number of environments: {n_envs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Buffer size: {buffer_size:,}")
    print(f"Tensorboard log: {tensorboard_log}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*60}\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=10,
        progress_bar=True,
    )
    
    # Save final model
    final_model_path = os.path.join(save_dir, "sac_final")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    return model


class ProgressBarCallback(BaseCallback):
    """
    Custom callback for displaying a progress bar during training.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.pbar = None
    
    def _on_training_start(self) -> None:
        try:
            from tqdm import tqdm
            self.pbar = tqdm(total=self.locals.get('total_timesteps', 0))
        except ImportError:
            self.pbar = None
    
    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.update(1)
        return True
    
    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


def quick_train_sac(
    timesteps: int = 10000,
    seed: int = 42,
    verbose: int = 1,
) -> SAC:
    """
    Quick training function for testing and smoke tests.
    
    Args:
        timesteps: Number of training steps (default: 10000)
        seed: Random seed
        verbose: Verbosity level
    
    Returns:
        Trained SAC model
    """
    return train_sac(
        total_timesteps=timesteps,
        n_envs=1,
        seed=seed,
        learning_starts=100,
        eval_freq=5000,
        save_freq=5000,
        verbose=verbose,
        tensorboard_log=None,
    )
