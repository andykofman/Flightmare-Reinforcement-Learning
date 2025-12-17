"""
SAC Training for FlightRL v2.

Soft Actor-Critic (SAC) training implementation using Stable-Baselines3.
SAC is well-suited for continuous control tasks like quadrotor navigation.

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
from stable_baselines3.common.logger  import configure

from flightrl_v2.envs.gymnasium_wrapper import make_flight_env_for_sb3


def make_env(
        config_path: Optional[str] = None,
        render: bool = False,
        seed: Optional[int] = None,
        rank: int = 0,
) -> Callable: 
    """
    Utility function for creating a FlightRL v2 environment for SB3.

    Args:
        config_path: Path to the environment configuration file.
        render: Whether to enable rendering.
        seed: Random seed for the environment.
        rank: Rank of the environment (for vectorized envs).
    Returns:
        Callable that creates the environment.
    """
    def _init():
        env = make_flight_env_for_sb3(
            config_path=config_path,
            render=render,
            seed=seed + rank if seed is not None else None,
        )
        env = Monitor(env) # Track episode statistics
        return env
    return _init

def train_sac( 
        total_timesteps: int = 1_000_000,
        config_path: Optional[str] = None,
        log_dir: Optional[str] = None,
        save_dir: str = "./models/sac",
        render: bool = False,
        n_envs: int = 4,
        use_multiprocessing: bool = False,
        seed: Optional[int] = None,
        eval_freq: int = 10_000,
        save_freq: int = 50_000,
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 10_000,
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
    Train a Soft Actor-Critic (SAC) agent on a FlightRL v2 environment.

    Args:
        total_timesteps: Total number of training timesteps.
        config_path: Path to the environment configuration file.
        log_dir: Directory for logging training progress.
        save_dir: Directory to save trained models.
        render: Whether to enable rendering during training.
        n_envs: Number of parallel environments for training.
        use_multiprocessing: Whether to use multiprocessing for vectorized envs.
        seed: Random seed for reproducibility.
        eval_freq: Frequency (in timesteps) to evaluate the agent.
        save_freq: Frequency (in timesteps) to save the model.
        learning_rate: Learning rate for the SAC optimizer.
        buffer_size: Size of the replay buffer.
        learning_starts: Number of steps before training starts.
        batch_size: Batch size for training updates.
        tau: Soft update coefficient for target networks.
        gamma: Discount factor for rewards.
        train_freq: Frequency (in timesteps) to perform training updates.
        gradient_steps: Number of gradient steps per training update.
        policy_kwargs: Additional arguments for the policy network.
        tensorboard_log: Directory for TensorBoard logs.
        verbose: Verbosity level (0, 1, or 2).
        **kwargs: Additional keyword arguments for SAC.

        Returns:
            Trained SAC model.

        Example:
            >>> model = train_sac(total_timesteps=500_000, n_envs=8, seed=42)
            >>> model.save("sac_flightrl_model")

            """
    
    # Create directories if they don't exist
    os.makedirs(save_dir, exist_ok=True)
    if log_dir is None:
        log_dir = os.path.join(save_dir, "logs")  # will add a logging system later

    # configure logger
    if tensorboard_log is None:
        tensorboard_log = os.path.join(log_dir, "tensorboard")

    # Create vectorized environment
    # Use SubprocVecEnv for multiprocessing, else DummyVecEnv
    if n_envs == 1:
        env = make_flight_env_for_sb3(
            config_path=config_path,
            render=render,
            seed=seed, 
        )
        env = Monitor(env, log_dir)
    else: # for parallel envs
        env_fns = [
            make_env(config_path, render and i == 0, seed, i) # visualize only the first env
            for i in range(n_envs) # i is the rank
        ]
    
        if use_multiprocessing:
            env = SubprocVecEnv(env_fns)
        else:
            env = DummyVecEnv(env_fns)

    # creat evaluation environment
    eval_env = make_flight_env_for_sb3(
        config_path=config_path,
        render=False,
        seed=seed + 1000 if seed is not None else None, 
    )

    eval_env = Monitor(eval_env)

 
    # Default policy architecture if none provided
    if policy_kwargs is None:
        policy_kwargs = dict(
            net_arch=[256, 256], # Two hidden layers with 256 units each

        )

    # Create SAC model
    model = SAC(
        policy="MlpPolicy",
        env=env, 
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        seed=seed,
        **kwargs
    )

    # Setup callbacks

    callbacks = [] 

    # Checkpoint callback to save the model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_dir, 
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback to evaluate the agent periodically and save the best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)
    callback = CallbackList(callbacks)


    # Train the agent
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
        timesteps: int = 10_000,
        seed: int = 42,
        verbose: int = 1,

) -> SAC:
    """
    Quick training run for SAC on FlightRL v2 for testing purposes.

    Args:
        timesteps: Number of training timesteps.
        seed: Random seed for reproducibility.
        verbose: Verbosity level.

    Returns:
        Trained SAC model.
    """ 

    return train_sac(
        total_timesteps=timesteps,
        n_envs=1,
        use_multiprocessing=False,
        seed=seed,
        render=False,
        verbose=verbose,
        tensorboard_log=None,   
        
    )