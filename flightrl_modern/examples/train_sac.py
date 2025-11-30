#!/usr/bin/env python3
"""
Train SAC Agent for Target Reaching Task

Goal: Train a quadrotor to spawn from random positions and reach target [0,0,5],
      then stabilize there.

Success Criteria:
- Distance to target < 0.5m
- Velocity < 0.5 m/s  
- Maintain for at least 1 second

Usage:
    # Quick test (10k steps)
    python train_sac.py --timesteps 10000
    
    # Full training (1M steps)
    python train_sac.py --timesteps 1000000 --n_envs 4
    
    # With rendering
    python train_sac.py --timesteps 100000 --render
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from flightrl_modern.envs.gymnasium_wrapper import (
    make_flight_env_for_sb3,
    configure_random_seed,
)


# =============================================================================
# SUCCESS TRACKING CALLBACK
# =============================================================================

class TargetReachingCallback(BaseCallback):
    """
    Custom callback to track success rate for target reaching task.
    
    Success: Drone reaches within 0.5m of target [0,0,5] with velocity < 0.5 m/s
    """
    
    TARGET = np.array([0.0, 0.0, 5.0])
    DISTANCE_THRESHOLD = 0.5  # meters
    VELOCITY_THRESHOLD = 0.5  # m/s
    
    def __init__(self, verbose: int = 0, log_freq: int = 1000):
        super().__init__(verbose)
        self.log_freq = log_freq
        
        # Tracking metrics
        self.episode_count = 0
        self.success_count = 0
        self.total_steps = 0
        
        # Per-episode tracking
        self.episode_min_distance = float('inf')
        self.episode_reached_target = False
        self.steps_at_target = 0
        
        # History for logging
        self.success_history = []
        self.distance_history = []
        
    def _on_step(self) -> bool:
        self.total_steps += 1
        
        # Get current observation
        obs = self.locals.get('new_obs', self.locals.get('obs'))
        if obs is None:
            return True
            
        # Handle vectorized envs
        if obs.ndim > 1:
            obs = obs[0]
        
        # Extract position and velocity from observation
        # obs format: [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, vel_x, vel_y, vel_z, ...]
        pos = obs[:3]
        vel = obs[6:9]
        
        # Calculate distance to target
        distance = np.linalg.norm(pos - self.TARGET)
        velocity_mag = np.linalg.norm(vel)
        
        # Track minimum distance this episode
        self.episode_min_distance = min(self.episode_min_distance, distance)
        
        # Check if at target
        at_target = distance < self.DISTANCE_THRESHOLD and velocity_mag < self.VELOCITY_THRESHOLD
        
        if at_target:
            self.steps_at_target += 1
            # Success if stable at target for 50 steps (1 second at 50Hz)
            if self.steps_at_target >= 50 and not self.episode_reached_target:
                self.episode_reached_target = True
        else:
            self.steps_at_target = 0
        
        # Check for episode end
        dones = self.locals.get('dones', self.locals.get('done'))
        if dones is not None:
            done = dones[0] if hasattr(dones, '__len__') else dones
            if done:
                self._on_episode_end()
        
        # Periodic logging
        if self.total_steps % self.log_freq == 0:
            self._log_progress()
        
        return True
    
    def _on_episode_end(self):
        """Called when an episode ends"""
        self.episode_count += 1
        
        if self.episode_reached_target:
            self.success_count += 1
        
        # Record history
        self.success_history.append(1 if self.episode_reached_target else 0)
        self.distance_history.append(self.episode_min_distance)
        
        # Reset episode tracking
        self.episode_min_distance = float('inf')
        self.episode_reached_target = False
        self.steps_at_target = 0
    
    def _log_progress(self):
        """Log progress to tensorboard and console"""
        if self.episode_count == 0:
            return
            
        success_rate = self.success_count / self.episode_count * 100
        
        # Recent success rate (last 100 episodes)
        recent_successes = self.success_history[-100:] if len(self.success_history) >= 100 else self.success_history
        recent_rate = np.mean(recent_successes) * 100 if recent_successes else 0
        
        # Average min distance
        recent_distances = self.distance_history[-100:] if len(self.distance_history) >= 100 else self.distance_history
        avg_min_dist = np.mean(recent_distances) if recent_distances else float('inf')
        
        # Log to tensorboard
        if self.logger is not None:
            self.logger.record("target/success_rate_total", success_rate)
            self.logger.record("target/success_rate_recent", recent_rate)
            self.logger.record("target/avg_min_distance", avg_min_dist)
            self.logger.record("target/episodes", self.episode_count)
            self.logger.record("target/successes", self.success_count)
        
        if self.verbose > 0:
            print(f"\n[Step {self.total_steps:,}] Episodes: {self.episode_count}, "
                  f"Success Rate: {success_rate:.1f}% (recent: {recent_rate:.1f}%), "
                  f"Avg Min Dist: {avg_min_dist:.2f}m")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            "total_episodes": self.episode_count,
            "total_successes": self.success_count,
            "success_rate": self.success_count / max(self.episode_count, 1) * 100,
            "avg_min_distance": np.mean(self.distance_history) if self.distance_history else float('inf'),
        }


# =============================================================================
# ENVIRONMENT CREATION
# =============================================================================

def make_env(config_path: Optional[str], render: bool, seed: int, rank: int):
    """Create environment factory for vectorized envs"""
    def _init():
        env = make_flight_env_for_sb3(
            config_path=config_path,
            render=render and rank == 0,
            seed=seed + rank if seed is not None else None,
        )
        return Monitor(env)
    return _init


def get_default_config_path() -> str:
    """Get path to target reaching config"""
    flightmare_path = os.environ.get("FLIGHTMARE_PATH")
    if flightmare_path:
        config_path = os.path.join(flightmare_path, "flightlib/configs/target_reaching.yaml")
        if os.path.exists(config_path):
            return config_path
    return None


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_target_reaching(
    total_timesteps: int = 500000,
    config_path: Optional[str] = None,
    n_envs: int = 1,
    seed: int = 42,
    render: bool = False,
    log_dir: str = "./logs/target_reaching",
    save_dir: str = "./models/target_reaching",
    # SAC Hyperparameters (tuned for target reaching)
    learning_rate: float = 3e-4,
    buffer_size: int = 100000,
    learning_starts: int = 5000,
    batch_size: int = 256,
    gamma: float = 0.99,
    tau: float = 0.005,
    # Training settings
    eval_freq: int = 5000,
    save_freq: int = 25000,
    verbose: int = 1,
) -> SAC:
    """
    Train SAC agent for target reaching task.
    
    Args:
        total_timesteps: Total training steps
        config_path: Path to config (uses target_reaching.yaml by default)
        n_envs: Number of parallel environments
        seed: Random seed
        render: Enable rendering (slow)
        log_dir: Directory for logs
        save_dir: Directory for saved models
        learning_rate: SAC learning rate
        buffer_size: Replay buffer size
        learning_starts: Steps before training starts
        batch_size: Batch size for updates
        gamma: Discount factor
        tau: Soft update coefficient
        eval_freq: Evaluation frequency
        save_freq: Checkpoint save frequency
        verbose: Verbosity level
    
    Returns:
        Trained SAC model
    """
    # Setup
    configure_random_seed(seed)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Use target reaching config by default
    if config_path is None:
        config_path = get_default_config_path()
        if config_path:
            print(f"Using config: {config_path}")
    
    # Create training environment
    if n_envs == 1:
        env = make_flight_env_for_sb3(
            config_path=config_path,
            render=render,
            seed=seed,
        )
        env = Monitor(env, log_dir)
    else:
        env_fns = [make_env(config_path, render, seed, i) for i in range(n_envs)]
        env = DummyVecEnv(env_fns)
    
    # Create evaluation environment
    eval_env = make_flight_env_for_sb3(
        config_path=config_path,
        render=False,
        seed=seed + 10000,
    )
    eval_env = Monitor(eval_env)
    
    # Create SAC model with good defaults for this task
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(
            net_arch=[256, 256],  # Two hidden layers
        ),
        tensorboard_log=log_dir,
        verbose=verbose,
        seed=seed,
    )
    
    # Setup callbacks
    callbacks = []
    
    # Success tracking callback
    success_callback = TargetReachingCallback(verbose=verbose, log_freq=2000)
    callbacks.append(success_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_dir,
        name_prefix="sac_target",
        save_replay_buffer=False,  # Save space
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
    )
    callbacks.append(eval_callback)
    
    callback = CallbackList(callbacks)
    
    # Print training info
    print("\n" + "="*70)
    print("  FLIGHTMARE TARGET REACHING TRAINING")
    print("="*70)
    print(f"  Task: Reach target [0, 0, 5] and stabilize")
    print(f"  Success: Distance < 0.5m, Velocity < 0.5 m/s for 1 second")
    print("-"*70)
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Environments: {n_envs}")
    print(f"  Seed: {seed}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Buffer size: {buffer_size:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gamma: {gamma}")
    print("-"*70)
    print(f"  Log directory: {log_dir}")
    print(f"  Save directory: {save_dir}")
    print("="*70 + "\n")
    
    # Train!
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=10,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    # Save final model
    final_path = os.path.join(save_dir, "sac_final")
    model.save(final_path)
    
    # Print summary
    summary = success_callback.get_summary()
    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print("="*70)
    print(f"  Total Episodes: {summary['total_episodes']}")
    print(f"  Successful Episodes: {summary['total_successes']}")
    print(f"  Success Rate: {summary['success_rate']:.1f}%")
    print(f"  Average Min Distance: {summary['avg_min_distance']:.2f}m")
    print("-"*70)
    print(f"  Final model saved to: {final_path}.zip")
    print(f"  Best model saved to: {save_dir}/best_model.zip")
    print("="*70 + "\n")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    return model


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SAC for Target Reaching Task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Training parameters
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Total training timesteps')
    parser.add_argument('--n_envs', type=int, default=1,
                        help='Number of parallel environments')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Environment
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML')
    parser.add_argument('--render', action='store_true',
                        help='Enable Unity rendering')
    
    # SAC hyperparameters
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--buffer_size', type=int, default=100000,
                        help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--learning_starts', type=int, default=5000,
                        help='Steps before training starts')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./logs/target_reaching',
                        help='Log directory')
    parser.add_argument('--save_dir', type=str, default='./models/target_reaching',
                        help='Model save directory')
    parser.add_argument('--eval_freq', type=int, default=5000,
                        help='Evaluation frequency')
    parser.add_argument('--save_freq', type=int, default=25000,
                        help='Checkpoint save frequency')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level')
    
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    
    model = train_target_reaching(
        total_timesteps=args.timesteps,
        config_path=args.config,
        n_envs=args.n_envs,
        seed=args.seed,
        render=args.render,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_starts=args.learning_starts,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        verbose=args.verbose,
    )
    
    print("\nTo evaluate the trained model:")
    print(f"  python evaluate_model.py --model {args.save_dir}/best_model.zip")
    print("\nTo view training progress:")
    print(f"  tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()
