#!/usr/bin/env python3
"""
Train a SAC agent on Flightmare quadrotor environment

This is the main training script for modern RL with Stable-Baselines3.
Replaces the legacy run_drone_control.py (TF1 + stable-baselines v2).

Usage:
    python train_sac.py --timesteps 1000000 --n_envs 4 --seed 42
    python train_sac.py --config /path/to/config.yaml --render
"""

import os
import argparse
from flightrl_modern.algorithms.train_sac import train_sac
from flightrl_modern.envs.gymnasium_wrapper import configure_random_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SAC agent on Flightmare",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Training parameters
    parser.add_argument(
        '--timesteps',
        type=int,
        default=1000000,
        help='Total number of training timesteps'
    )
    parser.add_argument(
        '--n_envs',
        type=int,
        default=1,
        help='Number of parallel environments'
    )
    parser.add_argument(
        '--multiprocessing',
        action='store_true',
        help='Use SubprocVecEnv (multiprocessing) instead of DummyVecEnv'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    # Environment parameters
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to environment config YAML file'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Enable Unity rendering (slow, only recommended for debugging)'
    )
    
    # SAC hyperparameters
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=3e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--buffer_size',
        type=int,
        default=1000000,
        help='Size of replay buffer'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning_starts',
        type=int,
        default=10000,
        help='Number of steps before training starts'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor'
    )
    parser.add_argument(
        '--tau',
        type=float,
        default=0.005,
        help='Soft update coefficient'
    )
    
    # Logging and saving
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs/sac',
        help='Directory for logs'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./models/sac',
        help='Directory to save models'
    )
    parser.add_argument(
        '--eval_freq',
        type=int,
        default=10000,
        help='Evaluate every N timesteps'
    )
    parser.add_argument(
        '--save_freq',
        type=int,
        default=50000,
        help='Save checkpoint every N timesteps'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbosity level (0: no output, 1: info, 2: debug)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    configure_random_seed(args.seed)
    
    # Print configuration
    print("\n" + "="*60)
    print("Flightmare SAC Training")
    print("="*60)
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Random seed: {args.seed}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Buffer size: {args.buffer_size:,}")
    print(f"Gamma: {args.gamma}")
    print(f"Tau: {args.tau}")
    print(f"Log directory: {args.log_dir}")
    print(f"Save directory: {args.save_dir}")
    print("="*60 + "\n")
    
    # Train the model
    model = train_sac(
        total_timesteps=args.timesteps,
        config_path=args.config,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        render=args.render,
        n_envs=args.n_envs,
        use_multiprocessing=args.multiprocessing,
        seed=args.seed,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        verbose=args.verbose,
    )
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    print(f"Model saved to: {args.save_dir}")
    print(f"Logs saved to: {args.log_dir}")
    print("\nTo evaluate the trained model, run:")
    print(f"  python evaluate_sac.py --model {args.save_dir}/best_model.zip")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
