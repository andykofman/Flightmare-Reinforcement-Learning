"""  
Unified training script for flightrl_v2.


Usage: 
    python train.py --task <task_name> --algorithm <algorithm_name> --timesteps <num_timesteps> 
    python train.py --config <config_file.yaml>
"""

import argparse
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from flightrl_v2.algorithms import train_ppo, train_sac, train_td3
from flightrl_v2.configs import load_config

ALGORITHMS = {
    "sac": train_sac,   
    "ppo": train_ppo, # Stub - will raise NotImplementedError
    "td3": train_td3, # Stub - will raise NotImplementedError
}

def main():
    parser = argparse.ArgumentParser(description="Train RL agent for quadrotor tasks.")

    # Config file (highest priority)
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to the configuration YAML file."
    )

    # Command line overrides
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="hover",
        choices=["hover", "target_reaching", "obstacle_avoidance"],
        help="Name of the task to train on."
    )

    # Algorithm selection
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        default="sac",
        choices=list(ALGORITHMS.keys()),
        help="RL algorithm to use for training."
    )

    # Training timesteps
    parser.add_argument(
        "--timesteps", "-ts",
        type=int,
        default=1_000_000,
        help="Total number of training timesteps."
    )

    # Number of parallel environments
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments to use for training."
    )

    # Seed
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )

    # Save directory
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./models",
        help="Directory to save trained models and logs."
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for TensorBoard logs (defaults to save_dir/logs)"
    )

    args = parser.parse_args()

    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Override config with command line arguments
    training_config = config.get("training", {})
    algorithm = args.algorithm or training_config.get("algorithm", "sac")
    timesteps = args.timesteps or training_config.get("total_timesteps", 1_000_000)
    n_envs = args.n_envs or training_config.get("n_envs", 1)
    seed = args.seed or training_config.get("seed", 42)
    save_dir = args.save_dir or training_config.get("save_dir", "./models")
    log_dir = args.log_dir or training_config.get("log_dir")  # None if not specified


    # Get training function
    train_fn = ALGORITHMS.get(algorithm)
    if train_fn is None:
        print(f"Unknown algorithm: {algorithm}")
        print(f"Available: {list(ALGORITHMS.keys())}")
        return 1

    print(f"Starting training:")
    print(f"  Task: {args.task}")
    print(f"  Algorithm: {algorithm}")
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Environments: {n_envs}")
    print(f"  Seed: {seed}")
    print(f"  Save dir: {save_dir}")
    print(f"  Log dir: {log_dir or save_dir + '/logs (auto)'}")

    # Run training
    try:
        model = train_fn(
            total_timesteps=timesteps,
            n_envs=n_envs,
            seed=seed,
            save_dir=save_dir,
            log_dir=log_dir,
        )
        print(f"\nTraining complete! Model saved to {args.save_dir}")
    except NotImplementedError as e:
        print(f"\nError: {e}")
        print("Try using --algorithm sac (the only fully implemented algorithm)")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())