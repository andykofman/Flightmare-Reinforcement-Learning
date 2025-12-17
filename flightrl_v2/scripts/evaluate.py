#!/usr/bin/env python3
"""
Unified evaluation script for flightrl_v2.

Usage:
    python evaluate.py --model ./models/sac/best_model.zip --episodes 10
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC

from flightrl_v2.algorithms import evaluate_policy
from flightrl_v2.envs import make_flight_env_for_sb3


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")

    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to trained model (.zip)"
    )
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render with Unity"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Handle deterministic flag
    deterministic = not args.stochastic

    print(f"Evaluating model: {args.model}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Deterministic: {deterministic}")
    print(f"  Render: {args.render}")

    # Load model
    model = SAC.load(args.model)

    # Create environment
    env = make_flight_env_for_sb3(seed=args.seed, render=args.render)

    # Evaluate
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.episodes,
        deterministic=deterministic
    )

    print(f"\nResults:")
    print(f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()
    return 0


if __name__ == "__main__":
    exit(main())