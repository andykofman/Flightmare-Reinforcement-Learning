#!/usr/bin/env python3
"""
Evaluate a trained SAC agent on Flightmare

This script loads a trained model and evaluates it on the environment.

Usage:
    python evaluate_sac.py --model ./models/sac/best_model.zip --episodes 10
    python evaluate_sac.py --model ./models/sac/best_model.zip --render --deterministic
"""

import os
import argparse
from stable_baselines3 import SAC
from flightrl_modern.envs.gymnasium_wrapper import make_flight_env_for_sb3
from flightrl_modern.algorithms.evaluate import evaluate_policy, test_policy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained SAC agent on Flightmare",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.zip file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to environment config YAML file'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of episodes to evaluate'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Enable Unity rendering'
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='Use deterministic actions (no exploration noise)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed'
    )
    parser.add_argument(
        '--save_video',
        action='store_true',
        help='Save video recordings of episodes'
    )
    parser.add_argument(
        '--video_folder',
        type=str,
        default='./videos',
        help='Folder to save videos'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    print("\n" + "="*60)
    print("Flightmare SAC Evaluation")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Render: {args.render}")
    print("="*60 + "\n")
    
    # Load the trained model
    print("Loading model...")
    model = SAC.load(args.model)
    print("Model loaded successfully!\n")
    
    # Create evaluation environment
    print("Creating environment...")
    env = make_flight_env_for_sb3(
        config_path=args.config,
        render=args.render,
        seed=args.seed,
    )
    print("Environment created!\n")
    
    # Connect to Unity if rendering
    if args.render:
        print("Connecting to Unity...")
        env.connect_unity()
        print("Unity connected!\n")
    
    # Run evaluation
    if args.save_video or args.render:
        # Detailed testing with visualization
        episode_infos = test_policy(
            model=model,
            env=env,
            n_episodes=args.episodes,
            deterministic=args.deterministic,
            render=args.render,
            save_video=args.save_video,
            video_folder=args.video_folder,
        )
    else:
        # Simple evaluation for statistics
        mean_reward, std_reward = evaluate_policy(
            model=model,
            env=env,
            n_eval_episodes=args.episodes,
            deterministic=args.deterministic,
            render=False,
        )
    
    # Cleanup
    if args.render:
        print("\nDisconnecting from Unity...")
        env.disconnect_unity()
    env.close()
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
