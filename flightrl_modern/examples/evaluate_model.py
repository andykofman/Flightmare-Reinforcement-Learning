#!/usr/bin/env python3
"""
Evaluate a Trained Model on Target Reaching Task

Runs episodes and reports success rate, tracking metrics, and optionally renders.

Usage:
    python evaluate_model.py --model ./models/target_reaching/best_model.zip
    python evaluate_model.py --model ./models/target_reaching/best_model.zip --render --episodes 10
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import SAC
from flightrl_modern.envs.gymnasium_wrapper import make_flight_env_for_sb3


# =============================================================================
# EVALUATION
# =============================================================================

TARGET = np.array([0.0, 0.0, 5.0])
DISTANCE_THRESHOLD = 0.5  # meters
VELOCITY_THRESHOLD = 0.5  # m/s
STABLE_STEPS_REQUIRED = 50  # 1 second at 50Hz


def evaluate_episode(env, model, render: bool = False, max_steps: int = 500) -> Dict:
    """
    Run a single evaluation episode.
    
    Returns:
        Dictionary with episode metrics
    """
    obs, info = env.reset()
    
    trajectory = {
        'positions': [],
        'velocities': [],
        'distances': [],
        'actions': [],
        'rewards': [],
    }
    
    total_reward = 0
    steps_at_target = 0
    reached_target = False
    reach_step = None
    min_distance = float('inf')
    
    for step in range(max_steps):
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Extract current state
        pos = obs[:3]
        vel = obs[6:9]
        distance = np.linalg.norm(pos - TARGET)
        velocity_mag = np.linalg.norm(vel)
        
        # Record trajectory
        trajectory['positions'].append(pos.copy())
        trajectory['velocities'].append(vel.copy())
        trajectory['distances'].append(distance)
        trajectory['actions'].append(action.copy())
        
        min_distance = min(min_distance, distance)
        
        # Check if at target
        at_target = distance < DISTANCE_THRESHOLD and velocity_mag < VELOCITY_THRESHOLD
        if at_target:
            steps_at_target += 1
            if steps_at_target >= STABLE_STEPS_REQUIRED and not reached_target:
                reached_target = True
                reach_step = step
        else:
            steps_at_target = 0
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        trajectory['rewards'].append(reward)
        
        if render:
            env.render()
        
        if terminated or truncated:
            break
    
    return {
        'success': reached_target,
        'reach_step': reach_step,
        'total_reward': total_reward,
        'min_distance': min_distance,
        'final_distance': trajectory['distances'][-1] if trajectory['distances'] else float('inf'),
        'episode_length': len(trajectory['positions']),
        'trajectory': trajectory,
    }


def evaluate_model(
    model_path: str,
    config_path: Optional[str] = None,
    n_episodes: int = 20,
    render: bool = False,
    max_steps: int = 500,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate a trained model over multiple episodes.
    
    Args:
        model_path: Path to saved model (.zip)
        config_path: Path to environment config
        n_episodes: Number of evaluation episodes
        render: Enable rendering
        max_steps: Maximum steps per episode
        verbose: Print progress
    
    Returns:
        Dictionary with aggregate statistics
    """
    # Load model
    if verbose:
        print(f"\nLoading model from: {model_path}")
    model = SAC.load(model_path)
    
    # Create environment
    env = make_flight_env_for_sb3(
        config_path=config_path,
        render=render,
        seed=12345,
    )
    
    # Run episodes
    results = []
    
    if verbose:
        print(f"\nRunning {n_episodes} evaluation episodes...")
        print("-" * 60)
    
    for ep in range(n_episodes):
        result = evaluate_episode(env, model, render=render, max_steps=max_steps)
        results.append(result)
        
        if verbose:
            status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
            print(f"Episode {ep+1:3d}: {status} | "
                  f"Min Dist: {result['min_distance']:.2f}m | "
                  f"Final Dist: {result['final_distance']:.2f}m | "
                  f"Reward: {result['total_reward']:.1f} | "
                  f"Steps: {result['episode_length']}")
    
    env.close()
    
    # Compute statistics
    successes = sum(1 for r in results if r['success'])
    success_rate = successes / n_episodes * 100
    
    min_distances = [r['min_distance'] for r in results]
    final_distances = [r['final_distance'] for r in results]
    rewards = [r['total_reward'] for r in results]
    lengths = [r['episode_length'] for r in results]
    
    stats = {
        'n_episodes': n_episodes,
        'successes': successes,
        'success_rate': success_rate,
        'min_distance_mean': np.mean(min_distances),
        'min_distance_std': np.std(min_distances),
        'final_distance_mean': np.mean(final_distances),
        'final_distance_std': np.std(final_distances),
        'reward_mean': np.mean(rewards),
        'reward_std': np.std(rewards),
        'episode_length_mean': np.mean(lengths),
        'results': results,
    }
    
    return stats


def print_summary(stats: Dict):
    """Print evaluation summary"""
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Episodes: {stats['n_episodes']}")
    print(f"  Successes: {stats['successes']}")
    print(f"  Success Rate: {stats['success_rate']:.1f}%")
    print("-" * 60)
    print(f"  Min Distance:   {stats['min_distance_mean']:.3f} ± {stats['min_distance_std']:.3f} m")
    print(f"  Final Distance: {stats['final_distance_mean']:.3f} ± {stats['final_distance_std']:.3f} m")
    print(f"  Episode Reward: {stats['reward_mean']:.1f} ± {stats['reward_std']:.1f}")
    print(f"  Episode Length: {stats['episode_length_mean']:.0f} steps")
    print("=" * 60)
    
    # Verdict
    if stats['success_rate'] >= 90:
        print("  ✓ EXCELLENT: Model reliably reaches target!")
    elif stats['success_rate'] >= 70:
        print("  ✓ GOOD: Model often reaches target")
    elif stats['success_rate'] >= 50:
        print("  ⚠ MODERATE: Model sometimes reaches target")
    else:
        print("  ✗ NEEDS IMPROVEMENT: Model rarely reaches target")
    print("=" * 60 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on target reaching task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path to saved model (.zip)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to environment config')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Number of evaluation episodes')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='Maximum steps per episode')
    parser.add_argument('--render', action='store_true',
                        help='Enable rendering')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-episode output')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)
    
    stats = evaluate_model(
        model_path=args.model,
        config_path=args.config,
        n_episodes=args.episodes,
        render=args.render,
        max_steps=args.max_steps,
        verbose=not args.quiet,
    )
    
    print_summary(stats)
    
    return 0 if stats['success_rate'] >= 50 else 1


if __name__ == "__main__":
    sys.exit(main())

