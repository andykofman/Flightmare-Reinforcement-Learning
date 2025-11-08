#!/usr/bin/env python3
"""
Modern Drone Control Script for Flightmare

This is an updated version of the legacy run_drone_control.py that uses:
- PyTorch + Stable-Baselines3 instead of TensorFlow 1.x + stable-baselines v2
- SAC algorithm instead of PPO2
- Gymnasium instead of gym 0.11

The script maintains similar functionality but with modern libraries.

Usage:
    # Training
    python run_drone_control_modern.py --train 1 --seed 42
    
    # Evaluation
    python run_drone_control_modern.py --train 0 --weight ./saved/best_model.zip --render 1
"""

import os
import argparse
import numpy as np
from ruamel.yaml import YAML, dump, RoundTripDumper

# Modern imports
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure as configure_logger

from flightrl_modern.envs.gymnasium_wrapper import (
    make_flight_env_for_sb3,
    configure_random_seed,
)
from flightrl_modern.algorithms.evaluate import test_policy


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train',
        type=int,
        default=1,
        help="1 to train new model, 0 to test pre-trained model"
    )
    parser.add_argument(
        '--render',
        type=int,
        default=0,
        help="Enable Unity Render (1=yes, 0=no)"
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved'),
        help="Directory where to save the checkpoints and training metrics"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help="Random seed"
    )
    parser.add_argument(
        '-w',
        '--weight',
        type=str,
        default='./saved/best_model.zip',
        help='Trained weight path for testing'
    )
    parser.add_argument(
        '--timesteps',
        type=int,
        default=25000000,
        help='Total training timesteps'
    )
    return parser


def main():
    args = parser().parse_args()
    
    # Load environment configuration
    flightmare_path = os.environ.get("FLIGHTMARE_PATH")
    if flightmare_path is None:
        # Try to infer from script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        flightmare_path = os.path.abspath(os.path.join(script_dir, "..", ".."))
        os.environ["FLIGHTMARE_PATH"] = flightmare_path
        print(f"Setting FLIGHTMARE_PATH={flightmare_path}")
    
    config_path = os.path.join(flightmare_path, "flightlib/configs/vec_env.yaml")
    cfg = YAML().load(open(config_path, 'r'))
    
    # Adjust config based on train/test mode
    if not args.train:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1
    
    if args.render:
        cfg["env"]["render"] = "yes"
    else:
        cfg["env"]["render"] = "no"
    
    # Set random seed
    configure_random_seed(args.seed)
    
    # Create environment
    config_yaml_str = dump(cfg, Dumper=RoundTripDumper)
    
    # For training or testing
    if args.train:
        print("\n" + "="*60)
        print("Training Mode - SAC on Flightmare")
        print("="*60)
        print(f"Total timesteps: {args.timesteps:,}")
        print(f"Seed: {args.seed}")
        print(f"Save directory: {args.save_dir}")
        print("="*60 + "\n")
        
        # Create directories
        os.makedirs(args.save_dir, exist_ok=True)
        
        # Create environment
        env = make_flight_env_for_sb3(
            config_dict=cfg,
            render=bool(args.render),
            seed=args.seed,
        )
        env = Monitor(env, args.save_dir)
        
        # Create evaluation environment
        eval_env = make_flight_env_for_sb3(
            config_dict=cfg,
            render=False,
            seed=args.seed + 1000,
        )
        eval_env = Monitor(eval_env)
        
        # Configure logger
        configure_logger(args.save_dir)
        
        # Create SAC model
        # Using similar architecture to legacy PPO2: [128, 128] for both pi and vf
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=1000000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            policy_kwargs=dict(
                net_arch=[128, 128],  # Similar to legacy: [128, 128] for both
            ),
            tensorboard_log=args.save_dir,
            verbose=1,
            seed=args.seed,
        )
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=args.save_dir,
            name_prefix="sac_checkpoint",
            save_replay_buffer=True,
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=args.save_dir,
            log_path=args.save_dir,
            eval_freq=10000,
            n_eval_episodes=5,
            deterministic=True,
        )
        
        # Train
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, eval_callback],
            log_interval=10,
            progress_bar=True,
        )
        
        # Save final model
        final_path = os.path.join(args.save_dir, "sac_final")
        model.save(final_path)
        print(f"\nFinal model saved to: {final_path}")
        
        env.close()
        eval_env.close()
        
    else:
        # Testing mode
        print("\n" + "="*60)
        print("Testing Mode - Evaluating Trained Model")
        print("="*60)
        print(f"Model: {args.weight}")
        print(f"Render: {bool(args.render)}")
        print("="*60 + "\n")
        
        # Load trained model
        if not os.path.exists(args.weight):
            print(f"Error: Model file not found: {args.weight}")
            print("Please train a model first or specify correct --weight path")
            return
        
        model = SAC.load(args.weight)
        print(f"Model loaded from: {args.weight}\n")
        
        # Create test environment
        env = make_flight_env_for_sb3(
            config_dict=cfg,
            render=bool(args.render),
            seed=args.seed,
        )
        
        # Run evaluation
        test_policy(
            model=model,
            env=env,
            n_episodes=5,
            deterministic=True,
            render=bool(args.render),
        )
        
        env.close()


if __name__ == "__main__":
    main()
