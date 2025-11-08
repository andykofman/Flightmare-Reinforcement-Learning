"""
Policy evaluation utilities for Flightmare

Provides functions to evaluate trained policies and collect statistics.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization


def evaluate_policy(
    model: BaseAlgorithm,
    env: gym.Env,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Tuple[float, float]:
    """
    Evaluate a trained policy.
    
    Args:
        model: The RL model to evaluate
        env: The environment
        n_eval_episodes: Number of episodes to evaluate
        deterministic: Use deterministic actions
        render: Render the environment
        return_episode_rewards: Return list of episode rewards
        warn: Output warnings
    
    Returns:
        mean_reward: Mean episode reward
        std_reward: Standard deviation of episode rewards
        (optionally) episode_rewards: List of episode rewards
        (optionally) episode_lengths: List of episode lengths
    """
    is_vec_env = isinstance(env, VecEnv)
    
    if is_vec_env:
        assert env.num_envs == 1, "You must pass only one environment for evaluation"
    
    episode_rewards = []
    episode_lengths = []
    
    for episode_idx in range(n_eval_episodes):
        # 1. Start episode
        obs, info = env.reset() #### Reset the environment, returns the initial observation and info
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        while not done:
            # 2. Agent takes action, based on observation
            action, _states = model.predict(obs, deterministic=deterministic) #### a = π(s) (same state → same action)  
            # 3. Environment updates, returns next observation, reward, terminated, truncated, and info
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            #4. Agent learns from (obs, action, reward, next_obs)
            done = terminated or truncated
            
            if render:
                env.render()
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if warn and episode_idx == 0:
            print(f"First episode reward: {episode_reward:.2f}, length: {episode_length}")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    
    if warn:
        print(f"\nEvaluation over {n_eval_episodes} episodes:")
        print(f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"  Mean length: {mean_length:.1f} +/- {std_length:.1f}")
    
    if return_episode_rewards:
        return mean_reward, std_reward, episode_rewards, episode_lengths
    
    return mean_reward, std_reward


def test_policy(
    model: BaseAlgorithm,
    env: gym.Env,
    n_episodes: int = 5,
    deterministic: bool = True,
    render: bool = True,
    save_video: bool = False,
    video_folder: str = "./videos",
) -> List[Dict[str, Any]]:
    """
    Test a trained policy and collect detailed statistics.
    
    Args:
        model: The RL model to test
        env: The environment
        n_episodes: Number of test episodes
        deterministic: Use deterministic actions
        render: Render the environment
        save_video: Save video recordings
        video_folder: Folder to save videos
    
    Returns:
        List of episode info dictionaries
    """
    if save_video:
        import os
        os.makedirs(video_folder, exist_ok=True)
        try:
            from gymnasium.wrappers import RecordVideo
            env = RecordVideo(
                env,
                video_folder=video_folder,
                episode_trigger=lambda x: True,  # Record all episodes
            )
        except ImportError:
            print("Warning: RecordVideo requires imageio-ffmpeg. Skipping video recording.")
            save_video = False
    
    episode_infos = []
    
    for episode_idx in range(n_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode_idx + 1}/{n_episodes}")
        print(f"{'='*60}")
        
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        actions_history = []
        rewards_history = []
        
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            actions_history.append(action.copy())
            rewards_history.append(reward)
            
            done = terminated or truncated
            
            if render:
                env.render()
        
        # Collect episode statistics
        episode_info = {
            'episode': episode_idx + 1,
            'reward': episode_reward,
            'length': episode_length,
            'mean_action': np.mean(actions_history, axis=0),
            'std_action': np.std(actions_history, axis=0),
            'mean_reward_per_step': episode_reward / episode_length,
            'terminated': terminated,
            'truncated': truncated,
        }
        
        episode_infos.append(episode_info)
        
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length}")
        print(f"  Mean reward per step: {episode_reward/episode_length:.3f}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Summary Statistics")
    print(f"{'='*60}")
    
    all_rewards = [ep['reward'] for ep in episode_infos]
    all_lengths = [ep['length'] for ep in episode_infos]
    
    print(f"  Mean reward: {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")
    print(f"  Mean length: {np.mean(all_lengths):.1f} +/- {np.std(all_lengths):.1f}")
    print(f"  Min reward: {np.min(all_rewards):.2f}")
    print(f"  Max reward: {np.max(all_rewards):.2f}")
    
    return episode_infos


def record_video(
    model: BaseAlgorithm,
    env: gym.Env,
    video_length: int = 1000,
    prefix: str = "agent",
    video_folder: str = "./videos",
):
    """
    Record a video of the agent.
    
    Args:
        model: Trained model
        env: Environment
        video_length: Length of video in steps
        prefix: Video file prefix
        video_folder: Folder to save video
    """
    import os
    from gymnasium.wrappers import RecordVideo
    
    os.makedirs(video_folder, exist_ok=True)
    
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=prefix,
        episode_trigger=lambda x: True,
    )
    
    obs, _ = env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    env.close()
    print(f"Video saved to {video_folder}")
