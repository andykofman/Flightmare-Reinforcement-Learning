#!/usr/bin/env python3
"""
Rollout Recorder for Flightmare Models

Records per-step state data from trained models running headlessly.
Outputs CSV files with drone pose, velocity, actions, and rewards.

Usage:
    from flightrl_v2.tools.rollout_recorder import record_rollouts
    
    episodes = record_rollouts(
        model_path="./logs/hovering/latest/checkpoints/final_model.zip",
        n_episodes=3,
        output_dir="./visualizations"
    )
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict

from stable_baselines3 import SAC


@dataclass
class EpisodeSummary:
    """Summary statistics for a single episode."""
    episode: int
    total_reward: float
    episode_length: int
    max_altitude_deviation: float
    max_horizontal_deviation: float
    terminal_reason: str
    csv_file: str
    
    # Additional metrics
    mean_reward_per_step: float = 0.0
    final_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    initial_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Target metrics
    target_position: Optional[Tuple[float, float, float]] = None
    final_distance_to_target: Optional[float] = None
    min_distance_to_target: Optional[float] = None


@dataclass
class RolloutConfig:
    """Configuration for rollout recording."""
    deterministic: bool = True
    max_steps: int = 300
    seed: Optional[int] = None
    dt: float = 0.02  # Simulation timestep
    
    # Observation indices (base quadrotor env)
    obs_pos_start: int = 0
    obs_pos_end: int = 3
    obs_euler_start: int = 3
    obs_euler_end: int = 6
    obs_vel_start: int = 6
    obs_vel_end: int = 9
    obs_omega_start: int = 9
    obs_omega_end: int = 12
    
    # Target-seeking specific (optional)
    obs_target_rel_start: int = 12
    obs_target_rel_end: int = 15
    obs_target_dist: int = 15
    obs_target_dir_start: int = 16
    obs_target_dir_end: int = 18
    
    # Default hover goal (from C++ quadrotor_env: goal_state_ << 0.0, 0.0, 5.0, ...)
    hover_goal: Tuple[float, float, float] = (0.0, 0.0, 5.0)


class RolloutRecorder:
    """
    Records rollouts from a trained model to CSV files.
    
    Captures per-step data including:
    - Drone position (x, y, z)
    - Orientation (Euler ZYX)
    - Linear and angular velocities
    - Actions taken
    - Rewards received
    - Termination flags
    """
    
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        config: Optional[RolloutConfig] = None,
        env_config_path: Optional[str] = None,
    ):
        """
        Initialize the rollout recorder.
        
        Args:
            model_path: Path to trained SB3 model (.zip)
            output_dir: Directory to save rollout CSVs
            config: Rollout configuration
            env_config_path: Path to environment YAML config
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.config = config or RolloutConfig()
        self.env_config_path = env_config_path
        
        # Will be set during recording
        self.model = None
        self.env = None
        self.episodes: List[EpisodeSummary] = []
        
        # Detect observation dimension to determine env type
        self._obs_dim = None
        self._is_target_seeking = False
    
    def _load_model(self):
        """Load the trained model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"Loading model: {self.model_path}")
        self.model = SAC.load(str(self.model_path))
        print("Model loaded successfully")
    
    def _create_env(self):
        """Create the headless environment."""
        print("Creating headless environment...")
        
        # Use the standard flightrl_v2 environment
        from flightrl_v2.envs.gymnasium_wrapper import make_flight_env_for_sb3
        
        self.env = make_flight_env_for_sb3(
            config_path=self.env_config_path,
            render=False,  # Always headless
            seed=self.config.seed,
            max_episode_steps=self.config.max_steps,
        )
        
        # Detect observation dimension
        self._obs_dim = self.env.observation_space.shape[0]
        self._is_target_seeking = self._obs_dim > 12
        
        print(f"Environment created (obs_dim={self._obs_dim}, target_seeking={self._is_target_seeking})")
    
    def _parse_observation(self, obs: np.ndarray) -> Dict[str, float]:
        """
        Parse observation vector into named fields.
        
        Args:
            obs: Observation array from environment
            
        Returns:
            Dictionary with named state values
        """
        cfg = self.config
        
        data = {
            # Position
            "pos_x": float(obs[cfg.obs_pos_start]),
            "pos_y": float(obs[cfg.obs_pos_start + 1]),
            "pos_z": float(obs[cfg.obs_pos_start + 2]),
            # Orientation (Euler ZYX)
            "euler_z": float(obs[cfg.obs_euler_start]),
            "euler_y": float(obs[cfg.obs_euler_start + 1]),
            "euler_x": float(obs[cfg.obs_euler_start + 2]),
            # Linear velocity
            "vel_x": float(obs[cfg.obs_vel_start]),
            "vel_y": float(obs[cfg.obs_vel_start + 1]),
            "vel_z": float(obs[cfg.obs_vel_start + 2]),
            # Angular velocity
            "omega_x": float(obs[cfg.obs_omega_start]),
            "omega_y": float(obs[cfg.obs_omega_start + 1]),
            "omega_z": float(obs[cfg.obs_omega_start + 2]),
        }
        
        # Add target-seeking specific fields if available
        if self._is_target_seeking and self._obs_dim >= 18:
            data.update({
                "target_rel_x": float(obs[cfg.obs_target_rel_start]),
                "target_rel_y": float(obs[cfg.obs_target_rel_start + 1]),
                "target_rel_z": float(obs[cfg.obs_target_rel_start + 2]),
                "target_dist": float(obs[cfg.obs_target_dist]),
                "target_dir_cos": float(obs[cfg.obs_target_dir_start]),
                "target_dir_sin": float(obs[cfg.obs_target_dir_start + 1]),
            })
        
        return data
    
    def _record_episode(self, episode_idx: int) -> Tuple[pd.DataFrame, EpisodeSummary]:
        """
        Record a single episode.
        
        Args:
            episode_idx: Episode index
            
        Returns:
            Tuple of (DataFrame with step data, EpisodeSummary)
        """
        obs, info = self.env.reset()
        
        steps = []
        cumulative_reward = 0.0
        step = 0
        
        # Track deviations from initial position
        initial_pos = np.array([obs[0], obs[1], obs[2]])
        max_altitude_dev = 0.0
        max_horizontal_dev = 0.0
        
        # Extract target position
        target_pos = None
        if 'target' in info:
            target_pos = np.array(info['target'])
        elif self._is_target_seeking:
            # Compute target from relative position in observation
            # target = current_pos + relative_pos
            target_pos = initial_pos + np.array([obs[12], obs[13], obs[14]])
        else:
            # Hovering task: use the fixed hover goal (0, 0, 5) from C++ environment
            target_pos = np.array(self.config.hover_goal)
        
        while step < self.config.max_steps:
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=self.config.deterministic)
            
            # Execute step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            cumulative_reward += reward
            
            # Parse observation
            state_data = self._parse_observation(obs)
            
            # Add target position to record (constant throughout episode)
            if target_pos is not None:
                state_data["target_x"] = float(target_pos[0])
                state_data["target_y"] = float(target_pos[1])
                state_data["target_z"] = float(target_pos[2])
                # Compute distance to target for this step
                current_pos = np.array([obs[0], obs[1], obs[2]])
                state_data["target_dist"] = float(np.linalg.norm(current_pos - target_pos))
            
            # Build step record
            step_record = {
                "step": step,
                "time": step * self.config.dt,
                **state_data,
                "action_0": float(action[0]),
                "action_1": float(action[1]),
                "action_2": float(action[2]),
                "action_3": float(action[3]),
                "reward": float(reward),
                "cumulative_reward": float(cumulative_reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
            
            steps.append(step_record)
            
            # Track deviations
            current_pos = np.array([obs[0], obs[1], obs[2]])
            altitude_dev = abs(current_pos[2] - initial_pos[2])
            horizontal_dev = np.sqrt((current_pos[0] - initial_pos[0])**2 + 
                                     (current_pos[1] - initial_pos[1])**2)
            max_altitude_dev = max(max_altitude_dev, altitude_dev)
            max_horizontal_dev = max(max_horizontal_dev, horizontal_dev)
            
            obs = next_obs
            step += 1
            
            if terminated or truncated:
                break
        
        # Create DataFrame
        df = pd.DataFrame(steps)
        
        # Determine terminal reason
        if terminated:
            terminal_reason = "terminated"
        elif truncated:
            terminal_reason = "truncated"
        else:
            terminal_reason = "max_steps"
        
        # Create summary
        final_pos = (float(obs[0]), float(obs[1]), float(obs[2]))
        
        # Calculate target-related metrics
        target_tuple = None
        final_dist = None
        min_dist = None
        if target_pos is not None:
            target_tuple = (float(target_pos[0]), float(target_pos[1]), float(target_pos[2]))
            final_dist = float(np.linalg.norm(np.array(final_pos) - target_pos))
            # Get min distance from dataframe if target_dist column exists
            if "target_dist" in df.columns:
                min_dist = float(df["target_dist"].min())
            else:
                min_dist = final_dist
        
        summary = EpisodeSummary(
            episode=episode_idx,
            total_reward=cumulative_reward,
            episode_length=step,
            max_altitude_deviation=max_altitude_dev,
            max_horizontal_deviation=max_horizontal_dev,
            terminal_reason=terminal_reason,
            csv_file=f"rollout_ep{episode_idx:02d}.csv",
            mean_reward_per_step=cumulative_reward / max(step, 1),
            final_position=final_pos,
            initial_position=(float(initial_pos[0]), float(initial_pos[1]), float(initial_pos[2])),
            target_position=target_tuple,
            final_distance_to_target=final_dist,
            min_distance_to_target=min_dist,
        )
        
        return df, summary
    
    def record(self, n_episodes: int = 3) -> List[EpisodeSummary]:
        """
        Record multiple episodes and save to disk.
        
        Args:
            n_episodes: Number of episodes to record
            
        Returns:
            List of episode summaries
        """
        # Setup
        self._load_model()
        self._create_env()
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        session_dir = self.output_dir / timestamp
        session_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nRecording {n_episodes} episodes to: {session_dir}")
        print("-" * 60)
        
        self.episodes = []
        
        for ep_idx in range(n_episodes):
            print(f"Recording episode {ep_idx + 1}/{n_episodes}...", end=" ")
            
            df, summary = self._record_episode(ep_idx)
            
            # Save CSV
            csv_path = session_dir / summary.csv_file
            df.to_csv(csv_path, index=False)
            
            self.episodes.append(summary)
            
            print(f"reward={summary.total_reward:.2f}, "
                  f"length={summary.episode_length}, "
                  f"dist={summary.final_distance_to_target:.2f}m, "
                  f"reason={summary.terminal_reason}")
        
        # Save summary JSON
        self._save_summary(session_dir, timestamp)
        
        # Cleanup
        self.env.close()
        
        print("-" * 60)
        print(f"Recorded {n_episodes} episodes to: {session_dir}")
        
        return self.episodes
    
    def _save_summary(self, session_dir: Path, timestamp: str):
        """Save session summary to JSON."""
        # Extract run_id from model path if possible
        run_id = "unknown"
        try:
            # Try to find run_metadata.json in parent directories
            for parent in self.model_path.parents:
                metadata_path = parent / "run_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                        run_id = metadata.get("run_id", "unknown")
                    break
                # Also check parent folder name as run_id
                if (parent / "checkpoints").exists():
                    run_id = parent.name
                    break
        except Exception:
            pass
        
        # Helper to convert numpy types to Python native types
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(v) for v in obj]
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        summary = {
            "model_path": str(self.model_path.absolute()),
            "run_id": run_id,
            "timestamp": timestamp,
            "episodes": [convert_to_native(asdict(ep)) for ep in self.episodes],
            "config": convert_to_native(asdict(self.config)),
            "is_target_seeking": self._is_target_seeking,
            "obs_dim": int(self._obs_dim) if self._obs_dim else None,
        }
        
        summary_path = session_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved: {summary_path}")


def _resolve_model_alias(model_arg: str, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve model path from alias (handles run names and symlinks).
    
    Supports formats:
        @hovering/sac_run_og
        @hovering/latest
        ./logs/hovering/sac_run_og/checkpoints/best_model.zip
    
    Args:
        model_arg: Model path or alias
        base_dir: Base directory for logs (default: flightrl_modern/logs)
        
    Returns:
        Resolved model path
    """
    if not model_arg.startswith("@"):
        return Path(model_arg)
    
    # Parse alias format: @project/run_name
    parts = model_arg[1:].split("/", 1)
    if len(parts) == 2:
        project, run_or_alias = parts
    else:
        project = "hovering"
        run_or_alias = parts[0]
    
    # Determine base logs directory
    if base_dir is None:
        # Try to find flightrl_v2/logs relative to this file
        script_dir = Path(__file__).parent
        base_dir = script_dir.parent.parent / "logs"
    
    project_dir = base_dir / project
    
    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")
    
    # Check if it's a direct run name (folder exists)
    run_dir = project_dir / run_or_alias
    if run_dir.is_dir():
        checkpoints_dir = run_dir / "checkpoints"
        # Also check for best/ subdirectory (HER models)
        best_dir = checkpoints_dir / "best"
        search_dirs = [best_dir, checkpoints_dir] if best_dir.exists() else [checkpoints_dir]
        
        for search_dir in search_dirs:
            for model_name in ["final_model.zip", "best_model.zip", "sac_final.zip", "final.zip", "best.zip"]:
                model_path = search_dir / model_name
                if model_path.exists():
                    return model_path
        raise FileNotFoundError(f"No model found in {checkpoints_dir}")
    
    # Check if it's a symlink (latest, best)
    if run_or_alias in ["latest", "best"]:
        link_path = project_dir / run_or_alias
        if link_path.exists() or link_path.is_symlink():
            resolved_run_dir = link_path.resolve()
            checkpoints_dir = resolved_run_dir / "checkpoints"
            best_dir = checkpoints_dir / "best"
            search_dirs = [best_dir, checkpoints_dir] if best_dir.exists() else [checkpoints_dir]
            
            for search_dir in search_dirs:
                for model_name in ["final_model.zip", "best_model.zip", "sac_final.zip", "final.zip", "best.zip"]:
                    model_path = search_dir / model_name
                    if model_path.exists():
                        return model_path
            raise FileNotFoundError(f"No model found in {checkpoints_dir}")
    
    raise FileNotFoundError(f"Could not resolve '{model_arg}'. Run directory not found at {run_dir}")


def record_rollouts(
    model_path: str,
    n_episodes: int = 3,
    output_dir: Optional[str] = None,
    deterministic: bool = True,
    max_steps: int = 300,
    seed: Optional[int] = None,
    env_config_path: Optional[str] = None,
) -> Tuple[Path, List[EpisodeSummary]]:
    """
    Convenience function to record rollouts from a trained model.
    
    Args:
        model_path: Path to trained model or alias (e.g., "@hovering/sac_run_og")
        n_episodes: Number of episodes to record
        output_dir: Output directory (default: same as model's run dir)
        deterministic: Use deterministic policy
        max_steps: Maximum steps per episode
        seed: Random seed
        env_config_path: Path to environment config
        
    Returns:
        Tuple of (output directory path, list of episode summaries)
    """
    # Resolve alias if needed
    if model_path.startswith("@"):
        model_path = _resolve_model_alias(model_path)
    
    model_path = Path(model_path)
    
    # Determine output directory
    if output_dir is None:
        # Default to visualizations/ under the run directory
        run_dir = model_path.parent.parent  # checkpoints/ -> run_dir
        output_dir = run_dir / "visualizations"
    else:
        output_dir = Path(output_dir)
    
    # Create config
    config = RolloutConfig(
        deterministic=deterministic,
        max_steps=max_steps,
        seed=seed,
    )
    
    # Record
    recorder = RolloutRecorder(
        model_path=str(model_path),
        output_dir=str(output_dir),
        config=config,
        env_config_path=env_config_path,
    )
    
    episodes = recorder.record(n_episodes=n_episodes)
    
    # Return the session directory (last created)
    session_dirs = sorted(output_dir.iterdir())
    session_dir = session_dirs[-1] if session_dirs else output_dir
    
    return session_dir, episodes


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Record rollouts from a trained Flightmare model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model or alias (e.g., @hovering/sac_run_og)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to record"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for rollout CSVs"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    session_dir, episodes = record_rollouts(
        model_path=args.model,
        n_episodes=args.episodes,
        output_dir=args.output,
        deterministic=args.deterministic,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    
    print(f"\nRollouts saved to: {session_dir}")

