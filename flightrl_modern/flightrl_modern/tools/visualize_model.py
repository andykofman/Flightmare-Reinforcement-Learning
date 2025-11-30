#!/usr/bin/env python3
"""
Visualize Model - Main CLI Entry Point

Records rollouts from a trained model and generates interactive 3D visualizations.
Combines rollout recording and Plotly visualization into a single workflow.

Usage:
    # Visualize sac_run_og model
    python -m flightrl_modern.tools.visualize_model --model @hovering/sac_run_og
    
    # Visualize with more episodes
    python -m flightrl_modern.tools.visualize_model --model @hovering/sac_run_og --episodes 5
    
    # Specify output directory
    python -m flightrl_modern.tools.visualize_model \\
        --model ./logs/hovering/sac_run_og/checkpoints/best_model.zip \\
        --out ./visualizations/test
    
    # Skip re-simulation if rollouts already exist
    python -m flightrl_modern.tools.visualize_model \\
        --model @hovering/sac_run_og \\
        --skip-record \\
        --session ./logs/hovering/sac_run_og/visualizations/2025-11-29_093510
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple


def resolve_model_path(model_arg: str, base_dir: Optional[Path] = None) -> Tuple[Path, str]:
    """
    Resolve model path from argument (handles aliases and run names).
    
    Args:
        model_arg: Model path or alias (e.g., @hovering/sac_run_og)
        base_dir: Base directory for logs (default: flightrl_modern/logs)
        
    Returns:
        Tuple of (resolved model path, run_id)
    """
    if model_arg.startswith("@"):
        # Parse alias format: @project/run_name or @project/alias
        parts = model_arg[1:].split("/", 1)
        if len(parts) == 2:
            project, run_or_alias = parts
        else:
            project = "hovering"
            run_or_alias = parts[0]
        
        # Determine base logs directory
        if base_dir is None:
            script_dir = Path(__file__).parent
            base_dir = script_dir.parent.parent / "logs"
        
        project_dir = base_dir / project
        
        # Check if it's a direct run name (folder exists)
        run_dir = project_dir / run_or_alias
        if run_dir.is_dir():
            # Direct run name - look for model in checkpoints
            checkpoints_dir = run_dir / "checkpoints"
            # Also check for best/ subdirectory (HER models)
            best_dir = checkpoints_dir / "best"
            search_dirs = [best_dir, checkpoints_dir] if best_dir.exists() else [checkpoints_dir]
            
            for search_dir in search_dirs:
                for model_name in ["final_model.zip", "best_model.zip", "sac_final.zip", "final.zip", "best.zip"]:
                    model_path = search_dir / model_name
                    if model_path.exists():
                        run_id = run_or_alias
                        return model_path, run_id
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
                            run_id = resolved_run_dir.name
                            return model_path, run_id
                raise FileNotFoundError(f"No model found in {checkpoints_dir}")
        
        # Try aliases.json as fallback
        aliases_file = project_dir / "aliases.json"
        if aliases_file.exists():
            with open(aliases_file) as f:
                aliases = json.load(f)
            if run_or_alias in aliases:
                run_name = aliases[run_or_alias]
                run_dir = project_dir / run_name
                checkpoints_dir = run_dir / "checkpoints"
                for model_name in ["final_model.zip", "best_model.zip", "sac_final.zip", "final.zip", "best.zip"]:
                    model_path = checkpoints_dir / model_name
                    if model_path.exists():
                        return model_path, run_name
        
        raise FileNotFoundError(
            f"Could not resolve '@{project}/{run_or_alias}'. "
            f"Run directory not found at {run_dir}"
        )
    else:
        model_path = Path(model_arg)
    
    # Try to extract run_id from path
    run_id = "unknown"
    try:
        for parent in model_path.parents:
            metadata_path = parent / "run_metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    run_id = metadata.get("run_id", "unknown")
                break
            # Use parent folder name as run_id if it contains checkpoints
            if (parent / "checkpoints").exists():
                run_id = parent.name
                break
    except Exception:
        pass
    
    return model_path, run_id


def find_latest_session(output_dir: Path) -> Optional[Path]:
    """Find the most recent session directory."""
    if not output_dir.exists():
        return None
    
    sessions = [d for d in output_dir.iterdir() if d.is_dir()]
    if not sessions:
        return None
    
    # Sort by name (timestamp format ensures chronological order)
    sessions.sort(key=lambda d: d.name, reverse=True)
    return sessions[0]


def visualize_model(
    model: str,
    episodes: int = 3,
    output_dir: Optional[str] = None,
    skip_record: bool = False,
    session: Optional[str] = None,
    deterministic: bool = True,
    max_steps: int = 300,
    seed: Optional[int] = None,
    show_animation: bool = True,
    open_browser: bool = False,
) -> Path:
    """
    Main visualization workflow.
    
    Args:
        model: Path to trained model or alias
        episodes: Number of episodes to record
        output_dir: Output directory for visualizations
        skip_record: Skip recording, use existing session
        session: Existing session directory to visualize
        deterministic: Use deterministic policy
        max_steps: Maximum steps per episode
        seed: Random seed
        show_animation: Include animation in visualization
        open_browser: Open visualization in browser when done
        
    Returns:
        Path to generated index.html
    """
    from flightrl_modern.tools.rollout_recorder import record_rollouts
    from flightrl_modern.visualization.plotly_scene import (
        create_multi_episode_visualization,
        create_trajectory_visualization,
    )
    
    print("\n" + "=" * 70)
    print("FLIGHTMARE MODEL VISUALIZER")
    print("=" * 70)
    
    # Resolve model path
    model_path, run_id = resolve_model_path(model)
    print(f"Model: {model_path}")
    print(f"Run ID: {run_id}")
    
    # Determine session directory
    if session:
        session_dir = Path(session)
        if not session_dir.exists():
            raise FileNotFoundError(f"Session directory not found: {session_dir}")
        print(f"Using existing session: {session_dir}")
    elif skip_record:
        # Find latest session in output_dir
        if output_dir:
            base_dir = Path(output_dir)
        else:
            run_dir = model_path.parent.parent
            base_dir = run_dir / "visualizations"
        
        session_dir = find_latest_session(base_dir)
        if session_dir is None:
            raise FileNotFoundError(f"No existing sessions found in {base_dir}")
        print(f"Using latest session: {session_dir}")
    else:
        # Record new rollouts
        print(f"\nRecording {episodes} episodes...")
        print("-" * 70)
        
        session_dir, episode_summaries = record_rollouts(
            model_path=str(model_path),
            n_episodes=episodes,
            output_dir=output_dir,
            deterministic=deterministic,
            max_steps=max_steps,
            seed=seed,
        )
        
        print("-" * 70)
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    
    # Load session summary
    summary_path = session_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            session_summary = json.load(f)
        episode_summaries = session_summary.get("episodes", [])
    else:
        episode_summaries = []
    
    # Create individual episode visualizations
    csv_files = sorted(session_dir.glob("rollout_ep*.csv"))
    
    for csv_file in csv_files:
        # Find matching summary
        matching = [s for s in episode_summaries if s.get("csv_file") == csv_file.name]
        summary = matching[0] if matching else None
        
        create_trajectory_visualization(
            csv_path=str(csv_file),
            episode_summary=summary,
            show_animation=show_animation,
        )
    
    # Create multi-episode overview
    index_path = create_multi_episode_visualization(
        session_dir=str(session_dir),
        title=f"Model: {run_id}",
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {session_dir}")
    print(f"Main visualization: {index_path}")
    print(f"\nIndividual episodes:")
    for csv_file in csv_files:
        html_file = csv_file.with_suffix(".html")
        if html_file.exists():
            print(f"  - {html_file.name}")
    
    # Print episode summary
    if episode_summaries:
        print(f"\nEpisode Summary:")
        print("-" * 50)
        total_reward = 0
        for ep in episode_summaries:
            reward = ep.get("total_reward", 0)
            length = ep.get("episode_length", 0)
            reason = ep.get("terminal_reason", "?")
            final_dist = ep.get("final_distance_to_target")
            dist_str = f", dist={final_dist:.2f}m" if final_dist is not None else ""
            print(f"  Episode {ep.get('episode', '?')}: "
                  f"reward={reward:.2f}, length={length}{dist_str}, {reason}")
            total_reward += reward
        
        avg_reward = total_reward / len(episode_summaries)
        print("-" * 50)
        print(f"  Average reward: {avg_reward:.2f}")
    
    print("\n" + "=" * 70)
    
    # Open in browser if requested
    if open_browser:
        import webbrowser
        webbrowser.open(f"file://{index_path.absolute()}")
        print("Opened visualization in browser.")
    
    return index_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trained Flightmare models with 3D trajectory plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Visualize sac_run_og model
    python -m flightrl_modern.tools.visualize_model --model @hovering/sac_run_og
    
    # Visualize with more episodes
    python -m flightrl_modern.tools.visualize_model \\
        --model @hovering/sac_run_og --episodes 5
    
    # Use existing session (skip re-recording)
    python -m flightrl_modern.tools.visualize_model \\
        --model @hovering/sac_run_og --skip-record
    
    # Specify output directory
    python -m flightrl_modern.tools.visualize_model \\
        --model ./checkpoints/best_model.zip --out ./my_visualizations
        """,
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
        help="Number of episodes to record (default: 3)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--skip-record",
        action="store_true",
        help="Skip recording, use most recent existing session"
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Path to existing session directory to visualize"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy (default: True)"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy (with exploration noise)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Maximum steps per episode (default: 300)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Disable animation controls in visualization"
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open visualization in browser when complete"
    )
    
    args = parser.parse_args()
    
    # Handle deterministic/stochastic flags
    deterministic = not args.stochastic
    
    try:
        visualize_model(
            model=args.model,
            episodes=args.episodes,
            output_dir=args.out,
            skip_record=args.skip_record,
            session=args.session,
            deterministic=deterministic,
            max_steps=args.max_steps,
            seed=args.seed,
            show_animation=not args.no_animation,
            open_browser=args.open,
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

