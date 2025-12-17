#!/usr/bin/env python3
"""
List and Compare Training Runs

Utility script to view training runs, compare performance, and manage aliases.

Usage:
    # Lista all runs for a project
    python list_runs.py target_seeking

    # List only completed runs
    python list_runs.py target_seeking --status completed

    # List runs for all projects
    python list_runs.py --all

    # Show detailed info for a specific run
    python list_runs.py target_seeking --run sac_2025-11-24_143022

    # Create an alias for a run
    python list_runs.py target_seeking --alias baseline --run sac_2025-11-24_143022
    

"""


import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flightrl_v2.utils.log_manager import list_runs, print_runs_table
import json

def parse_args():
    parser = argparse.ArgumentParser(
        description="List and compare training runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("project", type=str, nargs='?', default=None,
                        help="Project name (target_seeking, hovering)")
    parser.add_argument("--all", action="store_true",
                        help="List runs for all projects")
    parser.add_argument("--status", type=str, default=None,
                        choices=["running", "completed", "failed", "interrupted"],
                        help="Filter by status")
    parser.add_argument("--run", type=str, default=None,
                        help="Show detailed info for specific run")
    parser.add_argument("--alias", type=str, default=None,
                        help="Create alias for specified run (requires --run)")
    parser.add_argument("--base-dir", type=str, default="logs",
                        help="Base directory for logs")
    
    return parser.parse_args()



def show_run_details(project: str, run_id: str, base_dir: str):
    """Show detailed information about a specific run"""
    base_path = Path(base_dir)
    run_dir = base_path / project / run_id
    metadata_path = run_dir / "run_metadata.json"
    
    if not metadata_path.exists():
        print(f"[ERROR] Run not found: {project}/{run_id}")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print("\n" + "="*70)
    print(f"RUN DETAILS: {run_id}")
    print("="*70)
    
    # Basic info
    print(f"\n[RUN] ID: {metadata.get('run_id', 'N/A')}")
    print(f"[RUN] Project: {metadata.get('project', 'N/A')}")
    print(f"[RUN] Algorithm: {metadata.get('algorithm', 'N/A')}")
    print(f"[RUN] Status: {metadata.get('status', 'N/A')}")
    
    # Timing
    print(f"\n[TIME] Start: {metadata.get('start_time', 'N/A')}")
    print(f"[TIME] End: {metadata.get('end_time', 'N/A')}")
    duration = metadata.get('duration_hours')
    if duration:
        print(f"[TIME] Duration: {duration:.2f} hours")
    
    # Git info
    if metadata.get('git_commit'):
        print(f"\n[GIT] Commit: {metadata.get('git_commit')}")
    if metadata.get('git_branch'):
        print(f"[GIT] Branch: {metadata.get('git_branch')}")
    
    # Tags and notes
    tags = metadata.get('tags', [])
    if tags:
        print(f"\n[TAGS] {', '.join(tags)}")
    notes = metadata.get('notes')
    if notes:
        print(f"[NOTES] {notes}")
    
    # Hyperparameters
    hyperparams = metadata.get('hyperparameters', {})
    if hyperparams:
        print("\n[CONFIG] Hyperparameters:")
        for key, value in sorted(hyperparams.items()):
            print(f"   {key}: {value}")
    
    # Final metrics
    final_metrics = metadata.get('final_metrics', {})
    if final_metrics:
        print("\n[METRICS] Final:")
        for key, value in sorted(final_metrics.items()):
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
    
    # Directory structure
    print("\n[PATHS] Directory Structure:")
    print(f"   Run Dir: {run_dir}")
    print(f"   Checkpoints: {run_dir / 'checkpoints'}")
    print(f"   TensorBoard: {run_dir / 'tensorboard'}")
    print(f"   Eval: {run_dir / 'eval'}")
    
    print("="*70 + "\n")


def create_alias(project: str, run_id: str, alias: str, base_dir: str):
    """Create an alias for a run"""
    base_path = Path(base_dir)
    project_dir = base_path / project
    aliases_file = project_dir / "aliases.json"
    
    # Load existing aliases
    if aliases_file.exists():
        with open(aliases_file, 'r') as f:
            aliases = json.load(f)
    else:
        aliases = {}
    
    # Update alias
    aliases[alias] = run_id
    
    # Save aliases
    with open(aliases_file, 'w') as f:
        json.dump(aliases, f, indent=2)
    
    print(f"[ALIAS] Created: @{alias} -> {run_id}")


def main():
    args = parse_args()
    
    # Handle alias creation
    if args.alias:
        if not args.run or not args.project:
            print("[ERROR] --alias requires both --run and project name")
            sys.exit(1)
        create_alias(args.project, args.run, args.alias, args.base_dir)
        return
    
    # Handle specific run details
    if args.run:
        if not args.project:
            print("[ERROR] --run requires project name")
            sys.exit(1)
        show_run_details(args.project, args.run, args.base_dir)
        return
    
    # List runs
    if args.all:
        # List all projects
        base_path = Path(args.base_dir)
        if not base_path.exists():
            print(f"[ERROR] Logs directory not found: {args.base_dir}")
            return
        
        projects = [d.name for d in base_path.iterdir() if d.is_dir()]
        
        for project in projects:
            print(f"\n{'='*70}")
            print(f"PROJECT: {project.upper()}")
            print(f"{'='*70}")
            runs = list_runs(project, base_dir=args.base_dir, status=args.status)
            print_runs_table(runs)
    else:
        if not args.project:
            print("[ERROR] Please specify a project name or use --all")
            sys.exit(1)
        
        print(f"\n{'='*70}")
        print(f"PROJECT: {args.project.upper()}")
        print(f"{'='*70}")
        runs = list_runs(args.project, base_dir=args.base_dir, status=args.status)
        print_runs_table(runs)


if __name__ == "__main__":
    main()
