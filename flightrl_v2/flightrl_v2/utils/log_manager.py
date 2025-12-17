#!/usr/bin/env python3
"""
Unified Logging Manager for Flightmare RL Training

Provides centralized logging with:
- Timestamped run directories: logs/project/algorithm_timestamp/
- Run metadata tracking (hyperparameters, metrics, git info)
- Symbolic links for latest/best runs
- Alias resolution (@latest, @best, @baseline, etc.)
- Automatic best model tracking based on sucess rate


Usage: 
    # In training script: 
    from flightrl_v2.utils.log_manager import LogManager

    logger = LogManager(
        project="target_seeking",
        algorithm="SAC",
        config=args,
        tags=["velocity_control"]
    
    )

    # Use the run directory
    model = SAC(..., tensorboard_log=logger.run_dir)
    
    # Log metrics
    logger.log_metrics({"success_rate": 0.85, "avg_reward": 145.3})
    
    # Save model and finalize
    logger.save_model(model, "final")
    logger.finalize()  # Marks complete, updates 'best' if needed
    
    # In evaluation script:
    model_path = resolve_model_alias("@latest", project="target_seeking")
"""


import os 
import json
import subprocess
from datetime import  datetime
from pathlib import Path
from token import OP
from typing import Dict, Any, Optional, List




class LogManager:
    """
    Manages logging directories and metadata for training runs.




    Creates structure:
    logs/
        └── {project}/
            ├── {algorithm}_{timestamp}/
            │   ├── checkpoints/
            │   ├── tensorboard/
            │   ├── eval/
            │   ├── run_metadata.json
            │   └── training_config.txt
            ├── latest -> {algorithm}_{timestamp}/
            ├── best -> {algorithm}_{timestamp}/
            └── aliases.json
    """
    
    def __init__(
        self,
        project: str,
        algorithm: str,
        base_dir: str = "../../logs",
        config: Optional[List[Any]]=None,
        tags: Optional[List[Any]] = None,
        notes: Optional[str]      = None,
        run_name: Optional[str]   = None, # Optionl custom run name 
    ):
        """
        Initialize LogManager.
        
        Args:
            project: Project name (e.g., "target_seeking", "hovering")
            algorithm: Algorithm name (e.g., "SAC", "HER", "PPO")
            base_dir: Base logging directory (relative or absolute)
            config: Training configuration object (argparse.Namespace or dict)
            tags: List of tags for this run
            notes: Optional notes about this run
            run_name: Optional custom run name (overrides timestamp)
        """ 

        self.project = project
        self.algorithm = algorithm
        self.tags = tags or []
        self.notes = notes


        # convert base_dir to absolute path

        if not os.path.isabs(base_dir):
            # Get the dir of the script that's calling this
            caller_id = Path(os.getcwd()) 
            self.base_dir = (caller_id / base_dir).resolve()
        
        else: 
            self.base_dir  = Path(base_dir)

        # Create project directory

        self.project_dir = self.base_dir / project
        self.project_dir.mkdir(parents=True, exist_ok=True)


        # Create timestamped run directory

        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir_name = run_name if run_name else f"{algorithm.lower()}_{timestamp}" # if a custom run name is given, use it
        self.run_dir = self.project_dir / run_dir_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sub-directories

        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        
        self.tensorboard_dir = self.run_dir / "tensorboard"
        self.tensorboard_dir.mkdir(exist_ok=True)
        
        self.eval_dir = self.run_dir / "eval"
        self.eval_dir.mkdir(exist_ok=True)
        


        # Intialize metadata

        self.metadata = {
            "run_id": run_dir_name,
            "project": project,
            "algorithm": algorithm,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "duration_hours": None,
            "status": "running",
            "hyperparameters": {},
            "final_metrics": {},
            "git_commit": self._get_git_commit(),
            "git_branch": self._get_git_branch(),
            "tags": self.tags,
            "notes": self.notes,
        }

        # Store config if provided

        if config is not None:
            if hasattr(config, '__dict__'):
                self.metadata["hyperparameters"] = vars(config)
            elif isinstance(config, dict):
                self.metadata["hyperparameters"] = config

        
        # Save initial metadata

        self._save_metadata()
        
        # Update 'latest' symlink
        
        self._update_symlink("latest")
        
        # Save training config to text file

        if config is not None:
            self._save_config_txt(config)

        print(f"[INFO] Run directory: {self.run_dir}")
        print(f"[INFO] Symlink: {self.project_dir / 'latest'} -> {run_dir_name}")
    

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.base_dir
            )
            return result.stdout.strip()
        except:
            return None

    def _get_git_branch(self) -> Optional[str]:
        """Get current git branch"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=self.base_dir
            )
            return result.stdout.strip()
        except:
            return None

    def _save_metadata(self):
        """Save metadata to JSON file"""
        metadata_path = self.run_dir / "run_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _save_config_txt(self, config):
        """Save human-readable config to text file"""
        config_path = self.run_dir / "training_config.txt"
        with open(config_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TRAINING CONFIGURATION\n")
            f.write("="*70 + "\n")
            
            if hasattr(config, '__dict__'):
                config_dict = vars(config)
            elif isinstance(config, dict):
                config_dict = config
            else:
                config_dict = {}
            
            for key, value in sorted(config_dict.items()):
                f.write(f"{key}: {value}\n")
            
            f.write("="*70 + "\n")
            f.write(f"Timestamp: {self.metadata['start_time']}\n")
            if self.metadata['git_commit']:
                f.write(f"Git Commit: {self.metadata['git_commit']}\n")
            if self.metadata['git_branch']:
                f.write(f"Git Branch: {self.metadata['git_branch']}\n")
    
    def _update_symlink(self, link_name: str):
        """Create or update a symbolic link in project directory"""
        link_path = self.project_dir / link_name
        
        # Remove existing symlink if it exists
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        
        # Create new symlink (relative to project_dir)
        try:
            # Use relative path for portability
            link_path.symlink_to(self.run_dir.name, target_is_directory=True)
        except OSError:
            # Fallback: create a text file with the path if symlinks not supported
            with open(link_path.with_suffix('.txt'), 'w') as f:
                f.write(str(self.run_dir))

    def log_metrics(self, metrics: Dict[str, float]):
        """
        Log final metrics (success rate, reward, etc.)

        Args: 
            metrics: Dictionary of metric names and values

        """

        self.metadata["final_metrics"].update(metrics)
        self._save_metadata()

    def save_model(self, model, name: str = "final"):
        """
        Save model to checkpoints directory

        Args:
            model: model object with .save() method
            name: model name (without .zip extension)

        """
        model_path = self.checkpoints_dir / name
        model.save(str(model_path))
        print(f"[MODEL] Saved checkpoint: {model_path}.zip")


    def finalize(self, status: str = "completed"):

        """

        Marl run as complete and update metadata.

        Args:
            status: Final status ("completed", "interrupted", "failed")
        """

        start_time = datetime.fromisoformat(self.metadata["start_time"])
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 3600
        
        self.metadata["end_time"] = end_time.isoformat()
        self.metadata["duration_hours"] = round(duration, 2)
        self.metadata["status"] = status
        
        self._save_metadata()


    # Update 'best' symlink if this run has better metrics

        if status == "completed" and "success_rate" in self.metadata["final_metrics"]:
            self._maybe_update_best()

        print(f"[RUN] Finalized: {status} (duration: {duration:.2f}h)")
        

    def _maybe_update_best(self):
        """Update 'best' symlink if this run outperforms previous best"""
        current_success = self.metadata["final_metrics"].get("success_rate", 0)
        
        best_link = self.project_dir / "best"
        
        # Check if 'best' exists and read its metadata
        if best_link.exists() or best_link.is_symlink():
            try:
                best_run_dir = best_link.resolve()
                best_metadata_path = best_run_dir / "run_metadata.json"
                
                if best_metadata_path.exists():
                    with open(best_metadata_path, 'r') as f:
                        best_metadata = json.load(f)
                    
                    best_success = best_metadata.get("final_metrics", {}).get("success_rate", 0)
                    
                    if current_success > best_success:
                        print(f"[BEST] New best model ({current_success:.2%} > {best_success:.2%})")
                        self._update_symlink("best")
                    else:
                        print(f"[METRICS] Current: {current_success:.2%}, Best: {best_success:.2%}")
                else:
                    # No previous best metadata, make this the best
                    self._update_symlink("best")
            except Exception as e:
                print(f"[WARN] Could not compare with previous best: {e}")
                self._update_symlink("best")
        else:
            # No previous best, make this the best
            self._update_symlink("best")
            print(f"[BEST] First completed run marked as best")

    def update_alias(self, alias_name: str):
        """
        Add or update an alias for this run.
        
        Args:
            alias_name: Alias name (e.g., "baseline", "production", "v2")
        """
        aliases_file = self.project_dir / "aliases.json"
        
        # Load existing aliases
        if aliases_file.exists():
            with open(aliases_file, 'r') as f:
                aliases = json.load(f)
        else:
            aliases = {}
        
        # Update alias
        aliases[alias_name] = self.run_dir.name
        
        # Save aliases
        with open(aliases_file, 'w') as f:
            json.dump(aliases, f, indent=2)
        
        print(f"[ALIAS] '{alias_name}' -> {self.run_dir.name}")


def resolve_model_alias(
    model_path: str,
    project: Optional[str] = None,
    base_dir: str = "../../logs"

) -> str:

    """

    Resolve model path aliases like @latest, @best, @baseline.

    Args: 
        model_path: Model path or alias (e.g., "@latest", "@best", "path/to/model.zip")
        project: Project name (required if using alias)
        base_dir: Base logging directory

    Returns:
        Resolved absolute path to model

    Example:
        resolve_model_alias("@latest", project="target_seeking")
        # Returns: "/path/to/logs/target_seeking/latest/checkpoints/final_model.zip"
    """

    # If not an alias, return as-is

    if not model_path.startswith("@"):
        return model_path
    
    # Extract alias name
    alias = model_path[1:] # remove '@'

    if project is None:
        raise ValueError("Project name required when using model alias")

    # Convert base_dir to absolute path
    if not os.path.isabs(base_dir):
        caller_dir = Path(os.getcwd())
        base_dir = (caller_dir / base_dir).resolve()
    else:
        base_dir = Path(base_dir)
    
    project_dir = base_dir / project

        # Handle built-in aliases (latest, best)
    if alias in ["latest", "best"]:
        link_path = project_dir / alias
        
        # Check if symlink exists
        if link_path.exists() or link_path.is_symlink():
            run_dir = link_path.resolve()
        else:
            # Check for fallback text file
            txt_path = link_path.with_suffix('.txt')
            if txt_path.exists():
                with open(txt_path, 'r') as f:
                    run_dir = Path(f.read().strip())
            else:
                raise FileNotFoundError(f"Alias '{alias}' not found in project '{project}'")
    else:
        # Handle custom aliases from aliases.json
        aliases_file = project_dir / "aliases.json"
        
        if not aliases_file.exists():
            raise FileNotFoundError(f"No aliases.json found in project '{project}'")
        
        with open(aliases_file, 'r') as f:
            aliases = json.load(f)
        
        if alias not in aliases:
            raise KeyError(f"Alias '{alias}' not found. Available: {list(aliases.keys())}")
        
        run_dir = project_dir / aliases[alias]

      # Look for model in checkpoints directory
    checkpoints_dir = run_dir / "checkpoints"
    
    # Try common model names
    for model_name in ["final_model.zip", "best_model.zip", "final.zip", "best.zip"]:
        model_path = checkpoints_dir / model_name
        if model_path.exists():
            return str(model_path)
    
    # If no model found, return the checkpoints directory and let the user specify
    raise FileNotFoundError(
        f"No model found in {checkpoints_dir}. "
        f"Please specify full path: {model_path}/checkpoints/<model_name>.zip"
    )

def list_runs(project: str, base_dir: str = "../../logs", status: Optional[str] = None):
    """
    List all runs for a project.
    
    Args:
        project: Project name
        base_dir: Base logging directory
        status: Filter by status ("running", "completed", "failed", "interrupted")
    
    Returns:
        List of run metadata dictionaries
    """
    if not os.path.isabs(base_dir):
        caller_dir = Path(os.getcwd())
        base_dir = (caller_dir / base_dir).resolve()
    else:
        base_dir = Path(base_dir)
    
    project_dir = base_dir / project
    
    if not project_dir.exists():
        return []
    
    runs = []
    
    # Iterate through run directories
    for run_dir in project_dir.iterdir():
        if run_dir.is_dir() and not run_dir.is_symlink():
            metadata_path = run_dir / "run_metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Filter by status if specified
                if status is None or metadata.get("status") == status:
                    runs.append(metadata)
    
    # Sort by start time (newest first)
    runs.sort(key=lambda x: x.get("start_time", ""), reverse=True)
    
    return runs


def print_runs_table(runs: List[Dict[str, Any]]):
    """
    Print runs in a formatted table.
    
    Args:
        runs: List of run metadata dictionaries
    """
    if not runs:
        print("No runs found.")
        return
    
    print("\n" + "="*120)
    print(f"{'Run ID':<35} {'Algorithm':<8} {'Status':<12} {'Duration':<10} {'Success Rate':<12} {'Start Time':<20}")
    print("="*120)
    
    for run in runs:
        run_id = run.get("run_id", "N/A")
        algorithm = run.get("algorithm", "N/A")
        status = run.get("status", "N/A")
        duration = f"{run.get('duration_hours', 0):.2f}h" if run.get('duration_hours') else "N/A"
        success_rate = run.get("final_metrics", {}).get("success_rate", None)
        success_str = f"{success_rate:.2%}" if success_rate is not None else "N/A"
        start_time = run.get("start_time", "N/A")[:19] if run.get("start_time") else "N/A"
        
        print(f"{run_id:<35} {algorithm:<8} {status:<12} {duration:<10} {success_str:<12} {start_time:<20}")
    
    print("="*120 + "\n")
