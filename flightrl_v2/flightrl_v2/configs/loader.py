"""
Configuration loader for flightrl_v2.

Provides functions for loading YAML configuration files with 
support for inheritance and overrides.  
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from ruamel.yaml import YAML

def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """ 
    Load a Yaml configuration file.

    Args: 
        config_path: Path to the Yaml configuration file.
        overrides: Optional dictionary of configuration overrides.  
    Returns:
        Loaded configuration as a dictionary.
    """

    yaml = YAML()
    yaml.preserve_quotes = True

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}") 
    
    with config_path.open('r') as f:
        config = yaml.load(f)

    if config is None:
        config = {}
    
    # Apply overrides if provided
    if overrides:
        _deep_update(config, overrides)
    return config

def save_config(
    config: Dict[str, Any],
    config_path: Union[str, Path]
) -> None:
    """ 
    Save a configuration dictionary to a Yaml file.

    Args: 
        config: Configuration dictionary to save.
        config_path: Path to save the Yaml configuration file.
    """

    yaml = YAML()
    yaml.default_flow_style = False

    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        yaml.dump(config, f)

def _deep_update(base: Dict, updates: Dict) -> Dict:
    """Recursively update a dictionary."""
    for key, value in updates.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def get_default_config_path(task_name: str) -> Path:
    """Get path to default config for a task."""
    configs_dir = Path(__file__).parent / "defaults"
    config_path = configs_dir / f"{task_name}.yaml"
    return config_path