"""  
Config validation utilities for flightrl_v2.

Provides schema validation for configurations files.

"""

from typing import Any, Dict, List, Optional

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def validate_config(
        config: Dict[str, Any],
        required_keys: Optional[List[str]] = None,
        schema: Optional[Dict[str, type]] = None
    ) -> bool:
    """
    Validate a configuration dictionary.

    Args:
        config: Configuration to validate
        required_keys: List of required top-level keys
        schema: Dict mapping keys to expected types

    Returns:
        True if valid

    Raises:
        ConfigValidationError: If validation fails
    """
    if required_keys:
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise ConfigValidationError(
                f"Missing required config keys: {missing}"
            )

    if schema:
        for key, expected_type in schema.items():
            if key in config and not isinstance(config[key], expected_type):
                raise ConfigValidationError(
                    f"Config key '{key}' should be {expected_type.__name__}, "
                    f"got {type(config[key]).__name__}"
                )

    return True

# Default schemas for common configurations
TRAINING_CONFIG_SCHEMA = {
    "total_timesteps": int,
    "n_envs": int,
    "learning_rate": float,
    "seed": int,
}

ENVIRONMENT_CONFIG_SCHEMA = {
    "max_episode_steps": int,
}