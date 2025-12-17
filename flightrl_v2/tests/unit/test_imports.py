"""
Test imports and basic module structure.

Verifies that all modules can be imported and basic dependencies are available.
"""

import pytest


def test_import_flightrl_v2():
    """Test that flightrl_v2 can be imported."""
    import flightrl_v2
    assert flightrl_v2.__version__ is not None


def test_import_envs():
    """Test that environment modules can be imported."""
    from flightrl_v2.envs import FlightEnvVec, make_flight_env


def test_import_algorithms():
    """Test that algorithm modules can be imported."""
    from flightrl_v2.algorithms import train_sac, evaluate_policy


def test_import_tasks():
    """Test that task modules can be imported."""
    from flightrl_v2.tasks import BaseTask, HoverTask


def test_import_configs():
    """Test that config modules can be imported."""
    from flightrl_v2.configs import load_config, save_config, validate_config


def test_import_tools():
    """Test that tools can be imported."""
    from flightrl_v2.tools import RolloutRecorder, visualize_model


def test_import_stable_baselines3():
    """Test that Stable-Baselines3 is available."""
    import stable_baselines3 as sb3
    assert sb3.__version__ is not None


def test_import_gymnasium():
    """Test that Gymnasium is available."""
    import gymnasium as gym
    assert gym.__version__ is not None


def test_import_torch():
    """Test that PyTorch is available."""
    import torch
    assert torch.__version__ is not None
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

