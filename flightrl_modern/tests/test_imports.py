"""
Test imports and basic module structure
"""

import pytest


def test_import_flightrl_modern():
    """Test that flightrl_modern can be imported"""
    import flightrl_modern
    assert flightrl_modern.__version__ is not None


def test_import_envs():
    """Test that environment modules can be imported"""
    from flightrl_modern.envs import FlightEnvVec, make_flight_env


def test_import_algorithms():
    """Test that algorithm modules can be imported"""
    from flightrl_modern.algorithms import train_sac, evaluate_policy


def test_import_stable_baselines3():
    """Test that Stable-Baselines3 is available"""
    import stable_baselines3 as sb3
    assert sb3.__version__ is not None


def test_import_gymnasium():
    """Test that Gymnasium is available"""
    import gymnasium as gym
    assert gym.__version__ is not None


def test_import_torch():
    """Test that PyTorch is available"""
    import torch
    assert torch.__version__ is not None
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
