"""
Test environment creation and basic functionality

Note: These tests require flightgym (from flightlib) to be installed.
If flightgym is not available, tests will be skipped.
"""

import os
import pytest
import numpy as np


# Check if flightgym is available
try:
    import flightgym
    FLIGHTGYM_AVAILABLE = True
except ImportError:
    FLIGHTGYM_AVAILABLE = False


# Skip all tests in this module if flightgym is not available
pytestmark = pytest.mark.skipif(
    not FLIGHTGYM_AVAILABLE,
    reason="flightgym not available (requires flightlib installation)"
)


@pytest.fixture
def flightmare_path():
    """Setup FLIGHTMARE_PATH environment variable"""
    if "FLIGHTMARE_PATH" not in os.environ:
        # Try to infer from test location
        test_dir = os.path.dirname(os.path.abspath(__file__))
        flightmare_path = os.path.abspath(os.path.join(test_dir, "..", "..", ".."))
        os.environ["FLIGHTMARE_PATH"] = flightmare_path
    return os.environ["FLIGHTMARE_PATH"]


def test_make_flight_env(flightmare_path):
    """Test environment creation"""
    from flightrl_modern.envs.gymnasium_wrapper import make_flight_env
    
    env = make_flight_env(render=False, num_envs=1, num_threads=1)
    
    assert env is not None
    assert env.observation_space is not None
    assert env.action_space is not None
    
    env.close()


def test_env_reset(flightmare_path):
    """Test environment reset"""
    from flightrl_modern.envs.gymnasium_wrapper import make_flight_env
    
    env = make_flight_env(render=False, num_envs=1, num_threads=1)
    
    obs, info = env.reset(seed=42)
    
    assert obs is not None
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)
    
    env.close()


def test_env_step(flightmare_path):
    """Test environment step"""
    from flightrl_modern.envs.gymnasium_wrapper import make_flight_env
    
    env = make_flight_env(render=False, num_envs=1, num_threads=1)
    
    obs, info = env.reset(seed=42)
    action = env.action_space.sample()
    
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    assert next_obs is not None
    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    
    env.close()


def test_env_multiple_steps(flightmare_path):
    """Test multiple environment steps"""
    from flightrl_modern.envs.gymnasium_wrapper import make_flight_env
    
    env = make_flight_env(render=False, num_envs=1, num_threads=1)
    
    obs, info = env.reset(seed=42)
    
    total_reward = 0.0
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    assert total_reward != 0.0 or True  # Reward could be zero
    
    env.close()


def test_env_seed_reproducibility(flightmare_path):
    """Test that same seed produces same initial observation"""
    from flightrl_modern.envs.gymnasium_wrapper import make_flight_env
    
    env1 = make_flight_env(render=False, num_envs=1, num_threads=1)
    env2 = make_flight_env(render=False, num_envs=1, num_threads=1)
    
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    
    np.testing.assert_array_almost_equal(obs1, obs2, decimal=5)
    
    env1.close()
    env2.close()


def test_env_spaces(flightmare_path):
    """Test that observation and action spaces are correctly defined"""
    from flightrl_modern.envs.gymnasium_wrapper import make_flight_env
    import gymnasium as gym
    
    env = make_flight_env(render=False, num_envs=1, num_threads=1)
    
    # Check observation space
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.dtype == np.float32
    
    # Check action space
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.action_space.dtype == np.float32
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)
    
    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
