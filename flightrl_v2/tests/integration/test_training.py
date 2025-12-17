"""
Test training functionality.

These are integration tests that verify the training pipeline works.
They use very short training runs to keep test time reasonable.
"""

import os
import pytest
import numpy as np


# Check if flightgym is available
try:
    import flightgym  # type: ignore
    FLIGHTGYM_AVAILABLE = True
except ImportError:
    FLIGHTGYM_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not FLIGHTGYM_AVAILABLE,
    reason="flightgym not available (requires flightlib installation)"
)


@pytest.fixture
def flightmare_path():
    """Setup FLIGHTMARE_PATH environment variable."""
    if "FLIGHTMARE_PATH" not in os.environ:
        test_dir = os.path.dirname(os.path.abspath(__file__))
        flightmare_path = os.path.abspath(os.path.join(test_dir, "..", "..", "..", ".."))
        os.environ["FLIGHTMARE_PATH"] = flightmare_path
    return os.environ["FLIGHTMARE_PATH"]


def test_sac_quick_train(flightmare_path, tmp_path):
    """Test that SAC can train for a few steps without errors."""
    from stable_baselines3 import SAC
    from flightrl_v2.envs.gymnasium_wrapper import make_flight_env_for_sb3
    from stable_baselines3.common.monitor import Monitor
    
    # Create environment
    env = make_flight_env_for_sb3(render=False, seed=42)
    env = Monitor(env)
    
    # Create model
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=1000,
        learning_starts=10,
        batch_size=32,
        verbose=0,
        seed=42,
    )
    
    # Train for a very short time
    model.learn(total_timesteps=50, progress_bar=False)
    
    # Test that model can predict
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)
    
    assert action is not None
    assert action.shape == env.action_space.shape
    
    env.close()


def test_evaluate_policy(flightmare_path):
    """Test policy evaluation function."""
    from stable_baselines3 import SAC
    from flightrl_v2.envs.gymnasium_wrapper import make_flight_env_for_sb3
    from flightrl_v2.algorithms.evaluate import evaluate_policy
    from stable_baselines3.common.monitor import Monitor
    
    # Create environment
    env = make_flight_env_for_sb3(render=False, seed=42)
    env = Monitor(env)
    
    # Create untrained model
    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        seed=42,
    )
    
    # Evaluate (should work even with untrained model)
    mean_reward, std_reward = evaluate_policy(
        model=model,
        env=env,
        n_eval_episodes=2,
        deterministic=True,
        warn=False,
    )
    
    assert isinstance(mean_reward, (int, float))
    assert isinstance(std_reward, (int, float))
    assert std_reward >= 0.0
    
    env.close()


def test_train_and_save(flightmare_path, tmp_path):
    """Test training and saving model."""
    from stable_baselines3 import SAC
    from flightrl_v2.envs.gymnasium_wrapper import make_flight_env_for_sb3
    from stable_baselines3.common.monitor import Monitor
    
    # Create environment
    env = make_flight_env_for_sb3(render=False, seed=42)
    env = Monitor(env)
    
    # Create and train model
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=1000,
        learning_starts=10,
        batch_size=32,
        verbose=0,
        seed=42,
    )
    
    model.learn(total_timesteps=50, progress_bar=False)
    
    # Save model
    model_path = tmp_path / "test_model"
    model.save(model_path)
    
    # Load model
    loaded_model = SAC.load(model_path)
    
    # Test loaded model
    obs, _ = env.reset()
    action1, _ = model.predict(obs, deterministic=True)
    action2, _ = loaded_model.predict(obs, deterministic=True)
    
    # Actions should be the same
    np.testing.assert_array_almost_equal(action1, action2, decimal=5)
    
    env.close()


def test_quick_train_function(flightmare_path):
    """Test the quick_train_sac utility function."""
    from flightrl_v2.algorithms.train_sac import quick_train_sac
    
    model = quick_train_sac(timesteps=50, seed=42, verbose=0)
    
    assert model is not None
    # Model should be able to predict
    assert hasattr(model, 'predict')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

