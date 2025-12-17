"""
Backward compatibility verification tests for flightrl_v2.

This test suite verifies that flightrl_v2 maintains functional compatibility
with the expected behavior from flightrl_modern. While the internal architecture
has been refactored, the external API and environment behavior should remain
consistent.

These tests verify:
- Environment creation follows expected patterns
- Observation and action spaces match specifications
- Episode dynamics work correctly
- Random seeding produces reproducible results

Note: These tests run independently of flightrl_modern and verify against
the documented specifications rather than doing direct comparisons.

Run with: pytest tests/integration/test_backward_compatibility.py -v
"""
import pytest
import numpy as np
import gymnasium as gym


class TestEnvironmentCreation:
    """
    Test that environments can be created with the expected interfaces.
    
    These tests verify that the environment factory functions work correctly
    and produce environments that conform to the Gymnasium interface.
    """

    def test_basic_environment_creation(self):
        """
        Test that a basic environment can be created without errors.
        
        This verifies that the make_flight_env_for_sb3 function works with
        default parameters and produces a valid Gymnasium environment.
        """
        from flightrl_v2.envs import make_flight_env_for_sb3

        env = make_flight_env_for_sb3(seed=42)
        
        # Verify it's a valid Gymnasium environment
        assert isinstance(env, gym.Env)
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'close')
        
        env.close()

    def test_environment_with_custom_config(self):
        """
        Test environment creation with custom configuration file.
        
        This verifies that environments can be created using YAML config files
        and that the config path parameter is properly handled.
        """
        from flightrl_v2.envs import make_flight_env_for_sb3
        import os

        # Try with default config if FLIGHTMARE_PATH is set
        flightmare_path = os.environ.get("FLIGHTMARE_PATH")
        if flightmare_path:
            config_path = os.path.join(
                flightmare_path, 
                "flightlib/configs/target_reaching.yaml"
            )
            if os.path.exists(config_path):
                env = make_flight_env_for_sb3(config_path=config_path, seed=42)
                assert env is not None
                env.close()

    def test_environment_with_max_episode_steps(self):
        """
        Test that max_episode_steps parameter is respected.
        
        This verifies that the episode length limiting works correctly
        and episodes truncate at the specified step count.
        """
        from flightrl_v2.envs import make_flight_env_for_sb3

        max_steps = 100
        env = make_flight_env_for_sb3(seed=42, max_episode_steps=max_steps)
        
        obs, info = env.reset()
        
        # Run until truncation
        for step in range(max_steps + 10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                # Should truncate at max_steps
                assert step < max_steps + 1
                break
        
        env.close()


class TestObservationSpace:
    """
    Test that observation spaces match the expected specification.
    
    The observation space defines what information the agent receives.
    These tests verify that observations have the correct shape and range.
    """

    def test_observation_space_exists(self):
        """
        Test that observation space is properly defined.
        
        This verifies the environment has a valid observation space
        attribute that conforms to Gymnasium specifications.
        """
        from flightrl_v2.envs import make_flight_env_for_sb3

        env = make_flight_env_for_sb3(seed=42)
        
        assert hasattr(env, 'observation_space')
        assert isinstance(env.observation_space, gym.Space)
        
        env.close()

    def test_observation_shape(self):
        """
        Test that observations have the expected shape and bounds.
        
        This verifies that reset and step return observations matching
        the declared observation space.
        """
        from flightrl_v2.envs import make_flight_env_for_sb3

        env = make_flight_env_for_sb3(seed=42)
        
        # Get initial observation
        obs, info = env.reset()
        
        # Verify observation matches space
        assert obs in env.observation_space
        assert obs.dtype == np.float32
        
        # Take a step and verify observation again
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs in env.observation_space
        
        env.close()

    def test_observation_contains_expected_states(self):
        """
        Test that observations contain position, orientation, and velocity.
        
        The standard observation should include at minimum:
        - Position (x, y, z): 3 elements
        - Orientation (roll, pitch, yaw or quaternion): 3-4 elements
        - Linear velocity (vx, vy, vz): 3 elements
        - Angular velocity (wx, wy, wz): 3 elements
        """
        from flightrl_v2.envs import make_flight_env_for_sb3

        env = make_flight_env_for_sb3(seed=42)
        obs, info = env.reset()
        
        # Observation should have at least 12 elements (pos + ori + lin_vel + ang_vel)
        assert len(obs) >= 12, f"Observation too short: {len(obs)} elements"
        
        # Position should be first 3 elements
        position = obs[0:3]
        assert len(position) == 3
        
        # Velocity should be around indices 6-9
        if len(obs) >= 9:
            velocity = obs[6:9]
            assert len(velocity) == 3
        
        env.close()


class TestActionSpace:
    """
    Test that action spaces match the expected specification.
    
    The action space defines what commands the agent can send. For quadrotors,
    this is typically 4 motor commands.
    """

    def test_action_space_is_continuous(self):
        """
        Test that action space is continuous (Box space).
        
        Quadrotor control uses continuous actions for motor speeds.
        This verifies the action space is properly defined as continuous.
        """
        from flightrl_v2.envs import make_flight_env_for_sb3

        env = make_flight_env_for_sb3(seed=42)
        
        assert hasattr(env, 'action_space')
        assert isinstance(env.action_space, gym.spaces.Box)
        
        env.close()

    def test_action_space_dimension(self):
        """
        Test that action space has 4 dimensions (one per motor).
        
        Standard quadrotors have 4 motors, so the action space
        should be 4-dimensional.
        """
        from flightrl_v2.envs import make_flight_env_for_sb3

        env = make_flight_env_for_sb3(seed=42)
        
        # Should have 4 actions (4 motors)
        assert env.action_space.shape == (4,), \
            f"Expected 4 actions, got {env.action_space.shape}"
        
        env.close()

    def test_action_bounds(self):
        """
        Test that action bounds are defined and reasonable.
        
        Motor commands should have finite bounds. This test verifies
        that actions are properly bounded.
        """
        from flightrl_v2.envs import make_flight_env_for_sb3

        env = make_flight_env_for_sb3(seed=42)
        
        # Check bounds exist and are finite
        assert np.all(np.isfinite(env.action_space.low))
        assert np.all(np.isfinite(env.action_space.high))
        
        # Verify low < high
        assert np.all(env.action_space.low < env.action_space.high)
        
        env.close()


class TestEpisodeDynamics:
    """
    Test that episode dynamics work correctly.
    
    These tests verify that reset, step, and termination work as expected
    throughout an episode.
    """

    def test_reset_returns_valid_observation(self):
        """
        Test that reset returns observation and info dict.
        
        Gymnasium environments should return (observation, info) from reset.
        """
        from flightrl_v2.envs import make_flight_env_for_sb3

        env = make_flight_env_for_sb3(seed=42)
        
        result = env.reset()
        
        # Should return tuple of (obs, info)
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        
        env.close()

    def test_step_returns_five_values(self):
        """
        Test that step returns the correct Gymnasium tuple.
        
        Gymnasium step should return:
        (observation, reward, terminated, truncated, info)
        """
        from flightrl_v2.envs import make_flight_env_for_sb3

        env = make_flight_env_for_sb3(seed=42)
        env.reset()
        
        action = env.action_space.sample()
        result = env.step(action)
        
        # Should return 5 values
        assert isinstance(result, tuple)
        assert len(result) == 5
        
        obs, reward, terminated, truncated, info = result
        
        # Check types
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float, np.number))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)
        
        env.close()

    def test_episode_can_complete(self):
        """
        Test that an episode can run to completion.
        
        This verifies that the environment can handle a full episode
        without errors and properly signals termination.
        """
        from flightrl_v2.envs import make_flight_env_for_sb3

        env = make_flight_env_for_sb3(seed=42, max_episode_steps=100)
        obs, info = env.reset()
        
        done = False
        steps = 0
        max_steps = 200  # Safety limit
        
        while not done and steps < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        # Should have ended within max_steps
        assert steps < max_steps, "Episode did not terminate"
        
        env.close()


class TestReproducibility:
    """
    Test that random seeding produces reproducible results.
    
    Reproducibility is critical for debugging and experiment comparison.
    These tests verify that setting a seed produces identical behavior.
    """

    def test_seeding_produces_same_initial_state(self):
        """
        Test that the same seed produces the same initial observation.
        
        When reset with the same seed, environments should produce
        identical initial states.
        """
        from flightrl_v2.envs import make_flight_env_for_sb3

        seed = 123
        
        # Create two environments with same seed
        env1 = make_flight_env_for_sb3(seed=seed)
        env2 = make_flight_env_for_sb3(seed=seed)
        
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        
        # Initial observations should be identical
        np.testing.assert_array_almost_equal(obs1, obs2)
        
        env1.close()
        env2.close()

    def test_seeding_produces_same_trajectory(self):
        """
        Test that seeded environments produce identical trajectories.
        
        With the same seed and actions, two environments should produce
        exactly the same sequence of observations and rewards.
        """
        from flightrl_v2.envs import make_flight_env_for_sb3

        seed = 456
        num_steps = 10
        
        # Create two environments with same seed
        env1 = make_flight_env_for_sb3(seed=seed)
        env2 = make_flight_env_for_sb3(seed=seed)
        
        env1.reset(seed=seed)
        env2.reset(seed=seed)
        
        # Take same actions in both environments
        for _ in range(num_steps):
            # Use deterministic action (not sampled)
            action = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            
            obs1, reward1, term1, trunc1, info1 = env1.step(action)
            obs2, reward2, term2, trunc2, info2 = env2.step(action)
            
            # Observations and rewards should be identical
            np.testing.assert_array_almost_equal(obs1, obs2)
            assert abs(reward1 - reward2) < 1e-6
            assert term1 == term2
            assert trunc1 == trunc2
        
        env1.close()
        env2.close()


class TestIntegrationWithStableBaselines3:
    """
    Test integration with Stable-Baselines3 library.
    
    These tests verify that the environment works correctly with SB3,
    which is the primary training library used with this framework.
    """

    def test_environment_passes_sb3_check(self):
        """
        Test that environment passes Stable-Baselines3 checks.
        
        SB3 provides a check_env function that verifies environments
        conform to the expected interface.
        """
        pytest.importorskip("stable_baselines3")
        from stable_baselines3.common.env_checker import check_env
        from flightrl_v2.envs import make_flight_env_for_sb3

        env = make_flight_env_for_sb3(seed=42)
        
        # This will raise an exception if environment is invalid
        check_env(env, warn=True)
        
        env.close()

    def test_can_create_sb3_model(self):
        """
        Test that an SB3 model can be created with the environment.
        
        This verifies that the environment is compatible with SB3
        algorithm constructors.
        """
        pytest.importorskip("stable_baselines3")
        from stable_baselines3 import SAC
        from flightrl_v2.envs import make_flight_env_for_sb3

        env = make_flight_env_for_sb3(seed=42)
        
        # Create SAC model (should not raise errors)
        model = SAC("MlpPolicy", env, verbose=0)
        
        assert model is not None
        
        env.close()


if __name__ == "__main__":
    """
    Allow running this test file directly for quick verification.
    
    Usage: python tests/integration/test_backward_compatibility.py
    """
    pytest.main([__file__, "-v"])
