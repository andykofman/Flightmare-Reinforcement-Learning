"""Unit tests for rewards module."""
import pytest


class TestBaseReward:
    """Tests for BaseReward class."""

    def test_reward_import(self):
        """Test that rewards can be imported."""
        from flightrl_v2.rewards import BaseReward, PositionReward
        assert BaseReward is not None
        assert PositionReward is not None

    def test_position_reward_not_implemented(self):
        """Test that PositionReward raises NotImplementedError."""
        from flightrl_v2.rewards import PositionReward
        import numpy as np

        reward = PositionReward()
        obs = np.zeros(12, dtype=np.float32)
        action = np.zeros(4, dtype=np.float32)

        with pytest.raises(NotImplementedError):
            reward.compute(obs, action, obs, {})

    def test_collision_reward_import(self):
        """Test that collision reward can be imported."""
        from flightrl_v2.rewards import CollisionReward
        assert CollisionReward is not None

    def test_composite_reward_import(self):
        """Test that composite reward can be imported."""
        from flightrl_v2.rewards import CompositeReward
        assert CompositeReward is not None

