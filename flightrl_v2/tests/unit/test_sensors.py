"""Unit tests for sensor module."""
import pytest


class TestBaseSensor:
    """Tests for BaseSensor class."""

    def test_sensor_import(self):
        """Test that sensors can be imported."""
        from flightrl_v2.sensors import BaseSensor, LidarSensor
        assert BaseSensor is not None
        assert LidarSensor is not None

    def test_lidar_not_implemented(self):
        """Test that LIDAR raises NotImplementedError."""
        from flightrl_v2.sensors import LidarSensor
        sensor = LidarSensor()
        with pytest.raises(NotImplementedError):
            sensor.get_observation()

    def test_depth_camera_import(self):
        """Test that depth camera can be imported."""
        from flightrl_v2.sensors import DepthCameraSensor
        assert DepthCameraSensor is not None

    def test_imu_import(self):
        """Test that IMU can be imported."""
        from flightrl_v2.sensors import IMUSensor
        assert IMUSensor is not None

