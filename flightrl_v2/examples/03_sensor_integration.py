#!/usr/bin/env python3
"""
Example 03: Sensor Integration (LIDAR, Depth Camera, IMU)

PLACEHOLDER - This example will be completed when Phase 1 (Sensor Pipeline) is implemented.

This example will demonstrate:
- Configuring LIDAR sensors for obstacle detection
- Using depth camera observations for navigation
- Processing IMU data for state estimation
- Fusing multiple sensor modalities
- Sensor data preprocessing and normalization

Prerequisites:
- Phase 1: Sensor pipeline must be implemented
- C++ pybind wrapper extended to expose sensor data
- flightrl_v2.sensors module with functional LidarSensor, DepthCameraSensor

Current Status:
- BaseSensor is a stub (abstract base class)
- LidarSensor is a stub (raises NotImplementedError)
- DepthCameraSensor is a stub (raises NotImplementedError)
- IMUSensor is a stub (raises NotImplementedError)

Usage (when implemented):
    python 03_sensor_integration.py --sensor lidar --num-rays 16
    python 03_sensor_integration.py --sensor depth --width 64 --height 48
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def show_current_status():
    """Show what's currently available vs. what's coming."""
    print("\n" + "="*70)
    print("  Example 03: Sensor Integration")
    print("="*70)
    
    print("\n[STATUS] Current Status:")
    print("-"*70)
    
    # Check sensor imports
    try:
        from flightrl_v2.sensors import BaseSensor, SensorConfig
        print("  [OK] BaseSensor - Available (abstract base class)")
        print("  [OK] SensorConfig - Available (base configuration)")
    except ImportError as e:
        print(f"  [X] Base sensor: {e}")
    
    try:
        from flightrl_v2.sensors import LidarSensor, LidarConfig
        print("  [STUB] LidarSensor - Stub (raises NotImplementedError)")
        
        # Try to create and show what happens
        sensor = LidarSensor()
        print(f"      - Name: {sensor.name}")
        print(f"      - Shape: {sensor.observation_shape}")
    except ImportError as e:
        print(f"  [X] LidarSensor: {e}")
    except Exception as e:
        print(f"      - Error on creation: {e}")
    
    try:
        from flightrl_v2.sensors import DepthCameraSensor, DepthCameraConfig
        print("  [STUB] DepthCameraSensor - Stub (raises NotImplementedError)")
    except ImportError as e:
        print(f"  [X] DepthCameraSensor: {e}")
    
    try:
        from flightrl_v2.sensors import IMUSensor, IMUConfig
        print("  [STUB] IMUSensor - Stub (raises NotImplementedError)")
    except ImportError as e:
        print(f"  [X] IMUSensor: {e}")
    
    print("\n[INFO] What sensors are planned:")
    print("-"*70)
    print("""
  1. LidarSensor:
     - Configurable number of rays (default: 10)
     - Adjustable max range (default: 10m)
     - Horizontal and vertical FOV settings
     - Returns distance measurements as NDArray
     
  2. DepthCameraSensor:
     - Configurable resolution (default: 64x48)
     - Adjustable FOV (default: 90°)
     - Min/max depth clipping
     - Returns depth image as NDArray
     
  3. IMUSensor:
     - Accelerometer readings (3-axis)
     - Gyroscope readings (3-axis)
     - Configurable noise levels
     - Returns 6D measurement vector
""")
    
    print("\n[INFO] How sensors will integrate:")
    print("-"*70)
    print("""
  # Future usage (Phase 1):
  from flightrl_v2.sensors import LidarSensor, LidarConfig
  from flightrl_v2.envs import make_flight_env_for_sb3
  
  # Configure LIDAR
  lidar_config = LidarConfig(
      num_rays=16,
      max_range=10.0,
      fov_horizontal=360.0,
  )
  lidar = LidarSensor(lidar_config)
  
  # Create environment with sensor
  env = make_flight_env_for_sb3(
      sensors=[lidar],  # Sensor list
      ...
  )
  
  # Observation will include sensor data
  obs, info = env.reset()
  # obs now contains base state + lidar readings
""")
    
    print("\n[TODO] Required for Implementation:")
    print("-"*70)
    print("""
  1. Extend pybind_wrapper.cpp to expose:
     - LIDAR ray casting from Unity/C++ physics
     - Depth buffer rendering
     - IMU simulation with noise models
     
  2. Update FlightEnvVec to:
     - Accept sensor configurations
     - Include sensor data in observations
     - Handle sensor reset/update cycles
""")
    
    print("\n" + "="*70)
    print("  For a working example (without sensors), run:")
    print("    python 01_basic_training.py --timesteps 10000")
    print("="*70 + "\n")


def main():
    """Main entry point."""
    show_current_status()


if __name__ == "__main__":
    main()

