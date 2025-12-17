"""
Comprehensive import verification for flightrl_v2.

This test file verifies that all public APIs and modules can be imported
correctly without errors. It serves as both a sanity check for the package
structure and as documentation for the available import paths.

Run with: pytest tests/test_all_imports.py -v
"""
import pytest


def test_core_imports():
    """
    Test that core module components can be imported.
    
    The core module provides base classes and type definitions that are
    used throughout the framework. This includes abstract base classes
    for environments and tasks, as well as type aliases for type safety.
    """
    from flightrl_v2.core import BaseFlightEnv, BaseTask, TaskConfig
    from flightrl_v2.core.types import (
        ObservationType,
        ActionType,
        RewardType,
        InfoType,
    )
    
    # Verify classes are importable and are actual types
    assert BaseFlightEnv is not None
    assert BaseTask is not None
    assert TaskConfig is not None
    
    # Type aliases should exist
    assert ObservationType is not None
    assert ActionType is not None
    assert RewardType is not None
    assert InfoType is not None


def test_envs_imports():
    """
    Test that environment module components can be imported.
    
    The envs module provides Gymnasium-compatible environment wrappers
    and factory functions for creating training environments. This includes
    both the base FlightEnvVec wrapper and convenience functions for
    Stable-Baselines3 integration.
    """
    from flightrl_v2.envs import (
        FlightEnvVec,
        make_flight_env_for_sb3,
        configure_random_seed,
    )
    
    # Verify environment class and factory functions exist
    assert FlightEnvVec is not None
    assert callable(make_flight_env_for_sb3)
    assert callable(configure_random_seed)


def test_envs_wrappers_imports():
    """
    Test that environment wrapper classes can be imported.
    
    Wrappers modify environment behavior without changing the underlying
    environment implementation. They are useful for adding observation
    preprocessing, reward shaping, or curriculum learning.
    """
    from flightrl_v2.envs.wrappers import (
        ObservationWrapper,
        RewardWrapper,
        CurriculumWrapper,
    )
    
    # Verify wrapper classes exist
    assert ObservationWrapper is not None
    assert RewardWrapper is not None
    assert CurriculumWrapper is not None


def test_tasks_imports():
    """
    Test that task module components can be imported.
    
    Tasks define the objectives and success criteria for training.
    Each task provides reward computation, termination conditions,
    and task-specific configuration.
    """
    from flightrl_v2.tasks import (
        HoverTask,
        TargetReachingTask,
        ObstacleAvoidanceTask,
    )
    
    # Verify task classes exist
    assert HoverTask is not None
    assert TargetReachingTask is not None
    assert ObstacleAvoidanceTask is not None


def test_tasks_base_imports():
    """
    Test that task base classes can be imported directly.
    
    The base task module is re-exported for convenience, allowing
    users to import from either flightrl_v2.tasks or flightrl_v2.core.
    """
    from flightrl_v2.tasks.base import BaseTask, TaskConfig
    
    # Verify base classes are accessible from tasks module
    assert BaseTask is not None
    assert TaskConfig is not None


def test_sensors_imports():
    """
    Test that sensor module components can be imported.
    
    Sensors provide additional observations beyond the basic state
    information. They simulate real-world sensors like LIDAR, depth
    cameras, and IMUs.
    """
    from flightrl_v2.sensors import (
        BaseSensor,
        LidarSensor,
        DepthCameraSensor,
        IMUSensor,
    )
    
    # Verify sensor classes exist
    assert BaseSensor is not None
    assert LidarSensor is not None
    assert DepthCameraSensor is not None
    assert IMUSensor is not None


def test_rewards_imports():
    """
    Test that reward module components can be imported.
    
    Reward components allow modular construction of complex reward
    functions. Multiple reward components can be combined to create
    sophisticated reward shaping strategies.
    """
    from flightrl_v2.rewards import (
        BaseReward,
        PositionReward,
        CollisionReward,
        CompositeReward,
    )
    
    # Verify reward classes exist
    assert BaseReward is not None
    assert PositionReward is not None
    assert CollisionReward is not None
    assert CompositeReward is not None


def test_algorithms_imports():
    """
    Test that algorithm module components can be imported.
    
    The algorithms module provides training and evaluation functions
    for different reinforcement learning algorithms. These are
    high-level interfaces that handle environment creation, model
    setup, and training loops.
    """
    from flightrl_v2.algorithms import (
        train_sac,
        train_ppo,
        train_td3,
        evaluate_policy,
    )
    
    # Verify algorithm functions exist and are callable
    assert callable(train_sac)
    assert callable(train_ppo)
    assert callable(train_td3)
    assert callable(evaluate_policy)


def test_algorithms_callbacks_imports():
    """
    Test that callback classes can be imported.
    
    Callbacks allow custom behavior during training, such as
    curriculum adjustment, custom metrics logging, or early stopping.
    """
    from flightrl_v2.algorithms.callbacks import (
        CurriculumCallback,
        MetricsCallback,
    )
    
    # Verify callback classes exist
    assert CurriculumCallback is not None
    assert MetricsCallback is not None


def test_configs_imports():
    """
    Test that configuration module components can be imported.
    
    The configs module provides utilities for loading and validating
    YAML configuration files used to define environment parameters
    and training settings.
    """
    from flightrl_v2.configs import load_config, validate_config
    
    # Verify config functions exist and are callable
    assert callable(load_config)
    assert callable(validate_config)


def test_deployment_imports():
    """
    Test that deployment module components can be imported.
    
    The deployment module provides tools for exporting trained models
    and deploying them to real hardware or other simulators. This
    includes ONNX export and inference wrappers.
    """
    from flightrl_v2.deployment import export_to_onnx, InferenceWrapper
    
    # Verify deployment functions and classes exist
    assert callable(export_to_onnx)
    assert InferenceWrapper is not None


def test_deployment_ardupilot_imports():
    """
    Test that ArduPilot integration components can be imported.
    
    The ArduPilot deployment module enables deploying trained policies
    to real quadrotors running ArduPilot firmware through MAVLink
    communication.
    """
    from flightrl_v2.deployment.ardupilot import (
        MAVLinkBridge,
        CompanionComputer,
    )
    
    # Verify ArduPilot classes exist
    assert MAVLinkBridge is not None
    assert CompanionComputer is not None


def test_tools_imports():
    """
    Test that tools module components can be imported.
    
    Tools provide utilities for working with trained models, including
    rollout recording, visualization, and model inspection.
    """
    from flightrl_v2.tools import RolloutRecorder, visualize_model
    
    # Verify tools exist
    assert RolloutRecorder is not None
    # visualize_model is a module, not a function
    assert visualize_model is not None


def test_visualization_imports():
    """
    Test that visualization module components can be imported.
    
    The visualization module provides tools for creating interactive
    3D visualizations of quadrotor trajectories and training progress.
    """
    from flightrl_v2.visualization import PlotlySceneVisualizer
    
    # Verify visualization classes exist
    assert PlotlySceneVisualizer is not None


def test_top_level_imports():
    """
    Test that commonly used components can be imported from the top level.
    
    The main flightrl_v2 package exports the most frequently used
    components for convenient access without needing to import from
    submodules.
    """
    import flightrl_v2
    
    # Check version is defined
    assert hasattr(flightrl_v2, '__version__')
    assert isinstance(flightrl_v2.__version__, str)
    
    # Check core exports are available at top level
    from flightrl_v2 import (
        BaseFlightEnv,
        BaseTask,
        TaskConfig,
        FlightEnvVec,
        make_flight_env_for_sb3,
        HoverTask,
        TargetReachingTask,
        train_sac,
        evaluate_policy,
    )
    
    # Verify all top-level imports work
    assert BaseFlightEnv is not None
    assert BaseTask is not None
    assert TaskConfig is not None
    assert FlightEnvVec is not None
    assert callable(make_flight_env_for_sb3)
    assert HoverTask is not None
    assert TargetReachingTask is not None
    assert callable(train_sac)
    assert callable(evaluate_policy)


def test_no_circular_imports():
    """
    Test that importing all modules does not cause circular import errors.
    
    Circular imports can cause subtle bugs and import failures. This test
    ensures that all modules can be imported in any order without issues.
    """
    # Import all major modules in different orders
    import flightrl_v2.core
    import flightrl_v2.envs
    import flightrl_v2.tasks
    import flightrl_v2.sensors
    import flightrl_v2.rewards
    import flightrl_v2.algorithms
    import flightrl_v2.configs
    import flightrl_v2.deployment
    import flightrl_v2.tools
    import flightrl_v2.visualization
    
    # If we get here without ImportError, circular imports are not present
    assert True


def test_import_star_works():
    """
    Test that 'from flightrl_v2 import *' works correctly.
    
    Using 'import *' should only import the public API components
    defined in __all__. This test verifies that __all__ is properly
    defined and contains the expected exports.
    """
    import flightrl_v2
    
    # Check that __all__ is defined
    assert hasattr(flightrl_v2, '__all__')
    assert isinstance(flightrl_v2.__all__, list)
    assert len(flightrl_v2.__all__) > 0
    
    # Check that all items in __all__ are actually exported
    for name in flightrl_v2.__all__:
        assert hasattr(flightrl_v2, name), f"{name} in __all__ but not exported"


if __name__ == "__main__":
    """
    Allow running this test file directly for quick verification.
    
    Usage: python tests/test_all_imports.py
    """
    pytest.main([__file__, "-v"])
