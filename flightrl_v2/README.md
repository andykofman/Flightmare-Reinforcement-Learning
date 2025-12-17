# flightrl_v2

**Modern Reinforcement Learning Framework for Flightmare Quadrotor Simulator**

A PyTorch + Stable-Baselines3 + Gymnasium based RL framework for training intelligent quadrotor control policies in the Flightmare simulator.

## Overview

flightrl_v2 is a comprehensive refactoring of the Flightmare reinforcement learning stack, designed to provide researchers and developers with a clean, extensible, and production-ready framework for training autonomous quadrotor agents.

### Key Design Principles

- **Modularity**: Clear separation between environments, tasks, sensors, and algorithms allows independent development and testing of components
- **Modern Standards**: Built on Gymnasium (not legacy Gym), Stable-Baselines3 2.0+, and PyTorch 2.0+ for compatibility with current RL research
- **Type Safety**: Comprehensive type hints and runtime validation reduce bugs and improve code clarity
- **Extensibility**: Adding new tasks, sensors, or reward functions requires minimal code changes
- **Maintainability**: Well-documented codebase with consistent patterns and comprehensive test coverage

### What This Framework Provides

flightrl_v2 bridges the gap between simulation and real-world deployment by providing:

1. **Environment Interface**: Gymnasium-compatible wrappers around the Flightmare C++ simulator, handling observation processing, action mapping, and episode management
2. **Task Definitions**: Modular task specifications that define objectives, success criteria, and reward structures independently from the environment
3. **Training Infrastructure**: Pre-configured training pipelines for SAC, PPO, and TD3 with sensible defaults and extensive customization options
4. **Analysis Tools**: Rollout recording, trajectory visualization, and performance metrics for understanding agent behavior
5. **Deployment Path**: Model export and hardware integration tools for transitioning from simulation to physical quadrotors

## Features

### Currently Available (Phase 0 - Complete)

**Core Framework:**
- Gymnasium-compatible environment wrappers with proper observation/action space handling
- Soft Actor-Critic (SAC) training with optimized hyperparameters for quadrotor control
- Vectorized environment support for parallel data collection across multiple simulation instances
- Comprehensive TensorBoard logging for training metrics, episode statistics, and performance tracking
- Automatic model checkpointing and best model selection based on evaluation performance

**Tasks:**
- **Hover Task**: Train quadrotor to maintain stable position at origin with minimal drift
- **Target Reaching**: Navigate to specified 3D coordinates and stabilize within tolerance bounds

**Environment Features:**
- Configurable episode lengths, observation spaces, and termination conditions
- Random seed support for reproducible experiments and controlled training conditions
- YAML-based configuration system for easy parameter adjustment without code changes

**Tools:**
- Rollout recorder for capturing agent trajectories with per-step state information
- Interactive 3D visualization using Plotly for trajectory analysis and debugging
- Model evaluation utilities with customizable success metrics

### Planned Development

**Phase 1 - Sensor Integration:**
- LIDAR sensor simulation for 360-degree obstacle detection with configurable resolution
- Depth camera interface for vision-based navigation and obstacle avoidance
- IMU sensor simulation with realistic noise models for state estimation
- Observation wrappers for processing and normalizing sensor data

**Phase 2 - Advanced Training:**
- Curriculum learning framework with automatic difficulty progression
- Multi-task learning support for training agents on multiple objectives simultaneously
- Reward shaping utilities for composing complex reward functions from simple components
- Custom callback system for task-specific metrics and training adjustments

**Phase 3 - Complex Tasks:**
- Obstacle avoidance in cluttered environments with dynamic obstacle placement
- Racing through gate sequences with time optimization
- Formation flight for multi-agent coordination
- Trajectory following with position and velocity constraints

**Phase 4 - Deployment:**
- ONNX model export for hardware-accelerated inference on edge devices
- ArduPilot integration via MAVLink for real quadrotor control
- Safety monitoring and intervention system for hardware testing
- Sim-to-real transfer analysis and domain adaptation tools

## Installation

### Prerequisites

1. **Flightmare Simulator**: The C++ physics engine (`flightgym`) must be built and installed
   ```bash
   # Build flightgym (from Flightmare root)
   cd flightlib
   mkdir build && cd build
   cmake ..
   make -j$(nproc)
   ```

2. **Python 3.8+**: Tested on Python 3.8, 3.9, 3.10

### Install flightrl_v2

```bash
# Navigate to flightrl_v2 directory
cd flightmare/flightrl_v2

# Install in development mode
pip install -e .

# Or install with development tools
pip install -e ".[dev]"
```

## Quick Start

### Your First Training Run

The fastest way to get started is using the comprehensive training example:

```bash
# Navigate to examples directory
cd flightmare/flightrl_v2/examples

# Quick test run (10,000 steps for verification)
python 01_basic_training.py --timesteps 10000

# Full training run (convergence typically around 1 million steps)
python 01_basic_training.py --timesteps 1000000 --n_envs 16 --max_episode_steps 600
```

This trains a quadrotor to reach target position [0, 0, 5] and stabilize. The training script includes:
- Custom success rate tracking callback
- Automatic model checkpointing every 25,000 steps
- Evaluation-based best model selection
- TensorBoard logging for training visualization

### Monitoring Training Progress

Open TensorBoard to visualize training metrics in real-time:

```bash
tensorboard --logdir logs/target_reaching
```

Navigate to `http://localhost:6006` in your browser to see:
- Episode reward curves showing learning progress
- Success rate metrics for the target reaching task
- Actor/critic loss values and entropy coefficients
- Episode length distributions and termination reasons

### Evaluating Trained Models

After training completes, evaluate performance and visualize trajectories:

```bash
# Run quantitative evaluation
cd ../scripts
python evaluate.py --model ../models/target_reaching/best_model.zip --episodes 10

# Generate interactive 3D trajectory visualization
python -m flightrl_v2.tools.visualize_model \
    --model models/target_reaching/best_model.zip \
    --episodes 5
```

The visualization tool creates an HTML file with interactive Plotly plots showing:
- 3D trajectory through space
- Position, velocity, and action timeseries
- Distance to target over time
- Per-episode statistics and success indicators

### Using the Python API

For programmatic training and custom experiments:

```python
from flightrl_v2.envs import make_flight_env_for_sb3
from stable_baselines3 import SAC

# Create environment with specific configuration
env = make_flight_env_for_sb3(
    seed=42,
    max_episode_steps=600,
    render=False  # Set True to enable Unity visualization
)

# Configure SAC algorithm with custom hyperparameters
model = SAC(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    buffer_size=100_000,
    batch_size=256,
    gamma=0.99,
    verbose=1,
    tensorboard_log="./logs"
)

# Train for specified timesteps
model.learn(total_timesteps=1_000_000)

# Save trained model
model.save("trained_policy")

# Evaluate trained policy
obs, info = env.reset()
for step in range(1000):
    # Get deterministic action from policy
    action, _states = model.predict(obs, deterministic=True)
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Reset if episode ends
    if terminated or truncated:
        print(f"Episode ended at step {step}")
        obs, info = env.reset()

env.close()
```

### Advanced Configuration

Customize training through command-line arguments:

```bash
# Train with custom hyperparameters
python examples/01_basic_training.py \
    --timesteps 5000000 \
    --n_envs 16 \
    --max_episode_steps 600 \
    --learning_rate 1e-4 \
    --buffer_size 200000 \
    --batch_size 512 \
    --seed 42

# Use custom environment configuration
python examples/01_basic_training.py \
    --config path/to/custom_config.yaml \
    --timesteps 1000000
```

Or modify configuration files directly:

```yaml
# flightlib/configs/target_reaching.yaml
rl:
  pos_coeff: -0.1      # Position error penalty weight
  ori_coeff: -0.01     # Orientation error penalty weight
  lin_vel_coeff: -0.01 # Linear velocity penalty weight
  act_coeff: -0.001    # Action magnitude penalty weight

quadrotor_dynamics:
  mass: 0.73           # Vehicle mass in kg
  motor_tau: 0.0001    # Motor time constant
```

## Package Structure

```
flightrl_v2/
├── flightrl_v2/              # Main package
│   ├── core/                 # Base classes and type definitions
│   ├── envs/                 # Environment wrappers
│   │   └── wrappers/         # Observation/reward wrappers
│   ├── tasks/                # Task definitions (hover, reaching, etc.)
│   ├── algorithms/           # Training algorithms
│   │   └── callbacks/        # Custom training callbacks
│   ├── configs/              # Configuration management
│   │   └── defaults/         # Default task configs
│   ├── sensors/              # Sensor interfaces (Phase 1)
│   ├── rewards/              # Modular reward functions (Phase 3)
│   ├── deployment/           # Model export & deployment (Phase 4/5)
│   │   └── ardupilot/        # ArduPilot integration (Phase 5)
│   ├── tools/                # Utilities (rollout recording, etc.)
│   └── visualization/        # Plotting and visualization
├── scripts/                  # Command-line scripts
│   ├── train.py             # Unified training script
│   ├── evaluate.py          # Policy evaluation
│   ├── export_model.py      # Model export (Phase 4)
│   └── benchmark.py         # Performance benchmarking
├── tests/                    # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── fixtures/            # Test fixtures
├── examples/                 # Tutorial examples
└── docs/                     # Documentation
```

## Development Status

This package follows a structured development roadmap with 6 phases. Current progress:

| Phase | Status | Completion | Description |
|-------|--------|------------|-------------|
| **Phase 0** | ✅ Complete | 100% | Directory structure, packaging, and initial setup |
| **Phase 1** | Active | 85% | Core modules, environments, algorithms, and tasks |
| **Phase 2** |  Planned | 0% | Sensor pipeline integration (LIDAR, depth camera, IMU) |
| **Phase 3** |  Planned | 0% | Curriculum learning and dynamic obstacle avoidance |
| **Phase 4** |  Planned | 0% | Modular reward composition system |
| **Phase 5** |  Planned | 0% | Model export (ONNX, TorchScript) and sim-to-real transfer |
| **Phase 6** |  Planned | 0% | ArduPilot integration and hardware deployment |

### Phase 1 Status (Current Focus)

**Completed Components:**
- ✅ Core type definitions (`BaseFlightEnv`, `BaseTask`, `TaskConfig`)
- ✅ Gymnasium environment wrappers with Stable-Baselines3 support
- ✅ SAC training algorithm with custom callbacks
- ✅ Hover task (stabilization at origin)
- ✅ Target reaching task (navigation to [0, 0, 5])
- ✅ Configuration management system (YAML-based)
- ✅ Training script (`01_basic_training.py`) with command-line interface
- ✅ Evaluation and visualization tools
- ✅ Comprehensive test suite (unit and integration tests)
- ✅ Documentation (README, examples guide, API reference)

**Remaining Phase 1 Work:**
-  PPO and TD3 algorithm implementations
-  Obstacle avoidance task (static obstacles)
-  Additional sensor modalities (RGB camera, depth)
-  Extended evaluation metrics and benchmarking tools

### Known Issues and Limitations

**Training Convergence:**
- Target reaching task typically requires 1-2 million timesteps for reliable convergence
- Episodes shorter than 600 steps may prevent successful target reaching
- Recommend using 16+ parallel environments for faster training

**Environment Stability:**
- Unity rendering may cause performance degradation in long training runs
- Recommend headless mode (`render=False`) for training experiments
- Use Unity visualization only for final evaluation and demonstration

**Backward Compatibility:**
- Original `flightrl` API is deprecated but coexists in repository
- Migration guide available in `flightrl_v2/docs/migration.md`
- All new development should use `flightrl_v2` package exclusively

## API Reference

### Main Imports

```python
# Environment creation
from flightrl_v2.envs import make_flight_env_for_sb3, FlightEnvVec

# Training algorithms
from flightrl_v2.algorithms import train_sac, evaluate_policy

# Tasks
from flightrl_v2.tasks import HoverTask, TargetReachingTask

# Core abstractions
from flightrl_v2.core import BaseFlightEnv, BaseTask, TaskConfig

# Configuration
from flightrl_v2.configs import load_config
```

### Key Classes

- **`BaseFlightEnv`**: Abstract base class for all flight environments
- **`BaseTask`**: Abstract base class for task definitions
- **`FlightEnvVec`**: Vectorized environment wrapper
- **`HoverTask`**: Hover/stabilization task
- **`TargetReachingTask`**: Navigation to target position

### Configuration Files

Default configurations are provided in `flightrl_v2/configs/defaults/`:
- `hover.yaml`: Hover task configuration
- `target_reaching.yaml`: Target reaching configuration
- `obstacle_avoidance.yaml`: Obstacle avoidance (Phase 2/3)

## Examples

The `examples/` directory contains tutorial scripts:

| Example | Status | Description |
|---------|--------|-------------|
| `01_basic_training.py` | Working | Basic SAC training tutorial |
| `02_custom_reward.py` | Stub | Custom reward functions |
| `03_sensor_integration.py` | Stub | Sensor usage examples |
| `04_curriculum_learning.py` | Stub | Curriculum learning setup |
| `05_obstacle_avoidance.py` | Stub | Obstacle avoidance training |

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/unit/
pytest tests/integration/

# Run with coverage
pytest --cov=flightrl_v2 tests/
```

## Contributing

This package follows a structured development plan outlined in `REFACTORING_TASKS.md`. Key principles:

1. **Backward Compatibility**: All existing functionality must work identically
2. **Clean Architecture**: Modular design with clear responsibilities
3. **Type Safety**: Comprehensive type hints
4. **Documentation**: Docstrings for all public APIs
5. **Testing**: Unit and integration tests for all features

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'flightgym'`

```bash
# Verify flightgym C++ extension is built and installed
cd flightmare/flightlib
pip install -e .
```

Test the installation:
```python
import flightgym
print(flightgym.__file__)  # Should print path to compiled .so/.pyd file
```

**Problem:** `ImportError: cannot import name 'BaseFlightEnv' from flightrl_v2`

```bash
# Ensure flightrl_v2 is properly installed
cd flightmare/flightrl_v2
pip install -e .
```

Run comprehensive import test:
```bash
pytest tests/test_all_imports.py -v
```

### Environment Creation Fails

**Problem:** Environment creation hangs or crashes

**Causes and Solutions:**
1. **Missing Unity standalone binary**
   - Rendering requires Flightmare Unity standalone executable
   - For training, use `render=False` to disable visualization
   - Download Unity build from Flightmare releases if needed

2. **Incorrect flightgym installation**
   - Rebuild flightgym with correct Python version
   - Ensure CMake finds correct Python executable during build
   - Check that `flightlib/build/` contains compiled libraries

3. **Configuration file issues**
   - Verify config file path exists: `flightlib/configs/target_reaching.yaml`
   - Check YAML syntax for errors
   - Ensure all required parameters present (see `configs/defaults/` for examples)

**Diagnostic steps:**
```python
# Test basic flightgym functionality
from flightgym import QuadrotorEnv_v1
env = QuadrotorEnv_v1()  # Should create without errors

# Test Gymnasium wrapper
from flightrl_v2.envs import make_flight_env_for_sb3
env = make_flight_env_for_sb3(seed=42, render=False)
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")  # Should print (18,) for default setup
```

### Training Performance Issues

**Problem:** Training is very slow (< 100 FPS)

**Solutions:**
1. **Enable vectorized environments:**
   ```bash
   # Use 16 parallel environments for ~16x speedup
   python examples/01_basic_training.py --n_envs 16 --timesteps 1000000
   ```

2. **Disable rendering during training:**
   - Ensure `render=False` in `make_flight_env_for_sb3()`
   - Rendering reduces throughput from ~10,000 FPS to ~50 FPS

3. **Reduce logging frequency:**
   - TensorBoard logging every step can slow training
   - Use `log_interval=1000` in SAC configuration

4. **Optimize buffer size:**
   - Large replay buffers (> 1M) increase memory usage and sampling time
   - For quadrotor tasks, 100k-200k buffer size is typically sufficient

**Problem:** Model not converging (rewards plateau at negative values)

**Solutions:**
1. **Increase training duration:**
   - Target reaching typically converges around 1-2 million steps
   - Monitor success rate in TensorBoard: `rollout/ep_rew_mean` and custom `success_rate` metric

2. **Extend episode length:**
   - Default 300 steps may be too short for reaching distant targets
   - Use `--max_episode_steps 600` or higher for challenging tasks

3. **Tune reward coefficients:**
   - Edit `flightlib/configs/target_reaching.yaml`
   - Adjust `pos_coeff`, `ori_coeff`, `lin_vel_coeff`, `act_coeff` weights
   - More negative coefficients = stronger penalties

4. **Verify physics parameters:**
   - Check `mass`, `motor_tau`, thrust limits in config file
   - Run `python validate_physics.py` to verify dynamics consistency

**Problem:** Training crashes with CUDA out of memory

**Solutions:**
- Reduce batch size: `--batch_size 128` (default is 256)
- Reduce number of environments: `--n_envs 8` instead of 16
- Use CPU-only training: Set `device="cpu"` in SAC initialization

### Monitoring Training Progress

Open TensorBoard to visualize training metrics:
```bash
tensorboard --logdir logs/target_reaching
```

Key metrics to monitor:
- **`rollout/ep_rew_mean`**: Average episode reward (should increase toward 0)
- **`train/actor_loss`**: Actor network loss (should decrease and stabilize)
- **`train/critic_loss`**: Critic network loss (should decrease and stabilize)
- **`train/ent_coef`**: Entropy coefficient (controls exploration vs exploitation)
- **Custom `success_rate`**: Percentage of episodes reaching target successfully

Healthy training shows:
- Episode reward increasing from large negative values toward 0
- Success rate increasing from 0% toward 80-100%
- Actor/critic losses decreasing in first 100k steps, then stabilizing
- Entropy coefficient gradually decreasing as policy becomes more deterministic

## Citation

This package builds upon the Flightmare quadrotor simulator. If you use this code in your research, please cite the original Flightmare paper:

```bibtex
@inproceedings{song2020flightmare,
  title={Flightmare: A flexible quadrotor simulator},
  author={Song, Yunlong and Naji, Selim and Kaufmann, Elia and Loquercio, Antonio and Scaramuzza, Davide},
  booktitle={Conference on Robot Learning},
  pages={1147--1157},
  year={2020},
  organization={PMLR}
}
```

This reinforcement learning framework (`flightrl_v2`) provides a modern Gymnasium-compatible interface and training utilities for the Flightmare simulator.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Original Flightmare framework by RPG, University of Zurich
- Stable-Baselines3 for RL algorithm implementations
- Gymnasium for modern RL environment API

## Contact

- **Author**: Ahmed Ali
- **Email**: ali.a@aucegypt.edu
- **Repository**: https://github.com/Neurobotix/Flightmare-Reinforcement-Learning
- **Original Flightmare**: https://github.com/uzh-rpg/flightmare

---

**Version**: 2.1.0  
**Last Updated**: December 17, 2025

## Recent Updates (v2.1.0)

### Unified Logging System
- **LogManager**: Centralized logging with timestamped run directories (`logs/project/algorithm_timestamp/`)
- **Metadata Tracking**: Automatic tracking of hyperparameters, metrics, git commit/branch info
- **Alias System**: Use `@latest`, `@best`, or custom aliases instead of full model paths
- **Best Model Tracking**: Automatic tracking and symlinking of best performing model based on success rate
- **Utility Scripts**: `list_runs.py` for viewing and comparing training runs
- **Documentation**: Comprehensive logging system docs and usage examples

For details, see [CHANGELOG.md](CHANGELOG.md)

