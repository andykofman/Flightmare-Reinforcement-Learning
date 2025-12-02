# CLAUDE.md

## CRITICAL:

Each time you want to search or retrieve an information from the codebase you have to use Gemini CLI,  use it in non-interactive mode by stating gemini -p "your prompt" and flag the directories for it as well using --include-directories. Make sure to specifically ask for what you want from it with accurate prompts and direct guidlines for better responses.





This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Flightmare is a flexible modular quadrotor simulator with a modern reinforcement learning stack. This is a **maintained fork** of the original UZH-RPG Flightmare project, featuring significant improvements to the RL implementation and build system.

**Key Components:**
- **flightlib**: C++ physics engine with Python bindings (via pybind11)
- **flightrl_modern**: Modern RL stack (PyTorch + Stable-Baselines3 + Gymnasium)
- **flightrl**: Legacy RL implementation (TensorFlow 1.x - deprecated)
- **flightros**: ROS Noetic wrapper for flightlib
- **flightrender**: Unity-based rendering engine (separate Unity project)

## Architecture

### C++ Physics Engine (flightlib)

The core simulation is written in C++ and exposed to Python via pybind11:

- **Dynamics**: `flightlib/src/dynamics/quadrotor_dynamics.cpp` - Physics simulation using RK4 integration
- **Environments**: `flightlib/src/envs/` - Base environment classes (vectorized support)
- **Objects**: `flightlib/src/objects/quadrotor.cpp` - Quadrotor state, sensors, cameras
- **Bridges**: `flightlib/src/bridges/unity_bridge.cpp` - Communication with Unity renderer via ZMQ (port 10253)
- **Sensors**: IMU, RGB cameras with configurable parameters

**State Representation:**
- Position, orientation (quaternion), linear/angular velocity
- Motor states (omega values mapped to thrust via polynomial)
- Configurable via YAML files in `flightlib/configs/`

### Python RL Stack (flightrl_modern)

Modern implementation using current best practices:

**Environment Interface:**
- `flightrl_modern/flightrl_modern/envs/flight_env_vec.py` - Gymnasium-compatible vectorized environment
- `flightrl_modern/flightrl_modern/envs/gymnasium_wrapper.py` - Helper functions for SB3 integration
- Wraps `flightgym.QuadrotorEnv_v1` C++ environment with Gymnasium API

**Algorithms:**
- SAC (Soft Actor-Critic) is the default and recommended algorithm
- Full Stable-Baselines3 support (PPO, TD3, etc.)
- Training scripts in `flightrl_modern/examples/train_sac.py`
- Evaluation utilities in `flightrl_modern/flightrl_modern/algorithms/evaluate.py`

**Key Design Patterns:**
1. Config files drive environment parameters (`flightlib/configs/*.yaml`)
2. Environment requires `FLIGHTMARE_PATH` environment variable
3. Unity renderer is optional (runs on port 10253, can connect `host.docker.internal`)
4. Vectorized environments for parallel training

### ROS Integration (flightros)

ROS Noetic wrapper provides:
- Topic publishing for quadrotor state
- Integration with `autopilot`, `quadrotor_common`, `quadrotor_msgs`
- Launch files in `flightros/launch/`
- Requires full ROS workspace build (catkin)

## Build Commands

### Environment Setup

```bash
# Required environment variable
export FLIGHTMARE_PATH=/path/to/flightmare
```

### Building flightlib (C++ Physics Engine)

```bash
cd flightlib
pip install .

# Development mode (for active development)
pip install -e .

# Verify installation
python -c "import flightgym; print('✓ flightgym OK')"
```

**Build Process:**
- Uses CMake with custom setup.py that cleans cached files
- Downloads dependencies: Eigen3, yaml-cpp, pybind11, googletest
- Requires: cmake, gcc/g++, python3-dev, libopencv-dev, libeigen3-dev
- Parallel build with `-j4`

### Building flightrl_modern (Python RL)

```bash
cd flightrl_modern
pip install .

# Development mode
pip install -e .

# With dev dependencies (pytest, black, flake8)
pip install -e ".[dev]"
```

### Building ROS Package (flightros)

```bash
# From catkin workspace root
catkin build flightros

# Or using catkin_make
catkin_make

# Source the workspace
source devel/setup.bash
```

### Docker Build (Recommended)

```bash
cd docker

# Full build (CPU + ROS + RL)
./build_container.ps1

# GPU build with CUDA 11.8
./build_container.ps1 -Gpu -CudaVersion cu118

# Minimal build (no ROS, no RL)
./build_container.ps1 -NoRos -NoRl

# Run container
docker run -it --rm -p 10253:10253 flightmare:latest

# Inside container: verify installation
/root/verify_installation.sh
```

## Testing

### Python RL Tests

```bash
cd flightrl_modern

# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_environment.py -v

# With coverage
pytest tests/ --cov=flightrl_modern --cov-report=html

# Quick smoke test (verifies everything works)
python examples/smoke_test.py
```

### Physics Validation

```bash
cd flightrl_modern
python validate_physics.py
```

Comprehensive physics validation suite testing:
- Fundamental physics (gravity, thrust, mass)
- Control response (motors, dynamics)
- Kinematic consistency (position/velocity/acceleration)
- System integration (multi-axis, tracking)

### C++ Tests

```bash
cd flightlib/build
cmake .. -DBUILD_TESTS=ON
make -j4
ctest --output-on-failure
```

## Training and Evaluation

### Training a SAC Agent

```bash
cd flightrl_modern/examples

# Quick test (10k steps)
python train_sac.py --timesteps 10000

# Full training (1M steps, 4 parallel envs)
python train_sac.py --timesteps 1000000 --n_envs 4 --seed 42

# With rendering (connects to Unity on port 10253)
python train_sac.py --timesteps 100000 --render
```

**Training artifacts saved to:**
- Models: `./models/sac/`
- Logs: `./logs/` (TensorBoard format)
- Checkpoints: `./checkpoints/`

### Evaluating Models

```bash
cd flightrl_modern/examples

# Evaluate best model
python evaluate_sac.py --model ../models/sac/best_model.zip --episodes 10 --render

# Evaluate specific checkpoint
python evaluate_model.py --model_path /path/to/model.zip --episodes 100
```

## Common Development Workflows

### Modifying Physics Parameters

1. Edit config file: `flightlib/configs/target_reaching.yaml` (or relevant config)
2. Rebuild flightlib if C++ code changed: `cd flightlib && pip install .`
3. Test changes: `python flightrl_modern/validate_physics.py`
4. Train with new config: Environment automatically loads updated YAML

### Adding New RL Algorithm

1. Create algorithm file in `flightrl_modern/flightrl_modern/algorithms/`
2. Use existing SB3 algorithms or implement custom
3. Add training script to `flightrl_modern/examples/`
4. Update `flightrl_modern/flightrl_modern/algorithms/__init__.py`

### Debugging Physics Issues

1. Run physics validation: `python validate_physics.py`
2. Check motor omega ranges in config (`motor_omega_min`, `motor_omega_max`)
3. Verify thrust mapping polynomial coefficients
4. Test with rendering enabled to visualize behavior
5. Check simulation timestep (`sim_dt` in config, default 0.02s = 50Hz)

### Working with Unity Renderer

Unity renderer runs separately and connects via ZMQ (port 10253):

1. Unity project location: `flightrender/` directory
2. Connection: Set `render: yes` in environment config or `make_flight_env(render=True)`
3. Docker: Unity connects to `host.docker.internal:10253`
4. Troubleshooting: Check ZMQ port availability, firewall rules

## Important Technical Details

### Environment Variable Requirements

```bash
export FLIGHTMARE_PATH=/path/to/flightmare  # Required for all builds
```

### Python Dependencies

**flightlib:**
- ruamel.yaml (config parsing)
- numpy (< 2.0.0 for compatibility)
- opencv-python (camera rendering)

**flightrl_modern:**
- gymnasium >= 0.28.0
- stable-baselines3 >= 2.0.0
- torch >= 2.0.0
- numpy >= 1.21.0, < 2.0.0

### Configuration System

All environment parameters are in YAML files:
- `flightlib/configs/target_reaching.yaml` - Default target reaching task
- Loaded automatically based on environment type
- Parameters: mass, arm_length, motor limits, thrust mapping, reward coefficients, simulation timestep

### Motor and Thrust Model

Motors use polynomial thrust mapping:
```
thrust = c[0] * omega^2 + c[1] * omega + c[2]
```
Coefficients in `thrust_map` config parameter. Motor response time controlled by `motor_tau` (default: 0.0001s for near-instant response).

### Coordinate System

- Position: [x, y, z] in world frame (NED-like, z-up)
- Orientation: Quaternion [w, x, y, z]
- Actions: Motor commands (4 values, mapped to thrust)
- Target reaching default: [0, 0, 5] (5m altitude)

### Build Caching Issues

The flightlib setup.py **automatically clears** cached external files and build files to prevent CMake errors. If you encounter build issues:

1. Clean manually: `rm -rf flightlib/build flightlib/externals/*`
2. Rebuild: `cd flightlib && pip install .`
3. Check environment variable: `echo $FLIGHTMARE_PATH`

### Legacy Code (flightrl)

**Do not use or modify `flightrl/` directory** - it contains deprecated TensorFlow 1.x code. Use `flightrl_modern/` instead. Migration guide available in `flightrl_modern/README.md`.

## Git Workflow

Current branch: `first-stable-v`
Main branch: `main` (use for PRs)

Modified files in current state:
- `flightlib/configs/target_reaching.yaml` - Physics parameters
- `flightrl_modern/examples/evaluate_model.py` - Evaluation utilities
- `flightrl_modern/examples/train_sac.py` - Training script
- `flightrl_modern/flightrl_modern/tools/rollout_recorder.py` - Recording tools
- `flightrl_modern/validate_physics.py` - Physics validation

## Platform Notes

**Windows (Current Platform):**
- Use PowerShell scripts: `docker/build_container.ps1`
- Python paths may use backslashes
- Docker Desktop required for containerized builds

**Linux (Docker Target):**
- Primary deployment platform
- All build scripts tested on Ubuntu 20.04
- ROS Noetic support

## References

- Original Paper: http://rpg.ifi.uzh.ch/docs/CoRL20_Yunlong.pdf
- Original Documentation: https://flightmare.readthedocs.io/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- Gymnasium: https://gymnasium.farama.org/
