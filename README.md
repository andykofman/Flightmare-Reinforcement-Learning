# Flightmare - Modern Reinforcement Learning Edition

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Maintained](https://img.shields.io/badge/Maintained-yes-green.svg)

**Flightmare** is a flexible modular quadrotor simulator. Flightmare is composed of two main components: a configurable rendering engine built on Unity and a flexible physics engine for dynamics simulation. Those two components are totally decoupled and can run independently from each other. Flightmare comes with several desirable features: (i) a large multi-modal sensor suite, including an interface to extract the 3D point-cloud of the scene; (ii) an API for reinforcement learning which can simulate hundreds of quadrotors in parallel; and (iii) an integration with a virtual-reality headset for interaction with the simulated environment. Flightmare can be used for various applications, including path-planning, reinforcement learning, visual-inertial odometry, deep learning, human-robot interaction, etc.

> **Note**: This is a maintained fork of the original Flightmare project (which is no longer actively maintained). This version includes significant improvements to the reinforcement learning stack and build system.

**[Original Website](https://uzh-rpg.github.io/flightmare/)** | 
**[Original Documentation](https://flightmare.readthedocs.io/)**

[![IMAGE ALT TEXT HERE](./docs/flightmare_main.png)](https://youtu.be/m9Mx1BCNGFU)

## What's New in This Edition

This maintained edition includes significant updates and improvements to the original Flightmare project:

### Modern Reinforcement Learning Stack (flightrl_v2)

The legacy reinforcement learning implementation has been completely rewritten as **`flightrl_v2`** - a production-ready, modular framework:

- **Modern Python stack**: PyTorch 2.0+, Stable-Baselines3 2.0+, Gymnasium 0.28+
- **Clean architecture**: Modular design with clear separation of concerns (core, envs, tasks, algorithms)
- **Type-safe**: Comprehensive type hints throughout the codebase
- **Well-tested**: Unit and integration tests with pytest
- **Fully documented**: Complete API documentation and usage examples
- **SAC algorithm**: Production-ready Soft Actor-Critic implementation for continuous control
- **Multiple tasks**: Hover, target reaching, and obstacle avoidance (Phase 1+)
- **Vectorized environments**: Fast parallel training with multiple environments

### Build System Improvements

All build issues have been resolved with a complete Docker-based build system:

- **Drag-and-drop Docker setup** in the `docker/` directory - build and run with minimal configuration
- **Multi-stage Docker builds** for optimized image sizes
- **Flexible build options**: CPU/GPU support, optional ROS/RL components
- **Automated verification scripts** to ensure correct installation
- **Comprehensive build documentation** and troubleshooting guides

See the [Docker Documentation](./docker/README.md) for detailed build instructions.

### Key Features

- **Production-ready RL stack**: Fully tested and verified implementation
- **Gymnasium-compatible interface**: Works seamlessly with modern RL libraries
- **GPU/CPU support**: CUDA acceleration for faster training
- **Vectorized environments**: Multi-environment support for parallel training
- **Comprehensive documentation**: Complete learning guides and API documentation
- **Backwards compatibility**: Migration guide for existing users

## Installation

### Docker Installation (Recommended)

The easiest way to get started is using Docker:

```powershell
# Build the container
cd docker
.\build_container.ps1

# Run the container
docker run -it --rm -p 10253:10253 flightmare:latest
```

See the [Docker Documentation](./docker/README.md) for detailed instructions, build options, and verification procedures.

### Manual Installation

For manual installation, refer to the original [Flightmare Wiki](https://github.com/uzh-rpg/flightmare/wiki) and the [flightrl_modern README](./flightrl_modern/README.md).

## Quick Start with flightrl_v2

### Training a Target Reaching Agent

```python
from flightrl_v2.envs import make_flight_env_for_sb3
from stable_baselines3 import SAC

# Create environment
env = make_flight_env_for_sb3(seed=42, max_episode_steps=600)

# Train SAC agent
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)

# Save model
model.save("trained_policy")
```

### Command Line Training

```bash
cd flightrl_v2/examples

# Quick test run
python 01_basic_training.py --timesteps 10000

# Full training run (target reaching task)
python 01_basic_training.py --timesteps 1000000 --n_envs 16 --max_episode_steps 600
```

### Monitoring Training

```bash
# View training progress in TensorBoard
tensorboard --logdir logs/target_reaching
```

### Evaluation and Visualization

```bash
cd ../scripts

# Evaluate trained model
python evaluate.py --model ../models/target_reaching/best_model.zip --episodes 10

# Generate 3D trajectory visualizations
python -m flightrl_v2.tools.visualize_model \
    --model models/target_reaching/best_model.zip \
    --episodes 5
```

See the [flightrl_v2 README](./flightrl_v2/README.md) for complete documentation and examples.

## Project Structure

```
flightmare/
├── flightlib/              # C++ physics engine (core simulation)
│   ├── include/           # Header files
│   ├── src/               # Source files
│   └── configs/           # Configuration files
├── flightrl_v2/           # Modern Python RL framework (v2.0)
│   ├── flightrl_v2/       # Main package
│   │   ├── core/          # Base classes and types
│   │   ├── envs/          # Gymnasium-compatible environments
│   │   ├── tasks/         # Task definitions (hover, reaching, etc.)
│   │   ├── algorithms/    # Training algorithms and callbacks
│   │   ├── configs/       # Configuration management
│   │   ├── tools/         # Utilities (visualization, etc.)
│   │   └── sensors/       # Sensor interfaces (Phase 2+)
│   ├── examples/          # Training examples and tutorials
│   ├── scripts/           # Command-line tools
│   ├── tests/             # Test suite
│   └── docs/              # Package documentation
├── docker/                # Docker build system
│   ├── Dockerfile         # Multi-stage build definition
│   ├── build_container.ps1 # Build script
│   └── README.md          # Docker documentation
└── docs/                  # Original Flightmare documentation
```

## Getting Started with flightrl_v2

### Installation

```bash
# Install flightlib (C++ backend)
cd flightlib
pip install -e .

# Install flightrl_v2
cd ../flightrl_v2
pip install -e .
```

### Your First Training Run

```bash
cd flightrl_v2/examples
python 01_basic_training.py --timesteps 10000  # Quick test
```

### Package Features

- **Clean API**: Simple, intuitive interfaces for environment creation and training
- **Multiple tasks**: Hover stabilization, target reaching, obstacle avoidance
- **Flexible configuration**: YAML-based configuration system
- **Comprehensive examples**: Step-by-step tutorials in `examples/` directory
- **Full documentation**: See [flightrl_v2/README.md](./flightrl_v2/README.md)

### Migration from Legacy Code

If you have code using the old `flightrl` package:

1. **Import changes**: `from flightrl_v2.envs import make_flight_env_for_sb3`
2. **Environment creation**: Use Gymnasium API (returns 5-tuple: obs, reward, terminated, truncated, info)
3. **Training**: Compatible with any Stable-Baselines3 algorithm
4. **Configuration**: Use YAML config files in `flightlib/configs/`

See the [flightrl_v2 README](./flightrl_v2/README.md) for detailed migration guide and examples.

## Documentation

- **Main Documentation**: [Flightmare Documentation](https://flightmare.readthedocs.io/)
- **flightrl_v2 Package**: [flightrl_v2/README.md](./flightrl_v2/README.md)
- **Training Examples**: [flightrl_v2/examples/README.md](./flightrl_v2/examples/README.md)
- **Docker Build System**: [docker/README.md](./docker/README.md)
- **API Reference**: See docstrings in `flightrl_v2/flightrl_v2/`

## Updates

- **2025-12-17**: **v2.0.0** - Complete rewrite as flightrl_v2 with modular architecture, comprehensive tests, and full documentation
- **2025-11-07**: Docker-based build system with drag-and-drop setup and build verification
- **2025-11-07**: All build issues resolved and verified
- **2025-10-21**: Initial modernization of RL stack with PyTorch + Stable-Baselines3 + Gymnasium
- **2020-11-17**: [Spotlight](https://youtu.be/8JyrjPLt8wo) Talk at CoRL 2020
- **2020-09-04**: Original Flightmare release

## Publication

If you use this code in a publication, please cite the following paper **[PDF](http://rpg.ifi.uzh.ch/docs/CoRL20_Yunlong.pdf)**

```
@inproceedings{song2020flightmare,
    title={Flightmare: A Flexible Quadrotor Simulator},
    author={Song, Yunlong and Naji, Selim and Kaufmann, Elia and Loquercio, Antonio and Scaramuzza, Davide},
    booktitle={Conference on Robot Learning},
    year={2020}
}
```

## Contributors

### flightrl_v2 Implementation

- **Ahmed Ali** - flightrl_v2 framework, modern RL stack, and Docker build system
  - Email: ali.a@aucegypt.edu
  - GitHub: [Neurobotix](https://github.com/Neurobotix/Flightmare-Reinforcement-Learning)

### Original Flightmare

- **Yunlong Song** - Original Flightmare implementation
- **Selim Naji** - Core development
- **Elia Kaufmann** - Core development
- **Antonio Loquercio** - Core development
- **Davide Scaramuzza** - Project lead

## License

This project is released under the MIT License. Please review the [License file](LICENSE) for more details.

## Project Status

This is a **maintained fork** of the original Flightmare project. The original repository at [uzh-rpg/flightmare](https://github.com/uzh-rpg/flightmare) is no longer actively maintained. This edition continues development with:

- **v2.0.0 Release**: Production-ready flightrl_v2 framework
- **Active maintenance**: Bug fixes and regular updates
- **Modern architecture**: Clean, modular, type-safe Python codebase
- **Comprehensive testing**: Unit and integration test suite
- **Full documentation**: Complete guides, examples, and API reference
- **Docker support**: Easy deployment and development environment

## Acknowledgments

This project builds upon the excellent work of the original Flightmare team at the Robotics and Perception Group, University of Zurich. The modernization effort focuses on updating the reinforcement learning stack to use current best practices and resolving build system issues for easier deployment and development. We are grateful to the original authors for their groundbreaking work in quadrotor simulation.
