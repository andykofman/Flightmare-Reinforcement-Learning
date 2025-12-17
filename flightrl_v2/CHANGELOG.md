# Changelog

All notable changes to flightrl_v2 will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-12-17

### Added - Unified Logging System
- **LogManager**: Centralized logging with timestamped run directories (`logs/project/algorithm_timestamp/`)
- **Run Metadata Tracking**: Automatic tracking of hyperparameters, metrics, git commit/branch info
- **Symbolic Links**: `@latest` and `@best` aliases automatically maintained for easy model access
- **Custom Aliases**: Create custom aliases (e.g., `@baseline`, `@production`) for important runs
- **Best Model Tracking**: Automatic tracking and symlinking of best performing model based on success rate
- **Directory Structure**: Organized subdirectories for checkpoints, tensorboard logs, and evaluation results
- **Utility Scripts**: `list_runs.py` for viewing and comparing training runs
- **Model Resolution**: `resolve_model_alias()` function for easy model path resolution
- **Human-Readable Config**: Automatic saving of training configuration to text file

### Infrastructure
- Git commit tracking for reproducibility
- Automatic directory creation and management
- Cross-platform symlink support (fallback to text files on Windows)
- Metadata persistence in JSON format

## [2.0.0] - 2025-12-17

### Added - Core Architecture (Major Refactoring)
- Modern Gymnasium-compatible environment wrappers
- Modular task system with base classes for easy extension
- Composite reward system for flexible reward shaping
- Sensor infrastructure with base classes for LIDAR, IMU, depth camera
- Type-safe configuration system with YAML support
- Comprehensive test suite with unit and integration tests

### Added - Training Algorithms
- SAC (Soft Actor-Critic) training with optimized hyperparameters
- PPO and TD3 training stubs (placeholder for future implementation)
- Evaluation utilities with deterministic and stochastic testing
- Callback system for metrics and curriculum learning (Phase 2/3 placeholder)

### Added - Tasks
- Hover task: Maintain stable position at origin
- Target reaching task: Navigate to 3D coordinates and stabilize
- Obstacle avoidance task (stub for Phase 3)

### Added - Tools & Utilities
- Rollout recorder for trajectory capture and visualization
- Plotly-based 3D trajectory visualization
- Model export utilities for deployment
- TensorBoard integration for training metrics

### Added - Documentation
- Complete API documentation
- Example scripts for training and evaluation
- Configuration guide with YAML schema

### Changed
- **Breaking**: Migrated from `flightrl_modern` to `flightrl_v2` namespace
- **Breaking**: All imports now use `from flightrl_v2.X import Y`
- Improved observation space handling with proper bounds
- Enhanced reward system with modular composition
- Standardized error messages and logging format

## [1.0.0] - 2025-11-XX (flightrl_modern)

### Initial Release
- Basic SAC training for target seeking and hovering
- Manual logging with fixed directory structure
- Environment wrappers for Flightmare C++ backend
- Basic evaluation scripts

---

## Version History

### v2.1.0 - Unified Logging System (Current)
Added comprehensive logging infrastructure with metadata tracking, aliases, and automatic best model management.

### v2.0.0 - Major Refactoring
Complete architectural overhaul with modular design, modern Gymnasium compatibility, and comprehensive testing.

### v1.0.0 - Initial Implementation (Legacy - flightrl_modern)
Basic RL training capabilities for Flightmare simulator.

---

## Upgrade Guide: v1.0.0 → v2.0.0

### Import Changes
```python
# Old (v1.0.0)
from flightrl_modern.target_seeking.envs import TargetSeekingEnv

# New (v2.0.0)
from flightrl_v2.envs import FlightEnvVec
from flightrl_v2.tasks import TargetReachingTask
```

### Logging Changes
```python
# Old (v1.0.0)
model_dir = "logs/sac_target_seeking"
model.save(f"{model_dir}/final_model")

# New (v2.0.0)
from flightrl_v2.utils import LogManager

logger = LogManager(
    project="target_reaching",
    algorithm="SAC",
    config=args
)
model = SAC(..., tensorboard_log=logger.tensorboard_dir)
logger.save_model(model, "final")
logger.finalize()
```

### Model Loading Changes
```bash
# Old (v1.0.0)
python evaluate.py --model logs/sac_target_seeking/final_model.zip

# New (v2.0.0)
python evaluate.py --model @latest
# or
python evaluate.py --model @best
```

---

## Notes

- All v2.x.x releases maintain backward compatibility within major version
- Breaking changes will increment major version (e.g., v3.0.0)
- See `/docs` for complete documentation
- Report issues at: https://github.com/andykofman/Flightmare-Reinforcement-Learning/issues
