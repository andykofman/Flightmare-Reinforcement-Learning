# FlightRL v2 Examples

This directory contains practical examples demonstrating how to use the flightrl_v2 framework for training reinforcement learning agents to control quadrotors in the Flightmare simulator.

## Available Examples

### 01_basic_training.py (Complete)

Full-featured training pipeline for the target reaching task using Soft Actor-Critic (SAC).

**What it demonstrates:**
- Creating and configuring training environments with proper seeding
- Setting up vectorized environments for parallel training
- Configuring SAC hyperparameters optimized for quadrotor control
- Implementing custom callbacks for task-specific metrics tracking
- Using multiple callbacks (checkpointing, evaluation, success tracking)
- TensorBoard integration for training visualization
- Model saving and evaluation setup

**Task:** Train quadrotor to reach target position [0, 0, 5] and stabilize within 0.5 meters with velocity under 0.5 m/s for at least 1 second.

**Usage examples:**
```bash
# Quick test run (10,000 steps for debugging)
python 01_basic_training.py --timesteps 10000

# Standard training (500,000 steps)
python 01_basic_training.py --timesteps 500000 --n_envs 4

# Recommended: Extended training for convergence (around 1 million steps)
python 01_basic_training.py --timesteps 1000000 --n_envs 16 --max_episode_steps 600

# Long training for best results (5 million steps)
python 01_basic_training.py --timesteps 5000000 --seed 42 --n_envs 16 --max_episode_steps 600

# Training with Unity rendering enabled (slower, useful for visualization)
python 01_basic_training.py --timesteps 100000 --render
```

**Key command-line arguments:**
- `--timesteps`: Total number of training steps to run
- `--n_envs`: Number of parallel environments (higher = faster training)
- `--max_episode_steps`: Maximum steps per episode before truncation
- `--seed`: Random seed for reproducibility
- `--learning_rate`: SAC optimizer learning rate (default: 3e-4)
- `--buffer_size`: Replay buffer size (default: 100,000)
- `--batch_size`: Training batch size (default: 256)
- `--log_dir`: Directory for TensorBoard logs (default: ./logs/target_reaching)
- `--save_dir`: Directory for saved models (default: ./models/target_reaching)

**Expected results:**
- Model typically converges around 1 million training steps
- Success rate should reach above 80% by convergence
- Best model is automatically saved to `models/target_reaching/best_model.zip`
- Training logs can be viewed with: `tensorboard --logdir logs/target_reaching`

### 02_custom_reward.py (Planned - Phase 2)

This example will demonstrate how to define custom reward functions for different tasks.

**Planned features:**
- Creating custom reward components
- Combining multiple reward terms with weights
- Implementing shaped rewards for faster learning
- Comparing different reward formulations

**Status:** Not yet implemented. Currently using default position-based rewards defined in the environment configuration.

### 03_sensor_integration.py (Planned - Phase 3)

This example will demonstrate integrating additional sensors for perception-based control.

**Planned features:**
- Adding LIDAR sensors for obstacle detection
- Using depth cameras for visual navigation
- Processing sensor data in observations
- Training policies that use sensor inputs

**Status:** Not yet implemented. Current examples use state-based observations only.

### 04_curriculum_learning.py (Planned - Phase 3)

This example will demonstrate curriculum learning strategies for complex tasks.

**Planned features:**
- Gradually increasing task difficulty during training
- Automatic difficulty adjustment based on performance
- Multi-stage training pipelines
- Transfer learning between difficulty levels

**Status:** Not yet implemented. Current training uses fixed difficulty.

### 05_obstacle_avoidance.py (Planned - Phase 4)

This example will demonstrate training for obstacle avoidance tasks.

**Planned features:**
- Dynamic obstacle environments
- Collision detection and penalties
- Navigation through cluttered spaces
- Safety-aware policy training

**Status:** Not yet implemented. Current environments do not include obstacles.

## Quick Start Guide

### Prerequisites

Ensure you have installed flightrl_v2 and its dependencies:

```bash
cd flightmare/flightrl_v2
pip install -e .
```

Set the FLIGHTMARE_PATH environment variable (if not already set):

```bash
# Linux/Mac
export FLIGHTMARE_PATH=/path/to/flightmare

# Windows PowerShell
$env:FLIGHTMARE_PATH="D:\path\to\flightmare"
```

### Running Your First Training

Start with a quick test to verify everything works:

```bash
cd examples
python 01_basic_training.py --timesteps 10000 --n_envs 1
```

This will train for 10,000 steps with a single environment. You should see output indicating training progress, episode statistics, and model checkpointing.

For serious training, use more steps and parallel environments:

```bash
python 01_basic_training.py --timesteps 1000000 --n_envs 16 --max_episode_steps 600
```

### Monitoring Training Progress

Training metrics are logged to TensorBoard. Start TensorBoard to monitor progress:

```bash
tensorboard --logdir logs/target_reaching
```

Open your browser to `http://localhost:6006` to view:
- Episode rewards over time
- Success rate metrics
- Loss curves (actor, critic, entropy)
- Learning rate schedules

### Evaluating Trained Models

After training completes, evaluate your trained model:

```bash
# Using the provided evaluation script
cd ../scripts
python evaluate.py --model ../models/target_reaching/best_model.zip --episodes 10
```

Or visualize the trained policy:

```bash
# Generate interactive 3D visualizations
python -m flightrl_v2.tools.visualize_model --model models/target_reaching/best_model.zip --episodes 5
```

This creates an HTML visualization showing the drone's trajectory, which you can open in a web browser.

## Project Structure

```
examples/
├── README.md                    # This file
├── 01_basic_training.py         # Complete SAC training example
├── 02_custom_reward.py          # (Planned) Custom reward functions
├── 03_sensor_integration.py     # (Planned) Sensor-based control
├── 04_curriculum_learning.py    # (Planned) Curriculum learning
└── 05_obstacle_avoidance.py     # (Planned) Obstacle avoidance
```

## Common Issues and Solutions

### Issue: "Model not found" or import errors

**Solution:** Ensure flightrl_v2 is installed in development mode:
```bash
cd flightmare/flightrl_v2
pip install -e .
```

### Issue: "FLIGHTMARE_PATH not set"

**Solution:** The framework will attempt to auto-detect the configuration path, but you can explicitly set it:
```bash
export FLIGHTMARE_PATH=/path/to/flightmare
```

Or provide the config path directly:
```bash
python 01_basic_training.py --config /path/to/target_reaching.yaml
```

### Issue: Training is very slow

**Solution:** Increase parallel environments for faster data collection:
```bash
python 01_basic_training.py --n_envs 16  # Use 16 parallel environments
```

Note: Each environment uses CPU resources, so match the number to your available cores.

### Issue: Model not converging

**Possible solutions:**
- Train for more steps (try 1-5 million steps)
- Increase maximum episode length: `--max_episode_steps 600`
- Adjust learning rate: `--learning_rate 1e-4` (lower) or `--learning_rate 1e-3` (higher)
- Increase replay buffer size: `--buffer_size 200000`

## Next Steps

After successfully training with `01_basic_training.py`:

1. **Experiment with hyperparameters**: Try different learning rates, buffer sizes, and network architectures
2. **Extend training duration**: Models often continue improving beyond 1 million steps
3. **Modify the task**: Edit the configuration file to change target positions or success criteria
4. **Try different algorithms**: Implement PPO or TD3 training scripts (templates available in `flightrl_v2/algorithms/`)
5. **Deploy to hardware**: Use the deployment tools in `flightrl_v2/deployment/ardupilot/` for real quadrotor testing

## Additional Resources

- **Main documentation**: `flightmare/flightrl_v2/README.md`
- **API reference**: `flightmare/flightrl_v2/docs/`
- **Configuration files**: `flightmare/flightlib/configs/`
- **Test scripts**: `flightmare/flightrl_v2/tests/`

## Contributing

If you create additional examples, please:
1. Follow the naming convention: `XX_descriptive_name.py`
2. Include comprehensive docstrings explaining the example's purpose
3. Add command-line arguments with sensible defaults
4. Update this README with usage instructions
5. Submit a pull request with your example

For questions or issues, please open an issue on the project repository.
