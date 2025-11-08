# Flightrl Modern - Examples

This directory contains example scripts for training and evaluating RL agents on Flightmare.

## Quick Start

### 1. Smoke Test (Verify Installation)

```bash
python smoke_test.py
```

This runs a quick test to verify that all components are installed and working correctly.

### 2. Training

#### Basic Training (SAC)
```bash
python train_sac.py --timesteps 1000000 --seed 42
```

#### Training with Multiple Environments
```bash
python train_sac.py --timesteps 5000000 --n_envs 4 --seed 42
```

#### Training with Rendering (Slow)
```bash
python train_sac.py --timesteps 100000 --render --seed 42
```

### 3. Evaluation

```bash
python evaluate_sac.py --model ./models/sac/best_model.zip --episodes 10
```

#### Evaluation with Rendering
```bash
python evaluate_sac.py --model ./models/sac/best_model.zip --episodes 5 --render --deterministic
```

#### Evaluation with Video Recording
```bash
python evaluate_sac.py --model ./models/sac/best_model.zip --episodes 3 --save_video --video_folder ./videos
```

### 4. Legacy-Compatible Script

For users migrating from the old `run_drone_control.py`:

```bash
# Training
python run_drone_control_modern.py --train 1 --seed 42 --timesteps 25000000

# Testing
python run_drone_control_modern.py --train 0 --weight ./saved/best_model.zip --render 1
```

## Scripts Overview

### Training & Evaluation
- **`smoke_test.py`**: Quick verification that everything is installed correctly
- **`train_sac.py`**: Main training script for SAC algorithm
- **`evaluate_sac.py`**: Evaluation script for trained models with Unity visualization support
- **`run_drone_control_modern.py`**: Modern version of legacy `run_drone_control.py`

### Analysis & Visualization
- **`plot_training_metrics.py`**: Plot training curves (reward, episode length, FPS) from TensorBoard or CSV logs
- **`analyze_test_results.py`**: Comprehensive diagnostics and recommendations for training issues

## Configuration

All scripts use the default configuration from `flightlib/configs/vec_env.yaml`.

You can override the configuration by:
1. Passing `--config /path/to/your/config.yaml`
2. Modifying the default config file

## Hyperparameters

### SAC Default Hyperparameters

- Learning rate: 3e-4
- Batch size: 256
- Buffer size: 1,000,000
- Gamma: 0.99
- Tau: 0.005

You can override these via command-line arguments. See `python train_sac.py --help` for all options.

## Training Tips

1. **Use multiple environments** (`--n_envs 4` or more) for faster training
2. **Monitor training** with TensorBoard:
   ```bash
   tensorboard --logdir ./logs/sac
   ```
3. **Save checkpoints** are saved every 50,000 steps by default
4. **Best model** is automatically saved based on evaluation performance

## Troubleshooting

### "flightgym module not found"
- Make sure flightlib is installed: `cd flightlib && pip install .`
- Set `FLIGHTMARE_PATH` environment variable

### CUDA errors
- For CPU-only training, PyTorch will automatically fall back to CPU
- For GPU training, ensure CUDA version matches PyTorch installation

### Out of memory
- Reduce `--n_envs`
- Reduce `--buffer_size`
- Reduce `--batch_size`

## Analysis Tools

### Plot Training Metrics

Visualize training progress with reward curves and statistics:

```bash
# Auto-detect latest training run
python plot_training_metrics.py

# Plot from specific TensorBoard logs
python plot_training_metrics.py --tensorboard runs/SAC_1234567890

# Plot from CSV file
python plot_training_metrics.py --csv saved/progress.csv
```

**Features:**
- Episode reward tracking (raw + smoothed)
- Episode length trends
- Training speed (FPS)
- Summary statistics (max, final, average)

**Requirements:** `pip install pandas matplotlib tensorboard`

### Analyze Test Results

Diagnose training issues and get actionable recommendations:

```bash
# Auto-detect and analyze
python analyze_test_results.py

# Analyze specific run
python analyze_test_results.py --tensorboard runs/SAC_1234567890

# Include trajectory visualization
python analyze_test_results.py --trajectories test_trajectories.png
```

**Diagnostics:**
- Episode length analysis (crash detection, stability)
- Reward function balance check
- Catastrophic forgetting detection
- Observation normalization verification
- Actionable recommendations

**Requirements:** `pip install pandas matplotlib tensorboard pillow`

### Typical Analysis Workflow

1. **During Training:** Monitor with TensorBoard
   ```bash
   tensorboard --logdir logs/sac/
   ```

2. **Plot Progress:** Visualize training curves
   ```bash
   python plot_training_metrics.py
   ```

3. **After Training:** Run comprehensive diagnostics
   ```bash
   python analyze_test_results.py
   ```

4. **Evaluate:** Test with Unity rendering
   ```bash
   python evaluate_sac.py --model saved/best_model.zip --render --episodes 10
   ```

## Unity Visualization

To use `--render` with Unity:

1. **Start Docker container** with port mapping:
   ```bash
   docker-compose up -d
   docker-compose exec flightmare bash
   ```

2. **Open Unity Editor** on host machine and load Flightmare scene

3. **Configure Unity** to connect to `127.0.0.1` (localhost)

4. **Run evaluation** with rendering:
   ```bash
   python evaluate_sac.py --model saved/best_model.zip --render
   ```

The script will automatically call `env.connect_unity()` when `--render` is enabled.

## Saved Files Structure

```
examples/
├── saved/
│   ├── sac_flightmare_1234567890/
│   │   ├── best_model.zip          # Best checkpoint
│   │   ├── final_model.zip         # Final checkpoint
│   │   ├── rl_model_500000_steps.zip
│   │   └── vec_normalize.pkl       # Normalization stats (if used)
├── logs/
│   └── sac/
│       ├── SAC_1/
│       │   └── events.out.tfevents.*   # TensorBoard logs
│       ├── SAC_2/
│       └── SAC_3/
└── training_metrics.png            # Generated plots
```
