# Flightmare Docker Build Verification

This document provides verification steps and expected outputs for the Flightmare Docker build.

## Build Information

- **Base Image**: Ubuntu 20.04 LTS
- **Python Version**: 3.9
- **ROS Version**: Noetic
- **RL Framework**: PyTorch + Stable-Baselines3 + Gymnasium

## Components

### Core Components

1. **flightlib** - C++ simulation core with Python bindings (`flightgym`)
2. **flightros** - ROS Noetic wrapper and integration
3. **flightrl_modern** - Modern RL implementation (PyTorch + SB3)

### Excluded Components

- **flightrl** (legacy) - Deprecated TensorFlow 1.x + stable-baselines 2.x implementation

## Verification Steps

### 1. Basic Import Tests

Run inside container:

```bash
# Test flightgym
python3 -c "import flightgym; print('✓ flightgym OK')"

# Test NumPy
python3 -c "import numpy as np; print('✓ NumPy', np.__version__)"

# Test PyTorch
python3 -c "import torch; print('✓ PyTorch', torch.__version__); print('  CUDA:', torch.cuda.is_available())"

# Test Stable-Baselines3
python3 -c "import stable_baselines3 as sb3; print('✓ SB3', sb3.__version__)"

# Test Gymnasium
python3 -c "import gymnasium as gym; print('✓ Gymnasium', gym.__version__)"

# Test flightrl_modern
python3 -c "import flightrl_modern; print('✓ flightrl_modern', flightrl_modern.__version__)"
```

### 2. Environment Creation Test

```bash
cd /root/flightmare/flightrl_modern/examples
python3 -c "
from flightrl_modern.envs.gymnasium_wrapper import make_flight_env
env = make_flight_env(render=False, num_envs=1)
obs, info = env.reset()
print('✓ Environment created successfully')
print(f'  Observation shape: {obs.shape}')
print(f'  Action space: {env.action_space}')
env.close()
"
```

### 3. SAC Training Smoke Test

```bash
cd /root/flightmare/flightrl_modern/examples
python3 smoke_test.py
```

Expected output:
```
==============================
Flightmare RL Smoke Test
==============================

[1/5] Testing flightgym import...
  ✓ flightgym imported successfully

[2/5] Testing PyTorch import...
  ✓ PyTorch 2.0.0
    CUDA available: False (or True if GPU build)

[3/5] Testing Stable-Baselines3 import...
  ✓ Stable-Baselines3 2.0.0+

[4/5] Testing environment creation...
  ✓ Environment created successfully
    Observation shape: (...)
    Step executed successfully

[5/5] Testing SAC training (100 steps)...
  Training for 100 steps...
  ✓ SAC training test passed

==============================
✓ All smoke tests passed!
==============================
```

### 4. ROS Integration Test (if ROS enabled)

```bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash

# Check ROS environment
echo $ROS_DISTRO  # Should print: noetic

# List packages
rospack list | grep flight

# Check if flightros built successfully
rospack find flightros
```

### 5. Unit Tests

```bash
cd /root/flightmare/flightrl_modern
pytest tests/ -v
```

## Expected Package Versions

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.9.x | From deadsnakes PPA |
| NumPy | 1.24.3 | Compatible with PyTorch |
| PyTorch | 2.0.0+ | CPU or CUDA build |
| Gymnasium | 0.28.0+ | Modern gym replacement |
| Stable-Baselines3 | 2.0.0+ | RL algorithms |
| ROS | Noetic | Ubuntu 20.04 compatible |

## Performance Benchmarks

### Build Time

- **First build** (no cache): 30-60 minutes
- **Incremental build** (with cache): 5-15 minutes
- **Build with --no-rl**: 20-40 minutes

### Image Size

- **Full build** (ROS + RL): ~8-12 GB
- **No ROS build**: ~5-8 GB
- **Minimal build** (--no-ros --no-rl): ~3-5 GB

### Training Performance

SAC on QuadrotorEnv_v1 (CPU):
- ~500-1000 steps/second (single env)
- ~2000-4000 steps/second (4 parallel envs)

## Troubleshooting

### Common Issues

#### 1. "flightgym module not found"

**Problem**: flightgym not installed

**Solution**:
```bash
cd /root/flightmare/flightlib
pip3 install --verbose .
```

#### 2. CUDA not available (GPU build)

**Problem**: PyTorch not seeing GPU

**Check**:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
nvidia-smi  # Check if driver loaded
```

**Solution**: Ensure `--gpus all` flag when running container:
```bash
docker run -it --rm --gpus all flightmare:latest
```

#### 3. ROS packages not found

**Problem**: catkin workspace not sourced

**Solution**:
```bash
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
```

#### 4. Out of Memory during training

**Problem**: Insufficient RAM for buffer

**Solution**: Reduce buffer size or batch size:
```python
model = SAC(..., buffer_size=100000, batch_size=128)
```

## Test Results Template

```markdown
## Build Test Report

**Date**: YYYY-MM-DD
**Build Configuration**:
- GPU: Yes/No
- ROS: Yes/No
- RL: Yes/No
- CUDA Version: cpu/cu118/cu121

**Test Results**:
- [x] flightgym import
- [x] PyTorch import
- [x] SB3 import
- [x] Environment creation
- [x] SAC smoke test (100 steps)
- [x] Unit tests
- [x] ROS integration (if enabled)

**Build Time**: XX:XX
**Image Size**: X.X GB

**Notes**:
- All tests passed
- (Any issues or observations)
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Docker Build

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: |
          cd docker
          ./build_container.ps1 -NoRos  # Fast build for CI
      
      - name: Run verification
        run: |
          docker run --rm flightmare:latest /root/verify_installation.sh
      
      - name: Run smoke test
        run: |
          docker run --rm flightmare:latest \
            python3 /root/flightmare/flightrl_modern/examples/smoke_test.py
```

## Manual Verification Checklist

- [ ] Docker image builds without errors
- [ ] All import tests pass
- [ ] Environment can be created and reset
- [ ] Environment.step() executes without errors
- [ ] SAC training runs for 100 steps
- [ ] Model can be saved and loaded
- [ ] ROS workspace builds (if ROS enabled)
- [ ] Unity port (10253) is exposed
- [ ] Image size is reasonable (<15 GB)
- [ ] Documentation is up to date

## Acceptance Criteria

✅ **Build succeeds** - Docker build completes with exit code 0

✅ **Imports work** - All core libraries can be imported

✅ **Environment functional** - reset() and step() work correctly

✅ **Training works** - SAC can train for 100+ steps

✅ **Tests pass** - pytest runs without failures

✅ **ROS builds** - catkin build succeeds (if ROS enabled)

## Support

For issues or questions:
- Check the main README.md
- Review Docker build logs
- Inspect container: `docker run -it --rm flightmare:latest /bin/bash`
- Check GitHub issues: https://github.com/uzh-rpg/flightmare

---

**Last Updated**: 2025-11-06
**Maintained By**: Flightmare Community
