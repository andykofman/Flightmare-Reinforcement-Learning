# Flightmare Docker

This directory contains everything needed to build and run Flightmare in Docker.

## Quick Start

```powershell
# Build the image (CPU, with ROS and RL)
.\docker\build_container.ps1

# Run the container
docker run -it --rm -p 10253:10253 flightmare:latest

# Inside container: verify installation
/root/verify_installation.sh

# Inside container: run smoke test
python3 /root/flightmare/flightrl_modern/examples/smoke_test.py
```

## Files

- **`Dockerfile`** - Multi-stage Docker build definition
- **`build_container.ps1`** - PowerShell build script with options
- **`verify_build.ps1`** - Verification script (run outside container)
- **`.dockerignore`** - Files excluded from build context
- **`BUILD_VERIFICATION.md`** - Detailed verification guide

## Build Options

### Basic Builds

```powershell
# Full build (CPU + ROS + RL)
.\build_container.ps1

# GPU build with CUDA 11.8
.\build_container.ps1 -Gpu -CudaVersion cu118

# Minimal build (no ROS, no RL)
.\build_container.ps1 -NoRos -NoRl
```

## Verification

```powershell
# Outside container
.\docker\verify_build.ps1

# Inside container
/root/verify_installation.sh
```

See **BUILD_VERIFICATION.md** for complete documentation.

---

**Last Updated:** 2025-11-06
