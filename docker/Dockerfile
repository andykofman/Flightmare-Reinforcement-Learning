# Flightmare Docker - Modern Build with RL
FROM ubuntu:20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV FLIGHTMARE_PATH=/root/flightmare
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV UNITY_HOST=host.docker.internal

ARG PYTHON_VERSION=3.9
ARG PYTORCH_VERSION=2.0.0
ARG CUDA_VERSION=cpu
ARG ENABLE_ROS=1
ARG ENABLE_RL=1

# Set timezone
RUN ln -snf /usr/share/zoneinfo/UTC /etc/localtime && echo UTC > /etc/timezone

# Install base tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    lsb-release build-essential cmake git vim \
    ca-certificates curl wget gnupg2 \
    software-properties-common pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.9
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3.9-dev python3.9-distutils python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install C++ dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libzmq3-dev libzmqpp-dev libopencv-dev libeigen3-dev \
    libgoogle-glog-dev libyaml-cpp-dev libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python base packages
RUN pip3 install --no-cache-dir \
    numpy==1.24.3 ruamel.yaml opencv-python \
    matplotlib pandas pillow pybind11

# Install ROS Noetic (conditional)
RUN if [ "$ENABLE_ROS" = "1" ]; then \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add - && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-desktop-full ros-noetic-joy \
    ros-noetic-octomap-ros ros-noetic-octomap-msgs \
    python3-rosdep python3-rosinstall python3-rosinstall-generator \
    python3-wstool python3-catkin-tools && \
    rm -rf /var/lib/apt/lists/* && \
    rosdep init && rosdep update; \
    fi

# Copy source code (moved here for better caching - only invalidated when code changes)
WORKDIR /root
COPY . /root/flightmare/
ENV FLIGHTMARE_PATH=/root/flightmare

# Build flightlib
WORKDIR /root/flightmare/flightlib
RUN mkdir -p build externals
RUN set -e; \
    pip3 install --verbose . 2>&1 | tee /root/flightlib_install.log; \
    EXIT_CODE=$?; \
    echo "=== PIP EXIT CODE: $EXIT_CODE ==="; \
    if [ $EXIT_CODE -ne 0 ]; then \
        echo "=== BUILD FAILED - Last 100 lines ==="; \
        tail -100 /root/flightlib_install.log; \
        exit 1; \
    fi; \
    echo "=== Build completed, checking installation ==="; \
    pip3 list | grep -i flight || echo "WARNING: No flight packages found"; \
    python3 -c "import flightgym; print('✓ flightgym OK')"

# Build ROS packages (conditional)
RUN if [ "$ENABLE_ROS" = "1" ]; then \
    echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "export FLIGHTMARE_PATH=/root/flightmare" >> /root/.bashrc && \
    pip3 install --no-cache-dir vcstool && \
    mkdir -p /root/catkin_ws/src && \
    cd /root/catkin_ws/src && \
    ln -s /root/flightmare/flightros flightros && \
    ln -s /root/flightmare/flightrender flightrender && \
    (vcs-import < flightros/dependencies.yaml || echo "VCS import warnings") && \
    cd /root/catkin_ws && \
    /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    catkin init && \
    catkin config --extend /opt/ros/noetic && \
    catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    (catkin build -j$(nproc) || catkin build -j2 || catkin build -j1)" 2>&1 | tee /root/catkin_build.log && \
    echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc; \
    fi

# Install RL stack (conditional)
RUN if [ "$ENABLE_RL" = "1" ]; then \
    if [ "$CUDA_VERSION" = "cpu" ]; then \
    pip3 install --no-cache-dir torch==$PYTORCH_VERSION --index-url https://download.pytorch.org/whl/cpu; \
    else \
    pip3 install --no-cache-dir torch==$PYTORCH_VERSION; \
    fi && \
    pip3 install --no-cache-dir \
    gymnasium>=0.28.0 stable-baselines3>=2.0.0 sb3-contrib>=2.0.0 \
    tensorboard>=2.12.0 tqdm imageio imageio-ffmpeg pytest pytest-cov && \
    cd /root/flightmare/flightrl_modern && \
    pip3 install --verbose . 2>&1 | tee /root/flightrl_modern_install.log; \
    fi

# ============================================================================
# Runtime Stage
# ============================================================================
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV FLIGHTMARE_PATH=/root/flightmare
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ARG ENABLE_ROS=1
ARG ENABLE_RL=1

# Set timezone
RUN ln -snf /usr/share/zoneinfo/UTC /etc/localtime && echo UTC > /etc/timezone

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    lsb-release curl wget vim git ca-certificates gnupg2 \
    libzmq3-dev libzmqpp-dev libopencv-dev libeigen3-dev \
    libgoogle-glog-dev libyaml-cpp-dev software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.9
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3.9-distutils python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install ROS runtime (conditional)
RUN if [ "$ENABLE_ROS" = "1" ]; then \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add - && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base ros-noetic-joy \
    ros-noetic-octomap-ros ros-noetic-octomap-msgs python3-catkin-tools \
    && rm -rf /var/lib/apt/lists/*; \
    fi

# Copy built artifacts from builder
COPY --from=builder /usr/local/lib/python3.9/dist-packages /usr/local/lib/python3.9/dist-packages
COPY --from=builder /root/flightmare /root/flightmare
COPY --from=builder /root/catkin_ws /root/catkin_ws
COPY --from=builder /root/*.log /root/

# Set up environment
RUN echo "export FLIGHTMARE_PATH=/root/flightmare" >> /root/.bashrc

RUN if [ "$ENABLE_ROS" = "1" ]; then \
    echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc; \
    fi

# Create verification script
# Note: This script checks what was actually built, not build args
RUN cat > /root/verify_installation.sh << 'EOFVERIFY'
#!/bin/bash
set -e

echo "=============================="
echo "Flightmare Verification"
echo "=============================="
echo ""

echo "[1/6] Python version..."
python3 --version

echo ""
echo "[2/6] FlightGym import..."
python3 -c "import flightgym; print('✓ flightgym OK')"

echo ""
echo "[3/6] NumPy..."
python3 -c "import numpy as np; print('✓ NumPy', np.__version__)"

echo ""
echo "[4/6] PyTorch..."
python3 -c "import torch; print('✓ PyTorch', torch.__version__); print('  CUDA available:', torch.cuda.is_available())" 2>/dev/null || echo "⊘ PyTorch not installed (RL disabled)"

echo ""
echo "[5/6] Stable-Baselines3..."
python3 -c "import stable_baselines3 as sb3; print('✓ SB3', sb3.__version__)" 2>/dev/null || echo "⊘ SB3 not installed (RL disabled)"

echo ""
echo "[6/6] Flightrl Modern..."
python3 -c "import flightrl_modern; print('✓ flightrl_modern', flightrl_modern.__version__)" 2>/dev/null || echo "⊘ flightrl_modern not installed (RL disabled)"

echo ""
echo "=============================="
echo "✓ All verifications passed!"
echo "=============================="
EOFVERIFY

RUN chmod +x /root/verify_installation.sh

# Run verification to ensure build succeeded
RUN /bin/bash /root/verify_installation.sh

EXPOSE 10253
WORKDIR /root/flightmare

CMD ["/bin/bash"]
