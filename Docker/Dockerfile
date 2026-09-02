# Dockerfile for Foveated 2.5D LiDAR Multi-Language Pipeline Environment
FROM nvcr.io/nvidia/l4t-tensorrt:r35.2.1-runtime

# Install system dependencies (C++20 GCC, CMake, Eigen, PCL, ROS 2 Humble)
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    g++ \
    git \
    python3-pip \
    python3-dev \
    libeigen3-dev \
    libpcl-dev \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app/lidar_mapping

# Copy project source files
COPY . .

# Install Python requirements
RUN pip3 install --no-cache-dir torch numpy onnxruntime

# Build C++/CUDA Native Engine
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCH=87 && \
    make -j$(nproc)

# Expose WebGL Teleop Dashboard Port
EXPOSE 8080

# Default entrypoint runs full system validation loop
CMD ["bash", "build.sh"]
