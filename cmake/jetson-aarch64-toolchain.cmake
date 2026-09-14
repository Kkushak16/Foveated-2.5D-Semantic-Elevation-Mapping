# cmake/jetson-aarch64-toolchain.cmake
# =====================================================================
# Cross-compilation toolchain for building the CUDA grid engine on an
# x86_64 host targeting NVIDIA Jetson aarch64 boards.
#
# Usage (from a laptop / CI server):
#   mkdir build-jetson && cd build-jetson
#   cmake .. -DCMAKE_TOOLCHAIN_FILE=cmake/jetson-aarch64-toolchain.cmake
#   make -j$(nproc)
#
# Prerequisites:
#   sudo apt install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
#
# Note: Native build directly on the Jetson board is simpler and safer
# for a hackathon timeline — use this only if the board is too slow to
# compile on (Jetson Nano can take 10+ minutes for C++ builds).
# =====================================================================

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Cross-compilers
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Sysroot / library search paths
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# For CUDA cross-compilation, the CUDA toolkit host path must point to
# the x86_64 nvcc while the device code targets the Jetson GPU arch.
# The CMakeLists.txt handles -gencode flags based on CMAKE_SYSTEM_PROCESSOR.
