cmake_minimum_required(VERSION 3.18)
project(foveated_lidar_mapping LANGUAGES CXX C CUDA)

# C++ Standard Requirements
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Optimized Native Compiler Flags for Embedded Automotive Linux (NVIDIA Orin/Xavier)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra -pthread")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -gencode arch=compute_87,code=sm_87")

# Include Directories
include_directories(
    cpp/include
    cuda/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# -----------------------------------------------------------------------------
# 1. C++ Ring Buffer Engine Library
# -----------------------------------------------------------------------------
add_library(ring_buffer_core STATIC
    cpp/src/ring_buffer.cpp
    cpp/src/lidar_driver.cpp
    cpp/src/tensorrt_engine.cpp
    cpp/src/ros2_node.cpp
)

# -----------------------------------------------------------------------------
# 2. CUDA Parallel Grid Projection Kernel Library
# -----------------------------------------------------------------------------
add_library(cuda_grid_projection STATIC
    cuda/src/grid_projection.cu
)
set_target_properties(cuda_grid_projection PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# -----------------------------------------------------------------------------
# 3. Main Executable Target
# -----------------------------------------------------------------------------
add_executable(foveated_lidar_onboard_node
    cpp/src/main.cpp
)

target_link_libraries(foveated_lidar_onboard_node PRIVATE
    ring_buffer_core
    cuda_grid_projection
)

message(STATUS "Configured Multi-Language Architecture (Python -> C++ -> CUDA -> TensorRT -> ROS 2 -> Web)")
