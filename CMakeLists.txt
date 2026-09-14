cmake_minimum_required(VERSION 3.18)
project(foveated_lidar_mapping LANGUAGES CXX C)

# C++ Standard Requirements
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimized Compiler Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -pthread")

# Include Directories
include_directories(
    cpp/include
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
# 2. CUDA Parallel Grid Projection Kernel Library (Optional / Conditional)
# -----------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)

    # Multi-platform CUDA architecture support:
    # - Jetson Orin (Nano/NX/AGX):  compute_87 (Ampere iGPU)
    # - Jetson Xavier (NX/AGX):     compute_72 (Volta iGPU)
    # - Cloud T4 (Colab):           compute_75 (Turing)
    # - Cloud L4:                   compute_89 (Ada Lovelace)
    # Auto-detect: if building on/for aarch64 (Jetson), use Jetson archs;
    # otherwise assume cloud/desktop GPU.
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(CUDA_ARCH_FLAGS "-gencode arch=compute_72,code=sm_72 -gencode arch=compute_87,code=sm_87")
        message(STATUS "CUDA: Targeting Jetson architectures (Xavier sm_72, Orin sm_87)")
    else()
        set(CUDA_ARCH_FLAGS "-gencode arch=compute_75,code=sm_75 -gencode arch=compute_87,code=sm_87 -gencode arch=compute_89,code=sm_89")
        message(STATUS "CUDA: Targeting cloud/desktop architectures (T4 sm_75, Orin sm_87, L4 sm_89)")
    endif()

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math ${CUDA_ARCH_FLAGS}")
    include_directories(cuda/include)

    add_library(cuda_grid_projection STATIC
        cuda/src/grid_projection.cu
    )
    set_target_properties(cuda_grid_projection PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_compile_definitions(ENABLE_CUDA=1)
    message(STATUS "CUDA Compiler detected: ${CMAKE_CUDA_COMPILER}. GPU acceleration enabled.")
else()
    message(STATUS "CUDA Compiler (nvcc) not found. Building with CPU fallback.")
    add_compile_definitions(ENABLE_CUDA=0)
endif()

# -----------------------------------------------------------------------------
# 3. Main Executable Target
# -----------------------------------------------------------------------------
add_executable(foveated_lidar_onboard_node
    cpp/src/main.cpp
)

target_link_libraries(foveated_lidar_onboard_node PRIVATE
    ring_buffer_core
)

if(CMAKE_CUDA_COMPILER)
    target_link_libraries(foveated_lidar_onboard_node PRIVATE
        cuda_grid_projection
    )
endif()

# -----------------------------------------------------------------------------
# 4. Testing (Enables ctest for CI/CD)
# -----------------------------------------------------------------------------
enable_testing()
add_test(NAME test_foveated_onboard_node COMMAND foveated_lidar_onboard_node)

message(STATUS "Configured Multi-Language Architecture (Python -> C++ -> CUDA[optional] -> TensorRT -> ROS 2 -> Web)")
