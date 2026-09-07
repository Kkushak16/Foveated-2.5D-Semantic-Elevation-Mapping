cmake_minimum_required(VERSION 3.5)
project(vehicle_detection_ros2)

# Find ament and ROS 2 packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

# Optionally enable CUDA/TensorRT linking – will be ignored on systems without them
option(USE_CUDA "Enable CUDA/TensorRT linking" ON)
if(USE_CUDA)
  find_package(CUDA REQUIRED)
  find_library(NVINFER_LIB nvinfer)
endif()

# Include directories for our core C++ code (relative to the workspace root)
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/src
)

# Source files for the node – we reuse the existing core implementation files
set(SRC_FILES
  src/foveated_vehicle_detect_node.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/src/tensorrt_engine.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/src/ros2_node.cpp
)

add_executable(vehicle_detection_node ${SRC_FILES})

# Link ROS 2 libraries
ament_target_dependencies(vehicle_detection_node
  rclcpp
  sensor_msgs
  nav_msgs
  cv_bridge
)

# Optimization flags – enable fast‑math and maximum O3 when CUDA is present
if(USE_CUDA)
  target_compile_options(vehicle_detection_node PRIVATE -O3 -use_fast_math)
endif()

# Link optional CUDA/TensorRT libraries if available
if(USE_CUDA)
  target_include_directories(vehicle_detection_node PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(vehicle_detection_node ${CUDA_LIBRARIES} ${NVINFER_LIB})
endif()

# Installation rules
install(TARGETS vehicle_detection_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

ament_package()
