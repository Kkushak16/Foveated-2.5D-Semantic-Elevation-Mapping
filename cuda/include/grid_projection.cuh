/**
 * @file grid_projection.cuh
 * @brief CUDA Kernel Headers for 3D-to-2.5D Parallel Point Cloud Projection & Grid Binning.
 * Executes on NVIDIA CUDA GPU with atomic operations for fast cell aggregation.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace lidar_mapping {
namespace cuda {

// Structure matching GPU grid cell memory
struct GPUGridCell {
    int32_t min_z_int;     // Atomic integer representation of float min_z
    int32_t max_z_int;     // Atomic integer representation of float max_z
    uint32_t point_count;  // Atomic point accumulator
    uint32_t class_histogram[4]; // Atomic class voting counter (4 classes)
};

/**
 * Launch CUDA Kernel for parallel 3D-to-2.5D point projection
 * 
 * @param d_points Device pointer to (N x 4) point cloud array (x, y, z, intensity)
 * @param d_labels Device pointer to (N) semantic class labels
 * @param num_points Total number of points in cloud (e.g. 100,000 - 1,000,000)
 * @param d_grid_cells Output device pointer to grid cells (400 x 400)
 * @param resolution_m Grid cell resolution in meters (e.g., 0.05, 0.15, 0.50)
 * @param grid_size Dimensions of 2D grid (400)
 * @param stream CUDA stream handle for asynchronous execution
 */
void launch_3d_to_2d_projection_kernel(
    const float* d_points,
    const uint8_t* d_labels,
    size_t num_points,
    GPUGridCell* d_grid_cells,
    float resolution_m,
    int grid_size,
    cudaStream_t stream = 0
);

} // namespace cuda
} // namespace lidar_mapping
