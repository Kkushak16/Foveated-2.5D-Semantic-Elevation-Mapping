/**
 * @file grid_projection.cu
 * @brief CUDA Kernel Implementation for 3D-to-2.5D Parallel Elevation Projection.
 */

#include "grid_projection.cuh"
#include <device_launch_parameters.h>
#include <stdio.h>

namespace lidar_mapping {
namespace cuda {

// Helper CUDA device function: convert float to int order-preserving for atomic min/max
__device__ __forceinline__ int32_t float_to_ordered_int(float val) {
    int32_t i = __float_as_int(val);
    return (i >= 0) ? i : (i ^ 0x7FFFFFFF);
}

__device__ __forceinline__ float ordered_int_to_float(int32_t val) {
    int32_t i = (val >= 0) ? val : (val ^ 0x7FFFFFFF);
    return __int_as_float(i);
}

// -----------------------------------------------------------------------------
// CUDA KERNEL: 3D-to-2.5D Parallel Projection with Atomic Binning
// -----------------------------------------------------------------------------
__global__ void projection_kernel(
    const float* __restrict__ points,
    const uint8_t* __restrict__ labels,
    size_t num_points,
    GPUGridCell* __restrict__ grid,
    float resolution_m,
    int grid_size
) {
    // Global 1D thread ID across grid/blocks
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;

    // Read point coordinates (x, y, z, intensity)
    float x = points[idx * 4 + 0];
    float y = points[idx * 4 + 1];
    float z = points[idx * 4 + 2];
    uint8_t sem_class = labels ? labels[idx] : 0;

    // Calculate grid cell indices relative to origin (center of grid)
    int gx = __float2int_rd(x / resolution_m) + grid_size / 2;
    int gy = __float2int_rd(y / resolution_m) + grid_size / 2;

    // Boundary check
    if (gx < 0 || gx >= grid_size || gy < 0 || gy >= grid_size) return;

    int cell_idx = gy * grid_size + gx;
    GPUGridCell* cell = &grid[cell_idx];

    // Atomic height bounds calculation
    int32_t z_int = float_to_ordered_int(z);
    atomicMin(&cell->min_z_int, z_int);
    atomicMax(&cell->max_z_int, z_int);

    // Atomic point count accumulation
    atomicAdd(&cell->point_count, 1);

    // Atomic class voting
    if (sem_class < 4) {
        atomicAdd(&cell->class_histogram[sem_class], 1);
    }
}

// Host launcher function
void launch_3d_to_2d_projection_kernel(
    const float* d_points,
    const uint8_t* d_labels,
    size_t num_points,
    GPUGridCell* d_grid_cells,
    float resolution_m,
    int grid_size,
    cudaStream_t stream
) {
    int threads_per_block = 256;
    int blocks_per_grid = (static_cast<int>(num_points) + threads_per_block - 1) / threads_per_block;

    projection_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_points, d_labels, num_points, d_grid_cells, resolution_m, grid_size
    );
}

} // namespace cuda
} // namespace lidar_mapping
