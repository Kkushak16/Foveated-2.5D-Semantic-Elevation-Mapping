/**
 * @file ring_buffer.hpp
 * @brief Foveated 2.5D Multi-Level Ring Buffer (MLRB) Engine Core in Modern C++17/20.
 * 
 * Provides zero-GC, deterministic latency, struct-of-arrays memory layout for low-latency
 * vehicle ego-motion updates and O(1) circular grid shifts.
 */

#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <memory>

namespace lidar_mapping {

struct CellSoA {
    float min_z;
    float max_z;
    float ground_z;
    float z_variance;
    uint8_t sem_class;
    float sem_prob;
    uint32_t point_count;
    float confidence;
    uint8_t overhang_flag;
};

class RingBufferLayer {
public:
    RingBufferLayer(int level, float range_m, float resolution_m, int grid_size = 400);
    ~RingBufferLayer() = default;

    // Direct pointer access for fast cache locality
    void reset();
    void update_ego_motion(double dx_m, double dy_m);
    
    inline int world_to_grid_x(double x) const {
        return static_cast<int>(std::floor((x - origin_x_) / resolution_m_)) % grid_size_;
    }
    
    inline int world_to_grid_y(double y) const {
        return static_cast<int>(std::floor((y - origin_y_) / resolution_m_)) % grid_size_;
    }

    inline int get_index(int gx, int gy) const {
        int wrapped_x = (gx % grid_size_ + grid_size_) % grid_size_;
        int wrapped_y = (gy % grid_size_ + grid_size_) % grid_size_;
        return wrapped_y * grid_size_ + wrapped_x;
    }

    void insert_point(float x, float y, float z, float intensity, uint8_t sem_class);

    // Layer metadata
    int level_;
    float range_m_;
    float resolution_m_;
    int grid_size_;
    
    // Ego origin offsets
    double origin_x_{0.0};
    double origin_y_{0.0};
    int offset_x_{0};
    int offset_y_{0};

    // Struct-of-Arrays (SoA) contiguous memory layout
    std::vector<float> min_z;
    std::vector<float> max_z;
    std::vector<float> ground_z;
    std::vector<float> z_variance;
    std::vector<uint8_t> sem_class;
    std::vector<float> sem_prob;
    std::vector<uint32_t> point_count;
    std::vector<float> confidence;
    std::vector<uint8_t> overhang_flag;
};

class MultiLevelRingBuffer {
public:
    MultiLevelRingBuffer();
    void process_scan(const float* points, size_t num_points, const uint8_t* labels = nullptr);
    void update_vehicle_pose(double x, double y, double yaw);

    std::vector<std::unique_ptr<RingBufferLayer>>& get_layers() { return layers_; }

private:
    std::vector<std::unique_ptr<RingBufferLayer>> layers_;
};

} // namespace lidar_mapping
