/**
 * @file ring_buffer.cpp
 * @brief Implementation of the C++17/20 Multi-Level Ring Buffer Grid Engine.
 */

#include "ring_buffer.hpp"
#include <iostream>
#include <limits>

namespace lidar_mapping {

RingBufferLayer::RingBufferLayer(int level, float range_m, float resolution_m, int grid_size)
    : level_(level), range_m_(range_m), resolution_m_(resolution_m), grid_size_(grid_size) {
    size_t total_cells = static_cast<size_t>(grid_size_ * grid_size_);
    min_z.resize(total_cells, std::numeric_limits<float>::max());
    max_z.resize(total_cells, -std::numeric_limits<float>::max());
    ground_z.resize(total_cells, 0.0f);
    z_variance.resize(total_cells, 0.0f);
    sem_class.resize(total_cells, 0);
    sem_prob.resize(total_cells, 0.0f);
    point_count.resize(total_cells, 0);
    confidence.resize(total_cells, 1.0f);
    overhang_flag.resize(total_cells, 0);
}

void RingBufferLayer::reset() {
    size_t total_cells = static_cast<size_t>(grid_size_ * grid_size_);
    std::fill(min_z.begin(), min_z.end(), std::numeric_limits<float>::max());
    std::fill(max_z.begin(), max_z.end(), -std::numeric_limits<float>::max());
    std::fill(point_count.begin(), point_count.end(), 0);
    std::fill(confidence.begin(), confidence.end(), 1.0f);
}

void RingBufferLayer::update_ego_motion(double dx_m, double dy_m) {
    origin_x_ += dx_m;
    origin_y_ += dy_m;
    
    // O(1) circular buffer index shift calculation
    int shift_x = static_cast<int>(dx_m / resolution_m_);
    int shift_y = static_cast<int>(dy_m / resolution_m_);
    
    offset_x_ = (offset_x_ + shift_x) % grid_size_;
    offset_y_ = (offset_y_ + shift_y) % grid_size_;
}

void RingBufferLayer::insert_point(float x, float y, float z, float intensity, uint8_t class_id) {
    float dist = std::sqrt(x * x + y * y);
    if (dist > range_m_) return; // Outside ring range boundary

    int gx = static_cast<int>(std::floor((x - origin_x_) / resolution_m_)) + grid_size_ / 2;
    int gy = static_cast<int>(std::floor((y - origin_y_) / resolution_m_)) + grid_size_ / 2;

    if (gx < 0 || gx >= grid_size_ || gy < 0 || gy >= grid_size_) return;

    int idx = get_index(gx, gy);

    min_z[idx] = std::min(min_z[idx], z);
    max_z[idx] = std::max(max_z[idx], z);
    point_count[idx]++;
    sem_class[idx] = class_id;
    sem_prob[idx] = 0.95f;

    // Check overhang condition: height spread > 2.0m with ground gap
    if ((max_z[idx] - min_z[idx]) > 2.0f) {
        overhang_flag[idx] = 1;
    }
}

MultiLevelRingBuffer::MultiLevelRingBuffer() {
    // Level 0: Near range (0-10m, 5cm res)
    layers_.push_back(std::make_unique<RingBufferLayer>(0, 10.0f, 0.05f, 400));
    // Level 1: Mid range (10-30m, 15cm res)
    layers_.push_back(std::make_unique<RingBufferLayer>(1, 30.0f, 0.15f, 400));
    // Level 2: Far range (30-100m, 50cm res)
    layers_.push_back(std::make_unique<RingBufferLayer>(2, 100.0f, 0.50f, 400));
}

void MultiLevelRingBuffer::process_scan(const float* points, size_t num_points, const uint8_t* labels) {
    for (size_t i = 0; i < num_points; ++i) {
        float x = points[i * 4 + 0];
        float y = points[i * 4 + 1];
        float z = points[i * 4 + 2];
        float intensity = points[i * 4 + 3];
        uint8_t cls = labels ? labels[i] : 0;

        float dist = std::sqrt(x * x + y * y);
        if (dist <= 10.0f) {
            layers_[0]->insert_point(x, y, z, intensity, cls);
        } else if (dist <= 30.0f) {
            layers_[1]->insert_point(x, y, z, intensity, cls);
        } else if (dist <= 100.0f) {
            layers_[2]->insert_point(x, y, z, intensity, cls);
        }
    }
}

void MultiLevelRingBuffer::update_vehicle_pose(double x, double y, double yaw) {
    for (auto& layer : layers_) {
        layer->update_ego_motion(x, y);
    }
}

} // namespace lidar_mapping
