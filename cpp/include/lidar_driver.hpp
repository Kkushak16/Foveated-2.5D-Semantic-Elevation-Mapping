/**
 * @file lidar_driver.hpp
 * @brief High-throughput C++ LiDAR Driver & Shared Memory Pinned Buffer.
 * Handles UDP socket ingress of sensor packets and stage passing into GPU memory.
 */

#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <memory>

namespace lidar_mapping {

struct LidarPacket {
    double timestamp;
    uint32_t packet_id;
    std::vector<float> points_xyzi; // x, y, z, intensity per point
};

class LidarDriver {
public:
    LidarDriver(const std::string& ip = "192.168.1.201", int port = 2368);
    ~LidarDriver();

    bool initialize();
    bool receive_packet(LidarPacket& packet);
    size_t get_buffer_size() const { return buffer_capacity_; }

private:
    std::string sensor_ip_;
    int port_;
    bool is_connected_{false};
    size_t buffer_capacity_{100000 * 4}; // 100k points in float4 format
    float* pinned_gpu_buffer_{nullptr};
};

} // namespace lidar_mapping
