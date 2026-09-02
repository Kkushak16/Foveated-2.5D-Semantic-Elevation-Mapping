/**
 * @file ros2_node.hpp
 * @brief ROS 2 Middleware Integration Node (Modern C++ rclcpp).
 * Publishes 2.5D foveated ego-grid state to motion planning and vehicle controls topics.
 */

#pragma once

#include <string>
#include <memory>
#include <iostream>

namespace lidar_mapping {

class FoveatedGridRos2Node {
public:
    FoveatedGridRos2Node(const std::string& node_name = "foveated_grid_node");
    ~FoveatedGridRos2Node() = default;

    void spin_once();
    void publish_grid_layer(int level, const float* min_z, const float* max_z, const uint8_t* sem_class, int grid_size);

private:
    std::string node_name_;
    uint64_t frame_count_{0};
};

} // namespace lidar_mapping
