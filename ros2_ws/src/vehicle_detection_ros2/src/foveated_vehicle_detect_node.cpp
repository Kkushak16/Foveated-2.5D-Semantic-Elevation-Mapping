#include <cuda_runtime.h>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <cv_bridge/cv_bridge.h>

#include "ring_buffer.hpp"
#include "tensorrt_engine.hpp"
#include "ros2_node.hpp"
#include "grid_projection.cuh"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

using namespace lidar_mapping;

class FoveatedVehicleDetectNode : public rclcpp::Node {
public:
  FoveatedVehicleDetectNode()
    : Node("foveated_vehicle_detect_node"),
      ring_buffer_(std::make_shared<MultiLevelRingBuffer>()),
      trt_engine_(PrecisionMode::INT8),
      ros2_layer_("vehicle_grid_node") {
    // Subscribers
    pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/lidar/points", rclcpp::QoS(10).best_effort(),
        std::bind(&FoveatedVehicleDetectNode::pointcloud_cb, this, std::placeholders::_1));
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "camera/image_raw", rclcpp::QoS(10).best_effort(),
        std::bind(&FoveatedVehicleDetectNode::image_cb, this, std::placeholders::_1));

    // Publisher for the occupancy grid (2.5D ego‑grid)
    grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
        "planning/foveated_ego_grid", rclcpp::QoS(1).best_effort());

    // Worker thread that processes queued frames
    worker_thread_ = std::thread([this] { this->process_loop(); });

    // Use a MultiThreadedExecutor for the node (enables parallel callbacks)
    executor_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
    executor_->add_node(this->shared_from_this());
    executor_thread_ = std::thread([this] { this->executor_->spin(); });

    // CUDA stream for non‑blocking kernel launches
    cudaStream_t cuda_stream_;
    cudaError_t cuda_err = cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking);
    if (cuda_err != cudaSuccess) {
      RCLCPP_ERROR(this->get_logger(), "Failed to create CUDA stream – falling back to default stream");
      cuda_stream_ = 0; // default stream
    }

  }

  ~FoveatedVehicleDetectNode() {
    shutdown_ = true;
    cv.notify_all();
    if (worker_thread_.joinable()) worker_thread_.join();
  }

private:
  // -------------------------------------------------------------------
  // Callbacks – just enqueue raw data for the background worker
  // -------------------------------------------------------------------
  void pointcloud_cb(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(queue_mtx_);
    pointcloud_queue_.push(msg);
    cv.notify_one();
  }

  void image_cb(const sensor_msgs::msg::Image::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(queue_mtx_);
    image_queue_.push(msg);
    cv.notify_one();
  }

  // -------------------------------------------------------------------
  // Main processing loop – runs in a separate thread
  // -------------------------------------------------------------------
  void process_loop() {
    while (!shutdown_) {
      std::unique_lock<std::mutex> lk(queue_mtx_);
      cv.wait(lk, [this] { return shutdown_ || !pointcloud_queue_.empty() || !image_queue_.empty(); });
      if (shutdown_) break;

      // Grab the newest point cloud (if any)
      sensor_msgs::msg::PointCloud2::SharedPtr pc_msg;
      if (!pointcloud_queue_.empty()) {
        pc_msg = pointcloud_queue_.front();
        pointcloud_queue_.pop();
      }

      // Grab the newest image (if any) – we only need it when we have a point cloud
      sensor_msgs::msg::Image::SharedPtr img_msg;
      if (!image_queue_.empty()) {
        img_msg = image_queue_.front();
        image_queue_.pop();
      }

      lk.unlock();

      if (!pc_msg) continue; // nothing to process

      // -----------------------------------------------------------------
      // 1) Convert PointCloud2 to raw float array (x,y,z,intensity)
      // -----------------------------------------------------------------
      const auto *raw_ptr = reinterpret_cast<const float*>(pc_msg->data.data());
      size_t num_points = pc_msg->width * pc_msg->height; // assuming unorganized cloud

    // Launch projection kernel (non‑blocking) – currently mock
    launch_3d_to_2d_projection_kernel(
        raw_ptr,
        pred_labels.data(),
        num_points,
        nullptr, // TODO: allocate GPU grid cells and pass pointer
        0.05f,
        400,
        cuda_stream_);


      // -----------------------------------------------------------------
      // 3) Feed points + predictions into the multi‑level ring buffer
      // -----------------------------------------------------------------
      ring_buffer_->process_scan(raw_ptr, num_points, pred_labels.data());

      // -----------------------------------------------------------------
      // 4) (Optional) Update ego‑motion – placeholder static pose here
      // -----------------------------------------------------------------
      ring_buffer_->update_vehicle_pose(0.0, 0.0, 0.0);

      // -----------------------------------------------------------------
      // 5) Publish the occupancy grid for downstream planners
      // -----------------------------------------------------------------
      for (size_t layer = 0; layer < ring_buffer_->get_layers().size(); ++layer) {
        auto &lay = ring_buffer_->get_layers()[layer];
        // For demo we just publish an empty grid – real data would be copied
        // into a nav_msgs::msg::OccupancyGrid and published.
        ros2_layer_.publish_grid_layer(static_cast<int>(layer), nullptr, nullptr, nullptr, lay->grid_size_);
      }
    }
  }

  // -------------------------------------------------------------------
  // Member variables
  // -------------------------------------------------------------------
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_pub_;

  std::shared_ptr<MultiLevelRingBuffer> ring_buffer_; // Core ring‑buffer engine
  TensorRTEngine trt_engine_;                     // TensorRT wrapper
  FoveatedGridRos2Node ros2_layer_;               // Simple publisher helper from core lib

  // Thread‑safe queues for incoming data
  std::queue<sensor_msgs::msg::PointCloud2::SharedPtr> pointcloud_queue_;
  std::queue<sensor_msgs::msg::Image::SharedPtr> image_queue_;
  std::mutex queue_mtx_;
  std::condition_variable cv;
  std::shared_ptr<rclcpp::executors::MultiThreadedExecutor> executor_;
  std::thread executor_thread_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FoveatedVehicleDetectNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
