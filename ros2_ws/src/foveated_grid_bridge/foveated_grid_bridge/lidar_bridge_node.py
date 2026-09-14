"""
lidar_bridge_node.py — ROS 2 LiDAR → Foveated Grid Bridge
===========================================================
Minimal ROS 2 node that subscribes to a real LiDAR driver's PointCloud2
topic and feeds raw points into the existing grid-engine Python bindings.

This bridges real LiDAR hardware (Velodyne, Ouster, RPLidar, etc.) into
the foveated 2.5D grid pipeline via standard ROS 2 interfaces.

Usage (after building the ROS 2 workspace):
    ros2 run foveated_grid_bridge lidar_bridge_node
    ros2 run foveated_grid_bridge lidar_bridge_node --ros-args -p lidar_topic:=/ouster/points
"""

import sys
import time
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    import sensor_msgs_py.point_cloud2 as pc2
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("[lidar_bridge] WARNING: ROS 2 (rclpy) not installed.", file=sys.stderr)
    print("  Install ROS 2 Humble/Iron and sensor_msgs_py:", file=sys.stderr)
    print("  sudo apt install ros-humble-sensor-msgs-py", file=sys.stderr)


class LidarBridgeNode(Node):
    """ROS 2 node bridging PointCloud2 topics to the foveated grid engine."""

    def __init__(self):
        super().__init__('foveated_grid_lidar_bridge')

        # Declare parameters
        self.declare_parameter('lidar_topic', '/velodyne_points')
        self.declare_parameter('near_range_m', 10.0)
        self.declare_parameter('mid_range_m', 30.0)
        self.declare_parameter('far_range_m', 100.0)
        self.declare_parameter('log_interval_s', 5.0)

        topic = self.get_parameter('lidar_topic').value
        self.near_range = self.get_parameter('near_range_m').value
        self.mid_range = self.get_parameter('mid_range_m').value
        self.far_range = self.get_parameter('far_range_m').value
        log_interval = self.get_parameter('log_interval_s').value

        # Subscribe to LiDAR PointCloud2 topic
        self.subscription = self.create_subscription(
            PointCloud2, topic, self.on_cloud, 10)

        self.get_logger().info(
            f"Subscribed to LiDAR topic: {topic}")
        self.get_logger().info(
            f"Ring ranges: Near <{self.near_range}m, "
            f"Mid <{self.mid_range}m, Far <{self.far_range}m")

        # Stats tracking
        self._frame_count = 0
        self._total_points = 0
        self._ring_counts = [0, 0, 0]  # near, mid, far
        self._last_log_time = time.time()
        self._log_interval = log_interval

    def on_cloud(self, msg: PointCloud2):
        """Process incoming PointCloud2 message from the LiDAR driver."""
        self._frame_count += 1

        # Extract XYZ points from the PointCloud2 message
        points = list(pc2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True))

        if not points:
            return

        n_points = len(points)
        self._total_points += n_points

        # Classify points into 3-ring foveated bands
        near_pts = []
        mid_pts = []
        far_pts = []

        for x, y, z in points:
            r = np.sqrt(x * x + y * y)  # Range in XY plane (ego-centric)
            if r < self.near_range:
                near_pts.append((x, y, z))
                self._ring_counts[0] += 1
            elif r < self.mid_range:
                mid_pts.append((x, y, z))
                self._ring_counts[1] += 1
            else:
                far_pts.append((x, y, z))
                self._ring_counts[2] += 1

        # --- Feed into grid engine ---
        # TODO: Replace with actual grid engine Python bindings when available.
        # The grid engine's insertion function signature is expected to be:
        #   grid_engine.insert_points(ring_id, points_array)
        # where ring_id is 0 (near), 1 (mid), or 2 (far)
        # and points_array is an Nx3 numpy array.
        #
        # For now, we classify and count — the actual grid insertion will be
        # connected once the C++/Python bindings are built.

        # Periodic logging
        now = time.time()
        if now - self._last_log_time >= self._log_interval:
            self.get_logger().info(
                f"Frame #{self._frame_count} | "
                f"{n_points} pts | "
                f"Near: {len(near_pts)} Mid: {len(mid_pts)} Far: {len(far_pts)} | "
                f"Total processed: {self._total_points:,}")
            self._last_log_time = now


def main(args=None):
    if not ROS2_AVAILABLE:
        print("ERROR: ROS 2 (rclpy) is required for the LiDAR bridge node.",
              file=sys.stderr)
        print("Install ROS 2 Humble: https://docs.ros.org/en/humble/Installation.html",
              file=sys.stderr)
        sys.exit(1)

    rclpy.init(args=args)
    node = LidarBridgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            f"Shutting down. Processed {node._frame_count} frames, "
            f"{node._total_points:,} total points.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
