"""
viewer_phase1.py — Raw Point Cloud Viewer (Phase 1 Deliverable)
================================================================
Part of: Foveated 2.5D LiDAR Grid Mapping for Autonomous Vehicle Perception
Member C · Phase 1

Interactive / headless viewer for raw 3D LiDAR point clouds (SemanticKITTI or synthetic).
Supports:
  - Open3D 3D interactive viewer with custom color maps (Height, Intensity, Ground Truth)
  - Matplotlib 3D projection renderer
  - Built-in pure NumPy BMP image exporter fallback (no dependencies required)

Usage:
    # Synthetic scene:
    python run.py viewer_phase1 --synthetic

    # With SemanticKITTI dataset:
    python run.py viewer_phase1 /path/to/dataset --seq 00 --frame 000000

    # Save visualization frame to image (headless):
    python run.py viewer_phase1 --synthetic --save-img viewer_frame.bmp
"""

import argparse
import logging
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np

# Local imports
from dataset_loader import SemanticKITTILoader, GROUND_LABEL_IDS

logger = logging.getLogger(__name__)


def save_rgb_bmp(filename: str, rgb_array: np.ndarray) -> None:
    """Save an RGB array [H, W, 3] as a 24-bit uncompressed BMP using standard library."""
    if not filename.endswith('.bmp'):
        base, _ = os.path.splitext(filename)
        filename = base + '.bmp'

    if rgb_array.dtype != np.uint8:
        rgb_array = np.clip(rgb_array * 255.0, 0, 255).astype(np.uint8)

    h, w, _ = rgb_array.shape
    bgr = rgb_array[::-1, :, ::-1]  # BMP is bottom-to-top BGR
    row_padding = (4 - (w * 3) % 4) % 4
    pixel_bytes = bytearray()
    pad = b'\x00' * row_padding

    for row in bgr:
        pixel_bytes.extend(row.tobytes())
        pixel_bytes.extend(pad)

    file_size = 54 + len(pixel_bytes)
    header = bytearray([
        0x42, 0x4D,
        file_size & 0xFF, (file_size >> 8) & 0xFF, (file_size >> 16) & 0xFF, (file_size >> 24) & 0xFF,
        0, 0, 0, 0,
        54, 0, 0, 0,
        40, 0, 0, 0,
        w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF, (w >> 24) & 0xFF,
        h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF, (h >> 24) & 0xFF,
        1, 0,
        24, 0,
        0, 0, 0, 0,
        len(pixel_bytes) & 0xFF, (len(pixel_bytes) >> 8) & 0xFF, (len(pixel_bytes) >> 16) & 0xFF, (len(pixel_bytes) >> 24) & 0xFF,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ])

    with open(filename, 'wb') as f:
        f.write(header)
        f.write(pixel_bytes)

    logger.info(f"Saved pure BMP image to {filename}")


# Fallback synthetic scene generator
def generate_demo_point_cloud(n_points: int = 80_000, sensor_height: float = 1.73) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a realistic synthetic 3D LiDAR point cloud."""
    rng = np.random.default_rng(42)
    ground_z = -sensor_height

    # 1. Road surface (ground_id = 40)
    n_road = int(n_points * 0.50)
    rx = rng.uniform(-50, 50, n_road)
    ry = rng.uniform(-50, 50, n_road)
    rz = ground_z + rng.normal(0, 0.03, n_road)
    ri = rng.uniform(0.1, 0.4, n_road)
    road_pts = np.column_stack([rx, ry, rz, ri])
    road_lbl = np.full(n_road, 40, dtype=np.uint32)

    # 2. Sidewalk (ground_id = 48)
    n_side = int(n_points * 0.10)
    sx = rng.uniform(-30, 30, n_side)
    sy = rng.uniform(6, 12, n_side)
    sz = ground_z + 0.15 + rng.normal(0, 0.02, n_side)
    si = rng.uniform(0.2, 0.5, n_side)
    side_pts = np.column_stack([sx, sy, sz, si])
    side_lbl = np.full(n_side, 48, dtype=np.uint32)

    # 3. Vehicles (car_id = 10)
    n_cars = int(n_points * 0.20)
    cx = rng.uniform(-25, 25, n_cars)
    cy = rng.uniform(-25, 25, n_cars)
    cz = rng.uniform(ground_z + 0.2, ground_z + 1.8, n_cars)
    ci = rng.uniform(0.5, 0.9, n_cars)
    car_pts = np.column_stack([cx, cy, cz, ci])
    car_lbl = np.full(n_cars, 10, dtype=np.uint32)

    # 4. Vegetation / Poles / Buildings (id = 70)
    n_veg = n_points - n_road - n_side - n_cars
    vx = rng.uniform(-40, 40, n_veg)
    vy = rng.uniform(-40, 40, n_veg)
    vz = rng.uniform(ground_z + 1.0, ground_z + 6.0, n_veg)
    vi = rng.uniform(0.1, 0.6, n_veg)
    veg_pts = np.column_stack([vx, vy, vz, vi])
    veg_lbl = np.full(n_veg, 70, dtype=np.uint32)

    points = np.vstack([road_pts, side_pts, car_pts, veg_pts]).astype(np.float32)
    labels = np.concatenate([road_lbl, side_lbl, car_lbl, veg_lbl])

    order = rng.permutation(points.shape[0])
    return points[order], labels[order]


def get_point_colors(
    points: np.ndarray,
    labels: Optional[np.ndarray] = None,
    mode: str = "height"
) -> np.ndarray:
    """Generate RGB colors for point cloud based on visualization mode."""
    N = points.shape[0]
    colors = np.zeros((N, 3), dtype=np.float64)

    if mode == "semantic" and labels is not None:
        ground_mask = np.isin(labels, list(GROUND_LABEL_IDS))
        veh_mask = np.isin(labels, [10, 11, 13, 15, 18, 20, 30, 31, 32])
        
        colors[ground_mask] = [0.15, 0.75, 0.25]  # Green
        colors[veh_mask] = [0.90, 0.20, 0.15]     # Red
        colors[~ground_mask & ~veh_mask] = [0.6, 0.6, 0.7] # Gray
        return colors

    z = points[:, 2]
    z_min, z_max = -2.0, 4.0
    norm_z = np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)
    
    colors[:, 0] = np.clip(2.0 * norm_z - 0.5, 0.0, 1.0)
    colors[:, 1] = np.clip(1.5 - np.abs(3.0 * norm_z - 1.5), 0.0, 1.0)
    colors[:, 2] = np.clip(1.5 - 2.0 * norm_z, 0.0, 1.0)
    return colors


def render_open3d(
    points: np.ndarray,
    labels: Optional[np.ndarray] = None,
    color_mode: str = "height",
    save_path: Optional[str] = None
) -> bool:
    """Render 3D Point Cloud using Open3D."""
    try:
        import open3d as o3d
    except ImportError:
        return False

    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(get_point_colors(points, labels, color_mode))

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Foveated LiDAR — Phase 1 Point Cloud Viewer", width=1280, height=720, visible=(save_path is None))
        vis.add_geometry(pcd)

        opt = vis.get_render_option()
        opt.point_size = 1.5
        opt.background_color = np.array([0.05, 0.05, 0.08])

        ctr = vis.get_view_control()
        ctr.set_zoom(0.35)
        ctr.set_front([0.5, -0.5, -0.7])
        ctr.set_up([0, 0, 1])

        if save_path:
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(save_path)
            logger.info(f"Saved visualization to {save_path}")
            vis.destroy_window()
            return True

        print("\n[Open3D Viewer] Press [Q] or close the window to exit.")
        vis.run()
        vis.destroy_window()
        return True
    except Exception as e:
        logger.warning(f"Open3D window creation failed: {e}")
        return False


def render_matplotlib(
    points: np.ndarray,
    labels: Optional[np.ndarray] = None,
    color_mode: str = "height",
    save_path: Optional[str] = None
) -> bool:
    """Fallback 3D point cloud renderer using Matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig = plt.figure(figsize=(12, 8), facecolor='#0d1117')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0d1117')

    if points.shape[0] > 15_000:
        indices = np.random.choice(points.shape[0], size=15_000, replace=False)
        pts_sub = points[indices]
        lbl_sub = labels[indices] if labels is not None else None
    else:
        pts_sub = points
        lbl_sub = labels

    cols = get_point_colors(pts_sub, lbl_sub, color_mode)

    ax.scatter(pts_sub[:, 0], pts_sub[:, 1], pts_sub[:, 2], c=cols, s=1.2, alpha=0.8)
    ax.set_title("Foveated LiDAR — Phase 1 Point Cloud Viewer", color='white', fontsize=14, pad=12)
    ax.set_xlabel("X (m)", color='white')
    ax.set_ylabel("Y (m)", color='white')
    ax.set_zlabel("Z (m)", color='white')
    
    ax.tick_params(colors='white')
    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-3, 5)

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='#0d1117')
        logger.info(f"Saved Matplotlib frame to {save_path}")
        plt.close()
    else:
        print("\n[Matplotlib Viewer] Displaying 3D point cloud window...")
        plt.show()

    return True


def render_pure_bmp(
    points: np.ndarray,
    labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
) -> None:
    """Pure NumPy 2D Projection BMP generator (Zero Dependencies)."""
    grid_sz = 600
    canvas = np.full((grid_sz, grid_sz, 3), [12, 15, 22], dtype=np.uint8)
    
    # Render top-down projection onto 600x600 canvas [-50m, 50m]
    x_px = np.clip(((points[:, 0] + 50.0) / 100.0 * grid_sz).astype(np.int32), 0, grid_sz - 1)
    y_px = np.clip(((points[:, 1] + 50.0) / 100.0 * grid_sz).astype(np.int32), 0, grid_sz - 1)
    
    colors = (get_point_colors(points, labels, "semantic") * 255).astype(np.uint8)
    canvas[y_px, x_px] = colors

    out_file = save_path or "viewer_p1_output.bmp"
    save_rgb_bmp(out_file, canvas)
    print(f"  [Fallback BMP Renderer] Rendered {points.shape[0]:,} points to {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 Raw Point Cloud Viewer")
    parser.add_argument("dataset_root", nargs="?", default=None, help="Path to SemanticKITTI dataset root")
    parser.add_argument("--seq", default="00", help="Sequence ID (default: 00)")
    parser.add_argument("--frame", default="000000", help="Frame ID (default: 000000)")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic scene")
    parser.add_argument("--color-mode", choices=["height", "intensity", "semantic"], default="semantic", help="Color coding mode")
    parser.add_argument("--save-img", default=None, help="Path to save rendered image")
    parser.add_argument("--mpl-fallback", action="store_true", help="Force Matplotlib renderer fallback")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if args.synthetic or args.dataset_root is None:
        print("[Phase 1] Generating synthetic point cloud frame...")
        points, labels = generate_demo_point_cloud()
    else:
        print(f"[Phase 1] Loading SemanticKITTI seq {args.seq} / frame {args.frame}...")
        loader = SemanticKITTILoader(args.dataset_root)
        points, labels = loader.load_frame(args.seq, args.frame)

    print(f"Point cloud loaded: {points.shape[0]:,} 3D points.")

    success = False
    if not args.mpl_fallback:
        success = render_open3d(points, labels, args.color_mode, args.save_img)

    if not success:
        success = render_matplotlib(points, labels, args.color_mode, args.save_img)

    if not success:
        render_pure_bmp(points, labels, args.save_img)

    print("[SUCCESS] Phase 1 point cloud viewer execution completed.")


if __name__ == "__main__":
    main()
