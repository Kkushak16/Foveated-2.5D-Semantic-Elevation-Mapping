import os
from PIL import Image

lidar_dir = r"d:\Antigravity\Lidar Mapping"
docs_img_dir = os.path.join(lidar_dir, "docs", "images")
os.makedirs(docs_img_dir, exist_ok=True)

# List of BMP files to convert to PNG
bmp_files = [
    ("viewer_p1.bmp", "viewer_phase1.png"),
    ("dashboard_p2.bmp", "dashboard_phase2.png"),
    ("dashboard_p3.bmp", "dashboard_phase3.png"),
    ("live_p3.bmp", "live_stream.png"),
    ("dashboard_frame_001.bmp", "frame_001.png"),
    ("dashboard_frame_002.bmp", "frame_002.png"),
    ("dashboard_frame_003.bmp", "frame_003.png"),
    (os.path.join("recorded_demo", "fallback_hud.bmp"), "fallback_hud.png"),
]

for src_rel, dst_filename in bmp_files:
    src_path = os.path.join(lidar_dir, src_rel)
    dst_path = os.path.join(docs_img_dir, dst_filename)
    if os.path.exists(src_path):
        try:
            with Image.open(src_path) as img:
                img.save(dst_path, "PNG")
                print(f"Converted {src_rel} -> docs/images/{dst_filename}")
        except Exception as e:
            print(f"Error converting {src_path}: {e}")
    else:
        print(f"Source file not found: {src_path}")
