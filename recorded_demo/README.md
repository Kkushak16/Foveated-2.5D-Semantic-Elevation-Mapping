# Pre-Rendered Recorded Demo Archive
**Foveated 2.5D LiDAR Grid Mapping Pipeline**

---

## Overview
This directory contains pre-rendered, high-resolution visual evidence of the pipeline running across sequential golden frames. These assets provide a zero-risk presentation fallback during hackathon live demos.

---

## Included Demo Assets

1. **`fallback_hud.bmp`**: Single frame full 3-ring HUD layout overview (Level 0: 5cm, Level 1: 15cm, Level 2: 50cm).
2. **`dashboard_frame_001.bmp` – `dashboard_frame_009.bmp`**: Frame-by-frame snapshot sequence showing live temporal confidence decay, ego-motion shift, and dynamic object tracking.

---

## Quick Display Instructions
To preview these rendered frames in terminal or fallback visualizer:
```powershell
.\.venv\Scripts\python.exe run.py dashboard_phase2 --save-img recorded_demo/fallback_hud.bmp
```
Or open any `.bmp` file directly in standard OS image viewers.
