@echo off
echo ========================================================
echo   Uploading LiDAR Project Screenshots to Repo
echo ========================================================
echo.

cd /d "d:\Antigravity\Lidar Mapping"

echo [1/3] Staging render frames, fallback HUD, and README updates...
git add .gitignore README.md *.bmp recorded_demo/*.bmp

echo [2/3] Committing changes...
git commit -m "docs: add rendered HUD dashboard frames and image gallery to README"

echo [3/3] Pushing to repository remote...
git push origin main

echo.
echo ========================================================
echo   LiDAR photos and documentation uploaded successfully!
echo ========================================================
