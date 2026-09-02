# Script to stage, commit, and push LiDAR photos to GitHub repository
$ErrorActionPreference = "Stop"

Set-Location "d:\Antigravity\Lidar Mapping"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Uploading LiDAR Project Photos to GitHub Repository" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/3] Staging all photo assets and updated README..." -ForegroundColor Yellow
git add .gitignore README.md *.bmp recorded_demo/*.bmp

Write-Host "[2/3] Committing changes..." -ForegroundColor Yellow
git commit -m "docs: include LiDAR dashboard render frames and visual gallery in repo"

Write-Host "[3/3] Pushing to remote (origin/main)..." -ForegroundColor Yellow
git push origin main

Write-Host ""
Write-Host "Successfully uploaded photos to Foveated-2.5D-Semantic-Elevation-Mapping repo!" -ForegroundColor Green
