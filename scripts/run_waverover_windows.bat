@echo off
REM =============================================================================
REM run_waverover_windows.bat — 1-Click Launcher for Wave Rover on Windows / Laptop
REM =============================================================================

echo =========================================================================
echo   [ROVER] Launching Wave Rover + Foveated LiDAR Bridge (Windows Host PC)
echo   [ROVER] Microcontroller Target: Arduino Uno / ESP32 (Auto-detecting COM port)
echo =========================================================================

REM 1. Activate Python virtual environment if available
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

REM 2. Start Python Hardware Bridge
echo [1/2] Starting Python Hardware Bridge (Port 8081)...
start "WaveRover Bridge" cmd /k "python python/waverover_bridge.py"

REM 3. Start Node.js Web Dashboard HUD
echo [2/2] Starting Web Dashboard Server (Port 8080)...
start "WaveRover WebHUD" cmd /k "node web/server/websocket_bridge.js 8080"

echo =========================================================================
echo   [ROVER] System Launched Successfully!
echo   * Web Dashboard HUD: http://localhost:8080
echo   * Live Camera Stream: http://localhost:8081/video_feed
echo   * Rover Telemetry API: http://localhost:8081/api/rover/status
echo =========================================================================
pause
