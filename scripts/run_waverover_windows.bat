@echo off
REM =============================================================================
REM run_waverover_windows.bat — 1-Click Launcher for Wave Rover on Windows / Laptop
REM =============================================================================

echo =========================================================================
echo   [ROVER] Launching Wave Rover + Foveated LiDAR Bridge (Windows Host PC)
echo   [ROVER] Microcontroller Target: Arduino Uno / ESP32 (Auto-detecting COM port)
echo =========================================================================

REM Change directory to the project root
cd /d "%~dp0.."

REM 1. Identify Python executable (prefer .venv)
set "PYTHON_EXE=python"
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
)

REM 2. Start Python Hardware Bridge
echo [1/2] Starting Python Hardware Bridge (Port 8081)...
start "WaveRover Bridge" cmd /k ""%PYTHON_EXE%" python/waverover_bridge.py"

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
