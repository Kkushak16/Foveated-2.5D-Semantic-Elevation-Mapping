@echo off
REM Launcher for Waveshare Wave Rover Hardware Bridge
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    echo [LAUNCHER] Using project virtual environment (.venv)...
    ".venv\Scripts\python.exe" "python\waverover_bridge.py" %*
) else (
    echo [LAUNCHER] .venv not found, trying py launcher...
    py "python\waverover_bridge.py" %*
)
pause
