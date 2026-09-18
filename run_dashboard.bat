@echo off
REM Launcher for Unified Perception Dashboard
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    echo [LAUNCHER] Using project virtual environment (.venv)...
    ".venv\Scripts\python.exe" app.py --port 8080 --platform laptop --source local %*
) else (
    echo [LAUNCHER] .venv not found, trying py launcher...
    py app.py --port 8080 --platform laptop --source local %*
)
pause
