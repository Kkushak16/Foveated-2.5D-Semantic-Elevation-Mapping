"""
run_dashboard.py — Launch LiDAR Teleop Web Dashboard Server & Open Browser
==========================================================================
Starts the local HTTP/WebSocket bridge server for the WebGL Teleop HUD
and automatically opens http://localhost:8080 in your default browser.
"""

import os
import sys
import time
import subprocess
import webbrowser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(SCRIPT_DIR, "web", "ui")
BRIDGE_SERVER = os.path.join(SCRIPT_DIR, "web", "server", "websocket_bridge.js")

PORT = 8080
URL = f"http://localhost:{PORT}"

def main():
    print("=" * 75)
    print("  🌐 LAUNCHING FOVEATED LIDAR TELEOP WEB DASHBOARD")
    print("=" * 75)

    # Try starting Node.js websocket_bridge.js first, fallback to python http.server
    server_proc = None
    try:
        print("[INFO] Attempting to start Node.js server (websocket_bridge.js)...")
        server_proc = subprocess.Popen(["node", BRIDGE_SERVER], cwd=SCRIPT_DIR)
        time.sleep(1)
    except Exception as e:
        print(f"[NOTICE] Node.js not available ({e}). Falling back to Python http.server...")
        server_proc = subprocess.Popen([sys.executable, "-m", "http.server", str(PORT), "--directory", WEB_DIR], cwd=SCRIPT_DIR)
        time.sleep(1)

    print(f"\n[SUCCESS] Web Dashboard server is running at: {URL}")
    print("[INFO] Opening dashboard in your default browser...")
    webbrowser.open(URL)

    print("\nPress Ctrl+C to stop the dashboard server.\n")
    try:
        server_proc.wait()
    except KeyboardInterrupt:
        print("\n[INFO] Stopping dashboard server...")
        server_proc.terminate()

if __name__ == "__main__":
    main()
