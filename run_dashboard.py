"""
run_dashboard.py — Launch Unified LiDAR & Camera Teleop Web Dashboard
======================================================================
Launches the local HTTP/WebSocket bridge server for the merged dashboard
and automatically opens the UI in your default web browser.
"""

import os
import sys
import time
import socket
import subprocess
import webbrowser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(SCRIPT_DIR, "web", "ui")
BRIDGE_SERVER = os.path.join(SCRIPT_DIR, "web", "server", "websocket_bridge.js")

def find_available_port(start_port=8080, max_attempts=10):
    """Finds an available TCP port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    return start_port

def main():
    print("=" * 75)
    print("  LAUNCHING UNIFIED DUAL-SENSOR FOVEATED PERCEPTION DASHBOARD")
    print("=" * 75)

    port = find_available_port(8080)
    url = f"http://localhost:{port}"

    server_proc = None
    env = os.environ.copy()
    env["PORT"] = str(port)

    try:
        print(f"[INFO] Launching dashboard server on port {port}...")
        server_proc = subprocess.Popen(["node", BRIDGE_SERVER, str(port)], cwd=SCRIPT_DIR, env=env)
        time.sleep(0.8)
    except Exception as e:
        print(f"[NOTICE] Node.js fallback to Python http.server ({e})...")
        server_proc = subprocess.Popen([sys.executable, "-m", "http.server", str(port), "--directory", WEB_DIR], cwd=SCRIPT_DIR)
        time.sleep(0.8)

    print(f"\n[SUCCESS] Unified Web Dashboard is active at: {url}")
    print("[INFO] Opening dashboard in your default browser...")
    webbrowser.open(url)

    print("\nPress Ctrl+C to stop the dashboard server.\n")
    try:
        server_proc.wait()
    except KeyboardInterrupt:
        print("\n[INFO] Stopping dashboard server...")
        if server_proc:
            server_proc.terminate()

if __name__ == "__main__":
    main()
