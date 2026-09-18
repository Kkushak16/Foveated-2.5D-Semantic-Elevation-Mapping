"""
run_dashboard.py — Launch Unified LiDAR & Camera Teleop Web Dashboard
======================================================================
Launches the local HTTP/WebSocket bridge server for the merged dashboard,
ensures the Wave Rover Arduino hardware bridge is running with active LED OK status,
and automatically opens the UI in your default web browser.
"""

import os
import sys
import time
import socket
import subprocess
import webbrowser
import urllib.request
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(SCRIPT_DIR, "web", "ui")
BRIDGE_SERVER = os.path.join(SCRIPT_DIR, "web", "server", "websocket_bridge.js")
ROVER_BRIDGE_SCRIPT = os.path.join(SCRIPT_DIR, "python", "waverover_bridge.py")

# Locate virtualenv Python or fallback
VENV_PY = os.path.join(SCRIPT_DIR, ".venv", "Scripts", "python.exe")
PYTHON_BIN = VENV_PY if os.path.isfile(VENV_PY) else sys.executable

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

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex(('127.0.0.1', port)) == 0

def trigger_led_ok(base_url, timeout=3.0):
    """Sends an automatic check/OK command so LEDs start working immediately."""
    endpoints = [
        f"{base_url}/api/rover/ping_led",
        "http://127.0.0.1:8081/api/rover/ping_led"
    ]
    for ep in endpoints:
        try:
            req = urllib.request.Request(ep, data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode('utf-8'))
                    print(f"[SUCCESS] Arduino LEDs activated: {data.get('message', 'OK')}")
                    return True
        except Exception:
            pass
    return False

def main(port=None):
    print("=" * 75)
    print("  LAUNCHING UNIFIED DUAL-SENSOR FOVEATED PERCEPTION DASHBOARD")
    print("=" * 75)

    if port is None:
        port = find_available_port(8080)
    url = f"http://localhost:{port}"

    server_proc = None
    rover_proc = None

    env = os.environ.copy()
    env["PORT"] = str(port)
    env["PYTHON"] = PYTHON_BIN
    env["YOLO_PYTHON"] = PYTHON_BIN

    # 1. Ensure Wave Rover Hardware Bridge is active on port 8081
    if not is_port_in_use(8081) and os.path.isfile(ROVER_BRIDGE_SCRIPT):
        print(f"[INFO] Launching Wave Rover hardware bridge ({PYTHON_BIN})...")
        try:
            rover_proc = subprocess.Popen([PYTHON_BIN, ROVER_BRIDGE_SCRIPT, "--web-port", "8081"], cwd=SCRIPT_DIR, env=env)
            time.sleep(1.2)
        except Exception as e:
            print(f"[WARNING] Could not start waverover_bridge.py directly: {e}")

    # 2. Launch dashboard web server
    try:
        print(f"[INFO] Launching dashboard server on port {port}...")
        server_proc = subprocess.Popen(["node", BRIDGE_SERVER, str(port)], cwd=SCRIPT_DIR, env=env)
        time.sleep(1.0)
    except Exception as e:
        print(f"[NOTICE] Node.js fallback to Python http.server ({e})...")
        server_proc = subprocess.Popen([PYTHON_BIN, "-m", "http.server", str(port), "--directory", WEB_DIR], cwd=SCRIPT_DIR)
        time.sleep(1.0)

    # 3. Automatically initialize Arduino LEDs with OK sign
    print("[INFO] Initializing Arduino Uno Q connection and activating LEDs with OK status...")
    for _ in range(6):
        if trigger_led_ok(url, timeout=1.5):
            break
        time.sleep(0.8)

    print(f"\n[SUCCESS] Unified Web Dashboard is active at: {url}")
    print("[INFO] Opening dashboard in your default browser...")
    webbrowser.open(url)

    print("\nPress Ctrl+C to stop the dashboard server.\n")
    try:
        server_proc.wait()
    except KeyboardInterrupt:
        print("\n[INFO] Stopping dashboard servers...")
        if server_proc:
            server_proc.terminate()
        if rover_proc:
            rover_proc.terminate()

if __name__ == "__main__":
    main()
