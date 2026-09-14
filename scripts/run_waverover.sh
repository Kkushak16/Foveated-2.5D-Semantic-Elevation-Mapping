#!/bin/bash
# =============================================================================
# run_waverover.sh — One-Click Launcher for Waveshare WAVE ROVER on Raspberry Pi
# =============================================================================

echo "========================================================================="
echo "  🏎️ Launching Waveshare WAVE ROVER Foveated LiDAR Perception Engine"
echo "  Hardware Target : Raspberry Pi 4 / 5 (Pure CPU Mode — No GPU Required)"
echo "  Microcontroller : Onboard ESP32 (/dev/ttyS0 @ 115200 baud)"
echo "========================================================================="

# 1. Ensure Serial Port Permissions
sudo chmod 666 /dev/ttyS0 2>/dev/null || true
sudo chmod 666 /dev/ttyUSB* 2>/dev/null || true
sudo chmod 666 /dev/video* 2>/dev/null || true

# 2. Get Local Wi-Fi IP Address
IP_ADDR=$(hostname -I | awk '{print $1}')
echo "📡 Rover Wi-Fi IP: http://${IP_ADDR}:8080"
echo "📹 Live Video Feed: http://${IP_ADDR}:8081/video_feed"
echo "========================================================================="

# 3. Launch Hardware Bridge (ESP32 Serial + Camera + Pure CPU Grid Engine)
python3 python/waverover_bridge.py &
PID_BRIDGE=$!

# 4. Launch Node.js Web Dashboard Server
node web/server/websocket_bridge.js 8080 &
PID_WEB=$!

trap "echo 'Stopping all rover processes...'; kill $PID_BRIDGE $PID_WEB; exit" INT TERM
wait
