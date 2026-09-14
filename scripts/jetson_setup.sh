#!/bin/bash
# =====================================================================
# jetson_setup.sh — JetPack Environment Setup for Foveated 2.5D Pipeline
# =====================================================================
# Run this script on a freshly flashed Jetson board (Orin Nano, Xavier NX,
# or AGX Orin) to set up the complete development environment.
#
# Prerequisites:
#   - JetPack 5.1.x or 6.0+ flashed via NVIDIA SDK Manager or SD card image
#   - Internet connection for package downloads
#
# Usage:
#   chmod +x scripts/jetson_setup.sh
#   ./scripts/jetson_setup.sh
# =====================================================================

set -e

echo "========================================================================="
echo "  🚀 Jetson Environment Setup — Foveated 2.5D Perception Pipeline"
echo "========================================================================="

# -------------------------------------------------------------------------
# 1. Verify JetPack / L4T version
# -------------------------------------------------------------------------
echo ""
echo "[STEP 1/6] Verifying JetPack / L4T installation..."
echo "---------------------------------------------------"

if command -v dpkg &>/dev/null; then
    echo "Installed JetPack components:"
    dpkg -l | grep nvidia-jetpack || echo "  (nvidia-jetpack meta-package not found — checking individual components)"
    dpkg -l | grep -i cuda || echo "  CUDA: not found"
    dpkg -l | grep -i cudnn || echo "  cuDNN: not found"
    dpkg -l | grep -i tensorrt || echo "  TensorRT: not found"
fi

if command -v nvcc &>/dev/null; then
    echo ""
    echo "CUDA Compiler:"
    nvcc --version
else
    echo "⚠️  nvcc not found — CUDA may not be installed correctly"
    echo "   Re-flash with JetPack using NVIDIA SDK Manager"
fi

echo ""
echo "L4T Version:"
cat /etc/nv_tegra_release 2>/dev/null || echo "  Could not read /etc/nv_tegra_release"

# -------------------------------------------------------------------------
# JetPack / CUDA Version Compatibility Matrix (reference)
# -------------------------------------------------------------------------
# | JetPack | L4T    | CUDA  | cuDNN | TensorRT | Jetson Models           |
# |---------|--------|-------|-------|----------|-------------------------|
# | 5.1.2   | 35.4.1 | 11.4  | 8.6   | 8.5      | Orin Nano/NX, AGX Orin  |
# | 5.1.3   | 35.5.0 | 11.4  | 8.6   | 8.5      | Orin Nano/NX, AGX Orin  |
# | 6.0     | 36.3   | 12.2  | 8.9   | 8.6      | Orin Nano/NX, AGX Orin  |
# | 4.6.3   | 32.7.3 | 10.2  | 8.2   | 8.4      | Xavier NX, AGX Xavier   |
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# 2. System update and base tooling
# -------------------------------------------------------------------------
echo ""
echo "[STEP 2/6] Updating system and installing base tools..."
echo "--------------------------------------------------------"
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    python3-pip \
    cmake \
    git \
    libopencv-dev \
    python3-opencv \
    build-essential \
    curl \
    wget \
    htop

# -------------------------------------------------------------------------
# 3. Install jetson-stats (jtop) for on-device telemetry
# -------------------------------------------------------------------------
echo ""
echo "[STEP 3/6] Installing jetson-stats (jtop) for telemetry..."
echo "-----------------------------------------------------------"
sudo pip3 install -U jetson-stats
echo "✅ jetson-stats installed. Run 'jtop' to verify."
echo "   The telemetry module (python/telemetry.py) uses jtop for:"
echo "   - GPU utilization, Power draw (W), SoC temperature (°C)"
echo "   - These replace the pynvml readings used in the cloud path"

# -------------------------------------------------------------------------
# 4. Clone repository and install Python dependencies
# -------------------------------------------------------------------------
echo ""
echo "[STEP 4/6] Cloning repository and installing dependencies..."
echo "-------------------------------------------------------------"

REPO_URL="https://github.com/Kkushak16/Foveated-2.5D-Semantic-Elevation-Mapping.git"
REPO_DIR="Foveated-2.5D-Semantic-Elevation-Mapping"

if [ ! -d "$REPO_DIR" ]; then
    git clone "$REPO_URL"
else
    echo "Repository already exists. Pulling latest..."
    cd "$REPO_DIR" && git pull && cd ..
fi

cd "$REPO_DIR"
pip3 install -r requirements.txt

# -------------------------------------------------------------------------
# 5. Build C++/CUDA grid engine natively on the board
# -------------------------------------------------------------------------
echo ""
echo "[STEP 5/6] Building C++/CUDA grid engine (native aarch64)..."
echo "--------------------------------------------------------------"
echo "This may take several minutes on Jetson Nano. Be patient."

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_PROCESSOR=aarch64
make -j$(nproc)
cd ..

echo "✅ C++/CUDA grid engine built successfully."

# -------------------------------------------------------------------------
# 6. Validate setup
# -------------------------------------------------------------------------
echo ""
echo "[STEP 6/6] Validating installation..."
echo "---------------------------------------"

# Test telemetry module
python3 python/telemetry.py --selftest

# Test camera access
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('✅ USB camera detected and accessible')
    cap.release()
else:
    print('⚠️  No USB camera detected (device 0). Connect a camera and retry.')
"

echo ""
echo "========================================================================="
echo "  ✅ JETSON SETUP COMPLETE"
echo "========================================================================="
echo ""
echo "  Quick start commands:"
echo "  ---------------------"
echo "  # USB webcam perception:"
echo "  python3 python/jetson_live_processor.py --camera-type usb --device 0"
echo ""
echo "  # CSI camera (RPi Camera v2):"
echo "  python3 python/jetson_live_processor.py --camera-type csi --device 0"
echo ""
echo "  # Dashboard server (accessible from another device on same network):"
echo "  python3 app.py --port 8080 --platform jetson --source local"
echo ""
echo "  # Monitor GPU/power/thermal:"
echo "  jtop"
echo ""
echo "  # Run soak test before demo day:"
echo "  python3 scripts/soak_test.py --platform jetson --duration 3600"
echo "========================================================================="
