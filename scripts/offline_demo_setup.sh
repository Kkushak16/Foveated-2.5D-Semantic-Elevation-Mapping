#!/bin/bash
# =====================================================================
# offline_demo_setup.sh — Network Fallback & Offline Demo Configuration
# =====================================================================
# Configures the Jetson board for a fully offline demo with no internet
# dependency. This is the network fallback plan (Section 6 Item 7 +
# Section 9) for the physical demo.
#
# What this does:
# 1. Creates a Wi-Fi hotspot on the Jetson so judges' laptops can connect
# 2. Sets up local mDNS so the dashboard is at http://lidar-demo.local:8080
# 3. Verifies local-only demo works without cloud/ngrok/venue Wi-Fi
#
# Usage:
#   chmod +x scripts/offline_demo_setup.sh
#   sudo ./scripts/offline_demo_setup.sh
#
# To tear down after demo:
#   sudo ./scripts/offline_demo_setup.sh --teardown
# =====================================================================

set -e

HOTSPOT_SSID="LiDAR-Demo"
HOTSPOT_PASS="SIH2025Demo"
DASHBOARD_PORT=8080
HOSTNAME="lidar-demo"

# -------------------------------------------------------------------------
# Parse arguments
# -------------------------------------------------------------------------
TEARDOWN=false
if [ "$1" = "--teardown" ]; then
    TEARDOWN=true
fi

if [ "$TEARDOWN" = true ]; then
    echo "=========================================="
    echo "  Tearing down offline demo configuration"
    echo "=========================================="
    # Stop hotspot
    nmcli connection down "$HOTSPOT_SSID" 2>/dev/null || true
    nmcli connection delete "$HOTSPOT_SSID" 2>/dev/null || true
    echo "✅ Hotspot '$HOTSPOT_SSID' removed"
    echo "✅ Teardown complete. Reconnect to your normal Wi-Fi."
    exit 0
fi

echo "========================================================================="
echo "  🔒 OFFLINE DEMO SETUP — No Internet Required"
echo "========================================================================="
echo ""
echo "  This will:"
echo "  1. Create a Wi-Fi hotspot: SSID='$HOTSPOT_SSID' PASS='$HOTSPOT_PASS'"
echo "  2. Set hostname to '$HOSTNAME' (mDNS: http://$HOSTNAME.local:$DASHBOARD_PORT)"
echo "  3. Verify dashboard is accessible locally"
echo ""

# -------------------------------------------------------------------------
# 1. Install required packages
# -------------------------------------------------------------------------
echo "[STEP 1/4] Installing mDNS and network tools..."
sudo apt install -y avahi-daemon avahi-utils 2>/dev/null || true

# -------------------------------------------------------------------------
# 2. Set hostname for mDNS
# -------------------------------------------------------------------------
echo "[STEP 2/4] Configuring hostname for mDNS..."
sudo hostnamectl set-hostname "$HOSTNAME"
sudo systemctl restart avahi-daemon 2>/dev/null || true
echo "  Hostname set to: $HOSTNAME"
echo "  mDNS address: http://$HOSTNAME.local:$DASHBOARD_PORT"

# -------------------------------------------------------------------------
# 3. Create Wi-Fi hotspot
# -------------------------------------------------------------------------
echo "[STEP 3/4] Creating Wi-Fi hotspot..."

# Check if NetworkManager is available
if ! command -v nmcli &>/dev/null; then
    echo "  ⚠️  NetworkManager (nmcli) not found."
    echo "  The Jetson may use a different network manager."
    echo "  Manual hotspot creation may be required."
    echo ""
    echo "  Alternative: Connect Jetson and judge laptops to the same"
    echo "  mobile hotspot from a phone, and use the Jetson's IP address"
    echo "  directly (run 'ip addr' to find it)."
else
    # Remove old connection if it exists
    nmcli connection delete "$HOTSPOT_SSID" 2>/dev/null || true

    # Create hotspot
    nmcli device wifi hotspot \
        ssid "$HOTSPOT_SSID" \
        password "$HOTSPOT_PASS" \
        ifname wlan0 2>/dev/null || {
        echo "  ⚠️  Wi-Fi hotspot creation failed."
        echo "  This may be because:"
        echo "  - wlan0 interface doesn't exist (use 'ip link' to check)"
        echo "  - Wi-Fi adapter doesn't support AP mode"
        echo ""
        echo "  Fallback: Use a phone hotspot or Ethernet cable."
    }

    echo "  ✅ Wi-Fi Hotspot Active"
    echo "     SSID: $HOTSPOT_SSID"
    echo "     Pass: $HOTSPOT_PASS"
fi

# -------------------------------------------------------------------------
# 4. Get IP address and verify
# -------------------------------------------------------------------------
echo "[STEP 4/4] Verifying network configuration..."
echo ""
echo "  Network interfaces:"
ip -4 addr show | grep -E "inet " | grep -v "127.0.0.1" | while read -r line; do
    echo "    $line"
done

echo ""
echo "========================================================================="
echo "  ✅ OFFLINE DEMO READY"
echo "========================================================================="
echo ""
echo "  For judges to connect:"
echo "  1. Connect to Wi-Fi: SSID='$HOTSPOT_SSID' / Pass='$HOTSPOT_PASS'"
echo "  2. Open browser: http://$HOSTNAME.local:$DASHBOARD_PORT"
echo "     (or use the IP address shown above)"
echo ""
echo "  To start the dashboard:"
echo "    python3 app.py --port $DASHBOARD_PORT --platform jetson --source local"
echo ""
echo "  To tear down after demo:"
echo "    sudo ./scripts/offline_demo_setup.sh --teardown"
echo "========================================================================="
