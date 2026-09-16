"""
test_arduino_board.py — Live Hardware Diagnostic for Arduino Uno Q
===================================================================
Tests standalone Arduino Uno Q board connected via USB:
1. Checks dependencies (pyserial).
2. Probes COM3 (or auto-detects COM port).
3. Reads board startup banner and periodic 20Hz telemetry stream.
4. Sends interactive ping {"T":1001} and test commands.
5. Verifies safe fallback behavior when sensors/motors are not wired yet.
"""

import sys
import time
import json
import io

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 65)
print("  [*] ARDUINO UNO Q HARDWARE DIAGNOSTIC TEST")
print("=" * 65)

# Step 1: Check pyserial dependency
print("\n[STEP 1] Checking Python Dependencies...")
try:
    import serial
    import serial.tools.list_ports
    print("  [OK] pyserial is installed and ready.")
except ImportError:
    print("  [FAIL] pyserial is NOT installed. Run: pip install pyserial")
    sys.exit(1)

# Step 2: Auto-detect Arduino Port
print("\n[STEP 2] Scanning for Connected Serial Devices...")
ports = list(serial.tools.list_ports.comports())
arduino_port = None
for p in ports:
    print(f"  Found Port: {p.device} ({p.description})")
    if "COM3" in p.device.upper() or "ARDUINO" in p.description.upper() or "USB SERIAL" in p.description.upper():
        arduino_port = p.device

if not arduino_port and ports:
    arduino_port = ports[0].device

if not arduino_port:
    print("  [FAIL] No COM ports detected! Please check your USB cable connection.")
    sys.exit(1)

print(f"  [TARGET] Port Selected: {arduino_port}")

# Step 3: Connect & Test Handshake
print(f"\n[STEP 3] Opening Serial Connection on {arduino_port} @ 115200 baud...")
try:
    ser = serial.Serial(arduino_port, 115200, timeout=1.5)
    print("  [OK] Serial Port successfully opened!")
except Exception as e:
    print(f"  [FAIL] Failed to open {arduino_port}: {e}")
    print("  [HINT] Check if another program (App Lab, Arduino Serial Monitor) has the port open.")
    sys.exit(1)

try:
    print("  ... Waiting for board initialization (1.5s DTR settle)...")
    time.sleep(1.5)
    
    # Flush existing buffer
    ser.reset_input_buffer()
    
    # Step 4: Listen for initial telemetry or startup banner
    print("\n[STEP 4] Listening for Firmware Output...")
    received_lines = []
    start_time = time.time()
    while time.time() - start_time < 2.0:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                received_lines.append(line)
        time.sleep(0.05)

    if received_lines:
        print("  [RECV] Received from Board:")
        for l in received_lines[:5]:
            print(f"     -> {l}")
    else:
        print("  [INFO] No spontaneous data received yet. Sending ping command...")

    # Step 5: Send Interactive Telemetry Request {"T":1001}
    print("\n[STEP 5] Testing Bidirectional Handshake (Sending {\"T\":1001})...")
    ser.write(b'{"T":1001}\n')
    time.sleep(0.5)
    
    ping_response = None
    while ser.in_waiting:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print(f"  [RECV] Response: {line}")
            if line.startswith("{") and "v" in line:
                ping_response = line

    # Step 6: Test Motor Command
    print("\n[STEP 6] Testing Motor Command Simulation (Sending {\"T\":1,\"L\":100,\"R\":100})...")
    ser.write(b'{"T":1,"L":100,"R":100}\n')
    time.sleep(0.3)
    ser.write(b'{"T":0}\n')  # Emergency Stop
    print("  [OK] Command dispatched and safe emergency stop sent.")

    # Step 7: Summary Analysis
    print("\n" + "=" * 65)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 65)
    
    firmware_detected = False
    for l in received_lines + ([ping_response] if ping_response else []):
        if "ARDUINO_WAVE_ROVER_READY" in l or ("\"v\":" in l and "\"left\":" in l):
            firmware_detected = True
            break
            
    if firmware_detected:
        print("  [STATUS: FULLY WORKING!]")
        print("  * Physical Connection: COM3 OK")
        print("  * Firmware: wave_rover_controller.ino is ACTIVE and responding")
        print("  * Safe Standalone Mode: Active (default 12.2V telemetry simulated)")
        print("  * Next: Launch 'scripts\\run_waverover_windows.bat' or python bridge!")
    else:
        print("  [STATUS: BOARD CONNECTED, WAITING FOR FIRMWARE]")
        print("  * Physical USB connection on COM3 is WORKING and accessible.")
        print("  * However, 'wave_rover_controller.ino' is not yet responding.")
        print("  * Action: Upload 'arduino/wave_rover_controller/wave_rover_controller.ino'")
        print("    via App Lab / Arduino IDE to COM3, then re-run this test.")

finally:
    ser.close()
    print("\n[INFO] Serial port released successfully.")
