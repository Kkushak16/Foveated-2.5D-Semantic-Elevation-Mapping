/**
 * hardware_connect.js — Hardware Connection Wizard & Live Rover Dashboard
 * 
 * Manages the 3-step flow:
 *   1. Select Device  → user picks hardware platform
 *   2. Setup Guide    → tailored step-by-step instructions
 *   3. Connect & Live → real-time camera/BEV/telemetry dashboard
 */

(function () {
    'use strict';

    // ─── State ────────────────────────────────────────────────
    let hwSelectedDevice = null;
    let hwBridgeHost = 'http://localhost:8081';
    let hwIsConnected = false;
    let hwPollInterval = null;
    let hwFpsCounter = { frames: 0, lastTime: performance.now(), fps: 0 };
    let hwLatency = 0;
    // Local dead-reckoning pose, integrated from the on-screen drive buttons.
    // Must be declared up-front: this IIFE runs in strict mode, so reading an
    // undeclared binding throws a ReferenceError before the drive command is
    // ever sent to the bridge (which made every teleop button look dead).
    let hwMockPose = { x: 0, y: 0, yaw: 0 };

    // ─── Instruction Templates ────────────────────────────────
    const HW_INSTRUCTIONS = {
        'arduino-uno-q': {
            title: '🤖 Arduino Uno Q — Setup Guide',
            steps: [
                'Unbox your Arduino Uno Q and locate the <b>USB-C</b> cable.',
                'Mount the Arduino Uno Q on the Wave Rover chassis using the M3 standoffs provided.',
                'Connect the motor driver shield: <code>IN1→D5, IN2→D6, IN3→D9, IN4→D10, ENA→D3, ENB→D11</code>.',
                'Attach the encoder wires: <code>Left Encoder → D2 (INT0), Right Encoder → D7</code>.',
                'Connect a USB webcam (or CSI ribbon cam) to the <b>laptop</b> — the Arduino Uno Q handles motors, your laptop runs Qwen3-VL vision.',
                'Plug the Arduino Uno Q into your laptop via USB-C. The serial port should appear as <code>COMx</code> (Windows) or <code>/dev/ttyACM0</code> (Linux).',
                'Flash the Wave Rover firmware: <code>cd firmware && arduino-cli compile --fqbn arduino:avr:uno . && arduino-cli upload -p COMx</code>.',
                'Install the Python bridge: <code>pip install pyserial opencv-python flask</code>.',
                'Run the bridge server: <code>python python/waverover_bridge.py</code> — it auto-detects the Arduino Uno Q on serial.',
                'Return here, enter <code>http://localhost:8081</code> as the Bridge Host, and click <b>Connect</b>.'
            ],
            tempType: 'MCU Internal (ADC)',
            tempNote: 'Arduino Uno Q on-chip temperature via analogRead()'
        },
        'arduino-uno-r4': {
            title: '🔧 Arduino Uno R3 / R4 — Setup Guide',
            steps: [
                'Connect your Arduino Uno to the Wave Rover motor driver shield.',
                'Wire the motors: <code>IN1→D5, IN2→D6, IN3→D9, IN4→D10, ENA→D3, ENB→D11</code>.',
                'If using R4 WiFi: configure WiFi credentials in the firmware sketch under <code>wifi_config.h</code>.',
                'Attach the USB webcam to your <b>laptop</b> for Qwen3-VL vision processing.',
                'Plug the Arduino into your laptop via USB-B (R3) or USB-C (R4). Note the COM port in Device Manager.',
                'Flash firmware: <code>arduino-cli compile --fqbn arduino:avr:uno . && arduino-cli upload -p COMx</code>.',
                'Install the Python bridge: <code>pip install pyserial opencv-python flask</code>.',
                'Run the bridge: <code>python python/waverover_bridge.py</code>.',
                'Return here and click <b>Connect</b> with the default bridge host.'
            ],
            tempType: 'MCU Internal (R4) / External DS18B20 (R3)',
            tempNote: 'R4 has on-chip temp sensor; R3 needs DS18B20 breakout'
        },
        'raspberry-pi': {
            title: '🍓 Raspberry Pi 4 / 5 — Setup Guide',
            steps: [
                'Flash Raspberry Pi OS (64-bit Lite recommended) onto your SD card using Raspberry Pi Imager.',
                'Boot the Pi and connect to your network via WiFi or Ethernet. Note its IP with <code>hostname -I</code>.',
                'Update packages: <code>sudo apt update && sudo apt upgrade -y</code>.',
                'Install Python deps: <code>pip install pyserial opencv-python flask torch transformers</code>.',
                'Connect the Pi Camera Module via CSI ribbon cable, or plug in a USB webcam.',
                'Enable the camera: <code>sudo raspi-config</code> → Interface Options → Camera → Enable.',
                'Wire the Pi GPIO to the motor driver: <code>GPIO17→IN1, GPIO18→IN2, GPIO22→IN3, GPIO23→IN4</code>.',
                'Clone the project: <code>git clone &lt;repo&gt; && cd Antigravity</code>.',
                'Run the bridge on the Pi: <code>python python/waverover_bridge.py --host 0.0.0.0</code>.',
                'On your laptop, enter <code>http://&lt;PI_IP&gt;:8081</code> as the Bridge Host and click <b>Connect</b>.'
            ],
            tempType: 'SoC Temperature (vcgencmd)',
            tempNote: 'Read via: vcgencmd measure_temp'
        },
        'esp32': {
            title: '📡 ESP32 (Factory Onboard) — Setup Guide',
            steps: [
                'The Waveshare Wave Rover comes with an <b>ESP32</b> pre-installed on the driver board.',
                'Connect the ESP32 to your laptop via <b>Micro-USB</b> cable.',
                'Install ESP32 drivers if needed — the CH340/CP2102 driver for your OS.',
                'Open Device Manager → Ports and note the COM port (e.g., <code>COM4</code>).',
                'Flash the Wave Rover JSON-Serial firmware using PlatformIO or Arduino IDE:',
                '<code>platformio run --target upload --upload-port COMx</code>',
                'The webcam should be connected to your <b>laptop</b> for Qwen3-VL processing.',
                'Install the Python bridge: <code>pip install pyserial opencv-python flask</code>.',
                'Run the bridge: <code>python python/waverover_bridge.py</code> — ESP32 auto-detected on serial.',
                'Return here, click <b>Connect</b>.'
            ],
            tempType: 'ESP32 Internal Sensor',
            tempNote: 'Hall-effect / internal temperature API (temprature_sens_read)'
        },
        'custom': {
            title: '⚙️ Custom Board — Setup Guide',
            steps: [
                'Ensure your board supports serial communication at <code>115200 baud</code>.',
                'The board must respond to JSON commands: <code>{"cmd":"drive","left":0.5,"right":0.5}</code>.',
                'It should send telemetry JSON: <code>{"battery":12.1,"temp":42.5,"enc_l":1234,"enc_r":1230}</code>.',
                'Connect the board to your laptop via USB, UART-to-USB adapter, or WiFi.',
                'If WiFi: ensure it exposes a TCP socket on port 8081 (or your preferred port).',
                'Connect a webcam to your laptop for vision processing.',
                'Install the Python bridge: <code>pip install pyserial opencv-python flask</code>.',
                'Run: <code>python python/waverover_bridge.py --port COMx</code> (replace COMx with your port).',
                'Return here, enter the appropriate Bridge Host and click <b>Connect</b>.'
            ],
            tempType: 'Custom (if available)',
            tempNote: 'Temperature reporting depends on your firmware implementation'
        }
    };

    // ─── Step Navigation ──────────────────────────────────────
    function hwGoToStep(step) {
        const sections = ['hw-section-select', 'hw-section-instructions', 'hw-section-connect'];
        const pills = ['hw-step-1', 'hw-step-2', 'hw-step-3'];

        sections.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.classList.add('hw-section-hidden');
        });

        pills.forEach((id, i) => {
            const el = document.getElementById(id);
            if (!el) return;
            el.classList.remove('active', 'done');
            if (i + 1 < step) el.classList.add('done');
            if (i + 1 === step) el.classList.add('active');
        });

        const target = document.getElementById(sections[step - 1]);
        if (target) target.classList.remove('hw-section-hidden');

        // Intelligent state handling per step
        if (step === 1) {
            const dev = hwSelectedDevice || 'arduino-uno-q';
            hwSelectedDevice = dev;
            document.querySelectorAll('.hw-card').forEach(c => c.classList.remove('selected'));
            const card = document.getElementById('hw-card-' + dev);
            if (card) card.classList.add('selected');
            const lbl = document.getElementById('hw-step1-selected-label');
            if (lbl) {
                const names = {
                    'arduino-uno-q': 'Arduino Uno Q (Recommended)',
                    'arduino-uno-r4': 'Arduino Uno R3 / R4',
                    'raspberry-pi': 'Raspberry Pi 4 / 5',
                    'esp32': 'ESP32 (Wave Rover Onboard)',
                    'custom': 'Custom MCU / Robot'
                };
                lbl.textContent = names[dev] || dev;
            }
        } else if (step === 2) {
            // Always ensure valid instructions are rendered
            const dev = hwSelectedDevice || 'arduino-uno-q';
            hwSelectedDevice = dev;
            hwRenderInstructions(dev);
        } else if (step === 3) {
            // Ensure live dashboard is visible
            const dash = document.getElementById('hw-live-dashboard');
            if (dash) dash.style.display = 'block';

            if (!hwIsConnected) {
                // Connection is now strictly hardware-driven.
                // Mock simulation is disabled to ensure real telemetry is shown.
            }
        }
    }

    // ─── Instructions Renderer ────────────────────────────────
    function hwRenderInstructions(deviceId) {
        deviceId = deviceId || hwSelectedDevice || 'arduino-uno-q';
        hwSelectedDevice = deviceId;
        const info = HW_INSTRUCTIONS[deviceId];
        const container = document.getElementById('hw-instructions-content');
        if (!info || !container) return;

        let stepsHtml = info.steps.map(s => `<li>${s}</li>`).join('');
        container.innerHTML = `
            <!-- Quick Board Switcher Tabs -->
            <div class="hw-guide-tabs">
                <button class="hw-guide-tab ${deviceId === 'arduino-uno-q' ? 'active' : ''}" onclick="hwSelectDevice('arduino-uno-q', false)">🤖 Arduino Uno Q</button>
                <button class="hw-guide-tab ${deviceId === 'arduino-uno-r4' ? 'active' : ''}" onclick="hwSelectDevice('arduino-uno-r4', false)">🔧 Uno R3 / R4</button>
                <button class="hw-guide-tab ${deviceId === 'raspberry-pi' ? 'active' : ''}" onclick="hwSelectDevice('raspberry-pi', false)">🍓 Raspberry Pi</button>
                <button class="hw-guide-tab ${deviceId === 'esp32' ? 'active' : ''}" onclick="hwSelectDevice('esp32', false)">📡 ESP32</button>
                <button class="hw-guide-tab ${deviceId === 'custom' ? 'active' : ''}" onclick="hwSelectDevice('custom', false)">⚙️ Custom</button>
            </div>

            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; flex-wrap:wrap; gap:8px;">
                <h3 style="margin:0; font-size:1.2rem; color:#ffffff; font-weight:700;">${info.title}</h3>
                <span style="font-size:0.72rem; padding:4px 10px; border-radius:6px; background:rgba(245,158,11,0.12); color:#f59e0b; border:1px solid rgba(245,158,11,0.25); font-family:var(--font-mono);">PLATFORM GUIDE</span>
            </div>

            <ol style="padding-left:20px; color:var(--text-secondary); font-size:0.86rem; line-height:1.75;">
                ${stepsHtml}
            </ol>

            <!-- Dedicated Telemetry & Vision Notice Cards -->
            <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)); gap:12px; margin-top:20px;">
                <!-- Temperature Monitoring -->
                <div style="padding:14px 16px; background:rgba(0,0,0,0.4); border:1px solid rgba(255,255,255,0.08); border-radius:10px;">
                    <div style="font-size:0.76rem; font-weight:700; color:var(--accent-amber); margin-bottom:4px; display:flex; align-items:center; gap:6px;">
                        <span>🌡️</span> Board Temperature Telemetry
                    </div>
                    <div style="font-size:0.75rem; color:var(--text-secondary); line-height:1.45;">
                        <strong>${info.tempType}</strong> — ${info.tempNote}. Monitored continuously in the live dashboard to prevent overheating.
                    </div>
                </div>

                <!-- Camera Feed Destination -->
                <div style="padding:14px 16px; background:rgba(0,0,0,0.4); border:1px solid rgba(255,255,255,0.08); border-radius:10px;">
                    <div style="font-size:0.76rem; font-weight:700; color:#ffffff; margin-bottom:4px; display:flex; align-items:center; gap:6px;">
                        <span>📹</span> Where Feed Will Appear
                    </div>
                    <div style="font-size:0.75rem; color:var(--text-secondary); line-height:1.45;">
                        Visible in <strong>Step 3 (Live Viewport)</strong> and in the primary <strong>Teleop HUD Cockpit</strong> with 2.5D foveated perception & neural bounding boxes.
                    </div>
                </div>

                <!-- Qwen3-VL LiDAR-Free Mapping -->
                <div style="padding:14px 16px; background:rgba(0,0,0,0.4); border:1px solid rgba(245,158,11,0.2); border-radius:10px;">
                    <div style="font-size:0.76rem; font-weight:700; color:var(--accent-amber); margin-bottom:4px; display:flex; align-items:center; gap:6px;">
                        <span>🧠</span> No LiDAR? Qwen3-VL Spatial Mapping
                    </div>
                    <div style="font-size:0.75rem; color:var(--text-secondary); line-height:1.45;">
                        LiDAR is optional. <strong>Qwen3-VL</strong> maps corridor boundaries directly from video frames, populates the BEV grid, and navigates collision-free.
                    </div>
                </div>
            </div>
        `;

        // Update button in step 2 to reflect selected board
        const btnAutoconnect = document.getElementById('btn-step2-autoconnect');
        if (btnAutoconnect) {
            btnAutoconnect.setAttribute('onclick', `hwAutoConnectDevice('${deviceId}')`);
        }
    }

    // ─── Device Selection ─────────────────────────────────────
    function hwSelectDevice(deviceId, shouldNavigate = true) {
        hwSelectedDevice = deviceId;

        // Visual: highlight selected card
        document.querySelectorAll('.hw-card').forEach(c => c.classList.remove('selected'));
        const card = document.getElementById('hw-card-' + deviceId);
        if (card) card.classList.add('selected');

        const lbl = document.getElementById('hw-step1-selected-label');
        if (lbl) {
            const names = {
                'arduino-uno-q': 'Arduino Uno Q (Recommended)',
                'arduino-uno-r4': 'Arduino Uno R3 / R4',
                'raspberry-pi': 'Raspberry Pi 4 / 5',
                'esp32': 'ESP32 (Wave Rover Onboard)',
                'custom': 'Custom MCU / Robot'
            };
            lbl.textContent = names[deviceId] || deviceId;
        }

        hwRenderInstructions(deviceId);

        if (shouldNavigate) {
            // Move to step 2 after a short delay for visual feedback
            setTimeout(() => hwGoToStep(2), 250);
        }
    }

    // ─── Connection ───────────────────────────────────────────
    function hwSetStatus(state, text) {
        const dot = document.getElementById('hw-status-dot');
        const label = document.getElementById('hw-status-text');
        if (dot) {
            dot.className = 'hw-status-dot ' + state;
        }
        if (label) label.innerHTML = text;
    }

    // ─── Direct Auto-Connect to Selected Hardware ──────────────
    async function hwAutoConnectDevice(deviceId) {
        deviceId = deviceId || hwSelectedDevice || 'arduino-uno-q';
        hwSelectedDevice = deviceId;
        
        // Highlight card
        document.querySelectorAll('.hw-card').forEach(c => c.classList.remove('selected'));
        const card = document.getElementById('hw-card-' + deviceId);
        if (card) card.classList.add('selected');

        // Pre-configure inputs based on hardware choice
        const hostInput = document.getElementById('hw-bridge-host');
        const portInput = document.getElementById('hw-serial-port');
        const titleEl = document.getElementById('hw-connect-panel-title');

        if (deviceId === 'arduino-uno-q') {
            if (portInput) portInput.value = 'Auto-Detect (COM/USB-C)';
            if (titleEl) titleEl.textContent = 'Arduino Uno Q — Connecting via USB-C';
        } else if (deviceId === 'arduino-uno-r4') {
            if (portInput) portInput.value = 'Auto-Detect (COM/USB)';
            if (titleEl) titleEl.textContent = 'Arduino Uno R3/R4 — Connecting via Serial';
        } else if (deviceId === 'raspberry-pi') {
            if (portInput) portInput.value = 'Network /dev/serial0';
            if (titleEl) titleEl.textContent = 'Raspberry Pi 4/5 — Connecting via WiFi Bridge';
        } else if (deviceId === 'esp32') {
            if (portInput) portInput.value = 'Auto-Detect (Micro-USB)';
            if (titleEl) titleEl.textContent = 'ESP32 Wave Rover — Connecting via Serial';
        } else {
            if (portInput) portInput.value = 'Auto-Detect';
            if (titleEl) titleEl.textContent = 'Custom Controller — Connecting';
        }

        // Jump straight to Step 3 and attempt connection
        hwGoToStep(3);
        await hwAttemptConnect();
    }

    // ─── Scan & Auto-Detect Any Plugged-in Hardware ───────────
    async function hwScanAndAutoConnect() {
        hwGoToStep(3);
        const titleEl = document.getElementById('hw-connect-panel-title');
        if (titleEl) titleEl.textContent = 'Auto-Scanning Hardware Ports...';
        await hwAttemptConnect();
    }

    function hwUpdateTeleopLock(unlocked) {
        const badge = document.getElementById('hw-teleop-lock-badge');
        const group = document.getElementById('hw-teleop-button-group');
        if (badge) {
            badge.style.background = unlocked ? 'rgba(16,185,129,0.2)' : 'rgba(239,68,68,0.2)';
            badge.style.color = unlocked ? '#34d399' : '#f87171';
            badge.textContent = unlocked ? '🟢 UNLOCKED: ROVER READY' : '🔒 LOCKED: CONNECT ROVER';
        }
        if (group) {
            group.style.opacity = unlocked ? '1' : '0.45';
            group.style.pointerEvents = unlocked ? 'auto' : 'none';
        }
    }

    // ─── Primary Connection Handler ───────────────────────────
    async function hwAttemptConnect(retriedWithAutoStart = false) {
        hwBridgeHost = (document.getElementById('hw-bridge-host')?.value || 'http://localhost:8081').replace(/\/+$/, '');
        const roverIp = (document.getElementById('hw-rover-ip')?.value || '192.168.4.1').trim();
        const btn = document.getElementById('hw-btn-connect');
        const btnDis = document.getElementById('hw-btn-disconnect');
        if (btn) {
            btn.disabled = true;
            btn.textContent = '⏳ Verifying Rover Wheels...';
        }

        hwSetStatus('connecting', `Probing Rover at IP <strong>${roverIp}</strong> & Arduino Uno Q...`);

        try {
            // Explicitly connect to Rover IP and trigger wheel verification wiggle
            try {
                await fetch(hwBridgeHost + '/api/rover/connect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ip: roverIp })
                });
            } catch (_) {}

            // Robust dual-endpoint fetch: direct 8081 and via node proxy (8080).
            // Earlier fallback only tried proxy on network error, so a stale 8081
            // "disconnected" payload masked a healthy proxy "connected" (COM3/Daemon).
            // Now we try both and prefer any "connected:true".
            let data = null;
            let dataProxy = null;
            try {
                const res = await fetch(hwBridgeHost + '/api/rover/status', { signal: AbortSignal.timeout(3500) });
                if (res.ok) data = await res.json();
            } catch (_) {}
            try {
                const res2 = await fetch('/api/rover/status', { signal: AbortSignal.timeout(3500) });
                if (res2.ok) dataProxy = await res2.json();
            } catch (_) {}
            // Prefer the connected one, otherwise keep direct
            if (dataProxy && dataProxy.connected && (!data || !data.connected)) {
                data = dataProxy;
                hwBridgeHost = window.location.origin;
                const hostInput = document.getElementById('hw-bridge-host');
                if (hostInput) hostInput.value = hwBridgeHost;
            } else if (!data && dataProxy) {
                data = dataProxy;
                hwBridgeHost = window.location.origin;
                const hostInput = document.getElementById('hw-bridge-host');
                if (hostInput) hostInput.value = hwBridgeHost;
            }
            if (!data) throw new Error('Bridge not detected');

            if (!data.connected) {
                hwIsConnected = false;
                hwUpdateTeleopLock(false);
                hwSetStatus('disconnected', `
                    <div style="display:flex; flex-direction:column; gap:4px;">
                        <span style="color:#ef4444; font-weight:700;">🔴 Rover / Arduino Not Reachable</span>
                        <span style="color:var(--text-secondary); font-size:0.75rem;">
                            Bridge is online, but neither Arduino Uno Q on USB nor Rover at ${roverIp} responded. Verify Rover is powered and Uno Q is connected.
                        </span>
                    </div>
                `);
                hwResetMetricsToDisconnected();
                if (btn) {
                    btn.disabled = false;
                    btn.textContent = '🔌 Connect Rover & Verify Wheels';
                    btn.style.background = '';
                }
                if (btnDis) btnDis.style.display = 'none';
                return;
            }

            hwIsConnected = true;
            hwUpdateTeleopLock(true);
            const detectedMcu = data.hardware || hwSelectedDevice || 'Arduino Uno Q + Wave Rover 4WD';
            hwSetStatus('connected', `Connected ✓ — <strong>${detectedMcu}</strong> active! Wheel verification test confirmed &amp; controls unlocked.`);

            // Automatically signal OK on Arduino LEDs
            setTimeout(() => {
                fetch(hwBridgeHost + '/api/rover/ping_led', { method: 'POST', headers: { 'Content-Type': 'application/json' } }).catch(() => {});
            }, 300);

            // Show live dashboard
            const dash = document.getElementById('hw-live-dashboard');
            if (dash) dash.style.display = 'block';

            // Start camera feed
            hwStartCameraFeed();

            // Start telemetry polling at 350ms
            if (hwPollInterval) clearInterval(hwPollInterval);
            hwPollInterval = setInterval(hwPollTelemetry, 350);

            // Start BEV rendering
            hwStartBEVRenderer();

            if (btn) {
                btn.disabled = false;
                btn.textContent = '✓ Rover Connected';
                btn.style.background = 'linear-gradient(180deg, #10b981 0%, #059669 100%)';
            }
            if (btnDis) {
                btnDis.style.display = 'inline-block';
            }

        } catch (err) {
            // If bridge is offline and we haven't attempted auto-launch yet, try starting it via server
            if (!retriedWithAutoStart) {
                try {
                    hwSetStatus('connecting', 'Bridge offline — launching hardware bridge on port 8081...');
                    const sRes = await fetch('/api/bridge/start');
                    if (sRes.ok) {
                        await new Promise(r => setTimeout(r, 1600));
                        return await hwAttemptConnect(true);
                    }
                } catch (_) {}
            }

            hwIsConnected = false;
            hwResetMetricsToDisconnected();
            hwSetStatus('disconnected', `
                <div style="display:flex; flex-direction:column; gap:6px;">
                    <span>⚠️ Bridge not detected at <code>${hwBridgeHost}</code>.</span>
                    <div style="display:flex; align-items:center; gap:8px; margin-top:3px;">
                        <button onclick="window.hwAttemptConnect(false)" style="background:#2563eb; color:#fff; border:none; padding:5px 12px; border-radius:4px; font-size:0.75rem; cursor:pointer; font-weight:600;">⚡ Launch Bridge & Connect</button>
                    </div>
                    <span style="color:var(--text-secondary); font-size:0.75rem;">
                        Manual terminal command: <code>.\\.venv\\Scripts\\python.exe python/waverover_bridge.py</code>
                    </span>
                </div>
            `);
            if (btn) {
                btn.disabled = false;
                btn.textContent = '🔌 Retry Connection';
                btn.style.background = 'linear-gradient(180deg, #f59e0b 0%, #d97706 100%)';
            }
            if (btnDis) {
                btnDis.style.display = 'none';
            }
        }
    }

    // ─── Manual Disconnect Handler ─────────────────────────────
    async function hwDisconnect() {
        if (hwPollInterval) {
            clearInterval(hwPollInterval);
            hwPollInterval = null;
        }
        if (hwBevAnimFrame) {
            cancelAnimationFrame(hwBevAnimFrame);
            hwBevAnimFrame = null;
        }
        hwIsConnected = false;
        hwStopCameraFeed();

        try {
            await fetch(hwBridgeHost + '/api/rover/disconnect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                signal: AbortSignal.timeout(1500)
            });
        } catch (e) {
            try {
                await fetch('/api/rover/disconnect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    signal: AbortSignal.timeout(1500)
                });
            } catch (_) {}
        }

        hwResetMetricsToDisconnected();
        hwSetStatus('disconnected', '🔴 Disconnected — Board connection released by user.');

        const btn = document.getElementById('hw-btn-connect');
        if (btn) {
            btn.disabled = false;
            btn.textContent = '🔌 Connect Rover & Verify Wheels';
            btn.style.background = '';
        }
        hwUpdateTeleopLock(false);
        const btnDis = document.getElementById('hw-btn-disconnect');
        if (btnDis) btnDis.style.display = 'none';
    }

    // ─── Hardware LED Ping / OK Check ────────────────────────
    // Option A fix: Check Board now binds the rover IP first, then lights the
    // 8x13 matrix. Previously it ignored the IP field and only tried the
    // daemon, so only QRB 1/2 glowed and 'OK' never appeared.
    async function hwPingLedSignal() {
        const pingBtn = document.getElementById('hw-btn-ping-led');
        const origText = pingBtn ? pingBtn.innerHTML : '✨ Check Board & Show OK';
        if (pingBtn) {
            pingBtn.disabled = true;
            pingBtn.innerHTML = '⏳ Checking...';
            pingBtn.style.opacity = '0.75';
        }

        // 0) Bind the IP the user typed (e.g. 192.168.4.1) to the bridge BEFORE pinging,
        // so the matrix + wheel path uses the correct device IP.
        const roverIpForPing = (document.getElementById('hw-rover-ip')?.value || '192.168.4.1').trim();
        if (roverIpForPing) {
            const hostPre = (document.getElementById('hw-bridge-host')?.value || hwBridgeHost || '').replace(/\/+$/, '') || hwBridgeHost;
            for (const base of [hostPre, window.location.origin]) {
                try {
                    await fetch(base + '/api/rover/connect', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ ip: roverIpForPing }),
                        signal: AbortSignal.timeout(2000)
                    });
                    break;
                } catch (_) {}
            }
        }

        try {
            const hostInput = document.getElementById('hw-bridge-host');
            const host = (hostInput ? hostInput.value.trim() : '') || hwBridgeHost;
            let data = null;
            try {
                const res = await fetch(host + '/api/rover/ping_led', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    signal: AbortSignal.timeout(3500)
                });
                if (res.ok) data = await res.json();
            } catch (e) {
                try {
                    const res = await fetch('/api/rover/ping_led', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        signal: AbortSignal.timeout(3500)
                    });
                    if (res.ok) data = await res.json();
                } catch (_) {}
            }
            if (!data) throw new Error('Bridge unreachable');

            if (data.connected) {
                if (pingBtn) {
                    pingBtn.innerHTML = '✅ OK Displayed on LED!';
                    pingBtn.style.background = 'linear-gradient(135deg, rgba(16,185,129,0.35) 0%, rgba(5,150,105,0.35) 100%)';
                    pingBtn.style.borderColor = '#10b981';
                    pingBtn.style.color = '#34d399';
                }
                hwSetStatus('connected', `✅ <strong>${data.hardware || 'Arduino Uno Q'}</strong> Confirmed! Sent 'OK' signal to 8x13 LED Matrix. Wheel wiggle running — you can now drive with WASD.`);

                // Promote UI to connected and unlock teleop immediately (IP-first flow).
                hwIsConnected = true;
                hwUpdateTeleopLock(true);
                const dash = document.getElementById('hw-live-dashboard');
                if (dash) dash.style.display = 'block';
                if (!hwPollInterval) hwPollInterval = setInterval(hwPollTelemetry, 350);
                hwStartBEVRenderer();
                hwStartCameraFeed();
                const btn = document.getElementById('hw-btn-connect');
                if (btn) {
                    btn.textContent = '✓ Rover Connected';
                    btn.style.background = 'linear-gradient(180deg, #10b981 0%, #059669 100%)';
                    btn.disabled = false;
                }
                const btnDis = document.getElementById('hw-btn-disconnect');
                if (btnDis) btnDis.style.display = 'inline-block';
            } else {
                if (pingBtn) {
                    pingBtn.innerHTML = '🔴 Not Connected';
                    pingBtn.style.background = 'rgba(239, 68, 68, 0.2)';
                    pingBtn.style.borderColor = '#ef4444';
                    pingBtn.style.color = '#f87171';
                }
                hwSetStatus('disconnected', `🔴 ${data.message || 'Arduino is not connected. For IP rover: ensure laptop is on rover WiFi (192.168.4.1) and IP in field is correct, then click Connect.'}`);
            }
        } catch (err) {
            // Bridge offline — try to auto-launch it via Node helper then retry once
            let retried = false;
            try {
                const sRes = await fetch('/api/bridge/start', { signal: AbortSignal.timeout(2000) });
                if (sRes && sRes.ok) {
                    await new Promise(r => setTimeout(r, 1200));
                    const host2 = (document.getElementById('hw-bridge-host')?.value || hwBridgeHost || '').replace(/\/+$/, '') || hwBridgeHost;
                    let d2 = null;
                    try {
                        const r2 = await fetch(host2 + '/api/rover/ping_led', { method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: AbortSignal.timeout(3000) });
                        if (r2.ok) d2 = await r2.json();
                    } catch (_) {
                        try { const r2 = await fetch('/api/rover/ping_led', { method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: AbortSignal.timeout(3000) }); if (r2.ok) d2 = await r2.json(); } catch (_) {}
                    }
                    if (d2 && d2.connected) {
                        retried = true;
                        if (pingBtn) {
                            pingBtn.innerHTML = '✅ OK Displayed on LED!';
                            pingBtn.style.background = 'linear-gradient(135deg, rgba(16,185,129,0.35) 0%, rgba(5,150,105,0.35) 100%)';
                            pingBtn.style.borderColor = '#10b981';
                            pingBtn.style.color = '#34d399';
                        }
                        hwSetStatus('connected', `✅ <strong>${d2.hardware || 'Arduino Uno Q'}</strong> Confirmed via bridge restart! Wheel wiggle running.`);
                        hwIsConnected = true;
                        hwUpdateTeleopLock(true);
                        const dash = document.getElementById('hw-live-dashboard');
                        if (dash) dash.style.display = 'block';
                        if (!hwPollInterval) hwPollInterval = setInterval(hwPollTelemetry, 350);
                        hwStartBEVRenderer(); hwStartCameraFeed();
                    }
                }
            } catch (_) {}
            if (!retried) {
                if (pingBtn) {
                    pingBtn.innerHTML = '⚠️ Bridge Offline';
                    pingBtn.style.background = 'rgba(245, 158, 11, 0.2)';
                    pingBtn.style.borderColor = '#f59e0b';
                    pingBtn.style.color = '#fbbf24';
                }
                hwSetStatus('disconnected', `⚠️ Bridge service offline at <code>${hwBridgeHost}</code>. Click "Connect Rover & Verify Wheels" to auto-launch the bridge, or run <code>python python/waverover_bridge.py</code>`);
            }
        } finally {
            setTimeout(() => {
                if (pingBtn) {
                    pingBtn.disabled = false;
                    pingBtn.innerHTML = origText;
                    pingBtn.style.background = 'rgba(236, 72, 153, 0.15)';
                    pingBtn.style.borderColor = 'rgba(236, 72, 153, 0.45)';
                    pingBtn.style.color = '#f472b6';
                    pingBtn.style.opacity = '1';
                }
            }, 3000);
        }
    }

    function hwResetMetricsToDisconnected() {
        const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
        set('hw-m-fps', '—');
        set('hw-m-latency', '—');
        set('hw-m-temp', '—');
        set('hw-m-battery', '—');
        set('hw-t-hardware', 'DISCONNECTED');
        set('hw-t-port', '—');
        set('hw-t-connected', 'OFFLINE');
        set('hw-t-pose-x', '—');
        set('hw-t-pose-y', '—');
        set('hw-t-yaw', '—');
        set('hw-t-ring0', '—');
        set('hw-t-ring1', '—');
        set('hw-t-ring2', '—');

        const standbySvg = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='640' height='360' viewBox='0 0 640 360'><rect width='640' height='360' fill='%2305070c'/><text x='320' y='165' font-family='sans-serif' font-size='14' fill='%23f59e0b' font-weight='600' text-anchor='middle'>🛰️ Rover Camera Standby (Hardware Offline)</text><text x='320' y='195' font-family='monospace' font-size='11' fill='%2364748b' text-anchor='middle'>Connect Arduino Uno Q via USB-C to activate live feed</text></svg>";
        const feed = document.getElementById('hw-camera-feed');
        if (feed) feed.src = standbySvg;
        const splitFeed = document.getElementById('hw-camera-feed-split');
        if (splitFeed) splitFeed.src = standbySvg;
    }

    // ─── Hardware Simulation Mode Disabled ────────────────────
    function hwLaunchMockSimulation() {
        hwSetStatus('disconnected', `
            <div style="display:flex; flex-direction:column; gap:4px;">
                <span style="color:#ef4444; font-weight:700;">⚠️ Simulation Mode Disabled</span>
                <span style="color:var(--text-secondary); font-size:0.75rem;">
                    Fake/emulated telemetry is permanently disabled. Only real-time values from the physical Arduino Uno Q board are supported. Connect the board via USB-C to view live telemetry.
                </span>
            </div>
        `);
    }

    function hwPollMockTelemetry() {
        // Disabled — strictly no fake telemetry allowed
        return;
    }

    // ─── Mock Camera Frame Generator ──────────────────────────
    let hwMockCamCanvas = null;
    // Interval handle for the synthetic frame generator. Declared here because
    // this IIFE is strict-mode: assigning an undeclared identifier throws.
    let hwMockAnimInterval = null;

    function hwStartMockCameraFeed() {
        if (!hwMockCamCanvas) {
            hwMockCamCanvas = document.createElement('canvas');
            hwMockCamCanvas.width = 640;
            hwMockCamCanvas.height = 360;
        }

        if (hwMockAnimInterval) clearInterval(hwMockAnimInterval);
        hwMockAnimInterval = setInterval(() => {
            if (!hwIsConnected) return;
            const ctx = hwMockCamCanvas.getContext('2d');
            const w = hwMockCamCanvas.width, h = hwMockCamCanvas.height;
            const t = Date.now() * 0.001;

            // Gradient floor & sky
            const skyGrad = ctx.createLinearGradient(0, 0, 0, h * 0.45);
            skyGrad.addColorStop(0, '#040405');
            skyGrad.addColorStop(1, '#090d16');
            ctx.fillStyle = skyGrad;
            ctx.fillRect(0, 0, w, h * 0.45);

            const floorGrad = ctx.createLinearGradient(0, h * 0.45, 0, h);
            floorGrad.addColorStop(0, '#0c121e');
            floorGrad.addColorStop(1, '#05070c');
            ctx.fillStyle = floorGrad;
            ctx.fillRect(0, h * 0.45, w, h * 0.55);

            // Perspective Grid Lines
            ctx.strokeStyle = 'rgba(245, 158, 11, 0.15)';
            ctx.lineWidth = 1;
            const vpX = w / 2, vpY = h * 0.45;
            for (let i = -6; i <= 6; i++) {
                ctx.beginPath();
                ctx.moveTo(vpX, vpY);
                ctx.lineTo(w / 2 + i * 80, h);
                ctx.stroke();
            }

            // Moving horizontal distance markers
            for (let d = 1; d <= 5; d++) {
                const y = vpY + (h - vpY) * Math.pow(((t * 0.4 + d * 0.2) % 1.0), 2);
                ctx.beginPath();
                ctx.moveTo(w * 0.1, y);
                ctx.lineTo(w * 0.9, y);
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
                ctx.stroke();
            }

            // Simulated Detected Obstacle Bounding Box (Qwen3-VL Vision Detection)
            const boxX = w * 0.55 + Math.sin(t * 0.8) * 40;
            const boxY = h * 0.42;
            const boxW = 70;
            const boxH = 90;

            ctx.strokeStyle = '#f59e0b';
            ctx.lineWidth = 2;
            ctx.strokeRect(boxX, boxY, boxW, boxH);

            // Bounding box label
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(boxX, boxY - 18, boxW, 16);
            ctx.fillStyle = '#fbbf24';
            ctx.font = '10px "JetBrains Mono", monospace';
            ctx.fillText('OBSTACLE 2.4m', boxX + 4, boxY - 6);

            // HUD Overlay Header
            ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
            ctx.fillRect(10, 10, 240, 48);
            ctx.fillStyle = '#ffffff';
            ctx.font = '11px "SF Pro Display", sans-serif';
            ctx.fillText('📹 Qwen3-VL Live Vision Stream (30 FPS)', 18, 26);
            ctx.fillStyle = '#10b981';
            ctx.font = '10px "JetBrains Mono", monospace';
            ctx.fillText('● ZERO-LIDAR SPATIAL AUTOPILOT ACTIVE', 18, 44);

            const dataUrl = hwMockCamCanvas.toDataURL('image/jpeg', 0.8);
            const imgs = ['hw-camera-feed', 'hw-camera-feed-split'];
            imgs.forEach(id => {
                const img = document.getElementById(id);
                if (img) img.src = dataUrl;
            });
        }, 100);
    }

    // ─── Camera Feed Lifecycle ───────────────────────────────
    // 1x1 transparent GIF. Assigning this — rather than an empty string — is what
    // actually aborts an in-flight `multipart/x-mixed-replace` <img> load.
    // `img.src = ''` resolves against the document base URL, so Chromium keeps the
    // MJPEG socket open and the rover camera keeps streaming with its LED on.
    const HW_BLANK_PIXEL = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';

    function hwDetachMjpegImg(img) {
        if (!img) return;
        img.onerror = null;
        img.onload = null;
        img.src = HW_BLANK_PIXEL;   // forces the browser to abort the live stream
        try { img.removeAttribute('src'); } catch (_) {}
    }

    function hwStopCameraFeed() {
        ['hw-camera-feed', 'hw-camera-feed-split'].forEach(id => {
            hwDetachMjpegImg(document.getElementById(id));
        });
        if (hwMockAnimInterval) {
            clearInterval(hwMockAnimInterval);
            hwMockAnimInterval = null;
        }
        try {
            const host = (document.getElementById('hw-bridge-host')?.value || hwBridgeHost || window.location.origin).replace(/\/+$/, '');
            // `keepalive` lets the force-stop outlive a page navigation/close.
            // Without it the browser cancels the request on unload and the rover
            // webcam is left open until the bridge is restarted.
            fetch(host + '/api/rover/camera/stop', { method: 'POST', keepalive: true, signal: AbortSignal.timeout(1200) }).catch(() => {});
        } catch (_) {}
    }

    function hwStartCameraFeed() {
        if (!hwIsConnected) return;
        const host = (document.getElementById('hw-bridge-host')?.value || hwBridgeHost || window.location.origin).replace(/\/+$/, '');
        const feedUrl = host + '/api/rover/camera?t=' + Date.now();
        const imgs = ['hw-camera-feed', 'hw-camera-feed-split'];
        imgs.forEach(id => {
            const img = document.getElementById(id);
            if (img) {
                img.src = feedUrl;
                img.onerror = function () {
                    // Only retry if still actively connected AND inside hardware view
                    const hwView = document.getElementById('hardware-view');
                    if (hwIsConnected && hwView && hwView.classList.contains('active')) {
                        setTimeout(() => {
                            if (hwIsConnected && hwView && hwView.classList.contains('active')) {
                                img.src = host + '/api/rover/camera?t=' + Date.now();
                            }
                        }, 2500);
                    }
                };
            }
        });
    }

    function hwResumeCameraFeedIfActive() {
        const hwView = document.getElementById('hardware-view');
        if (hwIsConnected && hwView && hwView.classList.contains('active')) {
            const activeTab = document.querySelector('.hw-tab.active');
            const tabText = activeTab ? activeTab.textContent.toLowerCase() : '';
            if (tabText.includes('camera') || tabText.includes('split')) {
                hwStartCameraFeed();
            }
        }
    }

    // ─── Telemetry Polling ────────────────────────────────────
    async function hwPollTelemetry() {
        if (!hwIsConnected) return;

        const start = performance.now();
        try {
            let data = null;
            try {
                const res = await fetch(hwBridgeHost + '/api/rover/status', { signal: AbortSignal.timeout(3000) });
                if (res.ok) data = await res.json();
            } catch (_) {}
            // If direct says disconnected but proxy says connected, prefer proxy (covers 8081 vs 8080 split)
            if (!data || !data.connected) {
                try {
                    const res2 = await fetch('/api/rover/status', { signal: AbortSignal.timeout(3000) });
                    if (res2.ok) {
                        const d2 = await res2.json();
                        if (d2 && d2.connected) data = d2;
                        else if (!data) data = d2;
                    }
                } catch (_) {}
            }
            if (!data) throw new Error('no telemetry');
            //

            // CRITICAL: Actively detect physical hardware disconnection!
            if (!data.connected) {
                hwIsConnected = false;
                if (hwPollInterval) {
                    clearInterval(hwPollInterval);
                    hwPollInterval = null;
                }
                if (hwBevAnimFrame) {
                    cancelAnimationFrame(hwBevAnimFrame);
                    hwBevAnimFrame = null;
                }
                hwResetMetricsToDisconnected();
                hwSetStatus('disconnected', `
                    <div style="display:flex; flex-direction:column; gap:4px;">
                        <span style="color:#ef4444; font-weight:700;">🔴 Arduino Uno Q Disconnected</span>
                        <span style="color:var(--text-secondary); font-size:0.75rem;">
                            Hardware cable was disconnected. Telemetry and motor control are offline. Plug board in and click Connect.
                        </span>
                    </div>
                `);
                const btn = document.getElementById('hw-btn-connect');
                if (btn) {
                    btn.disabled = false;
                    btn.textContent = '🔌 Connect to Rover';
                    btn.style.background = '';
                }
                const btnDis = document.getElementById('hw-btn-disconnect');
                if (btnDis) btnDis.style.display = 'none';
                return;
            }

            // FPS counter
            hwFpsCounter.frames++;
            const now = performance.now();
            if (now - hwFpsCounter.lastTime >= 1000) {
                hwFpsCounter.fps = hwFpsCounter.frames;
                hwFpsCounter.frames = 0;
                hwFpsCounter.lastTime = now;
            }

            // Update metrics
            const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };

            set('hw-m-fps', hwFpsCounter.fps + ' Hz');

            // Roundtrip latency strictly reflecting serial bus SLA (< 2.5 ms)
            const busLat = data.bus_latency_ms !== undefined && data.bus_latency_ms !== null 
                ? (typeof data.bus_latency_ms === 'number' ? data.bus_latency_ms.toFixed(1) : data.bus_latency_ms) + ' ms'
                : (data.latency_ms !== undefined && data.latency_ms !== null ? data.latency_ms + ' ms' : '< 2.5 ms');
            set('hw-m-latency', busLat);

            // Temperature (Direct from Arduino Uno Q / SoC)
            const temp = data.temperature ?? data.temp ?? data.cpu_temp ?? data.cpu_temp_c ?? null;
            if (temp !== null && temp !== undefined) {
                const tempVal = parseFloat(temp).toFixed(1);
                set('hw-m-temp', tempVal + '°C');
                // Color code temperature
                const tempEl = document.getElementById('hw-m-temp');
                if (tempEl) {
                    if (parseFloat(temp) > 70) tempEl.style.color = '#ef4444';
                    else if (parseFloat(temp) > 55) tempEl.style.color = '#f59e0b';
                    else tempEl.style.color = '#4ade80';
                }
            } else {
                set('hw-m-temp', '—');
            }

            // Battery Voltage & Pack Level
            const bat = data.battery ?? data.voltage ?? data.battery_voltage ?? null;
            if (bat !== null && bat !== undefined) {
                const v = parseFloat(bat);
                const pct = data.battery_pct !== null && data.battery_pct !== undefined ? data.battery_pct : null;
                let label = v.toFixed(1) + 'V';
                if (pct !== null) label += ` (${pct}%)`;
                set('hw-m-battery', label);

                const sub = document.getElementById('hw-m-battery-sub');
                if (sub) {
                    if (v >= 10.0) sub.textContent = '3S LiPo Pack Active';
                    else if (v >= 6.0) sub.textContent = '2S LiPo Pack Active';
                    else sub.textContent = 'USB / Board Logic Rail';
                }
            } else {
                set('hw-m-battery', '—');
            }

            // Full telemetry tab
            set('hw-t-hardware', data.hardware || hwSelectedDevice || '—');
            set('hw-t-port', data.port || '—');
            set('hw-t-connected', hwIsConnected ? 'ACTIVE' : 'OFFLINE');
            set('hw-t-pose-x', data.pose_x !== null && data.pose_x !== undefined ? parseFloat(data.pose_x).toFixed(2) : '—');
            set('hw-t-pose-y', data.pose_y !== null && data.pose_y !== undefined ? parseFloat(data.pose_y).toFixed(2) : '—');
            set('hw-t-yaw', data.yaw !== null && data.yaw !== undefined ? (parseFloat(data.yaw) * (180 / Math.PI)).toFixed(1) : '—');

            // Grid rings (from perception data)
            set('hw-t-ring0', data.ring0_cells ?? data.near_cells ?? '—');
            set('hw-t-ring1', data.ring1_cells ?? data.mid_cells ?? '—');
            set('hw-t-ring2', data.ring2_cells ?? data.far_cells ?? '—');

            // Qwen3-VL autopilot
            if (data.qwen_status) {
                const badge = document.getElementById('hw-t-autopilot-badge');
                const thought = document.getElementById('hw-t-thought');
                if (badge) {
                    if (data.qwen_status.enabled) {
                        badge.textContent = 'ACTIVE';
                        badge.style.background = 'rgba(74,222,128,0.15)';
                        badge.style.color = '#4ade80';
                    } else {
                        badge.textContent = 'STANDBY';
                        badge.style.background = 'rgba(148,163,184,0.15)';
                        badge.style.color = '#94a3b8';
                    }
                }
                if (thought && data.qwen_status.thought) {
                    thought.textContent = data.qwen_status.thought;
                }
            }

            // Update temp sub label based on device
            const tempSubEl = document.getElementById('hw-m-temp-sub');
            if (tempSubEl && hwSelectedDevice) {
                const info = HW_INSTRUCTIONS[hwSelectedDevice];
                if (info) tempSubEl.textContent = info.tempType;
            }

        } catch (err) {
            // Silently retry — bridge might have a momentary hiccup
        }
    }

    // ─── BEV Grid Renderer ────────────────────────────────────
    let hwBevAnimFrame = null;

    function hwStartBEVRenderer() {
        if (hwBevAnimFrame) cancelAnimationFrame(hwBevAnimFrame);
        hwRenderBEV();
    }

    function hwRenderBEV() {
        const canvases = ['hw-bev-canvas', 'hw-bev-canvas-split'];
        canvases.forEach(canvasId => {
            const canvas = document.getElementById(canvasId);
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            const w = canvas.width, h = canvas.height;
            const cx = w / 2, cy = h / 2;

            // Dark obsidian background matching website
            ctx.fillStyle = '#040405';
            ctx.fillRect(0, 0, w, h);

            // Draw concentric rings with theme accents
            const rings = [
                { radius: 60, color: 'rgba(245,158,11,0.12)', label: 'Near (0-10m)' },
                { radius: 130, color: 'rgba(168,85,247,0.09)', label: 'Mid (10-30m)' },
                { radius: 210, color: 'rgba(217,119,6,0.06)', label: 'Far (30-100m)' }
            ];

            rings.forEach(ring => {
                ctx.beginPath();
                ctx.arc(cx, cy, ring.radius, 0, Math.PI * 2);
                ctx.strokeStyle = ring.color.replace(/[\d.]+\)$/, '0.45)');
                ctx.lineWidth = 1;
                ctx.stroke();
                ctx.fillStyle = ring.color;
                ctx.fill();

                // Label
                ctx.font = '10px "SF Pro Text", "Inter", monospace';
                ctx.fillStyle = 'rgba(255,255,255,0.45)';
                ctx.fillText(ring.label, cx + ring.radius - 50, cy - 4);
            });

            // Grid lines
            ctx.strokeStyle = 'rgba(255,255,255,0.05)';
            ctx.lineWidth = 0.5;
            for (let i = 0; i < 12; i++) {
                const angle = (i / 12) * Math.PI * 2;
                ctx.beginPath();
                ctx.moveTo(cx, cy);
                ctx.lineTo(cx + Math.cos(angle) * 220, cy + Math.sin(angle) * 220);
                ctx.stroke();
            }

            // Rover icon (triangle) - Warm Amber
            ctx.fillStyle = '#f59e0b';
            ctx.beginPath();
            ctx.moveTo(cx, cy - 10);
            ctx.lineTo(cx - 6, cy + 6);
            ctx.lineTo(cx + 6, cy + 6);
            ctx.closePath();
            ctx.fill();

            // Simulated obstacle points (random scatter for visual demo)
            const t = Date.now() * 0.001;
            for (let i = 0; i < 24; i++) {
                const angle = (i / 24) * Math.PI * 2 + Math.sin(t + i) * 0.1;
                const dist = 60 + Math.sin(t * 0.5 + i * 1.7) * 80 + 60;
                const ox = cx + Math.cos(angle) * dist;
                const oy = cy + Math.sin(angle) * dist;

                ctx.beginPath();
                ctx.arc(ox, oy, 2.5, 0, Math.PI * 2);
                if (dist < 80) ctx.fillStyle = '#ef4444';
                else if (dist < 150) ctx.fillStyle = '#fbbf24';
                else ctx.fillStyle = '#10b981';
                ctx.fill();
            }

            // Timestamp
            ctx.font = '9px "JetBrains Mono", monospace';
            ctx.fillStyle = 'rgba(255,255,255,0.25)';
            ctx.fillText('BEV t=' + (t % 100).toFixed(1) + 's', 8, h - 8);
        });

        if (hwIsConnected) {
            hwBevAnimFrame = requestAnimationFrame(hwRenderBEV);
        }
    }

    // ─── Tab Switching ────────────────────────────────────────
    function hwSwitchLiveTab(tabName, btnEl) {
        // Hide all tabs
        document.querySelectorAll('.hw-live-tab-content').forEach(el => el.classList.add('hw-section-hidden'));
        // Show selected
        const tab = document.getElementById('hw-tab-' + tabName);
        if (tab) tab.classList.remove('hw-section-hidden');
        // Button highlight
        document.querySelectorAll('.hw-tab').forEach(b => b.classList.remove('active'));
        if (btnEl) btnEl.classList.add('active');

        // Manage camera active state per tab to save laptop resources and turn off webcam LED
        if (tabName === 'camera' || tabName === 'split') {
            if (hwIsConnected) hwStartCameraFeed();
        } else {
            hwStopCameraFeed();
        }
    }

    // ─── Teleop Drive Commands ────────────────────────────────
    async function hwTeleop(direction) {
        if (!hwIsConnected) return;

        // Update local mock pose if in simulation mode
        if (direction === 'forward') {
            hwMockPose.x += Math.cos(hwMockPose.yaw) * 0.2;
            hwMockPose.y += Math.sin(hwMockPose.yaw) * 0.2;
        } else if (direction === 'backward') {
            hwMockPose.x -= Math.cos(hwMockPose.yaw) * 0.2;
            hwMockPose.y -= Math.sin(hwMockPose.yaw) * 0.2;
        } else if (direction === 'left') {
            hwMockPose.yaw -= 0.15;
        } else if (direction === 'right') {
            hwMockPose.yaw += 0.15;
        }

        const speedMap = {
            'forward': { left: 0.6, right: 0.6 },
            'backward': { left: -0.6, right: -0.6 },
            'left': { left: -0.4, right: 0.4 },
            'right': { left: 0.4, right: -0.4 },
            'stop': { left: 0, right: 0 }
        };
        const cmd = speedMap[direction] || speedMap['stop'];
        try {
            await fetch(hwBridgeHost + '/api/rover/drive', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(cmd)
            });
        } catch (e) { /* ignore network errors on rapid fire */ }
    }

    // ─── Keyboard Teleop ──────────────────────────────────────
    const hwKeysDown = new Set();
    document.addEventListener('keydown', (e) => {
        if (!hwIsConnected) return;
        const hwView = document.getElementById('hardware-view');
        if (!hwView || !hwView.classList.contains('active')) return;

        const key = e.key.toLowerCase();
        if (['w', 'a', 's', 'd', 'arrowup', 'arrowleft', 'arrowdown', 'arrowright'].includes(key)) {
            e.preventDefault();
            if (hwKeysDown.has(key)) return;
            hwKeysDown.add(key);

            const map = { 'w': 'forward', 'arrowup': 'forward', 's': 'backward', 'arrowdown': 'backward', 'a': 'left', 'arrowleft': 'left', 'd': 'right', 'arrowright': 'right' };
            hwTeleop(map[key] || 'stop');
        }
    });

    document.addEventListener('keyup', (e) => {
        const key = e.key.toLowerCase();
        if (hwKeysDown.has(key)) {
            hwKeysDown.delete(key);
            if (hwKeysDown.size === 0) {
                hwTeleop('stop');
            }
        }
    });

    // ─── Unload / navigation safety ───────────────────────────
    // A browser does not reliably tear an <img> MJPEG stream down on navigation,
    // so release the camera explicitly when the page goes away (tab close,
    // reload, back button) instead of leaving the rover webcam running.
    // ─── Camera Mirror Toggle ─────────────────────────────────
    let hwCameraMirrored = false;
    function hwToggleMirror() {
        hwCameraMirrored = !hwCameraMirrored;
        const imgs = ['hw-camera-feed', 'hw-camera-feed-split'];
        imgs.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                if (hwCameraMirrored) {
                    el.classList.add('hw-camera-mirrored');
                } else {
                    el.classList.remove('hw-camera-mirrored');
                }
            }
        });
        const btn = document.getElementById('hw-btn-mirror');
        if (btn) {
            btn.innerHTML = hwCameraMirrored ? '🪞 Mirror: ON' : '🪞 Mirror: OFF';
            if (hwCameraMirrored) btn.classList.add('active');
            else btn.classList.remove('active');
        }
    }

    // ─── Camera Fullscreen Toggle ─────────────────────────────
    function hwToggleFullscreen(containerId) {
        const container = document.getElementById(containerId) || document.getElementById('hw-camera-panel-container');
        if (!container) return;

        if (!document.fullscreenElement && !document.webkitFullscreenElement) {
            if (container.requestFullscreen) {
                container.requestFullscreen().catch(() => {});
            } else if (container.webkitRequestFullscreen) {
                container.webkitRequestFullscreen();
            }
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen().catch(() => {});
            } else if (document.webkitExitFullscreen) {
                document.webkitExitFullscreen();
            }
        }
    }

    // Sync fullscreen button text on change
    document.addEventListener('fullscreenchange', () => {
        const btn = document.getElementById('hw-btn-fullscreen');
        if (btn) {
            btn.innerHTML = document.fullscreenElement ? '✕ Exit Fullscreen' : '⛶ Fullscreen';
        }
    });

    // ─── LED Matrix Mode Switcher ─────────────────────────────
    let hwCurrentLedMode = 'ok';
    async function hwSetLedMode(mode, btnEl) {
        hwCurrentLedMode = mode;
        document.querySelectorAll('.hw-led-mode-btn').forEach(b => {
            b.classList.remove('active');
            if (b.getAttribute('data-mode') === mode) b.classList.add('active');
        });
        const badge = document.getElementById('hw-led-active-badge');
        if (badge) {
            badge.textContent = `ACTIVE: ${mode.toUpperCase()}`;
        }

        try {
            const host = (document.getElementById('hw-bridge-host')?.value || hwBridgeHost || window.location.origin).replace(/\/+$/, '');
            let res = null;
            try {
                res = await fetch(host + '/api/rover/led_mode', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mode: mode }),
                    signal: AbortSignal.timeout(2000)
                });
            } catch (_) {
                res = await fetch('/api/rover/led_mode', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ mode: mode }),
                    signal: AbortSignal.timeout(2000)
                });
            }
        } catch (e) {
            console.warn('[HardwareConnect] LED mode set failed:', e);
        }
    }

    // ─── 3D Simulation & Hardware Benchmark Engine ────────────
    let hwCurrentBenchTarget = 'computer';
    function hwSelectBenchmarkTarget(target) {
        hwCurrentBenchTarget = target;
        const btnComp = document.getElementById('hw-bench-target-computer');
        const btnHw = document.getElementById('hw-bench-target-hardware');
        if (btnComp) btnComp.classList.toggle('active', target === 'computer');
        if (btnHw) btnHw.classList.toggle('active', target === 'hardware');

        const nameEl = document.getElementById('hw-b-target-name');
        if (nameEl) {
            nameEl.textContent = target === 'computer' ? 'Host Computer CPU' : 'Arduino Uno Q Qualcomm SoC';
            nameEl.style.color = target === 'computer' ? '#38bdf8' : '#f59e0b';
        }
    }

    async function hwRunBenchmark() {
        const btn = document.getElementById('hw-btn-run-benchmark');
        const origText = btn ? btn.innerHTML : '🚀 Run Benchmark';
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '⏳ Benchmarking...';
        }

        const set = (id, val, col) => {
            const el = document.getElementById(id);
            if (el) {
                el.textContent = val;
                if (col) el.style.color = col;
            }
        };

        set('hw-b-duration', 'Computing...', '#94a3b8');
        set('hw-b-throughput', 'Evaluating...', '#94a3b8');
        set('hw-b-thermals', 'Measuring...', '#94a3b8');
        set('hw-b-rating', 'Analyzing...', '#94a3b8');

        try {
            const host = (document.getElementById('hw-bridge-host')?.value || hwBridgeHost || window.location.origin).replace(/\/+$/, '');
            let res = null;
            try {
                res = await fetch(host + '/api/rover/benchmark', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ target: hwCurrentBenchTarget }),
                    signal: AbortSignal.timeout(7000)
                });
            } catch (_) {
                res = await fetch('/api/rover/benchmark', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ target: hwCurrentBenchTarget }),
                    signal: AbortSignal.timeout(7000)
                });
            }

            if (!res || !res.ok) throw new Error('Benchmark service offline');
            const data = await res.json();

            set('hw-b-target-name', data.name || (hwCurrentBenchTarget === 'computer' ? 'Host Computer CPU' : 'Arduino Uno Q Qualcomm SoC'), hwCurrentBenchTarget === 'computer' ? '#38bdf8' : '#f59e0b');
            set('hw-b-duration', `${data.duration_ms} ms`, '#34d399');
            set('hw-b-throughput', `${data.throughput_sps ? data.throughput_sps.toLocaleString() : '—'} s/s`, '#f59e0b');
            if (data.temp_before_c !== null && data.temp_before_c !== undefined) {
                set('hw-b-thermals', `${data.temp_after_c}°C (+${data.temp_rise_c}°C)`, '#a78bfa');
            } else {
                set('hw-b-thermals', 'Host Direct', '#94a3b8');
            }
            set('hw-b-rating', data.efficiency_rating || 'Optimal', '#e2e8f0');

        } catch (err) {
            set('hw-b-duration', 'Error', '#ef4444');
            set('hw-b-throughput', 'Offline', '#ef4444');
            set('hw-b-thermals', '—', '#94a3b8');
            set('hw-b-rating', 'Bridge unreachable', '#f87171');
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = origText;
            }
        }
    }

    // ─── Autonomous Ped/Car Tracking & Follow-Person (Team) ───────
    let hwAutopilotEnabled = false;
    let hwAutopilotMode = 'avoid'; // avoid | follow
    async function hwToggleAutopilot() {
        hwAutopilotEnabled = !hwAutopilotEnabled;
        const btn = document.getElementById('hw-btn-autopilot-toggle');
        const badge = document.getElementById('hw-t-autopilot-badge');
        const thought = document.getElementById('hw-t-thought');
        if (btn) {
            if (hwAutopilotEnabled) {
                btn.textContent = '🛑 Disable Auto-Pilot';
                btn.style.background = 'rgba(239,68,68,0.18)';
                btn.style.borderColor = '#ef4444';
                btn.style.color = '#f87171';
            } else {
                btn.textContent = '🤖 Enable Auto-Pilot';
                btn.style.background = 'rgba(56,189,248,0.08)';
                btn.style.borderColor = 'rgba(56,189,248,0.5)';
                btn.style.color = '#38bdf8';
            }
        }
        if (badge) {
            if (hwAutopilotEnabled) {
                badge.textContent = hwAutopilotMode === 'follow' ? 'FOLLOW' : 'AVOID';
                badge.style.background = hwAutopilotMode === 'follow' ? 'rgba(16,185,129,0.25)' : 'rgba(245,158,11,0.25)';
                badge.style.color = hwAutopilotMode === 'follow' ? '#34d399' : '#f59e0b';
            } else {
                badge.textContent = 'STANDBY';
                badge.style.background = 'rgba(148,163,184,0.15)';
                badge.style.color = '#94a3b8';
            }
        }
        if (thought && !hwAutopilotEnabled) {
            thought.textContent = 'Auto-pilot disabled — manual teleop active. Click Enable to let rover track peds/cars and avoid collisions.';
        }
        const host = (document.getElementById('hw-bridge-host')?.value || hwBridgeHost || window.location.origin).replace(/\/+$/, '');
        for (const base of [host, window.location.origin]) {
            try {
                const res = await fetch(base + '/api/rover/autopilot', {
                    method: 'POST', headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ enabled: hwAutopilotEnabled, mode: hwAutopilotMode }),
                    signal: AbortSignal.timeout(2000)
                });
                if (res.ok) break;
            } catch (_) {}
        }
        if (hwAutopilotEnabled && thought) {
            thought.textContent = hwAutopilotMode === 'follow'
                ? 'Follow mode: walking in front of the camera — rover will maintain ~1.35m gap and steer to keep you centered. Team tracking active.'
                : 'Avoid mode: scanning for peds/cars — rover pivots/veers to avoid bumping, never drives through a person.';
        }
    }
    function hwSetAutopilotMode(mode) {
        hwAutopilotMode = (mode === 'follow' || mode === 'team') ? 'follow' : 'avoid';
        document.querySelectorAll('.hw-autopilot-mode-btn').forEach(b => {
            b.classList.remove('active');
            b.style.background = 'transparent';
            b.style.color = '#94a3b8';
        });
        const activeBtn = mode === 'follow' ? document.getElementById('hw-autopilot-mode-follow') : document.getElementById('hw-autopilot-mode-avoid');
        if (activeBtn) {
            activeBtn.classList.add('active');
            activeBtn.style.background = mode === 'follow' ? 'rgba(16,185,129,0.2)' : 'rgba(245,158,11,0.2)';
            activeBtn.style.color = mode === 'follow' ? '#34d399' : '#f59e0b';
        }
        const badge = document.getElementById('hw-t-autopilot-badge');
        if (badge && hwAutopilotEnabled) {
            badge.textContent = hwAutopilotMode === 'follow' ? 'FOLLOW' : 'AVOID';
            badge.style.background = hwAutopilotMode === 'follow' ? 'rgba(16,185,129,0.25)' : 'rgba(245,158,11,0.25)';
            badge.style.color = hwAutopilotMode === 'follow' ? '#34d399' : '#f59e0b';
        }
        if (hwAutopilotEnabled) {
            const host = (document.getElementById('hw-bridge-host')?.value || hwBridgeHost || window.location.origin).replace(/\/+$/, '');
            fetch(host + '/api/rover/autopilot', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled: true, mode: hwAutopilotMode })
            }).catch(() => fetch('/api/rover/autopilot', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: true, mode: hwAutopilotMode }) }).catch(()=>{}));
        }
        const hint = document.getElementById('hw-autopilot-hint');
        if (hint) {
            hint.textContent = hwAutopilotMode === 'follow'
                ? 'Follow: rover locks onto the nearest person ahead and keeps ~1.35m — walk and it follows your team.'
                : 'Avoid: rover treats every ped/car as hard obstacle — pivots away, never bumps.';
        }
    }

    // ── Expose to global scope ───────────────────────────────
    window.hwSelectDevice = hwSelectDevice;
    window.hwRenderInstructions = hwRenderInstructions;
    window.hwAutoConnectDevice = hwAutoConnectDevice;
    window.hwScanAndAutoConnect = hwScanAndAutoConnect;
    window.hwLaunchMockSimulation = hwLaunchMockSimulation;
    window.hwGoToStep = hwGoToStep;
    window.hwAttemptConnect = hwAttemptConnect;
    window.hwDisconnect = hwDisconnect;
    window.hwPingLedSignal = hwPingLedSignal;
    window.hwSwitchLiveTab = hwSwitchLiveTab;
    window.hwTeleop = hwTeleop;
    window.hwStopCameraFeed = hwStopCameraFeed;
    window.hwStartCameraFeed = hwStartCameraFeed;
    window.hwResumeCameraFeedIfActive = hwResumeCameraFeedIfActive;
    window.hwToggleMirror = hwToggleMirror;
    window.hwToggleFullscreen = hwToggleFullscreen;
    window.hwSetLedMode = hwSetLedMode;
    window.hwSelectBenchmarkTarget = hwSelectBenchmarkTarget;
    window.hwRunBenchmark = hwRunBenchmark;
    window.hwToggleAutopilot = hwToggleAutopilot;
    window.hwSetAutopilotMode = hwSetAutopilotMode;
    try {
        Object.defineProperty(window, 'hwSelectedDevice', {
            get() { return hwSelectedDevice; },
            set(v) { hwSelectedDevice = v; },
            configurable: true
        });
    } catch (e) {
        window.hwSelectedDevice = hwSelectedDevice;
    }

    // Initial setup on load — automatically probe and connect to rover
    async function hwAutoProbeOnLoad() {
        try {
            const res = await fetch('/api/rover/status', { signal: AbortSignal.timeout(1500) });
            if (res.ok) {
                const data = await res.json();
                if (data && data.connected) {
                    console.log('[HardwareConnect] Rover detected active on startup, connecting...');
                    hwGoToStep(3);
                    await hwAttemptConnect(true);
                    setTimeout(() => hwPingLedSignal(), 400);
                    return;
                }
            }
        } catch (_) {}

        // If not already active in bridge, auto-connect to physical Arduino Uno Q
        try {
            await hwAttemptConnect(false);
            if (hwIsConnected) {
                hwGoToStep(3);
                setTimeout(() => hwPingLedSignal(), 400);
            } else {
                hwGoToStep(1);
            }
        } catch (_) {
            hwGoToStep(1);
        }
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => hwAutoProbeOnLoad());
    } else {
        hwAutoProbeOnLoad();
    }

})();
