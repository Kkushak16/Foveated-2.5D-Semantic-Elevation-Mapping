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
                // Auto-start live simulation mode so user immediately sees video, BEV grid & telemetry
                hwLaunchMockSimulation();
            }
        }
    }

    // ─── Instructions Renderer ────────────────────────────────
    function hwRenderInstructions(deviceId) {
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

    // ─── Primary Connection Handler ───────────────────────────
    async function hwAttemptConnect() {
        hwBridgeHost = (document.getElementById('hw-bridge-host')?.value || 'http://localhost:8081').replace(/\/+$/, '');
        const btn = document.getElementById('hw-btn-connect');
        if (btn) {
            btn.disabled = true;
            btn.textContent = '⏳ Probing Rover...';
        }

        hwSetStatus('connecting', 'Probing Python bridge at ' + hwBridgeHost + ' (Auto-detecting Arduino Uno Q / ESP32 / Pi)...');

        try {
            const res = await fetch(hwBridgeHost + '/api/rover/status', { signal: AbortSignal.timeout(3500) });
            if (!res.ok) throw new Error('HTTP ' + res.status);
            const data = await res.json();

            hwIsConnected = true;
            const detectedMcu = data.hardware || hwSelectedDevice || 'Wave Rover (Arduino Uno Q)';
            hwSetStatus('connected', `Connected ✓ — <strong>${detectedMcu}</strong> active on ${data.port || 'Auto Serial'}`);

            // Show live dashboard
            const dash = document.getElementById('hw-live-dashboard');
            if (dash) dash.style.display = 'block';

            // Start camera feed
            hwStartCameraFeed();

            // Start telemetry polling
            if (hwPollInterval) clearInterval(hwPollInterval);
            hwPollInterval = setInterval(hwPollTelemetry, 500);

            // Start BEV rendering
            hwStartBEVRenderer();

            if (btn) {
                btn.disabled = false;
                btn.textContent = '✓ Connected';
                btn.style.background = 'linear-gradient(180deg, #10b981 0%, #059669 100%)';
            }

        } catch (err) {
            hwIsConnected = false;
            hwSetStatus('disconnected', `
                <div style="display:flex; flex-direction:column; gap:6px;">
                    <span>⚠️ Bridge not detected at <code>${hwBridgeHost}</code>.</span>
                    <span style="color:var(--text-secondary); font-size:0.75rem;">
                        To connect physical hardware, run: <code>python python/waverover_bridge.py</code><br>
                        Or test immediately with the simulated rover:
                    </span>
                    <div>
                        <button class="hw-btn-primary" style="padding:5px 12px; font-size:0.75rem; margin-top:4px;" onclick="hwLaunchMockSimulation()">
                            🧪 Launch Simulation Mode
                        </button>
                    </div>
                </div>
            `);
            if (btn) {
                btn.disabled = false;
                btn.textContent = '🔌 Retry Connection';
                btn.style.background = 'linear-gradient(180deg, #f59e0b 0%, #d97706 100%)';
            }
        }
    }

    // ─── Mock Hardware Simulation Mode ────────────────────────
    let hwMockAnimInterval = null;
    let hwMockPose = { x: 0.0, y: 0.0, yaw: 0.0 };

    function hwLaunchMockSimulation() {
        hwIsConnected = true;
        hwSelectedDevice = hwSelectedDevice || 'arduino-uno-q';

        hwSetStatus('connected', `Connected ✓ — <strong>Arduino Uno Q (Simulation Mode)</strong> • Port: Virtual COM3`);

        const dash = document.getElementById('hw-live-dashboard');
        if (dash) dash.style.display = 'block';

        const btn = document.getElementById('hw-btn-connect');
        if (btn) {
            btn.textContent = '✓ Simulated Live';
            btn.style.background = 'linear-gradient(180deg, #10b981 0%, #059669 100%)';
        }

        // Start mock camera drawing loop
        hwStartMockCameraFeed();

        // Start mock telemetry loop
        if (hwPollInterval) clearInterval(hwPollInterval);
        hwPollInterval = setInterval(hwPollMockTelemetry, 500);

        // Start BEV rendering
        hwStartBEVRenderer();
    }

    function hwPollMockTelemetry() {
        if (!hwIsConnected) return;

        hwFpsCounter.fps = 30;
        hwLatency = Math.floor(2 + Math.random() * 3);

        const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
        set('hw-m-fps', hwFpsCounter.fps + ' FPS');
        set('hw-m-latency', hwLatency + ' ms');
        set('hw-m-temp', (42.0 + Math.sin(Date.now() * 0.001) * 1.5).toFixed(1) + '°C');
        set('hw-m-battery', '12.2 V');

        set('hw-t-hardware', (hwSelectedDevice || 'arduino-uno-q').toUpperCase() + ' (SIM)');
        set('hw-t-port', 'Virtual COM3 @ 115200');
        set('hw-t-connected', 'ACTIVE (SIM)');
        set('hw-t-pose-x', hwMockPose.x.toFixed(2));
        set('hw-t-pose-y', hwMockPose.y.toFixed(2));
        set('hw-t-yaw', (hwMockPose.yaw * (180 / Math.PI)).toFixed(1));

        set('hw-t-ring0', '1,420');
        set('hw-t-ring1', '3,840');
        set('hw-t-ring2', '8,190');

        const badge = document.getElementById('hw-t-autopilot-badge');
        const thought = document.getElementById('hw-t-thought');
        if (badge) {
            badge.textContent = 'ACTIVE (QWEN3-VL)';
            badge.style.background = 'rgba(16, 185, 129, 0.2)';
            badge.style.color = '#34d399';
        }
        if (thought) {
            const thoughts = [
                "Clear corridor detected ahead. Distance: 3.4m. Maintaining heading 0.0°.",
                "Left obstacle flagged at 1.8m. Applying gentle right yaw adjustment.",
                "3-Ring spatial map refreshed. Near safety margin: OK. Tracking waypoint.",
                "Optical flow motion gating confirmed static environment. Compute savings: 79.1%."
            ];
            const idx = Math.floor((Date.now() / 3000) % thoughts.length);
            thought.textContent = thoughts[idx];
        }
    }

    // ─── Mock Camera Frame Generator ──────────────────────────
    let hwMockCamCanvas = null;

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

    // ─── Camera Feed ──────────────────────────────────────────
    function hwStartCameraFeed() {
        const feedUrl = hwBridgeHost + '/api/rover/camera?t=' + Date.now();
        const imgs = ['hw-camera-feed', 'hw-camera-feed-split'];
        imgs.forEach(id => {
            const img = document.getElementById(id);
            if (img) {
                img.src = feedUrl;
                img.onerror = function () {
                    // Retry after delay
                    setTimeout(() => {
                        img.src = hwBridgeHost + '/api/rover/camera?t=' + Date.now();
                    }, 2000);
                };
            }
        });
    }

    // ─── Telemetry Polling ────────────────────────────────────
    async function hwPollTelemetry() {
        if (!hwIsConnected) return;

        const start = performance.now();
        try {
            const res = await fetch(hwBridgeHost + '/api/rover/status', { signal: AbortSignal.timeout(3000) });
            const data = await res.json();
            hwLatency = Math.round(performance.now() - start);

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
            set('hw-m-latency', hwLatency + ' ms');

            // Temperature
            const temp = data.temperature ?? data.temp ?? data.cpu_temp ?? null;
            if (temp !== null) {
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

            // Battery
            const bat = data.battery ?? data.voltage ?? null;
            if (bat !== null) {
                set('hw-m-battery', parseFloat(bat).toFixed(1) + 'V');
            }

            // Full telemetry tab
            set('hw-t-hardware', data.hardware || hwSelectedDevice || '—');
            set('hw-t-port', data.port || '—');
            set('hw-t-connected', hwIsConnected ? 'ACTIVE' : 'OFFLINE');
            set('hw-t-pose-x', (data.pose_x ?? data.x ?? 0).toFixed(2));
            set('hw-t-pose-y', (data.pose_y ?? data.y ?? 0).toFixed(2));
            set('hw-t-yaw', ((data.yaw ?? data.heading ?? 0) * (180 / Math.PI)).toFixed(1));

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

    // ─── Expose to global scope ───────────────────────────────
    window.hwSelectDevice = hwSelectDevice;
    window.hwRenderInstructions = hwRenderInstructions;
    window.hwAutoConnectDevice = hwAutoConnectDevice;
    window.hwScanAndAutoConnect = hwScanAndAutoConnect;
    window.hwLaunchMockSimulation = hwLaunchMockSimulation;
    window.hwGoToStep = hwGoToStep;
    window.hwAttemptConnect = hwAttemptConnect;
    window.hwSwitchLiveTab = hwSwitchLiveTab;
    window.hwTeleop = hwTeleop;
    window.hwSelectedDevice = hwSelectedDevice;

    // Initial setup on load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => hwGoToStep(1));
    } else {
        hwGoToStep(1);
    }

})();
