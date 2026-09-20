const http = require('http');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

const ROOT = path.resolve(__dirname, '..');
const UI_DIR = path.join(ROOT, 'web', 'ui');
const DOCS_IMG_DIR = path.join(ROOT, 'docs', 'images');
const UI_PORT = 8188;
const BRIDGE_PORT = 8081;
const CDP_PORT = 9333;

if (!fs.existsSync(DOCS_IMG_DIR)) {
  fs.mkdirSync(DOCS_IMG_DIR, { recursive: true });
}

const DEMO_FRAME = fs.existsSync(path.join(ROOT, 'results', 'demo_output.jpg'))
  ? fs.readFileSync(path.join(ROOT, 'results', 'demo_output.jpg'))
  : Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg==',
      'base64'
    );

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

function startMockBridge() {
  const server = http.createServer((req, res) => {
    const url = req.url.split('?')[0];
    if (req.method === 'OPTIONS') {
      res.writeHead(204, CORS);
      res.end();
      return;
    }
    const json = (obj, code = 200) => {
      res.writeHead(code, { 'Content-Type': 'application/json', ...CORS });
      res.end(JSON.stringify(obj));
    };

    if (url === '/api/rover/camera' || url === '/video_feed') {
      res.writeHead(200, {
        'Content-Type': 'multipart/x-mixed-replace; boundary=FRAME',
        'Cache-Control': 'no-cache, private',
        ...CORS,
      });
      const timer = setInterval(() => {
        if (res.writableEnded) return;
        try {
          res.write('--FRAME\r\nContent-Type: image/jpeg\r\nContent-Length: ' + DEMO_FRAME.length + '\r\n\r\n');
          res.write(DEMO_FRAME);
          res.write('\r\n');
        } catch (_) {
          clearInterval(timer);
        }
      }, 50);
      req.on('close', () => clearInterval(timer));
      return;
    }

    if (url === '/api/rover/status') {
      json({
        hardware: 'Arduino Uno Q',
        connected: true,
        port: 'COM3',
        status: 'CONNECTED',
        temperature: 41.5,
        cpu_temp: 41.5,
        battery: 12.1,
        voltage: 12.1,
        bus_latency_ms: 2.1,
        pose: { x: 0, y: 0, yaw_deg: 0 },
        pose_x: 0,
        pose_y: 0,
        yaw: 0,
      });
      return;
    }

    if (url === '/api/rover/ping_led' || url === '/api/rover/connect' || url === '/api/rover/drive' || url === '/api/rover/teleop') {
      json({ status: 'ok', connected: true });
      return;
    }

    if (url === '/api/rover/autopilot') {
      json({ autopilot_active: false, status: 'ok' });
      return;
    }

    if (url === '/api/grid/latest') {
      json({ rings: [3, 5, 1], total_points: 180, ego_pose: { x: 0, y: 0, yaw: 0 } });
      return;
    }

    json({ ok: true });
  });

  return new Promise((resolve) => {
    server.listen(BRIDGE_PORT, '127.0.0.1', () => resolve(server));
  });
}

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
};

function startUiServer() {
  const server = http.createServer((req, res) => {
    const url = req.url.split('?')[0];

    if (url.startsWith('/api/rover/') || url === '/video_feed') {
      const proxyReq = http.request(
        {
          host: '127.0.0.1',
          port: BRIDGE_PORT,
          path: req.url,
          method: req.method,
          headers: { ...req.headers, host: `127.0.0.1:${BRIDGE_PORT}` },
        },
        (proxyRes) => {
          res.writeHead(proxyRes.statusCode, proxyRes.headers);
          proxyRes.pipe(res);
        }
      );
      proxyReq.on('error', () => {
        if (!res.headersSent) {
          res.writeHead(502, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: true, connected: false }));
        }
      });
      res.on('close', () => proxyReq.destroy());
      req.pipe(proxyReq);
      return;
    }

    if (url === '/api/telemetry' || url === '/yolo-status.json') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true, platform: 'laptop', fps: 45.8, latency: 1.01 }));
      return;
    }

    const rel = url === '/' ? '/index.html' : url;
    const file = path.join(UI_DIR, path.normalize(rel).replace(/^([/\\])+/, ''));
    if (!file.startsWith(UI_DIR)) {
      res.writeHead(403);
      res.end('forbidden');
      return;
    }
    fs.readFile(file, (err, content) => {
      if (err) {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('not found');
        return;
      }
      res.writeHead(200, { 'Content-Type': MIME[path.extname(file).toLowerCase()] || 'application/octet-stream' });
      res.end(content);
    });
  });

  return new Promise((resolve) => {
    server.listen(UI_PORT, '127.0.0.1', () => resolve(server));
  });
}

function findChrome() {
  const candidates = [
    process.env.CHROME_PATH,
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
  ].filter(Boolean);
  const found = candidates.find((p) => fs.existsSync(p));
  if (!found) throw new Error('No Chrome/Edge binary found.');
  return found;
}

class CDP {
  constructor(ws) {
    this.ws = ws;
    this.nextId = 0;
    this.pending = new Map();
    ws.addEventListener('message', (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch (_) { return; }
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(`${msg.error.message}`));
        else resolve(msg.result);
      }
    });
  }

  send(method, params = {}) {
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async eval(expression) {
    const r = await this.send('Runtime.evaluate', {
      expression,
      returnByValue: true,
      awaitPromise: true,
    });
    if (r.exceptionDetails) {
      throw new Error(`Eval error: ${JSON.stringify(r.exceptionDetails)}`);
    }
    return r.result?.value;
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function captureScreenshot(cdp, outputPath) {
  const { data } = await cdp.send('Page.captureScreenshot', {
    format: 'png',
    fromSurface: true,
    captureBeyondViewport: false,
  });
  fs.writeFileSync(outputPath, Buffer.from(data, 'base64'));
  console.log(`[SAVED] ${outputPath} (${(fs.statSync(outputPath).size / 1024).toFixed(1)} KB)`);
}

async function run() {
  console.log('--- Starting servers ---');
  const bridgeServer = await startMockBridge();
  const uiServer = await startUiServer();
  console.log(`Bridge on http://127.0.0.1:${BRIDGE_PORT}, UI on http://127.0.0.1:${UI_PORT}`);

  const chromeBin = findChrome();
  const profileDir = fs.mkdtempSync(path.join(os.tmpdir(), 'capture-chrome-'));
  console.log('Launching browser:', chromeBin);

  const chromeProc = spawn(
    chromeBin,
    [
      '--headless=new',
      '--disable-gpu',
      '--no-first-run',
      '--no-default-browser-check',
      '--disable-extensions',
      '--window-size=1920,1080',
      '--force-device-scale-factor=1',
      '--enable-unsafe-swiftshader',
      '--use-angle=swiftshader',
      `--remote-debugging-port=${CDP_PORT}`,
      `--user-data-dir=${profileDir}`,
      'about:blank',
    ],
    { stdio: 'ignore' }
  );

  let wsUrl = null;
  for (let i = 0; i < 80; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${CDP_PORT}/json/list`);
      const targets = await res.json();
      const page = targets.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
      if (page) {
        wsUrl = page.webSocketDebuggerUrl;
        break;
      }
    } catch (_) {}
    await sleep(100);
  }

  if (!wsUrl) {
    throw new Error('Failed to connect to Chrome DevTools port');
  }

  console.log('Connecting DevTools WebSocket...');
  const ws = new WebSocket(wsUrl);
  await new Promise((resolve) => ws.addEventListener('open', resolve, { once: true }));
  const cdp = new CDP(ws);

  await cdp.send('Page.enable');
  await cdp.send('Runtime.enable');
  await cdp.send('Emulation.setDeviceMetricsOverride', {
    width: 1920,
    height: 1080,
    deviceScaleFactor: 1,
    mobile: false,
  });

  console.log('Navigating to UI...');
  await cdp.send('Page.navigate', { url: `http://127.0.0.1:${UI_PORT}/` });
  await sleep(2500);

  // 1. Landing Page Hero
  console.log('Capturing Landing Page Hero...');
  await cdp.eval(`
    if (typeof showPageView === 'function') showPageView('landing');
    window.scrollTo(0, 0);
  `);
  await sleep(1500);
  await captureScreenshot(cdp, path.join(DOCS_IMG_DIR, 'landing_hero.png'));

  // 2. Hardware Connect View - Step 1/2 Selection & Pinouts
  console.log('Capturing Hardware Step 1 Selection...');
  await cdp.eval(`
    if (typeof showPageView === 'function') showPageView('hardware');
    if (typeof hwGoToStep === 'function') hwGoToStep(1);
    const qCard = document.querySelector('.hw-device-card[data-device="arduino-uno-q"]') || document.querySelector('.hw-device-card');
    if (qCard) qCard.click();
    const pills = document.querySelector('.hw-step-pills') || document.querySelector('#hw-step-1');
    if (pills) pills.scrollIntoView({ behavior: 'instant', block: 'start' });
  `);
  await sleep(1500);
  await captureScreenshot(cdp, path.join(DOCS_IMG_DIR, 'hardware_setup_wizard.png'));

  // 3. Hardware Connect View - Step 3 Live Teleoperation Cockpit
  console.log('Capturing Hardware Step 3 Live Teleoperation Cockpit...');
  await cdp.eval(`
    if (typeof hwGoToStep === 'function') hwGoToStep(3);
    const connBtn = document.querySelector('#hw-btn-connect');
    if (connBtn) connBtn.click();
    setTimeout(() => {
      const splitTab = document.querySelectorAll('.hw-tab')[2];
      if (splitTab) splitTab.click();
    }, 400);
    const step3El = document.querySelector('#hw-step-panel-3') || document.querySelector('#hw-step-3');
    if (step3El) step3El.scrollIntoView({ behavior: 'instant', block: 'start' });
  `);
  await sleep(2500);
  await captureScreenshot(cdp, path.join(DOCS_IMG_DIR, 'hardware_teleop_cockpit.png'));

  // 4. 3D World Simulation
  console.log('Capturing 3D Simulation...');
  await cdp.eval(`
    if (typeof showPageView === 'function') showPageView('dashboard');
    if (typeof setDashboardView === 'function') setDashboardView('sim3d');
    window.scrollTo(0, 0);
  `);
  await sleep(3500); // Allow car and traffic physics simulation to run
  await captureScreenshot(cdp, path.join(DOCS_IMG_DIR, 'sim_3d_world_live.png'));

  // 5. Dual-Sensor Split View
  console.log('Capturing Dual-Sensor Split View...');
  await cdp.eval(`
    if (typeof setDashboardView === 'function') setDashboardView('dual');
  `);
  await sleep(2000);
  await captureScreenshot(cdp, path.join(DOCS_IMG_DIR, 'dual_split_live.png'));

  // 6. LiDAR BEV Grid View
  console.log('Capturing LiDAR BEV Grid View...');
  await cdp.eval(`
    if (typeof setDashboardView === 'function') setDashboardView('bev');
  `);
  await sleep(2000);
  await captureScreenshot(cdp, path.join(DOCS_IMG_DIR, 'lidar_bev_grid_live.png'));

  // 7. Camera Foveation View
  console.log('Capturing Camera Foveation View...');
  await cdp.eval(`
    if (typeof setDashboardView === 'function') setDashboardView('camera');
  `);
  await sleep(2000);
  await captureScreenshot(cdp, path.join(DOCS_IMG_DIR, 'camera_foveation_live.png'));

  console.log('All screenshots captured successfully!');

  // Cleanup
  try { chromeProc.kill(); } catch (_) {}
  try { fs.rmSync(profileDir, { recursive: true, force: true }); } catch (_) {}
  bridgeServer.close();
  uiServer.close();
  process.exit(0);
}

run().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
