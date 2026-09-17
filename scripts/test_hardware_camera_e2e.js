#!/usr/bin/env node
/**
 * @file test_hardware_camera_e2e.js
 * @brief End-to-end regression test for the Hardware Connect view.
 *
 * Boots a mock rover bridge on :8081 (so no physical board is required) and a
 * static server for web/ui that proxies /api/rover/* the way
 * web/server/websocket_bridge.js does. Then drives a real headless Chrome over
 * the DevTools Protocol and CLICKS every button in the connect-hardware
 * section, asserting:
 *
 *   1. Every wizard / tab / drive / ping / disconnect button acts.
 *   2. Drive buttons POST the right payload to /api/rover/drive.
 *   3. The rover MJPEG stream is CLOSED (not just visualised as stopped) when
 *      the user switches tab, leaves the camera view, opens another page view,
 *      or switches camera source.
 *
 * Usage:  node scripts/test_hardware_camera_e2e.js
 * Env:    CHROME_PATH   override browser binary
 *         UI_PORT       static server port (default 8188)
 *         CDP_PORT      devtools port (default 9333)
 */

'use strict';

const http = require('http');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

const ROOT = path.resolve(__dirname, '..');
const UI_DIR = path.join(ROOT, 'web', 'ui');
const UI_PORT = parseInt(process.env.UI_PORT || '8188', 10);
// Preferred bridge port. A live `waverover_bridge.py` may already own 8081 on a
// developer machine; the mock then falls back to a free port so the test never
// fights the real bridge (nor energises a real rover's motors).
const PREFERRED_BRIDGE_PORT = parseInt(process.env.BRIDGE_PORT || '8081', 10);
const CDP_PORT = parseInt(process.env.CDP_PORT || '9333', 10);
const UI_ORIGIN = `http://127.0.0.1:${UI_PORT}`;

// Resolved at runtime by startMockBridge().
let BRIDGE_PORT = PREFERRED_BRIDGE_PORT;

// 1x1 PNG — a guaranteed-decodable frame for the multipart stream.
const PNG_FRAME = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg==',
  'base64'
);

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

const log = (...a) => console.log(...a);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------
// Mock rover bridge (:8081) — records what the browser actually asked for
// ---------------------------------------------------------------------------
function startMockBridge() {
  const state = {
    statusHits: 0,
    pingLed: 0,
    disconnects: 0,
    cameraStops: 0,
    streamOpens: 0,
    openStreams: 0,
    drivePayloads: [],
    teleopPayloads: [],
    streams: [],
    hardwareOnline: true,
  };

  function closeEveryStream() {
    // Mirrors LiveCameraStreamer.stop_all_streams(): terminate the sends and
    // release the device, which is what stops the browser-side <img>.
    for (const s of state.streams) s.kill();
  }

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

    // -- MJPEG stream: stays open until the client aborts or a stop arrives --
    if (url === '/api/rover/camera' || url === '/video_feed') {
      state.streamOpens++;
      state.openStreams++;
      const rec = { closed: false, kill: null };
      state.streams.push(rec);

      res.writeHead(200, {
        'Content-Type': 'multipart/x-mixed-replace; boundary=FRAME',
        'Cache-Control': 'no-cache, private',
        Pragma: 'no-cache',
        ...CORS,
      });

      const timer = setInterval(() => {
        if (rec.closed || res.writableEnded) return;
        try {
          res.write('--FRAME\r\n');
          res.write('Content-Type: image/png\r\n');
          res.write(`Content-Length: ${PNG_FRAME.length}\r\n\r\n`);
          res.write(PNG_FRAME);
          res.write('\r\n');
        } catch (_) {
          kill();
        }
      }, 40);

      function kill() {
        if (rec.closed) return;
        rec.closed = true;
        state.openStreams--;
        clearInterval(timer);
        try { res.destroy(); } catch (_) {}
      }
      rec.kill = kill;

      req.on('close', kill);
      req.on('error', kill);
      res.on('close', kill);
      res.on('error', kill);
      return;
    }

    if (url === '/api/rover/status') {
      state.statusHits++;
      if (!state.hardwareOnline) {
        json({ connected: false, hardware: 'Arduino Uno Q', port: null, status: 'DISCONNECTED' });
        return;
      }
      json({
        mock: true,   // proves requests reach THIS stub, not a real bridge
        hardware: 'Arduino Uno Q',
        connected: true,
        port: 'COM5',
        status: 'CONNECTED',
        temperature: 48.2,
        cpu_temp: 48.2,
        battery: 11.8,
        voltage: 11.8,
        bus_latency_ms: 1.9,
        pose: { x: 0, y: 0, yaw_deg: 0 },
        pose_x: 0,
        pose_y: 0,
        yaw: 0,
      });
      return;
    }

    if (url === '/api/rover/ping_led') {
      state.pingLed++;
      json({ connected: true, hardware: 'Arduino Uno Q', message: 'LED flashed' });
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

    if (url === '/api/rover/camera/stop') {
      state.cameraStops++;
      closeEveryStream();
      json({ camera: 'stopped', status: 'ok' });
      return;
    }

    if (url === '/api/rover/disconnect') {
      state.disconnects++;
      closeEveryStream();
      json({ status: 'ok', connected: false });
      return;
    }

    if (url === '/api/rover/drive' || url === '/api/rover/teleop') {
      let body = '';
      req.on('data', (c) => { body += c; });
      req.on('end', () => {
        let parsed = null;
        try { parsed = JSON.parse(body || '{}'); } catch (_) {}
        if (url === '/api/rover/drive') state.drivePayloads.push(parsed);
        else state.teleopPayloads.push(parsed);
        json({ status: 'ok' });
      });
      return;
    }

    json({ error: 'Not Found' }, 404);
  });

  return { server, state };
}

/** True when something already answers HTTP on 127.0.0.1:`port`. */
function isPortOccupied(port) {
  return new Promise((resolve) => {
    const req = http.request(
      { host: '127.0.0.1', port, path: '/api/rover/status', method: 'GET', timeout: 700 },
      (res) => { res.resume(); resolve(true); }
    );
    req.on('timeout', () => { req.destroy(); resolve(false); });
    req.on('error', () => resolve(false));
    req.end();
  });
}

/**
 * Binds `server` to a port with nothing already serving on it.
 *
 * An explicit HTTP probe is required instead of relying on `listen()` failing:
 * Node sets SO_REUSEADDR on Windows, so a second server can silently bind to a
 * port a real `waverover_bridge.py` already owns and requests get split between
 * them unpredictably.
 */
async function listenOnFreePort(server, preferred, alternates) {
  for (const port of [preferred, ...alternates]) {
    if (await isPortOccupied(port)) continue;
    const ok = await new Promise((resolve) => {
      const onError = () => { server.removeListener('listening', onOk); resolve(false); };
      const onOk = () => { server.removeListener('error', onError); resolve(true); };
      server.once('error', onError);
      server.once('listening', onOk);
      server.listen(port, '127.0.0.1');
    });
    if (ok) return port;
  }
  throw new Error(`no free port among ${[preferred, ...alternates].join(', ')}`);
}

// ---------------------------------------------------------------------------
// Static UI server with the same /api/rover/* proxy as websocket_bridge.js
// ---------------------------------------------------------------------------
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml',
};

function startUiServer(bridgePort) {
  const server = http.createServer((req, res) => {
    const url = req.url.split('?')[0];

    // Proxy hardware/bridge endpoints to the Python bridge port, exactly like
    // the production Node bridge does — the dashboard uses relative paths
    // (e.g. /api/rover/camera) rather than an absolute bridge URL.
    if (url.startsWith('/api/rover/') || url.startsWith('/api/grid/') || url === '/video_feed') {
      const proxyReq = http.request(
        { host: '127.0.0.1', port: bridgePort, path: req.url, method: req.method, headers: req.headers },
        (proxyRes) => {
          res.writeHead(proxyRes.statusCode, { ...proxyRes.headers, 'Access-Control-Allow-Origin': '*' });
          proxyRes.pipe(res);
        }
      );
      proxyReq.on('error', () => {
        if (!res.headersSent) {
          res.writeHead(502, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: true, connected: false }));
        }
      });
      // Only tear the upstream down when the DOWNSTREAM response closes.
      // `req.on('close')` also fires the instant a bodyless GET finishes being
      // received, which aborts long-lived MJPEG streams (surfacing as 502).
      res.on('close', () => proxyReq.destroy());
      req.pipe(proxyReq);
      return;
    }

    if (url === '/api/telemetry' || url === '/yolo-status.json') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: false, platform: 'laptop' }));
      return;
    }

    if (url === '/api/rover/autopilot') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ autopilot_active: false }));
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

// ---------------------------------------------------------------------------
// Minimal DevTools Protocol client (Node >= 22 ships a global WebSocket)
// ---------------------------------------------------------------------------
class CDP {
  constructor(ws) {
    this.ws = ws;
    this.nextId = 0;
    this.pending = new Map();
    this.listeners = [];
    ws.addEventListener('message', (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch (_) { return; }
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        if (msg.error) reject(new Error(`${msg.error.message} (${JSON.stringify(msg.error.data ?? '')})`));
        else resolve(msg.result);
      } else if (msg.method) {
        for (const fn of this.listeners) fn(msg);
      }
    });
  }

  send(method, params = {}) {
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`CDP timeout: ${method}`));
        }
      }, 30000);
    });
  }

  once(method) {
    return new Promise((resolve) => {
      const fn = (msg) => {
        if (msg.method === method) {
          this.listeners = this.listeners.filter((f) => f !== fn);
          resolve(msg.params);
        }
      };
      this.listeners.push(fn);
    });
  }
}

function findBrowser() {
  const candidates = [
    process.env.CHROME_PATH,
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
    'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
  ].filter(Boolean);
  const found = candidates.find((p) => fs.existsSync(p));
  if (!found) throw new Error('No Chrome/Edge binary found. Set CHROME_PATH.');
  return found;
}

async function launchBrowser() {
  const bin = findBrowser();
  const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'hw-e2e-'));
  const proc = spawn(
    bin,
    [
      '--headless=new',
      '--disable-gpu',
      '--no-first-run',
      '--no-default-browser-check',
      '--disable-extensions',
      '--autoplay-policy=no-user-gesture-required',
      '--enable-unsafe-swiftshader',
      '--use-angle=swiftshader',
      `--remote-debugging-port=${CDP_PORT}`,
      `--user-data-dir=${profile}`,
      'about:blank',
    ],
    { stdio: 'ignore' }
  );

  for (let i = 0; i < 80; i++) {
    try {
      const r = await fetch(`http://127.0.0.1:${CDP_PORT}/json/list`);
      const targets = await r.json();
      const page = targets.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
      if (page) return { proc, wsUrl: page.webSocketDebuggerUrl, profile };
    } catch (_) { /* devtools not up yet */ }
    await sleep(250);
  }
  throw new Error('DevTools endpoint did not come up');
}

// ---------------------------------------------------------------------------
// Test run
// ---------------------------------------------------------------------------
const results = [];
let failures = 0;

function check(name, ok, detail = '') {
  results.push({ name, ok, detail });
  if (!ok) failures++;
  log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
}

async function main() {
  const { server: bridge, state } = startMockBridge();
  const alternates = [];
  for (let p = 8101; p <= 8115; p++) alternates.push(p);
  BRIDGE_PORT = await listenOnFreePort(bridge, PREFERRED_BRIDGE_PORT, alternates);
  if (BRIDGE_PORT !== PREFERRED_BRIDGE_PORT) {
    log(`[e2e] note: port ${PREFERRED_BRIDGE_PORT} is in use (real bridge running) — using mock bridge on ${BRIDGE_PORT}.`);
  }
  const uiServer = await startUiServer(BRIDGE_PORT);
  log(`[e2e] mock bridge  -> http://127.0.0.1:${BRIDGE_PORT}`);
  // Guard against SO_REUSEADDR splitting requests between this stub and a real
  // bridge that happens to be running: confirm the status payload is ours.
  const probe = await new Promise((resolve) => {
    http.get({ host: '127.0.0.1', port: BRIDGE_PORT, path: '/api/rover/status' }, (res) => {
      let body = '';
      res.on('data', (c) => { body += c; });
      res.on('end', () => {
        try { resolve(JSON.parse(body)); } catch { resolve({}); }
      });
    }).on('error', () => resolve({}));
  });
  if (probe.mock !== true) {
    log(`[e2e] FAILED to bind an isolated mock bridge on ${BRIDGE_PORT} (got ${JSON.stringify(probe)}).`);
    log('[e2e] stop the running waverover_bridge.py or set BRIDGE_PORT=<free port>.');
    process.exitCode = 1;
    bridge.close();
    uiServer.close();
    return;
  }
  log(`[e2e] dashboard UI -> ${UI_ORIGIN}`);

  const { proc, wsUrl } = await launchBrowser();
  const ws = new WebSocket(wsUrl);
  await new Promise((res, rej) => {
    ws.addEventListener('open', res, { once: true });
    ws.addEventListener('error', () => rej(new Error('websocket error')), { once: true });
  });
  const cdp = new CDP(ws);

  // Collect page exceptions so a broken button cannot pass silently.
  const pageErrors = [];
  cdp.listeners.push((msg) => {
    if (msg.method === 'Runtime.exceptionThrown') {
      const d = msg.params.exceptionDetails;
      pageErrors.push(d.exception?.description || d.text || 'unknown exception');
    }
    if (msg.method === 'Log.entryAdded' && msg.params.entry.level === 'error') {
      pageErrors.push('[console] ' + msg.params.entry.text);
    }
  });

  await cdp.send('Page.enable');
  await cdp.send('Runtime.enable');
  await cdp.send('Log.enable');

  const loaded = cdp.once('Page.loadEventFired');
  await cdp.send('Page.navigate', { url: `${UI_ORIGIN}/index.html` });
  await loaded;

  const evaluate = async (expression) => {
    const res = await cdp.send('Runtime.evaluate', {
      expression, awaitPromise: true, returnByValue: true, userGesture: true,
    });
    if (res.exceptionDetails) {
      throw new Error('Page exception: ' + (res.exceptionDetails.exception?.description || res.exceptionDetails.text));
    }
    return res.result.value;
  };

  const waitFor = async (label, predicate, timeoutMs = 5000) => {
    const started = Date.now();
    while (Date.now() - started < timeoutMs) {
      if (predicate()) return true;
      await sleep(50);
    }
    check(label, false, `timed out after ${timeoutMs}ms`);
    return false;
  };

  // -- Wait for the dashboard bundle to finish initialising -----------------
  const booted = await evaluate(
    `(async () => {
       for (let i = 0; i < 100; i++) {
         if (typeof window.showPageView === 'function' &&
             typeof window.hwAttemptConnect === 'function' &&
             window.engineInstance) return true;
         await new Promise(r => setTimeout(r, 100));
       }
       return { showPageView: typeof window.showPageView,
                hwAttemptConnect: typeof window.hwAttemptConnect,
                engineInstance: !!window.engineInstance };
     })()`
  );
  check('dashboard + hardware bundles initialised', booted === true, JSON.stringify(booted));

  check('engineInstance is published on window (showPageView can stop cameras)',
    await evaluate('!!window.engineInstance && typeof window.engineInstance.stopAllCameras === "function"'));

  // -- A. Hardware wizard: device cards -------------------------------------
  await evaluate(`window.showPageView('hardware')`);
  check('showPageView("hardware") activates the hardware view',
    await evaluate(`document.getElementById('hardware-view').classList.contains('active')`));

  const cardIds = ['arduino-uno-q', 'arduino-uno-r4', 'raspberry-pi', 'esp32', 'custom'];
  for (const id of cardIds) {
    // Cards auto-advance to step 2 after 250ms, so return to step 1 first.
    await evaluate(`window.hwGoToStep(1)`);
    await evaluate(`document.getElementById('hw-card-${id}').click()`);
  }
  // hwSelectDevice highlights via the `.selected` class on `.hw-card` and clears
  // it from every other card.
  check('device cards click + highlight the last card only',
    await evaluate(`document.getElementById('hw-card-custom').classList.contains('selected') &&
                    document.querySelectorAll('.hw-card.selected').length === 1`),
    await evaluate(`document.querySelectorAll('.hw-card.selected').length + ' card(s) selected'`));
  check('selected-device label reflects the last card clicked',
    (await evaluate(`document.getElementById('hw-step1-selected-label').textContent.trim()`)) ===
      'Custom MCU / Robot',
    await evaluate(`document.getElementById('hw-step1-selected-label').textContent.trim()`));
  await evaluate(`window.hwGoToStep(1)`);

  // -- B. Wizard navigation --------------------------------------------------
  await evaluate(`window.hwGoToStep(2)`);
  check('"Continue" (hwGoToStep(2)) reveals the instructions section',
    await evaluate(`!document.getElementById('hw-section-instructions').classList.contains('hw-section-hidden')`));
  await evaluate(`window.hwGoToStep(3)`);
  check('hwGoToStep(3) reveals the connect section',
    await evaluate(`!document.getElementById('hw-section-connect').classList.contains('hw-section-hidden')`));
  await evaluate(`window.hwGoToStep(1)`);
  check('"Switch Hardware" (hwGoToStep(1)) returns to device selection',
    await evaluate(`!document.getElementById('hw-section-select').classList.contains('hw-section-hidden')`));
  await evaluate(`window.hwGoToStep(3)`);

  // Point the wizard at this stub before exercising anything that talks to the
  // bridge directly (the ping/led and drive buttons use the resolved host, not
  // the page origin), so no request can leak onto a real bridge on port 8081.
  await evaluate(`document.getElementById('hw-bridge-host').value = 'http://localhost:${BRIDGE_PORT}'`);

  // -- C. Ping/Flash LED button ---------------------------------------------
  await evaluate(`document.getElementById('hw-btn-ping-led').click()`);
  await waitFor('"Check Board & Flash LED" reaches the bridge', () => state.pingLed > 0);
  check('ping-led button POSTed /api/rover/ping_led', state.pingLed > 0, `hits=${state.pingLed}`);
  await sleep(400);
  check('ping-led button restores its label ("Heart / OK LED Sent!")',
    (await evaluate(`document.getElementById('hw-btn-ping-led').innerHTML`)).includes('Heart') ||
    (await evaluate(`document.getElementById('hw-btn-ping-led').innerHTML`)).includes('OK'));

  // -- D. Connect ------------------------------------------------------------
  await evaluate(`document.getElementById('hw-bridge-host').value = 'http://localhost:${BRIDGE_PORT}'`);
  await evaluate(`document.getElementById('hw-btn-connect').click()`);
  const connected = await waitFor('Connect button connects to the bridge', () => state.statusHits > 0);
  if (connected) {
    await evaluate(`(async () => {
      for (let i = 0; i < 60; i++) {
        if (document.getElementById('hw-live-dashboard').style.display === 'block') return true;
        await new Promise(r => setTimeout(r, 100));
      }
      return false;
    })()`);
  }
  check('Connect button shows the live dashboard',
    await evaluate(`document.getElementById('hw-live-dashboard').style.display === 'block'`));
  check('Connect button reveals the Disconnect button',
    await evaluate(`document.getElementById('hw-btn-disconnect').style.display !== 'none'`));

  // -- E. Camera stream while on the camera tab ------------------------------
  await waitFor('camera tab opens the MJPEG stream', () => state.openStreams > 0);
  check('camera feed <img> is bound to the live stream',
    (await evaluate(`document.getElementById('hw-camera-feed').src`)).includes('/api/rover/camera'));

  // -- F. Tab switching must STOP the camera --------------------------------
  const stopsBeforeTab = state.cameraStops;
  await evaluate(`document.querySelector(".hw-tab[onclick*='bev']").click()`);
  await waitFor('switching to BEV closes the MJPEG stream', () => state.openStreams === 0);
  check('switching tab BEV closes the rover camera stream', state.openStreams === 0,
    `open=${state.openStreams}`);
  check('switching tab BEV force-stops the camera on the bridge',
    state.cameraStops > stopsBeforeTab, `stops=${state.cameraStops}`);
  check('detached feed <img> no longer points at the stream',
    !(await evaluate(`document.getElementById('hw-camera-feed').src`)).includes('/api/rover/camera'));

  await evaluate(`document.querySelector(".hw-tab[onclick*='split']").click()`);
  await waitFor('split tab restarts the stream', () => state.openStreams > 0);
  check('split tab re-opens the stream while stays on camera view', state.openStreams > 0);
  await evaluate(`document.querySelector(".hw-tab[onclick*='telemetry']").click()`);
  await waitFor('telemetry tab closes the stream', () => state.openStreams === 0);
  check('telemetry tab closes the stream', state.openStreams === 0);

  // -- G. Drive buttons ------------------------------------------------------
  await evaluate(`document.querySelector(".hw-tab[onclick*='camera']").click()`);
  const driveCountBefore = state.drivePayloads.length;
  const press = async (dir) => {
    await evaluate(`(() => {
      const b = [...document.querySelectorAll('.hw-teleop-btn')]
        .find(x => x.getAttribute('onmousedown') === "hwTeleop('${dir}')");
      if (!b) return 'MISSING';
      b.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
      b.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
      return 'OK';
    })()`);
  };
  for (const dir of ['forward', 'backward', 'left', 'right']) await press(dir);
  await sleep(700);
  const expected = {
    forward: { left: 0.6, right: 0.6 },
    backward: { left: -0.6, right: -0.6 },
    left: { left: -0.4, right: 0.4 },
    right: { left: 0.4, right: -0.4 },
  };
  const got = state.drivePayloads.slice(driveCountBefore);
  let driveOk = got.length >= 4;
  for (const dir of Object.keys(expected)) {
    const hit = got.find((p) => p && p.left === expected[dir].left && p.right === expected[dir].right);
    if (!hit) driveOk = false;
  }
  check('all four drive buttons POST the correct /api/rover/drive payloads', driveOk,
    `${got.length} payload(s): ${JSON.stringify(got)}`);

  await evaluate(`[...document.querySelectorAll('.hw-teleop-btn')]
     .find(x => x.getAttribute('onclick') === "hwTeleop('stop')").click()`);
  await sleep(400);
  check('STOP button posts a zero-velocity command',
    state.drivePayloads.some((p) => p && p.left === 0 && p.right === 0),
    JSON.stringify(state.drivePayloads.slice(-2)));

  // -- H. Leaving the hardware view stops the camera -------------------------
  const stopsBeforeNav = state.cameraStops;
  await evaluate(`window.showPageView('dashboard')`);
  await waitFor('leaving hardware view closes the stream', () => state.openStreams === 0);
  check('switching to the dashboard closes the rover camera stream', state.openStreams === 0);
  check('switching to the dashboard force-stops the camera on the bridge',
    state.cameraStops > stopsBeforeNav, `stops=${state.cameraStops}`);
  check('dashboard view is active and hardware view is not',
    await evaluate(`document.getElementById('dashboard-view').classList.contains('active')
      && !document.getElementById('hardware-view').classList.contains('active')`));

  // -- I. Dashboard: rover camera source lifecycle ---------------------------
  const dashboardCamState = () => evaluate(`(() => {
    const e = window.engineInstance || {};
    const img = e.roverCamImg;
    return JSON.stringify({
      source: e.cameraSource, mode: e.viewMode, released: e.roverReleased,
      imgSrc: img ? img.src : null, currentSrc: img ? img.currentSrc : null,
    });
  })()`);

  await evaluate(`window.switchCameraSource('rover')`);
  await waitFor('dashboard source "rover" opens the stream', () => state.openStreams > 0);
  check('dashboard camera source "rover" subscribes to the MJPEG stream', state.openStreams > 0,
    `open=${state.openStreams} ${await dashboardCamState()}`);

  await evaluate(`window.switchCameraSource('synthetic')`);
  await waitFor('dashboard source "synthetic" closes the stream', () => state.openStreams === 0);
  check('switching camera source away from rover closes the stream', state.openStreams === 0);

  await evaluate(`window.switchCameraSource('rover')`);
  await waitFor('rover source re-opens the stream', () => state.openStreams > 0);
  const stopsBeforeViewMode = state.cameraStops;
  await evaluate(`window.setDashboardView('bev')`);
  await waitFor('BEV view mode releases the camera', () => state.openStreams === 0);
  check('switching dashboard view mode to BEV releases the rover camera', state.openStreams === 0,
    `open=${state.openStreams}`);
  check('BEV view mode force-stopped the camera on the bridge',
    state.cameraStops > stopsBeforeViewMode, `stops=${state.cameraStops}`);

  await evaluate(`window.setDashboardView('camera')`);
  await waitFor('camera view mode re-attaches the stream', () => state.openStreams > 0);
  check('returning to the camera view re-attaches the rover stream', state.openStreams > 0);

  await evaluate(`window.showPageView('landing')`);
  await waitFor('landing view releases the camera', () => state.openStreams === 0);
  check('leaving the dashboard for the landing view stops the rover camera', state.openStreams === 0);

  // -- J. Disconnect button --------------------------------------------------
  await evaluate(`window.showPageView('hardware')`);
  await sleep(600);
  const disBefore = state.disconnects;
  await evaluate(`document.getElementById('hw-btn-disconnect').click()`);
  await waitFor('Disconnect POSTs to the bridge', () => state.disconnects > disBefore);
  check('Disconnect button POSTed /api/rover/disconnect', state.disconnects > disBefore);
  await waitFor('Disconnect closes the stream', () => state.openStreams === 0);
  check('Disconnect button closes the rover camera stream', state.openStreams === 0);
  check('Disconnect hides the Disconnect button again',
    await evaluate(`document.getElementById('hw-btn-disconnect').style.display === 'none'`));

  // -- K. No silent script errors -------------------------------------------
  const ourErrors = pageErrors.filter((e) => /hardware_connect\.js|teleop_dashboard\.js/.test(e));
  check('no uncaught exceptions from hardware_connect.js / teleop_dashboard.js',
    ourErrors.length === 0, ourErrors.slice(0, 4).join(' | '));
  if (pageErrors.length > ourErrors.length) {
    log('[e2e] note: unrelated page messages: ' + pageErrors.filter((e) => !ourErrors.includes(e)).slice(0, 3).join(' | '));
  }

  log('');
  log(`[e2e] ${results.length - failures}/${results.length} checks passed`);
  ws.close();
  try { proc.kill(); } catch (_) {}
  uiServer.close();
  bridge.close();
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error('[e2e] harness error:', err && err.stack ? err.stack : err);
  process.exit(1);
});