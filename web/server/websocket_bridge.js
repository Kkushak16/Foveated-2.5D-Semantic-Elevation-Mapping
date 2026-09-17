/**
 * @file websocket_bridge.js
 * @brief Node.js WebSocket Bridge for Unified Teleoperation Dashboard.
 * Streams live 2.5D foveated grid state and vehicle pose to remote browser clients.
 *
 * ALSO spawns the Python YOLO semantic vision server
 * (python/yolo_vision_server.py) and serves its /yolo-status.json so the
 * browser knows where to open the WebSocket for per-pixel car / person shape
 * recognition (wheels & headlights included).
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const net = require('net');
const { spawn } = require('child_process');

const PORT = parseInt(process.env.PORT || process.argv[2] || '8080', 10);
const UI_DIR = path.join(__dirname, '../ui');
const ROOT = path.resolve(__dirname, '..', '..');

// ---------------------------------------------------------------------------
// YOLO semantic vision server bootstrap
// ---------------------------------------------------------------------------
function findFreePort(start, tries = 12) {
    for (let p = start; p < start + tries; p++) {
        const srv = net.createServer();
        try {
            srv.listen(p);
            srv.close();
            return p;
        } catch (err) {
            try { srv.close(); } catch (_) {}
        }
    }
    return start;
}

const YOLO_PORT = parseInt(process.env.YOLO_PORT || String(findFreePort(8090)), 10);
const PY_SCRIPT = path.join(ROOT, 'python', 'yolo_vision_server.py');
const VENV_PY = path.join(ROOT, '.venv', 'Scripts', 'python.exe');
const PYTHON = process.env.YOLO_PYTHON ||
    (fs.existsSync(VENV_PY) ? VENV_PY : (process.platform === 'win32' ? 'python' : 'python3'));
const YOLO_BACKEND = process.env.YOLO_BACKEND || 'auto';

let yoloSpawned = false;

function startYoloServer() {
    if (yoloSpawned || !fs.existsSync(PY_SCRIPT)) return;
    yoloSpawned = true;
    let proc = null;
    try {
        proc = spawn(PYTHON, [PY_SCRIPT, '--port', String(YOLO_PORT), '--backend', YOLO_BACKEND], {
            cwd: ROOT,
            stdio: ['ignore', 'inherit', 'inherit']
        });
    } catch (err) {
        console.log(`[yolo-server] could not start (${err.message}) — browser will use its built-in cascade fallback.`);
        return;
    }
    proc.on('exit', (code) => {
        console.log(`[yolo-server] vision server exited (code ${code}).`);
    });
    console.log(`[yolo-server] Python YOLO semantic vision server → ws://127.0.0.1:${YOLO_PORT} (backend=${YOLO_BACKEND})`);
}

// ---------------------------------------------------------------------------
// Static file HTTP server
// ---------------------------------------------------------------------------
const server = http.createServer((req, res) => {
    let reqUrl = req.url.split('?')[0];

    // Serve the YOLO status JSON written by the Python server, with a graceful
    // fallback payload so the browser never hits a 404 while it starts up.
    if (reqUrl === '/yolo-status.json') {
        const stFile = path.join(UI_DIR, 'yolo_status.json');
        fs.readFile(stFile, (err, content) => {
            if (err) {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({
                    ok: false,
                    reason: 'yolo-server-not-ready',
                    port: YOLO_PORT,
                }));
            } else {
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(content, 'utf-8');
            }
        });
        return;
    }

    // Proxy rover and hardware endpoints to Python Bridge (Port 8081)
    if (reqUrl.startsWith('/api/rover/') || reqUrl.startsWith('/api/grid/') || reqUrl === '/video_feed') {
        const proxyReq = http.request({
            host: '127.0.0.1',
            port: 8081,
            path: req.url,
            method: req.method,
            headers: {
                ...req.headers,
                host: '127.0.0.1:8081'
            }
        }, (proxyRes) => {
            res.writeHead(proxyRes.statusCode, {
                ...proxyRes.headers,
                'Access-Control-Allow-Origin': '*'
            });
            proxyRes.pipe(res, { end: true });
        });
        proxyReq.on('error', (err) => {
            if (!res.headersSent) {
                res.writeHead(502, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
                res.end(JSON.stringify({ error: true, connected: false, message: 'Bridge offline on 8081' }));
            }
        });
        // Teardown is driven by the downstream response only. Listening on
        // `req.on('close')` also fires as soon as a bodyless GET has been fully
        // received, which destroys the upstream socket mid-response and breaks
        // long-lived MJPEG streams (`/api/rover/camera`, `/video_feed`).
        res.on('close', () => {
            proxyReq.destroy();
        });
        req.pipe(proxyReq, { end: true });
        return;
    }

    // -----------------------------------------------------------------------
    // Telemetry API — serves live GPU/system metrics from the Python module
    // (Section 9 of physical-testing.md: frontend telemetry upgrades)
    // -----------------------------------------------------------------------
    if (reqUrl === '/api/telemetry') {
        const platform = process.env.FOVEATED_PLATFORM || 'auto';
        const { execFile } = require('child_process');
        execFile(PYTHON, [
            path.join(ROOT, 'python', 'telemetry.py'),
            '--platform', platform, '--json'
        ], { timeout: 5000, cwd: ROOT }, (err, stdout) => {
            res.writeHead(200, {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Cache-Control': 'no-cache'
            });
            if (err) {
                res.end(JSON.stringify({
                    error: true,
                    platform: platform,
                    vram_used_mb: 0, vram_total_mb: 0,
                    gpu_util_pct: 0, power_draw_w: 0, soc_temp_c: 0
                }));
            } else {
                res.end(stdout.trim());
            }
        });
        return;
    }

    // Platform info endpoint — returns the current deployment tier
    if (reqUrl === '/api/platform') {
        const platform = process.env.FOVEATED_PLATFORM || 'laptop';
        const source = process.env.FOVEATED_SOURCE || 'simulator';
        res.writeHead(200, {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        });
        res.end(JSON.stringify({ platform, source }));
        return;
    }

    let filePath = path.join(UI_DIR, reqUrl === '/' ? 'index.html' : reqUrl);
    let extname = path.extname(filePath);
    let contentType = 'text/html';

    switch (extname) {
        case '.js': contentType = 'text/javascript'; break;
        case '.css': contentType = 'text/css'; break;
        case '.json': contentType = 'application/json'; break;
        case '.png': contentType = 'image/png'; break;
    }

    fs.readFile(filePath, (err, content) => {
        if (err) {
            if (err.code === 'ENOENT') {
                res.writeHead(404, { 'Content-Type': 'text/html' });
                res.end('<h1>404 Not Found</h1>', 'utf-8');
            } else {
                res.writeHead(500);
                res.end(`Server Error: ${err.code}`);
            }
        } else {
            res.writeHead(200, {
                'Content-Type': contentType,
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            });
            res.end(content, 'utf-8');
        }
    });
});

server.listen(PORT, () => {
    console.log('========================================================================');
    console.log(`  🌐 LiDAR & Camera Unified Teleop Dashboard: http://localhost:${PORT}`);
    console.log('  - Bridge Mode : WebSocket / Zero-Overhead HTTP Stream');
    console.log('  - Client UI   : HTML5 / Canvas / WebGL 3-Ring HUD');
    console.log('========================================================================');
    startYoloServer();
});
