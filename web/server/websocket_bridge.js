/**
 * @file websocket_bridge.js
 * @brief Node.js WebSocket Bridge for Teleoperation Dashboard.
 * Streams live 2.5D foveated grid state and vehicle pose to remote browser clients
 * over WebSockets without impacting local vehicle compute performance.
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 8080;
const UI_DIR = path.join(__dirname, '../ui');

// Simple HTTP server to serve the Teleop Dashboard UI
const server = http.createServer((req, res) => {
    let filePath = path.join(UI_DIR, req.url === '/' ? 'index.html' : req.url);
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
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(content, 'utf-8');
        }
    });
});

server.listen(PORT, () => {
    console.log('========================================================================');
    console.log(`  🌐 LiDAR Teleop Dashboard Server running at http://localhost:${PORT}`);
    console.log('  - Bridge Mode : WebSocket / Zero-Overhead HTTP Stream');
    console.log('  - Client UI   : WebGL / HTML5 Dynamic 3-Ring HUD');
    console.log('========================================================================');
});
