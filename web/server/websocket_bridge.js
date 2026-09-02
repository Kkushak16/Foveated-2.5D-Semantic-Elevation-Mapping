/**
 * @file websocket_bridge.js
 * @brief Node.js WebSocket Bridge for Unified Teleoperation Dashboard.
 * Streams live 2.5D foveated grid state and vehicle pose to remote browser clients.
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = parseInt(process.env.PORT || process.argv[2] || '8080', 10);
const UI_DIR = path.join(__dirname, '../ui');

const server = http.createServer((req, res) => {
    let reqUrl = req.url.split('?')[0];
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
            res.writeHead(200, { 'Content-Type': contentType });
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
});
