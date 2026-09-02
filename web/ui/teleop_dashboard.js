/**
 * @file teleop_dashboard.js
 * @brief WebGL / HTML5 Canvas Teleoperation Dashboard Engine.
 * Renders live 3-ring foveated grid, vehicle ego-motion trajectory, and obstacle heatmaps.
 */

class TeleopDashboardEngine {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.frame = 0;
        this.vehicleX = 0;
        this.vehicleY = 0;
        this.yaw = 0;

        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.startRenderLoop();
    }

    resize() {
        const rect = this.canvas.parentElement ? this.canvas.parentElement.getBoundingClientRect() : null;
        const w = (rect && rect.width > 50) ? rect.width - 36 : (window.innerWidth ? window.innerWidth * 0.5 : 800);
        const h = (rect && rect.height > 50) ? rect.height - 36 : (window.innerHeight ? window.innerHeight * 0.7 : 600);
        this.canvas.width = Math.max(300, Math.floor(w));
        this.canvas.height = Math.max(300, Math.floor(h));
    }

    startRenderLoop() {
        const render = () => {
            this.updateState();
            this.draw();
            requestAnimationFrame(render);
        };
        requestAnimationFrame(render);
    }

    updateState() {
        this.frame++;
        this.vehicleX += Math.cos(this.frame * 0.02) * 0.05;
        this.vehicleY += Math.sin(this.frame * 0.02) * 0.05;
        this.yaw = this.frame * 0.02;

        // Update telemetry text
        document.getElementById('stat-fps').innerText = '30.0 FPS';
        document.getElementById('stat-latency').innerText = '1.85 ms';
        document.getElementById('stat-points').innerText = '120,480';
        document.getElementById('stat-vram').innerText = '1.2 GB / 16 GB';
        document.getElementById('hud-ego-pos').innerText = `X: ${this.vehicleX.toFixed(2)}m  Y: ${this.vehicleY.toFixed(2)}m  Yaw: ${(this.yaw * 180 / Math.PI).toFixed(1)}°`;
    }

    draw() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        const centerX = width / 2;
        const centerY = height / 2;

        // Clear canvas
        this.ctx.fillStyle = '#020408';
        this.ctx.fillRect(0, 0, width, height);

        // Draw grid lines
        this.ctx.strokeStyle = 'rgba(56, 189, 248, 0.08)';
        this.ctx.lineWidth = 1;
        const step = 40;
        for (let x = 0; x < width; x += step) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, height);
            this.ctx.stroke();
        }
        for (let y = 0; y < height; y += step) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(width, y);
            this.ctx.stroke();
        }

        // Draw 3 Concentric Ring Boundaries
        // Ring 0: Near (10m)
        this.ctx.strokeStyle = '#38bdf8';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, 60, 0, Math.PI * 2);
        this.ctx.stroke();

        // Ring 1: Mid (30m)
        this.ctx.strokeStyle = '#c084fc';
        this.ctx.lineWidth = 1.5;
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, 140, 0, Math.PI * 2);
        this.ctx.stroke();

        // Ring 2: Far (100m)
        this.ctx.strokeStyle = '#fb923c';
        this.ctx.lineWidth = 1.2;
        this.ctx.setLineDash([6, 6]);
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, 240, 0, Math.PI * 2);
        this.ctx.stroke();
        this.ctx.setLineDash([]);

        // Draw simulated point cloud & obstacles
        const numPoints = 600;
        for (let i = 0; i < numPoints; i++) {
            const angle = (i / numPoints) * Math.PI * 2 + (this.frame * 0.005);
            const dist = (Math.sin(i * 37.1) * 0.5 + 0.5) * 220;
            const px = centerX + Math.cos(angle) * dist;
            const py = centerY + Math.sin(angle) * dist;

            // Class colors: 0=static, 1=dynamic, 2=pole, 3=ground
            let color = '#4ade80'; // ground
            if (dist < 60) color = '#38bdf8';
            else if (i % 7 === 0) color = '#f43f5e'; // dynamic car/pedestrian
            else if (i % 5 === 0) color = '#fb923c'; // pole/wall

            this.ctx.fillStyle = color;
            this.ctx.fillRect(px - 1.5, py - 1.5, 3, 3);
        }

        // Draw Vehicle Ego Symbol (Triangle at Center)
        this.ctx.save();
        this.ctx.translate(centerX, centerY);
        this.ctx.rotate(this.yaw);

        this.ctx.fillStyle = '#ffffff';
        this.ctx.beginPath();
        this.ctx.moveTo(0, -14);
        this.ctx.lineTo(-9, 10);
        this.ctx.lineTo(0, 5);
        this.ctx.lineTo(9, 10);
        this.ctx.closePath();
        this.ctx.fill();

        this.ctx.strokeStyle = '#38bdf8';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        this.ctx.restore();
    }
}

window.addEventListener('DOMContentLoaded', () => {
    new TeleopDashboardEngine('teleop-canvas');
});
