/**
 * @file teleop_dashboard.js
 * @brief Unified Dual-Sensor Foveated Teleop Dashboard Engine.
 * Renders live 3-ring LiDAR grid, Camera Foveated Spatial Masking & Motion Gating,
 * and handles interactive telemetry controls & view switching.
 */

class UnifiedTeleopEngine {
    constructor() {
        this.bevCanvas = document.getElementById('teleop-canvas');
        this.camCanvas = document.getElementById('camera-canvas');

        this.bevCtx = this.bevCanvas.getContext('2d');
        this.camCtx = this.camCanvas.getContext('2d');

        this.frame = 0;
        this.isRunning = true;
        this.viewMode = 'bev'; // 'bev' | 'camera' | 'dual'
        this.pointDensity = 100000;

        this.vehicleX = 0;
        this.vehicleY = 0;
        this.yaw = 0;

        this.init();
    }

    init() {
        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.startLoop();
    }

    resize() {
        const wrapper = document.getElementById('canvas-wrapper');
        const rect = wrapper ? wrapper.getBoundingClientRect() : null;

        let w = rect ? rect.width : window.innerWidth * 0.5;
        let h = rect ? rect.height : window.innerHeight * 0.7;

        if (this.viewMode === 'dual') {
            w = (w - 10) / 2;
        }

        const targetW = Math.max(300, Math.floor(w));
        const targetH = Math.max(300, Math.floor(h));

        this.bevCanvas.width = targetW;
        this.bevCanvas.height = targetH;

        this.camCanvas.width = targetW;
        this.camCanvas.height = targetH;
    }

    setViewMode(mode) {
        this.viewMode = mode;
        const hudMode = document.getElementById('hud-mode');

        if (mode === 'bev') {
            this.bevCanvas.classList.remove('hidden-canvas');
            this.camCanvas.classList.add('hidden-canvas');
            if (hudMode) hudMode.innerText = 'VIEW: 3-RING LIDAR BEV EGO-GRID';
        } else if (mode === 'camera') {
            this.bevCanvas.classList.add('hidden-canvas');
            this.camCanvas.classList.remove('hidden-canvas');
            if (hudMode) hudMode.innerText = 'VIEW: CAMERA FOVEATED OPTICAL FLOW MASK';
        } else if (mode === 'dual') {
            this.bevCanvas.classList.remove('hidden-canvas');
            this.camCanvas.classList.remove('hidden-canvas');
            if (hudMode) hudMode.innerText = 'VIEW: DUAL-SENSOR FUSION SPLIT';
        }

        this.resize();
    }

    setPointDensity(density) {
        this.pointDensity = parseInt(density, 10);
        document.getElementById('stat-points').innerText = this.pointDensity.toLocaleString();
    }

    toggleSimulation() {
        this.isRunning = !this.isRunning;
        const btn = document.getElementById('btn-play-pause');
        if (btn) {
            btn.innerText = this.isRunning ? '⏸️ Pause Stream' : '▶️ Resume Stream';
        }
    }

    startLoop() {
        const render = () => {
            if (this.isRunning) {
                this.updateState();
            }
            this.drawBEV();
            this.drawCameraFoveation();
            requestAnimationFrame(render);
        };
        requestAnimationFrame(render);
    }

    updateState() {
        this.frame++;
        this.vehicleX += Math.cos(this.frame * 0.02) * 0.04;
        this.vehicleY += Math.sin(this.frame * 0.02) * 0.04;
        this.yaw = (this.frame * 0.015) % (Math.PI * 2);

        // Update telemetry display values
        const fps = (70 + Math.sin(this.frame * 0.1) * 3).toFixed(1);
        const latency = (1.80 + Math.sin(this.frame * 0.05) * 0.08).toFixed(2);

        document.getElementById('stat-fps').innerText = `${fps} FPS`;
        document.getElementById('stat-latency').innerText = `${latency} ms`;
        document.getElementById('hud-ego-pos').innerText = `X: ${this.vehicleX.toFixed(2)}m | Y: ${this.vehicleY.toFixed(2)}m | Yaw: ${(this.yaw * 180 / Math.PI).toFixed(1)}°`;
    }

    drawBEV() {
        if (this.bevCanvas.classList.contains('hidden-canvas')) return;

        const ctx = this.bevCtx;
        const w = this.bevCanvas.width;
        const h = this.bevCanvas.height;
        const cx = w / 2;
        const cy = h / 2;

        // Background
        ctx.fillStyle = '#020408';
        ctx.fillRect(0, 0, w, h);

        // Grid overlay
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.07)';
        ctx.lineWidth = 1;
        const step = 40;
        for (let x = 0; x < w; x += step) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, h);
            ctx.stroke();
        }
        for (let y = 0; y < h; y += step) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(w, y);
            ctx.stroke();
        }

        // 3 Concentric Ring Boundaries
        const rNear = Math.min(w, h) * 0.15;
        const rMid = Math.min(w, h) * 0.32;
        const rFar = Math.min(w, h) * 0.44;

        // Near Ring (0-10m)
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cx, cy, rNear, 0, Math.PI * 2);
        ctx.stroke();

        // Mid Ring (10-30m)
        ctx.strokeStyle = '#c084fc';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(cx, cy, rMid, 0, Math.PI * 2);
        ctx.stroke();

        // Far Ring (30-100m)
        ctx.strokeStyle = '#fb923c';
        ctx.lineWidth = 1.2;
        ctx.setLineDash([6, 6]);
        ctx.beginPath();
        ctx.arc(cx, cy, rFar, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        // Render point cloud & obstacles based on point density
        const numPoints = Math.min(750, Math.floor(this.pointDensity / 120));
        for (let i = 0; i < numPoints; i++) {
            const angle = (i / numPoints) * Math.PI * 2 + (this.frame * 0.003);
            const dist = (Math.sin(i * 37.1) * 0.5 + 0.5) * rFar * 0.95;
            const px = cx + Math.cos(angle) * dist;
            const py = cy + Math.sin(angle) * dist;

            let color = '#4ade80'; // ground
            if (dist < rNear) color = '#38bdf8';
            else if (i % 7 === 0) color = '#f43f5e'; // dynamic car/pedestrian
            else if (i % 5 === 0) color = '#c084fc'; // mid obstacles

            ctx.fillStyle = color;
            ctx.fillRect(px - 1.5, py - 1.5, 3, 3);
        }

        // Draw Ego Vehicle Symbol
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(this.yaw);

        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.moveTo(0, -14);
        ctx.lineTo(-9, 10);
        ctx.lineTo(0, 5);
        ctx.lineTo(9, 10);
        ctx.closePath();
        ctx.fill();

        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.restore();
    }

    drawCameraFoveation() {
        if (this.camCanvas.classList.contains('hidden-canvas')) return;

        const ctx = this.camCtx;
        const w = this.camCanvas.width;
        const h = this.camCanvas.height;

        // Base Simulated Camera Road Frame
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, w, h);

        // Perspective Road Lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 2;

        ctx.beginPath();
        ctx.moveTo(w * 0.45, h * 0.45);
        ctx.lineTo(w * 0.1, h * 0.85);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(w * 0.55, h * 0.45);
        ctx.lineTo(w * 0.9, h * 0.85);
        ctx.stroke();

        // 1. Semantic Spatial Masking Overlays (Sky & Hood Removal)
        // Sky Mask (Top 35%)
        ctx.fillStyle = 'rgba(2, 6, 23, 0.85)';
        ctx.fillRect(0, 0, w, h * 0.35);

        ctx.fillStyle = 'rgba(244, 63, 94, 0.4)';
        ctx.font = '12px Inter, sans-serif';
        ctx.fillText('🚫 SKY REGION ELIMINATED (35% Pixel Savings)', 16, 24);

        // Hood Mask (Bottom 15%)
        ctx.fillStyle = 'rgba(2, 6, 23, 0.85)';
        ctx.fillRect(0, h * 0.85, w, h * 0.15);

        ctx.fillStyle = 'rgba(244, 63, 94, 0.4)';
        ctx.fillText('🚫 VEHICLE HOOD ELIMINATED (15% Pixel Savings)', 16, h - 12);

        // Active Camera ROI Boundary
        ctx.strokeStyle = '#4ade80';
        ctx.setLineDash([4, 4]);
        ctx.strokeRect(0, h * 0.35, w, h * 0.50);
        ctx.setLineDash([]);

        ctx.fillStyle = '#4ade80';
        ctx.fillText('✅ ACTIVE CAM ROI (50% Retained)', w - 210, h * 0.38);

        // 2. Farnebäck Optical Flow Bounding Boxes (Moving Dynamic Targets)
        const motionX1 = w * 0.35 + Math.sin(this.frame * 0.03) * 30;
        const motionY1 = h * 0.52;

        ctx.strokeStyle = '#f43f5e';
        ctx.lineWidth = 2;
        ctx.strokeRect(motionX1, motionY1, 60, 40);

        ctx.fillStyle = '#f43f5e';
        ctx.fillRect(motionX1, motionY1 - 18, 120, 18);
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 10px monospace';
        ctx.fillText('FLOW GATED: CAR', motionX1 + 4, motionY1 - 5);

        // 3-Ring Crop Resolutions
        ctx.strokeStyle = '#38bdf8';
        ctx.strokeRect(w * 0.2, h * 0.65, w * 0.6, h * 0.18);
        ctx.fillStyle = '#38bdf8';
        ctx.fillText('NEAR CROP: 1.0x Full Res', w * 0.2 + 6, h * 0.65 + 16);
    }
}

// Global Engine Instance & View Handlers
let engineInstance = null;

window.addEventListener('DOMContentLoaded', () => {
    engineInstance = new UnifiedTeleopEngine();
});

function setDashboardView(mode) {
    document.querySelectorAll('.view-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`btn-view-${mode}`).classList.add('active');
    if (engineInstance) {
        engineInstance.setViewMode(mode);
    }
}

function updatePointDensity(val) {
    if (engineInstance) {
        engineInstance.setPointDensity(val);
    }
}

function toggleSimulation() {
    if (engineInstance) {
        engineInstance.toggleSimulation();
    }
}
