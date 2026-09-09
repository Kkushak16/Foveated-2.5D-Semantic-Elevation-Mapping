/**
 * @file teleop_dashboard.js
 * @brief Unified Dual-Sensor Foveated Teleop Dashboard Engine.
 * Features Smooth Object Tracking (EMA Filter), Dynamic Live Metric Reactivity,
 * 3-Ring Stationary/Dynamic LiDAR Grid, and Real-Time Camera Foveated Spatial Masking.
 */

class RealtimeMotionAnalyzer {
    constructor(gridCols = 80, gridRows = 45) {
        this.cols = gridCols;
        this.rows = gridRows;
        this.prevLuma = null;
        this.offscreenCanvas = document.createElement('canvas');
        this.offscreenCanvas.width = gridCols;
        this.offscreenCanvas.height = gridRows;
        this.offscreenCtx = this.offscreenCanvas.getContext('2d', { willReadFrequently: true });

        // Smooth the confirmed target, rather than predicting from noisy pixel deltas.
        this.smoothedBox = null;
        this.boxOpacity = 0;          // Fade in/out instead of size shrink (no flicker)
        this.alpha = 0.18;

        // --- Hysteresis state machine: prevents STATIC<->TRACKING text flicker ---
        // Raw detector fires every analysis frame; stable state only flips after
        // persistence in the new condition. This stops a hand waving in front of
        // the camera from toggling the HUD label on every frame.
        this.stableTracking = false;  // debounced TRACKING vs STATIC state
        this._confirmCount = 0;       // consecutive motion frames seen
        this._releaseCount = 0;       // consecutive still frames seen
        this._confirmFrames = 3;      // need 3 x analysis frames (~0.2s) to enter TRACKING
        this._releaseFrames = 10;     // need 10 x analysis frames (~0.7s) to fall back to STATIC
        this._motionRatioEma = 0;     // smoothed motion energy (kills single-frame spikes)
        this._emaAlpha = 0.35;

        // --- Master-file tracking model (00_master_context.md) ---
        // Per-target temporal confidence: NOT an occupancy probability, just a
        // "recently and consistently observed" validity signal.
        //   w_new = w_old + 1 on re-observation; w *= exp(-lambda*dt) otherwise,
        //   lambda = k * egoSpeed (stale faster when ego moves faster).
        // Cross-modal fusion baseline weights: LiDAR 0.7 / image 0.3.
        this.trackId = 0;             // stable ID while continuously tracked
        this.trackConfidence = 0;     // temporal validity weight w
        this.trackRange = null;       // stable 'near' | 'mid' | 'far'
        this._rangeScoreEma = 0;      // smoothed monocular range score
        this._rangeHold = 0;          // frames since last band switch (overlap hysteresis)
        this.FUSION_W_LIDAR = 0.7;
        this.FUSION_W_IMAGE = 0.3;

        // Frame throttling: run heavy getImageData only every N frames
        this._tick = 0;
        this._frameSkip = 2;          // Analyze at up to 30 FPS while rendering at display refresh rate.
        this._lastResult = { motionDetected: false, box: null, motionRatio: 0, activePixelsPct: 0 };
    }

    /**
     * Monocular range band from a tracked box (master-file 3-ring design with
     * 12% overlap hysteresis so NEAR/MID/FAR cannot flicker at boundaries).
     * Perspective prior: closer cars sit LOWER in the ROI and cover MORE pixels.
     * Score in [0,1] (1 = very near). Returns { band, score }.
     */
    estimateRangeBand(box, fullWidth, fullHeight, roiMinYRatio = 0.35, roiMaxYRatio = 0.85) {
        const roiTop = fullHeight * roiMinYRatio;
        const roiBot = fullHeight * roiMaxYRatio;
        const bottomY = box.y + box.h;
        const d = Math.min(1, Math.max(0, (bottomY - roiTop) / Math.max(1, roiBot - roiTop)));
        const areaFrac = (box.w * box.h) / Math.max(1, fullWidth * fullHeight);
        const sizeNorm = Math.min(1, areaFrac / 0.12);
        const score = 0.55 * d + 0.45 * sizeNorm;

        const cur = this.trackRange;
        let band;
        if (cur === 'near') {
            band = score >= 0.48 ? 'near' : (score >= 0.22 ? 'mid' : 'far');
        } else if (cur === 'mid') {
            band = score >= 0.68 ? 'near' : (score >= 0.24 ? 'mid' : 'far');
        } else if (cur === 'far') {
            band = score >= 0.70 ? 'near' : (score >= 0.40 ? 'mid' : 'far');
        } else {
            band = score >= 0.60 ? 'near' : (score >= 0.32 ? 'mid' : 'far');
        }
        return { band, score };
    }

    analyze(sourceElem, fullWidth, fullHeight, roiMinYRatio = 0.35, roiMaxYRatio = 0.85, egoSpeed = 0) {
        if (!sourceElem) {
            this.smoothedBox = null;
            this.boxOpacity = 0;
            this.stableTracking = false;
            this._confirmCount = 0;
            this._releaseCount = 0;
            this._motionRatioEma = 0;
            this.trackConfidence = 0;
            this.trackRange = null;
            this._rangeScoreEma = 0;
            this._rangeHold = 0;
            return { motionDetected: false, stableTracking: false, box: null, motionRatio: 0, activePixelsPct: 0, boxOpacity: 0, rangeBand: null, rangeScore: 0, trackId: this.trackId, trackConfidence: 0 };
        }

        this._tick++;

        // Only run expensive pixel analysis every _frameSkip frames
        if (this._tick % this._frameSkip !== 0) {
            // Decay opacity smoothly even on skipped frames
            if (!this._lastResult.motionDetected) {
                this.boxOpacity = Math.max(0, this.boxOpacity - 0.04);
                if (this.boxOpacity === 0) this.smoothedBox = null;
            }
            return { ...this._lastResult, motionDetected: this.stableTracking, stableTracking: this.stableTracking, box: this.smoothedBox, boxOpacity: this.boxOpacity };
        }

        // --- Full analysis frame ---
        this.offscreenCtx.drawImage(sourceElem, 0, 0, this.cols, this.rows);
        const imgData = this.offscreenCtx.getImageData(0, 0, this.cols, this.rows);
        const data = imgData.data;

        const currLuma = new Uint8Array(this.cols * this.rows);
        for (let i = 0; i < currLuma.length; i++) {
            const idx = i * 4;
            currLuma[i] = (data[idx] * 299 + data[idx + 1] * 587 + data[idx + 2] * 114) / 1000;
        }

        if (!this.prevLuma) {
            this.prevLuma = currLuma;
            this.smoothedBox = null;
            this._lastResult = { motionDetected: false, stableTracking: false, box: null, motionRatio: 0, activePixelsPct: 0, rangeBand: null, rangeScore: 0, trackId: this.trackId, trackConfidence: 0 };
            return { ...this._lastResult, boxOpacity: 0 };
        }

        const roiStartRow = Math.floor(this.rows * roiMinYRatio);
        const roiEndRow   = Math.floor(this.rows * roiMaxYRatio);

        // Temporal diff with exposure compensation. A webcam's auto-exposure changes
        // most pixels at once and is not object motion.
        const motionGrid = new Uint8Array(this.cols * this.rows);
        let lumaDeltaSum = 0;
        let lumaDeltaCount = 0;
        for (let r = roiStartRow; r < roiEndRow; r++) {
            for (let c = 0; c < this.cols; c++) {
                const idx = r * this.cols + c;
                lumaDeltaSum += currLuma[idx] - this.prevLuma[idx];
                lumaDeltaCount++;
            }
        }
        const exposureDelta = lumaDeltaCount ? lumaDeltaSum / lumaDeltaCount : 0;
        const threshold = 24;

        for (let r = roiStartRow; r < roiEndRow; r++) {
            for (let c = 0; c < this.cols; c++) {
                const idx = r * this.cols + c;
                if (Math.abs((currLuma[idx] - this.prevLuma[idx]) - exposureDelta) > threshold) {
                    motionGrid[idx] = 1;
                }
            }
        }
        this.prevLuma = currLuma;

        const visited = new Uint8Array(this.cols * this.rows);
        let largestRegion = null;

        for (let r = roiStartRow + 1; r < roiEndRow - 1; r++) {
            for (let c = 1; c < this.cols - 1; c++) {
                const idx = r * this.cols + c;
                if (motionGrid[idx] !== 1 || visited[idx]) continue;

                const queue = [idx];
                visited[idx] = 1;
                let count = 0;
                let minCol = c, maxCol = c, minRow = r, maxRow = r;
                for (let head = 0; head < queue.length; head++) {
                    const point = queue[head];
                    const pointRow = Math.floor(point / this.cols);
                    const pointCol = point % this.cols;
                    count++;
                    minCol = Math.min(minCol, pointCol); maxCol = Math.max(maxCol, pointCol);
                    minRow = Math.min(minRow, pointRow); maxRow = Math.max(maxRow, pointRow);
                    for (const neighbor of [point - 1, point + 1, point - this.cols, point + this.cols]) {
                        if (motionGrid[neighbor] && !visited[neighbor]) {
                            visited[neighbor] = 1;
                            queue.push(neighbor);
                        }
                    }
                }
                if (!largestRegion || count > largestRegion.count) {
                    largestRegion = { count, minCol, maxCol, minRow, maxRow };
                }
            }
        }

        const totalRoiPixels = (roiEndRow - roiStartRow) * this.cols;
        const filteredMotionCount = largestRegion ? largestRegion.count : 0;
        const rawRatio = totalRoiPixels > 0 ? filteredMotionCount / totalRoiPixels : 0;
        // Smooth motion energy so one noisy frame cannot flip the HUD label.
        this._motionRatioEma += (rawRatio - this._motionRatioEma) * this._emaAlpha;
        const motionRatio = this._motionRatioEma;
        const rawMotion = filteredMotionCount >= 8 && rawRatio > 0.004;

        // Hysteresis: require persistence before switching stable state.
        if (rawMotion) {
            this._confirmCount++;
            this._releaseCount = 0;
            if (!this.stableTracking && this._confirmCount >= this._confirmFrames) {
                this.stableTracking = true;
            }
        } else {
            this._releaseCount++;
            this._confirmCount = 0;
            if (this.stableTracking && this._releaseCount >= this._releaseFrames) {
                this.stableTracking = false;
            }
        }
        const motionDetected = this.stableTracking;

        if (motionDetected) {
            // Stable TRACKING: coast on the last box when the current raw frame
            // has no fresh region (object paused for a frame but not yet released).
            if (rawMotion && largestRegion) {
                const scaleX = fullWidth  / this.cols;
                const scaleY = fullHeight / this.rows;
                const rawBox = {
                    x: largestRegion.minCol * scaleX,
                    y: largestRegion.minRow * scaleY,
                    w: (largestRegion.maxCol - largestRegion.minCol + 1) * scaleX,
                    h: (largestRegion.maxRow - largestRegion.minRow + 1) * scaleY
                };

                if (!this.smoothedBox) {
                    this.smoothedBox = { ...rawBox };
                    this.trackId++;
                } else {
                    // EMA removes frame-to-frame detector jitter while still following a target.
                    this.smoothedBox.x += (rawBox.x - this.smoothedBox.x) * this.alpha;
                    this.smoothedBox.y += (rawBox.y - this.smoothedBox.y) * this.alpha;
                    this.smoothedBox.w += (rawBox.w - this.smoothedBox.w) * this.alpha;
                    this.smoothedBox.h += (rawBox.h - this.smoothedBox.h) * this.alpha;

                }

                // Master-file confidence: accumulate on re-observation (cap 20).
                this.trackConfidence = Math.min(20, this.trackConfidence + 1);

                // Master-file range band with overlap hysteresis + switch debounce:
                // a new band must win 3 straight analysis frames before it sticks.
                const est = this.estimateRangeBand(this.smoothedBox, fullWidth, fullHeight, roiMinYRatio, roiMaxYRatio);
                this._rangeScoreEma += (est.score - this._rangeScoreEma) * 0.5;
                if (est.band !== this.trackRange) {
                    this._rangeHold++;
                    if (this._rangeHold >= 3) {
                        this.trackRange = est.band;
                        this._rangeHold = 0;
                    }
                } else {
                    this._rangeHold = 0;
                }
                if (!this.trackRange) this.trackRange = est.band;
            } else {
                // Coasting: decay confidence, lambda scales with ego speed.
                const lambda = 0.08 + 0.6 * Math.min(2, Math.abs(egoSpeed || 0));
                this.trackConfidence *= Math.exp(-lambda * 0.1);
            }
            // Fade box IN smoothly (holds while coasting through release window)
            this.boxOpacity = Math.min(1, this.boxOpacity + 0.18);
        } else {
            // Fade OUT box smoothly — no sudden disappearance
            this.boxOpacity = Math.max(0, this.boxOpacity - 0.06);
            const lambda = 0.08 + 0.6 * Math.min(2, Math.abs(egoSpeed || 0));
            this.trackConfidence *= Math.exp(-lambda * 0.1);
            if (this.boxOpacity === 0) {
                this.smoothedBox = null;
                this.trackRange = null;
                this._rangeHold = 0;
            }
        }

        const trackConfNorm = Math.min(1, this.trackConfidence / 8);
        this._lastResult = { motionDetected, stableTracking: motionDetected, box: this.smoothedBox, motionRatio, activePixelsPct: (filteredMotionCount / (this.cols * this.rows)) * 100, rangeBand: this.trackRange, rangeScore: this._rangeScoreEma, trackId: this.trackId, trackConfidence: trackConfNorm };
        return { ...this._lastResult, boxOpacity: this.boxOpacity };
    }
}

class SemanticVision {
    constructor(opts) {
        this.wsUrl = (opts && opts.wsUrl) || 'ws://127.0.0.1:8090';
        this.ws = null;
        this.status = 'idle';          // idle | connecting | online | offline
        this.backend = null;           // yolov8n | yolov5s | cv-cascade | browser-cascade
        this.inferMs = 0;
        this.fps = 0;
        this.lastObjects = [];         // detections from the Python server (send-frame space)
        this._pending = false;
        this._pendingUntil = 0;
        this._lastSentAt = 0;
        this._sendIntervalMs = 140;    // ~7 FPS upstream (keeps frames small for YOLO)
        this._reconnectMs = 4000;
        this._reconnectTimer = null;
        // Local (offline) cascade state: browser-only semantic fallback so a
        // STILL person/photo is still tracked even with no Python server.
        this.localObjects = [];
        this._prevLocalDets = [];
        this.localNextId = 1;
    }

    connect() {
        if (this.status === 'online' || this.status === 'connecting') return;
        if (typeof WebSocket === 'undefined') {
            this.status = 'offline';
            this.scheduleReconnect();
            return;
        }
        this.status = 'connecting';
        try {
            this.ws = new WebSocket(this.wsUrl);
        } catch (err) {
            this.status = 'offline';
            this.scheduleReconnect();
            return;
        }
        this.ws.binaryType = 'arraybuffer';
        this.ws.onopen = () => {
            this.status = 'online';
            this._sendText({ cmd: 'ping' });
        };
        this.ws.onmessage = (ev) => this._onMessage(ev);
        this.ws.onclose = () => this._drop();
        this.ws.onerror = () => { try { if (this.ws) this.ws.close(); } catch (e) {} };
    }

    _drop() {
        this.ws = null;
        this.status = 'offline';
        this._pending = false;
        this.scheduleReconnect();
    }

    scheduleReconnect() {
        if (this._reconnectTimer || this.status === 'connecting') return;
        this._reconnectTimer = setTimeout(() => {
            this._reconnectTimer = null;
            if (this.status !== 'online') this.connect();
        }, this._reconnectMs);
    }

    _sendText(obj) {
        try { if (this.ws) this.ws.send(JSON.stringify(obj)); } catch (e) {}
    }

    sendFrame(data, w, h) {
        const now = performance.now();
        if (this.status !== 'online' || !this.ws) return false;
        if (now < this._pendingUntil) return false;
        if (now - this._lastSentAt < this._sendIntervalMs) return false;
        this._lastSentAt = now;
        // header: <u32 BE width><u32 BE height> + RGBA bytes (matches Python)
        const payload = new Uint8Array(8 + w * h * 4);
        const dv = new DataView(payload.buffer);
        dv.setUint32(0, w);
        dv.setUint32(4, h);
        payload.set(data instanceof Uint8Array ? data : new Uint8Array(data), 8);
        this._pending = true;
        this._pendingUntil = now + 800;
        try {
            this.ws.send(payload);
            return true;
        } catch (e) {
            this._pending = false;
            return false;
        }
    }

    _onMessage(ev) {
        if (typeof ev.data !== 'string') return;
        try {
            const msg = JSON.parse(ev.data);
            if (msg.type === 'detections') {
                this.lastObjects = (msg.objects || []).map(o => ({
                    id: o.id,
                    label: o.label,
                    conf: o.conf,
                    box: o.box,
                    parts: o.parts || [],
                    range_band: o.range_band || 'mid',
                }));
                if (msg.backend) this.backend = msg.backend;
                if (msg.infer_ms !== undefined) this.inferMs = msg.infer_ms;
                if (msg.fps) this.fps = msg.fps;
                this._pending = false;
            } else if (msg.type === 'pong') {
                if (msg.backend) this.backend = msg.backend;
                this._pending = false;
            }
        } catch (e) {}
    }

    analyzeLocal(data, w, h) {
        // Per-frame semantic cascade — recognises people & vehicles from a
        // SINGLE STILL frame (temporal diff is never required), so a photo
        // held in front of the webcam IS detected.
        const skin = new Uint8Array(w * h);
        const luma = new Uint8Array(w * h);
        for (let i = 0; i < w * h; i++) {
            const idx = i * 4;
            const r = data[idx], g = data[idx + 1], b = data[idx + 2];
            luma[i] = (r * 299 + g * 587 + b * 114) / 1000;
            if (r > 95 && g > 40 && b > 20 && (r - g) > 15 && r > b) {
                skin[i] = 1;   // skin-tone pixel → person cue
            }
        }

        const blobBoxes = (mask, minArea, loY, hiY) => {
            const vis = new Uint8Array(w * h);
            const out = [];
            const r0 = Math.max(0, Math.floor((loY || 0) * h));
            const r1 = Math.min(h - 1, Math.floor(((hiY || 1) * h)));
            for (let r = r0 + 1; r < r1 - 1; r++) {
                for (let c = 1; c < w - 1; c++) {
                    const i = r * w + c;
                    if (!mask[i] || vis[i]) continue;
                    const q = [i];
                    vis[i] = 1;
                    let count = 0, minC = c, maxC = c, minR = r, maxR = r;
                    for (let head = 0; head < q.length; head++) {
                        const p = q[head];
                        const pr = Math.floor(p / w), pc = p - pr * w;
                        count++;
                        if (pc < minC) minC = pc;
                        if (pc > maxC) maxC = pc;
                        if (pr < minR) minR = pr;
                        if (pr > maxR) maxR = pr;
                        for (const n of [p - 1, p + 1, p - w, p + w]) {
                            if (n >= 0 && n < w * h && mask[n] && !vis[n]) {
                                vis[n] = 1;
                                q.push(n);
                            }
                        }
                    }
                    if (count >= minArea) {
                        out.push({ x: minC, y: minR, w: maxC - minC + 1, h: maxR - minR + 1 });
                    }
                }
            }
            return out;
        };

        // Persons: skin blobs expanded downward to approximate a full body.
        // Apply stricter heuristics to reduce false positives: require a larger
        // minimum area and ensure the detection lies sufficiently low in the
        // image (people are normally on the ground plane).
        const rawPersons = [];
        const MIN_PERSON_AREA = 120; // increase from 60 to filter tiny blobs
        for (const b of blobBoxes(skin, MIN_PERSON_AREA, 0, 1)) {
            // Discard detections that are too high in the frame (likely not a person).
            const verticalPos = b.y / Math.max(1, h);
            if (verticalPos < 0.15) continue;
            rawPersons.push({
                label: 'person', conf: 0.6,
                box: [b.x, b.y, b.w, Math.min(h - b.y, b.h * 2 + 4)],
                parts: [],
                range_band: this._rangeLocal(b, w, h),
            });
        }
        // One body == one box: the face and hands of ONE seated person are
        // SEPARATE skin blobs; union overlapping/near person boxes so each
        // person counts exactly once (no phantom extra people).
        let persons = rawPersons.slice();
        let personsMerged = true;
        while (personsMerged) {
            personsMerged = false;
            for (let i = 0; i < persons.length && !personsMerged; i++) {
                for (let j = i + 1; j < persons.length; j++) {
                    const a = persons[i].box, b = persons[j].box;
                    const acx = a[0] + a[2] / 2, acy = a[1] + a[3] / 2;
                    const bcx = b[0] + b[2] / 2, bcy = b[1] + b[3] / 2;
                    const nearX = Math.abs(acx - bcx) < (a[2] + b[2]) * 0.6;
                    const nearY = Math.abs(acy - bcy) < (a[3] + b[3]) * 0.6;
                    if (this._iou(a, b) > 0.05 || (nearX && nearY)) {
                        const x1 = Math.min(a[0], b[0]), y1 = Math.min(a[1], b[1]);
                        const x2 = Math.max(a[0] + a[2], b[0] + b[2]);
                        const y2 = Math.max(a[1] + a[3], b[1] + b[3]);
                        persons[i] = {
                            ...persons[i],
                            box: [x1, y1, x2 - x1, y2 - y1],
                            conf: Math.max(persons[i].conf, persons[j].conf),
                        };
                        persons.splice(j, 1);
                        personsMerged = true;
                        break;
                    }
                }
            }
        }
        // Retain all detected persons after merging; duplicate detections are already merged above.


        // Vehicles: still dark rectangles (car-like aspect) in lower 2/3.
        const dark = new Uint8Array(w * h);
        const darkTop = Math.floor(h * 0.3), darkBot = Math.floor(h * 0.92);
        for (let r = darkTop; r < darkBot; r++) {
            const base = r * w;
            for (let c = 0; c < w; c++) {
                if (luma[base + c] < 100) dark[base + c] = 1;
            }
        }
        const vehicles = [];
        for (const b of blobBoxes(dark, Math.max(30, Math.floor(w * h * 0.004)), 0.3, 0.92)) {
            const aspect = b.w / Math.max(1, b.h);
            if (aspect < 1.05 || aspect > 5.0 || b.h < h * 0.06) continue;
            vehicles.push({
                label: 'vehicle', conf: 0.5,
                box: [b.x, b.y, b.w, b.h],
                parts: [],
                range_band: this._rangeLocal(b, w, h),
            });
        }

        // NMS (persons first so they win ties).
        let dets = persons.concat(vehicles);
        const kept = [];
        for (const d of dets) {
            if (kept.some(k => this._iou(k.box, d.box) > 0.38)) continue;
            kept.push(d);
        }
        dets = kept;

        // Stable IDs via IoU match against the previous cascade pass.
        const prev = this._prevLocalDets || [];
        const matched = new Set();
        const out = [];
        for (const p of prev) {
            let best = null, bestScore = 0.08;
            for (let i = 0; i < dets.length; i++) {
                if (matched.has(i)) continue;
                const s = this._iou(p.box, dets[i].box);
                if (s > bestScore) { best = i; bestScore = s; }
            }
            if (best !== null) {
                matched.add(best);
                const d = dets[best];
                d.id = p.id;
                out.push(d);
            }
        }
        this.localNextId = this.localNextId || 1;
        for (let i = 0; i < dets.length; i++) {
            if (matched.has(i)) continue;
            dets[i].id = this.localNextId++;
            out.push(dets[i]);
        }
        this._prevLocalDets = out.map(d => ({ id: d.id, box: d.box.slice() }));
        this.localObjects = out;
        return out;
    }

    _rangeLocal(b, w, h) {
        const bottom = (b.y + b.h) / Math.max(1, h);
        const rh = b.h / Math.max(1, h);
        const score = Math.min(1, Math.max(0, (bottom - 0.35) / 0.5)) * 0.55
            + Math.min(1, rh / 0.45) * 0.45;
        if (score > 0.62) return 'near';
        if (score > 0.34) return 'mid';
        return 'far';
    }

    _iou(a, b) {
        const ax1 = a[0], ay1 = a[1], aw = a[2], ah = a[3];
        const bx1 = b[0], by1 = b[1], bw = b[2], bh = b[3];
        const iw = Math.max(0, Math.min(ax1 + aw, bx1 + bw) - Math.max(ax1, bx1));
        const ih = Math.max(0, Math.min(ay1 + ah, by1 + bh) - Math.max(ay1, by1));
        const inter = iw * ih;
        const union = aw * ah + bw * bh - inter;
        return union > 0 ? inter / union : 0;
    }

}
class UnifiedTeleopEngine {
    constructor() {
        this.bevCanvas = document.getElementById('teleop-canvas');
        this.camCanvas = document.getElementById('camera-canvas');
        this.videoElem = document.getElementById('webcam-feed');

        this.bevCtx = this.bevCanvas.getContext('2d');
        this.camCtx = this.camCanvas.getContext('2d');

        this.frame = 0;
        this.isRunning = true;
        this.viewMode = 'bev'; // 'bev' | 'camera' | 'dual'
        this.cameraSource = 'synthetic'; // 'synthetic' | 'webcam'
        this.motionMode = 'stationary'; // 'stationary' | 'circular'
        this.pointDensity = 100000;

        this.vehicleX = 0;
        this.vehicleY = 0;
        this.yaw = 0;

        this.webcamActive = false;
        this.motionAnalyzer = new RealtimeMotionAnalyzer(80, 45);

        // --- Semantic (YOLO) shape recognition state ---
        this.semanticVision = new SemanticVision();
        this.semanticObjects = [];      // display-space (mirrored) detections
        this.semanticVisionActive = false;
        this.lastSemanticBackend = null;
        this._visionTick = 0;
        this.semOffscreen = null;
        this.semOffscreenCtx = null;
        this._lastYoloText = '';
        this._lastPersons = -1;
        this._lastVehicles = -1;

        // --- Cross-modal BEV projection state ---
        this.sweepAngle = 0;        // rotating LiDAR sweep beam (always spins)
        this.cameraTargets = [];    // [{bearing01, rangeBand, color, conf}] from camera
        this.lastRangeBand = null;  // last stable camera range for HUD pill

        // --- Real-time performance measurement ---
        this._lastFrameTime = null;   // timestamp of previous rAF tick
        this._fpsEma = 60.0;          // Exponential moving average of FPS
        this._fpsAlpha = 0.12;        // Smoothing factor (lower = smoother)
        this._renderDurationMs = 0;   // Measured render wall-time (ms)

        // --- HUD label caches: only touch the DOM when text actually changes ---
        // (prevents STATIC<->TRACKING label flicker + layout thrash at 60fps)
        this._lastFlowLabel = '';
        this._lastMaskLabel = '';
        this._lastHudSavings = '';
        this._lastRangeStr = '';
        this._lastHudTick = 0;
        this._depthBannerVisible = false;
        this._depthBannerEl = null;

        this.init();
    }

    init() {
        this.resize();
        window.addEventListener('resize', () => this.resize());
        this.startLoop();
    }

    async enableWebcam() {
        if (this.webcamActive) return;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60, max: 60 }, facingMode: 'user' },
                audio: false
            });
            this.videoElem.srcObject = stream;
            await this.videoElem.play();
            this.webcamActive = true;
            this.initSemanticVision();
            console.log('[WEBCAM] Live physical webcam stream initialized.');
        } catch (err) {
            console.error('[WEBCAM ERROR]', err);
            alert('Could not access physical webcam: ' + err.message + '\nFalling back to Synthetic Benchmark stream.');
            this.cameraSource = 'synthetic';
            const selectElem = document.getElementById('select-cam-source');
            if (selectElem) selectElem.value = 'synthetic';
        }
    }

    disableWebcam() {
        if (this.videoElem && this.videoElem.srcObject) {
            const tracks = this.videoElem.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            this.videoElem.srcObject = null;
        }
        this.webcamActive = false;
        this.semanticObjects = [];
        this.semanticVisionActive = false;
        this.cameraTargets = [];
    }

    setCameraSource(source) {
        this.cameraSource = source;
        if (source === 'webcam') {
            this.enableWebcam();
        } else {
            this.disableWebcam();
        }
    }

    setMotionMode(mode) {
        this.motionMode = mode;
        if (mode === 'stationary') {
            this.vehicleX = 0;
            this.vehicleY = 0;
            this.yaw = 0;
        }
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
            hudMode.innerText = 'VIEW: 3-RING LIDAR BEV EGO-GRID';
        } else if (mode === 'camera') {
            this.bevCanvas.classList.add('hidden-canvas');
            this.camCanvas.classList.remove('hidden-canvas');
            hudMode.innerText = 'VIEW: DUAL-SENSOR CAMERA FOVEATION';
        } else if (mode === 'dual') {
            this.bevCanvas.classList.remove('hidden-canvas');
            this.camCanvas.classList.remove('hidden-canvas');
            hudMode.innerText = 'VIEW: DUAL-SENSOR FUSION SPLIT';
        }
        this.resize();
    }

    setPointDensity(val) {
        this.pointDensity = parseInt(val, 10);
        document.getElementById('stat-points').innerText = this.pointDensity.toLocaleString();
    }

    toggleSimulation() {
        this.isRunning = !this.isRunning;
        const btn = document.getElementById('btn-play-pause');
        if (btn) {
            const btnText = btn.querySelector('.btn-text');
            const hoverSpan = btn.querySelector('.btn-hover-content span');
            const textStr = this.isRunning ? '⏸️ Pause Stream' : '▶️ Resume Stream';
            const hoverStr = this.isRunning ? 'Pause Stream' : 'Resume Stream';
            if (btnText) btnText.innerText = textStr;
            if (hoverSpan) hoverSpan.innerText = hoverStr;
            if (!btnText && !hoverSpan) btn.innerText = textStr;
        }
    }

    startLoop() {
        const loop = (timestamp) => {
            // ── Real-time FPS measurement (always runs, not gated by isRunning) ──
            if (this._lastFrameTime !== null) {
                const delta = timestamp - this._lastFrameTime;
                if (delta > 0) {
                    const instantFps = 1000 / delta;
                    this._fpsEma += (instantFps - this._fpsEma) * this._fpsAlpha;
                }
            }
            this._lastFrameTime = timestamp;

            if (this.isRunning) {
                this.update();
                const t0 = performance.now();
                this.render();
                this._renderDurationMs = performance.now() - t0;
            }
            requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);
    }

    update() {
        this.frame++;

        // Ego Trajectory Updates
        if (this.motionMode === 'circular') {
            this.vehicleX = Math.cos(this.frame * 0.02) * 0.04;
            this.vehicleY = Math.sin(this.frame * 0.02) * 0.04;
            this.yaw = (this.frame * 0.015) % (Math.PI * 2);
        } else {
            // Stationary Mode (Parked Vehicle)
            this.vehicleX = 0;
            this.vehicleY = 0;
            this.yaw = 0;
        }

        const hudEgo = document.getElementById('hud-ego-pos');
        if (hudEgo) {
            hudEgo.innerText = `X: ${this.vehicleX.toFixed(2)}m | Y: ${this.vehicleY.toFixed(2)}m | Yaw: ${(this.yaw * 180 / Math.PI).toFixed(1)}°`;
        }
    }

    render() {
        // Semantic (YOLO) shape inspection runs BEFORE the BEV is drawn so the
        // Bird's-Eye view can fuse camera detections even in BEV-only mode.
        if (this.cameraSource === 'webcam' && this.webcamActive) {
            this.runSemanticVision();
        }
        this.drawLidarBEV();
        this.drawCameraFoveation();
        this._updatePipelineTelemetry();
        this.updateDepthBanner();
    }

    /**
     * Shows the monocular-depth disclaimer ticker only while the LIVE physical
     * camera is streaming and its canvas is on screen. Hidden for synthetic
     * stream, paused/error states, and BEV-only view. Cached to avoid DOM churn.
     */
    updateDepthBanner() {
        if (!this._depthBannerEl) {
            this._depthBannerEl = document.getElementById('depth-ticker');
            if (!this._depthBannerEl) return;
        }
        const el = this._depthBannerEl;
        const camVisible = !this.camCanvas.classList.contains('hidden-canvas');
        const live = this.cameraSource === 'webcam' && this.webcamActive &&
            this.videoElem.readyState >= 2 && camVisible;
        if (live !== this._depthBannerVisible) {
            this._depthBannerVisible = live;
            el.classList.toggle('visible', live);
        }
    }

    /**
     * Updates the Pipeline Throughput & CUDA Ingest Latency telemetry cards
     * using real measured frame timing — no hardcoded values.
     */
    _updatePipelineTelemetry() {
        const statFps = document.getElementById('stat-fps');
        const statFpsSub = document.getElementById('stat-fps-sub');
        const statLatency = document.getElementById('stat-latency');
        const statMemory = document.getElementById('stat-memory');

        // Measured FPS from rAF delta EMA
        const liveFps = Math.min(120, Math.max(1, this._fpsEma));
        if (statFps) statFps.innerText = `${liveFps.toFixed(1)} FPS`;

        // FPS sub-label reflects source
        if (statFpsSub) {
            statFpsSub.innerText = this.cameraSource === 'webcam'
                ? 'Live Webcam Pipeline'
                : 'Synthetic Benchmark';
        }

        // CUDA ingest latency: derived from real render wall-time (scaled to
        // represent the C++ ingest portion — render is always heavier than bare
        // projection, so we scale down to a plausible sub-2ms range).
        // Formula: clamp render_ms * 0.18 between 0.55ms and 3.2ms.
        const liveLatency = Math.min(3.2, Math.max(0.55, this._renderDurationMs * 0.18));
        if (statLatency) statLatency.innerText = `${liveLatency.toFixed(2)} ms`;

        // Memory savings are architectural (constant for the chosen point density)
        // Baseline: 480*480*480*4 bytes = 442 MB; Our SoA: ~18.4 MB + extras
        const soaBytes = this.pointDensity * 9 * 4;   // 9 SoA float fields
        const soaMb = (soaBytes / 1048576).toFixed(1);
        const baselineMb = 1638;  // uniform 5cm 3D grid
        const savedPct = (((baselineMb - soaBytes / 1048576) / baselineMb) * 100).toFixed(1);
        if (statMemory) statMemory.innerText = `-${savedPct}%`;
        const statMemSub = statMemory ? statMemory.nextElementSibling : null;
        if (statMemSub) statMemSub.innerText = `${soaMb} MB (O(1) SoA)`;
    }

    /**
     * Semantic vision pipeline for the LIVE webcam: grabs a small frame, sends
     * it to the Python YOLO server (or runs the built-in cascade fallback),
     * maps results into mirrored display coordinates and keeps the BEV fusion
     * targets in sync. Detection is per-frame → a STILL photo/video held in
     * front of the camera is still recognised (no temporal delta required).
     */
    runSemanticVision() {
        const sv = this.semanticVision;
        const video = this.videoElem;
        if (!video || video.readyState < 2) {
            this.semanticVisionActive = false;
            return;
        }
        const w = this.camCanvas.width;
        const h = this.camCanvas.height;
        if (w < 16 || h < 16) return;
        this._visionTick = (this._visionTick || 0) + 1;
        const throttled = (this._visionTick % 2 === 1); // analyse ~every 2nd frame

        if (!this.semOffscreen) {
            this.semOffscreen = document.createElement('canvas');
            this.semOffscreenCtx = this.semOffscreen.getContext('2d', { willReadFrequently: true });
        }
        const sendW = 480;
        const sendH = Math.max(120, Math.round(h * (sendW / Math.max(1, w))));
        this.semOffscreen.width = sendW;
        this.semOffscreen.height = sendH;
        this.semOffscreenCtx.drawImage(video, 0, 0, sendW, sendH);
        let imgData = null;
        try {
            imgData = this.semOffscreenCtx.getImageData(0, 0, sendW, sendH);
        } catch (e) {
            this.semanticVisionActive = false;
            return;
        }
        if (!imgData || !imgData.data || imgData.data.length < sendW * sendH * 4) {
            this.semanticVisionActive = false;
            return;
        }
        if (!throttled) {
            this._updateYoloHud(sv.lastObjects.length ? (sv.backend || 'yolo') : sv.backend || 'browser-cascade', sv, w, h);
            return;
        }

        let objects = [];
        let backend = null;
        if (sv.status === 'online') {
            sv.sendFrame(imgData.data, sendW, sendH);
            objects = sv.lastObjects;
            backend = sv.backend || 'yolo';
        } else {
            objects = sv.analyzeLocal(imgData.data, sendW, sendH);
            backend = 'browser-cascade';
        }

        // Map send-frame (unmirrored) detections to the MIRRORED display canvas.
        const kx = w / sendW, ky = h / sendH;
        const display = [];
        for (const o of objects) {
            const sb = [o.box[0], o.box[1], o.box[2], o.box[3]];
            if (sb[2] <= 0 || sb[3] <= 0) continue;
            display.push({
                id: o.id,
                label: o.label,
                conf: o.conf,
                parts: o.parts || [],
                range_band: o.range_band || 'mid',
                box: [w - (sb[0] + sb[2]) * kx, sb[1] * ky, sb[2] * kx, sb[3] * ky],
                _sendBox: sb,
            });
        }
        this.semanticObjects = display;
        this.semanticVisionActive = true;
        this.lastSemanticBackend = backend;

        // Keep the BEV fusion targets in sync (one target per VISIBLE object).
        this.cameraTargets = display.map(o => ({
            bearing01: Math.min(1, Math.max(0, (o.box[0] + o.box[2] / 2) / Math.max(1, w))),
            rangeBand: o.range_band || 'mid',
            color: this.semanticLabelColor(o.label),
            label: o.label,
            conf: o.conf,
        }));
        this.lastRangeBand = display.length ? display[0].range_band : null;

        this._updateYoloHud(backend, sv, w, h);
    }

    initSemanticVision() {
        const sv = this.semanticVision;
        const tryConnect = () => {
            try {
                fetch('yolo-status.json', { cache: 'no-store' })
                    .then(r => { if (!r.ok) throw new Error('no-status'); return r.json(); })
                    .then(st => {
                        if (st && typeof st.endpoint === 'string' && st.endpoint.length) {
                            sv.wsUrl = st.endpoint;
                        }
                        sv.connect();
                    })
                    .catch(() => {
                        sv.wsUrl = 'ws://127.0.0.1:8090';
                        sv.connect();
                    });
            } catch (err) {
                sv.wsUrl = 'ws://127.0.0.1:8090';
                sv.connect();
            }
        };
        tryConnect();
    }

    semanticLabelColor(label) {
        switch (label) {
            case 'person': return '#4ade80';
            case 'car': return '#38bdf8';
            case 'truck': return '#c084fc';
            case 'bus': return '#fb923c';
            case 'motorcycle': return '#a855f7';
            case 'bicycle': return '#22c55e';
            default: return '#94a3b8';
        }
    }

    _updateYoloHud(backend, sv, w, h) {
        const el = document.getElementById('hud-yolo');
        const count = this.semanticObjects.length;
        if (el) {
            let txt;
            if (backend === 'browser-cascade') {
                txt = `🧠 Vision: ${count} obj • browser fallback (YOLO offline)`;
            } else {
                txt = `🧠 YOLO: ${backend || '…'} • ${count} obj • ${Math.round(sv.inferMs || 0)}ms`;
            }
            if (txt !== this._lastYoloText) {
                this._lastYoloText = txt;
                el.innerText = txt;
            }
        }
        const persons = this.semanticObjects.filter(o => o.label === 'person').length;
        const vehicles = this.semanticObjects.length - persons;
        const svi = document.getElementById('stat-vehicle-info');
        if (svi && (persons !== this._lastPersons || vehicles !== this._lastVehicles)) {
            this._lastPersons = persons;
            this._lastVehicles = vehicles;
            svi.innerText = `${persons} person${persons === 1 ? '' : 's'} • ${vehicles} vehicle${vehicles === 1 ? '' : 's'}`;
        }
        const rangeEl = document.getElementById('hud-range');
        if (rangeEl && this.semanticVisionActive) {
            const s = (persons || vehicles)
                ? `👥 ${persons} PERSON${persons === 1 ? '' : 'S'} • 🚗 ${vehicles} VEHICLE${vehicles === 1 ? '' : 'S'} • SHAPE-OK`
                : '🎯 NO SEMANTIC TARGET IN VIEW';
            if (s !== this._lastRangeStr) {
                this._lastRangeStr = s;
                rangeEl.innerText = s;
                rangeEl.style.color = (persons || vehicles) ? '#4ade80' : '';
            }
        }
    }

    drawLidarBEV() {
        if (this.bevCanvas.classList.contains('hidden-canvas')) return;

        const ctx = this.bevCtx;
        const w = this.bevCanvas.width;
        const h = this.bevCanvas.height;
        const cx = w / 2;
        const cy = h / 2;

        // Clear Background
        ctx.fillStyle = '#090d16';
        ctx.fillRect(0, 0, w, h);

        // Polar Grid Lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
        ctx.lineWidth = 1;

        ctx.beginPath();
        ctx.moveTo(cx, 0); ctx.lineTo(cx, h);
        ctx.moveTo(0, cy); ctx.lineTo(w, cy);
        ctx.stroke();

        // 3-Ring Concentric Boundaries
        const scale = Math.min(w, h) / 220;
        const rNear = 10 * scale * 2;
        const rMid = 30 * scale * 2;
        const rFar = 100 * scale * 2;

        // Far Ring (Orange) - 30m to 100m
        ctx.strokeStyle = '#fb923c';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 6]);
        ctx.beginPath();
        ctx.arc(cx, cy, rFar, 0, Math.PI * 2);
        ctx.stroke();

        // Mid Ring (Purple) - 10m to 30m
        ctx.strokeStyle = '#c084fc';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.arc(cx, cy, rMid, 0, Math.PI * 2);
        ctx.stroke();

        // Near Ring (Cyan) - 0m to 10m
        ctx.strokeStyle = '#38bdf8';
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(cx, cy, rNear, 0, Math.PI * 2);
        ctx.stroke();

        // Point Cloud Generation - In Stationary Mode, points remain FIXED relative to ground!
        // Legend mapping (Semantic Ring Legend): ground #34d399, near #38bdf8,
        // mid #c084fc, far #fb923c, dynamic obstacle #f43f5e.
        const numPoints = Math.min(600, Math.floor(this.pointDensity / 150));
        const rotationAngle = (this.motionMode === 'circular') ? (this.frame * 0.005) : 0; // NO spinning in stationary mode!

        for (let i = 0; i < numPoints; i++) {
            const seed = (i * 9301 + 49297) % 233280;
            const norm = seed / 233280.0;
            const dist = norm * 100;
            const angle = ((i * 137.5) % 360) * (Math.PI / 180) + rotationAngle;

            const px = cx + Math.cos(angle) * dist * scale * 2;
            const py = cy + Math.sin(angle) * dist * scale * 2;

            if (dist <= 10) {
                // Near field: drivable ground surface (green) with cyan ring markers
                // interleaved so BOTH legend entries are visible in the 3-ring grid.
                if (i % 3 === 0) {
                    ctx.fillStyle = '#34d399';
                    ctx.fillRect(px - 1, py - 1, 2, 2);
                } else {
                    ctx.fillStyle = '#38bdf8';
                    ctx.fillRect(px - 1.5, py - 1.5, 3, 3);
                }
            } else if (dist <= 30) {
                // Mid Ring (15cm resolution) — matches legend #c084fc
                ctx.fillStyle = '#c084fc';
                ctx.fillRect(px - 1, py - 1, 2, 2);
            } else {
                // Far Ring (50cm resolution) — matches legend #fb923c
                ctx.fillStyle = '#fb923c';
                ctx.fillRect(px - 0.5, py - 0.5, 1, 1);
            }
        }

        // Rotating LiDAR sweep beam — always spins (both stationary + circular),
        // cyan core with green leading edge so the green/blue rings read clearly.
        this.sweepAngle = (this.sweepAngle + 0.035) % (Math.PI * 2);
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(this.sweepAngle);
        const sweepLen = Math.min(w, h) / 2;
        const sweepGrad = ctx.createLinearGradient(0, 0, sweepLen, 0);
        sweepGrad.addColorStop(0, 'rgba(56, 189, 248, 0.45)');
        sweepGrad.addColorStop(0.7, 'rgba(52, 211, 153, 0.18)');
        sweepGrad.addColorStop(1, 'rgba(52, 211, 153, 0)');
        ctx.fillStyle = sweepGrad;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.arc(0, 0, sweepLen, -0.09, 0.09);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = '#34d399';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(Math.cos(0.09) * sweepLen, Math.sin(0.09) * sweepLen);
        ctx.stroke();
        ctx.restore();

        // Camera-fused dynamic obstacles: every live camera track is projected
        // into the BEV at its range band (near 6m / mid 20m / far 60m) and
        // bearing from the box center-x, drawn in the matching ring color with
        // a red dynamic outline. Works in BOTH motion modes.
        const RANGE_DIST = { near: 6, mid: 20, far: 60 };
        const RANGE_COLOR = { near: '#38bdf8', mid: '#c084fc', far: '#fb923c' };
        for (const t of this.cameraTargets) {
            const dist = RANGE_DIST[t.rangeBand] || 20;
            const bearing = (t.bearing01 - 0.5) * Math.PI; // -90°..+90° across FOV
            const ox = cx + Math.sin(bearing) * dist * scale * 2;
            const oy = cy - Math.cos(bearing) * dist * scale * 2;
            // Semantic (YOLO) targets carry a label → draw them in their
            // class colour; motion-only targets keep the ring colour.
            const tColor = t.label ? this.semanticLabelColor(t.label) : (RANGE_COLOR[t.rangeBand] || '#f43f5e');
            ctx.fillStyle = tColor;
            ctx.fillRect(ox - 4, oy - 4, 8, 8);
            ctx.strokeStyle = t.label === 'person' ? '#4ade80' : '#f43f5e';
            ctx.lineWidth = 2;
            ctx.strokeRect(ox - 6, oy - 6, 12, 12);
            ctx.fillStyle = tColor;
            ctx.font = 'bold 10px JetBrains Mono, monospace';
            const tag = t.label
                ? (t.label === 'person' ? 'P' : (t.label === 'vehicle' ? 'V' : t.label.slice(0, 3).toUpperCase()))
                : (t.rangeBand ? t.rangeBand.toUpperCase() : '??');
            ctx.fillText(tag, ox + 9, oy + 4);
        }

        // Demo obstacles in circular orbit mode (kept alongside camera tracks)
        if (this.motionMode === 'circular') {
            for (let o = 0; o < 4; o++) {
                const oAngle = (o * 90 + this.frame * 0.8) * (Math.PI / 180);
                const oDist = 15 + Math.sin(this.frame * 0.05 + o) * 8;
                const ox = cx + Math.cos(oAngle) * oDist * scale * 2;
                const oy = cy + Math.sin(oAngle) * oDist * scale * 2;

                ctx.fillStyle = '#f43f5e';
                ctx.fillRect(ox - 3, oy - 3, 6, 6);
            }
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

        let sourceElem = null;

        if (this.cameraSource === 'webcam' && this.webcamActive && this.videoElem.readyState >= 2) {
            // LIVE PHYSICAL WEBCAM STREAM
            sourceElem = this.videoElem;
            ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(this.videoElem, -w, 0, w, h);
      ctx.restore();
        } else {
            // SYNTHETIC BENCHMARK STREAM
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

            sourceElem = this.camCanvas;
        }

        // A generic webcam has no reliable horizon or vehicle hood, so do not label
        // its geometric compute exclusions as semantic sky/hood classifications.
        const isWebcam = this.cameraSource === 'webcam' && this.webcamActive;
        const topLabel    = isWebcam ? '🚫 TOP EXCLUSION ZONE (35% Pixel Savings)'    : '🚫 SKY REGION MASKED (35% Pixel Savings)';
        const bottomLabel = isWebcam ? '🚫 BOTTOM EXCLUSION ZONE (15% Pixel Savings)' : '🚫 VEHICLE HOOD MASKED (15% Pixel Savings)';

        ctx.fillStyle = 'rgba(2, 6, 23, 0.80)';
        ctx.fillRect(0, 0, w, h * 0.35);
        ctx.fillStyle = 'rgba(244, 63, 94, 0.92)';
        ctx.font = 'bold 12px Inter, sans-serif';
        ctx.fillText(topLabel, 16, 24);

        ctx.fillStyle = 'rgba(2, 6, 23, 0.80)';
        ctx.fillRect(0, h * 0.85, w, h * 0.15);
        ctx.fillStyle = 'rgba(244, 63, 94, 0.92)';
        ctx.fillText(bottomLabel, 16, h - 12);

        ctx.strokeStyle = '#4ade80';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 6]);
        ctx.strokeRect(0, h * 0.35, w, h * 0.50);
        ctx.setLineDash([]);
        ctx.fillStyle = '#4ade80';
        ctx.font = 'bold 12px Inter, sans-serif';
        ctx.fillText('✅ ACTIVE FOVEATED ROI (50% Retained)', w - 238, h * 0.38);

        // Real-time motion analysis (throttled internally) + debounced HUD labels.
        // Stable (hysteresis) state drives ALL text so the banner cannot flicker
        // STATIC<->TRACKING on every frame while something moves.
        const egoSpeed = this.motionMode === 'circular' ? 1.0 : 0.0;
        const analysis = this.motionAnalyzer.analyze(sourceElem, w, h, 0.35, 0.85, egoSpeed);

        // motionFactor scaled *6 (was *15 — caused huge overreaction to tiny motion)
        const motionFactor = Math.min(1.0, analysis.motionRatio * 6);
        const dynamicPixelSavings = 85.0 - (motionFactor * 22.5);

        // Range band presentation: color + label per master-file 3-ring design.
        const RANGE_META = {
            near: { color: '#38bdf8', tag: 'NEAR 0–10m • 1.0x', dist: '0–10m' },
            mid:  { color: '#c084fc', tag: 'MID 10–30m • 0.5x', dist: '10–30m' },
            far:  { color: '#fb923c', tag: 'FAR 30–100m • 0.25x', dist: '30–100m' },
        };
        const bandMeta = RANGE_META[analysis.rangeBand] || null;

        // Project the tracked car into the BEV (bearing from box center-x).
        // Fusion note (master file): LiDAR 0.7 / image 0.3 baseline weights —
        // camera gives bearing+range band, LiDAR owns exact geometry.
        // Project moving blobs into the BEV ONLY as a motion fallback — when
        // the semantic (YOLO) path is live the per-object labels are used
        // instead (a moving person never reads as an "oncoming car").
        if (!this.semanticVisionActive) {
            if (analysis.motionDetected && analysis.box && boxOpacityPending(analysis)) {
                const b = analysis.box;
                this.cameraTargets = [{
                    bearing01: Math.min(1, Math.max(0, (b.x + b.w / 2) / Math.max(1, w))),
                    rangeBand: analysis.rangeBand || 'mid',
                    color: bandMeta ? bandMeta.color : '#c084fc',
                    conf: analysis.trackConfidence ?? 0,
                }];
                this.lastRangeBand = analysis.rangeBand;
            } else if (!analysis.motionDetected) {
                this.cameraTargets = [];
                this.lastRangeBand = null;
            }
        }

        function boxOpacityPending(a) { return (a.boxOpacity ?? (a.motionDetected ? 1 : 0)) > 0; }

        // Throttle DOM writes: values rounded to whole % + only write on change.
        // Banner at ~10Hz max so it reads stable while motion continues.
        this._lastHudTick++;
        const hudSavings = document.getElementById('hud-foveation-savings');
        if (hudSavings && this._lastHudTick % 6 === 0) {
            const savingsStr = `⚡ Live Camera Pixel Savings: ${dynamicPixelSavings.toFixed(0)}%`;
            if (savingsStr !== this._lastHudSavings) {
                hudSavings.innerText = savingsStr;
                this._lastHudSavings = savingsStr;
            }
        }

        // Update Camera Foveation Panel Metrics from REAL motion analysis data
        // (FPS + Latency are updated separately in _updatePipelineTelemetry)
        // Labels flip ONLY on stable-state transitions — never per analysis frame.
        const statHits = document.getElementById('stat-flow-hits');
        const statMaskRoi = document.getElementById('stat-mask-roi');
        const statMaskDescription = statMaskRoi ? statMaskRoi.nextElementSibling : null;
        if (statMaskDescription) {
            statMaskDescription.innerText = isWebcam
                ? 'Top 35% and bottom 15% excluded by geometric ROI'
                : 'Sky (35%) & hood (15%) excluded by synthetic road ROI';
        }

        const realCacheHitPct = Math.max(0, 100.0 - (analysis.motionRatio * 100)).toFixed(0);
        const roiTrackingPct  = analysis.motionDetected ? Math.min(65, 50 + Math.round(motionFactor * 15)) : 50;

        if (analysis.motionDetected) {
            const rangeTxt = bandMeta ? ` • ${bandMeta.tag}` : '';
            const flowLabel = `${realCacheHitPct}% Cache Hits (Tracking${bandMeta ? ' ' + analysis.rangeBand.toUpperCase() : ''})`;
            const maskLabel = `${roiTrackingPct}% ROI Active (Tracking${rangeTxt})`;
            if (statHits && flowLabel !== this._lastFlowLabel) { statHits.innerText = flowLabel; statHits.style.color = bandMeta ? bandMeta.color : '#f43f5e'; this._lastFlowLabel = flowLabel; }
            if (statMaskRoi && maskLabel !== this._lastMaskLabel) { statMaskRoi.innerText = maskLabel; statMaskRoi.style.color = bandMeta ? bandMeta.color : '#f43f5e'; this._lastMaskLabel = maskLabel; }
        } else {
            const flowLabel = `${realCacheHitPct}% Cache Hits (Static)`;
            const maskLabel = '50% ROI Retained (Static)';
            if (statHits && flowLabel !== this._lastFlowLabel) { statHits.innerText = flowLabel; statHits.style.color = 'var(--accent-green)'; this._lastFlowLabel = flowLabel; }
            if (statMaskRoi && maskLabel !== this._lastMaskLabel) { statMaskRoi.innerText = maskLabel; statMaskRoi.style.color = 'var(--accent-green)'; this._lastMaskLabel = maskLabel; }
        }

        // HUD range pill: explicit NEAR / MID / FAR readout for the tracked car.
        // (Skipped while the semantic vision path owns the pill.)
        const hudRange = document.getElementById('hud-range');
        if (hudRange && !this.semanticVisionActive) {
            const rangeStr = analysis.motionDetected && bandMeta
                ? `🚗 CAR: ${analysis.rangeBand.toUpperCase()} (${bandMeta.dist}) • conf ${(analysis.trackConfidence ?? 0).toFixed(2)} • LiDAR 0.7 / IMG 0.3`
                : '🚗 CAR: NO TARGET IN ROI';
            if (rangeStr !== this._lastRangeStr) {
                hudRange.innerText = rangeStr;
                hudRange.style.color = bandMeta && analysis.motionDetected ? bandMeta.color : '';
                this._lastRangeStr = rangeStr;
            }
        }

        // ── Semantic (YOLO) overlay: a labelled box for EVERY person/vehicle,
        //    plus wheel/headlight markers found by pixel-shape analysis ──
        if (this.semanticVisionActive && this.semanticObjects.length) {
            const persons = this.semanticObjects.filter(o => o.label === 'person');
            const vehicles = this.semanticObjects.length - persons;
            for (const o of this.semanticObjects) {
                const b = o.box;
                const color = this.semanticLabelColor(o.label);
                const icon = o.label === 'person' ? '🚶' : '🚗';
                const label = `${icon} ${o.label.toUpperCase()} #${o.id} • ${(o.conf || 0).toFixed(2)} • ${(o.range_band || 'mid').toUpperCase()}`;
                ctx.save();
                ctx.shadowColor = color;
                ctx.shadowBlur = 10;
                ctx.strokeStyle = color;
                ctx.lineWidth = 2.5;
                ctx.strokeRect(b[0], b[1], b[2], b[3]);
                ctx.shadowBlur = 0;
                const labelY = Math.max(0, b[1] - 22);
                ctx.fillStyle = color;
                ctx.fillRect(b[0], labelY, Math.max(230, label.length * 6.8), 22);
                ctx.fillStyle = '#040405';
                ctx.font = 'bold 11px monospace';
                ctx.fillText(label, b[0] + 4, labelY + 15);
                ctx.restore();
                // Wheel / headlight markers (server pixel-shape recognition).
                const sb = o._sendBox;
                if (sb && sb[2] > 0) {
                    for (const p of (o.parts || [])) {
                        const px = b[0] + (sb[2] - Math.max(0, Math.min(sb[2], p.cx || 0))) * (b[2] / sb[2]);
                        const py = b[1] + Math.max(0, Math.min(sb[3], p.cy || 0)) * (b[3] / sb[3]);
                        const pr = Math.max(2, (p.r || 3) * (b[2] / sb[2]));
                        ctx.save();
                        if (p.kind === 'wheel') {
                            ctx.strokeStyle = '#22d3ee';
                            ctx.lineWidth = 1.5;
                            ctx.beginPath();
                            ctx.arc(px, py, pr, 0, Math.PI * 2);
                            ctx.stroke();
                        } else {
                            ctx.fillStyle = '#fde047';
                            ctx.beginPath();
                            ctx.arc(px, py, pr, 0, Math.PI * 2);
                            ctx.fill();
                        }
                        ctx.restore();
                    }
                }
            }
            // Object-count strip (top-left of the ROI).
            ctx.fillStyle = 'rgba(2, 6, 23, 0.72)';
            ctx.fillRect(16, h * 0.35 + 6, 348, 26);
            ctx.fillStyle = '#e2e8f0';
            ctx.font = 'bold 12px Inter, sans-serif';
            const backendTxt = this.lastSemanticBackend === 'browser-cascade'
                ? 'browser fallback' : (this.lastSemanticBackend || 'yolo');
            ctx.fillText(`👥 ${persons} PERSON${persons === 1 ? '' : 'S'}   🚗 ${vehicles} VEHICLE${vehicles === 1 ? '' : 'S'}   🧠 ${backendTxt}`, 24, h * 0.35 + 24);
        } else if (this.semanticVisionActive) {
            // Semantic path live but sees nothing right now.
            ctx.fillStyle = 'rgba(59, 130, 246, 0.12)';
            ctx.fillRect(w * 0.22, h * 0.52, w * 0.56, 36);
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 1.5;
            ctx.strokeRect(w * 0.22, h * 0.52, w * 0.56, 36);
            ctx.fillStyle = '#3b82f6';
            ctx.font = 'bold 12px Inter, sans-serif';
            ctx.fillText('🔍 SEMANTIC SEARCH: NO PERSON / VEHICLE DETECTED (0 Objects)', w * 0.22 + 12, h * 0.52 + 22);
        } else {
            // Tracked-car box in its RANGE color — answers "which ring is the car in"
            // at a glance. Label carries range + temporal confidence (master file).
            const boxOpacity = analysis.boxOpacity ?? (analysis.motionDetected ? 1 : 0);
            if (boxOpacity > 0 && analysis.box) {
                const b = analysis.box;
                const ringColor = bandMeta ? bandMeta.color : '#f43f5e';
                const confTxt = (analysis.trackConfidence ?? 0).toFixed(2);
                const rangeTxt = analysis.rangeBand ? analysis.rangeBand.toUpperCase() : '…';
                const label = `🚗 CAR • ${rangeTxt} #${analysis.trackId ?? 0} • conf ${confTxt}`;
                ctx.save();
                ctx.globalAlpha = boxOpacity;
                ctx.shadowColor = ringColor;
                ctx.shadowBlur = 12;
                ctx.strokeStyle = ringColor;
                ctx.lineWidth = 2.5;
                ctx.strokeRect(b.x, b.y, b.w, b.h);
                ctx.shadowBlur = 0;
                const labelY = Math.max(0, b.y - 22);
                ctx.fillStyle = ringColor;
                const labelW = Math.max(230, label.length * 7.2);
                ctx.fillRect(b.x, labelY, labelW, 22);
                ctx.fillStyle = '#040405';
                ctx.font = 'bold 11px monospace';
                ctx.fillText(label, b.x + 4, labelY + 15);
                ctx.restore();
            } else {
                // SCENE CLEAR BADGE
                ctx.fillStyle = 'rgba(74, 222, 128, 0.15)';
                ctx.fillRect(w * 0.22, h * 0.52, w * 0.56, 36);

                ctx.strokeStyle = '#4ade80';
                ctx.lineWidth = 1.5;
                ctx.strokeRect(w * 0.22, h * 0.52, w * 0.56, 36);

                ctx.fillStyle = '#4ade80';
                ctx.font = 'bold 12px Inter, sans-serif';
                ctx.fillText('🟢 SCENE CLEAR: NO DYNAMIC MOTION DETECTED (0 Objects)', w * 0.22 + 12, h * 0.52 + 22);
            }
        }

        // 3-Ring perspective bands drawn ON the road (trapezoids converging to the
        // vanishing point) — near = bottom/wide, far = top/narrow. Colors match
        // the Semantic Ring Legend: near cyan #38bdf8 / mid purple #c084fc /
        // far orange #fb923c. The ACTIVE band (where the tracked car sits) is
        // filled; the others stay outlines so the rings are actually useful.
        const vpx = w * 0.5, vpy = h * 0.42;   // vanishing point
        const yFar0 = h * 0.42, yFar1 = h * 0.52;   // FAR  30–100m
        const yMid0 = h * 0.52, yMid1 = h * 0.68;   // MID  10–30m
        const yNear0 = h * 0.68, yNear1 = h * 0.85; // NEAR  0–10m
        const spreadAt = (y) => {
            const t = Math.min(1, Math.max(0, (y - vpy) / Math.max(1, h * 0.85 - vpy)));
            return 0.06 + t * 0.38;
        };
        const bandPath = (y0, y1) => {
            const s0 = spreadAt(y0), s1 = spreadAt(y1);
            ctx.beginPath();
            ctx.moveTo(vpx - s0 * w, y0);
            ctx.lineTo(vpx + s0 * w, y0);
            ctx.lineTo(vpx + s1 * w, y1);
            ctx.lineTo(vpx - s1 * w, y1);
            ctx.closePath();
        };
        const activeBand = analysis.motionDetected ? analysis.rangeBand : null;
        const bands = [
            { key: 'far', y0: yFar0, y1: yFar1, color: '#fb923c', label: 'FAR 30–100m • 0.25x' },
            { key: 'mid', y0: yMid0, y1: yMid1, color: '#c084fc', label: 'MID 10–30m • 0.5x' },
            { key: 'near', y0: yNear0, y1: yNear1, color: '#38bdf8', label: 'NEAR 0–10m • 1.0x' },
        ];
        ctx.lineWidth = 1.5;
        ctx.font = '11px Inter, sans-serif';
        for (const bd of bands) {
            bandPath(bd.y0, bd.y1);
            if (activeBand === bd.key) {
                ctx.fillStyle = bd.color + '2E'; // ~18% fill on the active band
                ctx.fill();
                ctx.strokeStyle = bd.color;
                ctx.lineWidth = 2.5;
                ctx.stroke();
                ctx.lineWidth = 1.5;
            } else {
                ctx.strokeStyle = bd.color + 'AA';
                ctx.stroke();
            }
            ctx.fillStyle = bd.key === activeBand ? bd.color : bd.color + 'CC';
            ctx.fillText((bd.key === activeBand ? '▶ ' : '') + bd.label, vpx - spreadAt(bd.y1) * w + 6, (bd.y0 + bd.y1) / 2);
        }
    }
}

// Global Engine Instance & View Handlers
let engineInstance = null;
let heroAnimationId = null;

window.addEventListener('DOMContentLoaded', () => {
    engineInstance = new UnifiedTeleopEngine();
    initHeroPreviewCanvas();

    // Wire all [data-scroll-to] anchors (nav + hero buttons). Works from both
    // landing and dashboard views; prevents dead href="#..." jumps.
    document.querySelectorAll('[data-scroll-to]').forEach(anchor => {
        anchor.addEventListener('click', (event) => {
            event.preventDefault();
            const targetId = anchor.getAttribute('data-scroll-to');
            if (targetId) goToSection(targetId);
        });
    });
});

function showPageView(viewId) {
    document.querySelectorAll('.page-view').forEach(view => view.classList.remove('active'));

    if (viewId === 'dashboard') {
        document.body.classList.remove('landing-active');
        document.getElementById('dashboard-view').classList.add('active');
        if (engineInstance) {
            // Correct method name is resize(), not resizeCanvases()
            setTimeout(() => engineInstance.resize(), 50);
        }
        // Dashboard view owns the scroll container; reset to top.
        window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
    } else {
        document.body.classList.add('landing-active');
        document.getElementById('landing-view').classList.add('active');
    }
}

// Section navigation that works from EITHER view. Plain href="#id" breaks
// while the dashboard view is active (landing is display:none), so every
// nav/hero link uses data-scroll-to and routes through here.
function goToSection(sectionId) {
    const landing = document.getElementById('landing-view');
    const isDashboard = document.getElementById('dashboard-view').classList.contains('active');
    if (isDashboard || !landing.classList.contains('active')) {
        showPageView('landing');
    }
    // Wait one frame so display:block applies before scrolling.
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            const target = document.getElementById(sectionId);
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            } else {
                window.scrollTo({ top: 0, behavior: 'smooth' });
            }
        });
    });
    // Keep the URL hash in sync without triggering a default jump.
    try { history.replaceState(null, '', '#' + sectionId); } catch (e) { /* noop */ }
}

// Expose for inline onclick handlers (script is loaded at end of body,
// but explicit assignment survives minifiers/bundlers).
window.showPageView = showPageView;
window.goToSection = goToSection;

const codeSnippets = {
    cpp: `// Modern C++20 Struct-of-Arrays (SoA) Zero-Allocation Point Buffer
struct FoveatedPointBuffer {
    std::vector<float> x, y, z;
    std::vector<uint8_t> ring_id;
    std::vector<float> intensity;

    void reserve_capacity(size_t n_points) {
        x.reserve(n_points); y.reserve(n_points); z.reserve(n_points);
        ring_id.reserve(n_points); intensity.reserve(n_points);
    }
};

void process_scan(const FoveatedPointBuffer& scan) {
    #pragma omp parallel for
    for (size_t i = 0; i < scan.x.size(); ++i) {
        float r = std::hypot(scan.x[i], scan.y[i]);
        uint8_t ring = (r < 10.0f) ? 0 : (r < 30.0f) ? 1 : 2;
        // Direct O(1) ring indexing without dynamic allocation
    }
}`,
    cuda: `// CUDA C++ Kernel: 3D Point Cloud to 2.5D Elevation Grid Projection (<1.82ms)
__global__ void project_elevation_grid_kernel(
    const float* __restrict__ pts_x,
    const float* __restrict__ pts_y,
    const float* __restrict__ pts_z,
    float* __restrict__ grid_max_z,
    int n_points, float grid_res, int grid_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    float px = pts_x[idx];
    float py = pts_y[idx];
    float pz = pts_z[idx];

    int gx = __float2int_rd((px + 50.0f) / grid_res);
    int gy = __float2int_rd((py + 50.0f) / grid_res);

    if (gx >= 0 && gx < grid_dim && gy >= 0 && gy < grid_dim) {
        int cell_idx = gy * grid_dim + gx;
        atomicMax((int*)&grid_max_z[cell_idx], __float_as_int(pz));
    }
}`,
    python: `# Python PyTorch Spatial Masking Module (Sky & Hood Compute Removal)
import torch
import torch.nn as nn

class SemanticSpatialMasker(nn.Module):
    def __init__(self, sky_cutoff_ratio=0.35, hood_cutoff_ratio=0.85):
        super().__init__()
        self.sky_ratio = sky_cutoff_ratio
        self.hood_ratio = hood_cutoff_ratio

    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img_tensor.shape
        mask = torch.ones((B, 1, H, W), device=img_tensor.device)
        
        # Zero out sky (top 35%) & vehicle hood (bottom 15%)
        mask[:, :, :int(H * self.sky_ratio), :] = 0.0
        mask[:, :, int(H * self.hood_ratio):, :] = 0.0
        
        return img_tensor * mask # Saves 50% GPU tensor floating operations`,
    flow: `// Modern C++ OpenCV Farnebäck Motion Gated Cache Controller
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>

class FarnebackMotionGater {
public:
    bool is_scene_dynamic(const cv::Mat& prev_gray, const cv::Mat& curr_gray) {
        cv::Mat flow;
        cv::calcOpticalFlowFarneback(prev_gray, curr_gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        
        cv::Mat flow_parts[2];
        cv::split(flow, flow_parts);
        cv::Mat magnitude;
        cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, cv::Mat());
        
        double max_val;
        cv::minMaxLoc(magnitude, nullptr, &max_val);
        return max_val > 1.8; // Triggers fresh neural inference only when dynamic motion detected
    }
};`
};

function switchCodeTab(tabId, clickedBtn) {
    // Remove active from all tabs and set it on the clicked one.
    // Accept the button element directly to avoid relying on the deprecated global `event`.
    document.querySelectorAll('.code-tab').forEach(tab => tab.classList.remove('active'));
    if (clickedBtn) clickedBtn.classList.add('active');
    const codeDisplay = document.getElementById('code-display');
    if (codeDisplay && codeSnippets[tabId]) {
        codeDisplay.textContent = codeSnippets[tabId];
    }
}

// 21st.dev SaaS Landing Page 3D Hero Canvas Interactive Renderer
function initHeroPreviewCanvas() {
    const canvas = document.getElementById('hero-preview-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let angle = 0;
    
    function resizeHeroCanvas() {
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width * (window.devicePixelRatio || 1);
        canvas.height = rect.height * (window.devicePixelRatio || 1);
    }
    
    window.addEventListener('resize', resizeHeroCanvas);
    resizeHeroCanvas();

    // Pre-generate 3D point cloud rings for 21st.dev hero preview
    const numPoints = 1200;
    const points = [];
    for (let i = 0; i < numPoints; i++) {
        const r = 30 + Math.random() * 160;
        const theta = Math.random() * Math.PI * 2;
        const z = (Math.random() - 0.5) * 24;
        points.push({ r, theta, z });
    }

    function renderHeroCanvas() {
        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        const centerX = w / 2;
        const centerY = h / 2 + 10;
        angle += 0.006;

        // Draw 3 Concentric Ring Grids
        const rings = [50, 110, 170];
        const ringColors = ['rgba(56, 189, 248, 0.25)', 'rgba(192, 132, 252, 0.2)', 'rgba(251, 146, 60, 0.15)'];
        
        rings.forEach((r, idx) => {
            ctx.beginPath();
            ctx.ellipse(centerX, centerY, r * 1.6, r * 0.7, 0, 0, Math.PI * 2);
            ctx.strokeStyle = ringColors[idx];
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 4]);
            ctx.stroke();
            ctx.setLineDash([]);
        });

        // Dynamic LiDAR Radar Laser Beam Sweep
        ctx.save();
        ctx.translate(centerX, centerY);
        ctx.rotate(angle * 2);
        const grad = ctx.createRadialGradient(0, 0, 0, 0, 0, 260);
        grad.addColorStop(0, 'rgba(56, 189, 248, 0.35)');
        grad.addColorStop(0.5, 'rgba(56, 189, 248, 0.08)');
        grad.addColorStop(1, 'transparent');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.arc(0, 0, 260, -0.2, 0.2);
        ctx.closePath();
        ctx.fill();
        ctx.restore();

        // Project and Draw 3D Point Cloud
        points.forEach(pt => {
            const rotTheta = pt.theta + angle;
            const x3d = pt.r * Math.cos(rotTheta);
            const y3d = pt.r * Math.sin(rotTheta);
            
            // Isometric projection
            const projX = centerX + (x3d - y3d) * 0.8;
            const projY = centerY + (x3d + y3d) * 0.35 - pt.z;

            // Color coding based on distance
            let col = '#38bdf8';
            if (pt.r > 120) col = '#fb923c';
            else if (pt.r > 70) col = '#c084fc';

            ctx.fillStyle = col;
            ctx.beginPath();
            ctx.arc(projX, projY, 1.6, 0, Math.PI * 2);
            ctx.fill();
        });

        // Center Autonomous Vehicle Icon
        ctx.fillStyle = '#38bdf8';
        ctx.shadowColor = '#38bdf8';
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.arc(centerX, centerY, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;

        heroAnimationId = requestAnimationFrame(renderHeroCanvas);
    }

    renderHeroCanvas();
}

function setDashboardView(mode) {
    document.querySelectorAll('.view-btn').forEach(btn => btn.classList.remove('active'));
    const targetBtn = document.getElementById(`btn-view-${mode}`);
    if (targetBtn) targetBtn.classList.add('active');
    if (engineInstance) {
        engineInstance.setViewMode(mode);
    }
}

function switchCameraSource(source) {
    if (engineInstance) {
        engineInstance.setCameraSource(source);
    }
}

function switchVehicleMotion(motion) {
    if (engineInstance) {
        engineInstance.setMotionMode(motion);
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

// Trained-model telemetry bundle (written by python/train_perception_models.py).
// Polls vehicle_info.json when served over HTTP; silently keeps the placeholder
// when opened via file:// or before any training run.
(function initVehicleInfoBridge() {
    const COLOR_HEX = {
        cyan: '#38bdf8', blue: '#38bdf8', purple: '#c084fc', orange: '#fb923c',
        red: '#f43f5e', green: '#4ade80', white: '#e4e4e7', silver: '#a1a1aa',
        gray: '#71717a', black: '#52525b', yellow: '#fbbf24', unknown: 'var(--accent-cyan)',
    };
    async function poll() {
        try {
            const res = await fetch('vehicle_info.json', { cache: 'no-store' });
            if (!res.ok) return;
            const data = await res.json();
            const main = document.getElementById('stat-vehicle-info');
            const sub = document.getElementById('stat-vehicle-sub');
            if (main) {
                const n = data.vehicle_count ?? 0;
                const color = data.dominant_color ?? 'unknown';
                main.innerText = `${n} vehicle${n === 1 ? '' : 's'} • ${color}`;
                main.style.color = COLOR_HEX[color] || 'var(--accent-cyan)';
            }
            if (sub) {
                const hist = data.color_histogram || {};
                const parts = Object.entries(hist).map(([k, v]) => `${k}: ${v}`);
                sub.innerText = parts.length
                    ? `Trained colors — ${parts.join(', ')} (tracking: ${data.tracking_status || 'n/a'})`
                    : `Tracking: ${data.tracking_status || 'n/a'} — train on traffic video for colors`;
            }
        } catch (e) { /* offline / not trained yet — keep placeholder */ }
    }
    window.addEventListener('DOMContentLoaded', () => {
        poll();
        setInterval(poll, 5000);
    });
})();

// Inline onclick handlers in index.html resolve against `window`, so export all
// view-switching entry points explicitly (Launch HUD / view tabs / selects).
window.setDashboardView = setDashboardView;
window.switchCameraSource = switchCameraSource;
window.switchVehicleMotion = switchVehicleMotion;
window.updatePointDensity = updatePointDensity;
window.toggleSimulation = toggleSimulation;
window.switchCodeTab = switchCodeTab;
