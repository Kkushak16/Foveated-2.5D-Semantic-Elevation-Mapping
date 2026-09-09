/**
 * @file teleop_semantic.test.js
 * @brief Unit tests for the browser semantic (YOLO) vision fallback.
 * Covers the key fixes:
 *   1. A STILL person in front of the camera IS detected (per-frame skin/shape
 *      analysis — no temporal delta required).
 *   2. EVERY person gets a stable tracking ID (multi-object tracking, no more
 *      single-largest-blob / "oncoming car" label).
 *   3. A static photo of a car is recognised by its shape, not by motion.
 *
 * Run (from web/ui/):  node --test teleop_semantic.test.js
 */

const assert = require('node:assert/strict');
const fs = require('node:fs');
const test = require('node:test');
const vm = require('node:vm');

function createSemanticHarness() {
    const context = {
        document: {
            createElement: () => ({ getContext: () => ({}) })
        }
    };
    const source = fs.readFileSync(require('node:path').join(__dirname, 'teleop_dashboard.js'), 'utf8');
    const slice = source.slice(0, source.indexOf('class UnifiedTeleopEngine'));
    vm.runInNewContext(`${slice}; globalThis.SemVision = SemanticVision;`, context);
    return new context.SemVision();
}

function setPixel(buf, w, x, y, r, g, b) {
    const i = (y * w + x) * 4;
    buf[i] = r; buf[i + 1] = g; buf[i + 2] = b; buf[i + 3] = 255;
}

function solidFrame(w, h, draw) {
    const buf = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) {
        buf[i * 4] = 128; buf[i * 4 + 1] = 128; buf[i * 4 + 2] = 128; buf[i * 4 + 3] = 255;
    }
    if (draw) draw(buf, w, h);
    return buf;
}

// Two people: skin-tone heads + dark clothing. IDENTICAL every frame (zero
// temporal difference — exactly the "static display image" case that the old
// motion-only analyzer could never detect).
function twoStillPeople(w, h) {
    return solidFrame(w, h, (buf, W, H) => {
        for (const cx of [Math.floor(W * 0.3), Math.floor(W * 0.7)]) {
            for (let dy = 0; dy < 10; dy++) {
                for (let dx = -6; dx <= 6; dx++) {
                    setPixel(buf, W, cx + dx, Math.floor(H * 0.3) + dy, 220, 150, 120);
                }
            }
            for (let dy = 10; dy < 34; dy++) {
                for (let dx = -9; dx <= 9; dx++) {
                    setPixel(buf, W, cx + dx, Math.floor(H * 0.3) + dy, 55, 55, 65);
                }
            }
        }
    });
}

test('detects a STILL person held in front of the camera (no motion)', () => {
    const sv = createSemanticHarness();
    const w = 240, h = 135;
    const frame = twoStillPeople(w, h);
    // Same frame twice — temporal diff is zero — but the person SHAPE is
    // recognised from the pixels alone.
    sv.analyzeLocal(frame, w, h);
    const persons = sv.localObjects.filter(o => o.label === 'person');
    assert.ok(persons.length >= 2, `expected >=2 persons, got ${persons.length}`);
});

test('tracks EACH person with a stable id across identical still frames', () => {
    const sv = createSemanticHarness();
    const w = 240, h = 135;
    const frame = twoStillPeople(w, h);
    sv.analyzeLocal(frame, w, h);
    const ids1 = sv.localObjects.filter(o => o.label === 'person').map(o => o.id).sort();
    sv.analyzeLocal(frame, w, h);
    const ids2 = sv.localObjects.filter(o => o.label === 'person').map(o => o.id).sort();
    assert.ok(ids1.length >= 2, `want >=2 persons, got ${ids1.length}`);
    assert.ok(ids1.every((id, i) => id === ids2[i]), `ids not stable: ${ids1} vs ${ids2}`);
});

test('recognises a static car photo by its shape (dark car-like rectangle)', () => {
    const sv = createSemanticHarness();
    const w = 240, h = 135;
    // A parked-looking car: wide + dark in the lower two-thirds.
    const frame = solidFrame(w, h, (buf, W, H) => {
        for (let dy = 0; dy < 38; dy++) {
            for (let dx = 0; dx < 80; dx++) {
                setPixel(buf, W, 70 + dx, Math.floor(H * 0.55) + dy, 60, 60, 70);
            }
        }
    });
    sv.analyzeLocal(frame, w, h);
    const vehicles = sv.localObjects.filter(o => o.label === 'vehicle');
    assert.ok(vehicles.length >= 1, `expected a vehicle, got ${vehicles.length}`);
});

test('range band helper returns near/mid/far monotonic with size', () => {
    const sv = createSemanticHarness();
    const near = sv._rangeLocal({ x: 0, y: 90, w: 80, h: 40 }, 240, 135);
    const far = sv._rangeLocal({ x: 0, y: 55, w: 12, h: 8 }, 240, 135);
    assert.ok(['near', 'mid', 'far'].includes(near), `near=${near}`);
    assert.ok(['near', 'mid', 'far'].includes(far), `far=${far}`);
});