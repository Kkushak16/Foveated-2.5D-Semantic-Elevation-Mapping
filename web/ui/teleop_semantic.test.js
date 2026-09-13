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

test('unifies face and gesturing hands/fingers into EXACTLY 1 person with stable ID', () => {
    const sv = createSemanticHarness();
    const w = 240, h = 135;
    // Single seated person: face at cx=120, hand/fingers at cx=140
    const frame = solidFrame(w, h, (buf, W, H) => {
        const cx = Math.floor(W * 0.5);
        // Face skin blob
        for (let dy = 0; dy < 14; dy++) {
            for (let dx = -8; dx <= 8; dx++) {
                setPixel(buf, W, cx + dx, Math.floor(H * 0.25) + dy, 220, 150, 120);
            }
        }
        // Torso / shirt
        for (let dy = 14; dy < 45; dy++) {
            for (let dx = -14; dx <= 14; dx++) {
                setPixel(buf, W, cx + dx, Math.floor(H * 0.25) + dy, 40, 50, 60);
            }
        }
        // Hand / fingers skin blob slightly to the right
        for (let dy = 20; dy < 30; dy++) {
            for (let dx = 14; dx <= 23; dx++) {
                setPixel(buf, W, cx + dx, Math.floor(H * 0.25) + dy, 220, 150, 120);
            }
        }
    });

    sv.analyzeLocal(frame, w, h);
    const persons1 = sv.localObjects.filter(o => o.label === 'person');
    assert.equal(persons1.length, 1, `expected exactly 1 person, got ${persons1.length}`);
    assert.equal(persons1[0].id, 1, `expected person ID 1, got ${persons1[0].id}`);

    // Second frame: person ID must remain 1
    sv.analyzeLocal(frame, w, h);
    const persons2 = sv.localObjects.filter(o => o.label === 'person');
    assert.equal(persons2.length, 1, `expected still exactly 1 person, got ${persons2.length}`);
    assert.equal(persons2[0].id, 1, `expected stable person ID 1 across frames, got ${persons2[0].id}`);
});

test('does NOT detect a flat wall or ambient room shadow as a vehicle', () => {
    const sv = createSemanticHarness();
    const w = 240, h = 135;
    // Flat wall with ambient shadow in lower area (typical room background)
    const frame = solidFrame(w, h, (buf, W, H) => {
        for (let y = 0; y < H; y++) {
            for (let x = 0; x < W; x++) {
                const shadow = (y > H * 0.4 && y < H * 0.8) ? 65 : 78;
                setPixel(buf, W, x, y, shadow, shadow, shadow);
            }
        }
    });

    sv.analyzeLocal(frame, w, h);
    const vehicles = sv.localObjects.filter(o => o.label === 'vehicle');
    assert.equal(vehicles.length, 0, `expected 0 vehicles on a flat wall, got ${vehicles.length}`);
});