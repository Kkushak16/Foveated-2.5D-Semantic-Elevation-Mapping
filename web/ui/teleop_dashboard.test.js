const assert = require('node:assert/strict');
const fs = require('node:fs');
const test = require('node:test');
const vm = require('node:vm');

function createAnalyzer() {
    let pixels = new Uint8ClampedArray(80 * 45 * 4);
    const context = {
        document: {
            createElement: () => ({
                getContext: () => ({
                    drawImage() {},
                    getImageData: () => ({ data: pixels })
                })
            })
        }
    };
    const source = fs.readFileSync('teleop_dashboard.js', 'utf8');
    const analyzerSource = source.slice(0, source.indexOf('class UnifiedTeleopEngine'));
    vm.runInNewContext(`${analyzerSource}; globalThis.Analyzer = RealtimeMotionAnalyzer;`, context);

    return {
        analyzer: new context.Analyzer(80, 45),
        setPixel(row, column, value) {
            const index = (row * 80 + column) * 4;
            pixels[index] = value;
            pixels[index + 1] = value;
            pixels[index + 2] = value;
            pixels[index + 3] = 255;
        }
    };
}

test('tracks a contiguous motion region while rejecting isolated noise', () => {
    const { analyzer, setPixel } = createAnalyzer();
    const source = {};

    analyzer.analyze(source, 800, 450); // skipped frame
    analyzer.analyze(source, 800, 450); // establishes the baseline
    setPixel(24, 40, 255); // isolated noise must not create a target
    analyzer.analyze(source, 800, 450);
    assert.equal(analyzer.analyze(source, 800, 450).motionDetected, false);

    for (let row = 22; row < 26; row++) {
        for (let column = 38; column < 43; column++) setPixel(row, column, 255);
    }
    // Hysteresis: stable TRACKING needs 3 consecutive analysis frames
    // (frameSkip=2, so pump skipped+analysis pairs). The whole block toggles
    // intensity each analysis frame so every frame carries a fresh 20-px delta,
    // simulating an object that keeps moving (a static held frame would give
    // zero diff on the second analysis, which is correctly NOT motion).
    let result = analyzer.analyze(source, 800, 450);
    for (let i = 0; i < 4; i++) {
        analyzer.analyze(source, 800, 450); // skipped
        const v = (i % 2 === 0) ? 140 : 255;
        for (let row = 22; row < 26; row++) {
            for (let column = 38; column < 43; column++) setPixel(row, column, v);
        }
        result = analyzer.analyze(source, 800, 450); // analysis
    }

    assert.equal(result.motionDetected, true);
    assert.ok(result.box.w < 100, 'the box is based on the moving region, not all changed pixels');
    assert.ok(result.box.h < 100, 'the box is based on the moving region, not all changed pixels');
});

test('hysteresis holds TRACKING briefly then releases to STATIC', () => {
    const { analyzer, setPixel } = createAnalyzer();
    const source = {};

    analyzer.analyze(source, 800, 450);
    analyzer.analyze(source, 800, 450);
    for (let row = 22; row < 26; row++) {
        for (let column = 38; column < 43; column++) setPixel(row, column, 255);
    }
    let result = analyzer.analyze(source, 800, 450);
    for (let i = 0; i < 4; i++) {
        analyzer.analyze(source, 800, 450);
        const v = (i % 2 === 0) ? 140 : 255;
        for (let row = 22; row < 26; row++) {
            for (let column = 38; column < 43; column++) setPixel(row, column, v);
        }
        result = analyzer.analyze(source, 800, 450);
    }
    assert.equal(result.motionDetected, true);

    // Clear motion: single still frame must NOT flip back to STATIC immediately.
    for (let row = 22; row < 26; row++) {
        for (let column = 38; column < 43; column++) setPixel(row, column, 0);
    }
    analyzer.analyze(source, 800, 450);
    result = analyzer.analyze(source, 800, 450);
    assert.equal(result.motionDetected, true, 'single still frame must not drop TRACKING');

    // Sustained stillness releases back to STATIC.
    for (let i = 0; i < 12; i++) {
        analyzer.analyze(source, 800, 450);
        result = analyzer.analyze(source, 800, 450);
    }
    assert.equal(result.motionDetected, false);
});

test('does not treat a uniform webcam exposure adjustment as motion', () => {
    const { analyzer, setPixel } = createAnalyzer();
    const source = {};

    analyzer.analyze(source, 800, 450);
    analyzer.analyze(source, 800, 450);
    for (let row = 15; row < 38; row++) {
        for (let column = 0; column < 80; column++) setPixel(row, column, 255);
    }
    analyzer.analyze(source, 800, 450);
    const result = analyzer.analyze(source, 800, 450);

    assert.equal(result.motionDetected, false);
});
