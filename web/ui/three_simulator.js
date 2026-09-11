/**
 * @file three_simulator.js
 * @brief High-Fidelity 3D Autonomous Vehicle Perception Simulator in Three.js / WebGL.
 * Features:
 *   1. Ultra-Crisp High-DPI Rendering (sRGB, ACES Filmic Tone Mapping, Soft PCF Shadows).
 *   2. Vivid Color Palette: Emerald grass landscape, dark asphalt road, red/white checker curbs, beige sidewalks, neon city towers.
 *   3. Holographic 3-Ring Concentric LiDAR Grid (Cyan Near 0-10m, Violet Mid 10-30m, Orange Far 30-70m) - cleanly separated from the road.
 *   4. Mathematically closed, continuous urban boulevard circuit with seamless transitions.
 *   5. True Non-Mirrored Controls: Right steers right, Left steers left; front wheels visually turn.
 *   6. Deterministic Autopilot Loop: Follows continuous closed path, slows down for corners, yields to pedestrians.
 *   7. 8 Animated 3D Pedestrians crossing at zebra crosswalks and walking along sidewalks.
 *   8. 720p HD Windshield Sensor Camera with exact distance & semantic ring telemetry tags.
 */

(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        define(['three'], factory);
    } else if (typeof exports === 'object') {
        module.exports = factory(require('three'));
    } else {
        root.ThreeSimulator = factory(root.THREE);
    }
}(typeof self !== 'undefined' ? self : this, function (THREE) {
    'use strict';

    if (!THREE) {
        console.warn('[ThreeSimulator] THREE.js is not loaded yet.');
        return null;
    }

    class ThreeSimulator {
        constructor(canvasId = 'sim-3d-canvas') {
            this.canvas = typeof canvasId === 'string' ? document.getElementById(canvasId) : canvasId;
            if (!this.canvas) {
                this.canvas = document.createElement('canvas');
                this.canvas.id = 'sim-3d-canvas';
            }

            this.width = this.canvas.clientWidth || 960;
            this.height = this.canvas.clientHeight || 600;

            // --- High-Resolution WebGL Renderer ---
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            this.renderer = new THREE.WebGLRenderer({
                canvas: this.canvas,
                antialias: true,
                alpha: false,
                powerPreference: 'high-performance'
            });
            this.renderer.setPixelRatio(dpr);
            this.renderer.setSize(this.width, this.height, true);
            this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
            this.renderer.toneMappingExposure = 0.92;
            if (THREE.sRGBEncoding) {
                this.renderer.outputEncoding = THREE.sRGBEncoding;
            }
            this.renderer.shadowMap.enabled = true;
            this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

            // --- Scene & Atmospheric Lighting (Bright Daylight Sky) ---
            this.scene = new THREE.Scene();
            this.scene.background = new THREE.Color(0x7dd3fc);
            this.scene.fog = new THREE.FogExp2(0xe0f2fe, 0.0012);

            // --- Cameras ---
            this.cameraMode = 'chase'; // 'chase' | 'cockpit' | 'overhead'
            this.userCamera = new THREE.PerspectiveCamera(60, this.width / this.height, 0.2, 450);
            this.userCamera.position.set(-28, 6.5, 60);

            // 720p HD Windshield Sensor Camera
            this.windshieldCam = new THREE.PerspectiveCamera(66, 16 / 9, 0.1, 180);
            this.windshieldTarget = new THREE.WebGLRenderTarget(1280, 720, {
                minFilter: THREE.LinearFilter,
                magFilter: THREE.LinearFilter,
                format: THREE.RGBAFormat
            });
            this.offscreenCanvas = document.createElement('canvas');
            this.offscreenCanvas.width = 1280;
            this.offscreenCanvas.height = 720;
            this.offscreenCtx = this.offscreenCanvas.getContext('2d');

            // --- Ego Vehicle State ---
            this.ego = {
                x: -34,
                z: 40,
                y: 0,
                yaw: 0,              // facing North (towards -Z)
                speed: 0,
                cruiseSpeed: 9.6,    // ~35 km/h cruising speed
                turnSpeed: 5.8,      // ~21 km/h cornering speed
                accel: 7.5,
                decel: 10.0,
                friction: 2.6,
                steerAngle: 0,
                steerSpeed: 3.8,
                wheelRotation: 0,
                mesh: null,
                lidarPuck: null,
                lidarSweep: null,
                lidarRingsGroup: null,
                wheels: []
            };

            // --- Control Modes ---
            this.autopilot = true;
            this.keys = { forward: false, backward: false, left: false, right: false, brake: false };
            this.manualOverrideTimer = 0;

            // --- Track Waypoints (Closed Continuous Loop) ---
            this.circuitWaypoints = this._generateCircuitWaypoints();
            this.currentWpIndex = this._findClosestForwardWaypoint(this.ego.x, this.ego.z, this.ego.yaw);

            // --- Actors & Visuals ---
            this.pedestrians = [];
            this.trafficVehicles = [];
            this.clouds = [];
            this.potholes = [];
            this.trees = [];
            this.sweepAngle = 0;
            this.isEvadingPedestrian = false;
            this.collisionAlert = false;

            // Build Virtual World
            this._initLighting();
            this._initSkyDome();
            this._initClouds();
            this._initContinuousCircuit();
            this._initPotholes();
            this._initRoadsideTreesAndHighwayScenery();
            this._initCityEnvironment();
            this._initEgoVehicle();
            this._initPedestrians();
            this._initTrafficVehicles();
            this._initHolographicLidarRings();
            this._setupControls();

            this._lastTime = performance.now();
        }

        // ---------------------------------------------------------------------
        // 1. Mathematically Continuous Closed Circuit Waypoints (Expanded 150m Corridor)
        // ---------------------------------------------------------------------
        _generateCircuitWaypoints() {
            const waypoints = [];
            const R = 34;
            const straightHalf = 75;

            // 1. West Straight (Driving North from Z = +75 to Z = -75 along X = -R)
            for (let z = straightHalf; z >= -straightHalf; z -= 5) {
                waypoints.push({ x: -R, z: z });
            }

            // 2. North Curve (Sweeping right from X = -R to X = +R around center (0, -straightHalf))
            const curveSteps = 24;
            for (let i = 1; i < curveSteps; i++) {
                const theta = (i / curveSteps) * Math.PI;
                waypoints.push({
                    x: -R * Math.cos(theta),
                    z: -straightHalf - R * Math.sin(theta)
                });
            }

            // 3. East Straight (Driving South from Z = -75 to Z = +75 along X = +R)
            for (let z = -straightHalf; z <= straightHalf; z += 5) {
                waypoints.push({ x: R, z: z });
            }

            // 4. South Curve (Sweeping right from X = +R to X = -R around center (0, straightHalf))
            for (let i = 1; i < curveSteps; i++) {
                const theta = (i / curveSteps) * Math.PI;
                waypoints.push({
                    x: R * Math.cos(theta),
                    z: straightHalf + R * Math.sin(theta)
                });
            }

            return waypoints;
        }

        _findClosestForwardWaypoint(x, z, yaw) {
            let bestIdx = 0;
            let minDist = Infinity;
            const fwdX = -Math.sin(yaw);
            const fwdZ = -Math.cos(yaw);
            for (let i = 0; i < this.circuitWaypoints.length; i++) {
                const wp = this.circuitWaypoints[i];
                const dx = wp.x - x;
                const dz = wp.z - z;
                const dist = Math.hypot(dx, dz);
                const dot = dx * fwdX + dz * fwdZ;
                if (dot > 0 && dist < minDist) {
                    minDist = dist;
                    bestIdx = i;
                }
            }
            return bestIdx;
        }

        // ---------------------------------------------------------------------
        // 2. Lighting & Sky Dome (Luminous Bright Daylight Sky)
        // ---------------------------------------------------------------------
        _initLighting() {
            const ambient = new THREE.AmbientLight(0xffffff, 0.68);
            this.scene.add(ambient);

            const sunLight = new THREE.DirectionalLight(0xfffaed, 1.25);
            sunLight.position.set(65, 100, 45);
            sunLight.castShadow = true;
            sunLight.shadow.mapSize.width = 2048;
            sunLight.shadow.mapSize.height = 2048;
            sunLight.shadow.camera.near = 15;
            sunLight.shadow.camera.far = 260;
            const d = 90;
            sunLight.shadow.camera.left = -d;
            sunLight.shadow.camera.right = d;
            sunLight.shadow.camera.top = d;
            sunLight.shadow.camera.bottom = -d;
            sunLight.shadow.bias = -0.0003;
            this.scene.add(sunLight);

            // Soft white daylight ambient bounce
            const skyFill = new THREE.DirectionalLight(0xe0f2fe, 0.42);
            skyFill.position.set(-50, 40, -50);
            this.scene.add(skyFill);
        }

        _initSkyDome() {
            const skyGeo = new THREE.SphereGeometry(320, 32, 24);
            const skyMat = new THREE.ShaderMaterial({
                side: THREE.BackSide,
                uniforms: {
                    topColor: { value: new THREE.Color(0x0284c7) },     // Deep vibrant sky blue
                    bottomColor: { value: new THREE.Color(0x7dd3fc) },  // Soft light sky blue
                    horizonColor: { value: new THREE.Color(0xffffff) }, // Brilliant luminous white horizon
                    offset: { value: 20 },
                    exponent: { value: 0.55 }
                },
                vertexShader: `
                    varying vec3 vWorldPosition;
                    void main() {
                        vec4 worldPosition = modelMatrix * vec4(position, 1.0);
                        vWorldPosition = worldPosition.xyz;
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                    }
                `,
                fragmentShader: `
                    uniform vec3 topColor;
                    uniform vec3 bottomColor;
                    uniform vec3 horizonColor;
                    uniform float offset;
                    uniform float exponent;
                    varying vec3 vWorldPosition;
                    void main() {
                        float h = normalize(vWorldPosition + offset).y;
                        vec3 sky = mix(bottomColor, topColor, max(pow(max(h, 0.0), exponent), 0.0));
                        if (h < 0.35) {
                            sky = mix(horizonColor, sky, h / 0.35);
                        }
                        gl_FragColor = vec4(sky, 1.0);
                    }
                `
            });
            const sky = new THREE.Mesh(skyGeo, skyMat);
            this.scene.add(sky);
        }

        _initClouds() {
            this.clouds = [];
            const cloudGroup = new THREE.Group();
            const cloudMat = new THREE.MeshStandardMaterial({
                color: 0xffffff,
                roughness: 0.98,
                metalness: 0.0,
                transparent: true,
                opacity: 0.95
            });

            // 28 realistic, fluffy, bright white cumulus cloud clusters in daylight sky
            for (let i = 0; i < 28; i++) {
                const puffCluster = new THREE.Group();
                const numSpheres = 6 + Math.floor(Math.random() * 4);
                const clusterRadius = 8 + Math.random() * 8;

                for (let s = 0; s < numSpheres; s++) {
                    const r = clusterRadius * (0.55 + Math.random() * 0.45);
                    const sph = new THREE.Mesh(new THREE.SphereGeometry(r, 8, 8), cloudMat);
                    sph.position.set(
                        (Math.random() - 0.5) * clusterRadius * 1.8,
                        (Math.random() - 0.5) * (clusterRadius * 0.35),
                        (Math.random() - 0.5) * clusterRadius * 1.5
                    );
                    puffCluster.add(sph);
                }

                const cx = (Math.random() - 0.5) * 360;
                const cy = 44 + Math.random() * 26;
                const cz = (Math.random() - 0.5) * 360;
                puffCluster.position.set(cx, cy, cz);
                cloudGroup.add(puffCluster);
                this.clouds.push(puffCluster);
            }

            this.scene.add(cloudGroup);
        }

        // ---------------------------------------------------------------------
        // 3. Continuous Circuit Road, Colorful Curbs & Landscape
        // ---------------------------------------------------------------------
        _initContinuousCircuit() {
            const trackGroup = new THREE.Group();

            // Vibrant Emerald Grass Landscape (fixes the dark-on-dark empty void!)
            const terrainGeo = new THREE.PlaneGeometry(600, 600);
            const terrainMat = new THREE.MeshStandardMaterial({
                color: 0x15803d, // Rich forest emerald grass
                roughness: 0.88,
                metalness: 0.05
            });
            const terrain = new THREE.Mesh(terrainGeo, terrainMat);
            terrain.rotation.x = -Math.PI / 2;
            terrain.receiveShadow = true;
            trackGroup.add(terrain);

            // Neutral Highway Grey Asphalt (Dark Realistic Asphalt)
            const asphaltMat = new THREE.MeshStandardMaterial({
                color: 0x2a2f36, // Dark realistic highway asphalt
                roughness: 0.95,
                metalness: 0.02
            });

            const R = 34;
            const roadWidth = 14;
            const straightHalf = 75;

            // West Straight (150m long)
            const westStraight = new THREE.Mesh(new THREE.PlaneGeometry(roadWidth, straightHalf * 2), asphaltMat);
            westStraight.rotation.x = -Math.PI / 2;
            westStraight.position.set(-R, 0.01, 0);
            westStraight.receiveShadow = true;
            trackGroup.add(westStraight);

            // East Straight (150m long)
            const eastStraight = new THREE.Mesh(new THREE.PlaneGeometry(roadWidth, straightHalf * 2), asphaltMat);
            eastStraight.rotation.x = -Math.PI / 2;
            eastStraight.position.set(R, 0.01, 0);
            eastStraight.receiveShadow = true;
            trackGroup.add(eastStraight);

            // North Turn Ring (Seamless 180° arc)
            const northCurveGeo = new THREE.RingGeometry(R - roadWidth / 2, R + roadWidth / 2, 48, 1, 0, Math.PI);
            const northCurve = new THREE.Mesh(northCurveGeo, asphaltMat);
            northCurve.rotation.x = -Math.PI / 2;
            northCurve.position.set(0, 0.01, -straightHalf);
            northCurve.receiveShadow = true;
            trackGroup.add(northCurve);

            // South Turn Ring (Seamless 180° arc)
            const southCurveGeo = new THREE.RingGeometry(R - roadWidth / 2, R + roadWidth / 2, 48, 1, Math.PI, Math.PI);
            const southCurve = new THREE.Mesh(southCurveGeo, asphaltMat);
            southCurve.rotation.x = -Math.PI / 2;
            southCurve.position.set(0, 0.01, straightHalf);
            southCurve.receiveShadow = true;
            trackGroup.add(southCurve);

            // --- High-Contrast Road Markings ---
            const yellowMat = new THREE.MeshBasicMaterial({ color: 0xfacc15 });
            const whiteDashMat = new THREE.MeshBasicMaterial({ color: 0xffffff });

            // Double Solid Yellow Centerlines
            [-0.18, 0.18].forEach(offset => {
                const lineW = new THREE.Mesh(new THREE.PlaneGeometry(0.14, straightHalf * 2), yellowMat);
                lineW.rotation.x = -Math.PI / 2;
                lineW.position.set(-R + offset, 0.02, 0);
                trackGroup.add(lineW);

                const lineE = new THREE.Mesh(new THREE.PlaneGeometry(0.14, straightHalf * 2), yellowMat);
                lineE.rotation.x = -Math.PI / 2;
                lineE.position.set(R + offset, 0.02, 0);
                trackGroup.add(lineE);

                const ringN = new THREE.Mesh(new THREE.RingGeometry(R + offset - 0.07, R + offset + 0.07, 48, 1, 0, Math.PI), yellowMat);
                ringN.rotation.x = -Math.PI / 2;
                ringN.position.set(0, 0.02, -straightHalf);
                trackGroup.add(ringN);

                const ringS = new THREE.Mesh(new THREE.RingGeometry(R + offset - 0.07, R + offset + 0.07, 48, 1, Math.PI, Math.PI), yellowMat);
                ringS.rotation.x = -Math.PI / 2;
                ringS.position.set(0, 0.02, straightHalf);
                trackGroup.add(ringS);
            });

            // Dashed White Lane Lines on Straights (4-Lane Highway)
            for (let z = -straightHalf + 3; z <= straightHalf - 3; z += 6) {
                [-3.5, 3.5].forEach(offset => {
                    const dashW = new THREE.Mesh(new THREE.PlaneGeometry(0.22, 3.2), whiteDashMat);
                    dashW.rotation.x = -Math.PI / 2;
                    dashW.position.set(-R + offset, 0.02, z);
                    trackGroup.add(dashW);

                    const dashE = new THREE.Mesh(new THREE.PlaneGeometry(0.22, 3.2), whiteDashMat);
                    dashE.rotation.x = -Math.PI / 2;
                    dashE.position.set(R + offset, 0.02, z);
                    trackGroup.add(dashE);
                });
            }

            // High-Visibility Zebra Crosswalks
            this._buildZebraCrosswalk(trackGroup, -R, -40, roadWidth, 4.8);
            this._buildZebraCrosswalk(trackGroup, -R, -10, roadWidth, 4.8);
            this._buildZebraCrosswalk(trackGroup, -R, 30, roadWidth, 4.8);
            this._buildZebraCrosswalk(trackGroup, R, -35, roadWidth, 4.8);
            this._buildZebraCrosswalk(trackGroup, R, 15, roadWidth, 4.8);
            this._buildZebraCrosswalk(trackGroup, R, 50, roadWidth, 4.8);

            // Sidewalks & Curbs
            this._buildCurbsAndSidewalks(trackGroup, R, roadWidth, straightHalf);

            this.scene.add(trackGroup);
        }

        _buildZebraCrosswalk(parent, cx, cz, roadW, length) {
            const stripeMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
            const stripeCount = 10;
            const stripeWidth = (roadW - 2) / (stripeCount * 2);
            for (let i = 0; i < stripeCount; i++) {
                const stripe = new THREE.Mesh(new THREE.PlaneGeometry(stripeWidth, length), stripeMat);
                stripe.rotation.x = -Math.PI / 2;
                const sx = cx - ((roadW - 2) / 2) + (i * 2 + 1) * stripeWidth;
                stripe.position.set(sx, 0.025, cz);
                parent.add(stripe);
            }
        }

        _buildCurbsAndSidewalks(parent, R, roadW, straightHalf) {
            const sidewalkMat = new THREE.MeshStandardMaterial({ color: 0xcfd8dc, roughness: 0.65 });
            const redCurbMat = new THREE.MeshStandardMaterial({ color: 0xd97706, roughness: 0.5 });
            const whiteCurbMat = new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.5 });

            const rInner = R - roadW / 2;
            const rOuter = R + roadW / 2;
            const curbHeight = 0.22;

            // Straight Sidewalks
            const swWidth = 4.0;
            const swWOuter = new THREE.Mesh(new THREE.BoxGeometry(swWidth, 0.18, straightHalf * 2), sidewalkMat);
            swWOuter.position.set(-R - (roadW / 2 + swWidth / 2), 0.09, 0);
            swWOuter.receiveShadow = true;
            parent.add(swWOuter);

            const swEOuter = new THREE.Mesh(new THREE.BoxGeometry(swWidth, 0.18, straightHalf * 2), sidewalkMat);
            swEOuter.position.set(R + (roadW / 2 + swWidth / 2), 0.09, 0);
            swEOuter.receiveShadow = true;
            parent.add(swEOuter);

            // Elevated Curbs
            [-1, 1].forEach(side => {
                const curbW = new THREE.Mesh(new THREE.BoxGeometry(0.35, curbHeight, straightHalf * 2), redCurbMat);
                curbW.position.set(-R + side * (roadW / 2), curbHeight / 2, 0);
                parent.add(curbW);

                const curbE = new THREE.Mesh(new THREE.BoxGeometry(0.35, curbHeight, straightHalf * 2), redCurbMat);
                curbE.position.set(R + side * (roadW / 2), curbHeight / 2, 0);
                parent.add(curbE);
            });

            // Glowing Outer Curve LED Runner (Cyan)
            const runnerMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
            const northRunner = new THREE.Mesh(new THREE.RingGeometry(rOuter, rOuter + 0.35, 48, 1, 0, Math.PI), runnerMat);
            northRunner.rotation.x = -Math.PI / 2;
            northRunner.position.set(0, 0.04, -straightHalf);
            parent.add(northRunner);

            const southRunner = new THREE.Mesh(new THREE.RingGeometry(rOuter, rOuter + 0.35, 48, 1, Math.PI, Math.PI), runnerMat);
            southRunner.rotation.x = -Math.PI / 2;
            southRunner.position.set(0, 0.04, straightHalf);
            parent.add(southRunner);
        }

        // ---------------------------------------------------------------------
        // 3.5. Road Potholes with 2.5D LiDAR Elevation Deficit Scanners
        // ---------------------------------------------------------------------
        _initPotholes() {
            this.potholes = [
                { x: -32.2, z: 8.0, depth: '-12cm', label: 'pothole', radius: 0.95 },
                { x: -35.8, z: -32.0, depth: '-10cm', label: 'pothole', radius: 0.85 },
                { x: 35.8, z: -15.0, depth: '-15cm', label: 'pothole', radius: 1.1 },
                { x: 32.4, z: 35.0, depth: '-12cm', label: 'pothole', radius: 0.9 }
            ];

            const potholeGroup = new THREE.Group();

            this.potholes.forEach(ph => {
                ph.pos = new THREE.Vector3(ph.x, 0.02, ph.z);

                // Deep Asphalt Depression Crater (Black recessed tarmac)
                const craterGeo = new THREE.CircleGeometry(ph.radius, 24);
                const craterMat = new THREE.MeshStandardMaterial({
                    color: 0x09090b,
                    roughness: 0.98,
                    metalness: 0.1
                });
                const crater = new THREE.Mesh(craterGeo, craterMat);
                crater.rotation.x = -Math.PI / 2;
                crater.position.set(ph.x, 0.022, ph.z);
                potholeGroup.add(crater);

                // Fractured asphalt rim with rough edge
                const rimGeo = new THREE.RingGeometry(ph.radius * 0.9, ph.radius * 1.35, 24);
                const rimMat = new THREE.MeshStandardMaterial({
                    color: 0x334155,
                    roughness: 0.95,
                    metalness: 0.05
                });
                const rim = new THREE.Mesh(rimGeo, rimMat);
                rim.rotation.x = -Math.PI / 2;
                rim.position.set(ph.x, 0.025, ph.z);
                potholeGroup.add(rim);

                // 2.5D LiDAR Elevation Deficit Indicator Ring (Glowing Hazard Rose)
                const lidarScanGeo = new THREE.RingGeometry(ph.radius * 1.05, ph.radius * 1.20, 32);
                const lidarScanMat = new THREE.MeshBasicMaterial({
                    color: 0xf43f5e,
                    side: THREE.DoubleSide
                });
                const scanRing = new THREE.Mesh(lidarScanGeo, lidarScanMat);
                scanRing.rotation.x = -Math.PI / 2;
                scanRing.position.set(ph.x, 0.035, ph.z);
                potholeGroup.add(scanRing);

                // Vertical elevation depth grid lines dipping down
                for (let a = 0; a < Math.PI * 2; a += Math.PI / 3) {
                    const lx = Math.cos(a) * ph.radius * 0.7;
                    const lz = Math.sin(a) * ph.radius * 0.7;
                    const dropGeo = new THREE.BufferGeometry().setFromPoints([
                        new THREE.Vector3(ph.x + lx, 0.03, ph.z + lz),
                        new THREE.Vector3(ph.x + lx, -0.12, ph.z + lz)
                    ]);
                    const dropLine = new THREE.Line(dropGeo, new THREE.LineBasicMaterial({ color: 0xef4444, linewidth: 2 }));
                    potholeGroup.add(dropLine);
                }
            });

            this.scene.add(potholeGroup);
        }

        // ---------------------------------------------------------------------
        // 3.6. Roadside Trees & Highway Scenery (Vibrant Colorful Species)
        // ---------------------------------------------------------------------
        _initRoadsideTreesAndHighwayScenery() {
            const sceneryGroup = new THREE.Group();
            this.trees = [];
            const straightHalf = 75;

            // 1. Pine Conifer Tree (Dense forest pine green with rich brown trunk)
            const buildPine = (x, z, scale = 1.0) => {
                const pine = new THREE.Group();
                const trunk = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.24 * scale, 0.36 * scale, 2.4 * scale, 8),
                    new THREE.MeshStandardMaterial({ color: 0x451a03, roughness: 0.92 }) // Rich warm brown bark
                );
                trunk.position.y = 1.2 * scale;
                trunk.castShadow = true;
                pine.add(trunk);

                const tierColors = [0x14532d, 0x166534, 0x15803d]; // Dense Natural Pine Greens
                const tierRadii = [2.4, 1.8, 1.3];
                const tierHeights = [2.8, 2.4, 2.0];
                const tierY = [2.5, 4.2, 5.8];

                for (let t = 0; t < 3; t++) {
                    const cone = new THREE.Mesh(
                        new THREE.ConeGeometry(tierRadii[t] * scale, tierHeights[t] * scale, 8),
                        new THREE.MeshStandardMaterial({ color: tierColors[t], roughness: 0.85 })
                    );
                    cone.position.y = tierY[t] * scale;
                    cone.castShadow = true;
                    pine.add(cone);
                }

                pine.position.set(x, 0, z);
                sceneryGroup.add(pine);
                this.trees.push({ pos: new THREE.Vector3(x, 2.5, z), radius: 2.2 * scale, label: 'tree' });
            };

            // 2. Deciduous Leafy Tree (Vibrant Natural Botanical Greens with Rich Brown Trunk)
            const buildDeciduous = (x, z, scale = 1.0, palette = 'emerald') => {
                const tree = new THREE.Group();
                const trunk = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.32 * scale, 0.48 * scale, 3.2 * scale, 8),
                    new THREE.MeshStandardMaterial({ color: 0x451a03, roughness: 0.92 }) // Rich warm brown bark
                );
                trunk.position.y = 1.6 * scale;
                trunk.castShadow = true;
                tree.add(trunk);

                let cMain, cAlt;
                if (palette === 'forest') {
                    cMain = 0x14532d; // Deep Pine Green
                    cAlt = 0x15803d;  // Forest Canopy Green
                } else if (palette === 'summer_oak') {
                    cMain = 0x166534; // Summer Oak Green
                    cAlt = 0x15803d;  // Vibrant Foliage Green
                } else if (palette === 'woodland') {
                    cMain = 0x166534; // Deep Woodland Green
                    cAlt = 0x14532d;  // Dark Pine Green
                } else {
                    cMain = 0x15803d; // Emerald Foliage
                    cAlt = 0x166534;  // Deep Emerald
                }

                const mat1 = new THREE.MeshStandardMaterial({ color: cMain, roughness: 0.85 });
                const mat2 = new THREE.MeshStandardMaterial({ color: cAlt, roughness: 0.85 });

                const clusters = [
                    { x: 0, y: 4.0, z: 0, r: 2.4, m: mat1 },
                    { x: -1.3, y: 3.6, z: 0.7, r: 1.8, m: mat2 },
                    { x: 1.3, y: 3.7, z: -0.7, r: 1.9, m: mat1 },
                    { x: 0, y: 5.2, z: 0, r: 1.7, m: mat2 }
                ];

                clusters.forEach(c => {
                    const sph = new THREE.Mesh(new THREE.SphereGeometry(c.r * scale, 8, 8), c.m);
                    sph.position.set(c.x * scale, c.y * scale, c.z * scale);
                    sph.castShadow = true;
                    tree.add(sph);
                });

                tree.position.set(x, 0, z);
                sceneryGroup.add(tree);
                this.trees.push({ pos: new THREE.Vector3(x, 2.6, z), radius: 2.4 * scale, label: 'tree' });
            };

            // 3. Natural Roadside Shrubs & Hedges
            const buildRoadsideShrub = (x, z, scale = 1.0, color = 0x15803d) => {
                const shrub = new THREE.Mesh(
                    new THREE.SphereGeometry(1.1 * scale, 8, 6),
                    new THREE.MeshStandardMaterial({ color: color, roughness: 0.85 })
                );
                shrub.scale.set(1.4, 0.8, 1.2);
                shrub.position.set(x, 0.6 * scale, z);
                shrub.castShadow = true;
                sceneryGroup.add(shrub);
                this.trees.push({ pos: new THREE.Vector3(x, 0.8, z), radius: 1.2 * scale, label: 'tree' });
            };

            // Highway Side Sequences (Lush Natural Botanical Foliage)
            const speciesList = ['forest', 'emerald', 'summer_oak', 'woodland'];

            // 1. West Highway Trees & Shrubs (X = -44 to -52)
            for (let z = -105; z <= 105; z += 10) {
                const xOff = -45 - (Math.abs(z * 11) % 5);
                const s = 0.85 + (Math.abs(z * 7) % 4) * 0.09;
                const spec = speciesList[Math.abs(Math.floor(z / 10)) % speciesList.length];

                if ((z / 10) % 2 === 0) {
                    buildPine(xOff, z, s, z > 0);
                } else {
                    buildDeciduous(xOff, z, s, spec);
                }
                // Natural green roadside decorative shrub along curb
                buildRoadsideShrub(-41.5, z + 5, 0.8, 0x15803d);
            }

            // 2. East Highway Trees & Shrubs (X = +44 to +52)
            for (let z = -105; z <= 105; z += 10) {
                const xOff = 45 + (Math.abs(z * 13) % 5);
                const s = 0.85 + (Math.abs(z * 9) % 4) * 0.09;
                const spec = speciesList[Math.abs(Math.floor(z / 10) + 2) % speciesList.length];

                if ((z / 10) % 2 === 0) {
                    buildDeciduous(xOff, z, s, spec);
                } else {
                    buildPine(xOff, z, s, z < 0);
                }
                // Natural green roadside decorative shrub along curb
                buildRoadsideShrub(41.5, z + 5, 0.8, 0x166534);
            }

            // 3. Central Median Trees & Foliage (Inside loop)
            for (let z = -55; z <= 55; z += 15) {
                buildDeciduous(-20, z, 0.8, 'forest');
                buildDeciduous(20, z, 0.8, 'summer_oak');
                buildRoadsideShrub(0, z, 1.1, 0x15803d);
            }

            // 4. North & South Outer Curve Green Belts
            const curveSteps = 20;
            for (let i = 0; i <= curveSteps; i++) {
                const theta = (i / curveSteps) * Math.PI;
                const rN = 48 + (i % 3) * 3;
                const xN = -rN * Math.cos(theta);
                const zN = -straightHalf - rN * Math.sin(theta);
                buildPine(xN, zN, 0.95);

                const rS = 48 + (i % 3) * 3;
                const xS = rS * Math.cos(theta);
                const zS = straightHalf + rS * Math.sin(theta);
                buildDeciduous(xS, zS, 0.95, (i % 2 === 0) ? 'emerald' : 'woodland');
            }

            // 5. Galvanized Steel Highway Guardrails
            const railMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.88, roughness: 0.22 });
            [-41.2, 41.2].forEach(rx => {
                const rail = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.35, straightHalf * 2), railMat);
                rail.position.set(rx, 0.65, 0);
                rail.castShadow = true;
                sceneryGroup.add(rail);

                for (let rz = -straightHalf; rz <= straightHalf; rz += 5) {
                    const post = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.65, 0.12), railMat);
                    post.position.set(rx, 0.32, rz);
                    sceneryGroup.add(post);
                }
            });

            // 6. Overhead Highway Digital Direction Gantry
            const gantryMat = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.85, roughness: 0.25 });
            const gantry = new THREE.Group();
            const pL = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.24, 7.2, 8), gantryMat);
            pL.position.set(-42, 3.6, -30);
            gantry.add(pL);
            const pR = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.24, 7.2, 8), gantryMat);
            pR.position.set(-26, 3.6, -30);
            gantry.add(pR);
            const crossBeam = new THREE.Mesh(new THREE.BoxGeometry(16.5, 0.45, 0.45), gantryMat);
            crossBeam.position.set(-34, 7.0, -30);
            gantry.add(crossBeam);

            const signBoard = new THREE.Mesh(
                new THREE.BoxGeometry(9.5, 2.4, 0.14),
                new THREE.MeshStandardMaterial({ color: 0x047857, roughness: 0.35 })
            );
            signBoard.position.set(-34, 6.9, -29.9);
            gantry.add(signBoard);

            sceneryGroup.add(gantry);
            this.scene.add(sceneryGroup);
        }

        // ---------------------------------------------------------------------
        // 4. Rich Architectural City Metropolis (Realistic Colorful Buildings & Lighting)
        // ---------------------------------------------------------------------
        _initCityEnvironment() {
            const scenery = new THREE.Group();
            const straightHalf = 75;

            // Central Park (Inside the highway loop: |x| <= 18, |z| <= 55)
            const parkTile = new THREE.Mesh(
                new THREE.BoxGeometry(32, 0.16, 110),
                new THREE.MeshStandardMaterial({ color: 0x166534, roughness: 0.85 })
            );
            parkTile.position.set(0, 0.08, 0);
            parkTile.receiveShadow = true;
            scenery.add(parkTile);

            // Modern Curved Glass Pavilion with Blue Glow
            const pav = new THREE.Mesh(
                new THREE.BoxGeometry(14, 8.5, 22),
                new THREE.MeshStandardMaterial({ color: 0x0284c7, metalness: 0.92, roughness: 0.12, transparent: true, opacity: 0.88 })
            );
            pav.position.set(0, 4.3, 0);
            pav.castShadow = true;
            scenery.add(pav);

            // Water Fountain / Pool with glowing water
            const pool = new THREE.Mesh(
                new THREE.CylinderGeometry(6.0, 6.0, 0.5, 24),
                new THREE.MeshStandardMaterial({ color: 0x06b6d4, metalness: 0.95, roughness: 0.08 })
            );
            pool.position.set(0, 0.28, 30);
            scenery.add(pool);

            // Building Palette Definitions (Architectural Realism!)
            const buildingStyles = [
                { base: 0x0284c7, glass: 0x38bdf8, name: 'Cyan Mirror Highrise' },
                { base: 0x1e3a8a, glass: 0x93c5fd, name: 'Sapphire Corporate Tower' },
                { base: 0xc2410c, glass: 0xfde047, name: 'Terracotta Brick Tower' },
                { base: 0xd97706, glass: 0xfef08a, name: 'Amber Sandstone Complex' },
                { base: 0x334155, glass: 0x67e8f9, name: 'Graphite Tech Center' },
                { base: 0xf8fafc, glass: 0x0284c7, name: 'Modernist Ivory Tower' }
            ];

            const winMatWarm = new THREE.MeshBasicMaterial({ color: 0xfef08a });
            const winMatCyan = new THREE.MeshBasicMaterial({ color: 0x67e8f9 });
            const winMatWhite = new THREE.MeshBasicMaterial({ color: 0xffffff });
            const winMats = [winMatWarm, winMatCyan, winMatWhite];

            // Helper to build a highly realistic skyscraper
            const buildSkyscraper = (x, z, w, d, h, styleIdx) => {
                const style = buildingStyles[styleIdx % buildingStyles.length];
                const bldgGroup = new THREE.Group();

                // Main Tower Body
                const bodyMat = new THREE.MeshStandardMaterial({
                    color: style.base,
                    metalness: 0.75,
                    roughness: 0.28
                });
                const body = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), bodyMat);
                body.position.y = h / 2;
                body.castShadow = true;
                body.receiveShadow = true;
                bldgGroup.add(body);

                // Multiple Rows of Illuminated Windows (Facade detail!)
                const numFloors = Math.floor(h / 3.6);
                for (let f = 1; f < numFloors; f++) {
                    const fy = f * 3.6;
                    const winMat = winMats[(f + styleIdx) % winMats.length];

                    // Front & Back illuminated window bands
                    [-d / 2 - 0.04, d / 2 + 0.04].forEach(wz => {
                        const winBand = new THREE.Mesh(new THREE.PlaneGeometry(w * 0.85, 1.2), winMat);
                        if (wz < 0) winBand.rotation.y = Math.PI;
                        winBand.position.set(0, fy, wz);
                        bldgGroup.add(winBand);
                    });

                    // Left & Right illuminated window bands
                    [-w / 2 - 0.04, w / 2 + 0.04].forEach(wx => {
                        const winBand = new THREE.Mesh(new THREE.PlaneGeometry(d * 0.85, 1.2), winMat);
                        winBand.rotation.y = wx < 0 ? -Math.PI / 2 : Math.PI / 2;
                        winBand.position.set(wx, fy, 0);
                        bldgGroup.add(winBand);
                    });
                }

                // Rooftop Architectural Penthouse & HVAC Chillers
                const roofH = 3.5;
                const penthouse = new THREE.Mesh(
                    new THREE.BoxGeometry(w * 0.6, roofH, d * 0.6),
                    new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.85, roughness: 0.3 })
                );
                penthouse.position.set(0, h + roofH / 2, 0);
                bldgGroup.add(penthouse);

                // Radio Transmission Antenna with Blinking Red Beacon
                const antenna = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.1, 0.22, 9.0, 8),
                    new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.9 })
                );
                antenna.position.set(0, h + roofH + 4.5, 0);
                bldgGroup.add(antenna);

                const beacon = new THREE.Mesh(
                    new THREE.SphereGeometry(0.35, 8, 8),
                    new THREE.MeshBasicMaterial({ color: 0xef4444 })
                );
                beacon.position.set(0, h + roofH + 9.0, 0);
                bldgGroup.add(beacon);

                bldgGroup.position.set(x, 0, z);
                scenery.add(bldgGroup);
            };

            // 1. West Corridor Skyline Towers (Spaced out to keep environment open & scenic)
            let bIdx = 0;
            for (let z = -90; z <= 90; z += 60) {
                const w = 18 + (Math.abs(z) % 6);
                const d = 18 + (Math.abs(z * 3) % 6);
                const h = 32 + (Math.abs(z * 7) % 30);
                buildSkyscraper(-72, z, w, d, h, bIdx++);
            }

            // 2. East Corridor Skyline Towers (Spaced out to keep environment open & scenic)
            for (let z = -90; z <= 90; z += 60) {
                const w = 18 + (Math.abs(z * 5) % 6);
                const d = 18 + (Math.abs(z * 2) % 6);
                const h = 34 + (Math.abs(z * 9) % 32);
                buildSkyscraper(72, z, w, d, h, bIdx++);
            }

            // Streetlights with Warm Glowing Illumination along sidewalks
            const poleMat = new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.85 });
            const lampMat = new THREE.MeshBasicMaterial({ color: 0xfef08a });

            [-42.5, 42.5].forEach(x => {
                for (let z = -straightHalf; z <= straightHalf; z += 25) {
                    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.14, 5.8, 8), poleMat);
                    pole.position.set(x, 2.9, z);
                    pole.castShadow = true;
                    scenery.add(pole);

                    const lampArm = new THREE.Mesh(new THREE.BoxGeometry(x > 0 ? -1.2 : 1.2, 0.1, 0.1), poleMat);
                    lampArm.position.set(x > 0 ? x - 0.6 : x + 0.6, 5.7, z);
                    scenery.add(lampArm);

                    const lamp = new THREE.Mesh(new THREE.SphereGeometry(0.28, 8, 8), lampMat);
                    lamp.position.set(x > 0 ? x - 1.2 : x + 1.2, 5.6, z);
                    scenery.add(lamp);
                }
            });

            this.scene.add(scenery);
        }

        // ---------------------------------------------------------------------
        // 5. High-Fidelity Ego Vehicle (Contoured Sports Sedan with Sport Wheels)
        // ---------------------------------------------------------------------
        _initEgoVehicle() {
            const car = new THREE.Group();

            const paintMat = new THREE.MeshStandardMaterial({
                color: 0x0284c7,
                metalness: 0.88,
                roughness: 0.22
            });
            const darkCarbonMat = new THREE.MeshStandardMaterial({
                color: 0x0f172a,
                metalness: 0.82,
                roughness: 0.3
            });

            // 1. Lower Body Chassis (Contoured wedge)
            const lowerChassis = new THREE.Mesh(new THREE.BoxGeometry(1.98, 0.46, 4.4), paintMat);
            lowerChassis.position.y = 0.48;
            lowerChassis.castShadow = true;
            lowerChassis.receiveShadow = true;
            car.add(lowerChassis);

            // Side Skirts & Carbon Trim
            [-1.0, 1.0].forEach(sx => {
                const sideSkirt = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.16, 2.8), darkCarbonMat);
                sideSkirt.position.set(sx, 0.26, 0);
                car.add(sideSkirt);
            });

            // 2. Sculpted Hood (Sloped front)
            const hood = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.18, 1.4), paintMat);
            hood.position.set(0, 0.72, -1.35);
            hood.rotation.x = 0.08;
            hood.castShadow = true;
            car.add(hood);

            // Front Grille & Air Intake
            const grille = new THREE.Mesh(new THREE.BoxGeometry(1.3, 0.22, 0.08), darkCarbonMat);
            grille.position.set(0, 0.42, -2.21);
            car.add(grille);

            // 3. Cabin / Greenhouse (Curved tinted glass canopy)
            const glassMat = new THREE.MeshStandardMaterial({
                color: 0x0b132b,
                roughness: 0.08,
                metalness: 0.95,
                transparent: true,
                opacity: 0.92
            });
            const cabin = new THREE.Mesh(new THREE.BoxGeometry(1.54, 0.52, 2.1), glassMat);
            cabin.position.set(0, 1.05, -0.05);
            cabin.castShadow = true;
            car.add(cabin);

            // Roof Cap (Colored metal roof panel)
            const roofCap = new THREE.Mesh(new THREE.BoxGeometry(1.42, 0.06, 1.4), paintMat);
            roofCap.position.set(0, 1.32, -0.05);
            car.add(roofCap);

            // Windshield Frame Pillars
            [-0.72, 0.72].forEach(px => {
                const pillar = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.54, 0.08), darkCarbonMat);
                pillar.rotation.x = 0.52;
                pillar.position.set(px, 1.02, -1.02);
                car.add(pillar);
            });

            // 4. Aerodynamic Side Mirrors
            [-1.02, 1.02].forEach(mx => {
                const mirrorMount = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.06, 0.06), darkCarbonMat);
                mirrorMount.position.set(mx, 0.94, -0.85);
                car.add(mirrorMount);

                const mirrorCap = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.12, 0.14), paintMat);
                mirrorCap.position.set(mx + (mx > 0 ? 0.1 : -0.1), 0.94, -0.85);
                car.add(mirrorCap);
            });

            // 5. Dual Crystalline LED Projector Headlights
            const hlMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
            const hlHaloMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });

            [-0.72, 0.72].forEach(hx => {
                const hlBox = new THREE.Mesh(new THREE.BoxGeometry(0.36, 0.14, 0.12), hlHaloMat);
                hlBox.position.set(hx, 0.62, -2.18);
                car.add(hlBox);

                const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.06, 8, 8), hlMat);
                bulb.position.set(hx, 0.62, -2.23);
                car.add(bulb);
            });

            // Forward Spotlight Beam
            const spotLight = new THREE.SpotLight(0x38bdf8, 4.2, 45, Math.PI / 6.5, 0.35);
            spotLight.position.set(0, 0.75, -2.0);
            spotLight.target.position.set(0, 0, -30);
            car.add(spotLight);
            car.add(spotLight.target);

            // 6. Rear OLED Continuous Light Bar & Diffuser
            const tl = new THREE.Mesh(new THREE.BoxGeometry(1.82, 0.08, 0.06), new THREE.MeshBasicMaterial({ color: 0xf43f5e }));
            tl.position.set(0, 0.68, 2.21);
            car.add(tl);

            const spoiler = new THREE.Mesh(new THREE.BoxGeometry(1.65, 0.05, 0.22), darkCarbonMat);
            spoiler.position.set(0, 0.88, 2.12);
            car.add(spoiler);

            const diffuser = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.18, 0.15), darkCarbonMat);
            diffuser.position.set(0, 0.32, 2.18);
            car.add(diffuser);

            // 7. Sport Alloy Wheels with Visible Red Brake Calipers
            const tireMat = new THREE.MeshStandardMaterial({ color: 0x18181b, roughness: 0.85 });
            const rimMat = new THREE.MeshStandardMaterial({ color: 0xd1d5db, metalness: 0.92, roughness: 0.12 });
            const caliperMat = new THREE.MeshStandardMaterial({ color: 0xdc2626, roughness: 0.3 });

            const tireGeo = new THREE.CylinderGeometry(0.37, 0.37, 0.28, 24);
            tireGeo.rotateZ(Math.PI / 2);

            const wheelOffsets = [
                { x: -1.02, y: 0.37, z: -1.4, isFront: true },
                { x:  1.02, y: 0.37, z: -1.4, isFront: true },
                { x: -1.02, y: 0.37, z:  1.4, isFront: false },
                { x:  1.02, y: 0.37, z:  1.4, isFront: false }
            ];

            this.ego.wheels = [];
            wheelOffsets.forEach(pos => {
                const wGroup = new THREE.Group();
                wGroup.position.set(pos.x, pos.y, pos.z);

                const tire = new THREE.Mesh(tireGeo, tireMat);
                tire.castShadow = true;
                wGroup.add(tire);

                const rimLip = new THREE.Mesh(new THREE.CylinderGeometry(0.26, 0.26, 0.29, 16), rimMat);
                rimLip.rotateZ(Math.PI / 2);
                wGroup.add(rimLip);

                for (let sp = 0; sp < 5; sp++) {
                    const spoke = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.24, 0.05), rimMat);
                    spoke.rotation.x = (sp / 5) * Math.PI * 2;
                    spoke.position.x = pos.x > 0 ? 0.12 : -0.12;
                    wGroup.add(spoke);
                }

                const caliper = new THREE.Mesh(new THREE.BoxGeometry(0.12, 0.16, 0.1), caliperMat);
                caliper.position.set(pos.x > 0 ? 0.04 : -0.04, 0.12, -0.1);
                wGroup.add(caliper);

                car.add(wGroup);
                this.ego.wheels.push({ group: wGroup, isFront: pos.isFront });
            });

            // 8. Streamlined Rooftop Autonomous LiDAR Sensor Pod
            const podMount = new THREE.Mesh(
                new THREE.CylinderGeometry(0.24, 0.28, 0.14, 16),
                darkCarbonMat
            );
            podMount.position.set(0, 1.39, 0.12);
            car.add(podMount);

            const puckHead = new THREE.Mesh(
                new THREE.CylinderGeometry(0.18, 0.18, 0.12, 16),
                new THREE.MeshBasicMaterial({ color: 0x00f0ff })
            );
            puckHead.position.set(0, 1.50, 0.12);
            car.add(puckHead);
            this.ego.lidarPuck = puckHead;

            // 360° Rotating Laser Sweep Cone
            const sweepGeo = new THREE.ConeGeometry(35, 0.15, 32, 1, true, 0, Math.PI / 6);
            sweepGeo.rotateX(Math.PI / 2);
            const sweepMat = new THREE.MeshBasicMaterial({
                color: 0x38bdf8,
                transparent: true,
                opacity: 0.30,
                side: THREE.DoubleSide
            });
            const sweepMesh = new THREE.Mesh(sweepGeo, sweepMat);
            sweepMesh.position.set(0, 1.48, 0.12);
            car.add(sweepMesh);
            this.ego.lidarSweep = sweepMesh;

            car.position.set(this.ego.x, 0, this.ego.z);
            car.rotation.y = this.ego.yaw;

            this.scene.add(car);
            this.ego.mesh = car;
        }

        // ---------------------------------------------------------------------
        // 6. Holographic 3 Semantic Rings (EXACTLY 3 RINGS: Ring 0, 1, 2)
        // ---------------------------------------------------------------------
        _initHolographicLidarRings() {
            const ringsGroup = new THREE.Group();

            // Dark Semantic LiDAR Rings with High-Contrast Channel & Crisp Accent Rails
            // Matches BEV 3-Ring design: deep dark slate/black annular lane (0x020617)
            // so the rings read as crisp, dark demarcations instead of washed-out light lines.
            const ringConfigs = [
                {
                    r: 10.0,
                    darkInner: 8.6,
                    darkOuter: 11.4,
                    railInner: 9.85,
                    railOuter: 10.15,
                    color: 0x38bdf8,
                    accentHex: '#38bdf8',
                    y: 0.12,
                    label: 'NEAR 0–10m (5cm)',
                    labelDist: 10
                },
                {
                    r: 30.0,
                    darkInner: 28.5,
                    darkOuter: 31.5,
                    railInner: 29.85,
                    railOuter: 30.15,
                    color: 0xc084fc,
                    accentHex: '#c084fc',
                    y: 0.10,
                    label: 'MID 10–30m (15cm)',
                    labelDist: 30
                },
                {
                    r: 70.0,
                    darkInner: 68.0,
                    darkOuter: 72.0,
                    railInner: 69.80,
                    railOuter: 70.20,
                    color: 0xfb923c,
                    accentHex: '#fb923c',
                    y: 0.08,
                    label: 'FAR 30–70m (50cm)',
                    labelDist: 70
                }
            ];

            ringConfigs.forEach(cfg => {
                // 1. Prominent Dark Annular Lane Body (crisp dark black/slate channel)
                const darkLaneMat = new THREE.MeshBasicMaterial({
                    color: 0x020617,
                    transparent: true,
                    opacity: 0.94,
                    side: THREE.DoubleSide,
                    depthWrite: false
                });
                const darkLaneGeo = new THREE.RingGeometry(cfg.darkInner, cfg.darkOuter, 96);
                const darkLaneMesh = new THREE.Mesh(darkLaneGeo, darkLaneMat);
                darkLaneMesh.rotation.x = -Math.PI / 2;
                darkLaneMesh.position.y = cfg.y;
                ringsGroup.add(darkLaneMesh);

                // 2. Dark inner & outer border rims (pure dark edge definition)
                const darkBorderMat = new THREE.MeshBasicMaterial({
                    color: 0x000000,
                    transparent: true,
                    opacity: 0.98,
                    side: THREE.DoubleSide,
                    depthWrite: false
                });
                const borderInnerGeo = new THREE.RingGeometry(cfg.darkInner - 0.15, cfg.darkInner + 0.05, 96);
                const borderInnerMesh = new THREE.Mesh(borderInnerGeo, darkBorderMat);
                borderInnerMesh.rotation.x = -Math.PI / 2;
                borderInnerMesh.position.y = cfg.y + 0.005;
                ringsGroup.add(borderInnerMesh);

                const borderOuterGeo = new THREE.RingGeometry(cfg.darkOuter - 0.05, cfg.darkOuter + 0.15, 96);
                const borderOuterMesh = new THREE.Mesh(borderOuterGeo, darkBorderMat);
                borderOuterMesh.rotation.x = -Math.PI / 2;
                borderOuterMesh.position.y = cfg.y + 0.005;
                ringsGroup.add(borderOuterMesh);

                // 3. Crisp Coloured Center Rail / Indicator Line atop the dark lane
                const railMat = new THREE.MeshBasicMaterial({
                    color: cfg.color,
                    transparent: true,
                    opacity: 0.96,
                    side: THREE.DoubleSide,
                    depthWrite: false
                });
                const railGeo = new THREE.RingGeometry(cfg.railInner, cfg.railOuter, 96);
                const railMesh = new THREE.Mesh(railGeo, railMat);
                railMesh.rotation.x = -Math.PI / 2;
                railMesh.position.y = cfg.y + 0.015;
                ringsGroup.add(railMesh);

                // 4. Subtle semantic colored tint across the dark lane for depth
                const tintMat = new THREE.MeshBasicMaterial({
                    color: cfg.color,
                    transparent: true,
                    opacity: 0.16,
                    side: THREE.DoubleSide,
                    depthWrite: false
                });
                const tintGeo = new THREE.RingGeometry(cfg.darkInner, cfg.darkOuter, 96);
                const tintMesh = new THREE.Mesh(tintGeo, tintMat);
                tintMesh.rotation.x = -Math.PI / 2;
                tintMesh.position.y = cfg.y + 0.01;
                ringsGroup.add(tintMesh);

                // 5. Dark Text label pill sprites at cardinal positions (North and East)
                const labelPositions = [
                    { x: 0, z: -cfg.labelDist },  // North
                    { x: cfg.labelDist, z: 0 }     // East
                ];
                labelPositions.forEach(pos => {
                    const canvas = document.createElement('canvas');
                    canvas.width = 256;
                    canvas.height = 48;
                    const lctx = canvas.getContext('2d');
                    lctx.clearRect(0, 0, 256, 48);

                    // Solid dark obsidian pill
                    lctx.fillStyle = 'rgba(2, 6, 23, 0.95)';
                    lctx.roundRect(0, 4, 256, 40, 8);
                    lctx.fill();

                    // Bold border in ring accent color
                    lctx.strokeStyle = cfg.accentHex;
                    lctx.lineWidth = 2.5;
                    lctx.roundRect(0, 4, 256, 40, 8);
                    lctx.stroke();

                    // Crisp high-legibility text
                    lctx.fillStyle = '#f8fafc';
                    lctx.font = 'bold 20px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
                    lctx.textAlign = 'center';
                    lctx.fillText(cfg.label, 128, 31);

                    const tex = new THREE.CanvasTexture(canvas);
                    tex.minFilter = THREE.LinearFilter;
                    const spriteMat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false });
                    const sprite = new THREE.Sprite(spriteMat);
                    sprite.position.set(pos.x, 2.2, pos.z);
                    sprite.scale.set(8, 1.5, 1);
                    ringsGroup.add(sprite);
                });
            });

            ringsGroup.position.set(this.ego.x, 0, this.ego.z);
            this.scene.add(ringsGroup);
            this.ego.lidarRingsGroup = ringsGroup;
        }

        // ---------------------------------------------------------------------
        // 7. 12 Anatomical 3D Pedestrians with Articulated Limbs & Walk Cycle
        // ---------------------------------------------------------------------
        _initPedestrians() {
            this.pedestrians = [];

            const pedConfigs = [
                // Pedestrian 1: Crossing West Zebra #1 (Z = -40)
                { startX: -41.0, startZ: -40, targetX: -27.0, targetZ: -40, speed: 1.35, jacketColor: 0x0284c7, pantsColor: 0x1e293b, capColor: 0x0f172a, label: 'person', bag: true },
                // Pedestrian 2: Crossing West Zebra #2 (Z = -10) - direct approach!
                { startX: -27.0, startZ: -10, targetX: -41.0, targetZ: -10, speed: 1.40, jacketColor: 0xf43f5e, pantsColor: 0x334155, capColor: 0x991b1b, label: 'person', bag: false },
                // Pedestrian 3: Crossing West Zebra #3 (Z = 30)
                { startX: -41.0, startZ: 30, targetX: -27.0, targetZ: 30, speed: 1.30, jacketColor: 0xf59e0b, pantsColor: 0x0f172a, capColor: 0x78350f, label: 'person', bag: true },
                // Pedestrian 4: Crossing East Zebra #1 (Z = -35)
                { startX: 27.0, startZ: -35, targetX: 41.0, targetZ: -35, speed: 1.30, jacketColor: 0x10b981, pantsColor: 0x1e293b, capColor: 0x064e3b, label: 'person', bag: false },
                // Pedestrian 5: Crossing East Zebra #2 (Z = 15)
                { startX: 41.0, startZ: 15, targetX: 27.0, targetZ: 15, speed: 1.45, jacketColor: 0xa855f7, pantsColor: 0x334155, capColor: 0x581c87, label: 'person', bag: true },
                // Pedestrian 6: Crossing East Zebra #3 (Z = 50)
                { startX: 27.0, startZ: 50, targetX: 41.0, targetZ: 50, speed: 1.25, jacketColor: 0x06b6d4, pantsColor: 0x1e293b, capColor: 0x0e7490, label: 'person', bag: false },
                // Pedestrian 7: Walking along North sidewalk curve (Z = -100)
                { startX: -22, startZ: -104, targetX: 22, targetZ: -104, speed: 1.15, jacketColor: 0xeab308, pantsColor: 0x1e293b, capColor: 0x713f12, label: 'person', bag: true },
                // Pedestrian 8: Walking along South sidewalk curve (Z = 100)
                { startX: 22, startZ: 104, targetX: -22, targetZ: 104, speed: 1.20, jacketColor: 0xec4899, pantsColor: 0x0f172a, capColor: 0x831843, label: 'person', bag: false },
                // Pedestrian 9: Jogging in Central Park West path
                { startX: -7, startZ: -45, targetX: -7, targetZ: 45, speed: 2.10, jacketColor: 0x3b82f6, pantsColor: 0x0f172a, capColor: 0x1d4ed8, label: 'person', bag: false },
                // Pedestrian 10: Jogging in Central Park East path
                { startX: 7, startZ: 42, targetX: 7, targetZ: -42, speed: 1.95, jacketColor: 0x10b981, pantsColor: 0x334155, capColor: 0x047857, label: 'person', bag: false },
                // Pedestrian 11: Stroller near Central Park fountain
                { startX: -12, startZ: 30, targetX: 12, targetZ: 30, speed: 0.95, jacketColor: 0xf97316, pantsColor: 0x1e293b, capColor: 0xc2410c, label: 'person', bag: true },
                // Pedestrian 12: Sidewalk pedestrian along West curb
                { startX: -42.5, startZ: 10, targetX: -42.5, targetZ: -10, speed: 1.05, jacketColor: 0x8b5cf6, pantsColor: 0x334155, capColor: 0x6d28d9, label: 'person', bag: false }
            ];

            pedConfigs.forEach(cfg => {
                const pedGroup = new THREE.Group();

                const skinMat = new THREE.MeshStandardMaterial({ color: 0xd4a373, roughness: 0.75 });
                const jacketMat = new THREE.MeshStandardMaterial({ color: cfg.jacketColor, roughness: 0.65 });
                const pantsMat = new THREE.MeshStandardMaterial({ color: cfg.pantsColor, roughness: 0.85 });
                const shoeMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.55 });
                const soleMat = new THREE.MeshBasicMaterial({ color: 0xf8fafc });
                const capMat = new THREE.MeshStandardMaterial({ color: cfg.capColor, roughness: 0.7 });
                const glassMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.9, roughness: 0.1 });

                // 1. Head, Sunglasses & Baseball Cap
                const head = new THREE.Mesh(new THREE.SphereGeometry(0.15, 16, 16), skinMat);
                head.position.y = 1.62;
                head.castShadow = true;
                pedGroup.add(head);

                // Sunglasses / Visor
                const glasses = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.04, 0.08), glassMat);
                glasses.position.set(0, 1.63, -0.12);
                pedGroup.add(glasses);

                const capDome = new THREE.Mesh(new THREE.SphereGeometry(0.155, 16, 16, 0, Math.PI * 2, 0, Math.PI / 2), capMat);
                capDome.position.y = 1.635;
                pedGroup.add(capDome);

                const capVisor = new THREE.Mesh(new THREE.BoxGeometry(0.17, 0.025, 0.13), capMat);
                capVisor.position.set(0, 1.65, -0.15);
                pedGroup.add(capVisor);

                // 2. Neck & Anatomical Torso
                const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.08, 0.09, 8), skinMat);
                neck.position.y = 1.48;
                pedGroup.add(neck);

                const torsoGroup = new THREE.Group();
                torsoGroup.position.set(0, 1.18, 0);

                const torso = new THREE.Mesh(new THREE.BoxGeometry(0.44, 0.52, 0.24), jacketMat);
                torso.castShadow = true;
                torsoGroup.add(torso);

                // Reflective Safety Stripe
                const stripe = new THREE.Mesh(new THREE.BoxGeometry(0.45, 0.05, 0.25), new THREE.MeshBasicMaterial({ color: 0xf8fafc }));
                stripe.position.y = 0.04;
                torsoGroup.add(stripe);

                // Backpack Accessory
                if (cfg.bag) {
                    const bag = new THREE.Mesh(
                        new THREE.BoxGeometry(0.30, 0.38, 0.15),
                        new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.8 })
                    );
                    bag.position.set(0, 0.06, 0.18);
                    bag.castShadow = true;
                    torsoGroup.add(bag);
                }
                pedGroup.add(torsoGroup);

                // 3. Two-Segment Articulated Arms (Shoulder + Elbow)
                const buildArm = (isLeft) => {
                    const armGroup = new THREE.Group();
                    const xPos = isLeft ? -0.28 : 0.28;
                    armGroup.position.set(xPos, 1.40, 0);

                    // Upper arm
                    const upperGeo = new THREE.BoxGeometry(0.10, 0.28, 0.10);
                    upperGeo.translate(0, -0.14, 0);
                    const upper = new THREE.Mesh(upperGeo, jacketMat);
                    upper.castShadow = true;
                    armGroup.add(upper);

                    // Forearm (pivoting at elbow)
                    const forearmGroup = new THREE.Group();
                    forearmGroup.position.set(0, -0.28, 0);
                    const lowerGeo = new THREE.BoxGeometry(0.085, 0.26, 0.085);
                    lowerGeo.translate(0, -0.13, 0);
                    const lower = new THREE.Mesh(lowerGeo, jacketMat);
                    lower.castShadow = true;
                    forearmGroup.add(lower);

                    // Hand
                    const hand = new THREE.Mesh(new THREE.SphereGeometry(0.05, 8, 8), skinMat);
                    hand.position.set(0, -0.28, 0);
                    forearmGroup.add(hand);

                    // Resting elbow flexion
                    forearmGroup.rotation.x = 0.22;
                    armGroup.add(forearmGroup);

                    return { main: armGroup, forearm: forearmGroup };
                };

                const armL = buildArm(true);
                const armR = buildArm(false);
                pedGroup.add(armL.main);
                pedGroup.add(armR.main);

                // 4. Two-Segment Articulated Legs (Hip + Knee Flexion)
                const buildLeg = (isLeft) => {
                    const hipGroup = new THREE.Group();
                    const xPos = isLeft ? -0.12 : 0.12;
                    hipGroup.position.set(xPos, 0.90, 0);

                    // Thigh
                    const thighGeo = new THREE.BoxGeometry(0.14, 0.40, 0.14);
                    thighGeo.translate(0, -0.20, 0);
                    const thigh = new THREE.Mesh(thighGeo, pantsMat);
                    thigh.castShadow = true;
                    hipGroup.add(thigh);

                    // Calf / Knee group (pivots at knee)
                    const kneeGroup = new THREE.Group();
                    kneeGroup.position.set(0, -0.40, 0);
                    const calfGeo = new THREE.BoxGeometry(0.12, 0.38, 0.12);
                    calfGeo.translate(0, -0.19, 0);
                    const calf = new THREE.Mesh(calfGeo, pantsMat);
                    calf.castShadow = true;
                    kneeGroup.add(calf);

                    // Sneaker with rubber sole & white stripe
                    const shoe = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.08, 0.23), shoeMat);
                    shoe.position.set(0, -0.40, -0.04);
                    kneeGroup.add(shoe);

                    const sole = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.03, 0.24), soleMat);
                    sole.position.set(0, -0.44, -0.04);
                    kneeGroup.add(sole);

                    hipGroup.add(kneeGroup);
                    return { hip: hipGroup, knee: kneeGroup };
                };

                const legL = buildLeg(true);
                const legR = buildLeg(false);
                pedGroup.add(legL.hip);
                pedGroup.add(legR.hip);

                pedGroup.position.set(cfg.startX, 0, cfg.startZ);
                this.scene.add(pedGroup);

                this.pedestrians.push({
                    mesh: pedGroup,
                    torso: torsoGroup,
                    armL: armL.main,
                    armR: armR.main,
                    legL: legL.hip,
                    legR: legR.hip,
                    calfL: legL.knee,
                    calfR: legR.knee,
                    x: cfg.startX,
                    z: cfg.startZ,
                    startX: cfg.startX,
                    startZ: cfg.startZ,
                    targetX: cfg.targetX,
                    targetZ: cfg.targetZ,
                    speed: cfg.speed,
                    walkPhase: Math.random() * Math.PI * 2,
                    direction: 1,
                    label: cfg.label
                });
            });
        }

        // ---------------------------------------------------------------------
        // 8. Traffic Fleet (Dynamic Cruising Sedans + Parked Highway Vehicles)
        // ---------------------------------------------------------------------
        _initTrafficVehicles() {
            this.trafficVehicles = [];

            const createCarMesh = (color, metalness = 0.85, roughness = 0.22) => {
                const car = new THREE.Group();
                const carMat = new THREE.MeshStandardMaterial({ color, metalness, roughness });
                const darkTrimMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.8 });

                // Main body
                const body = new THREE.Mesh(new THREE.BoxGeometry(1.92, 0.52, 4.3), carMat);
                body.position.y = 0.48;
                body.castShadow = true;
                car.add(body);

                // Tinted cabin glass
                const cabin = new THREE.Mesh(
                    new THREE.BoxGeometry(1.5, 0.50, 2.1),
                    new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.1, metalness: 0.95, transparent: true, opacity: 0.9 })
                );
                cabin.position.set(0, 0.98, -0.05);
                car.add(cabin);

                // Headlights (White LED)
                const hlMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
                [-0.68, 0.68].forEach(hx => {
                    const hl = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.12, 0.1), hlMat);
                    hl.position.set(hx, 0.55, -2.15);
                    car.add(hl);
                });

                // Taillights (Red LED)
                const tlMat = new THREE.MeshBasicMaterial({ color: 0xf43f5e });
                const tl = new THREE.Mesh(new THREE.BoxGeometry(1.75, 0.08, 0.06), tlMat);
                tl.position.set(0, 0.60, 2.15);
                car.add(tl);

                // Wheels (4 rubber tires with alloy rims)
                const wheelGeo = new THREE.CylinderGeometry(0.34, 0.34, 0.24, 16);
                const wheelMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.9 });
                [[-0.92, -1.3], [0.92, -1.3], [-0.92, 1.3], [0.92, 1.3]].forEach(([wx, wz]) => {
                    const wheel = new THREE.Mesh(wheelGeo, wheelMat);
                    wheel.rotation.z = Math.PI / 2;
                    wheel.position.set(wx, 0.34, wz);
                    car.add(wheel);
                });

                return car;
            };

            const vehicleConfigs = [
                // 1. Moving Traffic Car 1: Cruising Silver Metallic Sedan on West Lane (Moving North)
                { x: -31.5, z: 45, speed: 7.2, dir: -1, yaw: 0, color: 0x94a3b8, label: 'vehicle' },
                // 2. Moving Traffic Car 2: Cruising Deep Navy Blue Sedan on East Lane (Moving South)
                { x: 31.5, z: -35, speed: 6.8, dir: 1, yaw: Math.PI, color: 0x1e3a8a, label: 'vehicle' },
                // 3. Parked Red Sedan on East outer curb
                { x: 39.5, z: -15, speed: 0, dir: 0, yaw: 0, color: 0xb91c1c, label: 'vehicle' },
                // 4. Parked Pearl White SUV on West outer curb
                { x: -39.5, z: 22, speed: 0, dir: 0, yaw: Math.PI, color: 0xf8fafc, label: 'vehicle' },
                // 5. Parked Sunset Bronze Coupe on East outer curb
                { x: 39.5, z: 60, speed: 0, dir: 0, yaw: 0, color: 0xd97706, label: 'vehicle' },
                // 6. Parked Emerald Green Crossover on West outer curb
                { x: -39.5, z: -55, speed: 0, dir: 0, yaw: Math.PI, color: 0x047857, label: 'vehicle' }
            ];

            vehicleConfigs.forEach(cfg => {
                const mesh = createCarMesh(cfg.color);
                mesh.position.set(cfg.x, 0, cfg.z);
                mesh.rotation.y = cfg.yaw;
                this.scene.add(mesh);

                this.trafficVehicles.push({
                    mesh,
                    x: cfg.x,
                    z: cfg.z,
                    speed: cfg.speed,
                    dir: cfg.dir,
                    yaw: cfg.yaw,
                    label: cfg.label
                });
            });
        }

        // ---------------------------------------------------------------------
        // 9. Non-Mirrored Controls & Key Handlers
        // ---------------------------------------------------------------------
        _setupControls() {
            window.addEventListener('keydown', (e) => {
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

                let handled = true;
                switch (e.code) {
                    case 'KeyW':
                    case 'ArrowUp':
                        this.keys.forward = true;
                        this.autopilot = false;
                        this.manualOverrideTimer = 5.0;
                        break;
                    case 'KeyS':
                    case 'ArrowDown':
                        this.keys.backward = true;
                        this.autopilot = false;
                        this.manualOverrideTimer = 5.0;
                        break;
                    case 'KeyA':
                    case 'ArrowLeft':
                        this.keys.left = true;
                        this.autopilot = false;
                        this.manualOverrideTimer = 5.0;
                        break;
                    case 'KeyD':
                    case 'ArrowRight':
                        this.keys.right = true;
                        this.autopilot = false;
                        this.manualOverrideTimer = 5.0;
                        break;
                    case 'Space':
                        this.keys.brake = true;
                        this.autopilot = false;
                        break;
                    case 'KeyP':
                        this.autopilot = !this.autopilot;
                        this._updateDrivingHud();
                        break;
                    case 'KeyC':
                        this.cycleCameraMode();
                        break;
                    default:
                        handled = false;
                }
                if (handled) e.preventDefault();
            });

            window.addEventListener('keyup', (e) => {
                switch (e.code) {
                    case 'KeyW':
                    case 'ArrowUp':
                        this.keys.forward = false;
                        break;
                    case 'KeyS':
                    case 'ArrowDown':
                        this.keys.backward = false;
                        break;
                    case 'KeyA':
                    case 'ArrowLeft':
                        this.keys.left = false;
                        break;
                    case 'KeyD':
                    case 'ArrowRight':
                        this.keys.right = false;
                        break;
                    case 'Space':
                        this.keys.brake = false;
                        break;
                }
            });
        }

        cycleCameraMode() {
            const modes = ['chase', 'cockpit', 'overhead'];
            const nextIdx = (modes.indexOf(this.cameraMode) + 1) % modes.length;
            this.cameraMode = modes[nextIdx];
            this._updateDrivingHud();
        }

        // ---------------------------------------------------------------------
        // 10. Update Loop: Non-Mirrored Controls & Autopilot Tracking
        // ---------------------------------------------------------------------
        update(dt = 0.016) {
            if (this.manualOverrideTimer > 0) {
                this.manualOverrideTimer -= dt;
            }

            if (this.autopilot && this.manualOverrideTimer <= 0) {
                this._updateAutopilot(dt);
            } else {
                this._updateManualDriving(dt);
            }

            // Hard collision physics: prevents penetrating pedestrians, parked cars, or guardrails!
            this._resolveCollisions(dt);

            // Apply vehicle transform
            this.ego.mesh.position.set(this.ego.x, 0, this.ego.z);
            this.ego.mesh.rotation.y = this.ego.yaw;

            // Animate wheels & steering angle
            this.ego.wheelRotation += (this.ego.speed * dt) / 0.36;
            this.ego.wheels.forEach(w => {
                w.group.children[0].rotation.x = this.ego.wheelRotation;
                if (w.isFront) {
                    // Non-mirrored steering of front wheels
                    w.group.rotation.y = -this.ego.steerAngle;
                }
            });

            // Spin LiDAR sensor optical puck & laser sweep cone
            this.sweepAngle = (this.sweepAngle + dt * 18.0) % (Math.PI * 2);
            if (this.ego.lidarPuck) this.ego.lidarPuck.rotation.y = this.sweepAngle;
            if (this.ego.lidarSweep) this.ego.lidarSweep.rotation.y = this.sweepAngle;

            // Move Holographic LiDAR rings with vehicle
            if (this.ego.lidarRingsGroup) {
                this.ego.lidarRingsGroup.position.set(this.ego.x, 0, this.ego.z);
            }

            // Drift atmospheric 3D clouds
            if (this.clouds && this.clouds.length) {
                this.clouds.forEach(c => {
                    c.position.x += 1.8 * dt;
                    if (c.position.x > 200) c.position.x = -200;
                });
            }

            // Animate Pedestrians with Articulated Gait (Forward Oriented!)
            this._updatePedestrians(dt);

            // Animate Dynamic Traffic Vehicles
            this._updateTrafficVehicles(dt);

            // Update Cameras
            this._updateCameras();

            // Update Driving Telemetry HUD
            this._updateDrivingHud();
        }

        _resolveCollisions(dt) {
            this.collisionAlert = false;

            const sinYaw = Math.sin(this.ego.yaw);
            const cosYaw = Math.cos(this.ego.yaw);

            // Vehicle collision disks along centerline: front bumper, center chassis, rear bumper
            const carDisks = [
                { x: this.ego.x - sinYaw * 1.5, z: this.ego.z - cosYaw * 1.5, r: 1.15 },
                { x: this.ego.x, z: this.ego.z, r: 1.15 },
                { x: this.ego.x + sinYaw * 1.5, z: this.ego.z + cosYaw * 1.5, r: 1.15 }
            ];

            // 1. Collision with Pedestrians: HARD PHYSICAL ARREST (Never penetrates!)
            this.pedestrians.forEach(ped => {
                carDisks.forEach(disk => {
                    const dx = this.ego.x - ped.x;
                    const dz = this.ego.z - ped.z;
                    const dist = Math.hypot(dx, dz);
                    const minDist = disk.r + 0.65; // 1.80m minimum safe separation

                    if (dist < minDist && dist > 0.001) {
                        this.collisionAlert = true;
                        const overlap = minDist - dist;
                        const nx = dx / dist;
                        const nz = dz / dist;

                        // Immediately push car backwards along collision normal
                        this.ego.x += nx * overlap * 0.85;
                        this.ego.z += nz * overlap * 0.85;

                        // Hard arrest velocity (slight elastic rebound, no penetration)
                        this.ego.speed = Math.min(0, this.ego.speed * -0.2);

                        // Human takes safe step back
                        ped.x -= nx * overlap * 0.15;
                        ped.z -= nz * overlap * 0.15;
                    }
                });
            });

            // 2. Collision with Parked Traffic Vehicles
            this.trafficVehicles.forEach(tv => {
                const dx = this.ego.x - tv.x;
                const dz = this.ego.z - tv.z;
                const dist = Math.hypot(dx, dz);
                const minDist = 3.8;
                if (dist < minDist && dist > 0.001) {
                    this.collisionAlert = true;
                    const overlap = minDist - dist;
                    this.ego.x += (dx / dist) * overlap;
                    this.ego.z += (dz / dist) * overlap;
                    this.ego.speed = Math.min(0, this.ego.speed * -0.3);
                }
            });

            // 3. Collision with Highway Guardrails (|X| > 39.0)
            const maxTrackX = 39.0;
            if (Math.abs(this.ego.x) > maxTrackX) {
                this.collisionAlert = true;
                this.ego.x = Math.sign(this.ego.x) * maxTrackX;
                // Guide vehicle smoothly back toward road centerline
                this.ego.yaw = this.ego.x < 0 ? 0.0 : Math.PI;
                this.ego.steerAngle = 0;
                this.ego.speed = Math.max(3.0, this.ego.speed * 0.75);
            }
        }

        _updateAutopilot(dt) {
            const totalWp = this.circuitWaypoints.length;
            const curWp = this.circuitWaypoints[this.currentWpIndex];
            const distToCur = Math.hypot(curWp.x - this.ego.x, curWp.z - this.ego.z);

            // Advance strictly sequential waypoint when within 6.5m
            if (distToCur < 6.5) {
                this.currentWpIndex = (this.currentWpIndex + 1) % totalWp;
            }

            // Lookahead point 2 steps ahead (~10m) for smooth arc following
            const lookaheadIdx = (this.currentWpIndex + 2) % totalWp;
            const targetWp = this.circuitWaypoints[lookaheadIdx];

            const dx = targetWp.x - this.ego.x;
            const dz = targetWp.z - this.ego.z;

            // Compute base heading error to target
            const targetYaw = Math.atan2(-dx, -dz);
            let diffYaw = targetYaw - this.ego.yaw;
            while (diffYaw > Math.PI) diffYaw -= Math.PI * 2;
            while (diffYaw < -Math.PI) diffYaw += Math.PI * 2;

            let baseSteer = THREE.MathUtils.clamp(-diffYaw * 1.4, -0.45, 0.45);
            let desiredSpeed = (Math.abs(diffYaw) > 0.22) ? this.ego.turnSpeed : this.ego.cruiseSpeed;

            // ── ACTIVE HUMAN PROXIMITY DETECTION: SLOW DOWN & EVADE DIRECTION ──
            const egoPos = new THREE.Vector3(this.ego.x, 0, this.ego.z);
            const egoFwd = new THREE.Vector3(-Math.sin(this.ego.yaw), 0, -Math.cos(this.ego.yaw));
            const egoRight = new THREE.Vector3(Math.cos(this.ego.yaw), 0, -Math.sin(this.ego.yaw));

            let evasionSteer = 0;
            let nearestHazardDist = 999;
            this.isEvadingPedestrian = false;

            this.pedestrians.forEach(ped => {
                const pedPos = new THREE.Vector3(ped.x, 0, ped.z);
                const toPed = new THREE.Vector3().subVectors(pedPos, egoPos);
                const forwardDist = toPed.dot(egoFwd);
                const lateralDist = toPed.dot(egoRight);

                // If pedestrian is within 16 meters ahead and inside vehicle driving corridor (< 5.8m lateral)
                if (forwardDist > 0.3 && forwardDist < 16.0 && Math.abs(lateralDist) < 5.8) {
                    if (forwardDist < nearestHazardDist) {
                        nearestHazardDist = forwardDist;
                        this.isEvadingPedestrian = true;

                        // 1. Slow down progressively as human gets closer
                        if (forwardDist < 6.0) {
                            desiredSpeed = 0.8; // Safe crawl (< 3 km/h)
                        } else if (forwardDist < 11.0) {
                            desiredSpeed = 2.2; // Evasion crawl (~8 km/h)
                        } else {
                            desiredSpeed = 4.2; // Pre-braking (~15 km/h)
                        }

                        // 2. Change direction: Steer away from human
                        const steerNudge = lateralDist >= 0 ? -0.36 : 0.36;
                        const proximityWeight = 1.0 - (forwardDist / 16.0);
                        evasionSteer = steerNudge * proximityWeight;
                    }
                }
            });

            // Combine waypoint steering with evasion steering
            this.ego.steerAngle = THREE.MathUtils.clamp(baseSteer + evasionSteer, -0.52, 0.52);
            this.ego.yaw -= this.ego.steerAngle * (this.ego.speed * 0.14) * dt * 3.6;

            // Smooth Acceleration / Braking
            if (this.ego.speed < desiredSpeed) {
                this.ego.speed = Math.min(desiredSpeed, this.ego.speed + this.ego.accel * 0.7 * dt);
            } else {
                this.ego.speed = Math.max(desiredSpeed, this.ego.speed - this.ego.decel * 1.1 * dt);
            }

            // Advance along heading
            this.ego.x -= Math.sin(this.ego.yaw) * this.ego.speed * dt;
            this.ego.z -= Math.cos(this.ego.yaw) * this.ego.speed * dt;
        }

        _updateManualDriving(dt) {
            // Acceleration / Reverse
            if (this.keys.forward) {
                this.ego.speed = Math.min(15.0, this.ego.speed + this.ego.accel * dt);
            } else if (this.keys.backward) {
                this.ego.speed = Math.max(-4.5, this.ego.speed - this.ego.accel * dt);
            } else if (this.keys.brake) {
                if (this.ego.speed > 0) {
                    this.ego.speed = Math.max(0, this.ego.speed - this.ego.decel * 1.6 * dt);
                } else {
                    this.ego.speed = Math.min(0, this.ego.speed + this.ego.decel * 1.6 * dt);
                }
            } else {
                // Rolling friction
                if (this.ego.speed > 0) {
                    this.ego.speed = Math.max(0, this.ego.speed - this.ego.friction * dt);
                } else if (this.ego.speed < 0) {
                    this.ego.speed = Math.min(0, this.ego.speed + this.ego.friction * dt);
                }
            }

            // NON-MIRRORED STEERING
            if (this.keys.right) {
                this.ego.steerAngle = Math.min(0.5, this.ego.steerAngle + this.ego.steerSpeed * dt);
            } else if (this.keys.left) {
                this.ego.steerAngle = Math.max(-0.5, this.ego.steerAngle - this.ego.steerSpeed * dt);
            } else {
                this.ego.steerAngle *= 0.8;
            }

            if (Math.abs(this.ego.speed) > 0.1) {
                // Non-mirrored steering in both forward and reverse!
                this.ego.yaw -= this.ego.steerAngle * (this.ego.speed * 0.15) * dt * 3.6;
            }

            this.ego.x -= Math.sin(this.ego.yaw) * this.ego.speed * dt;
            this.ego.z -= Math.cos(this.ego.yaw) * this.ego.speed * dt;
        }

        _updatePedestrians(dt) {
            this.pedestrians.forEach(ped => {
                const dx = ped.targetX - ped.startX;
                const dz = ped.targetZ - ped.startZ;
                const totalDist = Math.hypot(dx, dz);

                // Fluid human walking cadence (~1.8-2.0 steps/sec)
                ped.walkPhase += dt * ped.speed * 3.8;

                ped.x += (dx / totalDist) * ped.speed * ped.direction * dt;
                ped.z += (dz / totalDist) * ped.speed * ped.direction * dt;

                const curDist = Math.hypot(ped.x - ped.startX, ped.z - ped.startZ);
                if (curDist >= totalDist && ped.direction === 1) {
                    ped.direction = -1;
                } else if (curDist <= 0.25 && ped.direction === -1) {
                    ped.direction = 1;
                }

                // Smooth orientation: always faces forward along velocity vector
                const vx = (dx / totalDist) * ped.direction;
                const vz = (dz / totalDist) * ped.direction;
                ped.mesh.rotation.y = Math.atan2(-vx, -vz);

                // Biomechanical Walking Gait with knee and elbow articulation
                const thighAngle = Math.sin(ped.walkPhase) * 0.44;
                if (ped.legL) ped.legL.rotation.x = thighAngle;
                if (ped.legR) ped.legR.rotation.x = -thighAngle;

                // Knee flexion: calf bends backward when swinging forward to clear ground
                if (ped.calfL) ped.calfL.rotation.x = Math.max(0, -thighAngle * 0.85);
                if (ped.calfR) ped.calfR.rotation.x = Math.max(0, thighAngle * 0.85);

                // Arm counter-stride swing with natural outward angle
                if (ped.armL) {
                    ped.armL.rotation.x = -thighAngle * 0.65;
                    ped.armL.rotation.z = -0.05;
                }
                if (ped.armR) {
                    ped.armR.rotation.x = thighAngle * 0.65;
                    ped.armR.rotation.z = 0.05;
                }

                // Subtle organic torso rotation and sway
                if (ped.torso) {
                    ped.torso.rotation.y = Math.sin(ped.walkPhase) * 0.06;
                    ped.torso.rotation.z = Math.cos(ped.walkPhase) * 0.02;
                }

                // Natural vertical step bounce
                ped.mesh.position.y = Math.abs(Math.sin(ped.walkPhase)) * 0.025;
                ped.mesh.position.x = ped.x;
                ped.mesh.position.z = ped.z;
            });
        }

        _updateTrafficVehicles(dt) {
            if (!this.trafficVehicles) return;
            this.trafficVehicles.forEach(tv => {
                if (tv.speed > 0) {
                    tv.z += tv.dir * tv.speed * dt;
                    if (tv.dir < 0 && tv.z < -85) {
                        tv.z = 85;
                    } else if (tv.dir > 0 && tv.z > 85) {
                        tv.z = -85;
                    }
                    tv.mesh.position.set(tv.x, 0, tv.z);
                }
            });
        }

        _updateCameras() {
            // Forward vector for ego vehicle
            const fwdX = -Math.sin(this.ego.yaw);
            const fwdZ = -Math.cos(this.ego.yaw);

            // Windshield Sensor Camera - mounted cleanly at front bumper/hood level looking forward
            this.windshieldCam.position.set(
                this.ego.x + fwdX * 1.95,
                1.32,
                this.ego.z + fwdZ * 1.95
            );
            this.windshieldCam.rotation.y = this.ego.yaw;

            // User View Camera (Aspect-aware framing for Dual Split Screen vs Full Viewport)
            const aspect = this.width / Math.max(1, this.height);
            const isNarrowSplit = aspect < 1.3;

            if (this.cameraMode === 'chase') {
                const camDist = isNarrowSplit ? 16.5 : 9.5;
                const targetY = isNarrowSplit ? 6.4 : 4.6;
                const targetX = this.ego.x - fwdX * camDist;
                const targetZ = this.ego.z - fwdZ * camDist;

                this.userCamera.position.lerp(new THREE.Vector3(targetX, targetY, targetZ), 0.14);
                this.userCamera.lookAt(this.ego.x, 1.2, this.ego.z);
            } else if (this.cameraMode === 'cockpit') {
                // Driver eye perspective looking forward with zero interior clipping
                this.userCamera.position.set(
                    this.ego.x + fwdX * 1.95,
                    1.32,
                    this.ego.z + fwdZ * 1.95
                );
                this.userCamera.rotation.y = this.ego.yaw;
            } else if (this.cameraMode === 'overhead') {
                const camH = isNarrowSplit ? 52 : 38;
                this.userCamera.position.set(this.ego.x, camH, this.ego.z + (isNarrowSplit ? 24 : 18));
                this.userCamera.lookAt(this.ego.x, 0, this.ego.z);
            }
        }

        _updateDrivingHud() {
            const speedKmh = Math.abs(this.ego.speed * 3.6);
            const speedEl = document.getElementById('sim-speed-val');
            if (speedEl) speedEl.innerText = `${speedKmh.toFixed(0)} km/h`;

            const steerEl = document.getElementById('sim-steer-val');
            if (steerEl) steerEl.innerText = `${(this.ego.steerAngle * 180 / Math.PI).toFixed(1)}°`;

            const modeEl = document.getElementById('sim-mode-badge');
            if (modeEl) {
                if (this.collisionAlert) {
                    modeEl.innerText = 'COLLISION: IMPACT ARRESTED 💥';
                    modeEl.style.color = '#ef4444';
                    modeEl.style.borderColor = '#ef4444';
                    modeEl.classList.add('autopilot-active');
                } else if (this.autopilot) {
                    if (this.isEvadingPedestrian) {
                        modeEl.innerText = 'AUTOPILOT: EVADING HUMAN 🚨';
                        modeEl.style.color = '#f43f5e';
                        modeEl.style.borderColor = '#f43f5e';
                    } else {
                        modeEl.innerText = 'AUTOPILOT: CRUISE';
                        modeEl.style.color = '';
                        modeEl.style.borderColor = '';
                    }
                    modeEl.classList.add('autopilot-active');
                } else {
                    modeEl.innerText = 'MANUAL: TELEOP';
                    modeEl.style.color = '';
                    modeEl.style.borderColor = '';
                    modeEl.classList.remove('autopilot-active');
                }
            }

            const camEl = document.getElementById('sim-cam-btn-text');
            if (camEl) camEl.innerText = `CAM: ${this.cameraMode.toUpperCase()}`;
        }

        // ---------------------------------------------------------------------
        // 11. Public API: Windshield Render & Detected Objects Telemetry
        // ---------------------------------------------------------------------
        renderWindshield() {
            // Temporarily hide ego vehicle mesh, ground rings, and lidar sweep so NO vehicle hood or window pane ever obstructs camera foveation
            const egoMeshVis = this.ego.mesh ? this.ego.mesh.visible : true;
            const ringsVis = this.ego.lidarRingsGroup ? this.ego.lidarRingsGroup.visible : true;
            const sweepVis = this.ego.lidarSweep ? this.ego.lidarSweep.visible : true;

            if (this.ego.mesh) this.ego.mesh.visible = false;
            if (this.ego.lidarRingsGroup) this.ego.lidarRingsGroup.visible = false;
            if (this.ego.lidarSweep) this.ego.lidarSweep.visible = false;

            const curTarget = this.renderer.getRenderTarget();
            this.renderer.setRenderTarget(this.windshieldTarget);
            this.renderer.render(this.scene, this.windshieldCam);
            this.renderer.setRenderTarget(curTarget);

            if (this.ego.mesh) this.ego.mesh.visible = egoMeshVis;
            if (this.ego.lidarRingsGroup) this.ego.lidarRingsGroup.visible = ringsVis;
            if (this.ego.lidarSweep) this.ego.lidarSweep.visible = sweepVis;

            const w = this.windshieldTarget.width;
            const h = this.windshieldTarget.height;

            if (!this._pixelBuf || this._pixelBuf.length !== w * h * 4) {
                this._pixelBuf = new Uint8Array(w * h * 4);
                this._imgData = this.offscreenCtx.createImageData(w, h);
            }

            this.renderer.readRenderTargetPixels(this.windshieldTarget, 0, 0, w, h, this._pixelBuf);

            const data = this._imgData.data;
            for (let y = 0; y < h; y++) {
                const srcRow = (h - 1 - y) * w * 4;
                const dstRow = y * w * 4;
                data.set(this._pixelBuf.subarray(srcRow, srcRow + w * 4), dstRow);
            }
            this.offscreenCtx.putImageData(this._imgData, 0, 0);

            return this.offscreenCanvas;
        }

        getDetectedObjects() {
            const results = [];
            const actors = [
                ...this.pedestrians.map(p => ({ x: p.x, y: 0.95, z: p.z, label: p.label })),
                ...this.trafficVehicles.map(v => ({ x: v.x, y: 0.8, z: v.z, label: v.label })),
                ...this.potholes.map(ph => ({ x: ph.x, y: 0.05, z: ph.z, label: 'pothole', depth: ph.depth }))
            ];

            const egoPos = new THREE.Vector3(this.ego.x, 0, this.ego.z);
            const egoFwd = new THREE.Vector3(-Math.sin(this.ego.yaw), 0, -Math.cos(this.ego.yaw));

            actors.forEach(act => {
                const actPos = new THREE.Vector3(act.x, act.y, act.z);
                const rel = new THREE.Vector3().subVectors(actPos, egoPos);
                const dist = rel.length();

                if (dist > 85) return;

                const dot = rel.dot(egoFwd);
                if (dot <= 0.25) return; // behind vehicle

                const pProj = actPos.clone().project(this.windshieldCam);
                if (pProj.z < 0 || pProj.z > 1) return;
                if (pProj.x < -1.15 || pProj.x > 1.15 || pProj.y < -1.15 || pProj.y > 1.15) return;

                const screenX = (pProj.x + 1) / 2;
                const screenY = (-pProj.y + 1) / 2;

                // Foveated 3-Ring Semantic Hierarchy
                let rangeBand = 'far';
                let ringId = 2;
                let resolution = '50cm';
                if (dist <= 10.0) {
                    rangeBand = 'near';
                    ringId = 0;
                    resolution = '5cm';
                } else if (dist <= 30.0) {
                    rangeBand = 'mid';
                    ringId = 1;
                    resolution = '15cm';
                }

                let boxW, boxH;
                if (act.label === 'pothole') {
                    boxW = THREE.MathUtils.clamp(140 / Math.max(1, dist), 28, 110);
                    boxH = THREE.MathUtils.clamp(80 / Math.max(1, dist), 18, 60);
                } else if (act.label === 'person') {
                    boxW = THREE.MathUtils.clamp(160 / Math.max(1, dist), 26, 180);
                    boxH = THREE.MathUtils.clamp(320 / Math.max(1, dist), 50, 320);
                } else {
                    boxW = THREE.MathUtils.clamp(200 / Math.max(1, dist), 34, 220);
                    boxH = THREE.MathUtils.clamp(240 / Math.max(1, dist), 38, 220);
                }

                results.push({
                    label: act.label,
                    depth: act.depth || null,
                    dist: dist,
                    ringId: ringId,
                    resolution: resolution,
                    bearing01: screenX,
                    rangeBand: rangeBand,
                    confidence: THREE.MathUtils.clamp(0.98 - (dist / 140), 0.82, 0.99),
                    box: {
                        x: screenX,
                        y: screenY,
                        w: boxW,
                        h: boxH
                    }
                });
            });

            // Detect roadside trees ahead in windshield camera frustum (sample at most 2 closest trees to keep forward road view crystal clear!)
            if (this.trees && this.trees.length) {
                const candidates = [];
                this.trees.forEach(tree => {
                    const treePos = tree.pos;
                    if (!treePos) return;
                    const rel = new THREE.Vector3().subVectors(treePos, egoPos);
                    const dist = rel.length();
                    if (dist < 4.0 || dist > 24.0) return;
                    const dot = rel.dot(egoFwd);
                    if (dot <= 0.3) return; // In front of vehicle

                    const pProj = treePos.clone().project(this.windshieldCam);
                    if (pProj.z < 0 || pProj.z > 1) return;
                    if (pProj.x < -1.15 || pProj.x > 1.15 || pProj.y < -1.15 || pProj.y > 1.15) return;

                    candidates.push({ tree, dist, pProj });
                });

                candidates.sort((a, b) => a.dist - b.dist);
                candidates.slice(0, 2).forEach(({ tree, dist, pProj }) => {
                    const screenX = (pProj.x + 1) / 2;
                    const screenY = (-pProj.y + 1) / 2;

                    const boxW = THREE.MathUtils.clamp(160 / Math.max(1, dist), 28, 90);
                    const boxH = THREE.MathUtils.clamp(200 / Math.max(1, dist), 36, 120);

                    results.push({
                        label: 'tree',
                        dist: dist,
                        ringId: dist <= 10 ? 0 : 1,
                        resolution: dist <= 10 ? '5cm' : '15cm',
                        bearing01: screenX,
                        rangeBand: dist <= 10 ? 'near' : 'mid',
                        confidence: 0.98,
                        masked: true,
                        maskType: 'Static Scene / Vegetation Caching',
                        box: {
                            x: screenX,
                            y: screenY,
                            w: boxW,
                            h: boxH
                        }
                    });
                });
            }

            return results;
        }

        resize(width, height) {
            this.width = width;
            this.height = height;
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            this.renderer.setPixelRatio(dpr);
            this.renderer.setSize(width, height, false);
            const aspect = width / height;
            this.userCamera.aspect = aspect;

            // Adaptive FOV: if aspect is narrow (e.g. dual split screen < 1.3), expand vertical FOV
            // to keep the entire highway, sidewalks, roadside trees, and city skyline fully visible!
            if (aspect < 1.3) {
                this.userCamera.fov = Math.min(82, 60 + (1.3 - aspect) * 35);
            } else {
                this.userCamera.fov = 60;
            }
            this.userCamera.updateProjectionMatrix();
        }

        render() {
            const isCockpit = this.cameraMode === 'cockpit';
            if (isCockpit && this.ego.mesh) this.ego.mesh.visible = false;

            this.renderer.setRenderTarget(null);
            this.renderer.render(this.scene, this.userCamera);

            if (isCockpit && this.ego.mesh) this.ego.mesh.visible = true;
        }
    }

    return ThreeSimulator;
}));
