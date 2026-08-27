# 04 — Visualization Dashboard (Member C · Phases 1-3)

Use after 00_master_context.md.

---

```
<task>
Build the real-time visualization dashboard for our foveated 2.5D grid,
progressively: Phase 1 = raw point cloud viewer, Phase 2 = placeholder grid
render + HUD stub, Phase 3 = live grid render wired to Member B's
getGridSnapshot() API.
</task>

<requirements_phase1>
- Streamlit web frontend (or Matplotlib fallback) viewer that loads and displays
  raw point cloud frames with interactive controls at `localhost:8501`.
</requirements_phase1>

<requirements_phase2>
- Streamlit top-down 2D render of a 3-ring grid (drawing ring boundaries and a
  checkerboard at each resolution) to validate the visual layout on web UI before real data is wired in.
- HUD overlay stub displaying `st.metric()` cards showing FPS, latency (ms), and RAM memory (MB) — hardcoded/fake
  values for now, real values wired in Phase 3.
</requirements_phase2>

<requirements_phase3>
- Wire the Streamlit dashboard to Member B's `getGridSnapshot(ring_level)` API (via
  In-Process Zero-Copy NumPy Array views / direct binding).
- Color-coding: green = drivable road, red/orange = dynamic object
  (car/pedestrian), gray = static obstacle (wall/pole), yellow/amber = vertical poles.
- Fine grid lines near ego vehicle (Level 0), progressively coarser grid
  lines further out (Levels 1-2), so the foveated effect is visually obvious on the web canvas.
- Live Streamlit HUD: `st.metric()` cards displaying real FPS, real per-frame latency, real process/MLRB memory footprint,
  and % memory reduction vs. uniform-grid baseline.
- Target 10+ FPS render on CPU-only hardware running locally on `localhost:8501`.
</requirements_phase3>

<deliverables>
- viewer_phase1.py
- dashboard_phase2.py
- dashboard_phase3.py (final live version)
- A short note on which IPC method you chose to talk to the C++ grid engine,
  and why
</deliverables>

<output_format>
Full code per phase file. For Phase 3, include a short architecture diagram
in ASCII showing how Python dashboard <-> C++ grid engine <-> data flow.
</output_format>
```
