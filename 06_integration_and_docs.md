# 06 — Integration, Docs & Demo Prep (All Members · Days 24-25)

Use after 00_master_context.md, after all other phases are done.

---

```
<task>
Do a final end-to-end integration pass and produce the documentation/demo
artifacts we need to present the project.
</task>

<requirements>
1. Write an end-to-end run script (`run_pipeline.sh` or `main.py`) that
   chains: dataset load -> ground seg -> clustering/classification ->
   grid engine -> dashboard, on our golden test scene, from a single command.
2. Write a top-level README.md covering: problem statement (with the simple
   4th-grader analogy), architecture diagram, tech stack table, how to
   install dependencies, how to run the pipeline, how to run benchmarks,
   and the headline results (memory reduction %, FPS, mIoU).
3. Write a short CONTRIBUTING.md or team-notes.md documenting which member
   owns which module, for future reference.
4. List every known limitation/tradeoff honestly (e.g. accuracy tradeoff from
   skipping deep learning, CPU-only performance ceiling) — this should be
   framed as intentional engineering tradeoffs, not bugs.
5. Build the hackathon room demo — no live sensor/vehicle in the room, so use
   a recorded-playback strategy:
   a. RECORDED FALLBACK (mandatory, build first): pre-run the full pipeline
      once on the golden test scene, save every frame's grid state, dashboard
      render, and metrics (CSV from Phase 4 benchmarks) to disk. This is a
      zero-risk replay — must work standalone with no live compute, no
      network, no dependencies beyond playing back saved files.
   b. LIVE PLAYBACK (primary demo mode): run the actual pipeline live on a
      laptop with a Streamlit web interface (`localhost:8501`), feeding from the pre-loaded SemanticKITTI clip instead
      of a real sensor. Judges see real inference with Streamlit `st.metric()` cards (FPS counter, memory graph
      updating live) — clean, low-risk, zero web/network dependency. This is what you present with unless it fails, then
      fall back to (a) immediately without breaking narration.
   c. Optional bonus (only if Phase 4 has slack time): small physical rig
      (cheap LiDAR + webcam on an RC car or handheld) driven live around the
      room. Treat as a wow-moment add-on at the END of the demo, never as the
      main proof — must not replace (a) or (b).
   Test (a) and (b) run start-to-finish at least once the day before
   presenting, on the actual laptop that will be used, not a different dev
   machine.
6. Write a comparison table/chart as part of the demo script: your lite
   pipeline's memory/latency numbers side-by-side with the uniform-grid
   baseline from Phase 4 benchmarks — this is the single most convincing
   visual for judges, make it large and simple to read from across a room.
</requirements>

<deliverables>
- run_pipeline.sh (or main.py)
- README.md
- team-notes.md
- demo_script.md — must explicitly cover: analogy opener, live-playback demo
  flow (5b), the recorded-fallback trigger/switchover plan (5a), the
  baseline-comparison visual (6), and where the optional physical-rig
  bonus (5c) fits if ready
- recorded_demo/ — the saved frames/metrics/dashboard state for fallback (5a)
</deliverables>

<output_format>
Full content for each file, in order.
</output_format>
```
