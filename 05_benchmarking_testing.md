# 05 — Benchmarking & Testing (All Members · Phase 4)

Use after 00_master_context.md. Assumes Phases 1-3 are complete and all
components exist (ground seg, clustering/classification, grid engine, dashboard).

---

```
<task>
Build the benchmarking + test suite that validates the whole pipeline and
produces the metrics we need for the final report/demo.
</task>

<requirements>
1. Baseline comparison: implement (or point to) a naive uniform-resolution
   3D voxel grid (5cm cells across the full 0-100m range) as the comparison
   baseline.
2. Memory benchmark: measure RAM footprint (MB/frame) of our foveated grid
   vs. the uniform baseline, across a fixed "golden test scene" (one short
   KITTI sequence used consistently for all measurements).
3. Latency/FPS benchmark: log per-stage timing (ground seg, clustering,
   classification, grid insertion, render) and total per-frame time; target
   10+ FPS on CPU-only.
4. Accuracy benchmark: mIoU-style evaluation of our classifier vs.
   SemanticKITTI ground truth, plus a comparison table against published
   Cylinder3D/MinkowskiNet numbers (for context, not apples-to-apples).
5. Robustness tests:
   - Overhang test (bridge/tunnel synthetic scene) — verify correct flagging
   - Sparse far-cell test (object at 90m) — verify Kalman/EMA convergence
     over ~10 frames
   - Stress test — dense urban frame vs. sparse highway frame, check for
     memory spikes/crashes
6. Output all results to CSV (one row per frame/run) so charts can be
   generated for the final report without re-running anything.
</requirements>

<deliverables>
- benchmark_memory.py
- benchmark_latency.py
- benchmark_accuracy.py
- test_robustness.py (overhang, sparse-cell, stress tests)
- results/ directory with CSV outputs
- A short summary script that prints the headline numbers (e.g. "72% memory
  reduction vs. uniform grid, 14 FPS average, mIoU 0.78")
</deliverables>

<constraints>
- Reuse the exact same "golden test scene" across every benchmark script —
  do not let it drift between memory/latency/accuracy runs.
- Every benchmark script must be re-runnable standalone (no hidden state from
  a previous run required).
</constraints>

<output_format>
Full code per file, then the exact terminal command to run the full benchmark
suite end-to-end in one shot.
</output_format>
```
