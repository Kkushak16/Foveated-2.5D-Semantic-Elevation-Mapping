# 01 — Ground Segmentation & Dataset Pipeline (Member A · Phase 1)

Use after pasting 00_master_context.md.

---

```
<task>
Build the dataset loading + ground segmentation stage of our pipeline.
</task>

<requirements>
1. Write a Python loader for SemanticKITTI (.bin point cloud files + .label files)
   that yields (points[N,4] xyzi, labels[N]) per frame, batched by sequence.
   Option: isl-org/Open3D-ML has a built-in SemanticKITTI dataset loader and
   visualizer — evaluate reusing it instead of writing a loader from scratch,
   since it's already tested against this exact dataset format.
2. Integrate Patchwork++ (or a RANSAC plane-fit fallback if Patchwork++ isn't
   installed) to segment each frame into ground vs. non-ground points.
3. Add a visual + numeric validation step: overlay ground/non-ground coloring
   on a sample frame using Open3D, and print % of points classified as ground.
4. Keep everything CPU-runnable. No CUDA dependency.
</requirements>

<deliverables>
- `dataset_loader.py` — SemanticKITTI loader
- `ground_segmentation.py` — Patchwork++/RANSAC wrapper with a clean
  `segment_ground(points) -> (ground_mask)` function
- `validate_ground_seg.py` — quick visual + numeric sanity check script
- A short README section explaining how to run each script and expected output
</deliverables>

<constraints>
- Function signatures should be stable — Member B's grid engine and Member C's
  dashboard will consume `segment_ground()`'s output directly, so don't change
  its interface without flagging it.
- Handle missing/corrupt frames gracefully (skip + log, don't crash).
</constraints>

<output_format>
Give me the full code for each file, then a 3-5 line usage example.
</output_format>
```
