# 02 — Obstacle Clustering & Classification (Member A · Phase 2)

Use after 00_master_context.md. Assumes 01_ground_segmentation.md's
`segment_ground()` already exists.

---

```
<task>
Take the non-ground points from segment_ground() and turn them into
classified obstacle clusters (static obstacle / dynamic object / pole /
wall etc.) using classical clustering + a lightweight classifier — no
deep 3D network.
</task>

<requirements>
1. Implement Euclidean clustering (via PCL bindings) or DBSCAN (scikit-learn)
   on non-ground points to group them into candidate objects.
2. Extract hand-crafted features per cluster: bounding box height/width/length,
   point density, point count, z-variance, aspect ratio.
3. Train a Random Forest classifier (scikit-learn) on SemanticKITTI labels
   mapped to our 4 target classes: drivable ground (excluded, already
   handled), static obstacle, dynamic object, pole/wall.
4. Provide a clean `classify_clusters(points, ground_mask) -> List[{cluster_id,
   class, confidence, points}]` function — this is what Member B's grid
   engine will consume.
5. Include an evaluation script that reports per-class precision/recall and
   overall mIoU-style accuracy against SemanticKITTI validation labels.
</requirements>

<deliverables>
- `clustering.py` — Euclidean/DBSCAN clustering wrapper
- `feature_extraction.py` — per-cluster feature computation
- `train_classifier.py` — trains and saves the Random Forest model
- `classify_clusters.py` — inference-time wrapper exposing the function above
- `evaluate.py` — accuracy/precision/recall reporting script
</deliverables>

<constraints>
- Keep the whole stage CPU-only and fast enough for near-real-time use
  (target: under ~50ms per frame on a laptop CPU for clustering+classification
  combined — flag it clearly if that's not achievable with current approach).
- If accuracy is far below deep-learning baselines, note the tradeoff
  explicitly rather than silently underperforming.
- Do NOT use charlesq34/pointnet (original PointNet) anywhere in this
  pipeline — it's TensorFlow 1.2-era and its own README says GPU access is
  "highly recommended," which directly breaks our CPU-first/lite constraint.
  If a DL comparison baseline is wanted later, use Open3D-ML's pretrained
  RandLA-Net or KPConv checkpoints on SemanticKITTI instead (Phase 4, GPU
  optional single comparison run only, not part of the lite pipeline).
</constraints>

<output_format>
Full code per file, then a short table of expected accuracy vs. known
Cylinder3D/MinkowskiNet baselines from literature so we can document the
tradeoff.
</output_format>
```
