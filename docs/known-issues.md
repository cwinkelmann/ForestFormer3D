# Known issues (out of scope for the 2026-09-22 fixes)

Findings from the 2026-09-22 review that were deliberately left alone in Phase 1.
Each is inherited from OneFormer3D or only matters once the benchmark shows it does.

1. **Aux-layer dice scaling** (`oneformer3d/instance_criterion.py`, `get_layer_loss`): the
   intermediate decoder layers weight the dice term the same way OneFormer3D does; changing it
   changes the trained model, so it stays until a training benchmark motivates it.
2. **`gt_labels_3d` first-point label** (`oneformer3d/transforms_3d.py`, `PointInstClassMapping_`):
   the class of an instance is the semantic label of its first point, not the majority. Harmless
   for ForAINetV2 (every tree point is wood or leaf, both "thing"), wrong for datasets with mixed
   instances.
3. **Nondeterministic `index_copy_` ties** (`ForAINetV2OneFormer3D_XAwarequery.grid_sample`):
   when several points share a voxel the representative index is whichever `index_copy_` writes
   last, which is not deterministic on CUDA. Affects only which original index a voxel carries,
   not the averaged coordinates.
4. ~~**Python-loop `save_ply_withscore`**~~ **Fixed 2026-09-23.** It built the vertex array with a
   per-point tuple comprehension and wrote ASCII. The 2026-09-23 inference profile measured
   5.6-10.3 s per 100 m tile, so it was rewritten as a structured array in `oneformer3d/ply_io.py`
   and is now written binary little-endian.
5. **Semantic mIoU semantics in `UnifiedSegMetric`**: `mIoU` averages the 1-based classes that
   have GT points and counts `-1` (no vote) predictions as class 0 "unclassified"; it is not the
   same number `tools/final_eval.py` prints. Both are reported; neither was changed.

## Deferred from the Phase 1 reviews

6. **`save_bluepoints` has no call site.** `ForAINetV2OneFormer3D_XAwarequery._predict_full_plot`
   (and the older `predict()`) always writes results with `save_ply_withscore`. The bluepoint
   second pass (`tools/inference_bluepoint.sh`) still expects the remaining, unsegmented points
   to be written with `save_bluepoints` so they can be re-fed into `test_data/`; today nothing in
   `predict()` calls it. Wiring it in is a code change, not a config/env flag, and is out of scope
   here.
7. **`torch.cdist` per tile is `chunk x |pc3|`.** In `_predict_full_plot`, the nearest-neighbour
   lookup batches over `pc1` in slices of `self.chunk` against the *whole* `pc3` tile
   (`torch.cdist(pc1[s:s+self.chunk, :3], pc3[:, :3])`), so memory scales with tile size, not just
   `chunk`. The config's top-level `chunk = 20_000` (`configs/oneformer3d_qs_radius16_qp300_2many.py`)
   is never threaded into `model=dict(...)`, so it has no effect; the model always uses its
   constructor default (`chunk=20_000`, same value today, but silently so).
8. **`grid_size`/`max_points` hardcoded in `_predict_full_plot`.** Both are local constants
   (`grid_size = 0.2`, `max_points = 640_000`) inside the method, not config-driven or
   constructor arguments, unlike `chunk`. Lowering them for a small-GPU OOM workaround means
   editing `oneformer3d/oneformer3d.py`.
9. **`PointSample_` without replacement permutes N points per sample above 640k.** `PointSample_`
   (`oneformer3d/transforms_3d.py`) calls `np.random.choice(len(points), min(num_samples,
   len(points)), replace=False)`; when `len(points) <= num_samples` this still permutes every
   point (replace=False with `num_samples >= len(points)` returns a full permutation), so a
   scene under the cap gets randomly reordered even though nothing is actually subsampled.
10. **Converter `torch.load` without `weights_only`.** `tools/fix_spconv_checkpoint.py` calls
    `torch.load(args.in_path, map_location='cpu')` with no `weights_only=True`; fine for
    checkpoints from a trusted training run, but it will execute arbitrary pickled objects if
    pointed at an untrusted file.
11. **Module-scoped model fixture in the GPU smoke test.** `tests/gpu/test_smoke.py` builds the
    model once per module (`@pytest.fixture(scope="module")`) and shares it across tests in that
    file; a test that mutates `model.test_cfg` (several do) can leak state into a later test in
    the same module if execution order changes.
12. **`test_smoke` uses negative thresholds because untrained scores are negative logits.**
    `tests/gpu/test_smoke.py` and `tests/gpu/test_predict_full_plot.py` set
    `model.test_cfg["inst_score_thr"] = -1e6` and `model.test_cfg["score_th"] = -1e6` to force
    predictions through on a randomly initialised model, whose raw logits are negative; this is a
    test-only workaround and says nothing about the thresholds used for a trained checkpoint.

13. **`looks_raw` misclassifies a ground-free, normalized crop with no `-1`.**
    `oneformer3d/labels.py:79-94`: when a scene has no ground points (`semantic == 0` never true),
    `looks_raw` falls back to `not (instance < 0).any()` — normalized GT with no unassigned points
    (no `-1` anywhere) then reads as `raw=True`, and `normalize_instance_gt` wipes GT instance id
    `0`, silently dropping one whole tree. Real ForAINetV2 plots always contain ground points, so
    this only matters for a synthetic/edge-case crop; `tests/test_final_eval.py` documents the same
    hazard in a comment on the file-B case of
    `test_binary_semantic_totals_are_summed_over_files`.
14. **Result PLYs can carry `semantic_pred == -1` for points no tile voted on.**
    `SemanticVotes.resolve()` (`oneformer3d/tiling.py`) returns `-1` for points that were sampled
    out of every overlapping tile; `save_ply_withscore` (`oneformer3d/oneformer3d.py:1589,2894`)
    writes it through as `i4`, which round-trips fine in a `.ply`. Phase 3's LAS writer is
    specified as writing `semantic` as **uint8** (0 ground, 1 wood, 2 leaf); casting `-1` straight
    to `uint8` silently becomes `255`. Phase 3 must map `-1` to an explicit nodata value (e.g.
    `255`) rather than rely on the cast.
15. **Per-tile `torch.cuda.empty_cache()` and an unseeded `sample_region` in `_predict_full_plot`.**
    `oneformer3d/oneformer3d.py:2206` calls `torch.cuda.empty_cache()` once per tile inside the
    cylinder loop — on a 100 m plot with `radius=16`, `step=4` that is roughly 676 device syncs
    plus allocator teardown inside the loop Phase 2 will report as wall-clock; not a regression
    (the old code did the same), but worth hoisting out of the loop or throttled to every N tiles
    before trusting a timing benchmark. Separately, `sample_region(pc2, pc2_indices, max_points)`
    at `oneformer3d/oneformer3d.py:2167` is called without a `generator`, so a tile that exceeds
    `max_points = 640_000` after voxel downsampling is subsampled using the global torch RNG —
    nondeterministic run to run. Rare in practice, but passing a seeded CPU generator would make a
    re-run of `run_release_eval.sh` reproducible bit-for-bit.
16. **`instance_scores` means a different shape in full-plot vs. crop-mode `predict()`.**
    `_predict_full_plot` (`oneformer3d/oneformer3d.py:2234`) sets `instance_scores` to a
    per-*point* array (`score_np`, length `n_total`), while `_predict_crop`/`predict_by_feat` (e.g.
    `oneformer3d/oneformer3d.py:104,917,3384,3873`) sets it to a per-*instance* array. Nothing in
    the active config reads the key across both paths today (`UnifiedSegMetric` ignores it), so
    this is a naming trap rather than a live bug — but any new code touching `instance_scores` must
    check which `predict()` branch produced it.

## Operational notes discovered while fixing

- `data/ForAINetV2/batch_load_ForAINetV2_data.py` skips scans whose `_vert.npy` already exists.
  After changing the loader (ground and unlabeled vegetation are now instance `-1`) delete
  `data/ForAINetV2/forainetv2_instance_data/` before regenerating.
- `tools/test.py` used `torch.load` without importing torch before Phase 1; the in-memory
  permutation block that contained it is gone.
- **Degenerate cylinder regions no longer abort a batch.** On near-empty 100 m ALS sub-tiles
  (water, open ground, a few stray returns) a cylinder region can hold so few, so far-apart
  voxels that they all land on the last index of an odd spatial dimension inside the spconv
  UNet. Each downsampling level is `(S - 2) // 2 + 1`, so for odd `S` that last index has no
  output cell, every point is dropped and spconv raises
  `ValueError: Your points vanished here` (seen with `spatial_shape=[17, 16, 16]` from a
  35-voxel region and `[19, 16, 87]` from a 480-voxel region with a 140 m z outlier). Because
  one `tools/test.py` process serves every sub-tile of a km tile, that exception used to kill
  the whole batch. `_predict_full_plot` now pre-filters each region with
  `tiling.degenerate_region_reason` (fewer than `model.test_cfg.min_region_points`, default 64,
  points after the 0.2 m grid sample, or a voxel span that is a point or a line) and, as a
  backstop, catches `ValueError` out of `collate`/`extract_feat` for a single region. Only
  `ValueError` is caught — an OOM `RuntimeError` still propagates. A skipped region simply casts
  no vote; its points fall back to whatever the ~44 overlapping regions say, or to nodata
  (semantic `-1`, instance `-1`). A scan where *no* region survives still gets its result PLY,
  all points unlabelled, so the per-tile files stay complete for the merge.
  To spot this in a log, grep for `degenerate region`, `spconv rejected region` and the
  per-scan summary `N of M cylinder regions were degenerate and skipped`; only the first three
  pre-filter skips are spelled out, the summary carries the total.
