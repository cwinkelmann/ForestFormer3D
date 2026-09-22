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
4. **Python-loop `save_ply_withscore`** (`oneformer3d/oneformer3d.py`): builds the vertex array
   with a per-point tuple comprehension. Rewrite with structured numpy only if it shows up in the
   Phase 2 benchmark timings.
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

## Operational notes discovered while fixing

- `data/ForAINetV2/batch_load_ForAINetV2_data.py` skips scans whose `_vert.npy` already exists.
  After changing the loader (ground and unlabeled vegetation are now instance `-1`) delete
  `data/ForAINetV2/forainetv2_instance_data/` before regenerating.
- `tools/test.py` used `torch.load` without importing torch before Phase 1; the in-memory
  permutation block that contained it is gone.
