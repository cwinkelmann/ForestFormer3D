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
  per-scan summary `of M cylinder regions U were segmented, D were degenerate, R were rejected
  by spconv, E were empty` (the four counts reconcile to `M`). At most three lines of each kind
  are spelled out; the summary carries the totals. Measured margin on a healthy tile: the
  *smallest* of the 625 regions of the r12 100 m tile holds 3760 voxels, 59x the threshold, and
  a rerun of that tile skipped zero regions and matched the old output to within the known
  run-to-run noise.
  One behaviour change to know about: a skipped region casts **no semantic vote either**, so a
  genuinely isolated cluster of fewer than 64 voxels that used to get a semantic label — and,
  with `npoint_thr=10`, possibly a tiny instance — now comes out `semantic -1, instance -1`
  (nodata in the LAS/GeoTIFF path). On real tiles that is confined to stray returns over water
  and bare ground, which is the intent; if those points ever need a label, cast a ground vote
  (`votes.add_binary(pc1_indices, torch.zeros_like(pc1_indices, dtype=torch.bool))`) in the
  pre-filter branch instead of skipping outright.

## The torch 2.0.1 pin stays (decided 2026-10-20)

The question comes up whenever someone looks at the stack: torch 2.0.1 / mmcv 2.0.1 /
mmdet 3.0.0 are from 2023, so would a newer torch be faster? The pin is kept deliberately.

**MinkowskiEngine is the blocker.** The active model's `collate`
(`oneformer3d/oneformer3d.py:1868-1884`, `ForAINetV2OneFormer3D_XAwarequery`) does not
merely call a helper; it voxelises through `ME.TensorField(...).sparse()` and takes the
point->voxel `inverse_mapping` from it, which the whole predict path depends on.
MinkowskiEngine's last release is 0.5.4 (2021) and does not build against torch >= 2.1
without patching. `mmcv 2.0.1`'s prebuilt wheels also stop at torch 2.1; beyond that its
ops compile from source.

It is a blocker, but not an immovable one: the two subsections below measured both halves of
the problem, and found ME replaceable by ~30 lines of pure torch at no cost in correctness
or GPU speed, and mmcv's compiled ops never called at all. So the reason the pin stays is
the next paragraph, not an impossibility -- if a future torch is ever *needed* (a new GPU
architecture, a dependency that demands it), the path exists and is written down. `docs/torch-upgrade-path.md`
has the three routes, what each one costs and the trigger conditions.

**The expected gain is small.** The 2.90x already measured came from scheduling
(`region_batch`, `batched_decoder`), not from kernels, and the profile addendum in
`docs/benchmarks/2026-09-23-inference-profile.md` shows what is left spread over backbone,
sampling, collate, nearest-neighbour and mask post-processing at 7-17 % each -- no single
hotspot for faster kernels to attack. `torch.compile` cannot reach the sparse convolutions,
which are custom CUDA in spconv and MinkowskiEngine.

### Replacing MinkowskiEngine: equivalent, and faster where it runs (2026-10-20)

ME is used for voxelisation and the point->voxel inverse mapping only; the convolutions are
spconv. Two drop-in replacements were implemented and checked against ME inside the image
on a batch of random point sets, correctness judged the way the model uses it
(`features[inverse_mapping]` per point). Both are **exactly equivalent** -- same voxel
count, same coordinates, same `spatial_shape`, per-point features agreeing to 7.6e-06,
which is float summation order rather than disagreement. On CPU both are slower, which is
what the first version of this section concluded from; **on the GPU, where this code
actually runs, the torch replacement wins**:

| CPU, in the image | 1x193k (a sub-tile) | 8x80k (region_batch=8) | 1x640k (max_points) |
|---|---|---|---|
| MinkowskiEngine | 547 ms | 1651 ms | 633 ms |
| packed-int64 `torch.unique` + `scatter_mean` | 650 ms | 2362 ms | 1571 ms |
| spconv `PointToVoxel.generate_voxel_with_id` | 603 ms | 3801 ms | 1450 ms |

| **CUDA (RTX 4080 SUPER), in the image** | 1x193k | 8x80k (region_batch=8) | 1x640k |
|---|---|---|---|
| MinkowskiEngine | 1.7 ms | 2.7 ms | 2.1 ms |
| packed-int64 `torch.unique` + `scatter_mean` | **0.4 ms** | **1.0 ms** | **0.7 ms** |
| spconv `PointToVoxel.generate_voxel_with_id` | 3.5 ms | 12.7 ms | 10.8 ms |

So the performance argument for keeping ME does not survive contact with the device: the
pure-torch groupby is 2.5-4x faster there, and spconv's hash map 4-6x slower. Do not read
the torch column as a speed opportunity either -- 2.7 ms per batched pass, ~8 passes per
sub-tile, 100 sub-tiles is **about 1.5 s of a 5,220 s km tile**. The swap is
performance-neutral in the run; what it buys is the removal of a build dependency, not
time. CPU numbers are kept above because they are what a Mac test run would measure, and
they are 100-1000x the GPU ones for every implementation -- never benchmark this on CPU
and conclude anything about the run.

spconv loses badly at the batch shape because `PointToVoxel` is per-sample, so
`region_batch=8` builds eight hash maps. `torch.unique(coords, dim=0)` is worse again --
it sorts 640k x 4 rows lexicographically; packing the batch index and the three coordinates
into one int64 is what makes the torch variant competitive at all.

The maintained fork (`alpsaur/MinkowskiEngine`) does not help: its wheels are built for
torch 2.9.1+cu128 on glibc >= 2.38, and a source build needs torch >= 2.7. This image is
Ubuntu 22.04 (glibc 2.35) with torch 2.0.1. `torchsparse` 2.0.0b0 does build against our
torch, but it replaces the spconv *backbone* rather than the groupby, which would cost
checkpoint compatibility for a claimed 1.25x on part of the time.

Net: ME is replaceable by the packed-int64 groupby at no cost in correctness or speed, and
that is the move if the stack is ever unpinned (it is the only piece with no wheel for
torch >= 2.1 and no maintained upstream). It is not worth doing on its own, because nothing
downstream of it changes while torch stays at 2.0.1.

### mmcv's compiled half is replaceable; its Python half is the stack (2026-10-20)

Our own code imports exactly one thing from mmcv -- `BaseTransform` in
`oneformer3d/transforms_3d.py`, a pure-Python base class. But `import oneformer3d` pulls
**129** mmcv submodules including all of `mmcv.ops` and the compiled `mmcv._ext`, eagerly,
via mmdet3d; and mmdet3d 1.1.0 declares `mmcv >= 2.0.0rc4, < 2.1.0`. That version box, not
MinkowskiEngine, is what actually pins the stack.

An inference-only path that bypassed mmdet3d was considered and rejected on the user's
reasoning: **training still needs mmcv** (`tools/train.py`, `dist_train.sh`,
`UnifiedSegMetric` validation, the registry-driven configs, the planned WINMOL
retraining), so bypassing it for inference would mean two model-construction paths whose
drift stays silent until a retrained checkpoint fails to load. One old box beats two paths.

That settled the *bypass*. The follow-up question -- whether mmcv could be replaced rather
than bypassed -- was then measured, and the compiled half turns out not to be load-bearing
at all:

- Of the **71** mmcv symbols that `import oneformer3d` binds, **18** come from modules that
  reference `_ext`. All but two of those arrive through model-zoo modules this config never
  builds; the two live ones are `mmcv.utils.ext_loader` (via
  `mmdet3d.models.data_preprocessors.voxelize`, whose `voxel` flag is False in our config)
  and `mmcv.ops.box_iou_rotated` (via `BaseInstance3DBoxes`, which we construct but never
  ask for a rotated IoU).
- With `sys.modules["mmcv._ext"]` seeded by a ~20-line `sitecustomize.py` stub whose every
  attribute raises `RuntimeError("MMCV_EXT_CALLED: ...")`, the model builds
  (`ForAINetV2OneFormer3D_XAwarequery`, 17.50 M parameters) and **`docker/smoke.sh` passes
  unchanged** -- one `loss()` step and one full-plot `predict()`, i.e. both paths, 2 passed
  in 6.12 s against 6.90 s for the control. No op was ever called.

So the ~150 CUDA ops in `mmcv._ext` are dead weight for this project, and the real
dependency is mmengine's registry plus mmdet3d's pure-Python classes. Nothing is changed on
the strength of this: the stub lives in the session scratchpad, not the image, because
shipping a stub that lies about a C extension is a trap for the next person who enables
`voxel=True` or a rotated box. What it establishes is the *order of the blockers* -- mmcv is
soft, mmdet3d's `mmcv < 2.1.0` box is a declaration rather than a kernel, and
MinkowskiEngine (above) is the only piece with no path to torch >= 2.1 at all.

Two cheaper leads were noted at the same time and are NOT done:

- torch logs `UserWarning: Converting mask without torch.bool dtype to bool; this will
  negatively affect performance` from `torch._native_multi_head_attention`
  (`attention.cpp:150`) during every inference run. `query_decoder.py:281` already casts
  with `.bool()`, so it is a different call site. The 2026-10-20 smoke run narrows it: the
  warning is raised from `torch/nn/modules/activation.py:1160`, i.e. an
  `nn.MultiheadAttention.forward` fast path, NOT the hand-written
  `scaled_dot_product_attention` cross-attention in `forward_padded`. It also reproduces in
  `docker/smoke.sh` in 7 s, so `-W error::UserWarning` on that gate is the cheap way to get
  the stack. This is a slow attention path on the version already installed.
- The `region_batch` sweep on the T14's RTX 4080 SUPER peaked at 7.3 GB of 15.6 GiB at the
  default 8 and was only measured to 8 (`.claude/skills/ff3d-carrot/SKILL.md`), so 16 is
  worth a measurement there even though it was flat from 8 to 32 on an H100.
