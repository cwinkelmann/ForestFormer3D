# ALS-density evaluation of the released checkpoint (2026-09-23)

**Question.** The released ForestFormer3D checkpoint produces visibly poor results on the
Berlin ALS tiles (`docs/benchmarks/2026-09-22-tegel-als.md`,
`docs/benchmarks/2026-09-23-berlin-visual-report.md`), where there is no ground truth to
score against. Berlin differs from the training data in many ways at once; the cheapest one
to isolate is **point density**: ForAINetV2 plots carry 125-4400 pts/m2, the Berlin ALS
carries roughly 20-30 pts/m2. This run thins the 28 *labelled* ForAINetV2 test plots to
25 pts/m2 and re-scores the same checkpoint, so the density effect can be measured against
real ground truth.

## Method

Host: carrot (H100), image `forestformer3d:cu118`, GPUs 2 and 3. Checkpoint
`work_dirs/clean_forestformer/epoch_3000_converted.pth` (the converted Zenodo release, the
same file the 2026-09-22 baseline used). Config
`configs/oneformer3d_qs_radius16_qp300_2many.py`, `randomness.seed=0`, all other
`test_cfg` defaults unchanged.

### Thinning (`benchmark/thin_plots.py`)

For each of the 28 plots in `data/ForAINetV2/meta_data/test_list.txt`, the 2D footprint
area is the area of the convex hull of the xy coordinates (shapely; on plots above 20 k
points the hull is computed from per-0.1 m-column extreme points, which changes the area by
< 0.1 %). The target point count is `density * area`. Both modes keep **every** field of
the input PLY (`x y z semantic_seg treeID`), so the thinned plots go through the ordinary
*labelled* preprocessing path and carry their own ground truth.

* **uniform** — random subsample without replacement, seed 0. This is the control: it
  reduces density without changing what the sensor sees, so stems, understorey and ground
  survive in proportion.
* **canopy** — an **approximation** of an airborne view: points are binned into 0.5 m xy
  cells, each cell is sorted by z descending, and only its highest `k` points are kept, with
  the budget distributed over occupied cells proportionally to their point counts (at least
  1 per cell). Under-canopy and stem points are therefore mostly dropped.
  This is **not** a sensor simulation: a real ALS system fires pulses that produce several
  returns, so ground and understorey points do survive under canopy gaps, the footprint is
  not a square cell, and the hit probability depends on the leaf area above a point rather
  than on rank in z. What the approximation reproduces is the first-return-like bias toward
  the top of the canopy, which is the property under test.

Density 25 pts/m2, seed 0, both modes. Result: 29 556 113 -> 475 893 points over the 28
plots (**1.61 % kept**) in both modes; per-plot input density ranged from 125 pts/m2
(`Yuchen_2023_dls_merged_230209_panoptic_test`) to 4387 pts/m2
(`NIBIO_NIBIO_plot_17_annotated_test`), output density is 25.0 pts/m2 for every plot.
Outputs: `data/ForAINetV2/test_data_thin25{u,c}/<stem>_thin25{u,c}.ply` (9.2 MB each set).
Thinning both sets took about 10 min of CPU time for the whole test split.

### Preprocessing and inference

Each set was preprocessed with its own private scan list (`work_dirs/logs/thin/<mode>/scan_list.txt`,
28 thinned stems) and its own `--test_forainetv2_dir`, so the tracked
`data/ForAINetV2/meta_data/test_list.txt` was never touched (md5 verified identical before
and after) and the new `forainetv2_instance_data/<stem>_thin25*_*.npy` exports sit beside
the untouched full-density ones. `tools/create_data_forainetv2.py --test-list <list>
--splits test --out-dir work_dirs/logs/thin/<mode>/` writes a private info pkl, which
`tools/test.py` picks up through
`--cfg-options test_dataloader.dataset.ann_file=/workspace/work_dirs/logs/thin/<mode>/forainetv2_oneformer3d_infos_test.pkl`.
The two sets ran in parallel on GPU 3 (uniform) and GPU 2 (canopy); the extra
`region_step_factor=0.5` variant ran on GPU 2 afterwards.

## Results

`final_eval` = `tools/final_eval.py` over the 28 result PLYs. `mmengine` = the
`UnifiedSegMetric` summary line of the same `tools/test.py` run. Baseline row from
`docs/benchmarks/2026-09-22-carrot-ff3d.md` (the "fixed" variant).

| condition | pts/m2 | F1 | mPrecision | mRecall | mMWCov | mMUCov | mIoU | meanPQ (final_eval) |
|---|---|---|---|---|---|---|---|---|
| full density (baseline) | 125-4387 | **0.9034** | 0.9245 | 0.8832 | 0.9029 | 0.8353 | 0.8523 | 0.8978 |
| thin, uniform | 25 | **0.6007** | 0.8438 | 0.4664 | 0.7255 | 0.5336 | 0.7400 | 0.7223 |
| thin, canopy | 25 | **0.3344** | 0.6801 | 0.2217 | 0.2351 | 0.2327 | 0.6524 | 0.6225 |
| thin, canopy, `region_step_factor=0.5` | 25 | **0.2845** | 0.7095 | 0.1779 | 0.1574 | 0.1777 | 0.6521 | 0.6048 |

mmengine evaluator summary lines (same runs):

| condition | mIoU | mIoU_binary | mMWCov | mMUCov | mPrecision | mRecall | F1 | mPQ |
|---|---|---|---|---|---|---|---|---|
| full density (baseline) | 0.8523 | 0.9935 | 0.9029 | 0.8353 | 0.9245 | 0.8832 | 0.9034 | 0.8072 |
| thin, uniform | 0.7400 | 0.9699 | 0.7255 | 0.5336 | 0.8438 | 0.4664 | 0.6007 | 0.4974 |
| thin, canopy | 0.6524 | 0.9912 | 0.2351 | 0.2327 | 0.6801 | 0.2173 | 0.3293 | 0.2571 |
| thin, canopy, step 0.5 | 0.6521 | 0.9915 | 0.1574 | 0.1777 | 0.7095 | 0.1696 | 0.2738 | 0.2166 |

Per-class semantic IoU `[unused, ground, wood, leaf]`:

| condition | ground | wood | leaf | oAcc |
|---|---|---|---|---|
| full density | 0.988 | 0.626 | 0.943 | 0.952 |
| thin, uniform | 0.947 | 0.350 | 0.923 | 0.930 |
| thin, canopy | 0.984 | 0.045 | 0.928 | 0.932 |
| thin, canopy, step 0.5 | 0.984 | 0.045 | 0.927 | 0.932 |

### Wall time per plot (inference only, one H100)

| condition | mmengine `time` per sample | 28 plots total |
|---|---|---|
| full density | 113.1 s | ~53 min |
| thin, uniform | 2.33 s | 83 s |
| thin, canopy | 1.68 s | 63 s |
| thin, canopy, step 0.5 | 0.50 s | 30 s |

Inference cost tracks point count almost linearly: 1.6 % of the points, ~2 % of the time.

## Over-segmentation diagnostics

`benchmark/instance_diagnostics.py` matches predicted instances to GT trees per plot
(Hungarian assignment on the IoU matrix, `scipy.optimize.linear_sum_assignment`) and pools
over the 28 plots. `frag/tree` counts, for each GT tree, the predictions that put >= 20 %
of *their own* points inside it; `split%` is the fraction of GT trees with >= 2 such
fragments; `merged%` is the fraction of predictions that cover >= 20 % of two or more GT
trees; `h_pred` / `h_gt` are median instance heights (top z minus the plot's min z).

| condition | GT trees | pred inst. | matched @IoU>=0.5 | match % | frag/tree | split % | merged % | h_pred (m) | h_gt (m) |
|---|---|---|---|---|---|---|---|---|---|
| full density | 1207 | 1153 | 1066 | 88.3 % | 1.02 | 6.9 % | 11.2 % | 15.2 | 15.0 |
| thin, uniform | 1205 | 666 | 562 | 46.6 % | 0.70 | 1.9 % | 48.5 % | 19.0 | 14.6 |
| thin, canopy | 1008 | 322 | 219 | 21.7 % | 0.46 | 3.2 % | 49.1 % | 15.3 | 17.0 |
| thin, canopy, step 0.5 | 1008 | 241 | 171 | 17.0 % | 0.34 | 2.1 % | 52.3 % | 14.1 | 17.0 |

Two bookkeeping notes. (1) The GT tree count drops from 1207 to 1008 in canopy mode: about
200 small, fully suppressed trees lose *all* their points to the canopy filter, so they are
no longer in the thinned ground truth at all — the canopy rows are scored against an easier,
overstorey-only reference and are still the worst rows in the table. (2) `h_gt` differs
slightly between full (15.0 m) and thin-uniform (14.6 m) because thinning removes the single
topmost point of some trees.

## Conclusion (5 lines)

1. Density alone is devastating: at 25 pts/m2 with structure preserved (uniform), F1 falls
   from **0.903 to 0.601** — a 0.30 absolute drop — driven almost entirely by recall
   (0.883 -> 0.466) while precision holds (0.925 -> 0.844).
2. Adding the airborne viewing bias on top costs as much again: the canopy approximation
   gives F1 **0.334**, with wood IoU collapsing from 0.626 to 0.045 because the stems the
   model relies on are simply not in the cloud any more.
3. The failure mode on thinned labelled data is **under-segmentation, not
   over-segmentation**: predicted instances drop from 1153 to 666 (uniform) and 322
   (canopy) against ~1200 GT trees, `split%` *falls* (6.9 % -> 1.9 %) and `merged%`
   quadruples (11.2 % -> 48.5 %) — roughly half of all predictions swallow two or more
   trees, and the median uniform-thinned instance is 4 m *taller* than a GT tree because it
   spans several crowns.
4. So density explains the *severity* of the Berlin result but **not its shape**: Berlin
   shows crowns split in two, 382 instances below 2 m and a median height of 17.6 m against
   a 26.2 m CHM baseline (`docs/benchmarks/2026-09-23-berlin-visual-report.md` sections 5-7,
   `docs/benchmarks/2026-09-22-tegel-als.md`) — short, fragmented, crown-only instances —
   whereas density loss alone produces few, tall, merged ones. Whatever produces the Berlin fragmentation is
   therefore a domain effect beyond density — species and stand structure, leaf-off
   foliage, ALS return characteristics and geometry, or the tiling/merge behaviour on
   100 m tiles rather than 14-22 m plots.
5. `region_step_factor=0.5` is a 2.3x speed-up that costs another 0.05 F1 on thinned data
   (0.334 -> 0.284, merged% 49 -> 52), so the denser 0.25 lattice should stay the default
   for low-density clouds; use 0.5 only when throughput matters more than instance recall.

## Reproduction

```bash
# on carrot, /raid/cwinkelmann/ForestFormer3D
for pair in "uniform u" "canopy c"; do set -- $pair
  docker run --rm --entrypoint python -e PYTHONPATH=/workspace -w /workspace \
    -v $PWD:/workspace forestformer3d:cu118 benchmark/thin_plots.py \
      --src data/ForAINetV2/test_data --dst data/ForAINetV2/test_data_thin25$2 \
      --density 25 --mode $1 --seed 0 \
      --list data/ForAINetV2/meta_data/test_list.txt \
      --out-list work_dirs/logs/thin/$1/scan_list.txt
done
bash work_dirs/logs/thin/run_mode.sh uniform u 3          # preprocess + test.py + final_eval
bash work_dirs/logs/thin/run_mode.sh canopy  c 2
bash work_dirs/logs/thin/run_mode.sh canopy c 2 -step model.test_cfg.region_step_factor=0.5
docker run --rm --entrypoint python -e PYTHONPATH=/workspace -w /workspace \
  -v $PWD:/workspace forestformer3d:cu118 benchmark/instance_diagnostics.py \
    --run full=work_dirs/bench-release-fixed \
    --run thin-uniform=work_dirs/logs/thin/uniform/out \
    --run thin-canopy=work_dirs/logs/thin/canopy/out --per-plot
```

Raw outputs kept on carrot (not committed): `data/ForAINetV2/test_data_thin25u`,
`data/ForAINetV2/test_data_thin25c` (9.2 MB each) and `work_dirs/logs/thin/` (45 MB:
per-mode logs, scan lists, info pkls, result PLYs, `diag_*.json`, and the two driver
scripts `run_thin.sh` / `run_mode.sh` written for this session).

## Caveats

* Single run per condition, single seed. Full-plot inference is not reproducible run to run
  (`docs/benchmarks/2026-09-23-inference-profile.md`): two unseeded runs of the same code
  disagree on ~5.7 % of point assignments. The differences reported here (0.30 and 0.57 F1)
  are far outside that noise floor; the 0.05 F1 cost of `region_step_factor=0.5` is closer
  to it and should be repeated before being treated as settled.
* One density (25 pts/m2) only. The shape of the degradation curve between 25 and
  ~700 pts/m2 is unmeasured, so this says nothing about where the model stops working.
* The canopy mode is an approximation of an airborne view, not a sensor simulation (see
  Method); real ALS retains more under-canopy returns than it does, so the canopy row is
  probably a pessimistic bound rather than a prediction of ALS performance.
* GT trees that lose all their points in canopy mode are dropped from the reference, which
  flatters the canopy rows.
