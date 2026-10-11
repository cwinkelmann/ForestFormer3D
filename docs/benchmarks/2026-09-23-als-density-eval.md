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
  It also has a floor: once the budget drops below the number of occupied 0.5 m cells
  (about 4 pts/m2 on a forest plot) the mode flips to "one randomly chosen point per cell"
  and stops being canopy-biased at all. The script warns on stderr when that happens; it
  does not happen at 25 pts/m2.

Density 25 pts/m2, seed 0, both modes. Result: 29 556 113 -> 475 893 points over the 28
plots (**1.61 % kept**) in both modes; per-plot input density ranged from 125 pts/m2
(`Yuchen_2023_dls_merged_230209_panoptic_test`) to 4387 pts/m2
(`NIBIO_NIBIO_plot_17_annotated_test`). Output density is 25.0 pts/m2 for every plot by
construction (`kept == target == round(25 * area)`), so that is a statement of what was
asked for, not an independent check.
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

**Why the two evaluators disagree on the canopy rows — and only there.** Precision agrees
to 4 dp in all four conditions, and both tools derive it from the same counts: TP / P is
1066/1153 (full), 562/666 (uniform), 219/322 (canopy), 171/241 (step 0.5) — the same TP and
prediction counts the diagnostics table below reports. Recall also agrees for full density
and uniform. It diverges on the two canopy runs because the two tools use different
denominators. `tools/final_eval.py` accumulates `total_gt_ins` *inside* a loop it
`continue`s out of when a plot has no predicted instance of the thing class
(`tools/final_eval.py:241` guarding `:248`), so the GT trees of a plot that produced nothing
at all never enter the recall denominator, while `UnifiedSegMetric` counts them. Canopy:
219/988 = 0.2217 (final_eval) vs 219/1008 = 0.2173 (mmengine); step 0.5: 171/961 = 0.1779 vs
171/1008 = 0.1696. Full density and uniform have no empty plot, which is why they agree
exactly. The mmengine denominator is the honest one — final_eval's quirk flatters a run
precisely when it fails hardest — and it is the one the diagnostics table uses. Magnitude
0.005 / 0.011 F1; no conclusion changes.

**`meanPQ` (final_eval) vs `mPQ` (mmengine) are not the same quantity** and should not be
read as a contradiction (0.6225 vs 0.2571 for canopy). `tools/final_eval.py:403` averages PQ
over all classes present, **including the ground/stuff class**, whose PQ is ~0.98 here;
`oneformer3d/unified_metric.py:228` averages over the *thing* classes only. The stuff class
stays easy while the tree class collapses, so the two diverge exactly as the instance
metrics fall.

Per-class semantic IoU `[unused, ground, wood, leaf]`:

| condition | ground | wood | leaf | oAcc |
|---|---|---|---|---|
| full density | 0.988 | 0.626 | 0.943 | 0.952 |
| thin, uniform | 0.947 | 0.350 | 0.923 | 0.930 |
| thin, canopy | 0.984 | 0.045 | 0.928 | 0.932 |
| thin, canopy, step 0.5 | 0.984 | 0.045 | 0.927 | 0.932 |

### Wall time (inference only, one H100)

Two definitions, kept apart. "per sample" is mmengine's own mean `time` over the 28 test
samples (model forward + tiling + PLY write, no startup). "step wall clock" is the wall
clock of the whole `tools/test.py` container, which adds a fixed ~16-20 s of container
start, entrypoint patch, import and checkpoint load.

| condition | per sample | 28 samples at that rate | step wall clock |
|---|---|---|---|
| full density | 113.1 s | 3167 s | 3290 s (54 min 50 s) |
| thin, uniform | 2.33 s | 65 s | 83 s |
| thin, canopy | 1.68 s | 47 s | 63 s |
| thin, canopy, step 0.5 | 0.50 s | 14 s | 30 s |

Inference cost tracks point count almost linearly: 1.6 % of the points, 2.1 % of the
per-sample time. `region_step_factor=0.5` is a **3.4x** per-sample speed-up (1.68 -> 0.50 s)
which, because the fixed startup then dominates, is only **2.1x** on the step wall clock.

## Instance-level diagnostics

`benchmark/instance_diagnostics.py` matches predicted instances to GT trees per plot
(Hungarian assignment on the IoU matrix, `scipy.optimize.linear_sum_assignment`) and pools
over the 28 plots. Definitions:

* **frag/gt**, **split%gt** — for each GT tree, the number of predictions that put >= 20 %
  of *their own* points inside it; averaged over **all** GT trees, and the fraction of all
  GT trees with >= 2 such fragments.
* **det**, **frag/det**, **split%det** — the same two numbers restricted to the GT trees
  that got at least one fragment. The all-GT columns fall automatically when recall falls
  (a tree nothing came near contributes 0 fragments), so only the conditioned columns can be
  compared across conditions with different recall.
* **merged%** — fraction of predictions that cover >= 20 % of two or more GT trees.
* **dz_med**, **dz_mean** — (predicted top z − GT top z) over the *matched* pairs only.
* **h_pred**, **h_gt** — median instance height over the surviving predictions and over all
  GT trees. These are two different populations, so their difference moves with recall alone
  and is reported only to show that it does.

| condition | GT | pred | matched @IoU>=0.5 | match % | frag/gt | split%gt | det | frag/det | split%det | merged % | dz_med | dz_mean | h_pred | h_gt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full density | 1207 | 1153 | 1066 | 88.3 % | 1.02 | 6.9 % | 1142 | 1.08 | 7.3 % | 11.2 % | 0.00 m | −0.00 m | 15.2 m | 15.0 m |
| thin, uniform | 1205 | 666 | 562 | 46.6 % | 0.70 | 1.9 % | 811 | 1.04 | 2.8 % | 48.5 % | 0.00 m | +0.04 m | 19.0 m | 14.6 m |
| thin, canopy | 1008 | 322 | 219 | 21.7 % | 0.46 | 3.2 % | 423 | 1.10 | 7.6 % | 49.1 % | 0.00 m | +0.52 m | 15.3 m | 17.0 m |
| thin, canopy, step 0.5 | 1008 | 241 | 171 | 17.0 % | 0.34 | 2.1 % | 318 | 1.09 | 6.6 % | 52.3 % | 0.00 m | +0.65 m | 14.1 m | 17.0 m |

Reading the table:

* **Merging is the density failure mode.** `merged%` goes from 11.2 % at full density to
  48.5-52.3 % on every thinned run: about half of all surviving predictions swallow two or
  more GT trees. Prediction counts fall far below the GT counts (1153 -> 666 -> 322 against
  1207 / 1205 / 1008 trees).
* **Splitting does not get worse — except that it does not get better either.** Conditioned
  on trees the model actually found, `split%det` is 7.3 % at full density, **2.8 %** on
  uniform thinning and **7.6 %** on canopy thinning: the canopy run splits detected trees at
  the same rate as full density while finding only a fifth of them. `frag/det` is flat
  (1.04-1.10) across all four conditions. The unconditioned `split%gt` column falls from
  6.9 % to 1.9-3.2 %, but that is the recall collapse, not an improvement, which is why the
  conditioned columns exist.
* **Matched instances have the right height.** `dz_med` is 0.00 m in every condition and
  `dz_mean` never exceeds +0.65 m: when the model finds a tree, it gets its top within
  centimetres. The eye-catching `h_pred` vs `h_gt` gaps (19.0 vs 14.6 m on uniform, 15.3 vs
  17.0 m on canopy) are selection effects — the surviving predictions are biased toward
  large trees on uniform and the canopy reference contains taller trees — not evidence that
  instances are too tall or too short.

Bookkeeping notes.

1. `n_gt` counts GT trees that still have at least one point in the result cloud, so
   thinning removes trees outright: 1207 -> **1205** under uniform thinning (2 tiny trees
   lost every point to a 1.61 % random subsample) and -> **1008** under canopy thinning
   (about 200 small, fully suppressed trees lost every point to the top-of-cell filter).
   The canopy rows are therefore scored against an easier, overstorey-only reference of
   1008 trees and are still the worst rows in the table.
2. `h_gt` differs slightly between full (15.0 m) and thin-uniform (14.6 m) because thinning
   removes the single topmost point of some trees.
3. One canopy plot and two `step 0.5` plots produced **no instance at all** (20 and 47 GT
   trees respectively) — this is what `tools/final_eval.py` drops from its recall
   denominator, see the note under Results.
4. `mMUCov > mMWCov` only in the `step 0.5` row (0.1777 vs 0.1574). Both evaluators report
   the same pair independently, so it is not a transcription slip: at that level of
   degradation the few surviving detections sit on smaller trees, and MWCov weights each GT
   tree by its point count.

## Conclusion (5 lines)

1. Density alone is devastating: at 25 pts/m2 with structure preserved (uniform), F1 falls
   from **0.903 to 0.601** — a 0.30 absolute drop — driven almost entirely by recall
   (0.883 -> 0.466) while precision holds (0.925 -> 0.844).
2. Adding the airborne viewing bias on top costs as much again: the canopy approximation
   gives F1 **0.334** (mmengine 0.329), with wood IoU collapsing from 0.626 to 0.045 because
   the stems the model relies on are simply not in the cloud any more.
3. On **uniform** thinning the failure is unambiguously **merging / under-segmentation**:
   666 predictions for 1205 trees, 48.5 % of predictions covering two or more GT trees
   (11.2 % at full density), and splitting of detected trees *below* the full-density rate
   (2.8 % vs 7.3 %).
4. On **canopy** thinning the picture is mixed and is the one that matters for ALS: merging
   is just as bad (49.1 %), but detected trees are split about as often as at full density
   (7.6 % vs 7.3 %) while only 21.7 % of trees are found at all — so the model both misses
   most trees and keeps fragmenting the ones it finds. Matched-pair height deltas are
   ~0 m in every condition (median 0.00 m, mean <= +0.65 m), so density loss does **not**
   make instances systematically short.
5. Therefore density explains the *severity* of the Berlin ALS result and reproduces its
   merging, but it does **not** reproduce the short instances reported there (382 crowns
   below 2 m, median 17.6 m against a 26.2 m CHM baseline,
   `docs/benchmarks/2026-09-23-berlin-visual-report.md`). The open question is precise:
   **what makes Berlin instances short when 25 pts/m2 on labelled plots does not?** Candidates
   that this experiment does not separate are leaf-off foliage, species and stand structure,
   ALS return characteristics, and the 100 m tiling/merge path (14-22 m plots never exercise
   it). `region_step_factor=0.5` is a 3.4x per-sample speed-up costing another 0.05 F1 on
   thinned data, so keep 0.25 for low-density clouds. At full density the same factor
   costs no measurable F1 (it is inside the seed-to-seed spread) at a 3.7x speed-up,
   `docs/benchmarks/2026-09-23-inference-profile.md`; this row is the reason the shipped
   default stays 0.25 anyway.

## Reproduction

Every step is a tracked script; nothing depends on a file written ad hoc during the session.

```bash
# on carrot, /raid/cwinkelmann/ForestFormer3D
bash benchmark/thin_make_sets.sh                       # both thinned sets + scan lists
FF3D_GPU=3 bash benchmark/thin_eval.sh uniform u       # preprocess + test.py + final_eval
FF3D_GPU=2 bash benchmark/thin_eval.sh canopy  c
FF3D_GPU=2 bash benchmark/thin_eval.sh canopy c -step model.test_cfg.region_step_factor=0.5
docker run --rm --entrypoint python -e PYTHONPATH=/workspace -w /workspace \
  -v $PWD:/workspace forestformer3d:cu118 benchmark/instance_diagnostics.py \
    --run full=work_dirs/bench-release-fixed \
    --run thin-uniform=work_dirs/logs/thin/uniform/out \
    --run thin-canopy=work_dirs/logs/thin/canopy/out \
    --run canopy-step0.5=work_dirs/logs/thin/canopy-step/out \
    --per-plot --json work_dirs/logs/thin/diag_all.json
```

`FF3D_DRY_RUN=1` prints the docker commands of either shell script without running them.
`FF3D_THIN_DENSITY` / `FF3D_THIN_SEED` change the density and seed (both scripts must be
given the same value).

Raw outputs kept on carrot (not committed): `data/ForAINetV2/test_data_thin25u`,
`data/ForAINetV2/test_data_thin25c` (9.2 MB each) and `work_dirs/logs/thin/` (45 MB:
per-mode logs, scan lists, info pkls, result PLYs, `diag_all.json`).

## Caveats

* Single run per condition, single seed. Full-plot inference is not reproducible run to run
  (`docs/benchmarks/2026-09-23-inference-profile.md`): two unseeded runs of the same code
  disagree on ~5.7 % of point assignments. The differences reported here (0.30 and 0.57 F1)
  are far outside that noise floor; the 0.05 F1 cost of `region_step_factor=0.5` and the
  2.8 % vs 7.3 % `split%det` gap on uniform thinning are closer to it and should be
  repeated before being treated as settled.
* One density (25 pts/m2) only. The shape of the degradation curve between 25 and
  ~700 pts/m2 is unmeasured, so this says nothing about where the model stops working.
* The canopy mode is an approximation of an airborne view, not a sensor simulation (see
  Method); real ALS retains more under-canopy returns than it does, so the canopy row is
  probably a pessimistic bound rather than a prediction of ALS performance. Below about
  4 pts/m2 it degenerates into a random one-point-per-cell sample and stops being canopy
  biased at all (the script warns).
* GT trees that lose all their points in thinning are dropped from the reference (2 under
  uniform, ~200 under canopy), which flatters the thinned rows — especially canopy.
* The Berlin comparison is against a CHM local-maxima baseline, not ground truth; "Berlin
  instances are short" is a statement about the model vs that baseline
  (`docs/benchmarks/2026-09-22-tegel-als.md`,
  `docs/benchmarks/2026-09-23-berlin-visual-report.md`), and the two experiments differ in
  tile size (1 km tiles split to 100 m vs 14-22 m plots) as well as in domain.
* The fragment / merge thresholds (20 % of a prediction's own points; 20 % of a GT tree)
  and the 0.5 IoU match threshold are conventions, not standards; the *relative* movement
  between conditions is the result, not the absolute levels.
