---
name: ff3d-evaluation
description: Use when evaluating ForestFormer3D - the 28-plot benchmark on carrot, the ALS-density thinning study, instance diagnostics, the run-to-run nondeterminism noise floor, final_eval's recall-denominator quirk and the region_step_factor speed/quality trade-off.
---

# Evaluating the model

Read `ff3d-carrot` first (ssh, image, `benchmark/common.sh`, GPU etiquette).
Everything here runs on carrot in the container; all scripts are tracked.

## 1. The 28-plot benchmark (`docs/benchmarks/RUNBOOK-carrot.md`)

The labelled ForAINetV2 test split is 28 plots (the Zenodo `test_data/` holds 29 PLYs; the
test list names 28). Two things are measured separately:

```bash
cd /raid/cwinkelmann/ForestFormer3D
# one-time: data + checkpoint, old-code worktree
FF3D_ZENODO_CACHE=/raid/cwinkelmann/zenodo-16742708 bash benchmark/fetch_zenodo.sh
bash benchmark/setup_old_worktree.sh          # main @ 6a75c37 at /raid/cwinkelmann/ff3d-old-main
bash -c 'source benchmark/common.sh; FF3D_GPU=2 ff3d_preprocess'

# A. released checkpoint, old code vs fixed code (inference-only regressions)
FF3D_GPU=2 bash benchmark/run_release_eval.sh
tail -f work_dirs/logs/release-eval-*.log

# B. two 200-epoch trainings in parallel (training-loop regressions)
FF3D_GPU=3 bash benchmark/run_train_200.sh old
FF3D_GPU=4 bash benchmark/run_train_200.sh fixed

# C. collect into a dated report
DATE=$(date +%F)
docker run --rm --entrypoint python -e PYTHONPATH=/workspace -w /workspace \
  -v /raid/cwinkelmann/ForestFormer3D:/workspace \
  forestformer3d:cu118 benchmark/collect.py --root /workspace --date "$DATE" --allow-missing
cat docs/benchmarks/$DATE-carrot-ff3d.md
```

Preview anything first:
`FF3D_GPU=2 FF3D_DRY_RUN=1 FF3D_FOREGROUND=1 bash benchmark/run_release_eval.sh`.

Each script daemonizes itself with `nohup` and prints its log path, so an SSH drop is
harmless; every stage has a marker inside its own output dir
(`work_dirs/bench-release-fixed/.done-test`, `.../.done-eval`,
`work_dirs/bench-<variant>-200/.done-train`, ...) and a re-run skips finished stages.
`FF3D_FORCE=1` re-infers an existing output dir; `FF3D_DRY_RUN=1` writes nothing at all.
The full marker table is section 7 of the runbook.

Two `old`-specific behaviours, by design: a non-zero exit of the **old** `tools/test.py`
is tolerated (its evaluator always crashes with an `IndexError` *after* every result PLY is
written; the real postcondition is 28 PLYs), and a complete output dir is **adopted**
rather than deleted (`adopting 28 existing PLYs`).

Published results (2026-09-22, `docs/benchmarks/2026-09-22-carrot-ff3d.md`): released
checkpoint scored with the current `final_eval.py` - old code **F1 0.9064**, fixed code
**F1 0.9034** (precision 0.945 -> 0.925, recall 0.871 -> 0.883, mIoU 0.852 both).
200-epoch runs (with `prepare_epoch` overridden to 60, otherwise the instance decoder never
trains): test F1 old 0.834, fixed 0.811, one seed each.

Interpretation: "old variant" = original code **and** its original config from the
`6a75c37` worktree; only the Docker image is held constant. Both variants are scored with
the same current `tools/final_eval.py`, so the tables are comparable.

## 2. Nondeterminism - the noise floor

**Full-plot inference is not reproducible run to run.** `torch_cluster.fps` defaults to
`random_start=True`, so every run draws different query points; `randomness.seed=0` fixes
that but CUDA atomics and `grid_sample`'s `index_copy_` ties remain.
`randomness.deterministic=True` is not usable at all (it makes spconv's `torch.mm` raise).

Measured over ten 100 m sub-tiles (`docs/benchmarks/2026-09-23-inference-profile.md`):

| comparison | instance ids differing, after Hungarian re-matching | pairwise inst-F1@0.5 |
|---|---:|---:|
| same code, same seed, twice | **1.40-1.59 %** | 0.990 |
| old vs new code, seed 0 | 1.39 % | 0.987 |
| old vs new, **unseeded** | **5.73 %** | 0.959 |

(Raw `instance_pred` differs ~31 % before re-matching - about 95 % of that is renumbering.)

Consequences you must respect:

- **A 0.003 F1 difference is noise**, not a regression. The 0.9064 vs 0.9034 gap above is
  exactly that size and is reported as a failed gate, not as a finding.
- A whole-plot PLY diff can never prove a code change is equivalent. Prove equivalence with
  unit tests on the changed function (as `tests/test_result_ply.py` and the vectorised
  z-filter tests do) and quote the same-code control alongside any end-to-end diff.
- Differences of 0.05 F1 and below should be repeated before being believed; 0.30 F1 is far
  outside the floor.

## 3. ALS-density thinning study

Question: the released checkpoint does badly on Berlin ALS, where there is no ground truth.
Thinning the **labelled** test plots to ALS density (ForAINetV2 carries 125-4387 pts/m2,
Berlin ALS 20-30) isolates density from domain. Write-up:
`docs/benchmarks/2026-09-23-als-density-eval.md`.

```bash
cd /raid/cwinkelmann/ForestFormer3D
bash benchmark/thin_make_sets.sh                     # both sets + private scan lists
FF3D_GPU=3 bash benchmark/thin_eval.sh uniform u     # preprocess + test.py + final_eval
FF3D_GPU=2 bash benchmark/thin_eval.sh canopy  c
FF3D_GPU=2 bash benchmark/thin_eval.sh canopy c -step model.test_cfg.region_step_factor=0.5
```

`benchmark/thin_plots.py` has two modes: **uniform** (random subsample - the control,
structure preserved) and **canopy** (0.5 m xy cells, keep the highest `k` per cell - an
*approximation* of an airborne view, not a sensor simulation; below ~4 pts/m2 it degenerates
into one random point per cell and warns). The target count is `density * convex-hull area`.
Every field of the input PLY survives, so the thinned plots go through the ordinary
**labelled** path and carry their own ground truth. Output name
`<stem>_thin<density><u|c>.ply` (deliberately not ending in `_<digits>`, which the data
tools would strip as a block index).

`FF3D_THIN_DENSITY` (default 25) and `FF3D_THIN_SEED` (default 0) must be given to **both**
scripts. Neither ever touches `data/ForAINetV2/meta_data/test_list.txt`: the scan list is
`work_dirs/logs/thin/<mode>/scan_list.txt` and everything lands under
`work_dirs/logs/thin/<mode><suffix>/` (`out/*.ply`, `out/evaluation_total_test.txt`,
`t_start`, `t_end`). `FF3D_DRY_RUN=1` prints the docker commands. Thinning needs no GPU, so
`thin_make_sets.sh` uses `--entrypoint python` to bypass the CUDA-asserting entrypoint.

Headline result at 25 pts/m2 (final_eval F1): full density **0.9034** -> uniform **0.6007**
-> canopy **0.3344** -> canopy + `region_step_factor=0.5` **0.2845**. Wood IoU collapses
0.626 -> 0.045 on canopy: the stems the model relies on are simply not in the cloud.

## 4. Instance diagnostics

`final_eval` says how well; `benchmark/instance_diagnostics.py` says **how it fails**. It
reads result PLYs (which carry `semantic_gt` / `instance_gt` alongside the prediction), does
a Hungarian assignment on the IoU matrix per plot and pools over plots.

```bash
docker run --rm --entrypoint python -e PYTHONPATH=/workspace -w /workspace \
  -v $PWD:/workspace forestformer3d:cu118 benchmark/instance_diagnostics.py \
    --run full=work_dirs/bench-release-fixed \
    --run thin-uniform=work_dirs/logs/thin/uniform/out \
    --run thin-canopy=work_dirs/logs/thin/canopy/out \
    --per-plot --json work_dirs/logs/thin/diag_all.json
```

Columns: `gt, pred, matched@IoU>=0.5, frag/gt, split%gt, det, frag/det, split%det, merged%,
dz_med, dz_mean, h_pred, h_gt`. **Read only the `det`-conditioned columns across conditions
with different recall** - the all-GT fragment columns fall automatically when recall falls.
`h_pred` vs `h_gt` compares two different populations and moves with recall alone; only
`dz_med` / `dz_mean` (matched pairs) say whether instances are too short.

Findings: merging is the density failure mode (`merged%` 11.2 % -> 48.5-52.3 %); splitting of
*detected* trees does not get worse (`split%det` 7.3 % -> 2.8 % uniform, 7.6 % canopy);
matched instances have the right height everywhere (`dz_med` 0.00 m, `dz_mean` <= +0.65 m).
So density explains the severity and the merging seen in Berlin, but **not** the short
Berlin instances - that remains the open question.

## 5. `final_eval.py`'s recall-denominator quirk

`tools/final_eval.py` accumulates `total_gt_ins` *inside* a loop it `continue`s out of when
a plot produced no predicted instance of a thing class (`tools/final_eval.py:241` guarding
`:248`). The GT trees of a completely empty plot therefore never enter the recall
denominator, while mmengine's `UnifiedSegMetric` counts them. Canopy: 219/988 = 0.2217
(final_eval) vs 219/1008 = 0.2173 (mmengine). Precision agrees to 4 dp always; recall only
diverges when some plot produced nothing at all. **The mmengine denominator is the honest
one** - final_eval's quirk flatters a run precisely when it fails hardest.

Also not the same quantity: `meanPQ` from `final_eval.py:403` averages over **all** classes
including the ground/stuff class (PQ ~ 0.98), while `mPQ` from
`oneformer3d/unified_metric.py:228` averages over thing classes only. 0.6225 vs 0.2571 is
not a contradiction.

`UnifiedSegMetric` (`oneformer3d/unified_metric.py`, stuff `[0]`, things `[1,2]`) runs during
validation. `tools/final_eval.py <dir>` is the standalone evaluator over a directory of
result PLYs and rewrites `evaluation_total_test.txt` there on every run.

## 6. Inference profile and `region_step_factor`

`docs/benchmarks/2026-09-23-inference-profile.md`. The cylinder lattice pitch is
`radius * model.test_cfg.region_step_factor` (default **0.25** -> step 4 m -> 625 cylinders
per 100 m tile, every point seen ~44 times). Override without editing the config:
`--cfg-options model.test_cfg.region_step_factor=0.5`.

- `model.test_cfg.region_batch` (default **8**, since 2026-10-08) batches that many cylinders
  through the sparse collate + UNet + heads; the decoder and mask post-processing stay per
  region, so results are unchanged within the noise floor. Idle H100, one 100 m Berlin
  sub-tile: inference step 73 s -> 54 s, flat from 8 to 32 (addendum in the profile doc).
  `batched_decoder` (default True, same day) adds one fps call and a padded batched decoder
  pass per batch: another ~18 % off the inference step. Both are free speed-ups with no
  quality trade-off, unlike the step factor below.
- `0.5` -> 169 regions, measured **3.4x** faster per sample - and **-0.05 F1** on thinned
  data (0.3344 -> 0.2845), with 14 % fewer merged instances on an unlabeled tile.
  **Keep 0.25 for low-density clouds**; `0.333` is the conservative middle step if speed
  is needed.
- Any change to it must be checked with `tools/final_eval.py` on the labelled 28 plots
  (F1, MUCov, PQ) before becoming a default - and read the result against the noise floor
  in section 2.
- The big win already landed: vectorising the per-mask z filter in `pred_inst_sem_test`
  plus the binary PLY writer measured **6.7x** (316.7 -> 47.2 s per scan), a
  quality-neutral change proven by unit tests, not by an output diff.

## 7. Unlabeled ALS tiles have no ground truth

For Berlin tiles the mmengine evaluator runs against constant labels, so its numbers are
meaningless (harmless, but ignore them). The only per-tile quality signal there is the
report's CHM local-maxima baseline and its decision rule - see `ff3d-inference-km-tiles`
section 10 and `ff3d-outputs-and-viewers`.

## 8. Mosaic-wide comparisons and analytics (`benchmark/berlin_analytics.py`)

One script, no LAS reads, ~1 minute: per-method tree counts per km tile (grid heatmaps and
the SAT/FF3D ratio, also per survey footprint -- R12 0.96 vs R13 1.21 on 2026-10-05),
height / crown / points-per-tree distributions (bimodal heights: canopy ~26 m, understory
4-7 m), quality flags (flat blobs < 2 m & > 50 m2 = ground labelled leaf; noise-sized < 20
points), the pooled FF3D-vs-SAT agreement from the per-tile `instance_agreement.py` JSONs
(copy them from carrot `work_dirs/logs/sat/agreement33/` to `ALS_Data/berlin_agreement/
ff3d_vs_sat_33/`), seam metrics and stitch counts per method, the building mask per tile,
predicted p90 height per stand against the 2014 inventory height by dominant species, and
detection rate + height residual against the tree cadastre (nearest predicted top within
3 m: ~82 % FF3D, ~86 % SAT, bias ~0, MAE ~4 m, r ~0.5). Output is the report chapter
`berlin-analytics` (see `ff3d-pdf-report`) plus `assets/analytics/analytics.json`. Agreement
needs identical points in identical order: the 3-tile AMS3D set from the old un-haloed run
cannot be compared that way (the 44- and 57-tile runs compute all three pairs on carrot; `berlin_analytics.py`'s `AGREEMENT_TAG` selects the mosaic, `AGREEMENT_TAG_FALLBACK` keeps a pair that ran on fewer tiles, e.g. PointTreeFormer's 15).

## 9. Speed: throughput under sharing vs a clean benchmark

`benchmark/tile_timings.py` reads the production queue logs; its per-sub-tile seconds carry two
concurrency columns (queues sharing the same GPU / the host, time-weighted) because the
production runs were never alone: the 44-tile medians (FF3D 61 s, SAT 73 s, AMS3D 11 s per
100 m sub-tile) were measured with 5-7 queues on the host and two SAT containers per GPU, and
the 13-tile extension ran three FF3D queues on one GPU. Only rows with similar concurrency
compare. `benchmark/speed_benchmark.sh <gpu> <tiles>...` is the clean measurement: it waits
until none of our queues/chains runs, then per tile runs FF3D (one process), SAT (one container)
and AMS3D (48 workers) strictly in sequence on one GPU, sampling `nvidia-smi` every 10 s, into
`work_dirs/bench-<T>-<method>/` and `work_dirs/logs/bench/speed-<TS>.json` (seconds, s per
sub-tile, GPU utilisation mean/max, memory). Production results are never touched
(`SAT_OUT_PREFIX`, `AMS3D_OUT_PREFIX`). The methods chapter quotes the clean figures; the
production ones are "throughput as run".

**Measured 2026-10-08** (GPU 4, idle host, commit `b0570a1`, i.e. before `region_batch` and
`batched_decoder`), tiles 381_5828 (25.2 M points) and 379_5829 (15.1 M): FF3D 104.3 and
72.7 s per sub-tile, SAT 41.2 and 28.9 s, AMS3D 17.3 and 3.7 s. Normalise by the tile's
points and the deep methods are linear to the third digit -- **FF3D 414 s, SAT 164 s per
million points** (2.0 h and 0.8 h for the mosaic's mean 17.6 M-point tile) -- while AMS3D
is not (69 vs 21 s/M; it follows the tree count, 38.3 k vs 16.2 k). GPU utilisation of one
process: FF3D 21-38 % mean (max 87-90 %), SAT 24-28 % (max 43-47 %), memory under 4.2 GB.
Quote s/M points, never s/sub-tile, when comparing tiles.
