# How inference works in this checkout

This page explains the end-to-end inference path as it is used today: from a raw
ALS tile to a georeferenced LAS, a tree table, raster masks and a report. It
complements the runbooks (`docs/benchmarks/RUNBOOK-carrot.md` for the benchmark,
`docs/benchmarks/RUNBOOK-tegel.md` for ALS tiles) which hold the exact commands.

## Machines and environments

| Where | What runs there | Environment |
|---|---|---|
| GPU server carrot (`ssh carrot`, 8x H100) | every GPU step, the batch runs | Docker image `forestformer3d:cu118` (CUDA 11.8, torch 2.0.1, mmdet3d, MinkowskiEngine, spconv) started through `benchmark/common.sh`'s `ff3d_docker` with `FF3D_GPU=<n>`; checkout `/raid/cwinkelmann/ForestFormer3D`; host venv `/raid/cwinkelmann/ff3d-geo-venv` for the pure-Python `ff3d_geo` package |
| Mac (no GPU) | conversion, merging, masks, reports, tests | `.venv-cpu` (numpy, laspy, shapely, geopandas, pyogrio, rasterio); no torch |
| Lenovo T14 (RTX 4080 SUPER 16 GB) | optional second GPU worker | same image built from the `Dockerfile`; venv `~/ff3d-geo-venv`; setup started, build paused at step 12 of 19 |

The GPU steps never import `ff3d_geo`, and `ff3d_geo` never imports torch: the two
halves talk through files only.

## The model side (inside the container)

`tools/test.py <config> <checkpoint> --work-dir <out> --cfg-options test_dataloader.dataset.ann_file=<pkl>`

1. mmengine builds `ForAINetV2OneFormer3D_XAwarequery` from
   `configs/oneformer3d_qs_radius16_qp300_2many.py` and loads
   `work_dirs/clean_forestformer/epoch_3000_fix.pth` (the released checkpoint, already in
   the converted spconv layout; `tools/fix_spconv_checkpoint.py` converts a raw one).
2. The test dataset reads the info pkl, one entry per scan; each entry points at the
   per-scan `.npy` exports under `data/ForAINetV2/forainetv2_instance_data/`
   (`<scan>_vert.npy`, `_sem_label.npy`, `_ins_label.npy`, `_offsets.npy`, ...).
3. `predict()` runs `_predict_full_plot` (`oneformer3d/oneformer3d.py`): the whole scan is
   tiled into overlapping cylinders (`oneformer3d/tiling.py::generate_cylindrical_regions`,
   radius 16 m, step radius/4 = 4 m: 625 cylinders for a 100 m tile, so every point is
   processed about 44 times), each cylinder is sampled to at most
   640,000 points, run through the sparse-conv backbone and the query decoder
   (300 instance queries + 3 semantic queries), and the per-cylinder instance masks are
   merged across cylinders by score (`tiling.merge_instances_by_score`, overlap threshold
   0.3, score threshold 0.4). Semantics are per-point votes (`tiling.SemanticVotes`).
4. The result is written as `<out>/<scan>.ply` with `x y z` (float32, centered by
   `<scan>_offsets.npy` = `[mean_x, mean_y, min_z]`), `semantic_pred` (0 ground, 1 wood,
   2 leaf, -1 no vote), `instance_pred` (tree id, -1 none) and `score`.
5. The mmengine evaluator then scores the scan against its labels. For unlabeled ALS
   scans the labels are constants, so its numbers are meaningless but harmless.

Validation during training uses the non-tiled path (`tools/train.py` forces
`full_plot=False`).

## The geo side (`ff3d_geo`, host Python)

`python -m ff3d_geo run --las <tile>.las [...] --checkpoint <pth> --out <dir> --gpu <n>`
executes eight steps and prints each one before running it (`--dry-run` prints only):

1. `las_to_ply`: every input LAS (local coordinates 0..100 m, origin encoded in the file
   name as `E<easting>_N<northing>`) becomes `data/ForAINetV2/test_data/<stem>.ply` with
   `x y z` only, plus `<out>/<stem>.sidecar.json` (origin, EPSG, point count, source
   format) and `<stem>_classification.npy` (the ALS classes, kept for the report).
2. `prepare_inputs`: writes a private scan list `<out>/scan_list.txt` (one stem per line)
   and removes stale exports for those stems. The tracked
   `data/ForAINetV2/meta_data/test_list.txt` is never touched.
3. `preprocess` (container): `batch_load_ForAINetV2_data.py --unlabeled` exports the
   `.npy` files and superpoints for the listed stems, then
   `create_data_forainetv2.py --test-list <scan list> --splits test --out-dir <out>`
   writes `<out>/forainetv2_oneformer3d_infos_test.pkl` (the benchmark pkls under
   `data/ForAINetV2/` stay untouched).
4. `check_preprocess`: every stem has `_vert.npy` and `_offsets.npy`, and the pkl is
   newer than the scan list.
5. `inference` (container): one `tools/test.py` call for all stems (one model load,
   N result PLYs).
6. `results_to_las`: `<out>/<stem>.ply` + offsets + sidecar -> `<out>/<stem>.las`
   (LAS 1.4, point format 6, EPSG:25833 WKT, extra dims `treeID` int32 (-1 none),
   `semantic` uint8 (255 no vote), `score` float32; the ALS classification restored).
7. `trees_to_gpkg`: one point row per tree (`tree_id, x, y, top_z, height,
   crown_area_m2, n_points, mean_score`) in `<out>/<stem>_trees.gpkg`, height measured
   against a 1 m ground grid.
8. `report`: `<out>/<stem>_report.json/.md` with the model-vs-ALS ground agreement, the
   CHM local-maxima baseline (`ff3d_geo.baseline`), height statistics and the decision
   rule (`ff3d_geo.report.recommend`: tree count within 50 % of the CHM count and median
   height within 3 m of the CHM median).

Host steps isolate failures per tile; a failed docker step stops the run.

### km tiles

Berlin ALS 2021 tiles are 1 km squares with 23-25 M points, too big for one run. The
production path gives every tree a SEAMLESS id across the whole mosaic, sub-tile borders
and km-tile borders alike, by splitting with a 20 m halo and stitching over the overlap
instead of offsetting each sub-tile's ids into a disjoint range:

- `python -m ff3d_geo split --las <km tile> --out <dir> --buffer 20 --neighbours <km tiles>`
  cuts 100 m sub-tiles in local coordinates named `<prefix>_E<x>_N<y>_100m.las` (about 5 s
  per tile, sparse sub-tiles below 1000 points skipped, point count checked), each carrying
  a 20 m halo of extra points beyond its core. `--neighbours` names whichever of the eight
  surrounding km tiles are on disk, so a sub-tile on the source tile's border can fill its
  halo from the neighbour's own points instead of stopping at the km-tile line; a missing
  neighbour is simply skipped (the halo goes unfilled there). Also written:
  `<dir>/split_manifest.json` and, per sub-tile, `<stem>_ident.npy` (the point identity
  `ff3d_geo.stitch` needs).
- `run` takes all sub-tiles of a km tile as one batch, as before.
- Once every tile of the mosaic has been split and run (however many GPUs that took),
  `python -m ff3d_geo stitch --manifest <split_manifest.json of every split> --results
  <per-tile run --out dirs> --out <dir> --runtime-s <s>` matches instances across every
  adjacent sub-tile pair by their IoU over the shared halo points and unions the matches,
  so the SAME physical tree gets the SAME id in every sub-tile (and every km tile) that saw
  it. It rewrites each source km tile that owns at least one sub-tile core as one seamless
  `<T>.las` / `<T>_trees.gpkg` / `<T>_report.json/.md` (a km tile that only supplied halo
  points to a neighbour is not written). A tree spanning a km-tile border gets one tree-table
  row, in the tile holding most of its points, computed over all of its points. This is a single call over the WHOLE mosaic being
  stitched, not per km tile, so ids stay unique and dense across all of it; benchmark's
  production wrapper is `benchmark/berlin_stitch.sh`, run once all of `berlin_run_gpu.sh`'s
  per-GPU queues have finished.
- `python -m ff3d_geo border-check --las <stitched km tile> --json <out>.json` measures what
  is left of the seams: the strip of unlabelled vegetation along the sub-tile grid lines and
  how many crowns still get cut by them (should be near zero with the halo, unlike the old
  offset-id `merge` path).
- `python -m ff3d_geo masks --las <stitched km tile> --out <dir>` writes the instance GeoTIFF
  (int32 tree id of the highest point per 0.5 m cell, nodata -1), the semantic GeoTIFF
  (uint8 majority class, 255 empty) and the crown polygons (`<T>_crowns.gpkg`).

`python -m ff3d_geo merge` (offset each sub-tile's ids into a disjoint range, no halo) still
exists for a single split without `--buffer`, but is no longer part of the Berlin production
path.

Measured on carrot (H100), pre-halo: 55-60 s per 100 m sub-tile, about 1.6 h per km tile on
one GPU; km tiles run in parallel one per GPU. The profile of where that time goes is in
`docs/benchmarks/2026-09-23-inference-profile.md` (when present). With `--buffer 20` each
sub-tile carries ~2x its core point count (a 140x140 m halo box vs. a 100x100 m core), so
expect roughly **double** the per-sub-tile run time and the `inputs/berlin/sub/` disk
footprint of these pre-halo numbers until the haloed runs are actually measured.

## Where the results are

- carrot: `work_dirs/tegel-r12`, `work_dirs/tegel-r13` (Phase 3 tiles),
  `work_dirs/berlin-<km tile>/` (per-sub-tile run results, `<T>.las`/`_trees.gpkg` output by
  `run --out`), `work_dirs/berlin-mosaic/` (the stitched, seamless-id km tiles and their
  masks/border-check JSON, written by `benchmark/berlin_stitch.sh`), inputs under `inputs/`
  and `inputs/berlin/`.
- Mac: `~/work/hnee/ForestFormer3D_runs/berlin_out/` (copied merged files, reports,
  masks; per-tile subdirectories under `berlin-2021/`).
- Reports in this repo: `docs/benchmarks/2026-09-22-tegel-als.md` (r12/r13),
  `docs/benchmarks/2026-09-22-tegel-berlin-2021.md` (three km tiles around Revier 12),
  `docs/benchmarks/2026-09-22-carrot-ff3d.md` (benchmark old vs fixed code).
