# ForestFormer3D: review fixes, H100 benchmark, Tegel ALS geospatial inference

Date: 2026-09-22
Branch: `fix/review-findings` (from `main` at 6a75c37)
Status: approved design, ready for implementation planning

## 1. Goal

Turn the ForestFormer3D research checkout into something that can be run and trusted on the
H100 server `carrot`, prove it with a short benchmark against the released checkpoint, and
run the released model on the Berlin Tegel airborne LiDAR (ALS) tile with georeferenced
outputs.

Four phases, in order. Phases 2 and 3 both depend on 0 and 1.

| Phase | Deliverable | Verified by |
|-------|-------------|-------------|
| 0 Environment | CUDA 11.8 / PyTorch 2.0.1 Docker image that runs on H100 without manual post-install steps | container smoke test: one train iteration, one inference on a synthetic cylinder |
| 1 Fixes | The ~20 output-affecting and reproducibility findings from the 2026-09-22 review, each with a test | `pytest` on the Mac (CPU) plus GPU-marked tests in the container |
| 2 Benchmark | Two tables: released checkpoint old vs fixed inference; 200-epoch old vs fixed training | `docs/benchmarks/<date>-carrot-ff3d.md` with numbers, logs kept in `work_dirs` |
| 3 Tegel | `ff3d_geo` package and CLI; LAS + GeoPackage outputs for the Tegel and Spandau tiles; plausibility report | `pytest` for the package; QGIS-openable outputs; report in `docs/benchmarks/<date>-tegel-als.md` |

## 2. Facts the design relies on

- Repo: OneFormer3D fork on mmengine 0.7.3, mmdet 3.0.0, mmdet3d commit 22aaa47, mmcv 2.0.0,
  PyTorch 1.13.1+cu116. Active config `configs/oneformer3d_qs_radius16_qp300_2many.py`, model
  `ForAINetV2OneFormer3D_XAwarequery` in `oneformer3d/oneformer3d.py`.
- Dataset ForAINetV2: 46 train, 15 val, 27 test plots (`data/ForAINetV2/meta_data/*.txt`),
  downloaded from Zenodo record 16742708 together with `epoch_3000_fix.pth`.
- carrot: H100, 224 cores, Docker with NVIDIA runtime, reached with `ssh carrot`. Not
  resolvable from the Mac at design time (VPN or SSH alias needed). Working root on carrot:
  `/raid/cwinkelmann/ForestFormer3D` (checkout, with `data/ForAINetV2` and `work_dirs`
  inside it, matching the repo layout).
- nvcc 11.6 cannot target compute 9.0. The current Dockerfile builds MinkowskiEngine for
  8.0 only. A CUDA 11.8 toolchain is the first version that supports H100.
- Tegel tile `~/work/hnee/ForestFormer3D_runs/berlin_in/r12_tegel_E381300_N5828300_100m.las`:
  LAS 1.2, point format 1, 192,814 points, about 19 pts/m2, coordinates already local
  (0..100 m, z 0..37 m), no CRS VLR, ALS classes 2 (ground) and 3/4/5 (vegetation), up to 7
  returns. Georeference exists only in the filename: lower-left corner E 381300, N 5828300 in
  ETRS89 / UTM 33N (EPSG:25833). A Spandau tile with the same layout sits next to it.
- `~/work/hnee/FF3D_inference/ff3d_forestsens` is the authors' inference-only variant: same
  environment, LAS/LAZ input, `entrypoint_ff3d.sh` applies the site-packages patches,
  `save_las` in `tools/merge_prediction.py` keeps offsets but writes no CRS.

## 3. Phase 0: environment

### 3.1 New Dockerfile

`Dockerfile` is replaced. The current file is renamed `Dockerfile.a100-cu116` and kept
unchanged for reference.

Stack, all pinned:

| Component | Version |
|-----------|---------|
| base | `pytorch/pytorch:2.0.1-cuda11.8-cudnn8-devel` |
| mmcv | 2.0.1, cu118 / torch2.0 wheel index |
| mmengine | 0.7.3 |
| mmdet | 3.0.0 |
| mmdet3d | git commit 22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61 (unchanged) |
| spconv | `spconv-cu118==2.3.6` |
| MinkowskiEngine | commit 02fc608 (unchanged), `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"`, openblas, `--force_cuda` |
| torch-scatter | 2.1.1 built from source, `FORCE_CUDA=1`, same arch list |
| torch-cluster | 1.6.1 built from source, `FORCE_CUDA=1`, same arch list |
| torch-points-kernels | 0.7.0 built from source, `FORCE_CUDA=1`, same arch list |
| segmentator | Karbo123 commit 76efe46 built as today |
| extras | `laspy[lazrs]`, `tqdm`, `pytest`, existing numeric pins carried over where still compatible with numpy 1.24 |

Rules: no `--install-option` (removed in pip 23.1; MinkowskiEngine gets its flags through
`setup.py` environment variables or a `pip install .` after `git clone`), `pip uninstall -y`,
no `apt-key adv`, no `nvidia-utils` driver packages in the image, no infinite-sleep `CMD`
in the middle of the file. The image ends with `ENTRYPOINT ["/workspace/docker/entrypoint.sh"]`.

### 3.2 Entrypoint and patches

`docker/entrypoint.sh`: copies `replace_mmdetection_files/transforms_3d.py` into the
mmdet3d site-packages (only patch that remains, see 4.1), verifies the CUDA extensions
import (`torch_points_kernels.instance_iou`, `spconv`, `MinkowskiEngine`), then `exec "$@"`.

### 3.3 Smoke test

`tests/gpu/test_smoke.py` (marked `gpu`): builds the model from the active config, runs one
`loss()` step on a synthetic 5,000-point cylinder with two fake trees, and one `predict()`
in full-plot mode on the same points, asserting finite loss and a non-empty instance map.
`docker/smoke.sh` runs it inside the container. This is the Phase 0 gate on carrot.

## 4. Phase 1: fixes

Every item below lands as a failing test first, then the fix. CPU-testable items live in
`tests/`, GPU-only ones in `tests/gpu/`. Findings reference the 2026-09-22 review.

### 4.1 Remove the mmengine patches

The model reads the epoch with `MessageHub.get_current_instance().get_info('epoch')`
inside `loss()` instead of an `epoch` kwarg. `replace_mmdetection_files/loops.py` and
`base_model.py` are deleted; README step 4 shrinks to the one mmdet3d file. This also
removes the TypeError that made `tools/dist_train.sh` unusable.
Test: `loss()` called without `epoch` selects the warm-up branch when the hub reports an
epoch below `prepare_epoch` and the full branch above it.

### 4.2 Inference: `oneformer3d/tiling.py`

Extract from `predict()` into a pure-torch module with no mmengine dependency:

- `generate_cylindrical_regions(points_xy, radius, step)`.
- `sample_region(points, indices, num_points, generator)`: handles `<`, `==`, `>` the cap
  (fixes the unbound-variable case at exactly 640,000 points).
- `merge_instances_by_score(masks, scores, overlap_threshold)`: visits masks by descending
  score; a lower-scoring mask never overwrites points already assigned, it only claims
  unassigned points, and is dropped entirely if its overlap with assigned points exceeds
  the threshold (fixes the overwrite finding).
- `vote_semantics(...)`: binary-head votes and three-class votes are kept in separate
  label spaces; near-empty regions no longer inject "foreground" as "wood".

`predict()` becomes a thin orchestration that calls these, and:

- Full-plot mode is selected by `test_cfg.full_plot: bool` (default `True` for
  `test_cfg`, config sets it explicitly), replacing `'test' in lidar_path`.
- GT arrays are read only if present in `eval_ann_info`; unlabeled inputs work.
- `pred_pts_seg` is set to the merged full-plot semantic and instance arrays so
  `UnifiedSegMetric` scores the same thing that is written to the `.ply`.
- The per-tile `pred_pan_sem` pass and the timing prints are removed.
- `score_th` (instance score) and `overlap_threshold` (0.3 today) are both `test_cfg` keys;
  the misnamed `score_th2` disappears.
- `keep_inds` is defined on both branches of the `nms` switch.
- Zero predicted foreground in a training crop skips query sampling for that sample
  instead of dividing by zero.
- `prepare_epoch=0` means "no warm-up" (`is not None` check).

Tests (CPU): synthetic two-tree overlap for the merge; exact-cap sampling; region grid
coverage; vote label spaces.

### 4.3 Metrics and transforms

- `unified_metric.py`: instance ids after the crop pipeline start at 0 and ground is -1;
  the `g == 0` skip is removed, -1 is skipped, and a scene with GT trees but no predictions
  contributes zero coverage instead of being skipped. Unused constructor arguments
  (`min_num_points`, `id_offset`, `sem_mapping`, `inst_mapping`, `metric_meta`) are removed
  from the class and the config.
- `transforms_3d.py`: `PointSample_` samples without replacement; `GridSample` and
  `PointSample_` remap `ratio_inspoint` together with the instance ids they compact;
  `CylinderCrop` maps raw `treeID == 0` on vegetation points to ignore (-1 instance, and
  excluded from the semantic loss) instead of forming one instance.
- `filter_stuff_masks` / `get_iou_with_crop` consume the remapped ratios by id, not by
  position.

Tests (CPU): metric on a hand-built 3-tree scene where ids start at 0; coverage with zero
predictions; sampling uniqueness; ratio dict survives a voxel drop of a one-point tree.

### 4.4 Tooling

- `tools/test.py`: no in-memory weight permutation. `tools/fix_spconv_checkpoint.py` stays
  the one place that converts, and it refuses to convert a checkpoint whose weights already
  have the converted shape. `test_cfg.output_dir` defaults to `cfg.work_dir` but
  `--cfg-options model.test_cfg.output_dir=...` wins.
- `tools/final_eval.py`: binary-semantic and stuff accumulators are initialised once and
  accumulated across files; empty input directory exits with a clear message.
- `tools/prepare_safe_testfile_names.sh`: strips the extension before matching so files are
  renamed together with the list.
- `data/ForAINetV2/batch_load_ForAINetV2_data.py` and `load_forainetv2_data.py`: an
  `--unlabeled` flag writes constant labels for files without `semantic_seg`/`treeID`;
  `--loader fast` and the unused `torch`, `segmentator`, `open3d`, `Delaunay` imports are
  removed; `test_mode` is honoured by `tools/create_data_forainetv2.py` so a run with only
  `test_data` works.
- `tools/inference_bluepoint.sh`: shebang, `set -euo pipefail`, paths and thresholds as
  variables or `--cfg-options`, no `sed` on the tracked config.
- `tools/update_infos_to_v2.py`: `classes` is a tuple.
- `Dockerfile` (Phase 0) already covers the hidden dependencies.

Tests: `final_eval` on two tiny synthetic result files gives the summed totals; rename
script on a temp dir; `fix_spconv_checkpoint` refuses a double conversion; `batch_load
--unlabeled` on a label-free PLY produces the `.npy` set.

### 4.5 Hygiene in scope

Delete `oneformer3d/oneformer3d_speedup_v1.py`, `oneformer3d_withoutspeedup.py`,
`oneformer3d/mink_unet.py` and its import (SpConvUNet is the only backbone used),
`tools/merge_prediction_slow.py`, `tools/copy_predictions.py`,
`data/ForAINetV2/second_inference.py`, `data/ForAINetV2/compare_outputs.py`,
`data/ForAINetV2/run_and_compare.sh`, `data/ForAINetV2/plyutils.py` (import from
`tools/plyutils.py`), `segmentator/test_equivariance.py`, and the 22 tracked `.pyc` files.
`segment_any_tree_gpu.docker` is unstaged and not committed. `README.md` is updated to the
corrected commands and CLAUDE.md is committed.

Out of scope for Phase 1: the aux-layer dice scaling inherited from OneFormer3D, the
`gt_labels_3d` first-point label, nondeterministic `index_copy_` ties, the Python-loop
`save_ply_withscore` (rewritten with structured numpy only if it shows up in the benchmark
timings), and semantic mIoU semantics in `UnifiedSegMetric`. They are listed in
`docs/known-issues.md`.

## 5. Phase 2: benchmark on carrot

Layout on carrot: `/raid/cwinkelmann/ForestFormer3D` is a clone of this branch.
`data/ForAINetV2/{train_val_data,test_data}` and `work_dirs/clean_forestformer/epoch_3000_fix.pth`
come from Zenodo. `work_dirs/old-main` is a `git worktree` of `main` (6a75c37) used for
the old-code runs, executed in the same new image so only code differs.

Scripts in `benchmark/`:

- `fetch_zenodo.sh`: downloads and unpacks record 16742708 into the layout above, idempotent.
- `run_release_eval.sh`: preprocesses the test split once, then runs `tools/test.py` with
  `epoch_3000` (converted once with `fix_spconv_checkpoint.py` from the raw checkpoint if
  Zenodo ships a raw one, otherwise used as is) for old and fixed code, and scores both
  output directories with the fixed `final_eval.py`. Old code needs its own preprocessing
  because its `test.py` re-permutes weights; the script feeds it the raw checkpoint.
- `run_train_200.sh <old|fixed>`: 200 epochs, `val_interval=20`, `max_keep_ckpts=2`, under
  `nohup` with `work_dirs/bench-<old|fixed>-200/train.log`. Reports the last-epoch test F1
  through `run_release_eval.sh`'s scoring path.
- `collect.py`: reads the two `evaluation_total_test.txt` files and the mmengine
  `scalars.json` logs and writes `docs/benchmarks/<date>-carrot-ff3d.md` with the two tables
  and the wall-clock per epoch.

Success criteria: the released checkpoint's fixed-inference F1 is at least the old
inference's F1 on the same outputs (the merge fix should not reduce it); the fixed 200-epoch
run trains without NaN or crash and its val curve is reported next to the old one. No
target number is promised for the 200-epoch runs; they are a sanity comparison.

## 6. Phase 3: Tegel ALS geospatial inference

### 6.1 Package `ff3d_geo/`

Pure Python (numpy, laspy, shapely, pyproj), no CUDA, tested on the Mac.

- `origin.py`: `parse_origin(filename) -> (easting, northing)` from `E<int>_N<int>`;
  `--origin E N` overrides; error if neither.
- `convert.py`:
  - `las_to_ply(las_path, ply_path, sidecar_path, origin, epsg=25833)`: writes `x y z`
    as float64 in local coordinates (the pipeline's `.ply` reader expects local, small
    values), constant `semantic_seg=1`, `treeID=0`, and a JSON sidecar with origin, EPSG,
    scale/offset of the source, and the ALS `classification` array saved as `.npy` next to it.
  - `results_to_las(result_ply, sidecar, out_las)`: LAS 1.4, point format 6, coordinates
    restored to UTM with the source scale, extra dimensions `treeID` (int32, -1 = none),
    `semantic` (uint8: 0 ground, 1 wood, 2 leaf), `score` (float32), original ALS
    `classification` kept, CRS written as WKT VLR for the EPSG.
- `trees.py`: `trees_to_gpkg(out_las, gpkg_path)`: one row per `treeID >= 0`: `tree_id`,
  stem `x y` (median of the lowest 1 m of the tree's points), `top_z`, `height` (top minus
  the ground surface, from a 1 m grid of model-ground or ALS class-2 minimum z, whichever
  has points), `crown_area_m2` (2D convex hull), `n_points`, `mean_score`. Layer `trees`,
  CRS from the sidecar.
- `baseline.py`: CHM local-maxima tree count from the same tile (1 m CHM from ALS
  classes 3-5 minus class-2 DTM, 3 m window, min height 3 m) as the plausibility reference.
- `cli.py`: `python -m ff3d_geo run --las <file> --checkpoint <pth> --out <dir>
  [--origin E N] [--epsg 25833] [--config ...]`: convert, place in `data/ForAINetV2/test_data`,
  write `meta_data/test_list.txt`, run `batch_load --unlabeled` and `create_data`, run
  `tools/test.py --work-dir <out>`, then `results_to_las` and `trees_to_gpkg`. Also
  `python -m ff3d_geo convert ...` and `python -m ff3d_geo report ...` for the parts.

### 6.2 Run and report

Both tiles run on carrot with the released checkpoint. `docs/benchmarks/<date>-tegel-als.md`
reports per tile: point count, runtime, tree count vs CHM baseline, height distribution
(min, median, max), ground-vs-vegetation agreement between model semantics and ALS
classes, and a QGIS screenshot of the GeoPackage over the LAS. The report ends with a
recommendation on whether ALS fine-tuning is worth pursuing. Fine-tuning itself is out of
scope.

Tests (CPU): origin parsing; LAS round trip preserves coordinates to 1 cm and the CRS; a
synthetic result PLY with two trees yields two GeoPackage rows with the expected heights
and areas; CHM baseline finds two peaks in a synthetic two-cone tile.

## 7. Testing and tooling summary

- `pytest` at the repo root, `tests/` (CPU) and `tests/gpu/` (`-m gpu`, container only).
- `pyproject.toml` with `[tool.pytest.ini_options]` markers and `ff3d_geo` as an installable
  package (`pip install -e .`), no other packaging changes.
- No CI.

## 8. Risks

- CUDA 11.8 migration: mmcv 2.0.1 and spconv 2.3.6 have cu118 wheels; MinkowskiEngine at
  the pinned commit compiles under CUDA 11.8 with gcc 9 to 11. If MinkowskiEngine fails to
  build for 9.0, fall back to `8.0;8.6;8.9` plus PTX and note it in the benchmark doc.
- The released checkpoint may already be permuted (its name ends in `_fix`). The
  shape-aware converter in 4.4 detects this; the benchmark script does not assume either way.
- ALS density is roughly two orders of magnitude below the training data. Poor Tegel
  results are an expected outcome and still a valid deliverable; the report is the product.
- carrot is not reachable from the Mac at design time; all carrot steps need VPN or an SSH
  alias and are planned as the last tasks of Phases 0, 2 and 3.
