# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ForestFormer3D (ICCV 2025 oral): end-to-end semantic + instance segmentation of forest LiDAR point clouds (classes `ground`, `wood`, `leaf`; instances = individual trees). It is a fork of OneFormer3D built on the OpenMMLab stack (mmengine 0.7.3, mmdet 3.0.0, mmdet3d at a pinned commit, mmcv 2.0.1, PyTorch 2.0.1 / CUDA 11.8). Everything is registry-driven: configs name classes by string and mmengine builds them. License is CC BY-NC 4.0.

Tests live in `tests/` (CPU) and `tests/gpu/` (container). There is no linter or CI config in
the repo.

## Environment

The image is `forestformer3d:cu118` (`Dockerfile`): `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` plus the torch 2.0.1 / torchvision 0.15.2 `+cu118` wheels, with MinkowskiEngine, spconv, torch-scatter, torch-points-kernels, torch-cluster, and the `segmentator` C++ extension built for compute 8.0/8.6/8.9/9.0. The previous CUDA 11.6 image is kept as `Dockerfile.a100-cu116` (see the README's legacy section). Inside the container the repo is mounted at `/workspace` and `PYTHONPATH=/workspace` must be set (the Dockerfile's `ENV` sets it at build time; `docker/entrypoint.sh` itself never touches `PYTHONPATH` — it only applies the `transforms_3d.py` patch and links the segmentator build, see below). `tools/inference_bluepoint.sh` defaults `WORK_DIR` to the repo root it lives in, overridable with `WORK_DIR=`.

`docker/entrypoint.sh` overwrites one upstream file in site-packages on every container start: `replace_mmdetection_files/transforms_3d.py` → `mmdet3d/datasets/transforms/`. It makes flip/rotate/scale also transform the per-point `vote_label` offsets that this repo's crop/sample transforms create. No mmengine files are patched: the model reads the training epoch from `mmengine.logging.MessageHub` (`current_epoch_from_hub()` in `oneformer3d/oneformer3d.py`), so `tools/dist_train.sh` works.

### Testing

- Mac / no GPU: `python3 -m pytest tests` at the repo root runs the pure-Python CPU tests
  (tiling math, checkpoint converter, loader, `runner_options`, hygiene checks). A few tests
  need `plyfile`/`scipy`/`laspy`/`pytest` that aren't part of the base install; create
  `.venv-cpu` once (`python3 -m venv .venv-cpu && .venv-cpu/bin/pip install -r
  tests/requirements-cpu.txt`) and run `.venv-cpu/bin/python -m pytest -q tests` — tests using
  those packages skip cleanly without it.
- In the container: `pytest -m gpu tests/gpu` runs the GPU tests (model construction, tiling
  end to end, a full smoke scenario). `docker/smoke.sh` runs a one-loss-step + one-full-plot
  smoke test on a synthetic plot and is the quickest way to check a fresh image/GPU host.
- `docs/known-issues.md` lists findings that were deliberately left unfixed; read it before
  treating any of them as a new bug.

## Common commands

```bash
# 1. Raw .ply -> per-scan .npy (run from data/ForAINetV2). Uses meta_data/{train,val,test}_list.txt.
cd data/ForAINetV2 && python batch_load_ForAINetV2_data.py && cd ../..
# 2. .npy -> mmdet3d info pkls (forainetv2_oneformer3d_infos_{train,val,test}.pkl)
python tools/create_data_forainetv2.py forainetv2

# Train (single A100; reduce `radius` in the config on smaller GPUs)
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/oneformer3d_qs_radius16_qp300_2many.py --work-dir work_dirs/<name>
# Multi-GPU
bash tools/dist_train.sh configs/oneformer3d_qs_radius16_qp300_2many.py <num_gpus>

# Inference on everything in meta_data/test_list.txt; writes <scan>.ply result files into --work-dir
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/oneformer3d_qs_radius16_qp300_2many.py work_dirs/<name>/epoch_N.pth \
  --work-dir work_dirs/<output_dir>

# Two-pass inference for dense plots (see "Bluepoint iteration" below)
bash tools/inference_bluepoint.sh

# Offline F1/eval over a directory of result .ply files (writes a fresh evaluation_total_test.txt there each run)
python tools/final_eval.py work_dirs/<output_dir>

tensorboard --logdir=work_dirs/<name>/vis_data/ --host=0.0.0.0 --port=6006
```

`tools/test.py` loads the checkpoint as given: convert a freshly trained checkpoint once with `tools/fix_spconv_checkpoint.py` (it exits 2 on an already converted file). Result `.ply` files go to `--work-dir`, or to `--cfg-options model.test_cfg.output_dir=...` when that is set. Unlabeled scans: `batch_load_ForAINetV2_data.py --unlabeled`. Tests: `python3 -m pytest tests` on the Mac (CPU), `pytest -m gpu tests/gpu` inside the container.

Pretrained weights and the ForAINetV2 dataset come from Zenodo (README) and are expected at `work_dirs/clean_forestformer/epoch_3000_fix.pth` and `data/ForAINetV2/{train_val_data,test_data}`. Neither `work_dirs/` nor the data are in git.

## Architecture

### Registry wiring
`configs/*.py` set `custom_imports = dict(imports=['oneformer3d'])`, so a class is only usable from a config if `oneformer3d/__init__.py` imports it. When adding a new `@MODELS/@TRANSFORMS/@DATASETS/@METRICS.register_module()` class, add it to `__init__.py` or the config will fail with "not in registry". Names ending in `_` (`Pack3DDetInputs_`, `PointSample_`, `LoadAnnotations3D_`, `ForAINetV2SegDataset_`, `Det3DDataPreprocessor_`, `InstanceData_`) are patched subclasses of the mmdet3d originals.

### The active config
`configs/oneformer3d_qs_radius16_qp300_2many.py` is the one used in the paper and by every script. It builds `ForAINetV2OneFormer3D_XAwarequery` with one-to-many matching (`InstanceCriterionForAI_OneToManyMatch`). `configs/oneformer3d_radius16_qp300.py` is the older `ForAINetV2OneFormer3D` variant. There is no top-level `score_th` any more: thresholds live under `model.test_cfg` — `score_th` (0.4, per-tile instance score threshold), `inst_score_thr`, `overlap_threshold` (0.3, drop a mask when this fraction of its points is already assigned), `full_plot` (True; tiled whole-plot inference — `tools/train.py` forces it to `False` for validation). Override any of them without editing the file via `--cfg-options model.test_cfg.score_th=...`. Other knobs: top-level `radius` (cylinder crop radius, 16 m), `chunk = 20_000` (defined at the top of the config but not threaded into `model=dict(...)`; see `docs/known-issues.md`), `voxel_size=0.2`, `query_point_num=300`. Training runs 3000 epochs with val every 100, AdamW + PolyLR, EpochBasedTrainLoop.

### Model (`oneformer3d/oneformer3d.py`)
The file is about 3900 lines and contains several models; only `ForAINetV2OneFormer3D` and `ForAINetV2OneFormer3D_XAwarequery` (the latter starts around line 1744) matter here. The ScanNet/S3DIS classes are inherited from OneFormer3D and unused. Flow:

- `SpConvUNet` (`spconv_unet.py`) voxelizes at 0.2 m and extracts per-point features.
- `ForAINetv2QueryDecoder_XAwarequery` (`query_decoder.py`) runs 6 transformer layers over `query_point_num` sampled query points plus 3 semantic queries, producing mask logits + class logits.
- `ForAINetv2UnifiedCriterion_XAwarequery` (`unified_criterion.py`) sums the instance criterion (`instance_criterion.py`: Hungarian/one-to-many matching with `QueryClassificationCost`, `MaskBCECost`, `MaskDiceCost`) and the semantic criterion.
- `predict()` checks `self.test_cfg.get('full_plot', True)`; when true it dispatches to `_predict_full_plot`, which tiles the whole plot with `oneformer3d/tiling.py`'s `generate_cylindrical_regions` (step `radius/4`), runs each cylinder through `sample_region` capped at `max_points = 640_000` (a local constant, not config-driven) and `predict_by_feat_test` to get per-tile masks and scores, merges the overlapping per-tile instance masks across tiles by score with `tiling.merge_instances_by_score` (threshold from `model.test_cfg.overlap_threshold`), resolves per-point semantics via `tiling.SemanticVotes`, sets ground to instance `-1`, and writes `<scan>.ply` via `save_ply_withscore` into `self.test_cfg['output_dir']`, which `tools/test.py` sets from `--work-dir` (or `--cfg-options model.test_cfg.output_dir=...`). `tools/train.py` forces `full_plot=False` for validation so it runs the non-tiled path instead. `save_bluepoints` exists on the class but has no call site (`docs/known-issues.md`).

### Data pipeline
Raw plots are `.ply` files with fields `x, y, z, semantic_seg, treeID` (`semantic_seg` is 1-based and shifted by −1 on load; `treeID == 0` means unannotated).

1. `data/ForAINetV2/batch_load_ForAINetV2_data.py` reads the scan names from `meta_data/*_list.txt`, calls `load_forainetv2_data.export()` per scan, computes superpoints with `segmentator`, and writes `forainetv2_instance_data/<scan>_vert.npy`, `_sem_label.npy`, `_ins_label.npy`, bbox and offset arrays. It skips a scan whose `_vert.npy` already exists, so delete `forainetv2_instance_data/` before regenerating after a loader change. `--unlabeled` gives every scan whose PLY has no `semantic_seg`/`treeID` fields constant labels (semantic 0, instance -1) instead of failing; without it a scan missing those fields aborts with a message naming `--unlabeled`. There is no `--loader fast` any more; `load_forainetv2_data.py` imports `read_ply` from `tools/plyutils.py` (via a `sys.path` insertion of the repo root, so it works whether the script runs from the repo root or with cwd `data/ForAINetV2`).
2. `tools/create_data_forainetv2.py` → `converter_forainetv2.create_info_file` → `forainetv2_data_utils.ForAINetV2Data` builds the info pkls, then `update_infos_to_v2.update_forainetv2_infos` converts them to the mmdet3d v2 schema. By default it reads `meta_data/{train,val,test}_list.txt` and writes the pkls into `data/ForAINetV2`; works when only `test_data/` exists, since a split with no scans in its list is simply not built (no message). `--test-list PATH` overrides the test split's scan list, `--splits test` builds only that split (train/val are skipped the same silent way), and `--out-dir DIR` writes the pkl(s) elsewhere instead of `data/ForAINetV2` — this is how `ff3d_geo` (below) points a private test pkl at its own `--out` dir without touching the tracked lists or the benchmark's pkls.
3. `ForAINetV2SegDataset_` (`forainetv2_dataset.py`) subclasses mmdet3d's `ScanNetDataset` and reads those pkls.
4. Train pipeline (`transforms_3d.py`): `CylinderCrop` (random cylinder of `radius`) → `GridSample(0.2)` → `PointSample_(640000)` → `SkipEmptyScene_` → `PointInstClassMapping_` → flip/rotate/scale → `Pack3DDetInputs_`. The test pipeline has no cropping; the model tiles internally.

Test file base names ending in `_<digits>` break name handling; `tools/prepare_safe_testfile_names.sh` appends a `fixedname` tag to both the list and the files.

### Bluepoint iteration (`tools/inference_bluepoint.sh`)
Second-pass inference on points left unassigned by the first pass. It is entirely env-driven (`WORK_DIR`, `CONFIG_FILE`, `MODEL_PATH`, `ITERATIONS`, `SCORE_TH`, `BLUEPOINTS_DIR`, `DATA_ROOT`, `TEST_LIST_INIT`, `TEST_LIST`, `TEST_DATA_DIR`, `DRY_RUN=1` to print commands instead of running them) and never edits tracked files — it writes `test_list.txt` to a private temp file and passes thresholds to `tools/test.py` via `--cfg-options model.test_cfg.score_th=... model.test_cfg.output_dir=...` instead of `sed`-ing the config. Per scan in `meta_data/test_list_initial.txt` it reruns steps 1–2 of the data pipeline for just that scan, runs `tools/test.py`, looks for `<scan>_bluepoints_<i>.ply` in `BLUEPOINTS_DIR`, copies it into `test_data/` and repeats up to `ITERATIONS`; then `tools/merge_prediction.py` merges rounds and `tools/final_eval.py` scores each `round_<i>` directory. It still depends on `predict()` calling `save_bluepoints` for the remaining points, which nothing does today (`docs/known-issues.md`), so iteration 2+ currently has nothing new to pick up.

### Evaluation
`UnifiedSegMetric` (`unified_metric.py`, stuff=`[0]`, things=`[1,2]`) runs during validation. `tools/final_eval.py` is the standalone per-region F1 evaluator over result `.ply` files; each run overwrites `evaluation_total_test.txt` in the target directory with a single fresh result block. `tools/merge_prediction.py` hardcodes the container path `/workspace/data/ForAINetV2/forainetv2_instance_data` (for the per-scan `_offsets.npy` used when writing the merged `.las`); edit before use outside that layout.

### Geospatial inference (`ff3d_geo`)
`ff3d_geo/` is a separate, pure-Python package (no torch/CUDA) for running inference on real georeferenced ALS tiles (LAS/LAZ), as opposed to the ForAINetV2 benchmark plots. Install it with the `geo` extra in `pyproject.toml` (`pip install -e .[geo]`; `tests/requirements-cpu.txt` installs the same packages, so `.venv-cpu` already has it). `python -m ff3d_geo run --las <file> --checkpoint <pth> --out <dir> --gpu <N>` runs on the HOST in a plain CPU venv: it converts the LAS to a PLY plus a JSON georeferencing sidecar, delegates the two GPU steps (`batch_load_ForAINetV2_data.py` + `create_data_forainetv2.py`, then `tools/test.py`) to the `forestformer3d:cu118` container through `benchmark/common.sh`'s `ff3d_docker` helper, then georeferences the result back to LAS 1.4 and writes a tree GeoPackage plus a markdown/JSON report. It never reads or writes the tracked `data/ForAINetV2/meta_data/test_list.txt`: it writes its own private scan list into `--out` and points `create_data_forainetv2.py` at it with `--test-list`/`--splits test`/`--out-dir` (see the data pipeline step 2 note above). `--dry-run` prints the 8-step plan and touches nothing. `--las` accepts several tiles: one preprocess and one inference step cover the whole batch (one scan list with a line per tile) while the host-side steps still run per tile, so each tile keeps its own `<out>/<stem>.las`, `<stem>_trees.gpkg` and `<stem>_report.json/.md`. Failures are handled per step kind (`execute` in `ff3d_geo/cli.py`): the two docker steps stay fail-fast, while a host step that raises is recorded and the remaining host steps run anyway, so one bad tile in `results_to_las` costs neither the other tiles their LAS (`_run_per_tile` finishes the step first) nor the healthy tiles their GeoPackage and report; at the end one `RuntimeError` names every failed step. Two more subcommands bracket that batch for tiles larger than the ~100 m the model is trained on: `split --las <km tile> --out <dir>` (`ff3d_geo/split.py`) cuts a km tile into grid-aligned `<prefix>_E<x>_N<y>_100m.las` sub-tiles in local coordinates and prints their paths, and `merge --las <results>... --gpkg <gpkgs>... --out-las ... --out-gpkg ... [--report-json ...]` (`ff3d_geo/merge.py`) stitches them back with globally unique `treeID`s. `merge` applies the LAS id offsets to the GeoPackages POSITIONALLY, so the two lists must name the same sub-tiles in the same order; the CLI enforces that by checking `<stem>.las` against `<stem>_trees.gpkg` pairwise. `masks --las <result las> --out <dir> [--cell 0.5] [--prefix P]` (`ff3d_geo/raster.py`) is the optional last step: it reads a (merged) result LAS once and writes `<prefix>_instance_<tag>.tif` (int32, the `treeID` of the highest point per cell, nodata -1, and 0 is a valid id), `<prefix>_semantic_<tag>.tif` (uint8 majority class per cell, 255 where nothing voted, ties broken toward the higher class index) and `<prefix>_crowns.gpkg` layer `crowns` (one convex hull per tree; degenerate trees get a `cell_m` square flagged by `hull_is_point`). Both GeoTIFFs are EPSG:25833, north-up, LZW+tiled, on an extent snapped outward to a global `cell_m` lattice so neighbouring tiles line up; `_cell_tag` names the cell size `50cm`/`1m`. Binning is vectorised numpy (one `np.lexsort`, one `np.bincount`), so a 25 M point km tile takes ~12 s and ~2 GB. A LAS whose CRS has no EPSG code is refused (the two outputs could not then share a CRS); a LAS with no tree at all is fine and yields an all-nodata instance raster plus an empty crowns layer. Crown ids match `<T>_trees.gpkg`'s `tree_id`, but the instance raster's id set is a strict SUBSET (~4 % of trees are topped everywhere by a taller neighbour). `docs/benchmarks/RUNBOOK-tegel.md` is the full walkthrough (copying tiles to a GPU host, running, the km-tile split/batch/merge loop, filling in a report).

## OOM guidance (from README)
Training: lower `radius` in the config. Inference is per-cylinder with no batching: lower `chunk` (constructor default on the model; the config's top-level `chunk` is not wired to it, see `docs/known-issues.md`), lower the `max_points = 640_000` cap inside `_predict_full_plot` (`oneformer3d/oneformer3d.py`), or lower `radius`.

`docs/known-issues.md` collects findings from the fix pass that were deliberately left alone; check it before assuming an oddity is new.

---

# context-mode — MANDATORY routing rules

You have context-mode MCP tools available. These rules are NOT optional — they protect your context window from flooding. A single unrouted command can dump 56 KB into context and waste the entire session.

## BLOCKED commands — do NOT attempt these

### curl / wget — BLOCKED
Any Bash command containing `curl` or `wget` is intercepted and replaced with an error message. Do NOT retry.
Instead use:
- `ctx_fetch_and_index(url, source)` to fetch and index web pages
- `ctx_execute(language: "javascript", code: "const r = await fetch(...)")` to run HTTP calls in sandbox

### Inline HTTP — BLOCKED
Any Bash command containing `fetch('http`, `requests.get(`, `requests.post(`, `http.get(`, or `http.request(` is intercepted and replaced with an error message. Do NOT retry with Bash.
Instead use:
- `ctx_execute(language, code)` to run HTTP calls in sandbox — only stdout enters context

### WebFetch — BLOCKED
WebFetch calls are denied entirely. The URL is extracted and you are told to use `ctx_fetch_and_index` instead.
Instead use:
- `ctx_fetch_and_index(url, source)` then `ctx_search(queries)` to query the indexed content

## REDIRECTED tools — use sandbox equivalents

### Bash (>20 lines output)
Bash is ONLY for: `git`, `mkdir`, `rm`, `mv`, `cd`, `ls`, `npm install`, `pip install`, and other short-output commands.
For everything else, use:
- `ctx_batch_execute(commands, queries)` — run multiple commands + search in ONE call
- `ctx_execute(language: "shell", code: "...")` — run in sandbox, only stdout enters context

### Read (for analysis)
If you are reading a file to **Edit** it → Read is correct (Edit needs content in context).
If you are reading to **analyze, explore, or summarize** → use `ctx_execute_file(path, language, code)` instead. Only your printed summary enters context. The raw file content stays in the sandbox.

### Grep (large results)
Grep results can flood context. Use `ctx_execute(language: "shell", code: "grep ...")` to run searches in sandbox. Only your printed summary enters context.

## Tool selection hierarchy

1. **GATHER**: `ctx_batch_execute(commands, queries)` — Primary tool. Runs all commands, auto-indexes output, returns search results. ONE call replaces 30+ individual calls.
2. **FOLLOW-UP**: `ctx_search(queries: ["q1", "q2", ...])` — Query indexed content. Pass ALL questions as array in ONE call.
3. **PROCESSING**: `ctx_execute(language, code)` | `ctx_execute_file(path, language, code)` — Sandbox execution. Only stdout enters context.
4. **WEB**: `ctx_fetch_and_index(url, source)` then `ctx_search(queries)` — Fetch, chunk, index, query. Raw HTML never enters context.
5. **INDEX**: `ctx_index(content, source)` — Store content in FTS5 knowledge base for later search.

## Subagent routing

When spawning subagents (Agent/Task tool), the routing block is automatically injected into their prompt. Bash-type subagents are upgraded to general-purpose so they have access to MCP tools. You do NOT need to manually instruct subagents about context-mode.

## Output constraints

- Keep responses under 500 words.
- Write artifacts (code, configs, PRDs) to FILES — never return them as inline text. Return only: file path + 1-line description.
- When indexing content, use descriptive source labels so others can `ctx_search(source: "label")` later.

## ctx commands

| Command | Action |
|---------|--------|
| `ctx stats` | Call the `ctx_stats` MCP tool and display the full output verbatim |
| `ctx doctor` | Call the `ctx_doctor` MCP tool, run the returned shell command, display as checklist |
| `ctx upgrade` | Call the `ctx_upgrade` MCP tool, run the returned shell command, display as checklist |
