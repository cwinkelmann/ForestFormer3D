# ForestFormer3D Fixes, H100 Benchmark and Tegel ALS Inference: Master Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ForestFormer3D checkout runnable and trustworthy on the H100 server carrot, prove it with a two-run benchmark against the released checkpoint, and run the released model on the Berlin Tegel ALS tile with georeferenced LAS and GeoPackage outputs.

**Architecture:** Four sequential phases, each in its own plan file with its own tests and verification gate. Phase 0 rebuilds the Docker image on CUDA 11.8 / PyTorch 2.0.1. Phase 1 fixes the output-affecting review findings behind tests and extracts the tiling and merge logic into `oneformer3d/tiling.py`. Phase 2 runs the benchmark scripts on carrot. Phase 3 adds the pure-Python `ff3d_geo` package and CLI for LAS in, LAS plus GeoPackage out.

**Tech Stack:** PyTorch 2.0.1+cu118, mmengine 0.7.3, mmdet 3.0.0, mmdet3d 22aaa47, mmcv 2.0.1, spconv-cu118 2.3.6, MinkowskiEngine 02fc608, Docker, pytest; laspy, plyfile, shapely, pyproj, geopandas for `ff3d_geo`.

**Spec:** `docs/superpowers/specs/2026-09-22-ff3d-fixes-benchmark-tegel-design.md`

## Global Constraints

- Branch `fix/review-findings` from `main` at 6a75c37; one commit per task; commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Docker image tag `forestformer3d:cu118`, repo mounted at `/workspace`, `PYTHONPATH=/workspace`, entrypoint `/workspace/docker/entrypoint.sh`.
- carrot checkout `/raid/cwinkelmann/ForestFormer3D`; data under `data/ForAINetV2/{train_val_data,test_data,meta_data}`; released checkpoint at `work_dirs/clean_forestformer/epoch_3000_fix.pth`; Zenodo record 16742708.
- The Mac has no torch, mmengine or mmdet3d. CPU tests (`pytest`, default `-m 'not gpu'`) use numpy, plyfile, laspy and friends only; torch-only tests skip when torch is missing; anything needing mmengine/mmdet3d is under `tests/gpu/` with `@pytest.mark.gpu` and runs in the container.
- `MinkowskiEngine` and the from-source CUDA extensions build for `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"`.
- Result PLY contract: `<work_dir>/<scan>.ply` with `x,y,z` (f4, centered coordinates), `semantic_pred` (0 ground, 1 wood, 2 leaf), `instance_pred` (-1 none), `score`, and the per-scan offsets in `data/ForAINetV2/forainetv2_instance_data/<scan>_offsets.npy`.
- Tegel tile georeference: local coordinates plus origin from the filename `E<easting>_N<northing>`, EPSG:25833.

---

## Phase order and gates

Execute the four phase plans in this order. Do not start a phase before the previous gate passes.

### Phase 0: Environment

Plan: `docs/superpowers/plans/2026-09-22-ff3d-00-environment.md`

- [ ] All Phase 0 tasks complete
- [ ] Gate: `docker build -t forestformer3d:cu118 .` succeeds on carrot and `docker/smoke.sh` reports the GPU smoke test passing (one `loss()` step with finite loss, one full-plot `predict()` with a non-empty instance map)

### Phase 1: Fixes

Plan: `docs/superpowers/plans/2026-09-22-ff3d-01-fixes.md`

- [ ] All Phase 1 tasks complete
- [ ] Gate: `pytest` passes on the Mac; `pytest -m gpu tests/gpu` passes inside the container on carrot; `replace_mmdetection_files/` contains only `transforms_3d.py`; `git ls-files '*.pyc'` is empty; `segment_any_tree_gpu.docker` is not tracked

### Phase 2: Benchmark on carrot

Plan: `docs/superpowers/plans/2026-09-22-ff3d-02-benchmark.md`

- [ ] All Phase 2 tasks complete
- [ ] Gate: `docs/benchmarks/<date>-carrot-ff3d.md` exists with both tables filled from real runs; released-checkpoint F1 with fixed inference is at least the old-inference F1; the fixed 200-epoch run finished without NaN or crash

### Phase 3: Tegel ALS geospatial inference

Plan: `docs/superpowers/plans/2026-09-22-ff3d-03-tegel-geo.md`

- [ ] All Phase 3 tasks complete
- [ ] Gate: `python -m ff3d_geo run` produced `<out>/<stem>.las` (LAS 1.4, EPSG:25833 CRS, `treeID`/`semantic`/`score` extra dims) and `<out>/<stem>_trees.gpkg` for both the Tegel and Spandau tiles; `docs/benchmarks/<date>-tegel-als.md` contains the numbers, the QGIS screenshot and the fine-tuning recommendation

## Manual steps that need the user

- carrot is not reachable from the Mac at planning time. Phases 0, 2 and 3 each end with carrot tasks that need VPN or an SSH alias; the runbooks `docs/benchmarks/RUNBOOK-carrot.md` and `docs/benchmarks/RUNBOOK-tegel.md` list the exact commands.
- The QGIS screenshot in Phase 3 is taken by hand.
