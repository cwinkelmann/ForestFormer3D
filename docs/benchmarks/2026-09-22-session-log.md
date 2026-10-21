# Session log, 2026-09-22 to 2026-09-23: fixes, benchmark, Tegel and Berlin inference

What was done on branch `fix/review-findings` (fork `cwinkelmann/ForestFormer3D`), in
order, with the artefacts each step left behind. The plans and the design spec are under
`docs/superpowers/`.

## 0. Environment (Phase 0)

- New `Dockerfile` on `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` with torch 2.0.1+cu118,
  spconv-cu118, MinkowskiEngine, torch-scatter/cluster/points-kernels and `segmentator`
  built for compute 8.0/8.6/8.9/9.0; the old CUDA 11.6 file kept as `Dockerfile.a100-cu116`.
- `docker/entrypoint.sh` (patches one mmdet3d transform file, links the segmentator
  build, asserts CUDA), `docker/smoke.sh`.
- Image `forestformer3d:cu118` built on carrot; GPU test suite `tests/gpu` (25 tests).

## 1. Code fixes (Phase 1)

Output-affecting and reproducibility findings of the initial review, all behind tests:
evaluator scored only the last tile of a full-plot prediction; `tools/test.py` permuted
checkpoints in memory and forced the output dir; metric dropped ground-truth id 0;
tile merge overwrote higher-scoring instances; validation used a different query
selection; epoch read from a patched mmengine loop (now `MessageHub`); `final_eval.py`
per-file globals; dead and duplicate modules removed; tracked `.pyc` files removed;
label conventions centralised in `oneformer3d/labels.py`; tiling logic extracted to
`oneformer3d/tiling.py`. `docs/known-issues.md` lists what was deliberately left.
CPU tests run on the Mac without torch (`.venv-cpu`).

## 2. Benchmark on carrot (Phase 2)

`benchmark/` scripts (shared docker wrappers, Zenodo fetch with cache, old-code worktree
at `main` 6a75c37, release eval, 200-epoch runs with markers, `collect.py`).
Report: `docs/benchmarks/2026-09-22-carrot-ff3d.md`.

- Released checkpoint on the 28 test plots, scored with `final_eval.py` over the result
  PLYs of each variant: old code F1 0.9064, fixed code F1 0.9034 (precision 0.945 -> 0.925,
  recall 0.871 -> 0.883, mIoU 0.852 both). The gate "fixed >= old" fails by 0.003; the
  precision/recall shift is attributed (as a hypothesis) to the per-tile bookkeeping of
  the merge. The original evaluator crashes on full-plot output, so no mmengine metrics
  exist for the old code. The shipped `tools/test.py` also lacked `import torch`; the old
  worktree gets exactly that one line.
- 200-epoch runs (prepare_epoch overridden to 60, otherwise the instance decoder never
  trains in 200 epochs): both finished without non-finite loss; test F1 old 0.834,
  fixed 0.811, single seed each.

## 3. Tegel ALS tiles (Phase 3)

`ff3d_geo` package and CLI (see `docs/inference-pipeline.md`), `docs/benchmarks/RUNBOOK-tegel.md`.
Report `docs/benchmarks/2026-09-22-tegel-als.md`: r12 Tegel 402 trees vs CHM 397,
r13 Spandau 443 vs 400; median heights 3-4 m below the CHM medians -> recommendation to
fine-tune on ALS. QGIS screenshots pending.

## 4. Berlin ALS 2021 km tiles (Phase 4)

`split` / batched `run` / `merge` (`docs/superpowers/plans/2026-09-22-ff3d-04-berlin-batch.md`).
Report `docs/benchmarks/2026-09-22-tegel-berlin-2021.md`: three km tiles around Revier 12
Tegelsee (`3dm_33_380_5828`, `381_5828`, `381_5829`), 300 sub-tiles, 73.5 M points,
105,829 trees, 4 h 47 min on one H100. All three: tree count 6-11 % under the CHM
baseline (passes), median height 4.4-8.6 m low (fails); the gap is a shift of the whole
height distribution, not a tail of short instances.

## 5. Masks

`python -m ff3d_geo masks` writes instance/semantic GeoTIFFs and crown polygons
(`ff3d_geo/raster.py`); produced for the three tiles above on the Mac (about 12 s and
1.9 GB per tile).

## 6. In progress on 2026-09-23

- Tegel block, eight more km tiles (`379_5828`, `379_5829`, `380_5829`, `382_5828`,
  `382_5829`, `381_5830`, `383_5828`, `383_5829`): running on carrot GPUs 0, 2, 3, 4, 5, 6, 7
  in parallel (scripts `work_dirs/logs/berlin-run-gpu.sh`, logs `work_dirs/logs/berlin-gpu*.log`),
  each tile ending with merge + masks under `work_dirs/berlin-<T>/`.
- Inference profile (why 55-60 s per 100 m sub-tile): `docs/benchmarks/2026-09-23-inference-profile.md`.
- Lenovo T14 as a second worker: repo, checkpoint and venv in place, image build paused
  at step 12 of 19 (resume with `docker build -t forestformer3d:cu118 .` in `~/ForestFormer3D`).

## Conventions kept throughout

- No commit trailers; pushes only to `fork`.
- One Docker image for both code variants; GPU steps in the container, geo steps in a
  host venv; the tracked test list is never modified by the CLI.
- Every phase: fresh implementer per task, review per task, one whole-phase review.
