---
name: ff3d-ams3d-benchmark
description: Use when running or comparing the CPU-only AMS3D (adaptive mean shift 3D) crown segmentation baseline in ff3d_geo - the `python -m ff3d_geo ams3d` command, its parameter configs, runtime on carrot, where the outputs go, and how to compare it with ForestFormer3D or the R13 reference crowns.
---

# AMS3D benchmark method

`ff3d_geo/ams3d.py` is a port of the adaptive mean shift 3D spike from the
`GEE_animation` repo (branch `als-ams3d-spike`, `docs/experiments/ams3d_spike.py` and
`2026-09-22-ams3d-spike.md`). It needs no GPU and no Docker: numpy + scipy + laspy in the
host geo venv. The results are `docs/benchmarks/2026-09-23-ams3d-berlin.md`.

## Method in one paragraph

Height-normalise with a 1 m min-z grid of ALS class 2; keep classes 3/4/5 between 2 and
60 m above ground; run mean shift from every point with a cylinder kernel whose radius
`h_s = max(1, 0.12 h)` and half-height `h_r = max(1.5, 0.27 h)` grow with the current
mode height; union converged modes within 1.5 m horizontally / 6 m vertically; attach
clusters under 50 points to the nearest bigger one. That is the spike's **config C**, the
default. Points get `treeID` (-1 for ground / unclustered), `semantic` 0 ground / 2 leaf
/ 255 not classified (no wood class), `score` -1.

| config | kernel | merge xy / z | min points | note |
|---|---|---|---|---|
| `default` | 0.12 h / 0.27 h | 1 / 2 m | 20 | textbook; splits tall crowns vertically |
| `A` | 0.12 h / 0.5 h | 1 / 4 m | 20 | wider vertical window, 2-3x slower |
| `B` | 0.15 h / 0.5 h | 1 / 4 m | 50 | A + wider horizontal window |
| `C` | 0.12 h / 0.27 h | 1.5 / 6 m | 50 | recommended; matches the reference density |

Programmatic use: `segment_ams3d(xyz, classification, Ams3dParams())` returns the id per
point; `CONFIGS["A"]` etc. and `params_for("C", min_points=30)` build variants;
`run_ams3d_tile(las_in, las_out, params, buffer_m)` writes one result LAS.

## Running

```bash
# carrot, host venv (ff3d-carrot skill), no GPU needed
cd /raid/cwinkelmann/ForestFormer3D && source /raid/cwinkelmann/ff3d-geo-venv/bin/activate
# a 100 m local-coordinate tile (origin in the name) -> one sub-tile, no buffer
python -m ff3d_geo ams3d --las inputs/r12_tegel_E381300_N5828300_100m.las --out work_dirs/ams3d-r12 --workers 1
# a km tile: 100 m sub-tiles + 10 m buffer, 48 processes, log it
T=3dm_33_381_5829_1_be
nohup python -m ff3d_geo ams3d --las inputs/berlin/$T.las --out work_dirs/ams3d-$T --workers 48 \
  > work_dirs/logs/ams3d-$T.log 2>&1 &
tail -f work_dirs/logs/ams3d-$T.log      # one line per finished sub-tile with its seconds
```

Options: `--config default|A|B|C`, `--buffer M` (10), `--size M` (100), `--workers N`
(default `os.cpu_count() // 2`; use 48 on carrot when others are busy), `--keep-subtiles`.
The km tile is split with `ff3d_geo.split.split_las(..., buffer_m=10)`; each sub-tile is
segmented on its buffered extent and only its core points are written, so the merge holds
exactly the source points; trees on a sub-tile seam get one id per side (same border
effect as the ForestFormer3D split/merge).

Outputs in `--out`, same contract as `run` + `merge` + `masks`: `<T>.las`,
`<T>_trees.gpkg`, `<T>_crowns.gpkg`, `<T>_instance_50cm.tif`, `<T>_semantic_50cm.tif`,
`<T>_report.json` (with an extra `ams3d` block: config, params, workers, per-sub-tile
seconds) and `<T>_report.md`. The report's ground/vegetation agreement is trivially high
because AMS3D's ground IS ALS class 2; the CHM baseline and height medians are comparable.

## Runtime (carrot, 224-core host)

Measured 2026-09-23 (`docs/benchmarks/2026-09-23-ams3d-berlin.md`): r12 hectare 97 s and
r13 hectare 301 s single-process (121 k / 187 k vegetation points); km tile
`3dm_33_381_5829_1_be` (23.2 M points, 100 buffered sub-tiles of 57–354 k vegetation
points) 19.4 min wall with `--workers 48` — sub-tiles take 5 s to 19 min each (median 4
min, 27,000 CPU-s in total), so the tail of dense sub-tiles sets the wall time. Memory per
worker stays under ~2 GB (`chunk = 4000` seeds per neighbour query). Two things that bit:

* numpy's and scipy's bundled OpenBLAS each start a 64-thread pool in every spawned
  worker; the pipeline sets `OPENBLAS_NUM_THREADS/OMP_NUM_THREADS/MKL_NUM_THREADS=1`
  before creating the pool (without it 48 workers crawled at ~13 runnable processes).
* progress lines are flushed, so `grep -c 'vegetation pts' <log>` is the live counter;
  the sub-tile result files appear in `<out>/ams3d_subtiles_out/` as they finish.

Running from a separate worktree: if the main checkout on carrot has another agent's
untracked files blocking `git pull`, `git worktree add --detach /raid/cwinkelmann/ff3d-ams3d-worktree origin/fix/review-findings`
and run `python -m ff3d_geo ams3d` from there with absolute `--las`/`--out` paths
(`python -m` puts the cwd first on `sys.path`, so that worktree's `ff3d_geo` wins over the
venv's editable install).

Where the results are: `work_dirs/ams3d-r12`, `work_dirs/ams3d-r13`,
`work_dirs/ams3d-3dm_33_381_5829_1_be`, `work_dirs/ams3d-3dm_33_380_5828_1_be`,
`work_dirs/ams3d-3dm_33_381_5828_1_be` on carrot; the small files (reports, tree/crown
GeoPackages, GeoTIFFs) of `381_5829` are also on the Mac under
`/Volumes/2TB/winmol/ALS_Data/berlin_als_2021_ams3d/<T>/`; the merged LAS files (865 MB
each) stayed on carrot because that volume is at its 40 GB floor.

## Comparing

```bash
# agreement between two segmentations of the same points (Hungarian IoU >= 0.5)
python benchmark/ams3d_compare.py iou work_dirs/berlin-$T/$T.las work_dirs/ams3d-$T/$T.las --json work_dirs/ams3d-$T/agreement_vs_ff3d.json
# the spike's rule against the R13 crown file (Mac; the gpkg is on the 2TB volume)
python benchmark/ams3d_compare.py reference work_dirs/ams3d-r13/r13_spandau_E376400_N5827400_100m.las \
  "/Volumes/2TB/winmol/training_data/WINDWURF_Tegel/ALS segmentation 2017.gpkg" \
  --bbox 376400 5827400 376500 5827500 --trees work_dirs/ams3d-r13/r13_spandau_E376400_N5827400_100m_trees.gpkg
```

Tests: `pytest tests/test_geo_ams3d.py` (synthetic scenes, contract, buffer split,
pipeline); `pytest -m slow tests/test_geo_ams3d.py` reruns the spike's r12 patch and
checks the tree count / mean crown diameter against the spike's stats JSON (needs the
km tile on `/Volumes/2TB` or the local 100 m tile).
