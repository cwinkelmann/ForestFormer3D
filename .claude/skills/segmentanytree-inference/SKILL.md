---
name: segmentanytree-inference
description: Use when running SegmentAnyTree (Wielgosz et al. 2024, torch-points3d) as the second tree-segmentation method on carrot - the segment-any-tree:cu118 image and fork checkout, run_inference.sh mounts and GPU selection, the per-GPU km-tile queue, converting its output to the ForestFormer3D contract with sat_to_ff3d.py, where results live, timings, the instance agreement with ForestFormer3D and adding the results to the Potree site.
---

# SegmentAnyTree inference on carrot

Read `ff3d-carrot` first (ssh, GPU etiquette, the host geo venv). SegmentAnyTree is the
comparison method: same inputs as ForestFormer3D, same output contract after conversion,
so every downstream tool (tree table, masks, report, Potree, agreement) is shared.
Write-up: `docs/benchmarks/2026-09-23-segmentanytree-berlin.md`.

## 1. What runs

- **Code**: the fork `cwinkelmann/SegmentAnyTree`, branch `fix/pipeline-usability`
  (`661c88e`: index-based merge, CRS kept, Hopper build, GPU mean shift), checked out
  on carrot at `/raid/cwinkelmann/SegmentAnyTree_infer`. The sibling
  `/raid/cwinkelmann/SegmentAnyTree` is the **training** checkout and is bind-mounted
  into a running training container - never modify or pull it while that runs.
- **Checkpoint**: `model_file/PointGroup-PAPER.pt`, 665 MB, Git LFS upstream. A fresh
  clone holds a 134-byte pointer; copy the real file from the training checkout
  (`cp -f /raid/cwinkelmann/SegmentAnyTree/model_file/PointGroup-PAPER.pt model_file/`)
  or `git lfs pull`. `run_inference.sh` refuses to start on a pointer.
- **Image**: `segment-any-tree:cu118` (17.7 GB, built 2026-09-22 from the fork's
  `Dockerfile_cuda:11.8.0`: CUDA 11.8, torch 2.3.1, MinkowskiEngine with sm_90). The
  published `maciekwielgosz/segment-any-tree:latest` is CUDA 11.1 with sm 6.0-8.6
  kernels and does not run on the H100s; `maciekwielgosz/segment-any-tree-cuda11.8.0`
  is pulled on carrot but untested. Rebuild only if `docker image inspect
  segment-any-tree:cu118` fails: `cd /raid/cwinkelmann/SegmentAnyTree_infer && docker
  build -f Dockerfile_cuda:11.8.0 -t segment-any-tree:cu118 .` (tens of minutes; the
  checkpoint must be the real file, the build asserts its size).
- The container's own entrypoint is the Oracle bucket pipeline; we bypass it with
  `--entrypoint bash run_inference.sh <in> <out>` and mount the checkout over the
  image's copy at `/home/nibio/mutable-outside-world`, so the code that runs is the
  checkout, not the build snapshot.

## 2. One folder, by hand (the smoke test)

```bash
ssh carrot; cd /raid/cwinkelmann/ForestFormer3D
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv   # pick a card at 0 % / 0 MiB, never GPU 1
mkdir -p work_dirs/sat-smoke/in && cp inputs/r12_tegel_E381300_N5828300_100m.las work_dirs/sat-smoke/in/
docker run --rm --name sat-smoke-r12 --gpus device=7 \
  -e NUMBA_CACHE_DIR=/tmp/numba -e OMP_NUM_THREADS=16 --shm-size=64g \
  -v /raid/cwinkelmann/SegmentAnyTree_infer:/home/nibio/mutable-outside-world \
  -v /raid/cwinkelmann/ForestFormer3D/work_dirs:/work_dirs \
  --entrypoint bash segment-any-tree:cu118 run_inference.sh /work_dirs/sat-smoke/in /work_dirs/sat-smoke/out \
  > work_dirs/logs/sat/smoke-r12-$(date +%Y%m%d-%H%M%S).log 2>&1
```

30 s on an H100 for the 192,814-point r12 tile; expect `192814 points, ... 373 tree
instances` near the end of the log and `final_results/<name>_out.laz`. Sanity numbers:
LAS 1.4 pf6, point count = input, extra dims `PredSemantic`, `PredInstance`, 370-400
instances (the run is not bit-reproducible: 360 with the CPU mean shift, 373 with the
GPU one).

Input rules: a **flat** folder of `.las/.laz/.ply`, one plot per file, XYZ only needed,
no extra dots and no leading `NNN_` in file names, UTM or local coordinates both fine
(each file is shifted by its own minimum and shifted back). The output has no CRS when
the input has none (our 100 m sub-tiles), no per-instance score, ids from 1 with NMS
gaps, 0 = no instance. `<out>` is wiped at start (`clean_output_dir=true`).

## 3. km tiles: the per-GPU queue

The sub-tiles from `ff3d_geo split` (`inputs/berlin/sub/<T>/`, see
`ff3d-inference-km-tiles`) are the input folder; one `eval.py` process handles all ~100
files of a km tile (one model load). `benchmark/sat_run_gpu.sh` owns one GPU and walks
its tiles: `run_inference.sh` -> `benchmark/sat_to_ff3d.py` -> residue cleanup.

```bash
cd /raid/cwinkelmann/ForestFormer3D && mkdir -p work_dirs/logs/sat
TS=$(date +%Y%m%d-%H%M%S)
nohup bash benchmark/sat_run_gpu.sh 3 3dm_33_379_5828_1_be 3dm_33_379_5829_1_be > work_dirs/logs/sat/sat-gpu3-$TS.log 2>&1 &
nohup bash benchmark/sat_run_gpu.sh 7 3dm_33_381_5830_1_be                       > work_dirs/logs/sat/sat-gpu7-$TS.log 2>&1 &
```

Watch it (poll every few minutes, not in a loop):

```bash
grep -h '^===\|!!!' work_dirs/logs/sat/sat-gpu*.log            # milestones: sat run / convert / tile done (run Ns)
tail -c 600 work_dirs/logs/sat/sat-gpu3-*.log | tr '\r' '\n' | grep -o '[0-9]*/[0-9]* \[[^]]*\]' | tail -1   # cylinder progress bar
docker ps --format '{{.Names}} {{.Status}}' | grep sat-gpu
```

The progress bar counts 8 m cylinders (150-210 per 100 m sub-tile, 14-19 cylinders/s on
an H100 shared with other jobs); after the bar the tracker's per-file 1-NN and the merge
run on the CPU. `!!! <T> sat run failed (rc=..., M of N result files)` means the container
died or a sub-tile produced no output: rerun that tile alone. Nothing resumes inside a
tile; the queue continues with the next tile.

Timings (H100 shared with other jobs, GPU at 30-45 %): 100 sub-tiles of
`3dm_33_381_5830` took **1899 s** (31.7 min, ~19 s per sub-tile) for the container run,
against ForestFormer3D's 55-60 s per sub-tile; the conversion adds a few minutes per km
tile (per-sub-tile tree tables, merge, masks). The full per-tile table is in the
write-up.

Known failure, fixed in fork commit `fbeb22a`: a sub-tile in which no cylinder produced
an instance (open water, e.g. the Tegeler See sub-tiles of `3dm_33_379_5828`) made the
tracker raise `IndexError: index is out of bounds for dimension with size 0` in
`finalise`, which lost the whole 80-sub-tile batch. Checkouts older than that commit
must not be used for tiles with water or bare ground.

Residue: each run leaves `sat_raw/{input_data,utm2local,PointGroup-PAPER.pt,*.ply}`
(about 2.5 GB per km tile); the queue deletes them after a successful conversion
(`SAT_KEEP=1` keeps them). `sat_raw/final_results/` and `sat_raw/eval.log` stay.

## 4. Conversion to the ff3d contract

`benchmark/sat_to_ff3d.py` (host venv, no torch) writes `work_dirs/sat-<T>/<T>.las` -
LAS 1.4 pf6 EPSG:25833 with `treeID` int32 (-1 none; `PredInstance - 1`, then per
sub-tile offsets like `ff3d_geo.merge`), `semantic` uint8 (non-tree -> 0 ground,
tree -> 2 leaf, wood never; 255 unknown), `score` float32 (-1: none available), the
ALS `classification` kept - then `ff3d_geo` products next to it: `<T>_trees.gpkg`,
`<T>_report.json/.md`, `<T>_instance_50cm.tif`, `<T>_semantic_50cm.tif`,
`<T>_crowns.gpkg`. Rerun by hand after a failed convert:

```bash
source /raid/cwinkelmann/ff3d-geo-venv/bin/activate
T=3dm_33_381_5828_1_be
python benchmark/sat_to_ff3d.py --sat-las work_dirs/sat-$T/sat_raw/final_results/*_out.laz \
    --tile $T --out work_dirs/sat-$T --runtime-s $(grep "$T: tile done" work_dirs/logs/sat/sat-gpu*.log | sed 's/.*run \([0-9]*\)s.*/\1/')
```

`--absolute` for files that already carry UTM coordinates (no `E<x>_N<y>` in the name),
`--no-masks` to skip the rasters. Note the report's "ground vs vegetation agreement"
compares SegmentAnyTree's *non-tree* class with ALS ground, so buildings and low
vegetation count as disagreements. `mean_score` is -1 for every tree.

## 5. Agreement with ForestFormer3D

Both merged LAS files hold the same points in the same order (same sub-tiles, sorted),
so `benchmark/instance_agreement.py` can match instances one-to-one (Hungarian on the
IoU matrix, hit at IoU >= 0.5, the `instance_diagnostics.py` recipe):

```bash
python benchmark/instance_agreement.py --a work_dirs/berlin-$T/$T.las --b work_dirs/sat-$T/$T.las \
    --label-a ForestFormer3D --label-b SegmentAnyTree --json work_dirs/sat-$T/agreement_ff3d_vs_sat.json
```

It refuses when x/y/z differ. This is agreement, not accuracy: there is no ground truth.

## 6. Where things are

- carrot: `work_dirs/sat-<T>/` (contract LAS, gpkg, tifs, report, `sat_raw/final_results/`),
  `work_dirs/sat-r12/` (the smoke tile), logs `work_dirs/logs/sat/`.
- Mac / 2TB volume: `/Volumes/2TB/winmol/ALS_Data/berlin_als_2021_sat/<T>/` (merged
  products only: `rsync -av --exclude 'sat_raw' carrot:.../work_dirs/sat-$T/ .../berlin_als_2021_sat/$T/`;
  check `df -h /Volumes/2TB` first).

## 7. Potree site

Octree (carrot, seconds per tile) and the site variant (Mac):

```bash
# carrot
python3 benchmark/potree_convert_tile.py --las work_dirs/sat-$T/$T.las \
    --out work_dirs/logs/potree/out_sat/$T --potree-converter work_dirs/logs/potree/PotreeConverter_linux_x64/PotreeConverter
# Mac
rsync -a carrot:/raid/cwinkelmann/ForestFormer3D/work_dirs/logs/potree/out_sat/$T/ /Volumes/2TB/winmol/ALS_Data/berlin_potree/pointclouds_sat/$T/
.venv-cpu/bin/python benchmark/build_potree_site.py --site /Volumes/2TB/winmol/ALS_Data/berlin_potree \
    --variant sat --sat-dir /Volumes/2TB/winmol/ALS_Data/berlin_als_2021_sat [--tiles $T]
(cd docker/potree && docker compose up -d --build)   # or: python3 benchmark/serve_potree.py --root <site> --port 8080
```

`--variant sat` adds `variants.sat` to the tiles already in `data/tiles.json`
(octree under `pointclouds_sat/<T>/`, `data/<T>_sat_trees.geojson`, `_sat_crowns.geojson`,
`_sat_instance.png`) and refreshes `index.html`, whose **Method** buttons
(ForestFormer3D / SegmentAnyTree) reload the checked tiles from the other octree and
vectors; `?method=sat` in the URL starts on SegmentAnyTree. Tiles without the variant
are greyed out. CHM and orthophotos are shared.

## 8. Adding tiles

1. Sub-tiles must exist: `python -m ff3d_geo split --las inputs/berlin/$T.las --out inputs/berlin/sub/$T`.
2. Queue the tile on a free GPU (section 3); convert runs automatically.
3. Copy `work_dirs/sat-$T/` home (section 6), rebuild the Potree variant (section 7),
   append the tile to the write-up's table (`<T>_report.json` has the numbers).
