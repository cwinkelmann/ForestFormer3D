# SegmentAnyTree on the Berlin Tegel tiles, compared with ForestFormer3D

Date: 2026-09-23. Machine: carrot (H100 80 GB, GPUs 3 and 7), image
`segment-any-tree:cu118` built from the fork `cwinkelmann/SegmentAnyTree` @ `661c88e`
(checkout `/raid/cwinkelmann/SegmentAnyTree_infer`), published checkpoint
`model_file/PointGroup-PAPER.pt`. Inputs: the eleven Berlin ALS 2021 km tiles around
Tegel that ForestFormer3D processed (`docs/benchmarks/2026-09-23-berlin-visual-report.md`),
as the same 100 m sub-tiles (`inputs/berlin/sub/<T>/`). Scripts: `benchmark/sat_run_gpu.sh`
(queue), `benchmark/sat_to_ff3d.py` (contract conversion), skill
`.claude/skills/segmentanytree-inference/SKILL.md`. Logs: `work_dirs/logs/sat/`.

No ground truth exists for these tiles. Every number below is a plausibility figure
(CHM baseline, ALS ground agreement) or an *agreement* between two methods, never an
accuracy.

## 1. What SegmentAnyTree is and how the published model is packaged

**Method.** SegmentAnyTree (Wielgosz, Puliti, Xiang, Schindler, Astrup; *Remote Sensing
of Environment* 313, 2024, doi 10.1016/j.rse.2024.114367) is a PointGroup-style panoptic
model on the torch-points3d framework: a MinkowskiEngine sparse-conv U-Net backbone at
0.2 m voxels with three heads (semantic tree/non-tree, per-point offset to the instance
centre, per-point embedding), mean-shift clustering of the embeddings into instance
proposals, a ScoreNet that scores proposals, and NMS. It was trained on FOR-instance-style
plots (TLS, ULS, MLS and some ALS, resampled to several densities) in 8 m radius
cylinders, so it is "sensor and platform agnostic" by construction. At inference the
input plot is grid-sampled into 8 m cylinders on an 8 m step (each point is seen ~3
times), each cylinder is run through the network, and the per-cylinder instances are
merged plot-wide by score ("block merging": cross-IoU between proposals, NMS, then a
1-NN transfer from the 0.2 m voxels back to every input point; points more than 1 m from
a voxel, in clusters of fewer than 10 points, or predicted non-tree get no instance).

**Packaging.** The upstream repo (`SmartForest-no/SegmentAnyTree`) ships the checkpoint
`model_file/PointGroup-PAPER.pt` (665 MB, 100 epochs, trained on the `mls` job) through
Git LFS; a clone without `git lfs pull` holds a 134-byte pointer. The README's way to run
it is the pre-built Docker image `maciekwielgosz/segment-any-tree:latest` (CUDA 11.1,
torch 1.9, kernels compiled for sm 6.0-8.6 only, so it does **not** run on Hopper) with
the input folder mounted at `/home/nibio/mutable-outside-world/bucket_in_folder` and the
output at `.../bucket_out_folder`; the entrypoint `run_oracle_pipeline.sh` copies the
inputs in chunks of ten files into `run_inference.sh` and zips `final_results/` to the
output mount. A second image `maciekwielgosz/segment-any-tree-cuda11.8.0` ("not well
tested") exists; it is pulled on carrot but was not used, because the fork's own
`Dockerfile_cuda:11.8.0` had already been made to build and run on H100 (fork commit
`90a23fe`: torch 2.3.1+cu118, MinkowskiEngine compiled with `9.0` in
`TORCH_CUDA_ARCH_LIST`, torch-geometric pinned to 1.7.2, a torch-points-kernels device
fix; verified end to end on an H100 on 2026-09-22). That image is `segment-any-tree:cu118`
on carrot (17.7 GB, built 2026-09-22 17:45, with the checkpoint baked in). The fork's
`run_inference.sh <input_dir> <output_dir>` is what runs inside the container; we call it
with `--entrypoint bash` and bind-mount the fork checkout over the image's copy so the
code is the committed `661c88e`, not the snapshot the image was built from.

**Input contract.** A flat folder of `.las`/`.laz`/`.ply`, one plot per file, XYZ only
needed, every other attribute passed through. Georeferenced (UTM) coordinates are fine:
`pipeline_utm2local_parallel.py` subtracts the per-file minimum of x, y, z (stored in
`<name>_min_values.json`) and `merge_pt_ss_is` adds it back, so the output is numerically
in the input frame. Because the absolute z (after that shift) is a network feature, the
ground of a plot should sit near z = 0, which the per-file minimum gives for our tiles.
File names: no extra dots, no leading `NNN_` prefix (both are used as separators by the
pipeline scripts). The model has no notion of tiling beyond the 8 m cylinders, so a whole
file is one plot: the dataset loads it entirely, and the block merge and the final 1-NN
run over all its points.

**Output contract** (`final_results/<name>_out.laz`, LAS 1.4 point format 6, scale
0.001, offset = per-file minimum): the input attributes plus `PredSemantic` uint8
(0 non-tree, 1 tree) and `PredInstance` uint32 (tree id from 1, 0 none). Point order and
count equal the input (the fork's tracker writes an `orig_index`, and the merge assigns
by row instead of the upstream string join on float32 coordinates). The CRS VLRs of the
input are copied (upstream drops them); our 100 m sub-tiles carry none, so the output
has none and the origin is restored from the file name by our converter. There is no
per-instance score in the output (the ScoreNet scores are used for NMS and discarded).
Ids are not contiguous (NMS gaps: r12 has max id 397 for 373 instances).

**Runtime characteristics** (fork code, H100): the r12 100 m tile (192,814 points) takes
30 s wall clock for the whole container run including model load, 373 instances; a km
tile as 100 sub-tiles runs as one `eval.py` process (one model load, ~150-210 cylinders
per sub-tile at 14-19 cylinders/s). GPU memory stays under 2 GB. With the upstream
scikit-learn mean shift the same r12 tile took ~3 min (run of 2026-09-22, 360
instances); the fork's GPU mean shift (`661c88e`) is what makes km tiles practical.

**Pitfalls from `CODE_REVIEW.md` that still matter** (the rest are fixed in the fork):

- the LFS pointer instead of the checkpoint (guarded in `run_inference.sh` and the
  Dockerfile now; `SegmentAnyTree_infer` got its copy of the 665 MB file from the
  training checkout);
- upstream images do not run on Hopper (sm 6.0-8.6 cubins, no PTX);
- the dataset and transform config come from the `run_config` embedded in the
  checkpoint, not from `conf/eval.yaml`; only `data.fold`, `data.dataroot` and
  `checkpoint_dir` are injected;
- every run copies the 665 MB checkpoint into its output directory
  (`resume=True`), and keeps the input copies and local-coordinate PLYs: about 2.5 GB
  of residue per km tile, which `sat_run_gpu.sh` deletes after the conversion;
- clustering only runs for checkpoints with more than `prepare_epoch` (30) epochs; a
  model trained for fewer epochs returns zero instances without a warning (relevant to
  our own training runs, not to the published checkpoint);
- the final 1-NN over the whole file runs on the CPU with `num_workers=48`;
- hard limit of 80 instances per 8 m cylinder at training time only;
- the semantic head knows two classes (tree / non-tree): buildings, low vegetation and
  ground all become "non-tree".

## 2. Running it on carrot

Smoke test (r12, `inputs/r12_tegel_E381300_N5828300_100m.las`, GPU 7, log
`work_dirs/logs/sat/smoke-r12-20260923-165342.log`):

```
docker run --rm --name sat-smoke-r12 --gpus device=7 \
  -e NUMBA_CACHE_DIR=/tmp/numba -e OMP_NUM_THREADS=16 --shm-size=64g \
  -v /raid/cwinkelmann/SegmentAnyTree_infer:/home/nibio/mutable-outside-world \
  -v /raid/cwinkelmann/ForestFormer3D/work_dirs:/work_dirs \
  --entrypoint bash segment-any-tree:cu118 run_inference.sh /work_dirs/sat-smoke/in /work_dirs/sat-smoke/out
```

30 s wall clock, rc 0. Output `final_results/r12_tegel_E381300_N5828300_100m_out.laz`:
LAS 1.4 pf6, 192,814 points (= input, same order), extra dims `PredSemantic`,
`PredInstance`; ids 1..397, 373 instances, 107,628 of 121,368 tree points carry an
instance (13,740 tree points, 11 %, have none: the 1 m / 10-point rules above); no
point is "instance but non-tree". Converted to the ff3d contract (`work_dirs/sat-r12/`):

| | SegmentAnyTree | ForestFormer3D (`work_dirs/tegel-r12`) | CHM local maxima |
|---|---:|---:|---:|
| trees | 373 | 402 | 397 |
| height min / median / max (m) | 3.3 / 28.0 / 34.4 | 2.2 / 25.8 / 34.8 | 12.5 / 29.1 / 34.8 |
| ground vs ALS agreement | 97.3 % | (see `2026-09-22-tegel-als.md`) | |
| decision rule | usable (count ok, median 1.1 m off) | not usable (median 3.3 m off) | |
| wall time | 30 s | minutes (single tile path) | |

**Decision for km tiles: run the existing 100 m sub-tiles and merge afterwards.** A km
tile is 23-27 M points; SegmentAnyTree treats one file as one plot, so a native run would
hold the whole file in the dataset, run the plot-wide block merge (a cross-IoU matrix over
every proposal of the tile, tens of thousands of trees) and the 48-worker CPU 1-NN over
23 M points in one process. Nothing in the code caps that, but nothing has been tested at
that size either, and one failed 20-30 min run per tile would be the only way to find out.
The sub-tiles are also exactly what ForestFormer3D saw, so the two methods get the same
cuts and the same border effects (about 16 % of trees touch a 100 m border and stay
split, `2026-09-22-tegel-berlin-2021.md` section 3). `sat_to_ff3d.py` restores the UTM
origin from each sub-tile name and renumbers ids across sub-tiles like `ff3d_geo.merge`.

Queue: `benchmark/sat_run_gpu.sh` on GPUs 3 and 7 (the only idle cards; GPU 1 is never
used), launched 2026-09-23 16:58 CEST:

```
nohup bash benchmark/sat_run_gpu.sh 3 3dm_33_379_5828_1_be 3dm_33_379_5829_1_be 3dm_33_380_5828_1_be \
     3dm_33_380_5829_1_be 3dm_33_381_5828_1_be 3dm_33_381_5829_1_be > work_dirs/logs/sat/sat-gpu3-<ts>.log 2>&1 &
nohup bash benchmark/sat_run_gpu.sh 7 3dm_33_381_5830_1_be 3dm_33_382_5828_1_be 3dm_33_382_5829_1_be \
     3dm_33_383_5828_1_be 3dm_33_383_5829_1_be > work_dirs/logs/sat/sat-gpu7-<ts>.log 2>&1 &
```

## 3. Conversion to the ForestFormer3D contract

`benchmark/sat_to_ff3d.py --sat-las work_dirs/sat-<T>/sat_raw/final_results/*_out.laz --tile <T> --out work_dirs/sat-<T> --runtime-s <s>`:

| ff3d field | from SegmentAnyTree |
|---|---|
| `x`, `y` | file coordinates + the `E<x>_N<y>` origin of the sub-tile name (`ff3d_geo.origin.parse_origin`); `--absolute` for files already in UTM |
| `z` | unchanged |
| `classification` | passed through from the input |
| `treeID` int32 | `PredInstance - 1`, 0 -> -1; sub-tile `j` offset by `sum(max_id_i + 1, i < j)` in sorted file order (as `ff3d_geo.merge.merge_las`) |
| `semantic` uint8 | `PredSemantic` 0 (non-tree) -> 0 ground, 1 (tree) -> 2 leaf; class 1 (wood) never occurs; anything else -> 255 |
| `score` float32 | -1 everywhere (no per-instance score in the output) |
| CRS | WKT VLR for EPSG:25833 |

Consequence of the semantic mapping: the report's "ground vs vegetation agreement"
compares SegmentAnyTree's *non-tree* class with ALS class 2, so buildings, low
vegetation and water count as disagreements for SegmentAnyTree but not necessarily for
ForestFormer3D, whose ground class is a real ground class. The `mean_score` column of the
tree table is -1 for every SegmentAnyTree tree. Everything else (`<T>_trees.gpkg`,
`<T>_crowns.gpkg`, the two GeoTIFF masks, `<T>_report.json/.md`) comes from the unchanged
`ff3d_geo` functions. Tests: `tests/test_sat_to_ff3d.py`.

## 4. Results per km tile

All eleven tiles ran to completion (GPU 3: six tiles 17:18-20:56, GPU 7: five tiles
16:58-20:02 CEST). SegmentAnyTree numbers are the `work_dirs/sat-<T>/<T>_report.json`
of the merged tile; ForestFormer3D numbers are the same report keys from
`docs/benchmarks/2026-09-23-berlin-visual-report.md` / `work_dirs/berlin-<T>/`. "CHM"
is the local-maxima baseline on the same LAS (`ff3d_geo.baseline`), identical for both.
Wall time is the container run over the 100 m sub-tiles (`tile done (run Ns)` in the
queue log; the conversion adds 1-2 min per tile). ForestFormer3D's wall times are the
measured 92-99 min per km tile of the single-GPU run of 2026-09-22
(`2026-09-22-tegel-berlin-2021.md`, 55-60 s per sub-tile) for the three Revier 12 tiles;
the other eight were inferred in a seven-GPU block under heavy host contention (up to
6.5 h per tile, `ff3d-inference-km-tiles` section 4) or recovered from partial runs and
have no clean per-tile figure, so the column says "n/a".

| Tile | Points | CHM baseline | Trees SAT / FF3D | Median height SAT / FF3D / CHM (m) | Ground agreement SAT / FF3D | Wall time SAT / FF3D |
|---|---:|---:|---:|---:|---:|---:|
| `3dm_33_379_5828_1_be` | 12,671,526 | 20,420 | 17,925 / 17,754 | 20.1 / 16.6 / 23.1 | 90.0 % / 94.7 % | 50 min* / n/a |
| `3dm_33_379_5829_1_be` | 15,085,213 | 20,648 | 19,648 / 16,144 | 13.0 / 10.8 / 14.1 | 83.9 % / 88.9 % | 35 min / n/a |
| `3dm_33_380_5828_1_be` | 25,073,247 | 39,876 | 36,754 / 37,386 | 25.1 / 21.9 / 28.4 | 96.3 % / 97.6 % | 37 min / 99 min |
| `3dm_33_380_5829_1_be` | 16,283,451 | 19,772 | 16,992 / 16,503 | 10.9 / 8.4 / 11.7 | 86.2 % / 88.4 % | 23 min / n/a |
| `3dm_33_381_5828_1_be` | 25,175,784 | 41,616 | 34,984 / 37,058 | 27.4 / 25.0 / 29.4 | 96.8 % / 97.6 % | 36 min / 96 min |
| `3dm_33_381_5829_1_be` | 23,246,640 | 34,809 | 32,368 / 31,385 | 21.2 / 17.6 / 26.2 | 92.7 % / 96.2 % | 33 min / 92 min |
| `3dm_33_381_5830_1_be` | 18,638,283 | 30,516 | 27,467 / 26,440 | 17.8 / 14.7 / 19.8 | 88.6 % / 93.7 % | 32 min / n/a |
| `3dm_33_382_5828_1_be` | 20,188,173 | 32,297 | 27,614 / 27,538 | 22.9 / 20.4 / 25.5 | 94.1 % / 96.1 % | 33 min / n/a |
| `3dm_33_382_5829_1_be` | 27,012,312 | 39,719 | 36,493 / 34,752 | 24.2 / 21.9 / 27.6 | 96.4 % / 97.5 % | 41 min / n/a |
| `3dm_33_383_5828_1_be` | 16,085,492 | 26,946 | 20,112 / 18,470 | 14.2 / 12.9 / 15.8 | 87.8 % / 92.5 % | 25 min / n/a |
| `3dm_33_383_5829_1_be` | 27,084,463 | 40,119 | 39,981 / 37,022 | 22.4 / 20.6 / 26.1 | 95.6 % / 97.3 % | 36 min / 391 min** |
| **total** | 226,544,890 | 346,738 | 310,338 / 300,452 | | | 6.4 h / |

\* the retry after the water-tile crash (section 2), run while a 48-worker CPU job of
another user shared the host (cylinder rate fell to 4-6/s); the other tiles ran at
14-19 cylinders/s. \*\* seven-GPU contended block, not comparable.

Per-tile decision rule (`ff3d_geo.report.recommend`, count within 50 % of the CHM
baseline and median height within 3 m of the CHM median): SegmentAnyTree passes on
`379_5828` (2.9 m), `379_5829`, `380_5829`, `381_5828` (2.0 m), `381_5830` (2.0 m),
`382_5828` (2.6 m), `383_5828` and is "not usable" on `380_5828` (3.3 m), `381_5829`
(5.0 m), `382_5829` (3.4 m), `383_5829` (3.7 m): 7 of 11 against ForestFormer3D's 0 of
11 (4.4-8.6 m gaps on the three Revier 12 tiles). The count criterion passes everywhere
for both.

## 5. Instance-level agreement between the two methods

`benchmark/instance_agreement.py` matches the instances of the two merged LAS files
one-to-one (they hold the same points in the same order): every overlapping pair of the
two `treeID` arrays with its IoU, a pair counts when the two instances are each other's
best partner and the IoU is >= 0.5 (above 0.5 an instance can have only one such
partner, so this is the maximum matching that `instance_diagnostics.py` computes with
the Hungarian algorithm, without the dense 27k x 27k matrix), plus the
`instance_diagnostics.py` fragment counts (a B instance is a "piece" of an A instance
when >= 20 % of its points lie in it). **Agreement, not accuracy**: no ground truth
exists for these tiles, so a low figure says the methods disagree, not which is wrong.

### r12 (100 m, `work_dirs/tegel-r12` vs `work_dirs/sat-r12`)

| Metric | Value |
|---|---|
| points | 192,814 |
| instances ForestFormer3D / SegmentAnyTree | 402 / 373 |
| points with an instance ForestFormer3D / SegmentAnyTree / both | 107,410 / 107,628 / 95,397 |
| matched pairs (IoU >= 0.5) | 231 |
| matched share of ForestFormer3D / SegmentAnyTree | 57.5 % / 61.9 % |
| median IoU of matched pairs | 0.812 |
| SegmentAnyTree pieces per ForestFormer3D instance (>= 20 % overlap) | 0.96 |
| ForestFormer3D pieces per SegmentAnyTree instance | 0.97 |
| ForestFormer3D instances split by SegmentAnyTree (>= 2 pieces) | 18.4 % |
| SegmentAnyTree instances split by ForestFormer3D | 13.4 % |

Both methods put an instance on almost the same point set (107 k points each, 95 k in
common: the ALS canopy), and their tree counts bracket the CHM baseline (402 / 373 vs
397). Yet only 231 trees (58-62 %) are the *same* tree at IoU 0.5; where they match they
match well (median IoU 0.81). The rest is crown boundaries drawn differently: 18 % of the
ForestFormer3D trees are cut into two or more SegmentAnyTree pieces and 13 % the other
way round, with the mean pieces-per-instance just under 1 in both directions (i.e. each
method also has instances the other does not touch at the 20 % level at all).

### 3dm_33_381_5830_1_be (km tile, `work_dirs/berlin-…` vs `work_dirs/sat-…`, 5 s)

| Metric | Value |
|---|---|
| points | 18,638,283 |
| instances ForestFormer3D / SegmentAnyTree | 26,440 / 27,467 |
| points with an instance ForestFormer3D / SegmentAnyTree / both | 9,462,774 / 8,515,116 / 7,598,067 |
| matched pairs (IoU >= 0.5) | 13,639 |
| matched share of ForestFormer3D / SegmentAnyTree | 51.6 % / 49.7 % |
| median IoU of matched pairs | 0.765 |
| SegmentAnyTree pieces per ForestFormer3D instance (>= 20 % overlap) | 1.09 |
| ForestFormer3D pieces per SegmentAnyTree instance | 0.85 |
| ForestFormer3D instances split by SegmentAnyTree (>= 2 pieces) | 26.3 % |
| SegmentAnyTree instances split by ForestFormer3D | 8.4 % |

At km scale half of the trees are the same tree (13,639 pairs, 50-52 % of either side,
median IoU 0.77). The asymmetry is the interesting part: 26 % of the ForestFormer3D
instances are cut into two or more SegmentAnyTree pieces but only 8 % the other way
round, and SegmentAnyTree puts 0.95 M fewer points into instances (8.5 M vs 9.5 M).
Read together with the tree counts (27.5 k vs 26.4 k) this says SegmentAnyTree draws
*smaller* crowns - more of the low and outer canopy is left without an instance - and
splits some of the large ForestFormer3D crowns, rather than merging neighbours. Which
of the two is closer to the real trees cannot be decided from these files.

## 6. Reading the numbers

- **Tree counts are the same story for both methods.** SegmentAnyTree finds 310 k trees
  on the eleven tiles, ForestFormer3D 300 k, the CHM local maxima 347 k; per tile the
  two methods are within 1-3 k of each other and both sit 2-25 % below the baseline.
  Neither method explodes or collapses the instance count on ALS.
- **SegmentAnyTree's trees are taller by 2-4 m at the median on every tile**, and
  closer to the CHM: its median gap to the CHM median is 1.1-5.0 m (7 of 11 tiles
  inside the 3 m band) against ForestFormer3D's 3.2-8.6 m (0 of 11). Read with the
  agreement numbers (section 5: SegmentAnyTree puts ~10 % fewer points into instances
  and splits 26 % of the ForestFormer3D crowns), the picture is that SegmentAnyTree
  keeps the crown *tops* as separate trees where ForestFormer3D attaches them to a
  neighbour or starts a crown below its apex, and leaves more of the low canopy
  unassigned. Whether that is right cannot be settled without ground truth; it is the
  direction in which the ALS height distribution and the CHM point, and it is the same
  finding for both methods as in `2026-09-22-tegel-berlin-2021.md` section 5 - only
  less severe for SegmentAnyTree.
- **Ground agreement is lower for SegmentAnyTree on every tile (84-97 % vs 88-98 %)**,
  but that figure is not comparable: SegmentAnyTree's "ground" is its *non-tree* class
  (section 3), so buildings, low vegetation and water count against it. The tiles with
  the largest gap (`379_5829`, `380_5829`, `383_5828`: urban and water tiles) are exactly
  the ones with the most non-tree, non-ground surface. ForestFormer3D's ground class is a
  ground class, and its semantic head remains the cleaner one.
- **Speed**: 23-41 min per km tile for SegmentAnyTree on a shared H100 (19-25 s per 100 m
  sub-tile, GPU at 30-45 %, CPU-bound in the mean shift, block merge and the final
  1-NN) against 92-99 min for ForestFormer3D on an uncontended GPU (55-60 s per sub-tile,
  before the vectorised writer of commits `3da7f87`/`efb3fcf`, which the profile in
  `2026-09-23-inference-profile.md` puts well under 20 s). Both are practical for
  km-tile batches; neither number is a benchmark of the model alone.
- **What failed and was fixed**: a sub-tile without any predicted instance (open water)
  crashed the SegmentAnyTree tracker and lost the whole 80-sub-tile batch of `379_5828`
  (fork commit `fbeb22a` writes an empty result instead; ten such sub-tiles occurred in
  the rerun); the first version of `sat_to_ff3d.py` built the tree table over the merged
  km tile, which did not finish in 20 min (commit `3c95c1a` builds it per sub-tile and
  merges, 45 s per km tile).
- **Where to look next**: the 26 % of ForestFormer3D crowns that SegmentAnyTree splits are
  the natural set for a visual check in the Potree site (Method switch, tree-id colour,
  same tile from both methods); the height question needs the per-crown "predicted top
  vs CHM maximum inside the crown polygon" comparison proposed in the earlier report,
  which now can be run for both methods from the same GeoPackages.

## 7. Where the results are

- carrot: `work_dirs/sat-<T>/` (`<T>.las`, `<T>_trees.gpkg`, `<T>_crowns.gpkg`,
  `<T>_instance_50cm.tif`, `<T>_semantic_50cm.tif`, `<T>_report.json/.md`,
  `sat_raw/final_results/*_out.laz` + `eval.log`), `work_dirs/sat-r12/`, logs
  `work_dirs/logs/sat/`, octrees `work_dirs/logs/potree/out_sat/<T>/`.
- 2TB volume: `/Volumes/2TB/winmol/ALS_Data/berlin_als_2021_sat/<T>/` (the products
  above without `sat_raw`), Potree site `berlin_potree/pointclouds_sat/<T>/` +
  `data/<T>_sat_*` with the Method switch in `index.html`.
