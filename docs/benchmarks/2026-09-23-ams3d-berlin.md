# AMS3D (adaptive mean shift) on the Berlin tiles, next to ForestFormer3D (2026-09-23)

**Question.** How does a classical, CPU-only crown segmentation — adaptive mean shift 3D,
the method spiked on 2026-09-22 in the `GEE_animation` repo — compare with ForestFormer3D
on the same Berlin ALS 2021 tiles, at the level of tree counts, heights, runtime and
instance-level agreement, and (where the R13 reference crown file reaches) against the
same reference crowns?

**Answer.** With the spike's recommended config C, AMS3D finds fewer, larger crowns than
ForestFormer3D: 312 vs 402 trees on the Tegel hectare, 407 vs 443 on the Spandau hectare,
20,417 vs 31,385 on the km tile `3dm_33_381_5829_1_be`, 28,810 vs 37,386 and 31,337 vs
37,058 on the two Revier 12 tiles `380_5828` / `381_5828`, with median heights within
1–6 m of the CHM medians and never below 1.8 m (ForestFormer3D's report minima go to
-14 m). The two methods agree on a tree (point-set IoU
>= 0.5 under Hungarian matching) for roughly half of AMS3D's trees and a third of
ForestFormer3D's (30–39 % on the km tiles, 44–47 % on the hectares); the rest is mostly
ForestFormer3D splitting what AMS3D keeps as one crown (understorey and stacked
sub-crowns). Against the R13 reference crowns both land in the
same band (AMS3D 238, ForestFormer3D 245 of 407 crowns matched by the spike's apex rule),
so neither is validated by it — the reference is itself an unvalidated segmentation
product, and these are agreement figures, not accuracy. A km tile takes 19–25 min on 48
CPU cores, against ~94–102 min of GPU time for ForestFormer3D. One caveat on the spike's record:
its committed script, rerun today with the config C arguments, gives 398 trees on the
r13 hectare, not the 339 in its stats file (section 2), so the spike's C row above was
produced by an earlier state of the script.

## 1. The method and where it comes from

AMS3D (Ferraz et al. 2012, 2016) runs mean shift from every vegetation point with a
cylinder kernel whose size grows with the height above ground of the *current* mode:
horizontal radius `h_s = max(1 m, 0.12 h)`, vertical half-height `h_r = max(1.5 m, 0.27 h)`
(ratios read off the R13 reference crowns: mean diameter / height and mean crown length /
height). The Epanechnikov product kernel makes each step the plain centroid of the points
inside the cylinder. Converged modes are unioned within a merge tolerance and every point
takes the id of its mode's cluster; clusters below a point minimum are attached to the
nearest bigger one.

Provenance: `docs/experiments/ams3d_spike.py` and `2026-09-22-ams3d-spike.md` in the
`GEE_animation` repo, branch `als-ams3d-spike`. The spike ran four configs on a Spandau
hectare (r13, 376400–376500 E / 5827400–5827500 N) against the reference file and one on a
Tegeler Forst hectare (r12, 381300–381400 E / 5828300–5828400 N):

| config | change vs default | trees/ha (r13) | matched of 407 ref | recall | precision | mean diam [m] | height RMSE [m] |
|---|---|---|---|---|---|---|---|
| default | h_r = 0.27·h, merge 1 / 2 m, min 20 pts | 798 | 289 | 0.71 | 0.36 | 5.3 | 0.79 |
| A | h_r = 0.5·h, merge z 4 m | 518 | 227 | 0.56 | 0.44 | 5.6 | 1.02 |
| B | A + h_s = 0.15·h, min 50 pts | 290 | 180 | 0.44 | 0.62 | 7.8 | 0.84 |
| C | default kernel, merge 1.5 / 6 m, min 50 pts | 339 | 213 | 0.52 | 0.63 | 7.9 | 0.76 |

**Why config C is the default.** The textbook kernel splits tall leaf-off crowns
vertically: a point 12 m up a 30 m pine gets a 3 m vertical window and converges to a mode
inside the lower crown, so tall trees come out as 2–3 stacked clusters. Widening the
vertical bandwidth (A, B) helps less than merging modes afterwards and costs 2–3x runtime
because the neighbour sets grow. Merging modes within 1.5 m horizontally / 6 m vertically
and absorbing clusters under 50 points (C) removes the stacking and lands at the reference
density (339 vs 407 crowns/ha) with the best height agreement (RMSE 0.76 m); its crowns are
larger than the reference's (7.9 vs 5.8 m) because it also absorbs the understorey stems
that the reference keeps as separate 6–10 m trees.

The reference crown file the spike used is
`/Volumes/2TB/winmol/training_data/WINDWURF_Tegel/ALS segmentation 2017.gpkg` (layer
`ALS Kronenerfassung`, 538,587 MultiPolygons, EPSG:25833, extent 373000–379000 E /
5825000–5829472 N — R13 Spandau only, so it does not reach the Tegel tiles). Columns: `ID`,
`CentrdX/Y/Z`, `TreeH`, `CrwnBsH`, `CrwnLng`, `NPoints`, `CnvxHlA`, `CnvxHlP`, `CrwnRds`,
`CrwnDmt`, `TreeBB`, `DBH_prd`, `species`, `layer`, `path`. It is the closest thing to
reference crowns available for Spandau; nothing equivalent exists for Revier 12.

## 2. The port (`ff3d_geo/ams3d.py`)

`segment_ams3d(xyz, classification, Ams3dParams)` returns a per-point instance id (int32,
-1 for ground and anything not clustered); `Ams3dParams()` is config C, `CONFIGS` holds all
four, `params_for("C", min_points=30)` builds variants. `run_ams3d_tile()` reads one LAS,
height-normalises with the 1 m class-2 ground grid, segments classes 3/4/5 between 2 and
60 m above ground and writes the ForestFormer3D result contract (LAS 1.4 / pf6, EPSG:25833,
`treeID` int32, `semantic` uint8 with 0 ground / 2 leaf / 255 not classified — AMS3D has no
wood class and never labels understorey below 2 m — `score` float32 = -1, ALS
`classification` kept). `python -m ff3d_geo ams3d --las <tile> --out <dir> [--buffer 10]
[--workers N] [--config C]` runs a km tile: `split_las(buffer_m=10)` into 100 m sub-tiles
with 10 m of context, a `multiprocessing` pool over the sub-tiles, crop to the core, then
`merge_las`/`merge_trees`, `las_to_masks`, `build_report` — the same file set as the
ForestFormer3D `run` + `merge` + `masks` chain (`<T>.las`, `_trees.gpkg`, `_crowns.gpkg`,
`_instance_50cm.tif`, `_semantic_50cm.tif`, `_report.json/.md`), plus an `ams3d` block in
the report JSON with the config, parameters, worker count and per-sub-tile seconds.

**Differences from the spike: none in behaviour.** The mean shift, mode clustering and
small-cluster relabelling are the spike's code with the module globals moved into the
dataclass; the only numerical change is the local-coordinate origin (point-set minimum
instead of the patch corner), which is a translation and does not change the result.
Checked by rerunning the port on the spike's r12 patch (km tile 381/5828, 100 m + 10 m
buffer, config `default`): 658 trees with apex in the patch, mean crown diameter 5.399 m,
mean height 18.655 m, 931 clusters, 172,379 vegetation points — identical to
`ams3d_spike_r12_stats.json` to all printed digits (`tests/test_geo_ams3d.py`, marked
`slow`). The same check on the r13 patch with config C gives 399 trees, 577 clusters,
mean diameter 7.470 m — but `ams3d_spike_r13_C_stats.json` records 339 / 477 / 7.901 m.
Rerunning the *spike script itself* (its committed version, `python ams3d_spike.py
--patch r13 --merge_xy 1.5 --merge_z 6 --min_pts 50`) today gives 398 / 577 / 7.456 m,
matching the port to one tree and 1.4 cm (the residual is the translated origin changing
a few convergence decisions at the 5 cm tolerance). The spike's recorded C numbers
therefore come from an earlier, uncommitted state of its script — the file was committed
once, after all four runs — and the port is faithful to the committed code. The
reference matching of that rerun (237 of 407, recall 0.58, precision 0.60, RMSE 0.74 m)
is what config C actually does on that hectare; the write-up's 213 / 0.52 / 0.63 should
be read with that in mind. What the port adds around the algorithm:

* `threads` (cKDTree workers) defaults to 1 so a 48-process pool does not oversubscribe
  the host; the spike used every core for one patch.
* Sub-tile seams: each sub-tile is segmented on its buffered extent and only its core
  points are written, so the merged tile holds exactly the source points. A crown on a
  seam gets one id per side — the same border effect the ForestFormer3D split/merge
  pipeline has (`2026-09-22-tegel-berlin-2021.md`, section 3); the buffer makes the
  crowns near a seam complete on each side but does not stitch them.
* Ground/vegetation agreement in the report is trivially high: AMS3D's ground *is* ALS
  class 2. Compare the CHM baseline and height medians instead.

## 3. Results per tile

AMS3D config C on the same inputs as the ForestFormer3D runs (`inputs/r12_*`, `inputs/r13_*`
= the two 100 m tiles in local coordinates, no buffer available; `inputs/berlin/<T>.las` =
the km tiles, split into 100 m sub-tiles with a 10 m buffer). ForestFormer3D numbers from
`2026-09-22-tegel-als.md` (hectares) and `2026-09-22-tegel-berlin-2021.md` (km tiles).
"CHM" is the model-free local-maxima baseline of `ff3d_geo.baseline` (1 m CHM, 3 m window,
>= 3 m), identical for both methods. Ground agreement is model-ground vs ALS class 2 and
is 100 % for AMS3D by construction (its ground *is* class 2; 2–5 % of the points, the
vegetation below 2 m and the non-2/3/4/5 classes, are `semantic 255` and excluded).

| Tile | Method | Trees | CHM baseline | Height median (min / max), m | CHM median, m | Ground agreement | Wall time | Workers |
|---|---|---|---|---|---|---|---|---|
| r12 Tegel 1 ha | AMS3D C | 312 | 397 | 27.9 (5.3 / 34.7) | 29.1 | 100 % | 97 s | 1 CPU |
| r12 Tegel 1 ha | ForestFormer3D | 402 | 397 | 25.8 | 29.1 | 97.7 % | 67 s | 1 GPU |
| r13 Spandau 1 ha | AMS3D C | 407 | 400 | 22.6 (7.9 / 36.1) | 28.6 | 100 % | 301 s | 1 CPU |
| r13 Spandau 1 ha | ForestFormer3D | 443 | 400 | 24.6 | 28.6 | 98.2 % | 78 s | 1 GPU |
| `3dm_33_381_5829_1_be` 1 km² | AMS3D C | 20,417 | 34,809 | 22.3 (1.9 / 40.3) | 26.2 | 100 % | 19.4 min (1162 s) | 48 CPU |
| `3dm_33_381_5829_1_be` 1 km² | ForestFormer3D | 31,385 | 34,809 | 17.6 (-11.4 / 54.8) | 26.2 | 96.2 % | ~94 min | 1 GPU |
| `3dm_33_380_5828_1_be` 1 km² | AMS3D C | 28,810 | 39,876 | 24.3 (1.9 / 43.0) | 28.4 | 100 % | 24.8 min (1486 s) | 48 CPU |
| `3dm_33_380_5828_1_be` 1 km² | ForestFormer3D | 37,386 | 39,876 | 21.9 (-14.3 / 42.9) | 28.4 | 97.6 % | ~102 min | 4 GPUs x 25 sub-tiles |
| `3dm_33_381_5828_1_be` 1 km² | AMS3D C | 31,337 | 41,616 | 25.3 (1.8 / 41.8) | 29.4 | 100 % | 19.4 min (1166 s) | 48 CPU |
| `3dm_33_381_5828_1_be` 1 km² | ForestFormer3D | 37,058 | 41,616 | 25.0 (0.2 / 65.6) | 29.4 | 97.6 % | ~98 min | 1 GPU |

Spike numbers for the same hectares, for orientation (its patches carry the 10 m buffer
and count only crowns whose apex lies inside the hectare, so they are not the same
statistic as the tile counts above): r12 `default` 658 trees (port: 658), r13 `C` 339
recorded / 398 when the committed spike script is rerun today (port: 399, see section 2).

Runtime detail for `381_5829`: split 11 s; the 100 buffered sub-tiles (56,602–354,470
vegetation points each, median 215,967) took 5–1127 s each (median 244 s, 27,089 CPU-s in
total) in a 48-process pool, 1137 s wall; merge + masks + report 25 s. The first attempt
crawled at 13 runnable processes because numpy's and scipy's bundled OpenBLAS each spawned
a 64-thread pool per worker; the pipeline now pins them to one thread (commit 7996f3f).
The denser Revier 12 tiles took 24.8 min (`380_5828`, 35,799 CPU-s, sub-tiles 68–989 s)
and 19.4 min (`381_5828`, 40,583 CPU-s, 133–927 s) with the same 48 workers, run one
after the other.
AMS3D crown statistics on the km tile: median crown area 32.7 m², mean equivalent
diameter 7.44 m, median 296 points per tree — the same 7.4–7.9 m crowns as on the r13
hectare, i.e. larger than the 5.8 m reference mean.

Where the outputs are: `work_dirs/ams3d-{r12,r13,3dm_33_381_5829_1_be,3dm_33_380_5828_1_be,3dm_33_381_5828_1_be}`
on carrot, each with an `agreement_vs_ff3d.json` for the km tiles; the small files of
`381_5829` (reports, tree/crown GeoPackages, GeoTIFFs) are also under
`/Volumes/2TB/winmol/ALS_Data/berlin_als_2021_ams3d/3dm_33_381_5829_1_be/`. The merged
LAS files (865 MB each) were NOT copied: that volume was at 41 GB free before the copy
and 40 GB after, the agreed floor.

Seams: by construction no AMS3D crown crosses a 100 m sub-tile line (each sub-tile writes
only its core points), so a crown on a seam is one tree per side; 0.6 % of the hulls touch
a line. ForestFormer3D's split/merge has the same effect (16 % of its trees touch a
border, `2026-09-22-tegel-berlin-2021.md` section 3). The 10 m buffer makes the crowns
near a seam complete on each side but does not stitch them; stitching would need the
overlap-based id matching planned for ForestFormer3D (plan `fd954ce`), and would apply to
both methods alike.

## 4. Instance-level agreement with ForestFormer3D

`benchmark/ams3d_compare.py iou` matches the two segmentations of the same points by
point-set IoU with the Hungarian assignment at IoU >= 0.5 — the rule
`benchmark/instance_diagnostics.py` uses against ground truth — solved per connected
component of the overlap graph so the km tile's 31k x 20k matrix stays sparse. Neither
side is ground truth, so "matched" means the two methods drew the same tree, nothing
more.

| Tile | Trees FF3D / AMS3D | Matched (IoU >= 0.5) | Share of FF3D / of AMS3D | Mean IoU of matches | Points with an id, FF3D / AMS3D / both |
|---|---|---|---|---|---|
| r12 Tegel 1 ha | 402 / 312 | 175 | 43.5 % / 56.1 % | 0.75 | 107,410 / 121,308 / 105,510 |
| r13 Spandau 1 ha | 443 / 407 | 209 | 47.2 % / 51.4 % | 0.70 | 167,707 / 187,493 / 166,585 |
| `3dm_33_381_5829_1_be` | 31,385 / 20,417 | 9,550 | 30.4 % / 46.8 % | 0.73 | 12,645,220 / 14,031,877 / 12,014,782 |
| `3dm_33_380_5828_1_be` | 37,386 / 28,810 | 13,747 | 36.8 % / 47.7 % | 0.73 | 13,752,164 / 15,866,574 / 13,433,831 |
| `3dm_33_381_5828_1_be` | 37,058 / 31,337 | 14,404 | 38.9 % / 46.0 % | 0.72 | 13,777,125 / 16,155,840 / 13,551,202 |

Reading: nearly every point ForestFormer3D assigns to a tree is also assigned by AMS3D
(98 % on the hectares, 95 % on the km tile), and AMS3D assigns 10–15 % more points (it
labels every vegetation point above 2 m, ForestFormer3D leaves low-score points out). The
unmatched trees are therefore mostly a different *partition* of the same points: AMS3D's
crowns are larger and fewer, ForestFormer3D splits them. On the km tile the agreement
drops to 30 % of ForestFormer3D's trees, which is where ForestFormer3D's count also runs
furthest from the CHM baseline and its height median lowest (17.6 m vs 26.2 m CHM): the
sub-canopy trees it adds there have no AMS3D counterpart at IoU 0.5.

## 5. Both methods against the R13 reference crowns

`ALS segmentation 2017.gpkg` covers the r13 hectare (407 crowns with centroid inside).
`benchmark/ams3d_compare.py reference` applies the spike's rule: a predicted tree matches
when its apex (highest point) lies inside a reference polygon and the heights differ by
<= 3 m, one match per reference crown, the tallest prediction winning. Heights are the
`height` column of the `_trees.gpkg` (top z minus the ground grid at the stem).

| Method (r13 hectare) | Trees with apex in the hectare | Matched of 407 | Recall | Precision | Height RMSE of matches | Mean height (ref 22.8 m) |
|---|---|---|---|---|---|---|
| AMS3D C (port, `inputs/r13_*`, no buffer) | 407 | 238 | 0.58 | 0.58 | 0.75 m | 23.5 m |
| ForestFormer3D (`work_dirs/tegel-r13`) | 443 | 245 | 0.60 | 0.55 | 0.62 m | 22.8 m |
| spike C, recorded 2026-09-22 | 339 | 213 | 0.52 | 0.63 | 0.76 m | 24.0 m |
| spike C, committed script rerun 2026-09-23 | 398 | 237 | 0.58 | 0.60 | 0.74 m | 23.5 m |

Both methods sit within a handful of matches of each other on this hectare; the
reference itself keeps understorey stems as separate 6–10 m trees that neither method
resolves as such, which caps the recall of both at about 0.6. ForestFormer3D's height
RMSE is lower (0.62 vs 0.75 m) because its tree heights come from the model-ground grid
at the stem, AMS3D's from the class-2 grid at the stem of a larger crown.

## 6. Reading the numbers

* **AMS3D under-counts against the CHM baseline on the km tiles** (-41 % on `381_5829`,
  -28 % on `380_5828`, -25 % on `381_5828`) where ForestFormer3D is at -6 to -11 %. Its config C was tuned on the closed Spandau stand to
  merge stacked modes; on the more open Tegel pine stands and the settlement edges of
  `381_5829` the same 6 m vertical merge also swallows neighbours, and the 2 m height
  floor plus the 50-point minimum drop every small tree. The hectares show the milder
  version of this (312 vs 397 on r12).
* **Heights**: AMS3D's height medians are within 1.2–6 m of the CHM medians and its
  minima are >= 1.9 m by construction; ForestFormer3D's report on the km tile has a
  -11.4 m minimum and a 17.6 m median from stems placed on the model-ground grid, the
  known artefact discussed in `2026-09-22-tegel-berlin-2021.md` section 5. The report's
  "first pass usable" rule (count within +-50 % of CHM, median height within 3 m) says
  **no** for AMS3D on the km tile because of the count, and no for ForestFormer3D because
  of the height; on the hectares AMS3D passes both (r12: 312 vs 397 = -21 %, 27.9 vs
  29.1 m; r13: 407 vs 400, 22.6 vs 28.6 m fails the height rule by 6 m).
* **What AMS3D gives that ForestFormer3D does not**: no training data, no GPU, ~19 min per
  km tile on 48 cores, and explicit, physically named parameters. What it lacks: any
  wood/leaf separation, any understorey below 2 m, any per-instance confidence, and it
  never learned what a tree looks like — every parameter is a stand-dependent guess that
  the spike calibrated against one hectare of an unvalidated crown file.
* **Neither number is accuracy.** The two methods' agreement (30–47 % of ForestFormer3D's
  trees at IoU 0.5) and the reference matching (0.58–0.60 recall for both) bound how far
  apart two reasonable segmentations of the same leaf-off ALS are; a hand-checked plot is
  needed before either count is trusted.

## 7. Reproducing

```bash
# carrot, host venv, no GPU
cd /raid/cwinkelmann/ForestFormer3D && source /raid/cwinkelmann/ff3d-geo-venv/bin/activate
python -m ff3d_geo ams3d --las inputs/r12_tegel_E381300_N5828300_100m.las --out work_dirs/ams3d-r12 --workers 1
python -m ff3d_geo ams3d --las inputs/r13_spandau_E376400_N5827400_100m.las --out work_dirs/ams3d-r13 --workers 1
T=3dm_33_381_5829_1_be
nohup bash -c "time python -m ff3d_geo ams3d --las inputs/berlin/$T.las --out work_dirs/ams3d-$T --workers 48" > work_dirs/logs/ams3d-$T.log 2>&1 &
python benchmark/ams3d_compare.py iou work_dirs/berlin-$T/$T.las work_dirs/ams3d-$T/$T.las
# Mac (the reference gpkg lives on the 2TB volume)
python benchmark/ams3d_compare.py reference <r13 result.las> "/Volumes/2TB/winmol/training_data/WINDWURF_Tegel/ALS segmentation 2017.gpkg" \
  --bbox 376400 5827400 376500 5827500 --trees <r13 result_trees.gpkg>
```
