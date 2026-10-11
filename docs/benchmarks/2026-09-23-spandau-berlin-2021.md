# Berlin ALS 2021 inference over the eight km tiles around Revier 13 Spandau

Date: 2026-09-23 (run: 15:04 - 17:10 CEST)
Branch / commit: `fix/review-findings`; carrot's checkout was at
`318785c bench: ALS-density doc fixes` when the queues were launched.
Machine: carrot (8x H100), image `forestformer3d:cu118`, checkpoint
`work_dirs/clean_forestformer/epoch_3000_fix.pth`, host venv `/raid/cwinkelmann/ff3d-geo-venv`.
Commands: `work_dirs/logs/berlin-run-gpu.sh <gpu> <tile>...` (tracked as
`benchmark/berlin_run_gpu.sh`; see `docs/benchmarks/RUNBOOK-tegel.md` sections 8-9 and the
`ff3d-inference-km-tiles` skill), one `nohup`'d queue per GPU, logs
`work_dirs/logs/berlin-r13-gpu{0,3,4,5,6}-2026*.log`.

Eight 1 km x 1 km tiles covering the E 374-378 / N 5827-5829 block around Revier 13
Spandau: **153.6 M points, 741 sub-tiles, 233,858 predicted trees, 2 h 06 min wall clock**
across five GPUs. This is the second km-scale block after the three Revier 12 Tegelsee
tiles (`docs/benchmarks/2026-09-22-tegel-berlin-2021.md`), and the first run with the
vectorised z-filter and binary PLY writer.

## 1. GPU assignment and runtime

GPU 1 (another user) and GPU 2 (busy at 100 %) were excluded; GPU 7 was reserved for
another agent's job. The eight tiles were distributed over GPUs 0, 3, 4, 5 and 6, three
queues taking two tiles:

| GPU | Tiles (in queue order) |
|---|---|
| 0 | `374_5827`, `374_5828` |
| 3 | `375_5827`, `375_5828` |
| 4 | `376_5827`, `376_5828` |
| 5 | `377_5827` |
| 6 | `377_5828` |

| Tile | Sub-tiles | `run` (s) | s / sub-tile | Wall time (split -> tile done) |
|---|---|---|---|---|
| `3dm_33_374_5827_1_be` | 98 | 2746 | 28.0 | 46 min 11 s |
| `3dm_33_375_5827_1_be` | 100 | 3180 | 31.8 | 53 min 24 s |
| `3dm_33_376_5827_1_be` | 100 | 3474 | 34.7 | 58 min 20 s |
| `3dm_33_377_5827_1_be` | 100 | 3480 | 34.8 | 58 min 27 s |
| `3dm_33_374_5828_1_be` | 81 | 3367 | 41.6 | 56 min 40 s |
| `3dm_33_375_5828_1_be` | 100 | 3292 | 32.9 | 55 min 17 s |
| `3dm_33_376_5828_1_be` | 99 | 3380 | 34.1 | 56 min 48 s |
| `3dm_33_377_5828_1_be` | 63 | 1996 | 31.7 | 33 min 48 s |

Block wall clock 15:04:11 - 17:09:52 = **2 h 05 min 41 s** for 741 sub-tiles. All 741
succeeded; no `!!!` line appears in any queue log, no sub-tile was skipped by `run`, and
every tile produced its merged LAS, tree and crown GeoPackages, both 50 cm masks and the
report pair. Sub-tile counts below 100 are `split`'s `--min-points` floor doing its job:
tiles `374_5828` (81) and `377_5828` (63) reach beyond the Havel and the city boundary.

28-42 s per sub-tile against 55-60 s in the Tegel block is a 1.6-2.0x speed-up, well short
of the 6.7x the vectorisation measured in isolation - five queues sharing one host's
CPU and I/O is the limiting factor, the same host-latency effect the Tegel block saw at
seven GPUs. Per *tile* the gain is much larger because five tiles now run at once: 2 h 06
min for eight tiles here against 4 h 47 min for three tiles on one GPU.

## 2. Input and results per km tile

| Tile | Points | Sub-tiles | Trees (model) | CHM baseline | vs CHM | Median height, model / CHM (m) | Ground agreement | First pass usable | Wall time |
|---|---|---|---|---|---|---|---|---|---|
| `3dm_33_374_5827_1_be` | 17,522,555 | 98 | 26,870 | 29,899 | -10.1 % | 17.1 / 20.8 | 93.4 % | no | 46 min |
| `3dm_33_375_5827_1_be` | 21,086,767 | 100 | 30,011 | 32,949 | -8.9 % | 22.2 / 24.0 | 95.0 % | **yes** | 53 min |
| `3dm_33_376_5827_1_be` | 22,729,769 | 100 | 38,019 | 39,466 | -3.7 % | 22.8 / 25.6 | 96.8 % | **yes** | 58 min |
| `3dm_33_377_5827_1_be` | 23,080,082 | 100 | 40,001 | 41,082 | -2.6 % | 23.5 / 26.9 | 97.0 % | no | 58 min |
| `3dm_33_374_5828_1_be` | 12,676,377 | 81 | 13,387 | 18,491 | -27.6 % | 14.3 / 21.1 | 92.2 % | no | 57 min |
| `3dm_33_375_5828_1_be` | 22,021,451 | 100 | 30,959 | 36,185 | -14.4 % | 21.6 / 24.7 | 94.8 % | no | 55 min |
| `3dm_33_376_5828_1_be` | 21,216,255 | 99 | 33,407 | 37,513 | -10.9 % | 20.9 / 24.7 | 95.0 % | no | 57 min |
| `3dm_33_377_5828_1_be` | 13,305,340 | 63 | 21,204 | 23,477 | -9.7 % | 20.8 / 26.5 | 97.1 % | no | 34 min |
| **total / range** | **153,638,596** | **741** | **233,858** | **259,062** | -9.7 % | -1.9 to -6.8 | 92.2-97.1 % | 2 of 8 | 2 h 06 min |

All eight are LAS 1.4 / point format 6, EPSG:25833, and the merged point counts equal the
source point counts exactly. Nodata fraction (semantic 255) is 0.0 % on every tile.

### Which tiles intersect the Revier 13 AOI

The AOI (E 374469-377184, N 5827475-5828037) formally clips all eight km tiles, but the
N 5828 row is cut at 5828037, so those four tiles contribute a 37 m strip each - 2-4 % of
a tile. By intersected area the AOI is carried by the N 5827 row, and the three tiles that
hold essentially all of it are **`375_5827` (0.53 km2), `376_5827` (0.53 km2) and
`374_5827` (0.28 km2)**; `377_5827` adds a 184 m strip (0.10 km2) and the four 5828 tiles
0.007-0.037 km2 each. The three `_report.md` blocks below are those three tiles.

### 3dm_33_374_5827_1_be

| Metric | Value |
|---|---|
| Points | 17522555 |
| Inference runtime | 2746 s |
| Trees (model) | 26870 |
| Trees (CHM local maxima baseline) | 29899 |
| Height min / median / max (model, m) | 0.0 / 17.1 / 33.0 |
| Height min / median / max (CHM, m) | 3.0 / 20.8 / 32.9 |
| Ground vs vegetation agreement | 93.4 % |
| Nodata fraction (semantic 255, excluded above) | 0.0 % |

| Model \ ALS | class 2 (ground) | other |
|---|---|---|
| semantic 0 (ground) | 7011149 | 506497 |
| semantic 1/2 (wood/leaf) | 658287 | 9346622 |

First pass usable: **no** (tree count 26870 vs CHM baseline 29899 (ok); median height 17.1 m vs CHM median 20.8 m (outside 3 m))

### 3dm_33_375_5827_1_be

| Metric | Value |
|---|---|
| Points | 21086767 |
| Inference runtime | 3180 s |
| Trees (model) | 30011 |
| Trees (CHM local maxima baseline) | 32949 |
| Height min / median / max (model, m) | 0.0 / 22.2 / 38.0 |
| Height min / median / max (CHM, m) | 3.0 / 24.0 / 37.4 |
| Ground vs vegetation agreement | 95.0 % |
| Nodata fraction (semantic 255, excluded above) | 0.0 % |

| Model \ ALS | class 2 (ground) | other |
|---|---|---|
| semantic 0 (ground) | 7705644 | 569474 |
| semantic 1/2 (wood/leaf) | 494841 | 12316808 |

First pass usable: **yes** (tree count 30011 vs CHM baseline 32949 (ok); median height 22.2 m vs CHM median 24.0 m (ok))

### 3dm_33_376_5827_1_be

| Metric | Value |
|---|---|
| Points | 22729769 |
| Inference runtime | 3474 s |
| Trees (model) | 38019 |
| Trees (CHM local maxima baseline) | 39466 |
| Height min / median / max (model, m) | 0.1 / 22.8 / 38.5 |
| Height min / median / max (CHM, m) | 3.0 / 25.6 / 38.6 |
| Ground vs vegetation agreement | 96.8 % |
| Nodata fraction (semantic 255, excluded above) | 0.0 % |

| Model \ ALS | class 2 (ground) | other |
|---|---|---|
| semantic 0 (ground) | 7465744 | 573120 |
| semantic 1/2 (wood/leaf) | 149819 | 14541086 |

First pass usable: **yes** (tree count 38019 vs CHM baseline 39466 (ok); median height 22.8 m vs CHM median 25.6 m (ok))

## 3. Comparison with the Revier 12 Tegelsee block

The Tegel block (`docs/benchmarks/2026-09-22-tegel-berlin-2021.md`: `380_5828`,
`381_5828`, `381_5829`, 73.5 M points, 105,829 trees) and this Spandau block fail and pass
in the same places, but Spandau is measurably closer to the decision rule. The tree-count
criterion passes on all eleven tiles of both blocks, and in both the model lands *below*
the CHM baseline - 6.2-11.0 % under at Tegel, 2.6-27.6 % under here (median 9.9 %), so
this is still not over-segmentation. The height criterion is where they differ: at Tegel
the model median was 4.4, 6.5 and 8.6 m below the CHM median and **no** tile passed, while
here the gap is 1.9-6.8 m (median 3.6 m) and **two of eight tiles pass** - `375_5827`
(-1.9 m) and `376_5827` (-2.8 m), the two tiles that carry most of the Revier 13 AOI. The
extreme model heights are also tamer: Tegel produced 65.6 m and 54.8 m instances against
CHM maxima of 42.0 and 40.1 m, whereas here only `377_5827` (52.0 m model vs 37.7 m CHM)
shows that signature clearly and five tiles have model maxima within 1 m of the CHM
maximum, with model minima at or above 0 m on seven of eight tiles (`374_5828` is the
exception at -1.3 m; Tegel had -14.3 and -11.4 m). Ground-vs-vegetation agreement is
slightly worse - 92.2-97.1 % here against 96.2-97.6 % at Tegel, and the Tegel asymmetry
(vegetation called ground more often than the reverse) holds on seven tiles at 1.2-4.0x
but inverts on `374_5827`, where the model instead over-calls vegetation (658 k ground
points labelled vegetation against 506 k the other way); the two weakest tiles,
`374_5828` (92.2 %) and `374_5827`
(93.4 %), are the western Havel-edge tiles with water, reed and open shore. That the
verdict softened rather than flipped is consistent with the Tegel reading: the median gap
is a shift of the whole height distribution that depends on stand composition, not a
tail of short spurious instances, so a block of somewhat shorter, more open Spandau stands
lands nearer the 3 m band without the underlying behaviour having changed. The
recommendation from the Tegel report stands unchanged - **fine-tune the instance head on
ALS-density data**; the semantic head continues to transfer intact.

## 4. Where the results are

- carrot: `work_dirs/berlin-3dm_33_<E>_<N>_1_be/` (per-sub-tile files, merged files,
  masks), inputs `inputs/berlin/3dm_33_<E>_<N>_1_be.las`, split inputs
  `inputs/berlin/sub/<tile>/`, logs `work_dirs/logs/berlin-r13-gpu*.log`.
- 2TB volume: `/Volumes/2TB/winmol/ALS_Data/berlin_als_2021_ff3d/<tile>/` - one directory
  per km tile with `<T>.las`, `<T>_trees.gpkg`, `<T>_crowns.gpkg`,
  `<T>_instance_50cm.tif`, `<T>_semantic_50cm.tif` and `<T>_report.json` / `.md`
  (484-886 MB per tile, 5.9 GB in total). Per-sub-tile `*_100m*` files and all `.ply`
  files were deliberately not copied.
- Orthophotos for the same eight tiles:
  `/Volumes/2TB/winmol/ALS_Data/berlin_dop_2021/dop_2021_rgbi/` (DOP20 2021 leaf-off,
  4-band R,G,B,NIR, 72.5-75.2 MB per tile) and
  `/Volumes/2TB/winmol/ALS_Data/berlin_dop_2025_sommer/` (TrueDOP20 2025 summer,
  27.4-81.2 MB per tile), both fetched with
  `benchmark/fetch_berlin_dop.py --rgbi`, 16 tiles, no failures.

## 5. Buildings: ALKIS footprints masked out of the tree instances

The Berlin ALS 2021 point clouds have **no building class** (classes present: 2, 3, 4, 5,
7, 32; class 6 absent, roof points in the vegetation bins 3/4/5), so ForestFormer3D reads
a roof as a crown and predicts tree instances on buildings. The correction comes from
Berlin's official ALKIS footprints, fetched with `benchmark/fetch_berlin_buildings.py`
from the GDI WFS `https://gdi.berlin.de/services/wfs/alkis_gebaeude` (feature type
`alkis_gebaeude:gebaeude`) and applied with `python -m ff3d_geo buildings`; the mechanics
and the Tegel numbers are in
`docs/benchmarks/2026-09-22-tegel-berlin-2021.md` section 7.

Revier 13 is the opposite case to Tegel: the eight km tiles are almost pure forest, and
the WFS returns **0 footprints** for `376_5827`, `376_5828`, `377_5827` and `377_5828`,
1 each for `375_5827`/`375_5828` and 5 / 11 for `374_5827`/`374_5828`. Sixteen footprints
over the whole block, against 5 244 over the eleven Tegel tiles.

| Tile | Instances before | Removed | Partially masked | Points masked | Footprints in tile | Instances after |
|---|---|---|---|---|---|---|
| `3dm_33_374_5827_1_be` | 26 870 | 2 | 3 | 3 617 | 5 | 26 868 |
| `3dm_33_374_5828_1_be` | 13 387 | 6 | 19 | 9 344 | 11 | 13 381 |
| **total** | **40 257** | **8** | **22** | **12 961** | **16** | **40 249** |

So the building mask is effectively a no-op here: 8 instances of 40 257 on the two tiles
it could be run on, three orders of magnitude below Tegel's 3.6 %. It is still worth
running - it costs ~100 s per tile and it is the same command - but nothing in section 2
changes because of it.

Not yet masked, for lack of free space on the 2TB volume (the run stops below 40 GB
free): `375_5827`, `375_5828`, `376_5827`, `376_5828`, `377_5827`, `377_5828`. The four
tiles with no footprints at all cannot change; the two `375_*` tiles have one footprint
each.

## 6. Reproducing

```bash
# Mac: copy the tiles up
for T in 374_5827 375_5827 376_5827 377_5827 374_5828 375_5828 376_5828 377_5828; do
  scp /Volumes/2TB/winmol/ALS_Data/berlin_als_2021/3dm_33_${T}_1_be.las \
      carrot:/raid/cwinkelmann/ForestFormer3D/inputs/berlin/
done

# carrot: one queue per free GPU
cd /raid/cwinkelmann/ForestFormer3D
TS=$(date +%Y%m%d-%H%M%S)
nohup bash work_dirs/logs/berlin-run-gpu.sh 0 3dm_33_374_5827_1_be 3dm_33_374_5828_1_be \
  > work_dirs/logs/berlin-r13-gpu0-$TS.log 2>&1 &
# ... GPU 3 -> 375_*, GPU 4 -> 376_*, GPU 5 -> 3dm_33_377_5827_1_be,
#     GPU 6 -> 3dm_33_377_5828_1_be

# Mac: orthophotos
.venv-cpu/bin/python benchmark/fetch_berlin_dop.py --service dop_2021 --rgbi \
  --tiles 374_5827 375_5827 376_5827 377_5827 374_5828 375_5828 376_5828 377_5828 \
  --out-dir /Volumes/2TB/winmol/ALS_Data/berlin_dop_2021/dop_2021_rgbi
.venv-cpu/bin/python benchmark/fetch_berlin_dop.py --service truedop_2025_sommer --rgbi \
  --tiles 374_5827 375_5827 376_5827 377_5827 374_5828 375_5828 376_5828 377_5828 \
  --out-dir /Volumes/2TB/winmol/ALS_Data/berlin_dop_2025_sommer

# Mac: bring the merged products home, one directory per km tile
for T in 374_5827 375_5827 376_5827 377_5827 374_5828 375_5828 376_5828 377_5828; do
  K=3dm_33_${T}_1_be
  mkdir -p /Volumes/2TB/winmol/ALS_Data/berlin_als_2021_ff3d/$K
  rsync -a --exclude '*.ply' --exclude '*_100m*' \
    carrot:/raid/cwinkelmann/ForestFormer3D/work_dirs/berlin-$K/ \
    /Volumes/2TB/winmol/ALS_Data/berlin_als_2021_ff3d/$K/
done
```

QGIS screenshots and DOP crown overlays for this block are **pending**.

Note on free space: the 2TB volume was at 53 GB free after this block, so the residue on
carrot (`data/ForAINetV2/test_data/*_100m.ply` and the matching `.npy` exports, 1-2 GB per
km tile - see `ff3d-inference-km-tiles` section 6) has not been cleaned up yet and the
split inputs under `inputs/berlin/sub/` are still in place.
