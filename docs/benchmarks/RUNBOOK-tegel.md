# Runbook: ForestFormer3D on the Berlin ALS tiles (Tegel r12, Spandau r13)

Everything below was verified on the Mac except the GPU steps, which need `ssh carrot`
(VPN or SSH alias). Paths on carrot: checkout `/raid/cwinkelmann/ForestFormer3D`
(mounted at `/workspace` inside the container), checkpoint
`work_dirs/clean_forestformer/epoch_3000_fix.pth`, image `forestformer3d:cu118`.

`python -m ff3d_geo run` is a **host-side orchestrator**: it runs in a plain Python venv
on the host and launches only the two GPU steps in the container, through
`bash -c 'source benchmark/common.sh; ff3d_docker ...'` with `FF3D_GPU` taken from
`--gpu`. Nothing is ever pip-installed into the image.

## 0. Sanity on the Mac (no GPU)

```bash
cd ~/ForestFormer3D
python3 -m venv .venv-cpu && .venv-cpu/bin/pip install -r tests/requirements-cpu.txt   # once
.venv-cpu/bin/python -m pytest tests/test_geo_*.py -q            # 42 passed
.venv-cpu/bin/python -m ff3d_geo run \
  --las ~/work/hnee/ForestFormer3D_runs/berlin_in/r12_tegel_E381300_N5828300_100m.las \
  --checkpoint work_dirs/clean_forestformer/epoch_3000_fix.pth \
  --out work_dirs/tegel-r12 --gpu 5 --dry-run     # prints the 8 steps, runs nothing
```

## 1. Copy the tiles to carrot

```bash
ssh carrot mkdir -p /raid/cwinkelmann/ForestFormer3D/inputs
scp ~/work/hnee/ForestFormer3D_runs/berlin_in/r12_tegel_E381300_N5828300_100m.las \
    ~/work/hnee/ForestFormer3D_runs/berlin_in/r13_spandau_E376400_N5827400_100m.las \
    carrot:/raid/cwinkelmann/ForestFormer3D/inputs/
```

## 2. Update the checkout on carrot

```bash
ssh carrot
cd /raid/cwinkelmann/ForestFormer3D
git fetch origin && git checkout fix/review-findings && git pull --ff-only origin fix/review-findings
git log --oneline -1          # note the commit hash for the report
```

The host venv used by the CLI is `/raid/cwinkelmann/ff3d-geo-venv` (Python 3.12), made
once with `python3 -m venv /raid/cwinkelmann/ff3d-geo-venv && /raid/cwinkelmann/ff3d-geo-venv/bin/pip install -r tests/requirements-cpu.txt`.

## 3. Run both tiles

Run them **sequentially** on one free GPU. GPUs 2-4 may be busy with benchmark jobs;
`--gpu 5` pins this work to card 5 and nothing else is touched.

```bash
mkdir -p /raid/cwinkelmann/ForestFormer3D/work_dirs/logs
cd /raid/cwinkelmann/ForestFormer3D && source /raid/cwinkelmann/ff3d-geo-venv/bin/activate

python -m ff3d_geo run --las inputs/r12_tegel_E381300_N5828300_100m.las \
  --checkpoint work_dirs/clean_forestformer/epoch_3000_fix.pth \
  --out work_dirs/tegel-r12 --gpu 5 \
  2>&1 | tee work_dirs/logs/tegel-r12-$(date +%Y%m%d-%H%M%S).log

python -m ff3d_geo run --las inputs/r13_spandau_E376400_N5827400_100m.las \
  --checkpoint work_dirs/clean_forestformer/epoch_3000_fix.pth \
  --out work_dirs/tegel-r13 --gpu 5 \
  2>&1 | tee work_dirs/logs/tegel-r13-$(date +%Y%m%d-%H%M%S).log
```

`--las` accepts several tiles in one call (`--las a.las b.las ...`): they share one
preprocess and one inference step, and each tile still gets its own `<out>/<stem>.las`,
`<stem>_trees.gpkg` and `<stem>_report.json/.md`. That is what section 8 uses for the km
tiles; for the two 100 m tiles above, one call per tile keeps the logs readable.

For a long run, put it in the background instead and poll the log:
`nohup python -m ff3d_geo run ... > work_dirs/logs/<name>.log 2>&1 &`, then
`tail -5 work_dirs/logs/<name>.log` every minute or two. A 100 m ALS tile is minutes,
not hours; if `tools/test.py` starts iterating over more than one scan, the scan list
leaked - kill that run's PID only and check `<out>/scan_list.txt`.

Expected per run: the 8 step lines (`# python: ...` / `$ cd ...`), `batch_load` printing
the `--unlabeled` constant-label path, `create_data` (`--splits test`) silently building
only the test split (train/val are simply not built, no message), the `tools/test.py`
output, then the markdown report block ending in `First pass usable: **yes|no**`.

Files afterwards in `work_dirs/tegel-r12/` (a real run):

```
20260922_170517/                                     # mmengine log dir
empty_list.txt                                       # empty train/val scan lists
forainetv2_oneformer3d_infos_test.pkl                # test info pkl, private to this run
oneformer3d_qs_radius16_qp300_2many.py               # config copy dumped by tools/test.py
r12_tegel_E381300_N5828300_100m.las                  # georeferenced result
r12_tegel_E381300_N5828300_100m.ply                  # raw model result
r12_tegel_E381300_N5828300_100m.sidecar.json         # origin / EPSG / point count
r12_tegel_E381300_N5828300_100m_classification.npy   # ALS class per point, for the baseline
r12_tegel_E381300_N5828300_100m_report.json
r12_tegel_E381300_N5828300_100m_report.md
r12_tegel_E381300_N5828300_100m_trees.gpkg
scan_list.txt                                        # the single tile stem
```

Sanity numbers for r12 (model-independent): 192,814 points, CHM baseline 397 local maxima
with the default 1 m / 3 m / 3 m settings, ALS ground 66,337 points. If
`chm_baseline_count` differs, the classification did not survive the round trip.

If a run dies after inference, do not repeat inference - finish it from the host venv:

```bash
cd /raid/cwinkelmann/ForestFormer3D && source /raid/cwinkelmann/ff3d-geo-venv/bin/activate
python -m ff3d_geo georef \
  --result-ply work_dirs/tegel-r12/r12_tegel_E381300_N5828300_100m.ply \
  --sidecar work_dirs/tegel-r12/r12_tegel_E381300_N5828300_100m.sidecar.json \
  --offsets data/ForAINetV2/forainetv2_instance_data/r12_tegel_E381300_N5828300_100m_offsets.npy \
  --out-las work_dirs/tegel-r12/r12_tegel_E381300_N5828300_100m.las \
  --gpkg work_dirs/tegel-r12/r12_tegel_E381300_N5828300_100m_trees.gpkg
python -m ff3d_geo report \
  --las work_dirs/tegel-r12/r12_tegel_E381300_N5828300_100m.las \
  --gpkg work_dirs/tegel-r12/r12_tegel_E381300_N5828300_100m_trees.gpkg \
  --json work_dirs/tegel-r12/r12_tegel_E381300_N5828300_100m_report.json \
  --md work_dirs/tegel-r12/r12_tegel_E381300_N5828300_100m_report.md \
  --runtime-s <seconds of the tools/test.py step from the log>
```

## 4. What the run leaves behind

The CLI never touches the tracked `data/ForAINetV2/meta_data/test_list.txt`: it writes its
own `scan_list.txt` / `empty_list.txt` into `<out>` and passes them to
`batch_load_ForAINetV2_data.py` and `tools/create_data_forainetv2.py`, and the test info
pkl is written into `<out>` too, so the benchmark test set is unaffected and nothing needs
restoring.

The converted tiles do stay behind as untracked files in `data/ForAINetV2/test_data/` and
`data/ForAINetV2/forainetv2_instance_data/`. Delete them if you want the data dir clean:

```bash
rm data/ForAINetV2/test_data/r1?_*_100m.ply data/ForAINetV2/forainetv2_instance_data/r1?_*_100m_*.npy
```

## 5. Copy the results back to the Mac

```bash
mkdir -p ~/work/hnee/ForestFormer3D_runs/berlin_out
rsync -av --exclude '*.ply' carrot:/raid/cwinkelmann/ForestFormer3D/work_dirs/tegel-r12/ ~/work/hnee/ForestFormer3D_runs/berlin_out/tegel-r12/
rsync -av --exclude '*.ply' carrot:/raid/cwinkelmann/ForestFormer3D/work_dirs/tegel-r13/ ~/work/hnee/ForestFormer3D_runs/berlin_out/tegel-r13/
rsync -av carrot:/raid/cwinkelmann/ForestFormer3D/work_dirs/logs/ ~/work/hnee/ForestFormer3D_runs/berlin_out/logs/
```

Verify on the Mac (no torch needed, `.venv-cpu` is enough):

```bash
.venv-cpu/bin/python -c "import laspy; l = laspy.read('$HOME/work/hnee/ForestFormer3D_runs/berlin_out/tegel-r12/r12_tegel_E381300_N5828300_100m.las'); print(l.header.version, l.header.point_format.id, l.header.parse_crs().to_epsg(), l.header.point_count, l.header.mins, l.header.maxs, list(l.point_format.extra_dimension_names))"
# 1.4 6 25833 192814 [ 381300. 5828300. 0.] [ 381399.99 5828399.99 36.83] ['treeID', 'semantic', 'score']
.venv-cpu/bin/python -c "import geopandas as g; d = g.read_file('$HOME/work/hnee/ForestFormer3D_runs/berlin_out/tegel-r12/r12_tegel_E381300_N5828300_100m_trees.gpkg', layer='trees'); print(len(d), d.crs.to_epsg(), d.height.describe()[['min','50%','max']].round(1).tolist())"
```

The second line must match `n_trees` and the height quantiles in `..._report.json`.

## 6. Look at it in QGIS (3.28 or newer)

1. New project. Project > Properties > CRS: `EPSG:25833`.
2. Drag `r12_tegel_E381300_N5828300_100m.las` into the map (it opens as a point cloud
   layer; the CRS is read from the WKT VLR). Layer styling: Symbology "Attribute by Ramp",
   attribute `treeID` (or `semantic` for ground/wood/leaf).
3. Drag `r12_tegel_E381300_N5828300_100m_trees.gpkg` into the map, layer `trees`. Style:
   single symbol, size by `height` (Data defined override, `"height" / 2`), label `tree_id`.
4. Zoom to the LAS layer. Optional: add an XYZ basemap (OpenStreetMap) to see the tile
   sits in Tegel forest.
5. Project > Import/Export > Export Map to Image, 1600 px wide, save as
   `docs/benchmarks/assets/2026-09-22-tegel-r12-qgis.png`. Repeat for r13.
6. Optional 3D check: View > 3D Map Views > New 3D Map View, LAS layer with
   "Attribute by Ramp" on `treeID`.

## 7. Fill the report

Paste `<stem>_report.md` from each `berlin_out/tegel-r1?/` into
`docs/benchmarks/2026-09-22-tegel-als.md`, add the screenshots, complete the
recommendation with the decision rule printed in the block, commit.

## 8. Berlin ALS 2021 km tiles

The three 1 km x 1 km tiles around Revier 12 Tegelsee (`3dm_33_380_5828_1_be`,
`3dm_33_381_5828_1_be`, `3dm_33_381_5829_1_be`) are far bigger than the ~100 m extent
the model is trained on, so each one is cut into 100 m sub-tiles, inferred as a batch,
and stitched back together:

```bash
cd /raid/cwinkelmann/ForestFormer3D && source /raid/cwinkelmann/ff3d-geo-venv/bin/activate
mkdir -p work_dirs/logs
for T in 3dm_33_380_5828_1_be 3dm_33_381_5828_1_be 3dm_33_381_5829_1_be; do
  python -m ff3d_geo split --las inputs/berlin/$T.las --out inputs/berlin/sub/$T > work_dirs/logs/split-$T.txt
  python -m ff3d_geo run --las inputs/berlin/sub/$T/*.las --checkpoint work_dirs/clean_forestformer/epoch_3000_fix.pth \
      --out work_dirs/berlin-$T --gpu 5 2>&1 | tee work_dirs/logs/berlin-$T-$(date +%Y%m%d-%H%M%S).log
  python -m ff3d_geo merge --las work_dirs/berlin-$T/*_100m.las --gpkg work_dirs/berlin-$T/*_100m_trees.gpkg \
      --out-las work_dirs/berlin-$T/$T.las --out-gpkg work_dirs/berlin-$T/${T}_trees.gpkg \
      --report-json work_dirs/berlin-$T/${T}_report.json --report-md work_dirs/berlin-$T/${T}_report.md
done
```

Notes:

- `split` prints one sub-tile path per line (also captured in `split-$T.txt`); a ~23 M
  point km tile splits in about five seconds and yields up to 100 sub-tiles named
  `<prefix>_E<easting>_N<northing>_100m.las` in local coordinates, with sub-tiles under
  `--min-points` (default 1000) left out entirely - water, roofs and tile edges.
- `--las inputs/berlin/sub/$T/*.las` relies on the **shell** to expand the glob into one
  batched `run`: one preprocess and one `tools/test.py` pass for the whole km tile. The
  sub-tile LAS files in `work_dirs/berlin-$T/` are the georeferenced *results* named
  `<stem>.las`, distinct from those inputs under `inputs/berlin/sub/`.
- Both `merge` globs expand in the same sorted order, which is what makes them line up;
  the command still checks pairwise that `<stem>.las` goes with `<stem>_trees.gpkg` and
  refuses the merge otherwise, because the tree-id offsets are applied positionally.
- `merge` also takes `--runtime-s <seconds>`: without it the merged report's "Inference
  runtime" is `n/a`. The value is that km tile's inference wall time, from the run log
  (`work_dirs/logs/berlin-$T-*.log`, the time between the `inference` step's command
  line and the `results_to_las` one); equivalently, a sub-tile's
  `<stem>_report.json` `runtime_s` times the number of sub-tiles, since a batch charges
  each tile its share of the one shared inference step.
- GPU 5 only (`--gpu 5`): GPUs 2-4 may be running Phase 2 benchmark jobs.
- Expect about 55-60 s per sub-tile, so roughly 1 h 35 min per km tile including the
  shared preprocess and the host-side georeferencing/reporting (measured 2026-09-22 on
  carrot's H100: 92-99 min per km tile, 4 h 47 min for all three). Run the three tiles
  **sequentially**, as one `nohup`'d script under `work_dirs/logs/`, and poll the log
  every few minutes rather than in a tight loop.
- A batch has no resume. If the run dies part-way, the sub-tiles whose `<stem>.ply`
  already exists can be finished from the host venv with `ff3d_geo georef` / `report`
  (section 3), or the whole `run` can simply be repeated - it re-exports everything.
- On a host where a 100-sub-tile batch is too much in one go, split the list into chunks
  of ~25 and issue several `run` calls into the **same** `--out`. That is safe: each call
  rewrites `<out>/scan_list.txt` and `<out>/empty_list.txt` for its own stems only, and
  `results_to_las` / `trees_to_gpkg` / `report` only ever touch the stems of that call.
  The stems must differ between chunks (they do - one per sub-tile).
- A km-tile run leaves residue that nothing cleans up: the ~100 input PLYs it wrote into
  `data/ForAINetV2/test_data/` and their `_vert.npy` / `_offsets.npy` exports in
  `data/ForAINetV2/forainetv2_instance_data/`, on the order of 1-2 GB per km tile. Once
  `merge` has run and the merged files are copied back, clear them per tile with

  ```bash
  P=${T%_1_be}; rm data/ForAINetV2/test_data/${P}_E*_100m.ply data/ForAINetV2/forainetv2_instance_data/${P}_E*_100m_*.npy
  ```

  (`$P` is `$T` without the `_1_be` suffix, i.e. the sub-tile prefix `split` used - check one
  file name from `split-$T.txt` first). The tracked benchmark plots do not match that
  glob. The split *inputs* live in `inputs/berlin/sub/<T>/` and are separate - keep them
  if the tile might be re-run, delete that directory otherwise; `work_dirs/berlin-$T/`
  additionally holds the per-sub-tile result PLY/LAS/GeoPackages, which the rsync above
  deliberately does not copy back.

Copy the results back to the Mac (no PLYs, and no per-sub-tile LAS/GeoPackage - the
merged km files are what matters):

```bash
for T in 3dm_33_380_5828_1_be 3dm_33_381_5828_1_be 3dm_33_381_5829_1_be; do
  mkdir -p ~/work/hnee/ForestFormer3D_runs/berlin_out/berlin-2021/$T
  rsync -av --exclude '*.ply' --exclude '*_100m.las' --exclude '*_100m_trees.gpkg' \
    carrot:/raid/cwinkelmann/ForestFormer3D/work_dirs/berlin-$T/ \
    ~/work/hnee/ForestFormer3D_runs/berlin_out/berlin-2021/$T/
done
```

One destination directory **per km tile**: every `<out>` holds a `scan_list.txt`, an
`empty_list.txt`, a config copy and one `2026...` mmengine log dir per `run` call, and
copying all three tiles into one directory would overwrite the first three and pile up
the log dirs. The merged `<T>.las` / `<T>_trees.gpkg` / `<T>_report.json/.md` are the
files that matter and they are uniquely named either way.

Then fill in `docs/benchmarks/2026-09-22-tegel-berlin-2021.md` from the three
`<T>_report.md` blocks.

## 9. Raster masks and crown polygons (optional, no GPU)

`ff3d_geo masks` turns a result LAS into GIS-ready rasters: an int32 instance mask
(the `treeID` of the highest point per cell, `-1` = no tree), a uint8 semantic mask
(majority class per cell, `255` = no voted point) and a crown-polygon GeoPackage
(one convex hull per tree). Run it on the merged `<T>.las` — on the Mac after
section 5's rsync, or on carrot right after `merge`:

```bash
for T in 3dm_33_380_5828_1_be 3dm_33_381_5828_1_be 3dm_33_381_5829_1_be; do
  .venv-cpu/bin/python -m ff3d_geo masks \
      --las ~/work/hnee/ForestFormer3D_runs/berlin_out/berlin-2021/$T/$T.las \
      --out ~/work/hnee/ForestFormer3D_runs/berlin_out/berlin-2021/$T
done
```

Notes:

- Writes `<T>_instance_50cm.tif`, `<T>_semantic_50cm.tif` and `<T>_crowns.gpkg`
  next to the LAS. `--cell` changes the cell size (`0.5` -> `50cm` in the name,
  `1.0` -> `1m`), `--prefix` the file-name stem.
- Measured 2026-09-23 on the Mac (M-series, `.venv-cpu`): 11-13 s and 1.9-2.0 GB
  peak RSS per 23-25 M point km tile, giving 2000 x 2000 cells at 0.5 m; the two
  GeoTIFFs are ~3 MB together (LZW, tiled 256 x 256) and the GeoPackage 12-14 MB.
- The crown count equals the `<T>_trees.gpkg` row count and the ids are the same
  ids (37386 / 37058 / 31385 for the three tiles), so crowns join to the tree
  table on `tree_id`.
- Not every tree reaches the instance raster: about 4 % (1200-1600 per tile) are
  understorey trees whose every cell is topped by a taller neighbour. The raster's
  id set is therefore a strict subset of the crowns' — use the crowns (or the tree
  table) when completeness matters.
- Load all three into QGIS as in section 6: the masks are EPSG:25833, north-up and
  snapped to a global 0.5 m lattice, so neighbouring tiles line up cell for cell.
