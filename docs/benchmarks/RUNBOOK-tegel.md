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
