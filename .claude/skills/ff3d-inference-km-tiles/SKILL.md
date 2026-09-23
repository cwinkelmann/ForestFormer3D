---
name: ff3d-inference-km-tiles
description: Use when running ForestFormer3D inference on new ALS tiles on carrot - copying tiles in, the per-GPU split/run/merge/masks queue, watching logs, recovery, residue cleanup, copying results back, timings and the per-tile decision rule.
---

# Running inference on new km tiles

Read `ff3d-carrot` first for ssh, the image, `benchmark/common.sh` and GPU etiquette.
The full pipeline description is `docs/inference-pipeline.md`; the long-form runbook is
`docs/benchmarks/RUNBOOK-tegel.md`.

The model is trained on ~14–22 m plots, so a 1 km ALS tile is cut into **100 m sub-tiles**,
inferred as one batch, and stitched back together.

## 0. Preflight

```bash
ssh carrot
cd /raid/cwinkelmann/ForestFormer3D
git pull --ff-only origin fix/review-findings && git log --oneline -1
docker image inspect forestformer3d:cu118 --format '{{.Id}}'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv   # pick idle cards; GPU 1 is someone else's
ls work_dirs/clean_forestformer/epoch_3000_fix.pth
```

## 1. Copy the tiles in

From the Mac (tiles live on the 2TB volume, see `ff3d-als-download`):

```bash
ssh carrot mkdir -p /raid/cwinkelmann/ForestFormer3D/inputs/berlin
scp /Volumes/2TB/winmol/ALS_Data/berlin_als_2021/3dm_33_381_5830_1_be.las \
    /Volumes/2TB/winmol/ALS_Data/berlin_als_2021/3dm_33_382_5828_1_be.las \
    carrot:/raid/cwinkelmann/ForestFormer3D/inputs/berlin/
```

## 2. Launch one queue per GPU

`benchmark/berlin_run_gpu.sh` is the tracked production script (the live copy on carrot is
`work_dirs/logs/berlin-run-gpu.sh`, identical). One invocation owns **one GPU** and walks
its tiles sequentially: `split` → `run` → `merge` → `masks`.

```bash
cd /raid/cwinkelmann/ForestFormer3D
mkdir -p work_dirs/logs
TS=$(date +%Y%m%d-%H%M%S)
nohup bash benchmark/berlin_run_gpu.sh 5 3dm_33_381_5830_1_be \
  > work_dirs/logs/berlin-gpu5-$TS.log 2>&1 &
nohup bash benchmark/berlin_run_gpu.sh 6 3dm_33_382_5828_1_be 3dm_33_382_5829_1_be \
  > work_dirs/logs/berlin-gpu6-$TS.log 2>&1 &
```

Rules: one tile at a time per GPU; give each GPU its own tile list (never the same tile to
two GPUs — they share `data/ForAINetV2/test_data/` and would overwrite each other's
exports only if stems collided, and the sub-tile stems are derived from the km tile name,
so different km tiles are safe); skip GPU 1 always and any card already busy.

What the script does per tile `$T`:

```bash
python -m ff3d_geo split --las inputs/berlin/$T.las --out inputs/berlin/sub/$T \
       > work_dirs/logs/split-$T.txt
python -m ff3d_geo run --las inputs/berlin/sub/$T/*.las \
       --checkpoint work_dirs/clean_forestformer/epoch_3000_fix.pth \
       --out work_dirs/berlin-$T --gpu $GPU
python -m ff3d_geo merge --las work_dirs/berlin-$T/*_100m.las \
       --gpkg work_dirs/berlin-$T/*_100m_trees.gpkg \
       --out-las work_dirs/berlin-$T/$T.las --out-gpkg work_dirs/berlin-$T/${T}_trees.gpkg \
       --report-json work_dirs/berlin-$T/${T}_report.json \
       --report-md work_dirs/berlin-$T/${T}_report.md --runtime-s $R
python -m ff3d_geo masks --las work_dirs/berlin-$T/$T.las --out work_dirs/berlin-$T
```

It activates `/raid/cwinkelmann/ff3d-geo-venv` itself. `set -uo pipefail` (no `-e`): a
failing stage prints `!!! $T <stage> failed` and the loop moves to the next tile.

## 3. Watch it

```bash
tail -5 work_dirs/logs/berlin-gpu5-*.log        # poll every few minutes, NOT in a tight loop
grep -h 'tile done\|!!!' work_dirs/logs/berlin-gpu*.log
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
pgrep -af berlin_run_gpu ; docker ps
```

Milestones in a log, in order:

```
=== 2026-09-23T15:04:11 3dm_33_374_5827_1_be: split ===
=== 2026-09-23T15:04:15 3dm_33_374_5827_1_be: run (98 sub-tiles) ===
# python: las_to_ply ... -> data/ForAINetV2/test_data/<stem>.ply (+ sidecar ...)
$ cd /raid/... && docker run ... batch_load_ForAINetV2_data.py --unlabeled ...
$ ... tools/test.py ...            # one process, N scans; "The length of the dataset: 98"
# python: results_to_las / trees_to_gpkg / report
=== ... merge ===
=== ... tile done (run 5932s) ===
=== ... block finished ===
```

`tile done (run <N>s)` is the whole `run` step's wall time for that km tile — that number
is what `merge --runtime-s` records in the report.

## 4. Timings

- **2026-09-22, one GPU per tile, pre-vectorisation**: 55–60 s per 100 m sub-tile,
  **92–99 min per km tile** (4 h 47 min for three tiles).
- Running 7 tiles on 7 GPUs at once is host-latency bound, not GPU bound: those blocks
  measured up to 235 s per sub-tile (6.5 h per km tile). Parallelism across GPUs buys much
  less than it looks like it should.
- The vectorised z-filter + binary PLY writer (commits `3da7f87`, `efb3fcf`) measured
  **6.7x** end to end (316.7 → 47.2 s per scan on ten sub-tiles under contention), so an
  uncontended GPU should now be well under 20 s per sub-tile.
- **Measure, don't assume**: take the first `tile done (run Ns)` of the current run and
  divide by the sub-tile count from `wc -l work_dirs/logs/split-$T.txt`.

Profile of where the time goes: `docs/benchmarks/2026-09-23-inference-profile.md`.

## 5. Recovery

A batch `run` has **no resume**. If it dies:

- Sub-tiles whose `<out>/<stem>.ply` exists are already inferred — finish them from the
  host venv instead of re-inferring (see `ff3d_geo georef` / `report` in
  `ff3d-outputs-and-viewers`), or just re-run the whole `run`, which re-exports everything.
- If `run` finished but `merge`/`masks` failed, rerun just those by hand:

```bash
cd /raid/cwinkelmann/ForestFormer3D && source /raid/cwinkelmann/ff3d-geo-venv/bin/activate
T=3dm_33_381_5830_1_be
ls work_dirs/berlin-$T/*_100m.las | wc -l     # must equal the sub-tile count
python -m ff3d_geo merge --las work_dirs/berlin-$T/*_100m.las \
  --gpkg work_dirs/berlin-$T/*_100m_trees.gpkg \
  --out-las work_dirs/berlin-$T/$T.las --out-gpkg work_dirs/berlin-$T/${T}_trees.gpkg \
  --report-json work_dirs/berlin-$T/${T}_report.json \
  --report-md work_dirs/berlin-$T/${T}_report.md \
  --runtime-s $(grep "$T: tile done" work_dirs/logs/berlin-gpu*.log | sed 's/.*run \([0-9]*\)s.*/\1/')
python -m ff3d_geo masks --las work_dirs/berlin-$T/$T.las --out work_dirs/berlin-$T
```

Both `merge` globs must expand in the same sorted order; `merge` checks pairwise that
`<stem>.las` goes with `<stem>_trees.gpkg` and refuses otherwise (tree-id offsets are
applied positionally). Without `--runtime-s` the merged report says `Inference runtime:
n/a` — an equivalent value is a sub-tile's `report.json` `runtime_s` times the sub-tile
count.

If a host is too small for a ~100-sub-tile batch, issue several `run` calls with ~25
sub-tiles each into the **same** `--out`: each call rewrites only its own
`scan_list.txt` / `empty_list.txt` and only touches its own stems.

## 6. Residue cleanup (1–2 GB per km tile)

A km-tile run leaves the ~100 input PLYs and their `.npy` exports behind. After `merge`
and after the results are copied back:

```bash
T=3dm_33_381_5830_1_be
P=${T%_1_be}     # sub-tile prefix used by split, e.g. 3dm_33_381_5830
head -1 work_dirs/logs/split-$T.txt    # sanity-check the prefix against a real file name
rm data/ForAINetV2/test_data/${P}_E*_100m.ply \
   data/ForAINetV2/forainetv2_instance_data/${P}_E*_100m_*.npy
```

That glob cannot match the tracked benchmark plots. The split **inputs**
`inputs/berlin/sub/$T/` are separate: keep them if the tile may be re-run, otherwise
`rm -rf` that directory.

## 7. Copy the results home

```bash
# on the Mac
T=3dm_33_381_5830_1_be
mkdir -p ~/work/hnee/ForestFormer3D_runs/berlin_out/berlin-2021/$T
rsync -av --exclude '*.ply' --exclude '*_100m.las' --exclude '*_100m_trees.gpkg' \
  carrot:/raid/cwinkelmann/ForestFormer3D/work_dirs/berlin-$T/ \
  ~/work/hnee/ForestFormer3D_runs/berlin_out/berlin-2021/$T/
```

**One destination directory per km tile** — each `<out>` has its own `scan_list.txt`,
`empty_list.txt`, config copy and mmengine log dir, which would collide. The 2TB volume
mirror of these products is `/Volumes/2TB/winmol/ALS_Data/berlin_als_2021_ff3d/<tile>/`.

## 8. Single 100 m tiles (the r12/r13 path)

For a tile that is already ~100 m, skip `split`/`merge` entirely:

```bash
cd /raid/cwinkelmann/ForestFormer3D && source /raid/cwinkelmann/ff3d-geo-venv/bin/activate
python -m ff3d_geo run --las inputs/r12_tegel_E381300_N5828300_100m.las \
  --checkpoint work_dirs/clean_forestformer/epoch_3000_fix.pth \
  --out work_dirs/tegel-r12 --gpu 5 \
  2>&1 | tee work_dirs/logs/tegel-r12-$(date +%Y%m%d-%H%M%S).log
```

Minutes, not hours. The tile origin is parsed from the file name (`E<easting>_N<northing>`)
or given with `--origin E N`. Sanity numbers for r12: 192,814 points, 397 CHM local maxima,
66,337 ALS ground points — if `chm_baseline_count` differs, the ALS classification did not
survive the round trip. Preview any run with `--dry-run` (prints the 8 steps, runs nothing).

## 9. The batch `run` contract

`python -m ff3d_geo run` is a **host-side orchestrator** (plain venv) that shells the two
GPU steps into the container via `bash -c 'source benchmark/common.sh; ff3d_docker ...'`
with `FF3D_GPU` from `--gpu`. Its eight steps, each printed before it runs:

1. `las_to_ply` — each input LAS → `data/ForAINetV2/test_data/<stem>.ply` (xyz only) plus
   `<out>/<stem>.sidecar.json` (origin, EPSG, point count, source format) and
   `<stem>_classification.npy`.
2. `prepare_inputs` — writes `<out>/scan_list.txt` and removes stale exports for those
   stems. The tracked `data/ForAINetV2/meta_data/test_list.txt` is **never** touched.
3. `preprocess` (container) — `batch_load_ForAINetV2_data.py --unlabeled` then
   `create_data_forainetv2.py --test-list <scan list> --splits test --out-dir <out>`.
4. `check_preprocess` — every stem has `_vert.npy` and `_offsets.npy`; the pkl is newer
   than the scan list.
5. `inference` (container) — one `tools/test.py` for all stems: one model load, N PLYs.
6. `results_to_las` — `<out>/<stem>.las` (LAS 1.4, pf 6, EPSG:25833 WKT).
7. `trees_to_gpkg` — `<out>/<stem>_trees.gpkg`.
8. `report` — `<out>/<stem>_report.json` / `.md`.

Host steps isolate failures per tile; a failed docker step stops the run.

## 10. Per-tile decision rule

Each report ends with `First pass usable: **yes|no**` from `ff3d_geo.report.recommend`:
usable when the tree count is within **±50 %** of the CHM local-maxima baseline **and**
the median tree height is within **3 m** of the CHM median. On the Berlin 2021 tiles the
count passes (6–11 % under baseline) while the median height fails by 4.4–8.6 m — a shift
of the whole height distribution, not a tail of short instances. Paste the `<T>_report.md`
blocks into a dated `docs/benchmarks/<date>-<area>.md` and commit that.
