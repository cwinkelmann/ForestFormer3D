---
name: ff3d-inference-km-tiles
description: Use when running ForestFormer3D inference on new ALS tiles on carrot - copying tiles in, the per-GPU split/run queue with a halo, the mosaic-wide stitch/border-check/masks/buildings step, watching logs, recovery, residue cleanup, copying results back, timings and the per-tile decision rule.
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
its tiles sequentially: `split` → `run`. Splitting cuts each sub-tile with a **20 m halo**
(`--buffer 20`) and fills that halo from whichever of the eight surrounding km tiles already
exist under `inputs/berlin/` (`--neighbours`), so a sub-tile on the source tile's border is
not starved of context at the km-tile line. Merging the sub-tile results into one seamless
km tile is now a **separate, mosaic-wide** step (`benchmark/berlin_stitch.sh`, below), run
once every GPU's queue has finished — `merge`/`masks` are no longer part of this per-GPU
script.

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

**Never run the same km tile — or any two sub-tile sets that share stems — on two GPUs at
once**: `ff3d_geo/cli.py` deletes the stale `<stem>_*.npy` exports in the shared
`data/ForAINetV2/forainetv2_instance_data/` before preprocessing, so the second `run`
silently replaces the first one's point clouds mid-inference and both results are garbage.
Re-running one tile for comparison needs its own prefix (`ff3d_geo split --prefix ...`).

What the script does per tile `$T`:

```bash
IFS=_ read -r P1 P2 E N P5 P6 <<< "$T"     # 3dm 33 <easting_km> <northing_km> 1 be
NB=()                                       # existing neighbours among the eight offsets
for dE in -1 0 1; do for dN in -1 0 1; do
  [ "$dE" -eq 0 ] && [ "$dN" -eq 0 ] && continue
  CAND="inputs/berlin/${P1}_${P2}_$((E + dE))_$((N + dN))_${P5}_${P6}.las"
  [ -f "$CAND" ] && NB+=("$CAND")           # a missing neighbour is skipped
done; done
python -m ff3d_geo split --las inputs/berlin/$T.las --out inputs/berlin/sub/$T --buffer 20 \
       ${NB[@]+--neighbours "${NB[@]}"} > work_dirs/logs/split-$T.txt
python -m ff3d_geo run --las inputs/berlin/sub/$T/*.las \
       --checkpoint work_dirs/clean_forestformer/epoch_3000_fix.pth \
       --out work_dirs/berlin-$T --gpu $GPU
```

The `run` step's wall time (seconds) is appended to `work_dirs/logs/runtime-$T.txt`, one
integer, which `benchmark/berlin_stitch.sh` sums across the whole mosaic for
`stitch --runtime-s`. It activates `/raid/cwinkelmann/ff3d-geo-venv` itself. `set -uo
pipefail` (no `-e`): a failing stage prints `!!! $T <stage> failed` and the loop moves to
the next tile.

### Then stitch the whole mosaic

Once **every** GPU's queue has finished `split` + `run` for **every** tile of the mosaic
(however many GPUs and however many nights that took), stitch them all together in ONE call
so tree ids stay unique and dense across the whole mosaic, sub-tile borders and km-tile
borders alike — this replaces the old per-tile `merge`, which offset each sub-tile's ids
into a disjoint range and left trees on a border permanently split:

```bash
cd /raid/cwinkelmann/ForestFormer3D
bash benchmark/berlin_stitch.sh 3dm_33_381_5829_1_be 3dm_33_381_5830_1_be 3dm_33_382_5828_1_be \
  > work_dirs/logs/berlin-stitch-$(date +%Y%m%d-%H%M%S).log 2>&1
# also mask the Berlin ALKIS building footprints (see `ff3d-outputs-and-viewers` section 4):
FF3D_BUILDINGS=/path/to/alkis_buildings.gpkg bash benchmark/berlin_stitch.sh ...
```

Give it **every** tile that was split/run for this mosaic (not a subset — the manifest and
results dir of each one feeds the same `stitch` call, so a tile left out simply never gets
matched against its neighbours). It runs `python -m ff3d_geo stitch --manifest <every split's
split_manifest.json> --results <every split's run --out dir> --out work_dirs/berlin-mosaic
--runtime-s <sum of the tiles' runtime-$T.txt>` once, then per tile: `border-check` (writes
`work_dirs/berlin-mosaic/<T>_border.json`), `masks`, and — only when `FF3D_BUILDINGS` is
set — `buildings`.

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
=== ... tile done (run 5932s) ===
=== ... block finished ===
```

`tile done (run <N>s)` is the whole `run` step's wall time for that km tile, also appended
to `work_dirs/logs/runtime-$T.txt` for `berlin_stitch.sh --runtime-s`. Once all the GPU
queues are done, `berlin_stitch.sh`'s own log has the same `=== <timestamp> <tile>: <stage>
===` / `!!!` milestones, one `mosaic: stitch` pair around the single `stitch` call and then
`border-check` / `masks` / `buildings` / `tile done` per tile:

```
=== 2026-09-24T09:12:03 mosaic: stitch (3 tiles) ===
=== 2026-09-24T09:41:57 mosaic: stitch done ===
=== 2026-09-24T09:41:57 3dm_33_374_5827_1_be: border-check ===
=== 2026-09-24T09:42:05 3dm_33_374_5827_1_be: masks ===
=== 2026-09-24T09:42:19 3dm_33_374_5827_1_be: tile done ===
...
=== ... block finished ===
```

## 4. Timings

**Not yet re-measured with the halo.** All the numbers below predate `--buffer 20`
(commit `b1715a3`): a haloed sub-tile is its `-20..120` box on both axes, **≈1.96x the
core's point count**, all of which `run` feeds to the model — expect roughly **double**
the per-sub-tile run time (and the residue footprint, section 7) until the haloed runs are
actually timed.

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
  `split`'s output (`split_manifest.json`, the sub-tile LAS/ident files) does not need to be
  redone — only re-split a tile if `split` itself failed or its `--out` was removed.

If a host is too small for a ~100-sub-tile batch, issue several `run` calls with ~25
sub-tiles each into the **same** `--out`: each call rewrites only its own
`scan_list.txt` / `empty_list.txt` and only touches its own stems.

`benchmark/berlin_stitch.sh` has no resume either — it is one call over the WHOLE mosaic.
**It needs every sub-tile result of every manifest passed to it**: a sub-tile whose `run`
never finished (so its `<results dir>/<stem>.las` is missing) makes `ff3d_geo stitch` raise
`FileNotFoundError`, **naming the missing stem**, before it writes anything for ANY tile of
the mosaic. Finish or rerun that one tile's `run` (per the point above — the rest of the
mosaic's sub-tile results are untouched) and re-run the whole `berlin_stitch.sh` call with
the same tile list. Once the stitch itself has succeeded, `border-check`/`masks`/`buildings`
for an individual tile can be rerun by hand (the same commands `berlin_stitch.sh` runs,
against `work_dirs/berlin-mosaic/$T.las`) without repeating `stitch`.

## 6. Buildings: mask the ALKIS footprints out (Berlin tiles, after `stitch`)

The Berlin ALS 2021 tiles have **no building class** (classes present: 2, 3, 4, 5, 7, 32;
class 6 absent, roof points in 3/4/5), so the model predicts tree instances on roofs.
Run the mask on every Berlin km tile after `stitch`/`masks` and before the results are
published or compared — it is not part of `run`. `benchmark/berlin_stitch.sh` does this
automatically, per tile, right after its own `masks` step, when `FF3D_BUILDINGS` is set to
the footprint GeoPackage's path (see section 2); the manual command below is for a one-off
or a tile stitched without it.

Fetch the footprints once (official ALKIS WFS `https://gdi.berlin.de/services/wfs/
alkis_gebaeude`, feature type `alkis_gebaeude:gebaeude`; `--tiles` takes km-tile keys or
the groups `tegel`/`r13`/`all`, a tile already in the GeoPackage is skipped):

```bash
# on the Mac, .venv-cpu
python benchmark/fetch_berlin_buildings.py --tiles all \
  --out /Volumes/2TB/winmol/ALS_Data/berlin_buildings/alkis_buildings.gpkg
```

Then, per finished km tile (CPU, host venv, ~4–6 min and < 3 GB RSS per 20–25 M point
tile):

```bash
T=3dm_33_381_5829_1_be
D=/Volumes/2TB/winmol/ALS_Data/berlin_als_2021_ff3d/$T
python -m ff3d_geo buildings --las $D/$T.las \
  --buildings /Volumes/2TB/winmol/ALS_Data/berlin_buildings/alkis_buildings.gpkg \
  --out $D/masked
```

`masked/` gets the full product set (`<T>.las`, `_trees.gpkg`, `_crowns.gpkg`, both
GeoTIFFs, `_report.json/.md`); the unmasked originals stay in place, and instance ids are
**not** renumbered, so the two are comparable id for id. Details and the per-tile numbers:
`ff3d-outputs-and-viewers` section 4 and
`docs/benchmarks/2026-09-22-tegel-berlin-2021.md` section "Buildings".

## 7. Residue cleanup (1–2 GB per km tile pre-halo; expect roughly double until measured)

A km-tile run leaves the ~100 input PLYs and their `.npy` exports behind. After
`benchmark/berlin_stitch.sh` has stitched the mosaic and the results are copied back:

```bash
T=3dm_33_381_5830_1_be
P=${T%_1_be}     # sub-tile prefix used by split, e.g. 3dm_33_381_5830
head -1 work_dirs/logs/split-$T.txt    # sanity-check the prefix against a real file name
rm data/ForAINetV2/test_data/${P}_E*_100m.ply \
   data/ForAINetV2/forainetv2_instance_data/${P}_E*_100m_*.npy
```

That glob cannot match the tracked benchmark plots. The split **inputs**
`inputs/berlin/sub/$T/` are a different matter: they are not disposable until the
mosaic-wide stitch has actually run. `split_manifest.json` and, per sub-tile,
`<stem>_ident.npy` in that directory are what `ff3d_geo stitch` reads to recognise a halo
point of one sub-tile as the same physical point as a core point of its neighbour
(`ff3d_geo/stitch.py`) — **keep `inputs/berlin/sub/$T/` until `benchmark/berlin_stitch.sh`
has successfully stitched this tile**, not just until the tile's own `run` finished.
Deleting it early is not cheaply recoverable: re-running `split` regenerates the manifest
and sidecars, but only reproduces the same halo if the exact same `--neighbours` set is
given again — if a neighbour tile arrived on disk in the meantime, the halo point counts
change and `stitch`'s sub-tile cache rejects the mismatched results, forcing a full re-`run`
of that tile (1.5–3 h of GPU time) before it can be stitched at all.

## 8. Copy the results home

The stitched, seamless-id products live in `work_dirs/berlin-mosaic/` (one directory for the
whole mosaic, not one per tile) — copy it back as a whole, or a subset with `--include`:

```bash
# on the Mac
mkdir -p ~/work/hnee/ForestFormer3D_runs/berlin_out/berlin-mosaic
rsync -av carrot:/raid/cwinkelmann/ForestFormer3D/work_dirs/berlin-mosaic/ \
  ~/work/hnee/ForestFormer3D_runs/berlin_out/berlin-mosaic/
```

The per-sub-tile `run` output (`work_dirs/berlin-$T/`) is normally left on carrot — it feeds
`stitch` and residue cleanup (section 7), not publishing:

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

## 9. Single 100 m tiles (the r12/r13 path)

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

## 10. The batch `run` contract

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

## 11. Per-tile decision rule

Each report ends with `First pass usable: **yes|no**` from `ff3d_geo.report.recommend`:
usable when the tree count is within **±50 %** of the CHM local-maxima baseline **and**
the median tree height is within **3 m** of the CHM median. On the Berlin 2021 tiles the
count passes (6–11 % under baseline) while the median height fails by 4.4–8.6 m — a shift
of the whole height distribution, not a tail of short instances. Paste the `<T>_report.md`
blocks into a dated `docs/benchmarks/<date>-<area>.md` and commit that.
