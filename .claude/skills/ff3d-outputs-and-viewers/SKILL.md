---
name: ff3d-outputs-and-viewers
description: Use when working with ForestFormer3D outputs - the result PLY/LAS contract, tree tables, crowns, GeoTIFF masks and report JSON, the ff3d_geo convert/georef/report/masks recovery commands, the figure scripts, DOP overlays, QGIS and the Potree web viewer.
---

# Outputs, conversions and viewers

Everything below runs on the **host**, no GPU, no torch: the Mac's `.venv-cpu` or carrot's
`/raid/cwinkelmann/ff3d-geo-venv`.

## 1. What a run produces

Per 100 m sub-tile (or per single tile), in the run's `--out` directory:

| file | what it is |
|---|---|
| `<stem>.ply` | the **raw model output**: `x y z` (float32, centred by `<stem>_offsets.npy` = `[mean_x, mean_y, min_z]`), `semantic_pred` (0 ground, 1 wood, 2 leaf, −1 no vote), `instance_pred` (tree id, −1 none), `score`. Binary little-endian since commit `efb3fcf`. |
| `<stem>.sidecar.json` | origin (tile lower-left in EPSG:25833), EPSG, point count, source format |
| `<stem>_classification.npy` | the input ALS classification per point, kept for the CHM baseline |
| `<stem>.las` | **LAS 1.4, point format 6, EPSG:25833 WKT**, extra dims `treeID` int32 (−1 none), `semantic` uint8 (**255** = no vote; **3** = building after `ff3d_geo buildings`), `score` float32; the ALS classification restored |
| `<stem>_trees.gpkg` | layer `trees`, one point per tree |
| `<stem>_report.json` / `.md` | the per-tile report |
| `scan_list.txt`, `empty_list.txt`, a config copy, `2026…/` | run bookkeeping (mmengine log dir) |

Per km tile, after `merge` and `masks`: `<T>.las`, `<T>_trees.gpkg`, `<T>_report.json/.md`,
`<T>_instance_50cm.tif`, `<T>_semantic_50cm.tif`, `<T>_crowns.gpkg`.

**Tree table columns** (`<stem>_trees.gpkg`, layer `trees`, EPSG:25833):
`tree_id, x, y, top_z, height, crown_area_m2, n_points, mean_score` — one point geometry
at the tree top; `height` is measured against a 1 m ground grid.

**Crowns** (`<T>_crowns.gpkg`): one convex hull per tree, joined to the tree table on
`tree_id`. The crown count equals the tree-table row count.

**Masks** (0.5 m cells, EPSG:25833, north-up, snapped to a global 0.5 m lattice so
neighbouring tiles line up cell for cell):
`<T>_instance_50cm.tif` int32 = the `treeID` of the **highest** point per cell, nodata −1;
`<T>_semantic_50cm.tif` uint8 = majority class per cell, 255 = no voted point.
About 4 % of trees (1200–1600 per km tile) are understorey and never reach the instance
raster — its id set is a strict **subset** of the crowns'. Use the crowns or the tree table
when completeness matters.

**Report JSON keys**: `tile, n_points, n_trees, chm_baseline_count, height_stats,
chm_height_stats, ground_vs_vegetation_agreement, nodata_fraction, n_voted, confusion,
per_class_counts, runtime_s, recommendation`, plus `buildings` once the
ALKIS mask has run (section 4). `recommendation.first_pass_usable` is true
when the tree count is within ±50 % of the CHM local-maxima baseline **and** the median
height within 3 m of the CHM median.

## 2. Verify a result

```bash
cd ~/ForestFormer3D
.venv-cpu/bin/python -c "
import laspy; l = laspy.read('$HOME/work/hnee/ForestFormer3D_runs/berlin_out/tegel-r12/r12_tegel_E381300_N5828300_100m.las')
print(l.header.version, l.header.point_format.id, l.header.parse_crs().to_epsg(),
      l.header.point_count, l.header.mins, l.header.maxs, list(l.point_format.extra_dimension_names))"
# 1.4 6 25833 192814 [381300. 5828300. 0.] [381399.99 5828399.99 36.83] ['treeID', 'semantic', 'score']

.venv-cpu/bin/python -c "
import geopandas as g; d = g.read_file('.../r12_..._trees.gpkg', layer='trees')
print(len(d), d.crs.to_epsg(), d.height.describe()[['min','50%','max']].round(1).tolist())"
```

The row count and height quantiles must match `n_trees` / `height_stats` in the report.

## 3. `ff3d_geo` conversions (recovery and one-offs)

```bash
source /raid/cwinkelmann/ff3d-geo-venv/bin/activate     # or use .venv-cpu on the Mac
python -m ff3d_geo <cmd> --help                         # run|split|merge|masks|buildings|convert|georef|report
```

- **`convert`** — LAS → input PLY + sidecar, without running anything else:
  `--las <in.las> --ply <out.ply> --sidecar <out.sidecar.json> [--origin E N] [--epsg 25833]`.
  The origin is parsed from an `E<easting>_N<northing>` file name unless given.
- **`georef`** — result PLY → LAS 1.4 (use when a run died *after* inference; never re-infer):
  ```bash
  python -m ff3d_geo georef \
    --result-ply work_dirs/tegel-r12/<stem>.ply \
    --sidecar    work_dirs/tegel-r12/<stem>.sidecar.json \
    --offsets    data/ForAINetV2/forainetv2_instance_data/<stem>_offsets.npy \
    --out-las    work_dirs/tegel-r12/<stem>.las \
    --gpkg       work_dirs/tegel-r12/<stem>_trees.gpkg
  ```
- **`report`** — result LAS (+ tree gpkg, written if absent) → report JSON/MD:
  `--las ... --gpkg ... --json ... [--md ...] [--runtime-s <seconds of the tools/test.py step>]`.
  Without `--runtime-s` the report says `n/a`.
- **`masks`** — result LAS → the two GeoTIFFs + crown polygons:
  `--las <T>.las --out <dir> [--cell 0.5] [--prefix <stem>]`. `--cell 0.5` → `50cm` in the
  file names, `1.0` → `1m`. Measured on the Mac: 11–13 s and 1.9–2.0 GB peak RSS per
  23–25 M point km tile; the two GeoTIFFs ~3 MB together (LZW, tiled 256×256) and the
  GeoPackage 12–14 MB.
- **`buildings`** — the Berlin ALKIS building mask; see section 4.
- `split` / `merge` / `run` belong to the inference flow — see `ff3d-inference-km-tiles`.

## 4. Buildings (Berlin): masking ALKIS footprints out of the trees

The Berlin ALS 2021 tiles have **no building class**. Measured on three km tiles the
classes present are 2 (ground), 3, 4, 5, 7, 32 — class 6 (building) is absent and roof
points sit in 3/4/5. The model was trained on forest plots, so it reads a roof as a
crown and predicts tree instances on buildings (obvious in the allotment area of
`3dm_33_381_5829_1_be`). The fix is external geometry: Berlin's official ALKIS
footprints.

**Fetch the footprints once** (official WFS, not OSM):

```bash
.venv-cpu/bin/python benchmark/fetch_berlin_buildings.py --tiles all \
  --out /Volumes/2TB/winmol/ALS_Data/berlin_buildings/alkis_buildings.gpkg
```

`https://gdi.berlin.de/services/wfs/alkis_gebaeude`, feature type
`alkis_gebaeude:gebaeude`, GeoJSON, EPSG:25833, `CountDefault` 1 000 000 (the script
pages anyway). `--tiles` takes km-tile keys (`381_5829`) or the groups `tegel` (the
eleven Tegel tiles), `r13` (the eight Revier 13 tiles) and `all`; `--bbox MINX MINY
MAXX MAXY` takes a free bbox. One GeoPackage per run holds layer `buildings`
(deduplicated by `uuid`, with `gfk`/`bezgfk`, `bat`/`bezbat`, `baw`/`bezbaw`, `nam`,
`shape_area`) and layer `tiles` (coverage squares) — a tile already in `tiles` is
skipped, so a re-run is cheap; `--force` refetches.

**Mask a finished km tile** (run it *after* `merge`, before publishing anything):

```bash
.venv-cpu/bin/python -m ff3d_geo buildings \
  --las      /Volumes/2TB/.../berlin_als_2021_ff3d/<T>/<T>.las \
  --buildings /Volumes/2TB/winmol/ALS_Data/berlin_buildings/alkis_buildings.gpkg \
  --out      /Volumes/2TB/.../berlin_als_2021_ff3d/<T>/masked
```

Writes `<T>.las`, `<T>_trees.gpkg`, `<T>_crowns.gpkg`, both GeoTIFFs and
`<T>_report.json/.md` into `--out`; the originals are never touched, so masked and
unmasked are directly comparable. Options: `--buffer 1.0` (footprints are buffered,
because a roof edge overhangs its ground plan), `--min-roof-fraction 0.5`,
`--cell 0.5`, `--no-masks`.

What it does (`ff3d_geo.buildings.mask_buildings`): a point inside a buffered footprint
gets `semantic = 3` (**building** — a new value in the result contract, next to 0
ground / 1 wood / 2 leaf / 255 no vote) and `treeID = -1`, while its ALS
`classification` is left untouched; an instance with ≥ `--min-roof-fraction` of its
points inside footprints is dropped entirely. **Ids are not renumbered** — a surviving
tree keeps the id it had in the unmasked result. The returned dict (and the report's
`buildings` block, and the report markdown's "Buildings" row) carries
`n_footprints, points_masked, instances_before, instances_removed,
instances_partially_masked, instances_after`.

Because `semantic == 3` is not a model vote, the report excludes those points from the
ground-vs-non-ground agreement, the same way it excludes 255.

Numbers for the Berlin tiles: `docs/benchmarks/2026-09-22-tegel-berlin-2021.md`,
section "Buildings".

## 5. Visual report figures

```bash
cd ~/ForestFormer3D
.venv-cpu/bin/python benchmark/plot_berlin_visuals.py \
  --tile 3dm_33_381_5829_1_be \
  --data-dir ~/work/hnee/ForestFormer3D_runs/berlin_out/berlin-2021 \
  --out-dir docs/benchmarks/assets/berlin-visual
.venv-cpu/bin/python benchmark/plot_berlin_visuals.py --only 5,6   # a subset
```

Eight figures: 1 top-down point cloud of a km tile, 2 oblique 3D view of the densest 120 m
window, 3 side elevation of a 100 m × 20 m strip, 4 close-up with crown polygons, 5 crown
map over the CHM hillshade, 6 zoom on the Revier 12 Tegelsee AOI, 7 height and crown-area
distributions, 8 multi-tile overview. It reads `<tile>.las`, `<tile>_trees.gpkg`,
`<tile>_crowns.gpkg`, `<tile>_instance_50cm.tif`, `<tile>_report.json`; `--data-dir` may be
repeated (first hit wins for single-tile figures, all of them for the overview);
`--sub-points` caps points per figure, `--cache-dir` / `--no-cache` control the decimation
cache. Write-up: `docs/benchmarks/2026-09-23-berlin-visual-report.md`.

DOP overlays (`benchmark/plot_berlin_dop_overlays.py`) are covered in
`ff3d-orthophoto-download`.

## 6. QGIS (3.28+)

1. New project, Project ▸ Properties ▸ CRS `EPSG:25833`.
2. Drag `<T>.las` in — it opens as a point cloud layer, CRS read from the WKT VLR.
   Symbology "Attribute by Ramp" on `treeID` (or `semantic`).
3. Drag `<T>_trees.gpkg` (layer `trees`): single symbol, size by `height` (data-defined
   `"height" / 2`), label `tree_id`. Add `<T>_crowns.gpkg` and the two GeoTIFFs the same way.
4. Optional: an XYZ OpenStreetMap basemap; View ▸ 3D Map Views for a 3D check.
5. Export Map to Image, 1600 px wide, into `docs/benchmarks/assets/`.

## 7. Potree web viewer

An offline Potree 3D site over the Berlin tiles lives on the 2TB volume:

```bash
cd /Volumes/2TB/winmol/ALS_Data/berlin_potree
python3 -m http.server 8080        # then open http://localhost:8080/
```

A `file://` open does **not** work — the octree loader uses `fetch` with range requests and
web workers, which need HTTP.

Layout: `index.html` (the viewer), `build/potree/` (Potree 1.8.2 release build),
`libs/`, `pointclouds/<tile>/` (Potree 2 octrees: `metadata.json`, `hierarchy.bin`,
`octree.bin`), `data/<tile>_trees.geojson` / `_crowns.geojson` / `_chm.png|.json` /
`_instance.png|.json` / `_dop2021.png|.json` / `_dop2025.png|.json`, and
`data/tiles.json` (the manifest `index.html` reads). Point-colour modes: `tree id`,
`semantic`, `height`, `score`, `ALS class`; draped layers: CHM, instance mask, DOP 2021
(leaf-off), DOP 2025 (leaf-on).

The octrees themselves are built separately with **PotreeConverter 2.1.1**; the overlays
and the manifest are built by `benchmark/build_potree_site.py`:

```bash
.venv-cpu/bin/python benchmark/build_potree_site.py \
    --site /Volumes/2TB/winmol/ALS_Data/berlin_potree \
    --ff3d-dir /Volumes/2TB/winmol/ALS_Data/berlin_als_2021_ff3d \
    --dop2021-dir /Volumes/2TB/winmol/ALS_Data/berlin_dop_2021/dop_2021_rgb \
    --dop2025-dir /Volumes/2TB/winmol/ALS_Data/berlin_dop_2025_sommer
```

(`--tiles` for a subset, `--texture-px`, `--chm-cell`, `--dtm-cell`, `--jobs`,
`--skip-existing`.) That script and the site's own `README.md`
(`/Volumes/2TB/winmol/ALS_Data/berlin_potree/README.md`, which documents controls and
"Adding a tile") were written alongside these skills — **see `benchmark/build_potree_site.py`
once landed** if it is not yet tracked in the repo you have checked out.

## 8. Source-of-truth documents

`docs/inference-pipeline.md` (the whole path, file by file),
`docs/benchmarks/RUNBOOK-tegel.md` (commands, sections 4–9),
`docs/benchmarks/2026-09-23-berlin-visual-report.md`,
`docs/benchmarks/2026-09-23-berlin-dop-overlay.md`.
