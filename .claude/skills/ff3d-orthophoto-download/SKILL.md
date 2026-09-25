---
name: ff3d-orthophoto-download
description: Use when fetching Berlin digital orthophotos (DOP 2021 leaf-off, DOP 2025 spring, TrueDOP 2025 summer) as km-tile GeoTIFFs with benchmark/fetch_berlin_dop.py, discovering other years' WMS services, or building crown overlay figures on them.
---

# Berlin orthophotos (DOP) for the ALS tiles

Orthophotos are the visual reference the ForestFormer3D crowns are checked against. They
are fetched per **1 km ALS tile**, so a DOP GeoTIFF has exactly the extent of the ALS tile
of the same name: `3dm_33_<E>_<N>_1_be.tif`, EPSG:25833, 5000 × 5000 px at 20 cm.

## Services

Berlin's view services live under `https://gdi.berlin.de/services/wms/<service>`. The old
FIS-Broker endpoints (`fbinter.stadt-berlin.de/fb/wms/senstadt/...`) are **gone** — the
whole prefix returns 404; do not retry them.

| what | `--service` | `--layer` | notes |
|---|---|---|---|
| DOP 2021 (the reference) | `dop_2021` | `dop_2021_rgb`, `dop_2021_cir` | DOP20RGBI, 20 cm, **Bildflug 22.02.2021 — leaf-off** |
| Spring DOP 2025 | `dop_2025_fruehjahr` | `dop_2025_fruehjahr_rgb` | leaf-off spring flight |
| Summer TrueDOP 2025 | `truedop_2025_sommer` | `truedop_2025_sommer_rgb` | **leaf-on**, true orthophoto (no building lean) |

Other years, same pattern: `dop_2019`, `truedop_2020_sommer`, `truedop_2022`,
`truedop_2023`, `truedop_2024`, `truedop_2026`.

DOP 2021 details: WMS 1.3.0; CRS `EPSG:25833` (+ CRS:84, axis order E,N — 1.3.0 and 1.1.1
return byte-identical images for the same bbox); GetMap formats `image/geotiff`,
`image/tiff`, `image/gif`, `image/jpeg`, `image/png`; no `MaxWidth`/`MaxHeight` declared
(2500 px requests are used anyway); 0.20 m ground resolution, ±0.4 m positional accuracy,
delivered in a 2 km Blattschnitt; licence dl-de/zero-2-0; metadata record
`ef0c8276-78af-4444-969f-d01bb7d3c841`.

## Fetching

`benchmark/fetch_berlin_dop.py` requests each 1 km tile as four 2500 × 2500 px quadrants
(`--px-per-tile 5000 --chunk 2500`), assembles them and writes one tiled, LZW-compressed
uint8 GeoTIFF with the exact EPSG:25833 affine transform. It skips tiles whose output
exists (`--overwrite` to force), retries with exponential backoff (`--retries`, default 4)
and sleeps `--sleep` (default 1 s) between GetMap calls.

```bash
cd ~/ForestFormer3D

# DOP 2021, RGB, the eleven benchmark tiles (--tiles all)
.venv-cpu/bin/python benchmark/fetch_berlin_dop.py --tiles all \
    --out-dir /Volumes/2TB/winmol/ALS_Data/berlin_dop_2021/dop_2021_rgb

# DOP 2021 as 4-band RGBI
.venv-cpu/bin/python benchmark/fetch_berlin_dop.py --tiles all --rgbi \
    --out-dir /Volumes/2TB/winmol/ALS_Data/berlin_dop_2021/dop_2021_rgbi

# Summer TrueDOP 2025 (leaf-on), same tiles
.venv-cpu/bin/python benchmark/fetch_berlin_dop.py \
    --service truedop_2025_sommer --layer truedop_2025_sommer_rgb --tiles all \
    --out-dir /Volumes/2TB/winmol/ALS_Data/berlin_dop_2025_sommer

# Spring DOP 2025, one tile
.venv-cpu/bin/python benchmark/fetch_berlin_dop.py \
    --service dop_2025_fruehjahr --tiles 381_5829 \
    --out-dir /Volumes/2TB/winmol/ALS_Data/berlin_dop_2025_fruehjahr
```

`--tiles all` is the eleven-tile Berlin benchmark set hard-coded as `DEFAULT_TILES`
(`379_5828 379_5829 380_5828 380_5829 381_5828 381_5829 381_5830 382_5828 382_5829
383_5828 383_5829`); otherwise pass tile keys like `381_5829`. The tile key alone defines
the bbox (`E km * 1000`, `N km * 1000`, +1 km), so nothing needs to be looked up.

Expected output per tile: `[fetch] 3dm_33_381_5829_1_be from dop_2021/dop_2021_rgb`,
then `[done] .../3dm_33_381_5829_1_be.tif (3, 5000, 5000) 55.9 MB`, and a final
`11 written, 0 skipped, 0 failed`. Exit status is 1 if any tile failed.

### RGBI reconstruction

The product is 4-band RGBI but a WMS only serves *rendered* 3-band images, so there is no
RGBI GetMap. `--rgbi` fetches both layers and stacks R,G,B from `<service>_rgb` with band
1 (= NIR) of the `<service>_cir` rendering. That is a faithful NIR channel only as far as
the CIR rendering is linear — good enough for a glance at vegetation indices, **not** the
original 4-band raster.

## Output layout on the 2TB volume

```
/Volumes/2TB/winmol/ALS_Data/berlin_dop_2021/
  dop_2021_rgb/3dm_33_<E>_<N>_1_be.tif    3-band uint8, 5000x5000, ~55 MB
  dop_2021_rgbi/3dm_33_<E>_<N>_1_be.tif   4-band uint8 (R,G,B,NIR), ~73 MB
  report/                                 a copy of the overlay document + figures
/Volumes/2TB/winmol/ALS_Data/berlin_dop_2025_sommer/3dm_33_<E>_<N>_1_be.tif
```

## Georeferencing check (do this once per new year/service)

The DOP tile must land on exactly the same extent as the ALS-derived instance raster:

```bash
.venv-cpu/bin/python - <<'PY'
import rasterio
dop = "/Volumes/2TB/winmol/ALS_Data/berlin_dop_2021/dop_2021_rgb/3dm_33_381_5829_1_be.tif"
ins = ("/Users/christian/work/hnee/ForestFormer3D_runs/berlin_out/berlin-2021/"
       "3dm_33_381_5829_1_be/3dm_33_381_5829_1_be_instance_50cm.tif")
for p in (dop, ins):
    with rasterio.open(p) as s:
        print(s.crs, tuple(round(v) for v in s.bounds), s.width, s.height, s.count)
PY
```

Both must print `EPSG:25833 (381000, 5829000, 382000, 5830000)`. Visually, the allotment
street grid in the west, the clearing in the middle and the diagonal path across the south
of that tile fall on the same pixels in both. No offset has ever been needed.

## Overlay figures

`benchmark/plot_berlin_dop_overlays.py` writes four PNGs into
`docs/benchmarks/assets/berlin-dop/`:

```bash
.venv-cpu/bin/python benchmark/plot_berlin_dop_overlays.py          # all four
.venv-cpu/bin/python benchmark/plot_berlin_dop_overlays.py --only b c
```

| fig | file | content |
|---|---|---|
| a | `berlin-dop-381-5829-crowns.png` | full 1 km tile, crown outlines coloured by height |
| b | `berlin-dop-381-5829-zoom.png` | 150 m zoom, DOP beside the ALS instance raster |
| c | `berlin-dop-381-5829-drone2025.png` | same window on the 2025 drone ortho |
| d | `berlin-dop-mosaic-density.png` | eleven-tile mosaic + crown-centroid density |

Input paths are constants at the top of the script (`DOP_DIR`, `ALS_DIRS`, `DRONE`,
`OUT_DIR`) — edit them there for another tile set or another year. It needs
`rasterio`, `matplotlib`, `PIL`, so run it from `.venv-cpu`. The written-up results are
`docs/benchmarks/2026-09-23-berlin-dop-overlay.md`.

Interpretation caveat: 2021 imagery is **leaf-off**, so broadleaves are bare grey-brown
twig structure while pine stands stay dark green — a crown outline that looks empty on the
2021 DOP may be a perfectly good broadleaf. Compare against the leaf-on
`truedop_2025_sommer` before concluding a crown is spurious.

## Discovering a service that is not listed above

1. Query the catalogue (JSON) for the year or product name:
   `https://gdi.berlin.de/geonetwork/srv/ger/q?_content_type=json&fast=index&any=DOP%202025`
2. Take the service id from the record's WMS link and confirm the layer names:
   `https://gdi.berlin.de/services/wms/<service>?service=WMS&version=1.3.0&request=GetCapabilities`
   — the `<Name>` elements under each `<Layer>` are what `--layer` takes.
3. Fetch both URLs from a **Python file with `urllib`**; shell `curl`/`wget` are blocked in
   this environment.
