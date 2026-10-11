---
name: ff3d-als-download
description: Use when obtaining Berlin ALS 2021 km tiles (3dm_33_<E>_<N>_1_be.las) or the Brandenburg LGB tiles - the INSPIRE ATOM feed, benchmark/fetch_berlin_als.py, where the files live, CRS, point format and density.
---

# Downloading Berlin ALS 2021 km tiles

The input to every Berlin run is a 1 km LAS tile named `3dm_33_<E km>_<N km>_1_be.las`
(e.g. `3dm_33_381_5829_1_be.las` = the km square whose lower-left corner is
E 381000, N 5829000 in EPSG:25833).

## Where the data comes from

Senatsverwaltung für Stadtentwicklung Berlin, **"Airborne Laserscanning (ALS) Primäre 3D
Laserscan-Daten"**, published as an INSPIRE ATOM download service.

| | |
|---|---|
| Catalogue search | `https://gdi.berlin.de/geonetwork/srv/ger/q?_content_type=json&fast=index&any=Laserscanning` |
| Metadata record | `https://gdi.berlin.de/geonetwork/srv/api/records/85a97801-36bb-4627-8790-f5e53b1e38b3` |
| **ATOM service feed** | **`https://gdi.berlin.de/data/a_als/atom`** |
| Dataset feed | `https://gdi.berlin.de/data/a_als/atom/0.atom` |
| Licence | Datenlizenz Deutschland – Zero – 2.0 (dl-de/zero-2-0) |

**There is no per-tile URL.** The dataset feed links **nine regional ZIP packages** —
`Mitte`, `Nord`, `Nordost`, `Nordwest`, `Ost`, `Sued`, `Suedost`, `Suedwest`, `West`
(`https://gdi.berlin.de/data/a_als/atom/<Region>.zip`) — each several GB and each holding
its region's km tiles as **stored (uncompressed) zip members** named
`3dm_33_<E>_<N>_1_be.las`. 1066 tiles in total cover Berlin. The Tegel/Spandau tiles this
project uses are all in `Nordwest.zip` (68 tiles) and `West.zip` (114 tiles).

Because the members are stored uncompressed, a single tile can be pulled out with HTTP
**range requests** without downloading the multi-GB package. That is what the script does.

## The script

`benchmark/fetch_berlin_als.py` (stdlib only: `urllib`, `zipfile`, `xml.etree`).

```bash
# 1. What exists, and which package holds the tiles you want (downloads nothing)
python3 benchmark/fetch_berlin_als.py --list --tiles 381_5829 380_5828

# 2. Named tiles
python3 benchmark/fetch_berlin_als.py --tiles 381_5829 381_5830 \
    --out-dir /Volumes/2TB/winmol/ALS_Data/berlin_als_2021

# 3. Everything intersecting an EPSG:25833 bbox, in metres (minE minN maxE maxN)
python3 benchmark/fetch_berlin_als.py --bbox 379000 5828000 383000 5830000 \
    --out-dir /Volumes/2TB/winmol/ALS_Data/berlin_als_2021
```

What it does, in order: reads the service feed, follows it to the dataset feed, collects
the `.zip` links, opens each package over range requests and reads **only its central
directory** to build `{tile key -> zip, member, size}`, then copies out the requested
members. The index is cached in `<out-dir>/.als_index.json` (keyed by the package list;
`--refresh-index` rebuilds it), so only the first run pays the nine directory reads.

Behaviour worth knowing:

- A tile whose local size already equals the member size is **skipped** (`[skip] ... have`);
  a size mismatch is refetched. Downloads go to `<name>.part` and are renamed only after
  the size check, so an interrupted run leaves no half file in place.
- After every download the LAS magic (`LASF`), version and point count are read back and
  printed: `[done] 3dm_33_381_5826_1_be.las: 175 MB in 42 s, LAS 1.4, 4,616,907 points`.
- A tile key outside Berlin prints `[miss] ...: in no package` and does not fail the run;
  a genuine failure is reported per tile and sets exit status 1.
- Throughput measured 2026-09-23 from the Mac: ~4 MB/s, i.e. 40 s for a 175 MB tile and
  3–4 min for a 900 MB one.

**Verified**: `--tiles 381_5826` downloaded into a scratch dir gave a file byte-identical
(md5 `2953979032ac96dd0ffc1c60ffed874d`) to the copy already held on the 2TB volume.

## Where the tiles go

```
/Volumes/2TB/winmol/ALS_Data/berlin_als_2021/     37 tiles, 24 GB  (+ README.md)
```

Keep them on the 2TB volume, never in the repo. `README.md` in that folder records the
provenance; the original 37 tiles were fetched on 2026-09-22 with the ad-hoc
`fetch_als_tiles.py` / `fetch_r12_tiles.py` + `remote_zip.py` in the same folder, which
selected exactly the tiles referenced by the `layer` column of
`training_data/WINDWURF_Tegel/ALS segmentation 2017.gpkg`. `benchmark/fetch_berlin_als.py`
is the tracked, general replacement — prefer it.

To put tiles on carrot for inference, copy them there (see `ff3d-inference-km-tiles`):

```bash
scp /Volumes/2TB/winmol/ALS_Data/berlin_als_2021/3dm_33_381_5830_1_be.las \
    carrot:/raid/cwinkelmann/ForestFormer3D/inputs/berlin/
```

## Properties of the data

- **CRS: ETRS89 / UTM 33N, EPSG:25833, heights DHHN2016.** The LAS headers carry **no CRS
  record** (`laspy ... header.parse_crs()` returns `None`). Every tool in this repo
  assumes 25833 and writes it into the output LAS; do not "fix" the inputs.
- LAS 1.4, point format 6, ~1 km × 1 km, 175–950 MB per tile (4.6–25 M points).
- Flight **24/25 Feb and 2 Mar 2021 — leaf-off**. (The LAS headers say created 2021-07-01
  by TerraScan.)
- Classified: `2` ground, `3/4/5` low/medium/high vegetation, `7` low points, `0` default.
  `ff3d_geo convert` keeps this classification in `<stem>_classification.npy` and restores
  it in the result LAS; the CHM baseline in the report uses it.
- Density ~10 pts/m² averaged over a whole km tile (much of it water, roofs and streets);
  over forested 100 m sub-tiles 20–30 pts/m² (e.g. 275,684 points in one 100 m sub-tile of
  381_5829 = 27.6 pts/m²). That is 5–100x sparser than the ForAINetV2 training plots —
  see `ff3d-evaluation` for what that costs.
- A neighbouring crown segmentation exists: `ALS segmentation 2017.gpkg` under
  `training_data/WINDWURF_Tegel/`. Despite the name it was segmented from *this* 2021
  scan (crown heights match the 2021 point maxima to a 0.00 m median).

## Contrast: Brandenburg LGB tiles

Different state, different everything — do not mix conventions.

| | Berlin ALS 2021 | Brandenburg (LGB) |
|---|---|---|
| Name | `3dm_33_<E>_<N>_1_be.las` | `als_33<E>-<N>.laz` (e.g. `als_33424-5870.laz`) |
| Source | ATOM feed, regional zips | `https://data.geobasis-bb.de/geobasis/daten/als/laz/als_33<E>-<N>.zip` |
| Licence | dl-de/zero-2.0 | dl-de/by-2.0, "GeoBasis-DE / LGB" |
| Format | LAS 1.4 pf 6 | LAZ 1.4 pf 6 |
| Density | ~10 pts/m² (km mean) | ~21 pts/m² |
| Classes | 2 ground, 3/4/5 vegetation, 7 low | **0 unclassified (all vegetation and buildings)**, 1, 2 ground, 20 synthetic water — **no vegetation classes**, so canopy = non-ground |
| Flight date | Feb/Mar 2021, leaf-off | not in the headers (files rewritten by las2las 2020-06); see LGB's `bb_laserscandaten_aktualitaet.pdf` |

Local copy: `/Volumes/2TB/winmol/ALS_Data/brandenburg_als/` (Grumsin tiles, with a README
and `models_1m/` DTM/DSM/CHM rasters). The missing vegetation classes matter: any report
step that reads `classification == 2` for ground still works, but "vegetation classes"
filters do not.

## If the feed ever changes

Re-discover it through the GeoNetwork catalogue, which returns JSON:

```
https://gdi.berlin.de/geonetwork/srv/ger/q?_content_type=json&fast=index&any=Laserscanning
```

Look for the record's `link` entries; the ATOM download service is the one under
`gdi.berlin.de/data/<id>/atom`. Fetch it with `urllib` from a Python file — this
environment blocks shell `curl`/`wget`. The script only needs the service-feed URL
(`--feed`); everything below it is parsed.
