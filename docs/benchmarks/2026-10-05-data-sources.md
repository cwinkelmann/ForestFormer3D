# Data sources: where every input to this work comes from

Everything below is either an open dataset of the State of Berlin, a published model with
its released weights, or something computed in this repository from those. Nothing comes
from OpenStreetMap or any crowd-sourced source. Service URLs were last checked on the dates
given; the fetch scripts named here are the record of exactly what was requested.

## 1. Airborne laser scanning: Berlin ALS 2021

| | |
|---|---|
| Publisher | Senatsverwaltung für Stadtentwicklung, Bauen und Wohnen Berlin (Geoportal Berlin / GDI Berlin) |
| Product | "Airborne Laserscanning (ALS) Primäre 3D Laserscan-Daten" |
| Access | INSPIRE ATOM download service `https://gdi.berlin.de/data/a_als/atom` (dataset feed `.../atom/0.atom`), regional zips of km tiles |
| Metadata record | `https://gdi.berlin.de/geonetwork/srv/api/records/85a97801-36bb-4627-8790-f5e53b1e38b3` |
| Licence | Datenlizenz Deutschland – Zero – 2.0 (dl-de/zero-2-0) |
| Flight | 24/25 February and 2 March 2021 -- **leaf-off** |
| CRS | ETRS89 / UTM 33N (EPSG:25833), heights DHHN2016; the LAS headers carry no CRS, the code sets it |
| Tiles | 1 km squares named `3dm_33_<E>_<N>_1_be.las`; 33 used here (Tegel and Spandau), 23-25 M points each |
| Density | ~10 pts/m² averaged over a km tile (much of it water, roofs, streets); 20-30 pts/m² over forested 100 m sub-tiles |
| Classes present | 2 (ground), 3/4/5 (vegetation by height), 7 (noise), 32 -- **no class 6 (building)**: roof points sit in 3/4/5 |
| Fetch | `benchmark/fetch_berlin_als.py` (skill `ff3d-als-download`); copies on the 2TB volume `ALS_Data/berlin_als_2021/` and on carrot `inputs/berlin/` |

The missing building class is why the ALKIS footprints (section 3) exist in this work: the
model predicts tree instances on roofs because nothing in the point cloud says "roof".

For contrast, two Brandenburg tiles (Grumsin) from the LGB
(`https://data.geobasis-bb.de/geobasis/daten/als/laz/`, licence dl-de/by-2.0, attribution
"GeoBasis-DE / LGB") were looked at; they are not part of the Berlin results.

## 2. Orthophotos: Berlin DOP 2021 and TrueDOP 2025

| | DOP 2021 | TrueDOP 2025 summer |
|---|---|---|
| Publisher | Senatsverwaltung für Stadtentwicklung, Bauen und Wohnen Berlin | same |
| Access | WMS 1.3.0 `https://gdi.berlin.de/services/wms/dop_2021`, layers `dop_2021_rgb`, `dop_2021_cir` | WMS `https://gdi.berlin.de/services/wms/truedop_2025_sommer`, layer `truedop_2025_sommer_rgb` |
| Product | DOP20RGBI, 20 cm ground resolution | true orthophoto (buildings do not lean), 20 cm |
| Flight | 22 February 2021 -- **leaf-off**, same season as the ALS | summer 2025 -- **leaf-on** |
| Licence | dl-de/zero-2-0 | as published with the service |
| Fetch | `benchmark/fetch_berlin_dop.py` (skill `ff3d-orthophoto-download`), one 5000 x 5000 px GeoTIFF per km tile, EPSG:25833 |

A WMS serves rendered 3-band images, so the "RGBI" set is reconstructed: R, G, B from the
RGB rendering and the near-infrared band from the CIR rendering's first channel. It is
good for a visual check of vegetation, not for radiometrically calibrated indices. The
2025 summer delivery is 4-band as served and is drawn as RGB.

## 3. Building footprints: Berlin ALKIS

ALKIS is the *Amtliches Liegenschaftskatasterinformationssystem*, the official real-estate
cadastre every German state maintains; its building layer (Gebäude) is the legally
surveyed outline of each building. Berlin publishes it as open data through its spatial
data infrastructure.

| | |
|---|---|
| Publisher | Senatsverwaltung für Stadtentwicklung, Bauen und Wohnen Berlin, via GDI Berlin |
| Access | WFS `https://gdi.berlin.de/services/wfs/alkis_gebaeude`, feature type `alkis_gebaeude:gebaeude` (title "Gebäude"), GeoJSON output, EPSG:25833 |
| Attributes kept | `uuid` (ALKIS identifier), `gfk`/`bezgfk` (Gebäudefunktion, e.g. 1010 "Wohnhaus"), `bat`/`bezbat` (Bauart), `baw`/`bezbaw` (Bauweise), `nam`, `shape_area` |
| Service title / provider | "ALKIS Berlin Gebäude", Senatsverwaltung für Stadtentwicklung, Bauen und Wohnen Berlin (from the WFS GetCapabilities, checked 2026-10-05) |
| Licence | Datenlizenz Deutschland – Zero – 2.0 (dl-de/zero-2-0), "keine Zugriffsbeschränkungen" -- stated in the capabilities document |
| Catalogue | the same dataset is listed in the national catalogue geoportal.de (records there mirror the GDI Berlin metadata); the data itself was taken from the GDI Berlin service above, never from a catalogue download |
| Fetch | `benchmark/fetch_berlin_buildings.py`, one request per km-tile bounding box, paged; result `ALS_Data/berlin_buildings/alkis_buildings.gpkg` with layers `buildings` and `tiles` (which km boxes were fetched) |
| Coverage | **16,257 footprints over all 33 tiles** since 2026-10-05. The first fetch (2026-09-23) covered 23 tiles; the ten others were added on 2026-10-05 and the building mask re-run for them. Five tiles genuinely contain no building (376_5827, 376_5828, 377_5827, 377_5828, 381_5828). |
| Use | `ff3d_geo buildings`: every predicted point inside a buffered footprint becomes class "building", an instance mostly inside footprints is dropped (`--min-roof-fraction`), surviving ids are not renumbered |

With all 33 tiles covered, the mask removes **35,480 of the 832,130 ForestFormer3D trees
(4.26 %)**, leaving 796,650 -- counted as distinct tree ids across the mosaic. Count ids,
not table rows: the stitched per-tile tree tables list a tree straddling a km border once,
in the tile holding most of it, whereas the mask step regenerates each tile's table from
that tile's own LAS and lists such a tree on both sides (7,134 double-listed rows). The
densest built-up tiles lose the most: 383_5827 40 % of its "trees" (5,330 instances),
383_5826 3,982, 380_5830 3,800. Any "buildings masked" figure quoted before 2026-10-05
was computed with ten tiles unmasked and is superseded by these numbers.

## 4. Models

**ForestFormer3D** -- Xiang, Wielgosz, Puliti, Král, Krůček, Missarov, Astrup: "ForestFormer3D",
ICCV 2025 (oral), arXiv 2506.16991. Code is this repository (a fork; licence CC BY-NC 4.0).
The released weights `epoch_3000_fix.pth` and the ForAINetV2 training/benchmark plots come
from Zenodo record **16742708** (`https://zenodo.org/records/16742708`, fetched by
`benchmark/fetch_zenodo.sh`). No training was done here; the released checkpoint is used
as is on data 5-100x sparser than its training plots (see the density study).

**SegmentAnyTree** -- Wielgosz et al. 2024 (torch-points3d / PointGroup). Run from the fork
`cwinkelmann/SegmentAnyTree` at commit `661c88e` with the published checkpoint
`model_file/PointGroup-PAPER.pt` (665 MB, Git LFS upstream), image `segment-any-tree:cu118`.

**AMS3D** -- an in-house adaptive mean-shift crown segmentation (`ff3d_geo/ams3d.py`), ported
from a spike in the `GEE_animation` repository; CPU only, no learned weights. It is the
classical baseline, not a published method.

## 5. Reference data that exists but is NOT used in this report

**Is there a source listing every individual tree? No.** Berlin's tree cadastre
(*Baumbestand Berlin*, WFS `https://gdi.berlin.de/services/wfs/baumbestand`, feature types
`baumbestand:strassenbaeume` and `baumbestand:anlagenbaeume`, licence dl-de/zero-2-0,
checked 2026-10-05) lists the managed trees -- street trees and, in the service's own
words, "einen Teil der Bäume in Grünanlagen" -- with species, address, planting year and
height. It does not cover the forest: Berlin's forests are inventoried by stand, not by
tree, and no per-tree forest inventory is published. On the Tegel and Spandau tiles the
cadastre would therefore give species-labelled reference trees along streets and in parks
at the forest edge, and nothing inside it. It has not been fetched or used here.

The WINMOL 2025 field campaign delineated 974 crowns with species labels inside three
sample circles (*Probekreise*) in Revier 12 Tegelsee and Revier 13 Spandau
(`training_data/WINDWURF_Tegel/Revier_1{2,3}/202507_*.gpkg`). They are the only
species-labelled, hand-delineated reference on these tiles. No figure in this report uses
them: every "agreement" number compares two methods with each other and says nothing about
which is right. Connecting the methods to these circles is the obvious next step.

## 6. What this repository computed from the above

Per km tile and method: a result LAS 1.4 with `treeID`/`semantic`/`score` extra
dimensions (the ALS classification kept), a tree table (`_trees.gpkg`), crown polygons
(`_crowns.gpkg`), 0.5 m instance and semantic GeoTIFFs, a border-check JSON and a report;
mosaic-wide `stitch.json`. The 20 m halo split of each km tile, the mosaic-wide stitch, the
seam metrics, the instance agreement, the terrain rasters (DTM/DSM/CHM from the ALS), the
Potree site and the QGIS project are all derived here and documented in the chapters that
follow. Nothing was obtained from the NAS mirror or any third-party processed product.
