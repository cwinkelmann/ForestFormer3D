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
| Tiles | 1 km squares named `3dm_33_<E>_<N>_1_be.las`; 44 used here (Tegel and Spandau; 33 on 2026-09-24, eleven more on 2026-10-06), 4-27 M points each (18 M on average; forest tiles 20-27 M, the lake tile 381_5827 3.6 M) |
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
| Coverage | **16,257 footprints over all 33 tiles** since 2026-10-05 (`alkis_buildings_44tiles.gpkg` adds the 2,014 of the eleven R13 tiles now in processing, 18,271 rows). The first fetch (2026-09-23) covered 23 tiles; the ten others were added on 2026-10-05 and the building mask re-run for them. Five tiles genuinely contain no building (376_5827, 376_5828, 377_5827, 377_5828, 381_5828). |
| Use | `ff3d_geo buildings`: every predicted point inside a buffered footprint becomes class "building", an instance mostly inside footprints is dropped (`--min-roof-fraction`), surviving ids are not renumbered |

With all 33 tiles covered, the mask removes **35,480 of the 832,130 ForestFormer3D trees
(4.26 %)**, leaving 796,650 -- counted as distinct tree ids across the mosaic. Count ids,
not table rows: the stitched per-tile tree tables list a tree straddling a km border once,
in the tile holding most of it, whereas the mask step regenerates each tile's table from
that tile's own LAS and lists such a tree on both sides (7,134 double-listed rows). The
densest built-up tiles lose the most: 383_5827 40 % of its "trees" (5,330 instances),
383_5826 3,982, 380_5830 3,800. Any "buildings masked" figure quoted before 2026-10-05
was computed with ten tiles unmasked and is superseded by these numbers.

## 3b. What grows inside the forest: the Berlin forest stand map

The tree cadastre (section 5) stops at the forest edge. Inside it, the Berliner Forsten
inventory by stand, and that map is published:

| | |
|---|---|
| Product | "Alters- und Bestandesstruktur der Wälder – Forstbetriebskarte 2014", Umweltatlas Berlin; catalogue record `f15f6603-4640-3d64-9dfb-45575347a901` |
| Access | WFS `https://gdi.berlin.de/services/wfs/ua_forstbetriebskarte_2014`, feature types `c_hauptbaumarten` (stands with species by canopy layer), `a_mischbaumarten`, `b_forstverwalt` |
| Licence | dl-de/zero-2-0 (from the capabilities document, checked 2026-10-05) |
| Content per stand | id, forest district, stand type, area, age class, and for the main layer (`s1`), two lower layers and the standards: up to five species each with code, German name, mixing share (%), DBH (cm) and height (m) |
| Fetch | `benchmark/fetch_berlin_forest_stands.py` -> `ALS_Data/berlin_forest/forstbetriebskarte_2014.gpkg`; **740 stands, 2,227 ha** intersect the 33-tile mosaic, all in forest district Tegel |
| Caveat | a 2014 management inventory, seven years before the 2021 ALS; stand polygons are 1-20 ha, so it says what a stand is made of, not where any tree stands |

Area-weighted composition of the main canopy layer over those 740 stands: **Scots pine
53.9 %**, sessile oak 19.4 %, beech 6.1 %, European larch 4.0 %, birch 3.7 %, "oak"
unspecified 3.1 %, pedunculate oak 1.4 %, Douglas fir 1.2 %, then black alder, ash, red
oak and small-leaved lime below 1 % each. So the canopy ForestFormer3D segments here is
predominantly pine with oak, which matters for reading the results: the model was trained
on mixed temperate plots, and pine crowns are the sparse, high-crowned case.

The WINMOL survey footprints -- the areas the 2025 drone campaign flew, where the
Probekreise sit -- are `WINDWURF_Tegel/Revier_12/ortho/R12_footprint.gpkg` (515 ha) and
`Revier_13/Ortho/R13_footprint.gpkg` (1,034 ha), EPSG:32633. The administrative district as
a whole is not published, but the stand id encodes it (`00101301-0062-a-H010`: characters
4-5 are the Revier, 12 = Tegelsee, 13 = Spandau), so `benchmark/derive_berlin_reviere.py`
also dissolves the stands per Revier. Both go into `berlin_forest/reviere.gpkg` together
with a coverage layer: the km tiles each area needs and whether the mosaic processed them.
On 2026-10-05 that showed the 33-tile mosaic covering R12 Tegelsee entirely but **R13
Spandau only to 56 %**: 454 of its 1,034 ha lay in eleven km tiles that were downloaded but
never processed (375_5826 and 376_5826 entirely inside the footprint, 376_5825, 377_5826,
378_5827, 378_5826, 377_5825, 374_5826, 375_5825, 375_5829, 374_5825). Those eleven were
processed with all three methods and the mosaic re-stitched on 2026-10-06, so both
footprints are now complete (44 tiles). Reviere 11 and 15, north and south of the mosaic,
remain outside the download extent; the project's coverage layer draws them in red.

The 974 Probekreis crowns of section 5 tell the opposite story -- beech 70 %, pine 12 %,
birch 7 % -- and their median polygon is under 1 m²: they are the regeneration layer under
that pine canopy, not the canopy. The two sources describe different strata of the same
forest and should not be compared as if they were one population.

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

## 5. Reference data: what is used as a check, and what is not yet

**Is there a source listing every individual tree? No.** Berlin's tree cadastre
(*Baumbestand Berlin*, WFS `https://gdi.berlin.de/services/wfs/baumbestand`, feature types
`baumbestand:strassenbaeume` and `baumbestand:anlagenbaeume`, licence dl-de/zero-2-0,
checked 2026-10-05) lists the managed trees -- street trees and, in the service's own
words, "einen Teil der Bäume in Grünanlagen" -- with species, address, planting year and
height. It does not cover the forest: Berlin's forests are inventoried by stand, not by
tree, and no per-tree forest inventory is published. On the Tegel and Spandau tiles the
cadastre would therefore give species-labelled reference trees along streets and in parks
at the forest edge, and nothing inside it. It was fetched for the mosaic extent on
2026-10-05 (`benchmark/fetch_berlin_trees.py` -> `ALS_Data/berlin_trees/baumbestand_berlin.gpkg`:
9,103 street and 12,349 park trees) and is a layer group of the QGIS project. The analytics
chapter uses it as the one per-tree check available: detection rate of cadastre trees by a
predicted tree top within 3 m, and the height residual against the cadastre height. QGIS 3.44 reported the same feature types as live WFS layers
unavailable on load although the service answers every request, so the project reads the
GeoPackage.

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
