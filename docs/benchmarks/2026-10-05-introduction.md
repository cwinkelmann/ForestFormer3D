# Introduction

## The question

Berlin published its 2021 city-wide airborne laser scan as open data. If a
published tree-segmentation model can turn such a scan into a map of individual trees
without site-specific training, the city's forest and street trees could be inventoried
from data that already exists. This report asks three things of ForestFormer3D (Xiang et
al., ICCV 2025), a recent state-of-the-art model for individual tree segmentation in forest
point clouds:

1. Does it run at km-tile scale on ALS that is 5-100 times sparser than the plots it was
   trained on, and can the per-tile results be merged into one consistent map?
2. How does it compare with the other published deep model for the task (SegmentAnyTree)
   and with a classical baseline (adaptive mean shift, AMS3D), on identical inputs?
3. Are the results consistent with the independent references Berlin publishes -- the
   forest stand inventory and the tree cadastre?

## The study area

![The study area in EPSG:25833. Green: the forest stands of the Berlin Forstbetriebskarte 2014; grey: ALKIS building footprints; squares: the 1 km ALS tiles, labelled easting_northing of their south-west corner in km -- solid blue = processed by all methods and in the mosaic, dashed orange = downloaded and being processed, dashed red = not fetched; thick outlines: the WINMOL 2025 survey footprints of Revier 12 Tegelsee (blue, east) and Revier 13 Spandau (purple, west).](assets/analytics/analytics_study_area.png)

The map explains the shape of the mosaic: the processed block covers the Tegel forest and the whole R13 footprint (the southern R13 tiles, 375_5826 and 376_5826 among them entirely forest, were added in the second batch); the red tiles north and south hold forest of Reviere 11 and 15 outside both survey footprints and were not fetched. Buildings crowd the eastern tiles (Tegel, Heiligensee) and are absent from the forest tiles, which is why the building mask matters on some tiles and not at all on others.

The area is the Tegel and Spandau forest complex in north-western Berlin, chosen because
the WINMOL project surveyed two districts there in 2025 with drones and field circles:
Revier 12 Tegelsee (515 ha) and Revier 13 Spandau (1,034 ha). Fifty-seven km tiles of
the Berlin ALS 2021 were processed -- 33 in a first batch, the eleven that complete the
R13 footprint on 2026-10-06 and thirteen more on 2026-10-08 that square off the mosaic's
southern and eastern edges -- so both survey areas lie entirely inside the mosaic. The tiles run from closed
pine-oak forest in the west through lakeshore and park to the dense housing of Tegel in
the east, so the same mosaic exercises the model on forest, parkland, gardens and roofs.

By the 2014 forest stand map, the canopy inside the forest is 54 % Scots pine, 19 % sessile
oak and 6 % beech, with larch and birch below 5 % each; stands are 1-20 ha. Under that
canopy the WINMOL field circles found a beech-dominated regeneration layer. The scan was
flown leaf-off in February 2021 at about 10 points/m² averaged over a km tile and 20-30
points/m² over forest.

## Data at a glance

| input | source | used for |
|---|---|---|
| Berlin ALS 2021, {{methods.ff3d.tiles}} km tiles | GDI Berlin ATOM feed, dl-de/zero-2-0 | the segmentation |
| ALKIS building footprints (24,888 for 57 tiles) | GDI Berlin WFS | masking roof "trees" |
| Forstbetriebskarte 2014, 740 stands | Umweltatlas WFS | species and inventory height per stand |
| Baumbestand (tree cadastre), 21,452 trees | GDI Berlin WFS | detection and height check outside the forest |
| DOP 2021 (leaf-off) and TrueDOP 2025 summer orthophotos | GDI Berlin WMS | visual checks, viewer, QGIS |
| ForestFormer3D weights `epoch_3000_fix.pth` | Zenodo 16742708 | the model, used as released |
| SegmentAnyTree `PointGroup-PAPER.pt` | the authors' release | comparison method |
| PointTreeFormer results, 15 Tegel tiles | Stefan Reder (HNEE), run of 2026-10-05 | comparison method (results only) |
| WINMOL 2025 survey footprints and 974 field-circle crowns | project data | coverage; not yet used for accuracy |

The data-sources chapter gives the full provenance and licences; the methods chapter
describes how the inputs become the products; the analytics chapter holds every
comparison; the appendix reproduces the working studies with their operational detail.
