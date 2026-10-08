# Comparisons and analytics across the Berlin mosaic

Generated 2026-10-08 by `benchmark/berlin_analytics.py` from the stitched per-tile products on the 2TB volume; every number here is also in `assets/analytics/analytics.json`. The chapter compares the three segmentation methods with each other on identical points and, where an external reference exists, against the Berlin forest inventory and the tree cadastre. It says nothing about which method is *right* in the forest interior -- no per-tree ground truth exists there (data-sources chapter, section 5).

## 1. What was compared

| method | km tiles | trees | trees / km² | points / tree (median) | height p10 / p50 / p90 (m) | crown area p50 (m²) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ForestFormer3D | 57 | 1,427,107 | 25,037 | 285 | 4.8 / 18.4 / 28.6 | 26.1 |
| SegmentAnyTree | 57 | 1,555,522 | 27,290 | 212 | 6.8 / 19.7 / 28.6 | 19.4 |
| AMS3D | 57 | 836,106 | 14,669 | 305 | 10.7 / 21.5 / 29.8 | 36.6 |
| PointTreeFormer | 15 | 357,015 | 23,801 | 298 | 6.7 / 18.4 / 30.1 | 40.0 |

ForestFormer3D, SegmentAnyTree and AMS3D ran on the same 20 m-halo split of each km tile and went through the same mosaic-wide stitch, so per-point labels are directly comparable (SegmentAnyTree and AMS3D preserve point order; ForestFormer3D's result is re-ordered to the split); PointTreeFormer's per-tile results were re-ordered onto the same points. PointTreeFormer covers 15 of the tiles, the subset it has been run on. PointTreeFormer's ids are per km tile (its run had a 20 m buffer but no mosaic-wide stitch), so its border trees are counted on both sides of a km line.

## 2. Trees per km tile

![Trees per km tile. Left and middle: the number of tree instances each method reports per 1 km tile (the label is the count in thousands; the grid is the easting and northing of the tile's south-west corner in km, EPSG:25833; white cells are tiles outside the mosaic). Right: the SegmentAnyTree count divided by the ForestFormer3D count, red where SegmentAnyTree finds more trees, blue where it finds fewer; 1.00 would mean identical totals.](assets/analytics/analytics_tile_grid.png)

What we see: both methods agree on where the trees are -- the darkest cells are the closed forest in the west (R13) and north-east (the Tegel forest north of 5829), the palest are the lake tile 381_5827 (Tegeler See, 2-3 thousand trees) and the housing to the east. The ratio map is not noise: it is a block of red over the R13 forest tiles and a block of blue-to-white over R12 and the built-up tiles, i.e. the two methods differ systematically by forest, not tile by tile.

SegmentAnyTree finds 1.090x the ForestFormer3D tree count over the mosaic (per tile from 0.84 to 1.61). On the 21 tiles that are more than half forest by the stand map the ratio averages 1.08, on the other 36 tiles 1.12. The two surveyed forests differ: R12 0.97 (8 tiles with at least a quarter of the tile inside the footprint), R13 1.20 (15 tiles with at least a quarter of the tile inside the footprint). Where SegmentAnyTree reports more trees it has cut crowns into more pieces (section 4); where it reports fewer, ForestFormer3D's extra instances are mostly the small and flat ones of section 3.

## 3. Size distributions and quality flags

![Per-tree size distributions over the whole mosaic. Each panel is a normalised histogram (area 1) of one attribute of the tree table, one curve per method, so curves of methods with different tree counts are comparable in shape: height above ground in 1 m bins; crown area (convex hull of the instance's points) in 4 m² bins; points per instance on a logarithmic axis. n is the number of trees behind each curve.](assets/analytics/analytics_distributions.png)

What we see: the height panel has two modes for both methods, a tall one at 24-28 m (the pine and oak canopy) and a short one at 4-7 m (understory, hedges, young trees in gardens), with a trough at 12-18 m; ForestFormer3D puts more mass under 8 m and SegmentAnyTree more at 20-30 m. The crown-area panel shows SegmentAnyTree's crowns shifted to smaller areas (mode 8-12 m² against 20-30 m²) -- the signature of cutting crowns into pieces. The points-per-tree panel has a spike at the smallest sizes for ForestFormer3D (instances of 10-20 points, the noise-sized class of the table below) and otherwise the same log-linear decline for both.

| method | trees | flat blobs (< 2 m, > 50 m²) | noise-sized (< 20 points) | taller than 40 m |
| :--- | ---: | ---: | ---: | ---: |
| ForestFormer3D | 1,427,107 | 0 (0.0 %) | 48,101 (3.4 %) | 518 (0.04 %) |
| SegmentAnyTree | 1,555,522 | 0 (0.0 %) | 60,825 (3.9 %) | 573 (0.04 %) |
| AMS3D | 836,106 | 0 (0.0 %) | 239 (0.0 %) | 867 (0.10 %) |
| PointTreeFormer | 357,015 | 0 (0.0 %) | 10,761 (3.0 %) | 153 (0.04 %) |

Both height distributions are bimodal: a canopy mode near 26 m and a second mode at 4-7 m (understory, hedges, young trees in gardens). ForestFormer3D has more of the low mode and of instances under 2 m; SegmentAnyTree's crowns are smaller (crown area p50 20 vs 26 m²) because it cuts more of them (section 4).

The flat-blob flag is the artefact seen in the viewer: an instance with almost no height but a large footprint is bare ground (ALS class 2) that the model labelled as leaf, not a tree. The noise-sized flag counts instances too small to be a crown at 20-30 pts/m². The products of this chapter went through the stitch's minimum-height rule (2.0 m), which removed 21,447 instances mosaic-wide before any table was written; the counts above are after it, and the flat-blob row is what the 2 m rule does not reach (instances taller than 2 m with a wide, flat hull).

On the 15 tiles every method covers (379_5826, 379_5827, 379_5828, 379_5829, 380_5826, 380_5827, 380_5828, 380_5829, 381_5827, 381_5828, 381_5829, 382_5828, 382_5829, 383_5828, 383_5829):

![The same distributions restricted to the tiles every method covers, all methods. Same axes and binning as the previous figure.](assets/analytics/analytics_distributions_3way.png)

What we see: AMS3D has no short mode at all -- its height distribution starts at about 10 m -- and its crowns are the largest. Mean shift with a height-dependent bandwidth merges the understory into the canopy tree above it, which is why it reports the fewest trees and the tallest ones; the learned methods separate that layer. PointTreeFormer is the odd one out in height: a broad mode at 10-18 m, exactly where the other three have their trough, and the heaviest tail of large crowns (hulls over 75 m² are twice as frequent as in ForestFormer3D) -- consistent with it drawing the larger crowns in the agreement table and with mid-height instances that the others split into a canopy tree and an understory one.

| method | trees on these tiles | height p10 / p50 / p90 (m) | crown p50 (m²) |
| :--- | ---: | ---: | ---: |
| ForestFormer3D | 392,764 | 4.5 / 18.2 / 30.1 | 27.3 |
| SegmentAnyTree | 388,450 | 6.9 / 21.3 / 30.4 | 21.6 |
| AMS3D | 232,442 | 10.6 / 22.2 / 31.1 | 39.4 |
| PointTreeFormer | 357,015 | 6.7 / 18.4 / 30.1 | 40.0 |

## 4. Instance agreement: ForestFormer3D vs SegmentAnyTree

![Instance agreement per km tile between ForestFormer3D (blue) and SegmentAnyTree (magenta), computed on identical points. Top: the fraction of each method's trees that have a counterpart in the other method overlapping with IoU ≥ 0.5 (bars), and the median IoU of those matched pairs (black line, right-hand reading on the same 0-1 axis). Bottom: the fraction of each method's trees whose points are covered by two or more instances of the other method -- a tree the other method has split. Tiles are ordered by easting, then northing.](assets/analytics/analytics_agreement.png)

What we see: the matched fractions move together from tile to tile (0.35-0.68) and are lowest on the built-up tiles (380_5830, 383_5826/5827), where both methods segment small garden vegetation differently, and highest on closed forest; the median IoU of a match is flat at 0.71-0.78 everywhere, so when the two agree on a tree they agree on its extent. The bottom panel is the asymmetry the whole chapter turns on: the blue bars (ForestFormer3D trees split by SegmentAnyTree) are 2-4x the magenta ones on every tile, 0.32-0.38 on the R13 forest tiles.

Pooled over 57 tiles and 1,003,586,865 identical points: 819,179 tree pairs overlap with IoU ≥ 0.5, i.e. 56.9 % of ForestFormer3D's 1,439,822 trees and 52.3 % of SegmentAnyTree's 1,565,765; the median IoU of a matched pair is 0.744 (the median of the per-tile medians; all pooled figures weight tiles by their tree count, so they differ in the last digit from the unweighted tile means quoted in the appendix). 28.7 % of ForestFormer3D trees are covered by two or more SegmentAnyTree instances against 10.5 % the other way round: where the two disagree, SegmentAnyTree has mostly cut one crown into several, which is also why it reports more trees.

The other pairs, each on the tiles both methods cover (A = the first named):

| pair | tiles | A matched | B matched | median IoU | A split by B | B split by A |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ForestFormer3D vs AMS3D | 57 | 31.7 % | 53.9 % | 0.725 | 11.2 % | 34.3 % |
| ForestFormer3D vs PointTreeFormer | 15 | 51.1 % | 56.9 % | 0.732 | 17.8 % | 31.6 % |
| SegmentAnyTree vs AMS3D | 57 | 27.2 % | 50.3 % | 0.724 | 9.7 % | 39.3 % |

The split columns say who draws the larger crowns: a method whose trees are often covered by several instances of the other draws them large (AMS3D, PointTreeFormer), one that splits the other's trees draws them small (SegmentAnyTree).

## 5. Seams and the building mask

![Left: seam residue per km tile after the mosaic-wide stitch -- the percentage of crowns that a 100 m grid line (the sub-tile borders used for inference) still intersects, per method; the control value for an arbitrary line through the same crowns is about 8 %. Right: the percentage of ForestFormer3D instances on each tile that the ALKIS building mask removes because at least half of their points lie on a roof.](assets/analytics/analytics_seams_buildings.png)

What we see, left: ForestFormer3D sits at 10-12 % on every tile, i.e. 2-4 points above the control floor, and SegmentAnyTree at 13-16 % with outliers to 20 % -- its smaller, more numerous instances touch lines more often, and some of its matches across sub-tiles fail the IoU threshold. Neither shows the 30-40 % that an un-haloed split produced. Right: the forest tiles lose nothing (no buildings), the lakeside and housing tiles 5-20 %, and 383_5827 -- the densest housing -- 40 % of its instances were roofs or roof vegetation.

| method | trees after stitch | instances unified over halos | cross-km matches | crowns touching a grid line (mean) | strip excess (mean pp) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| ForestFormer3D | 1,427,107 | 1,172,396 | 108,154 | 10.8 % | 0.131 |
| SegmentAnyTree | 1,555,522 | 1,034,887 | 98,959 | 14.1 % | 0.019 |
| AMS3D | 836,106 | 833,314 | 78,810 | 11.0 % | 0.000 |

The seam metrics are what `ff3d_geo border-check` measures on the stitched LAS: the share of crowns a 100 m grid line still cuts and the excess of unlabelled points in a 2 m strip along the lines relative to the interior (the control floor from an offset lattice is about 8 % and 0 pp).

The ALKIS footprints cover every tile; the mask removes 55,442 of 1,439,822 ForestFormer3D instances (3.85 %; per-tile counts, so a tree on a km border is counted in both tiles) and re-labels 38,111,659 points as building. The densest built-up tiles lose the most (383_5827 40.8 %, 384_5826 30.1 %, 383_5826 23.4 %).

## 6. Against the forest inventory (Forstbetriebskarte 2014)

![Predicted canopy height against the forest inventory, one marker per stand (only stands with at least 20 predicted trees and an inventory height). Horizontal axis: the height of the main canopy layer from the 2014 Forstbetriebskarte, a stand mean of dominant trees. Vertical axis: the 90th percentile of ForestFormer3D tree heights inside that stand. Marker size scales with the number of predicted trees in the stand, colour is the stand's dominant species (six most frequent by area; grey = other). The dashed line is equality.](assets/analytics/analytics_stand_heights.png)

What we see: the cloud follows the diagonal -- taller stands in the inventory are taller in the predictions -- and sits 4.3 m above it, consistently for pine, oak and larch; beech (green, the 30-36 m stands) lies on the line. The offset is expected: the ALS is seven growing seasons younger than the inventory, and a 90th percentile of tree tops exceeds a mean of dominant heights. The few stands far above the line at low inventory height (10-15 m) are young stands where tall remnant trees dominate the predicted percentile.

| dominant species | stands | ha | FF3D trees | trees / ha | inventory height (m) | pred. median (m) | pred. p90 (m) | crown p50 (m²) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemeine Kiefer | 264 | 1,261 | 510,464 | 405 | 25.2 | 22.5 | 29.0 | 26.1 |
| Traubeneiche | 106 | 381 | 142,637 | 375 | 25.5 | 21.8 | 29.1 | 27.9 |
| Rotbuche | 19 | 78 | 28,172 | 362 | 33.4 | 20.1 | 32.9 | 28.4 |
| Eiche | 9 | 59 | 21,906 | 370 | 23.8 | 21.0 | 26.0 | 27.2 |
| Europäische Lärche | 39 | 58 | 23,025 | 395 | 24.3 | 23.0 | 29.2 | 26.8 |
| Gemeine Birke | 23 | 48 | 16,301 | 340 | 22.3 | 19.6 | 25.8 | 24.3 |
| Stieleiche | 8 | 36 | 12,445 | 350 | 25.7 | 17.3 | 27.8 | 27.7 |
| Roterle | 7 | 18 | 6,834 | 383 | 24.1 | 18.6 | 24.6 | 15.0 |

Every ForestFormer3D tree was joined to the stand it stands in (557 stands with at least 20 trees). The inventory height is the main canopy layer's height from the 2014 management inventory, a stand mean of dominant trees, so the 90th percentile of the predicted tree heights is the comparable statistic (seven years of growth separate the two, worth a few metres of height). Over all stands the predicted p90 is 4.3 m above the inventory height on average (correlation 0.62); the predicted median sits below it, as it should for a figure that includes the understory.

## 7. Against the tree cadastre (street and park trees)

![Street and park trees of the Berlin tree cadastre against the predictions. Left: for every cadastre tree with a predicted tree top within 3 m, the cadastre's height (horizontal, whole metres as recorded at inspection, hence the vertical stripes) against the height of that predicted tree (vertical), one dot per tree, blue ForestFormer3D and magenta SegmentAnyTree, dashed line = equality. Right: histogram of predicted minus cadastre height for the same pairs, 1 m bins.](assets/analytics/analytics_cadastre.png)

What we see: the stripes are the cadastre's integer heights, which stop at 30 m. Within each stripe the predicted heights spread widely, but their centre rises with the cadastre value and the residual histogram is symmetric about zero with a sharp peak (bias 0.15 m, MAE 4.2 m): most matched pairs agree to within a few metres. The dots far above the diagonal at cadastre heights of 4-8 m are young street trees whose nearest predicted top belongs to a taller neighbour's crown (the nearest-top rule assigns it anyway); the dots far below are large cadastre trees for which only a fragment was predicted. The two methods overlay almost exactly, so the scatter is the reference's and the matching rule's, not the model's.

| method | cadastre trees in the mosaic | with a predicted top ≤ 3 m away | street | park | height pairs | bias (m) | MAE (m) | r |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ForestFormer3D | 21,029 | 81.7 % | 82.2 % | 81.3 % | 15,402 | 0.15 | 4.21 | 0.52 |
| SegmentAnyTree | 21,029 | 86.3 % | 87.2 % | 85.7 % | 16,265 | 0.13 | 4.07 | 0.54 |

The cadastre lists managed trees with a surveyed position and a height that is updated at inspection, so a predicted tree top within 3 m of a cadastre position is counted as a detection (the nearest one only; a cadastre tree hidden under a larger neighbour's crown is a miss by construction). The height comparison uses the detected pairs with a cadastre height. Detection by genus, ForestFormer3D (eight most frequent):

| genus | cadastre trees | detected |
| :--- | ---: | ---: |
| Ahorn | 4,767 | 82.4 % |
| Eiche | 2,835 | 84.3 % |
| Linde | 2,690 | 81.8 % |
| Kiefer | 1,640 | 81.8 % |
| Robinie | 1,225 | 85.2 % |
| Hainbuche | 700 | 72.9 % |
| Birke | 655 | 84.1 % |
| Douglasie | 520 | 84.8 % |

## 8. The WINMOL survey footprints

| footprint | ha | method | footprint covered by the method's tiles | trees inside | trees / covered ha | height p10 / p50 / p90 (m) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| R12 | 515 | ForestFormer3D | 100 % | 193,229 | 375 | 6.1 / 22.6 / 31.5 |
| R12 | 515 | SegmentAnyTree | 100 % | 186,125 | 361 | 9.1 / 24.9 / 31.8 |
| R12 | 515 | AMS3D | 100 % | 131,396 | 255 | 13.1 / 24.2 / 32.0 |
| R12 | 515 | PointTreeFormer | 100 % | 180,312 | 350 | 9.4 / 21.6 / 31.4 |
| R13 | 1,034 | ForestFormer3D | 100 % | 372,827 | 361 | 6.8 / 21.1 / 28.4 |
| R13 | 1,034 | SegmentAnyTree | 100 % | 434,331 | 420 | 7.8 / 21.4 / 28.2 |
| R13 | 1,034 | AMS3D | 100 % | 254,522 | 246 | 12.3 / 22.2 / 29.0 |
| R13 | 1,034 | PointTreeFormer | 0 % | 0 | – | – / – / – |

Density is per covered hectare, so partially covered footprints (R13 Spandau until its eleven remaining tiles are processed; AMS3D on its tile subset) stay comparable; the tree counts are partial.

## 9. Caveats

* Agreement between methods is not accuracy: two methods can agree on a wrong split.
* The inventory heights are stand means from 2014 and the cadastre heights are inspection estimates; both references are coarser than the ALS-derived heights they are compared with.
* The flat-blob and noise flags are descriptive; no filtering was applied to any count in this report.
* A method that covers only part of the tiles (the table says how many) is not comparable with the whole-mosaic totals of the others; its per-tile rates are.
