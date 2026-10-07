# Discussion and outlook

## What the comparisons say

The two deep models see the same forest. Half of all trees match one-to-one at IoU ≥ 0.5
and the matches are tight (median IoU {{agreement.ff3d_sat.iou_median|3}}); the other half
is dominated by one pattern, SegmentAnyTree cutting a ForestFormer3D crown into pieces
({{agreement.ff3d_sat.split_a_frac|pct}} vs {{agreement.ff3d_sat.split_b_frac|pct}}). Which
of the two is right in those cases cannot be decided from the ALS alone: the
over-segmentation reading is supported by SegmentAnyTree's smaller crowns (p50
{{methods.sat.crown_q.1|0}} vs {{methods.ff3d.crown_q.1|0}} m²) and by its training data
(denser TLS/MLS/ULS scans, where crowns resolve into more detail), but a dense pine stand
with interlocking crowns can genuinely hold more stems than ForestFormer3D's larger
instances suggest. The difference is also not uniform: in the Spandau forest
SegmentAnyTree finds {{ratio_by_footprint.R13.ratio|2}}x as many trees, in Tegel
{{ratio_by_footprint.R12.ratio|2}}x. That the two forests behave differently is itself a
finding -- their stand structure differs -- and the WINMOL field circles in both are the
instrument to settle it.

PointTreeFormer, on the 15 Tegel tiles it covers, matches ForestFormer3D about as often as
SegmentAnyTree does but from the other side: it draws the larger crowns, so its instances are the
ones covered by several ForestFormer3D trees. Three methods thus bracket ForestFormer3D's crown
size -- SegmentAnyTree smaller, PointTreeFormer and AMS3D larger -- which again is a statement
about agreement, not about which size is right.

The classical baseline reports fewer, larger and taller instances and misses the understory
mode of the height distribution altogether; its role in this report is to show what a
model-free method gets from the same points, and the deep models add a second mode of
small trees at 4-7 m that mean shift does not separate.

## What the references say

Outside the forest the picture is clear: {{cadastre.ff3d.detected_frac|pct0}} of the
cadastre's street and park trees have a predicted top within 3 m, the undetected share is
flat across genera, and the heights agree on average with a scatter of
{{cadastre.ff3d.height_mae|1}} m that is of the order of the cadastre's own height classes.
Inside the forest the stand-level check is coarser but consistent: the predicted canopy
height follows the inventory by species, and the {{stands_bias_p90|1}} m by which it exceeds the 2014
figures is roughly what seven growing seasons would add. Neither reference says anything about
individual crowns in closed forest.

## Limitations

* **No ground truth in the forest.** Every method-to-method number is agreement. The 974
  WINMOL crowns were delineated in 2025 on drone imagery of the regeneration layer, so
  even they measure a different stratum than the ALS canopy; a dedicated stem-mapped plot
  on these tiles would be the real test.
* **The model is used far from its training domain.** The ForAINetV2 plots are 5-100x
  denser than the Berlin ALS; the thinning study (appendix) shows F1 falling with density
  and the production setting trades a little recall for 3.7x speed. The two artefacts --
  flat ground blobs and roof instances -- are both domain effects.
* **References are coarse and dated.** The stand map is a 2014 management inventory with
  1-20 ha polygons; the cadastre height is an inspection estimate in whole metres.
* **The mosaic edge is one-sided.** Sub-tiles on the outer edge of the mosaic have a thinner
  halo; the same holds for the border between the first 33 tiles and the eleven added
  later, whose neighbours were split before those tiles existed.
* **Nondeterminism.** Run-to-run variation of the model is below the differences reported
  here (appendix), but not zero.

## Outlook

1. Use the WINMOL field circles as reference: match predicted crowns to the 974 delineated
   ones, by species, and report precision and recall per method -- the first accuracy
   figure on these tiles.
3. Post-filters for production use: drop instances under 2 m with crowns over 50 m², apply
   the building mask by default, and decide a minimum instance size from the field data.
4. A species layer: the stand map gives expected species shares per stand; the summer
   orthophoto and the leaf-off/leaf-on pair give per-crown spectral evidence to assign
   species to individual predicted trees and check them against the cadastre's species.
5. A second epoch: Berlin's next ALS flight against the 2021 one on the same ids, for
   growth and loss per tree.
