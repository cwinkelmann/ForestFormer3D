# Seamless tree ids across sub-tiles: design

Date: 2026-09-23
Branch: `fix/review-findings`
Status: approved for planning (plan: `docs/superpowers/plans/2026-09-23-ff3d-05-seamless-ids.md`)

## 1. The two artefacts

The Potree viewer of the Berlin 2021 results shows (user's words) "every sub-tile has
a different colour scheme" and "obvious tile seams". They have two unrelated causes,
one in the viewer and one in the inference pipeline. Both were measured on
`3dm_33_381_5829_1_be` (23,246,640 points, 31,385 trees, 100 sub-tiles).

### 1a. Colour scheme per sub-tile (viewer, fixed in `afac64e`)

`ff3d_geo.merge.merge_las` gives sub-tile `k` the id offset `sum(max_id_j + 1, j < k)`,
so every 100 m sub-tile owns a contiguous id block: on the tile above the 100 blocks are
`0..390`, `391..562`, `563..750`, ... with `next.min - prev.max == 1` everywhere and no
overlap. The viewer's tree-id mode maps an id to one of `LUT_N = 8192` texture slots by
its position in `[min id, max id]` (3.83 ids per slot here, about 88 slots per sub-tile)
and coloured slot `i` with `hashColor(i * 2654435761 + 7)`, where `hashColor` multiplies
by the same Knuth constant again. Two multiplications by `A` modulo 2^32 compose into one
linear map, so the hue was `frac(0.326 - 0.000385 * i)`: a slow ramp over the slot index
(a full hue cycle every ~2600 slots, i.e. every ~30 sub-tiles) and the value channel a
ramp with period ~69 slots (about one sub-tile). Reproduced exactly in Python with JS
int32 semantics: the mean hue difference between consecutive slots was 0.000 (random:
0.25), the per-sub-tile hue circular std 0.01 turns, and the sub-tile mean hues walked
0.31, 0.28, 0.26, 0.24, ... in id-block order. A screenshot confirmed one tint per 100 m
column band.

Fix (commit `afac64e`, `benchmark/potree_index.html` and the identical site copy
`/Volumes/2TB/winmol/ALS_Data/berlin_potree/index.html`): slot colours come from a
lowbias32 xorshift-multiply mixer, so neighbouring slots are unrelated (mean hue
difference 0.251, value 0.117, both at the random expectation; per-88-slot-block hue
resultant 0.09 instead of 0.998). Tree markers use the same slot colour as their tree's
points. Verified with a headless screenshot: no column bands remain. The draped instance
PNG (`build_potree_site.py`) already uses a random LUT and needed no change.

### 1b. Geometric seams (pipeline)

1. **No instance crosses a 100 m line.** Each sub-tile is inferred alone and `merge_las`
   only renumbers: of 31,385 instances, 0 have points on both sides of any 100 m line.
2. **A strip along every line is under-segmented.** Fraction of vegetation points
   (ALS class 3/4/5, model semantic != ground) with `treeID == -1` by distance to the
   nearest internal 100 m line, 0.5 m bins: 0.374 at 0-0.5 m, 0.317, 0.269, 0.232, 0.205
   (2-2.5 m), 0.172 (3 m), 0.150 (5 m), 0.131 (8 m), 0.126 (10 m); the interior level
   (>= 10 m from any line) is 0.124. It is a gradient over ~10 m, not a hard band, and the
   km-tile borders show the same profile (0.349 at 0 m, 0.156 at 9 m, measured on lines
   >= 10 m from any perpendicular line). The strip within 10 m of a line holds 35.6 % of
   the vegetation points and carries a 6.3 percentage-point excess of unlabelled points,
   about 330 k points per km tile with no tree because of the tiling.
   Cause: the model only sees the sub-tile's own points, so a crown cut by the line is a
   half crown without its apex on one side (missed or below `score_th`) and the cylinder
   lattice (`generate_cylindrical_regions`, centres from `x_min` in steps of `radius/4`)
   covers the outermost 16 m with one-sided cylinders; the per-cylinder edge rejection
   in `_predict_full_plot` (`near_edge = dist > radius - 0.5`) only matters where the
   cylinder edge coincides with the tile edge. Nothing in `merge_instances_by_score`
   is border-specific.
3. **Split crowns.** 5,150 trees (16.4 %) have a point extent within 1 m of a line (the
   geometric expectation for uncut crowns of this size is 16.8 %, so the count itself is
   not inflated). Pairing crowns that end at a line with crowns starting on the other
   side of the same line with overlapping extents gives 1,843 pairs (899 across x-lines,
   944 across y-lines): about 3,700 ids are fragments of ~1,843 real trees, a 6.2 %
   over-count of the tree table.

## 2. Options

| Option | Removes | Cost | Verdict |
|---|---|---|---|
| A. Post-hoc merge of border fragments over the merged LAS (join instances that touch the same line segment) | split ids (3) | none at inference | Does not restore the ~330 k points per km tile that no instance claimed (2); pairing by extent is heuristic (1,843 pairs is an estimate, not a match). Rejected as the main fix; its metric survives as validation. |
| B. Core-plus-halo inference: sub-tiles carry a halo of `h` m around a 100 m core, taken from the km tile and its neighbours; a stitch keeps each point's label from the sub-tile whose core contains it and unifies instances across borders by their overlap on shared halo points | (1), (2), (3) | `(100 + 2h)^2 / 100^2` = 1.96x points and inference time at h = 20 m | **Recommended.** The border profile reaches the interior level at ~10 m, so any h >= 16 m (the cylinder radius) makes a core point as well covered as an interior point. |
| C. B with 200 m cores | as B | `240^2 / 200^2` = 1.44x | Same code path (`--size 200`); try after B works on 100 m. Per-tile GPU memory in `_predict_full_plot` grows with the tile's point count (votes, mask index lists), so it needs one measured run. |
| D. Whole-km-tile inference (no sub-tiles) | internal seams only | 23-25 M points in one `_predict_full_plot` call | km borders remain; 625 cylinders per 100 m become 62,500; untested memory. Rejected. |
| E. Larger `region_step_factor` to pay for B | speed | -0.05 F1 on the 28-plot benchmark (`docs/benchmarks/2026-09-23-inference-profile.md`) | Independent of B; B with factor 0.5 would be ~1.7x faster than today with seams removed, at the step's accuracy cost. Decide separately. |

## 3. Recommended design (B)

- `split` writes sub-tiles of core `size_m` (100) plus halo `buffer_m` (20) in local
  coordinates relative to the core's lower-left corner (halo points are at
  `-20..0` and `100..120`); file names are unchanged (`<prefix>_E<x>_N<y>_100m.las`, the
  token is the core origin, so `parse_origin` and `results_to_las` need nothing). Halo
  points come from the km tile itself and from `--neighbours` km tiles when given, so
  km-tile borders behave like internal ones. For every sub-tile point its global identity
  `(source tile index, point index in that file)` goes to `<stem>_ident.npy`
  (`IDENT_DTYPE = [("tile", "<u2"), ("index", "<u4")]`) and `<out>/split_manifest.json`
  lists the sources (key, path, point count) and the sub-tiles (stem, core origin,
  n_points, n_core). `min_points` applies to core points.
- `run` is unchanged: the model sees 140 m tiles; its edge effects now fall in the halo.
- `stitch` (new, host-only) reads one or more manifests and the sub-tile result LAS
  files, keys every point by `(tile, index)`, unifies instances of adjacent sub-tiles
  (8-neighbourhood) when their IoU over the points both sub-tiles predicted is
  >= `iou_threshold` (0.5) with at least `min_shared` (20) shared points, and assigns
  every component one global id. Global ids are dense `0..N-1` over the whole mosaic in
  a pseudo-random (hash-of-root) order so consecutive ids are spatially unrelated: the
  viewer's 8192-slot LUT then never maps one sub-tile onto one slot, which a
  `tile * 10^6 + subtile * 10^4 + local` scheme would (its per-tile range of 10^6 puts
  ~122 ids per slot and every sub-tile into ~3 colours). Output per source km tile:
  `<T>.las` (all source points in source order, extra dims as `results_to_las`),
  `<T>_trees.gpkg`, `<T>_report.json/.md`, plus `<out>/stitch.json` (pairs tested,
  unified, components, cross-km unifications, id map size). A tree on a km border has
  the same id in both km-tile LAS files. `merge` stays for old outputs (halo 0).
- Validation (`border-check`): the no-instance excess in the 10 m strip must drop from
  6.3 pp to ~0, the split-pair count from 1,843 to ~0, and border-touching crowns from
  16.4 % to the natural rate for a 1 m tolerance (about 8 %: each of a crown's four
  extent edges lands within 1 m of a line with probability 2/100). Run on
  `3dm_33_381_5829_1_be` with its seven available neighbours first; the r12 single-tile
  run has no halo and must be unchanged (402 trees, 397 CHM maxima).
