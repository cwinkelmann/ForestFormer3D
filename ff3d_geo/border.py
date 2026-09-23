"""Seam metrics for tiled inference: how badly a sub-tile's own 100 m borders cut
into instance segmentation, and how many instances a crown split by such a line
was fragmented into.

Every metric here works on ABSOLUTE coordinates and treats every multiple of
``size_m`` as a border line -- both axes, independent of any tile origin. That is
correct for the sub-tile grid this repo produces (``ff3d_geo.split``/``merge``):
sub-tile origins are ``floor(min / size_m) * size_m``, so their shared edges are
exactly the multiples of ``size_m`` in UTM easting/northing, and a km-tile border
lands on a multiple of ``size_m`` too whenever ``size_m`` divides 1000 (100 does).
No origin parameter is needed as a result.

``no_instance_profile`` measures artefact (2) of the design doc (a strip of
under-segmented, ``treeID == -1`` vegetation along every line); ``split_pairs``
measures artefact (3) (one crown fragmented into two ``treeID``s facing each other
across a line). ``border_check`` is the single entry point over a result LAS,
reading it once for both. See ``docs/superpowers/specs/2026-09-23-seamless-tree-ids-design.md``
section 1b for the numbers these are meant to reproduce.
"""

from __future__ import annotations

import numpy as np

# ALS classes treated as "vegetation-ish, non-ground" for the border profile:
# 3 low, 4 medium, 5 high vegetation.
VEGETATION_CLASSES = (3, 4, 5)


def _distance_to_grid(coord: np.ndarray, size_m: float) -> np.ndarray:
    """Distance of each ``coord`` value to the nearest multiple of ``size_m``."""
    rem = np.mod(coord, size_m)
    return np.minimum(rem, size_m - rem)


def no_instance_profile(
    x: np.ndarray,
    y: np.ndarray,
    tree_id: np.ndarray,
    vegetation: np.ndarray,
    size_m: float = 100.0,
    bin_m: float = 0.5,
    max_m: float = 10.0,
) -> dict:
    """Fraction of unlabelled (``treeID == -1``) vegetation points by distance to
    the nearest ``size_m`` grid line, in ``bin_m`` bins out to ``max_m``.

    Returns ``{"bins": [lo, ...], "frac": [...], "n": [...], "interior_frac": float,
    "strip_excess_pp": float}``. ``bins`` are the ``round(max_m / bin_m)`` bin lower
    edges; ``frac[i]`` is the ``treeID == -1`` fraction of vegetation points whose
    distance falls in ``[bins[i], bins[i] + bin_m)`` (0.0 for an empty bin); ``n[i]``
    is that bin's vegetation point count. ``interior_frac`` is the ``treeID == -1``
    fraction of vegetation points at least ``max_m`` from any line. ``strip_excess_pp``
    is ``100 * (frac(< max_m) - interior_frac)``, the percentage-point excess of
    unlabelled points the whole ``< max_m`` strip carries over the interior level.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    tree_id = np.asarray(tree_id)
    vegetation = np.asarray(vegetation, dtype=bool)

    dist = np.minimum(_distance_to_grid(x, size_m), _distance_to_grid(y, size_m))
    veg_dist = dist[vegetation]
    veg_unlabelled = tree_id[vegetation] == -1

    n_bins = int(round(max_m / bin_m))
    bins = [i * bin_m for i in range(n_bins)]
    frac = []
    n = []
    for i in range(n_bins):
        lo, hi = i * bin_m, (i + 1) * bin_m
        mask = (veg_dist >= lo) & (veg_dist < hi)
        count = int(mask.sum())
        n.append(count)
        frac.append(float(veg_unlabelled[mask].mean()) if count else 0.0)

    interior_mask = veg_dist >= max_m
    interior_frac = (float(veg_unlabelled[interior_mask].mean())
                     if interior_mask.any() else 0.0)

    strip_mask = veg_dist < max_m
    strip_frac = (float(veg_unlabelled[strip_mask].mean())
                 if strip_mask.any() else 0.0)
    strip_excess_pp = 100.0 * (strip_frac - interior_frac)

    return {
        "bins": bins,
        "frac": frac,
        "n": n,
        "interior_frac": interior_frac,
        "strip_excess_pp": strip_excess_pp,
    }


def _instance_extents(x: np.ndarray, y: np.ndarray, tree_id: np.ndarray):
    """Per-instance ``(ids, x_min, x_max, y_min, y_max)`` for ``tree_id >= 0``."""
    valid = tree_id >= 0
    ids, inverse = np.unique(tree_id[valid], return_inverse=True)
    n = len(ids)
    x_min = np.full(n, np.inf)
    x_max = np.full(n, -np.inf)
    y_min = np.full(n, np.inf)
    y_max = np.full(n, -np.inf)
    vx, vy = x[valid], y[valid]
    np.minimum.at(x_min, inverse, vx)
    np.maximum.at(x_max, inverse, vx)
    np.minimum.at(y_min, inverse, vy)
    np.maximum.at(y_max, inverse, vy)
    return ids, x_min, x_max, y_min, y_max


def _nearest_line(v: np.ndarray, size_m: float) -> np.ndarray:
    return np.round(v / size_m) * size_m


def _crosses_a_line(lo: np.ndarray, hi: np.ndarray, size_m: float) -> np.ndarray:
    """True where an interior multiple of ``size_m`` lies strictly inside ``(lo, hi)``."""
    eps = 1e-6 * size_m
    return np.floor((hi - eps) / size_m) > np.floor((lo + eps) / size_m)


def _pair_candidates(end_idx, end_line, end_lo, end_hi, start_idx, start_line, start_lo,
                     start_hi) -> list:
    """Candidate (overlap, end_instance_idx, start_instance_idx) triples: an ``end``
    instance (ending at a line) matched with a ``start`` instance (starting across the
    SAME line) whose PERPENDICULAR extent (the other axis) overlaps.

    ``end_lo``/``end_hi`` and ``start_lo``/``start_hi`` are that perpendicular extent --
    overlap there is the pairing signal, since the line itself only says the two
    fragments are adjacent, not that they belong to the same crown.
    """
    if len(end_idx) == 0 or len(start_idx) == 0:
        return []
    candidates = []
    for line in np.unique(np.concatenate([end_line, start_line])):
        e = np.where(end_line == line)[0]
        s = np.where(start_line == line)[0]
        if len(e) == 0 or len(s) == 0:
            continue
        lo = np.maximum(end_lo[e][:, None], start_lo[s][None, :])
        hi = np.minimum(end_hi[e][:, None], start_hi[s][None, :])
        overlap = hi - lo
        ei, si = np.where(overlap > 0)
        for k in range(len(ei)):
            i, j = int(end_idx[e[ei[k]]]), int(start_idx[s[si[k]]])
            if i == j:
                continue
            candidates.append((float(overlap[ei[k], si[k]]), i, j))
    return candidates


def _greedy_match(candidates: list, used: set) -> int:
    """Assign ``candidates`` (already the full cross-axis pool) best overlap first,
    each instance used at most once; return the number of pairs formed."""
    n_pairs = 0
    for _, i, j in sorted(candidates, key=lambda c: -c[0]):
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        n_pairs += 1
    return n_pairs


def split_pairs(x: np.ndarray, y: np.ndarray, tree_id: np.ndarray,
                size_m: float = 100.0, tol_m: float = 1.0) -> dict:
    """Count instances split by a ``size_m`` grid line: touching, crossing, and
    greedily paired fragments.

    Per-instance extents are the min/max of its points on each axis
    (``np.minimum.at``/``np.maximum.at``). An instance is "touching" when any of its
    four extent edges (``x_min``, ``x_max``, ``y_min``, ``y_max``) lies within
    ``tol_m`` of the nearest ``size_m`` line; "crossing" when a line lies strictly
    inside its x or y extent (points on both sides). Pairing matches, for every
    line, instances that END at it (an edge approaching from below/left) with
    instances that START across it (an edge approaching from above/right) whose
    extent on the OTHER axis overlaps, best overlap first, each instance used once.

    Returns ``{"n_trees": int, "n_touching": int, "n_crossing": int, "n_pairs": int,
    "touching_frac": float}``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    tree_id = np.asarray(tree_id)

    ids, x_min, x_max, y_min, y_max = _instance_extents(x, y, tree_id)
    n_trees = len(ids)
    if n_trees == 0:
        return {"n_trees": 0, "n_touching": 0, "n_crossing": 0, "n_pairs": 0,
                "touching_frac": 0.0}

    x_min_line = _nearest_line(x_min, size_m)
    x_max_line = _nearest_line(x_max, size_m)
    y_min_line = _nearest_line(y_min, size_m)
    y_max_line = _nearest_line(y_max, size_m)

    x_min_touch = np.abs(x_min - x_min_line) <= tol_m
    x_max_touch = np.abs(x_max - x_max_line) <= tol_m
    y_min_touch = np.abs(y_min - y_min_line) <= tol_m
    y_max_touch = np.abs(y_max - y_max_line) <= tol_m

    touching = x_min_touch | x_max_touch | y_min_touch | y_max_touch
    n_touching = int(touching.sum())

    x_cross = _crosses_a_line(x_min, x_max, size_m)
    y_cross = _crosses_a_line(y_min, y_max, size_m)
    n_crossing = int((x_cross | y_cross).sum())

    idx = np.arange(n_trees)
    # x-lines: instance ends at the line from the left (x_max_touch) faces one
    # starting from the right (x_min_touch); perpendicular extent is y. y-lines:
    # perpendicular extent is x. Both axes' candidates are pooled into ONE greedy
    # pass (best overlap first, globally) rather than resolved axis by axis, so an
    # instance with both an x- and a y-candidate goes to whichever is the better
    # match instead of whichever axis happens to be processed first.
    candidates = _pair_candidates(
        idx[x_max_touch], x_max_line[x_max_touch], y_min[x_max_touch], y_max[x_max_touch],
        idx[x_min_touch], x_min_line[x_min_touch], y_min[x_min_touch], y_max[x_min_touch],
    ) + _pair_candidates(
        idx[y_max_touch], y_max_line[y_max_touch], x_min[y_max_touch], x_max[y_max_touch],
        idx[y_min_touch], y_min_line[y_min_touch], x_min[y_min_touch], x_max[y_min_touch],
    )
    n_pairs = _greedy_match(candidates, set())

    return {
        "n_trees": n_trees,
        "n_touching": n_touching,
        "n_crossing": n_crossing,
        "n_pairs": n_pairs,
        "touching_frac": n_touching / n_trees,
    }


def border_check(las_path, size_m: float = 100.0) -> dict:
    """Read a result LAS once and return the merged :func:`no_instance_profile` and
    :func:`split_pairs` metrics, plus ``n_points``.

    ``vegetation`` is ``classification in {3, 4, 5} and semantic not in {0, 255}``
    (ALS vegetation-ish minus model-ground and model-nodata points).
    """
    import laspy

    las = laspy.read(str(las_path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    tree_id = np.asarray(las.treeID, dtype=np.int64)
    classification = np.asarray(las.classification, dtype=np.int64)
    semantic = np.asarray(las.semantic, dtype=np.int64)

    vegetation = np.isin(classification, VEGETATION_CLASSES) & (semantic != 0) & (semantic != 255)

    profile = no_instance_profile(x, y, tree_id, vegetation, size_m=size_m)
    pairs = split_pairs(x, y, tree_id, size_m=size_m)
    return {**profile, **pairs, "n_points": int(len(x))}
