"""Drop instances that are not trees by height.

ForestFormer3D was trained on forest plots where almost every non-ground point belongs to
a tree, so on the Berlin ALS it labels grass returns on meadows (a few decimetres above
the ground) as *leaf* and groups them into "trees": flat, wide instances paved over open
ground in the lattice of the 16 m inference cylinders. On the 33-tile mosaic 13,522 of
832,130 instances (1.6 %) were under 2 m tall, 1,716 of them with hulls over 50 m², at a
median score of 0.50 against 0.64 for all trees -- the score threshold cannot separate
them, their height can. SegmentAnyTree, whose semantic head knows "non-tree", has 869.

The rule is the tree table's own height: ``top_z`` minus the ground grid at the stem (the
median of the instance's lowest metre of points), exactly as ``trees_to_gpkg`` and the
stitch compute it, so a filtered tree table never disagrees with the filter. Points of a
dropped instance keep their semantic and get ``treeID = -1``; nothing is renumbered.
"""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np

from ff3d_geo.trees import ground_surface, instance_rows


def short_instance_ids(rows, min_height: float) -> list[int]:
    """Tree ids of the rows whose ``height`` is below ``min_height`` (0 = none)."""
    if min_height <= 0:
        return []
    return [int(r["tree_id"]) for r in rows if float(r["height"]) < min_height]


def drop_instances(las_path, out_las, drop_ids, min_height: float | None = None) -> dict:
    """Write ``out_las``: ``las_path`` with the instances in ``drop_ids`` unassigned.

    Their points get ``treeID = -1`` and ``score = 0`` and keep their semantic; every
    other point is untouched, so ids stay comparable before and after. ``out_las`` may
    equal ``las_path`` (the rewrite goes through a temp file). Returns the counts.
    """
    las_path, out_las = Path(las_path), Path(out_las)
    las = laspy.read(str(las_path))
    tree_id = np.asarray(las.treeID, dtype=np.int64)
    has_tree = tree_id >= 0
    before = int(np.unique(tree_id[has_tree]).size)
    drop_ids = np.asarray(sorted(set(int(i) for i in drop_ids)), dtype=np.int64)
    if drop_ids.size and has_tree.any():
        lut = np.zeros(int(max(tree_id.max(), drop_ids.max())) + 1, dtype=bool)
        lut[drop_ids] = True
        drop = has_tree & lut[np.where(has_tree, tree_id, 0)]
    else:
        drop = np.zeros(tree_id.size, dtype=bool)
    tree_id[drop] = -1
    las.treeID = tree_id.astype(np.int32)
    las.score = np.where(drop, 0.0, np.asarray(las.score, dtype=np.float32)).astype(np.float32)
    out_las.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_las.with_name(out_las.name + ".tmp")
    las.write(str(tmp))
    tmp.replace(out_las)
    return {
        "las": str(out_las),
        "n_points": int(tree_id.size),
        "instances_before": before,
        "instances_removed": int(drop_ids.size),
        "instances_after": int(np.unique(tree_id[tree_id >= 0]).size),
        "points_unassigned": int(np.count_nonzero(drop)),
        "min_height": None if min_height is None else float(min_height),
    }


def filter_short_instances(las_path, out_las, min_height: float = 2.0,
                           ground_grid_m: float = 1.0) -> dict:
    """``las_path`` -> ``out_las`` without the instances shorter than ``min_height`` m.

    Heights come from :func:`ff3d_geo.trees.instance_rows` over this LAS alone, so on a
    tile of a stitched mosaic a tree straddling the km border is measured on the points
    this tile holds; the stitch applies the same rule mosaic-wide before writing the
    tables (``stitch(..., min_height=...)``), which is the production path.
    """
    las = laspy.read(str(las_path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    tree_id = np.asarray(las.treeID, dtype=np.int64)
    semantic = np.asarray(las.semantic, dtype=np.int64)
    classification = np.asarray(las.classification, dtype=np.int64)
    score = np.asarray(las.score, dtype=np.float64)
    ground, extent = ground_surface(x, y, z, semantic, classification, ground_grid_m)
    rows = instance_rows(x, y, z, tree_id, score, ground, extent)
    del las
    drop = short_instance_ids(rows, min_height)
    return drop_instances(las_path, out_las, drop, min_height=min_height)
