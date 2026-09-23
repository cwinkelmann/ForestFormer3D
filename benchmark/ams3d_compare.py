"""Compare two instance segmentations of the same tile, or one against reference crowns.

``iou``: instance-level AGREEMENT between two result LAS files of the same points
(``ff3d_geo`` contract: ``treeID`` extra dim) by Hungarian IoU matching at 0.5,
the rule ``benchmark/instance_diagnostics.py`` uses against ground truth. Neither
side is ground truth here, so the numbers are agreement, not accuracy: "matched"
is the number of (a, b) pairs whose point-set IoU is >= 0.5 under the assignment
that maximises total IoU.

``reference``: the spike's matching rule against a crown polygon file (the R13
``ALS segmentation 2017.gpkg``): a predicted tree counts as matched when its apex
(highest point) lies inside a reference polygon and the heights differ by <= 3 m,
one match per reference crown (the tallest prediction wins). Heights are the
tree's ``height`` column from ``<T>_trees.gpkg`` when given, else top z minus the
lowest ground point within 2 m.

Usage:
    python benchmark/ams3d_compare.py iou A.las B.las [--json out.json]
    python benchmark/ams3d_compare.py reference result.las "ALS segmentation 2017.gpkg" \
        --bbox 376400 5827400 376500 5827500 [--trees result_trees.gpkg] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MATCH_IOU = 0.5
HEIGHT_TOL_M = 3.0


def _load(path):
    import laspy

    las = laspy.read(str(path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    ids = np.asarray(las.treeID, dtype=np.int64)
    cls = np.asarray(las.classification, dtype=np.int64)
    return x, y, z, ids, cls


def _align(a, b):
    """Row order of the two files can differ (sub-tile order); align on x, y, z."""
    ka = np.lexsort((a[2], a[1], a[0]))
    kb = np.lexsort((b[2], b[1], b[0]))
    if not (np.allclose(a[0][ka], b[0][kb]) and np.allclose(a[1][ka], b[1][kb])
            and np.allclose(a[2][ka], b[2][kb])):
        raise SystemExit("the two LAS files do not hold the same points")
    return ka, kb


def _compact(ids):
    out = np.full(ids.shape, -1, dtype=np.int64)
    valid = ids >= 0
    if valid.any():
        uniq, inv = np.unique(ids[valid], return_inverse=True)
        out[valid] = inv
        return out, int(uniq.size)
    return out, 0


def iou_agreement(las_a, las_b) -> dict:
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import coo_matrix

    a, b = _load(las_a), _load(las_b)
    if len(a[0]) != len(b[0]):
        raise SystemExit(f"point counts differ: {len(a[0])} vs {len(b[0])}")
    ka, kb = _align(a, b)
    pa, na = _compact(a[3][ka])
    pb, nb = _compact(b[3][kb])
    both = (pa >= 0) & (pb >= 0)
    inter = coo_matrix((np.ones(int(both.sum())), (pa[both], pb[both])), shape=(na, nb)).tocsr()
    size_a = np.bincount(pa[pa >= 0], minlength=na).astype(np.float64)
    size_b = np.bincount(pb[pb >= 0], minlength=nb).astype(np.float64)
    inter = inter.tocoo()
    iou_vals = inter.data / (size_a[inter.row] + size_b[inter.col] - inter.data)
    # Hungarian on the dense matrix is fine for a 100 m tile but not for ~30k x 30k
    # trees of a km tile; the candidate pairs with IoU > 0 are sparse, so solve the
    # assignment per connected component of the overlap graph instead.
    from scipy.sparse.csgraph import connected_components

    n = na + nb
    g = coo_matrix((np.ones(len(inter.data)), (inter.row, na + inter.col)), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    matched = 0
    matched_iou: list[float] = []
    order = np.argsort(comp[inter.row], kind="stable")
    rows, cols, vals = inter.row[order], inter.col[order], iou_vals[order]
    comps = comp[rows]
    bounds = np.flatnonzero(np.diff(comps)) + 1
    for r, c, v in zip(np.split(rows, bounds), np.split(cols, bounds), np.split(vals, bounds)):
        ur, ir = np.unique(r, return_inverse=True)
        uc, ic = np.unique(c, return_inverse=True)
        m = np.zeros((len(ur), len(uc)))
        m[ir, ic] = v
        ri, ci = linear_sum_assignment(-m)
        hit = m[ri, ci] >= MATCH_IOU
        matched += int(hit.sum())
        matched_iou.extend(m[ri, ci][hit].tolist())
    return {
        "las_a": str(las_a), "las_b": str(las_b), "n_points": int(len(a[0])),
        "trees_a": na, "trees_b": nb, "matched_iou50": matched,
        "share_a_matched": matched / na if na else None,
        "share_b_matched": matched / nb if nb else None,
        "mean_iou_of_matches": float(np.mean(matched_iou)) if matched_iou else None,
        "points_a_assigned": int((pa >= 0).sum()), "points_b_assigned": int((pb >= 0).sum()),
        "points_both_assigned": int(both.sum()),
    }


def reference_match(las_path, ref_gpkg, bbox, trees_gpkg=None) -> dict:
    import geopandas as gpd
    from shapely.geometry import box

    x, y, z, ids, cls = _load(las_path)
    aoi = box(*bbox)
    ref = gpd.read_file(str(ref_gpkg), bbox=tuple(bbox))
    ref = ref[ref.geometry.centroid.within(aoi)].copy()
    ref["geometry"] = ref.geometry.buffer(0)

    heights = None
    if trees_gpkg is not None:
        t = gpd.read_file(str(trees_gpkg), layer="trees")
        heights = dict(zip(t["tree_id"].astype(int), t["height"].astype(float)))
    from scipy.spatial import cKDTree

    ground = cls == 2
    gtree = cKDTree(np.column_stack([x[ground], y[ground]])) if ground.any() else None
    rows = []
    for tid in np.unique(ids[ids >= 0]):
        m = ids == tid
        top = np.argmax(z[m])
        ax, ay, az = x[m][top], y[m][top], z[m][top]
        if not (bbox[0] <= ax < bbox[2] and bbox[1] <= ay < bbox[3]):
            continue
        if heights is not None and int(tid) in heights:
            h = heights[int(tid)]
        elif gtree is not None:
            nn = gtree.query_ball_point([ax, ay], 2.0)
            h = az - (z[ground][nn].min() if nn else z[ground].min())
        else:
            h = az - z.min()
        rows.append(dict(tree_id=int(tid), height=float(h), apex_x=float(ax), apex_y=float(ay)))
    pred = gpd.GeoDataFrame(rows, geometry=gpd.points_from_xy([r["apex_x"] for r in rows],
                                                              [r["apex_y"] for r in rows]),
                            crs=ref.crs)
    hit = gpd.sjoin(pred, ref[["ID", "TreeH", "CrwnDmt", "geometry"]], predicate="within", how="inner")
    hit = hit[(hit.height - hit.TreeH).abs() <= HEIGHT_TOL_M]
    hit = hit.sort_values("height", ascending=False).drop_duplicates("ID")
    return {
        "las": str(las_path), "reference": str(ref_gpkg), "bbox": list(bbox),
        "n_ref": int(len(ref)), "n_pred_apex_in_patch": int(len(pred)), "n_matched": int(len(hit)),
        "recall": float(len(hit) / max(len(ref), 1)), "precision": float(len(hit) / max(len(pred), 1)),
        "ref_height_mean": float(ref.TreeH.mean()), "pred_height_mean": float(pred.height.mean()) if len(pred) else None,
        "matched_height_rmse": float(np.sqrt(((hit.height - hit.TreeH) ** 2).mean())) if len(hit) else None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("iou")
    p.add_argument("las_a", type=Path)
    p.add_argument("las_b", type=Path)
    p.add_argument("--json", type=Path, default=None)
    r = sub.add_parser("reference")
    r.add_argument("las", type=Path)
    r.add_argument("reference", type=Path)
    r.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    r.add_argument("--trees", type=Path, default=None, help="<T>_trees.gpkg for the height column")
    r.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "iou":
        out = iou_agreement(args.las_a, args.las_b)
    else:
        out = reference_match(args.las, args.reference, tuple(args.bbox), args.trees)
    text = json.dumps(out, indent=1)
    print(text)
    if args.json is not None:
        args.json.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
