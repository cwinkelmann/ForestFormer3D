#!/usr/bin/env python3
"""Convert Stefan Reder's PointTreeFormer results into the ForestFormer3D result contract.

The PointTreeFormer run over the Berlin ALS 2021 tiles (carrot,
``/storage/sreder/projects/ALS_Berlin/PointTreeFormer/clipped_<T>.laz``, 2026-10-05, 15 km
tiles of the Tegel block) writes one LAZ 1.4 / point format 6 per km tile **with a 20 m
buffer** (1,040 m square, 26.9 M points for a 25.1 M tile), no CRS record (the coordinates are
ETRS89 / UTM 33N), the ALS attributes kept and three extra dimensions:

    classification_prediction          int64  0 ground, 1 wood, 2 leaf  (ForAINet classes)
    classification_binary_prediction   int64  0 non-tree, 1 tree
    instance_id_prediction             int64  tree id from 0, -1 none; unique within the tile

Measured on 3dm_33_380_5828_1_be: the buffer-free core holds exactly the tile's 25,073,247
source points, so a per-point comparison with the other methods is possible once the core
is in the same point order as the ForestFormer3D result LAS -- ``--order-like`` does that by
matching the scaled integer coordinates and refuses to write when a point has no partner.

Output, the ``ff3d_geo`` contract so every downstream tool works unchanged:

    <out>/<T>.las                 LAS 1.4 pf6, EPSG:25833; treeID int32 (-1 none), semantic uint8
                                  (0/1/2 as above, 255 for anything else), score float32 = -1
                                  (PointTreeFormer writes no per-instance confidence)
    <out>/<T>_trees.gpkg, _crowns.gpkg, _instance_50cm.tif, _semantic_50cm.tif, _report.json/.md

``--min-height 2`` applies the same minimum instance height the stitched mosaics carry
(``ff3d_geo.filter``) so the methods stay comparable; the report records it.

    python benchmark/ptf_to_ff3d.py --laz /storage/sreder/projects/ALS_Berlin/PointTreeFormer/clipped_3dm_33_380_5828_1_be.laz \\
        --out work_dirs/ptf-mosaic/3dm_33_380_5828_1_be --order-like work_dirs/berlin-mosaic-44/3dm_33_380_5828_1_be.las --min-height 2
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import laspy
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ff3d_geo.convert import SEMANTIC_UNLABELLED, result_point_header  # noqa: E402

TILE_RE = re.compile(r"(3dm_33_(\d+)_(\d+)_1_be)")
EPSG = 25833


def tile_from_name(path: Path) -> tuple[str, tuple[float, float, float, float]]:
    m = TILE_RE.search(path.name)
    if not m:
        raise ValueError(f"no 3dm_33_<E>_<N>_1_be tile stem in {path.name}")
    e, n = int(m.group(2)) * 1000.0, int(m.group(3)) * 1000.0
    return m.group(1), (e, n, e + 1000.0, n + 1000.0)


def core_mask(x, y, bounds) -> np.ndarray:
    minx, miny, maxx, maxy = bounds
    return (x >= minx) & (x < maxx) & (y >= miny) & (y < maxy)


def order_like(xyz_int: np.ndarray, ref_int: np.ndarray) -> np.ndarray:
    """Index array ``idx`` such that ``xyz_int[idx] == ref_int`` row by row, by exact match of
    the scaled integer coordinates; raises when any reference point has no partner."""
    if xyz_int.shape != ref_int.shape:
        raise ValueError(f"point count differs: {xyz_int.shape[0]} vs reference {ref_int.shape[0]}")
    key = lambda a: a[:, 0].astype(np.int64) * 4_000_000_000_000 + a[:, 1].astype(np.int64) * 4_000_000 + a[:, 2].astype(np.int64)  # noqa: E731
    k, r = key(xyz_int), key(ref_int)
    order = np.argsort(k, kind="stable")
    pos = np.searchsorted(k[order], r)
    pos = np.clip(pos, 0, k.size - 1)
    idx = order[pos]
    if not np.array_equal(k[idx], r):
        raise ValueError(f"{int((k[idx] != r).sum()):,} reference points have no coordinate match in the PointTreeFormer file")
    return idx


def convert(laz: Path, out_dir: Path, order_like_las: Path | None = None, min_height: float = 0.0,
            cell_m: float = 0.5, masks: bool = True, log=print) -> dict:
    tile, bounds = tile_from_name(laz)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = laspy.read(str(laz))
    x, y, z = np.asarray(src.x), np.asarray(src.y), np.asarray(src.z)
    core = core_mask(x, y, bounds)
    log(f"  {tile}: {src.header.point_count:,} points, {int(core.sum()):,} in the core, {int((~core).sum()):,} in the buffer")
    inst = np.asarray(src.instance_id_prediction)[core].astype(np.int64)
    cls = np.asarray(src.classification_prediction)[core].astype(np.int64)
    sem = np.where((cls >= 0) & (cls <= 2), cls, SEMANTIC_UNLABELLED).astype(np.uint8)
    xi, yi, zi = (np.asarray(src.X)[core], np.asarray(src.Y)[core], np.asarray(src.Z)[core])
    classification = np.asarray(src.classification)[core]
    scales, offsets = src.header.scales, src.header.offsets
    idx = None
    if order_like_las is not None:
        with laspy.open(str(order_like_las)) as ref:
            rh = ref.header
            pts = ref.read()
            refi = np.column_stack([np.asarray(pts.X), np.asarray(pts.Y), np.asarray(pts.Z)]).astype(np.int64)
            if np.allclose(rh.scales, scales) and np.allclose(rh.offsets, offsets):
                mine = np.column_stack([xi, yi, zi]).astype(np.int64)
            else:
                # Different grids (PointTreeFormer 1 cm, our LAS 1 mm from the source): quantise
                # the reference exactly the way PointTreeFormer's writer quantised the same source
                # values -- round((value - ITS offset) / ITS scale) -- so .5 ties fall the same way;
                # the file's own integers are that quantisation already.
                q, o = np.asarray(scales), np.asarray(offsets)
                mine = np.column_stack([xi, yi, zi]).astype(np.int64)
                refi_cmp = np.column_stack([np.round((np.asarray(pts.x) - o[0]) / q[0]), np.round((np.asarray(pts.y) - o[1]) / q[1]),
                                            np.round((np.asarray(pts.z) - o[2]) / q[2])]).astype(np.int64)
                idx = order_like(mine, refi_cmp)
            if idx is None:
                idx = order_like(mine, refi)
            scales, offsets = rh.scales, rh.offsets
        log(f"  {tile}: reordered to the point order of {order_like_las.name}")
        xi, yi, zi = refi[:, 0], refi[:, 1], refi[:, 2]
        inst, sem, classification = inst[idx], sem[idx], classification[idx]
    header = result_point_header(EPSG, scales, offsets)
    las = laspy.LasData(header)
    las.X, las.Y, las.Z = xi, yi, zi
    las.classification = classification.astype(np.uint8)
    las.treeID = inst.astype(np.int32)
    las.semantic = sem
    las.score = np.full(inst.size, -1.0, dtype=np.float32)
    out_las = out_dir / f"{tile}.las"
    las.write(str(out_las))
    info = {"tile": tile, "n_points": int(inst.size), "n_buffer_points": int((~core).sum()),
            "n_instances": int(np.unique(inst[inst >= 0]).size), "reordered": idx is not None}
    height_filter = None
    if min_height > 0:
        from ff3d_geo.filter import filter_short_instances

        height_filter = filter_short_instances(out_las, out_las, min_height=min_height)
        log(f"  {tile}: {height_filter['instances_removed']} instances under {min_height} m removed")
    from ff3d_geo.report import build_report, write_report
    from ff3d_geo.trees import trees_to_gpkg

    gpkg = out_dir / f"{tile}_trees.gpkg"
    info["n_trees"] = trees_to_gpkg(out_las, gpkg)
    if masks:
        from ff3d_geo.raster import las_to_masks

        las_to_masks(out_las, out_dir, cell_m=cell_m, prefix=tile)
    rep = build_report(out_las, gpkg, height_filter=height_filter)
    rep["source"] = {"method": "PointTreeFormer", "file": str(laz), "author": "Stefan Reder"}
    write_report(rep, out_dir / f"{tile}_report.json", out_dir / f"{tile}_report.md")
    log(f"  {tile}: wrote {out_las.name}, {info['n_trees']} trees")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laz", required=True, type=Path, help="clipped_<T>.laz of PointTreeFormer")
    ap.add_argument("--out", required=True, type=Path, help="output directory (<out>/<T>.las ...)")
    ap.add_argument("--order-like", type=Path, default=None,
                    help="a result LAS of the same tile whose point order the output must follow "
                         "(ForestFormer3D's stitched <T>.las), for per-point comparisons")
    ap.add_argument("--min-height", type=float, default=0.0, help="drop instances shorter than this (m); the mosaics use 2")
    ap.add_argument("--cell", type=float, default=0.5)
    ap.add_argument("--no-masks", action="store_true")
    a = ap.parse_args(argv)
    convert(a.laz, a.out, a.order_like, a.min_height, a.cell, not a.no_masks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
