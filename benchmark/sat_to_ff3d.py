#!/usr/bin/env python3
"""Convert SegmentAnyTree output into the ForestFormer3D result contract.

SegmentAnyTree (Wielgosz et al. 2024, ``run_inference.sh`` of the
``cwinkelmann/SegmentAnyTree`` fork) writes ``final_results/<name>_out.laz``: LAS 1.4 /
point format 6 with the input attributes plus two extra dimensions,

  PredSemantic  uint8   0 = non-tree, 1 = tree
  PredInstance  uint32  tree id from 1, 0 = no instance

in the coordinates of its input. Our km tiles are inferred as the 100 m sub-tiles that
``ff3d_geo split`` cuts (local coordinates 0..100 m, origin in the file name), so a km
tile comes back as up to 100 such files.

This script turns one SegmentAnyTree file, or the set of sub-tile files of one km tile,
into the ``ff3d_geo`` contract so every downstream tool (tree table, masks, report,
Potree site, comparisons) works unchanged:

  <out>/<tile>.las              LAS 1.4 pf6, EPSG:25833, extra dims
                                  treeID   int32   -1 = none, unique within the tile
                                  semantic uint8   0 ground, 1 wood, 2 leaf, 255 unknown
                                  score    float32 SegmentAnyTree has no per-instance
                                                   score in its output, so -1 everywhere
                                the ALS ``classification`` is kept
  <out>/<tile>_trees.gpkg       ff3d_geo.trees.trees_to_gpkg
  <out>/<tile>_report.json/.md  ff3d_geo.report.build_report
  <out>/<tile>_instance_50cm.tif, _semantic_50cm.tif, _crowns.gpkg   ff3d_geo.raster.las_to_masks

Id semantics: SegmentAnyTree's ``PredInstance`` 0 becomes ``treeID`` -1 and every id
``k >= 1`` becomes ``k - 1``; sub-tile ``j`` then gets the offset
``sum(max_id_i + 1 for i < j)`` added, exactly like ``ff3d_geo.merge.merge_las``, so the
ids are unique across the km tile and gaps left by SegmentAnyTree's NMS stay gaps.

Semantic mapping: SegmentAnyTree has two classes, ForestFormer3D three.
``PredSemantic`` 0 (non-tree) -> ``semantic`` 0 (ground) and 1 (tree) -> 2 (leaf);
class 1 (wood) is never produced. Anything else (there is nothing else in a valid file)
-> 255. The "ground vs vegetation agreement" figure of the report therefore compares
SegmentAnyTree's *non-tree* class with the ALS ground class, which also counts
buildings, low vegetation and other non-tree objects as disagreements.

    python3 benchmark/sat_to_ff3d.py \\
        --sat-las work_dirs/sat-<T>/sat_raw/final_results/*_out.laz \\
        --tile <T> --out work_dirs/sat-<T> --runtime-s 2400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ff3d_geo.origin import parse_origin  # noqa: E402

#: SegmentAnyTree PredSemantic -> ForestFormer3D semantic (0 ground, 1 wood, 2 leaf).
SEMANTIC_MAP = {0: 0, 1: 2}
SEMANTIC_UNKNOWN = 255
#: The ``score`` extra dim for every point: SegmentAnyTree writes no per-instance score.
NO_SCORE = -1.0
DEFAULT_EPSG = 25833


def sat_tree_ids(pred_instance) -> np.ndarray:
    """``PredInstance`` (uint32, 0 = none, ids from 1) -> int32 ids with -1 for none."""
    ids = np.asarray(pred_instance).astype(np.int64) - 1
    return np.where(ids < 0, -1, ids).astype(np.int32)


def sat_semantic(pred_semantic) -> np.ndarray:
    """``PredSemantic`` (0 non-tree, 1 tree) -> ``semantic`` uint8 via ``SEMANTIC_MAP``."""
    sem = np.asarray(pred_semantic).astype(np.int64)
    out = np.full(sem.shape, SEMANTIC_UNKNOWN, dtype=np.uint8)
    for src, dst in SEMANTIC_MAP.items():
        out[sem == src] = dst
    return out


def resolve_origin(path, absolute: bool) -> tuple[float, float]:
    """Origin to add to a file's coordinates: from its ``E<x>_N<y>`` name token, or
    ``(0, 0)`` when the file is already in absolute coordinates (``--absolute``)."""
    if absolute:
        return 0.0, 0.0
    return parse_origin(Path(path).name)


def _read_sat(path, absolute: bool) -> dict:
    import laspy

    las = laspy.read(str(path))
    names = set(las.point_format.extra_dimension_names)
    missing = {"PredSemantic", "PredInstance"} - names
    if missing:
        raise ValueError(f"{path}: not a SegmentAnyTree output, missing extra dims {sorted(missing)}")
    e, n = resolve_origin(path, absolute)
    ids = sat_tree_ids(las.PredInstance)
    return {
        "path": str(path),
        "x": np.asarray(las.x, dtype=np.float64) + e,
        "y": np.asarray(las.y, dtype=np.float64) + n,
        "z": np.asarray(las.z, dtype=np.float64),
        "classification": np.asarray(las.classification, dtype=np.uint8),
        "treeID": ids,
        "semantic": sat_semantic(las.PredSemantic),
        "max_id": int(ids.max()) if ids.size else -1,
    }


def sat_to_las(sat_paths, out_las, absolute: bool = False, epsg: int = DEFAULT_EPSG) -> dict:
    """Write the SegmentAnyTree files ``sat_paths`` as one contract LAS ``out_las``.

    Files are processed in sorted name order; tree ids get per-file offsets so they
    are unique in the output (see the module docstring). Returns ``{"n_points",
    "n_trees", "n_files", "id_offsets": {path: offset}}``.
    """
    import laspy
    import pyproj

    sat_paths = sorted(Path(p) for p in sat_paths)
    if not sat_paths:
        raise ValueError("sat_to_las: no input files")
    if len({str(p) for p in sat_paths}) != len(sat_paths):
        raise ValueError("sat_to_las: an input file is listed more than once")

    parts = [_read_sat(p, absolute) for p in sat_paths]

    id_offsets: dict[str, int] = {}
    running = 0
    for part in parts:
        id_offsets[part["path"]] = running
        running += part["max_id"] + 1

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.floor([
        min(float(p["x"].min()) for p in parts),
        min(float(p["y"].min()) for p in parts),
        min(float(p["z"].min()) for p in parts),
    ])
    # Descriptions are kept under 32 bytes: PotreeConverter reads past an unterminated
    # 32-byte description (see benchmark/potree_convert_tile.py).
    header.add_extra_dim(laspy.ExtraBytesParams(
        name="treeID", type=np.int32, description="SegmentAnyTree tree id, -1 none"))
    header.add_extra_dim(laspy.ExtraBytesParams(
        name="semantic", type=np.uint8, description="0 ground 1 wood 2 leaf 255 n/a"))
    header.add_extra_dim(laspy.ExtraBytesParams(
        name="score", type=np.float32, description="no score in SegmentAnyTree"))
    header.add_crs(pyproj.CRS.from_epsg(int(epsg)))

    out_las = Path(out_las)
    out_las.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_las.with_name(out_las.name + ".tmp")
    n_points = 0
    n_trees = 0
    try:
        with laspy.open(str(tmp), mode="w", header=header) as writer:
            for part in parts:
                n = part["x"].size
                rec = laspy.ScaleAwarePointRecord.zeros(
                    n, point_format=header.point_format, scales=header.scales,
                    offsets=header.offsets)
                rec.x = part["x"]
                rec.y = part["y"]
                rec.z = part["z"]
                rec.classification = part["classification"]
                ids = part["treeID"].copy()
                ids[ids >= 0] += id_offsets[part["path"]]
                rec.treeID = ids
                rec.semantic = part["semantic"]
                rec.score = np.full(n, NO_SCORE, dtype=np.float32)
                writer.write_points(rec)
                n_points += n
                n_trees += int(np.unique(ids[ids >= 0]).size)
        tmp.replace(out_las)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {"n_points": n_points, "n_trees": n_trees, "n_files": len(parts),
            "id_offsets": id_offsets}


def build_products(tile: str, out_dir: Path, runtime_s: float | None, cell_m: float,
                   masks: bool = True) -> dict:
    """Tree table, report and masks next to ``<out_dir>/<tile>.las`` (the ff3d layout)."""
    from ff3d_geo.report import build_report, report_markdown, write_report
    from ff3d_geo.trees import trees_to_gpkg

    las = out_dir / f"{tile}.las"
    gpkg = out_dir / f"{tile}_trees.gpkg"
    n_rows = trees_to_gpkg(las, gpkg)
    report = build_report(las, gpkg, runtime_s=runtime_s)
    report["method"] = "SegmentAnyTree"
    write_report(report, out_dir / f"{tile}_report.json", out_dir / f"{tile}_report.md")
    info = {"trees_gpkg": gpkg, "n_rows": n_rows, "report": report}
    if masks:
        from ff3d_geo.raster import las_to_masks

        info["masks"] = las_to_masks(las, out_dir, cell_m=cell_m, prefix=tile)
    print(report_markdown(report))
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sat-las", nargs="+", required=True, type=Path,
                   help="SegmentAnyTree *_out.laz/.las files (all sub-tiles of one km tile)")
    p.add_argument("--tile", required=True, help="output stem, e.g. 3dm_33_381_5828_1_be")
    p.add_argument("--out", required=True, type=Path, help="output directory (work_dirs/sat-<tile>)")
    p.add_argument("--runtime-s", type=float, default=None, help="wall time of the SegmentAnyTree run")
    p.add_argument("--epsg", type=int, default=DEFAULT_EPSG)
    p.add_argument("--absolute", action="store_true",
                   help="inputs are already in absolute coordinates (no E<x>_N<y> origin in the name)")
    p.add_argument("--cell", type=float, default=0.5, help="mask cell size in metres")
    p.add_argument("--no-masks", action="store_true", help="skip the GeoTIFF masks and crowns")
    a = p.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    out_las = a.out / f"{a.tile}.las"
    info = sat_to_las(a.sat_las, out_las, absolute=a.absolute, epsg=a.epsg)
    print(f"{out_las}: {info['n_points']} points, {info['n_trees']} trees from "
          f"{info['n_files']} SegmentAnyTree file(s)")
    build_products(a.tile, a.out, a.runtime_s, a.cell, masks=not a.no_masks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
