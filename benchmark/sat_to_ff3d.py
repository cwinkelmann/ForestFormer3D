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
``k >= 1`` becomes ``k - 1``; each SegmentAnyTree file becomes one contract LAS of its
own (``<out>/sub/<stem>.las``, stem = the SegmentAnyTree name without ``_out``), the
tree table is built per sub-tile (``ff3d_geo.trees.trees_to_gpkg`` loops over the
trees of one file, which is seconds for a 100 m sub-tile but hours for a km tile), and
``ff3d_geo.merge.merge_las`` / ``merge_trees`` stitch them into ``<out>/<tile>.las`` and
``<out>/<tile>_trees.gpkg`` with sub-tile ``j`` offset by ``sum(max_id_i + 1 for i < j)``
-- the same path the ForestFormer3D km tiles take, so ids are unique across the km tile
and gaps left by SegmentAnyTree's NMS stay gaps. The per-sub-tile files are removed
afterwards unless ``--keep-subtiles``.

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


def sub_stem(sat_path) -> str:
    """``<name>_out.laz`` -> ``<name>`` (the input stem SegmentAnyTree derived it from)."""
    stem = Path(sat_path).stem
    return stem[:-4] if stem.endswith("_out") else stem


def _contract_header(part: dict, epsg: int):
    import laspy
    import pyproj

    header = laspy.LasHeader(point_format=6, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.floor([float(part["x"].min()), float(part["y"].min()), float(part["z"].min())])
    # Descriptions are kept under 32 bytes: PotreeConverter reads past an unterminated
    # 32-byte description (see benchmark/potree_convert_tile.py).
    header.add_extra_dim(laspy.ExtraBytesParams(
        name="treeID", type=np.int32, description="SegmentAnyTree tree id, -1 none"))
    header.add_extra_dim(laspy.ExtraBytesParams(
        name="semantic", type=np.uint8, description="0 ground 1 wood 2 leaf 255 n/a"))
    header.add_extra_dim(laspy.ExtraBytesParams(
        name="score", type=np.float32, description="no score in SegmentAnyTree"))
    header.add_crs(pyproj.CRS.from_epsg(int(epsg)))
    return header


def sat_file_to_las(sat_path, out_las, absolute: bool = False, epsg: int = DEFAULT_EPSG) -> dict:
    """One SegmentAnyTree file -> one contract LAS (ids ``PredInstance - 1``, no offset)."""
    import laspy

    part = _read_sat(sat_path, absolute)
    header = _contract_header(part, epsg)
    las = laspy.LasData(header)
    las.x, las.y, las.z = part["x"], part["y"], part["z"]
    las.classification = part["classification"]
    las.treeID = part["treeID"]
    las.semantic = part["semantic"]
    las.score = np.full(part["x"].size, NO_SCORE, dtype=np.float32)
    out_las = Path(out_las)
    out_las.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(out_las))
    return {"n_points": int(part["x"].size),
            "n_trees": int(np.unique(part["treeID"][part["treeID"] >= 0]).size)}


def sat_to_las(sat_paths, out_las, absolute: bool = False, epsg: int = DEFAULT_EPSG,
               out_gpkg=None, sub_dir=None, keep_subtiles: bool = False) -> dict:
    """Write the SegmentAnyTree files ``sat_paths`` as one contract LAS ``out_las``
    (and, with ``out_gpkg``, the merged tree table).

    Files are processed in sorted name order. Each becomes ``<sub_dir>/<stem>.las``
    (default ``<out_las dir>/sub/``), gets its tree table when ``out_gpkg`` is set, and
    ``ff3d_geo.merge`` stitches them with per-file id offsets so ids are unique in the
    output (see the module docstring). Returns ``{"n_points", "n_trees", "n_files",
    "id_offsets": {sat path: offset}}``.
    """
    import shutil

    from ff3d_geo.merge import merge_las, merge_trees

    sat_paths = sorted(Path(p) for p in sat_paths)
    if not sat_paths:
        raise ValueError("sat_to_las: no input files")
    if len({str(p) for p in sat_paths}) != len(sat_paths):
        raise ValueError("sat_to_las: an input file is listed more than once")
    out_las = Path(out_las)
    sub_dir = Path(sub_dir) if sub_dir is not None else out_las.parent / "sub"
    sub_dir.mkdir(parents=True, exist_ok=True)

    sub_las, sub_gpkg = [], []
    for sat_path in sat_paths:
        las_path = sub_dir / f"{sub_stem(sat_path)}.las"
        if las_path in sub_las:
            raise ValueError(f"two inputs map to the same sub-tile stem {las_path.stem}")
        sat_file_to_las(sat_path, las_path, absolute=absolute, epsg=epsg)
        sub_las.append(las_path)
        if out_gpkg is not None:
            from ff3d_geo.trees import trees_to_gpkg

            gpkg_path = sub_dir / f"{las_path.stem}_trees.gpkg"
            trees_to_gpkg(las_path, gpkg_path)
            sub_gpkg.append(gpkg_path)

    info = merge_las(sub_las, out_las)
    if out_gpkg is not None:
        info["n_rows"] = merge_trees(sub_gpkg, out_gpkg, info["id_offsets"])
    info["id_offsets"] = {str(s): o for s, o in zip(sat_paths, info["id_offsets"].values())}
    info["n_files"] = len(sat_paths)
    if not keep_subtiles:
        shutil.rmtree(sub_dir)
    return info


def build_products(tile: str, out_dir: Path, runtime_s: float | None, cell_m: float,
                   masks: bool = True) -> dict:
    """Report and masks next to ``<out_dir>/<tile>.las`` + ``_trees.gpkg`` (the ff3d layout)."""
    from ff3d_geo.report import build_report, report_markdown, write_report

    las = out_dir / f"{tile}.las"
    gpkg = out_dir / f"{tile}_trees.gpkg"
    report = build_report(las, gpkg, runtime_s=runtime_s)
    report["method"] = "SegmentAnyTree"
    write_report(report, out_dir / f"{tile}_report.json", out_dir / f"{tile}_report.md")
    info = {"trees_gpkg": gpkg, "report": report}
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
    p.add_argument("--keep-subtiles", action="store_true",
                   help="keep the per-sub-tile LAS/GeoPackage files under <out>/sub/")
    a = p.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    out_las = a.out / f"{a.tile}.las"
    info = sat_to_las(a.sat_las, out_las, absolute=a.absolute, epsg=a.epsg,
                      out_gpkg=a.out / f"{a.tile}_trees.gpkg", keep_subtiles=a.keep_subtiles)
    print(f"{out_las}: {info['n_points']} points, {info['n_trees']} trees from "
          f"{info['n_files']} SegmentAnyTree file(s)", flush=True)
    build_products(a.tile, a.out, a.runtime_s, a.cell, masks=not a.no_masks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
