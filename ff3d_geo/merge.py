"""Stitch per-sub-tile result LAS files and tree GeoPackages back into one km tile.

``merge_las`` concatenates the LAS 1.4 / point format 6 result files that
``ff3d_geo.convert.results_to_las`` writes (extra dims ``treeID`` int32,
``semantic`` uint8, ``score`` float32; CRS written as a WKT VLR), renumbering
``treeID`` so every non-negative id is unique across sub-tiles. ``merge_trees``
applies the same id offsets to the matching ``ff3d_geo.trees.trees_to_gpkg``
GeoPackages (layer ``trees``, columns ``TREE_COLUMNS``) so the merged tree
table's ``tree_id`` set matches the merged LAS's ``treeID >= 0`` set.
"""

from __future__ import annotations

import copy
from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import pandas as pd
import pyproj


def _describe_dim(dim) -> str:
    """``name type "description"`` for one extra dimension, for an error message."""
    return f'{dim.name} {dim.dtype} "{dim.description}"'


def _format_difference(expected, actual) -> str:
    """Say how point format ``actual`` differs from ``expected``, in one clause.

    laspy compares point formats dimension by dimension and a ``DimensionInfo``
    carries its ``description``, so two formats with identical dimension NAMES can
    still compare unequal -- that is exactly the bug the caller guards against.
    Printing the names alone would then show two identical lists, so spell out the
    first dimension that really differs, description included.
    """
    if expected.id != actual.id:
        return f"point format {actual.id} instead of {expected.id}"
    exp_dims = list(expected.extra_dimensions)
    act_dims = list(actual.extra_dimensions)
    if len(exp_dims) != len(act_dims):
        return (f"extra dims {[d.name for d in act_dims]} instead of "
                f"{[d.name for d in exp_dims]}")
    for exp, act in zip(exp_dims, act_dims):
        if exp != act:
            return (f"extra dim {_describe_dim(act)} instead of {_describe_dim(exp)} "
                    "(name, type AND description must match)")
    return "the standard dimensions differ"


def _reject_duplicates(what: str, argument: str, paths: list[Path]) -> None:
    """Raise unless ``paths`` are all distinct, naming the first repeated one.

    ``merge_las`` keys ``id_offsets`` by path string, so the same sub-tile passed
    twice (two overlapping globs, say) collapses to ONE entry: both copies would
    then be written with the same offset and their tree ids would collide, and
    ``merge_trees`` would only notice afterwards, via the length mismatch, with a
    wrong merged LAS already on disk. Checked before anything is read or written.
    """
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            raise ValueError(
                f"{what}: {argument} lists {path} more than once; every sub-tile must "
                "appear exactly once (overlapping shell globs are the usual cause)"
            )
        seen.add(key)


def merge_las(las_paths, out_las) -> dict:
    """Concatenate sub-tile result LAS files into one, with globally unique treeIDs.

    Each input is LAS 1.4 point format 6 with extra dims ``treeID`` (int32,
    -1 = none), ``semantic`` (uint8) and ``score`` (float32), CRS EPSG:25833.
    Sub-tile ``k``'s non-negative ids get ``offset_k = sum(max_id_j + 1 for j
    < k)`` added so ids never collide across sub-tiles; ``treeID == -1`` stays
    -1. The output header uses scale 0.001 and offsets = floor of the global
    minimum x/y/z, matching ``results_to_las``'s convention.

    Returns ``{"n_points": int, "n_trees": int, "id_offsets": {path_str: offset_k}}``.
    """
    las_paths = [Path(p) for p in las_paths]
    if not las_paths:
        raise ValueError("merge_las: las_paths is empty")
    _reject_duplicates("merge_las", "las_paths", las_paths)

    n_points = 0
    max_x = min_x = max_y = min_y = max_z = min_z = None
    max_ids: list[int] = []
    crs = None
    point_format = None
    version = None

    for p in las_paths:
        with laspy.open(str(p)) as reader:
            header = reader.header
            n_points += int(header.point_count)
            xs = (header.mins[0], header.maxs[0])
            ys = (header.mins[1], header.maxs[1])
            zs = (header.mins[2], header.maxs[2])
            min_x = xs[0] if min_x is None else min(min_x, xs[0])
            max_x = xs[1] if max_x is None else max(max_x, xs[1])
            min_y = ys[0] if min_y is None else min(min_y, ys[0])
            max_y = ys[1] if max_y is None else max(max_y, ys[1])
            min_z = zs[0] if min_z is None else min(min_z, zs[0])
            max_z = zs[1] if max_z is None else max(max_z, zs[1])
            if crs is None:
                crs = header.parse_crs()
            # The output point format is COPIED from the first input rather than
            # rebuilt from the extra dimensions' names: laspy compares point formats
            # dimension by dimension when writing, and a DimensionInfo carries its
            # description, so extra dims re-declared without the descriptions that
            # results_to_las gives them ("ForestFormer3D instance, -1 none", ...)
            # compare unequal and write_points fails with "Incompatible point formats".
            if point_format is None:
                point_format = copy.deepcopy(header.point_format)
                version = str(header.version)
            elif header.point_format != point_format:
                raise ValueError(
                    f"merge_las: {p} does not have the same point format as "
                    f"{las_paths[0]}: {_format_difference(point_format, header.point_format)}"
                    "; all sub-tile results must come from the same results_to_las version"
                )

        las = laspy.read(str(p))
        ids = np.asarray(las.treeID, dtype=np.int64)
        positive = ids[ids >= 0]
        max_ids.append(int(positive.max()) if positive.size else -1)

    id_offsets: dict[str, int] = {}
    running = 0
    for p, max_id in zip(las_paths, max_ids):
        id_offsets[str(p)] = running
        running += max_id + 1

    header = laspy.LasHeader(point_format=point_format, version=version)
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.floor([min_x, min_y, min_z])
    if crs is not None:
        header.add_crs(crs)
    else:
        header.add_crs(pyproj.CRS.from_epsg(25833))

    out_las = Path(out_las)
    out_las.parent.mkdir(parents=True, exist_ok=True)

    # Written to a temporary file and renamed only once the last sub-tile is in and
    # the header is finalised (like split_las), so a kill mid-merge cannot leave a
    # truncated <out_las> that looks like a complete km tile.
    tmp_las = out_las.with_name(out_las.name + ".tmp")
    all_ids: list[np.ndarray] = []
    try:
        with laspy.open(str(tmp_las), mode="w", header=header) as writer:
            for p in las_paths:
                las = laspy.read(str(p))
                offset = id_offsets[str(p)]
                ids = np.asarray(las.treeID, dtype=np.int32).copy()
                ids[ids >= 0] += offset
                las.treeID = ids
                writer.write_points(las.points)
                all_ids.append(ids)
        tmp_las.replace(out_las)
    except BaseException:
        tmp_las.unlink(missing_ok=True)
        raise

    merged_ids = np.concatenate(all_ids) if all_ids else np.empty(0, dtype=np.int32)
    n_trees = int(np.unique(merged_ids[merged_ids >= 0]).size)

    return {"n_points": n_points, "n_trees": n_trees, "id_offsets": id_offsets}


def merge_trees(gpkg_paths, out_gpkg, id_offsets: dict) -> int:
    """Concatenate sub-tile tree GeoPackages, offsetting ``tree_id`` by ``id_offsets``.

    Reads layer ``trees`` from each GeoPackage (pyogrio engine), adds the
    matching sub-tile's offset to ``tree_id``, concatenates in input order and
    writes layer ``trees`` to ``out_gpkg`` with the same columns and CRS.
    Returns the merged row count.

    ``id_offsets`` is ``merge_las``'s return value keyed by the *LAS* paths
    (in sub-tile order); ``gpkg_paths`` is the matching list of GeoPackage
    paths for the same sub-tiles in the same order, so offsets are applied
    positionally rather than by matching path strings.
    """
    gpkg_paths = [Path(p) for p in gpkg_paths]
    _reject_duplicates("merge_trees", "gpkg_paths", gpkg_paths)
    offsets = list(id_offsets.values())
    if len(offsets) != len(gpkg_paths):
        raise ValueError(
            f"merge_trees: {len(gpkg_paths)} gpkg_paths but {len(offsets)} id_offsets"
        )

    frames = []
    crs = None
    for p, offset in zip(gpkg_paths, offsets):
        gdf = gpd.read_file(str(p), layer="trees", engine="pyogrio")
        gdf = gdf.copy()
        gdf["tree_id"] = gdf["tree_id"].astype(np.int64) + offset
        if crs is None:
            crs = gdf.crs
        frames.append(gdf)

    merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=crs)

    out_gpkg = Path(out_gpkg)
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)
    merged.to_file(str(out_gpkg), layer="trees", driver="GPKG", engine="pyogrio")
    return len(merged)
