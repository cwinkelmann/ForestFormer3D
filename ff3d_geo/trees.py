"""Per-tree attributes from a georeferenced result LAS into a GeoPackage."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
from shapely.geometry import MultiPoint, Point

from ff3d_geo.grid import GridExtent, grid_extent, grid_reduce, nearest_fill

TREE_COLUMNS = ["tree_id", "x", "y", "top_z", "height", "crown_area_m2", "n_points", "mean_score"]


def ground_surface(x, y, z, semantic, classification, cell: float) -> tuple[np.ndarray, GridExtent]:
    """Ground z per cell: min z of model-ground (semantic==0) points; cells without
    any fall back to min z of ALS class-2 points; still-empty cells are filled from
    their nearest filled cell; if nothing is filled at all, the tile's min z.

    ``semantic == 255`` (nodata) points are never treated as ground: they simply
    fail the ``semantic == 0`` mask like any other non-ground semantic value.
    """
    extent = grid_extent(x, y, cell)
    ground = grid_reduce(extent, x, y, z, "min", mask=np.asarray(semantic) == 0)
    als = grid_reduce(extent, x, y, z, "min", mask=np.asarray(classification) == 2)
    empty = np.isnan(ground)
    ground[empty] = als[empty]
    if np.isnan(ground).all():
        ground[:] = float(np.min(z))
    else:
        ground = nearest_fill(ground)
    return ground, extent


def trees_to_gpkg(las_path, gpkg_path, ground_grid_m: float = 1.0) -> int:
    """Write one point row per ``treeID >= 0`` into layer ``trees``; return the row count.

    Columns: tree_id, x, y (stem = median of the tree's lowest 1 m of points),
    top_z, height (top_z minus ground z at the stem cell), crown_area_m2
    (2D convex hull; 0.0 when fewer than 3 distinct points), n_points, mean_score.
    CRS = the LAS CRS.
    """
    las = laspy.read(str(las_path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    tree_id = np.asarray(las.treeID, dtype=np.int64)
    semantic = np.asarray(las.semantic, dtype=np.int64)
    classification = np.asarray(las.classification, dtype=np.int64)
    score = np.asarray(las.score, dtype=np.float64)
    crs = las.header.parse_crs()

    ground, extent = ground_surface(x, y, z, semantic, classification, ground_grid_m)

    # Group the points by tree with ONE sort rather than a `tree_id == tid` pass per
    # tree: a km tile has ~31 k instances and ~23 M points, so the naive loop costs
    # ~31 k x 23 M comparisons (measured: 14 minutes of CPU for a single km tile).
    order = np.argsort(tree_id, kind="stable")
    sorted_ids = tree_id[order]
    first_real = int(np.searchsorted(sorted_ids, 0, side="left"))
    ids, starts = np.unique(sorted_ids[first_real:], return_index=True)
    starts = starts + first_real
    stops = np.append(starts[1:], sorted_ids.size)

    rows: list[dict] = []
    geoms: list[Point] = []
    for tid, start, stop in zip(ids, starts, stops):
        member = order[start:stop]
        tx, ty, tz = x[member], y[member], z[member]
        low = tz <= tz.min() + 1.0
        stem_x = float(np.median(tx[low]))
        stem_y = float(np.median(ty[low]))
        top_z = float(tz.max())
        ix, iy = extent.index(stem_x, stem_y)
        ground_z = float(ground[iy, ix])
        n_distinct = len(set(zip(tx.tolist(), ty.tolist())))
        if n_distinct >= 3:
            hull = MultiPoint(np.column_stack([tx, ty])).convex_hull
            crown_area = float(hull.area) if hull.geom_type == "Polygon" else 0.0
        else:
            crown_area = 0.0
        rows.append(
            {
                "tree_id": int(tid),
                "x": stem_x,
                "y": stem_y,
                "top_z": top_z,
                "height": top_z - ground_z,
                "crown_area_m2": crown_area,
                "n_points": int(member.size),
                "mean_score": float(score[member].mean()),
            }
        )
        geoms.append(Point(stem_x, stem_y))

    frame = {col: [row[col] for row in rows] for col in TREE_COLUMNS}
    gdf = gpd.GeoDataFrame(frame, geometry=gpd.GeoSeries(geoms, crs=crs), crs=crs)
    if not rows:
        gdf = gdf.astype(
            {
                "tree_id": "int64",
                "x": "float64",
                "y": "float64",
                "top_z": "float64",
                "height": "float64",
                "crown_area_m2": "float64",
                "n_points": "int64",
                "mean_score": "float64",
            }
        )
    gpkg_path = Path(gpkg_path)
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(gpkg_path), driver="GPKG", layer="trees")
    return len(rows)
