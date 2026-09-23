"""Raster segmentation masks and crown polygons from a georeferenced result LAS.

``las_to_masks`` turns one ``ff3d_geo.convert.results_to_las`` output (LAS 1.4 /
point format 6 with the ``treeID`` / ``semantic`` / ``score`` extra dims) into

* ``<prefix>_instance_<cell>.tif`` - int32, the ``treeID`` of the highest point
  in each cell, ``-1`` where the cell holds no tree point,
* ``<prefix>_semantic_<cell>.tif`` - uint8, the majority semantic class per cell,
  ``255`` where the cell holds no voted point,
* ``<prefix>_crowns.gpkg`` (layer ``crowns``) - one convex-hull polygon per tree.

The tree ids are the same ids as the ``tree_id`` column of ``ff3d_geo.trees``'
GeoPackage, so a mask and a tree table of the same LAS join on them directly. The
crown layer holds every tree, but the instance raster's id set is only a SUBSET of
it: a tree every one of whose cells is topped by a taller neighbour owns no cell at
all (about 4 % of the trees on the Berlin km tiles at 0.5 m - understorey). Use the
crowns, not the raster, when completeness matters.

Everything is binned with vectorised numpy (one ``np.lexsort`` and one
``np.bincount``); there is no Python loop over points anywhere, so a 25 M point
Berlin km tile at 0.5 m (2000 x 2000 cells) stays inside a few GB. The only
per-object Python loop is over *trees* (tens of thousands), for the hulls.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import laspy
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import MultiPoint, box

#: Fallback CRS when the LAS carries none. ETRS89 / UTM zone 33N, the Berlin ALS CRS.
DEFAULT_EPSG = 25833

#: Written where a cell has no ``semantic`` vote (and the sentinel ``results_to_las``
#: already uses for points the model left unlabelled).
SEMANTIC_NODATA = 255

#: Written where a cell holds no point with ``treeID >= 0``. ``0`` is a valid tree id,
#: so the nodata value has to sit outside the id range rather than being 0.
INSTANCE_NODATA = -1

#: Semantic classes that take part in the majority vote: 0 ground, 1 wood, 2 leaf.
N_SEMANTIC_CLASSES = 3

CROWN_COLUMNS = ["tree_id", "n_points", "top_z", "crown_area_m2", "hull_is_point"]


def _cell_tag(cell_m: float) -> str:
    """File-name tag for a cell size: ``0.5 -> '50cm'``, ``1.0 -> '1m'``, ``2.5 -> '2.5m'``.

    A literal ``0.5m`` in a file name is awkward to glob and sorts badly, so sub-metre
    cells are named in whole centimetres instead.
    """
    if cell_m <= 0:
        raise ValueError(f"cell_m must be positive, got {cell_m}")
    if cell_m < 1:
        return f"{int(round(cell_m * 100))}cm"
    return f"{cell_m:g}m"


def _raster_extent(header, cell_m: float) -> tuple[float, float, float, float, int, int]:
    """LAS header mins/maxs snapped OUTWARD to multiples of ``cell_m``.

    Returns ``(x0, y1, x1, y0, rows, cols)`` where ``(x0, y1)`` is the north-west
    corner, i.e. the origin of a north-up transform. Snapping outward (floor on the
    minimum, ceil on the maximum) keeps the grid aligned to a global ``cell_m``
    lattice, so masks of neighbouring tiles line up cell for cell.
    """
    x0 = float(np.floor(header.mins[0] / cell_m) * cell_m)
    y0 = float(np.floor(header.mins[1] / cell_m) * cell_m)
    x1 = float(np.ceil(header.maxs[0] / cell_m) * cell_m)
    y1 = float(np.ceil(header.maxs[1] / cell_m) * cell_m)
    cols = max(int(round((x1 - x0) / cell_m)), 1)
    rows = max(int(round((y1 - y0) / cell_m)), 1)
    # A tile whose extent is an exact multiple of cell_m in one axis snaps to itself;
    # keep x1/y1 consistent with the (possibly bumped to 1) cell counts.
    return x0, y1, x0 + cols * cell_m, y1 - rows * cell_m, rows, cols


def _cell_index(x, y, x0: float, y1: float, cell_m: float, rows: int, cols: int):
    """Flat (row-major, row 0 = northernmost) cell index per point, clamped in range.

    ``astype(int64)`` truncates toward zero rather than flooring, which is the same
    thing here: ``x0``/``y1`` come from the snapped extent, so both offsets are >= 0.
    """
    col = np.clip(((x - x0) / cell_m).astype(np.int64), 0, cols - 1)
    row = np.clip(((y1 - y) / cell_m).astype(np.int64), 0, rows - 1)
    return row * cols + col


def _highest_per_cell(flat, z, values, n_cells: int, nodata):
    """For each cell, ``values`` of the point with the largest ``z``; ``nodata`` elsewhere.

    ``np.lexsort`` orders the points by cell and, within a cell, by z, so the last
    point of each run is that cell's highest. Ties in z resolve to whichever point
    the (stable) sort left last, which for identical z is an arbitrary but
    deterministic choice.
    """
    out = np.full(n_cells, nodata, dtype=values.dtype)
    if flat.size == 0:
        return out
    order = np.lexsort((z, flat))
    cells = flat[order]
    last = np.flatnonzero(np.append(np.diff(cells) != 0, True))
    out[cells[last]] = values[order[last]]
    return out


def _majority_semantic(flat, semantic, n_cells: int) -> np.ndarray:
    """Per-cell majority of ``semantic`` over classes 0/1/2; ``255`` where nothing voted.

    Ties are broken toward the HIGHER class index (leaf over wood over ground): the
    argmax runs over the class axis reversed, so ``np.argmax``'s first-maximum rule
    picks the largest class index among the tied ones. Rationale: a cell that a tree
    reaches at all is more useful shown as canopy than as the ground below it.
    """
    voted = semantic < N_SEMANTIC_CLASSES
    counts = np.bincount(
        flat[voted] * N_SEMANTIC_CLASSES + semantic[voted].astype(np.int64),
        minlength=n_cells * N_SEMANTIC_CLASSES,
    ).reshape(n_cells, N_SEMANTIC_CLASSES)
    out = (N_SEMANTIC_CLASSES - 1 - np.argmax(counts[:, ::-1], axis=1)).astype(np.uint8)
    out[counts.sum(axis=1) == 0] = SEMANTIC_NODATA
    return out


def _write_tif(path: Path, data: np.ndarray, transform, crs, nodata) -> Path:
    """One-band, north-up, LZW-compressed, internally tiled GeoTIFF."""
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
        tiled=True,
    ) as dst:
        dst.write(data, 1)
    return path


def _crowns(x, y, z, tree_id, cell_m: float, crs) -> gpd.GeoDataFrame:
    """One convex-hull polygon per ``treeID >= 0``.

    A tree with fewer than 3 distinct XY points - or whose points are all collinear,
    which makes the convex hull a LineString rather than a Polygon - gets a
    ``cell_m``-sided square around its XY centroid instead, flagged by
    ``hull_is_point``. ``crown_area_m2`` is always the written geometry's area, so
    for those degenerate trees it is ``cell_m ** 2`` rather than 0.
    """
    keep = tree_id >= 0
    tid, tx, ty, tz = tree_id[keep], x[keep], y[keep], z[keep]
    order = np.argsort(tid, kind="stable")
    tid, tx, ty, tz = tid[order], tx[order], ty[order], tz[order]
    starts = np.flatnonzero(np.append(True, np.diff(tid) != 0))
    ends = np.append(starts[1:], tid.size) if starts.size else starts

    rows: list[dict] = []
    geoms = []
    for start, end in zip(starts, ends):
        gx, gy = tx[start:end], ty[start:end]
        hull = MultiPoint(np.column_stack([gx, gy])).convex_hull
        degenerate = hull.geom_type != "Polygon"
        if degenerate:
            cx, cy = float(gx.mean()), float(gy.mean())
            half = cell_m / 2.0
            hull = box(cx - half, cy - half, cx + half, cy + half)
        rows.append(
            {
                "tree_id": int(tid[start]),
                "n_points": int(end - start),
                "top_z": float(tz[start:end].max()),
                "crown_area_m2": float(hull.area),
                "hull_is_point": bool(degenerate),
            }
        )
        geoms.append(hull)

    frame = {col: [row[col] for row in rows] for col in CROWN_COLUMNS}
    gdf = gpd.GeoDataFrame(frame, geometry=gpd.GeoSeries(geoms, crs=crs), crs=crs)
    if not rows:
        gdf = gdf.astype(
            {
                "tree_id": "int64",
                "n_points": "int64",
                "top_z": "float64",
                "crown_area_m2": "float64",
                "hull_is_point": "bool",
            }
        )
    return gdf


def las_to_masks(las_path, out_dir, cell_m: float = 0.5, prefix: str | None = None) -> dict:
    """Write instance/semantic mask GeoTIFFs and a crown-polygon GeoPackage.

    ``las_path`` is a result LAS from ``ff3d_geo.convert.results_to_las`` (or from
    ``ff3d_geo.merge.merge_las``); it is read exactly once. ``prefix`` defaults to
    the LAS stem, and the cell size appears in the raster file names via
    ``_cell_tag`` (``0.5 -> 50cm``).

    Returns ``{"instance", "semantic", "crowns"}`` paths plus ``n_trees``,
    ``cell_m`` and ``shape`` (rows, cols).
    """
    las_path = Path(las_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix if prefix is not None else las_path.stem
    tag = _cell_tag(cell_m)

    las = laspy.read(str(las_path))
    las_crs = las.header.parse_crs()
    epsg = las_crs.to_epsg() if las_crs is not None else None
    raster_crs = CRS.from_epsg(epsg or DEFAULT_EPSG)
    vector_crs = las_crs if las_crs is not None else f"EPSG:{DEFAULT_EPSG}"

    x0, y1, _, _, rows, cols = _raster_extent(las.header, cell_m)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    tree_id = np.asarray(las.treeID, dtype=np.int32)
    semantic = np.asarray(las.semantic, dtype=np.uint8)
    del las  # the point record is the biggest object in the room; the arrays are copies

    flat = _cell_index(x, y, x0, y1, cell_m, rows, cols)
    n_cells = rows * cols

    has_tree = tree_id >= 0
    instance = _highest_per_cell(
        flat[has_tree], z[has_tree], tree_id[has_tree], n_cells, INSTANCE_NODATA
    ).reshape(rows, cols)
    semantic_grid = _majority_semantic(flat, semantic, n_cells).reshape(rows, cols)
    del flat, semantic

    transform = from_origin(x0, y1, cell_m, cell_m)
    instance_tif = _write_tif(
        out_dir / f"{prefix}_instance_{tag}.tif", instance, transform, raster_crs,
        INSTANCE_NODATA,
    )
    semantic_tif = _write_tif(
        out_dir / f"{prefix}_semantic_{tag}.tif", semantic_grid, transform, raster_crs,
        SEMANTIC_NODATA,
    )
    del instance, semantic_grid

    crowns = _crowns(x, y, z, tree_id, cell_m, vector_crs)
    crowns_gpkg = out_dir / f"{prefix}_crowns.gpkg"
    crowns.to_file(str(crowns_gpkg), driver="GPKG", layer="crowns", engine="pyogrio")

    return {
        "instance": instance_tif,
        "semantic": semantic_tif,
        "crowns": crowns_gpkg,
        "n_trees": len(crowns),
        "cell_m": float(cell_m),
        "shape": (rows, cols),
    }
