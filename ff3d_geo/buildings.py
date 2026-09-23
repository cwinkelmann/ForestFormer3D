"""Mask ALKIS building footprints out of a result LAS.

Why: the Berlin ALS 2021 point clouds have **no building class**.  Measured on three
km tiles the classes present are 2 (ground), 3, 4, 5, 7 and 32 -- class 6 (building)
is absent and roof points sit in the vegetation classes 3/4/5.  ForestFormer3D was
trained on forest plots, so it takes a roof for a crown and predicts tree instances
on buildings (clearly visible in the allotment area of ``3dm_33_381_5829_1_be``).

The fix is external: Berlin's official ALKIS building footprints, fetched by
``benchmark/fetch_berlin_buildings.py`` from
``https://gdi.berlin.de/services/wfs/alkis_gebaeude``.  Every predicted point inside
a footprint (buffered by ``buffer_m``, because a roof edge overhangs its ground plan
and the ALS point can sit a metre outside) is relabelled ``semantic = 3`` (building)
and loses its ``treeID``; an instance whose points lie inside footprints by at least
``min_roof_fraction`` is dropped entirely.

Instance ids are NOT renumbered: a kept tree keeps the id it had in the unmasked
result, so the masked and unmasked tree tables stay comparable row by row.
"""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np

# The result LAS ``semantic`` extra dim is 0 ground / 1 wood / 2 leaf / 255 nodata
# (see ff3d_geo.convert).  3 is free, and unlike 255 it is a real class, not nodata.
SEMANTIC_BUILDING = 3
BUILDINGS_LAYER = "buildings"
# Points are tested against the footprint index in chunks: shapely.points() on a
# whole 25 M point km tile would allocate ~2.5 GB of geometry objects at once.
CHUNK = 1_000_000


def load_footprints(buildings_gpkg, bbox=None, buffer_m: float = 1.0, layer: str | None = None):
    """Footprints from a GeoPackage, clipped to ``bbox`` and buffered by ``buffer_m``.

    ``bbox`` is ``(minx, miny, maxx, maxy)`` in the LAS CRS; it is grown by
    ``buffer_m`` before the spatial filter so a building just outside the tile whose
    buffer reaches in is still loaded.  Returns a GeoSeries (possibly empty).
    """
    import geopandas as gpd

    path = Path(buildings_gpkg)
    layers = [layer] if layer else [BUILDINGS_LAYER, None]
    frame = None
    last: Exception | None = None
    for name in layers:
        try:
            frame = gpd.read_file(str(path), layer=name) if name else gpd.read_file(str(path))
            break
        except Exception as exc:  # pragma: no cover - depends on the GeoPackage
            last = exc
    if frame is None:
        raise RuntimeError(f"{path}: no readable building layer ({last})")

    geoms = frame.geometry
    if bbox is not None:
        pad = max(buffer_m, 0.0)
        minx, miny, maxx, maxy = bbox
        geoms = geoms.cx[minx - pad:maxx + pad, miny - pad:maxy + pad]
    geoms = geoms[~geoms.is_empty & geoms.notna()]
    if buffer_m:
        geoms = geoms.buffer(buffer_m)
    return geoms.reset_index(drop=True)


def points_in_footprints(x: np.ndarray, y: np.ndarray, geoms) -> np.ndarray:
    """Boolean mask: point (x, y) lies in any of ``geoms``.

    Vectorised via a shapely 2 STRtree over the footprints -- the tree does the bbox
    filtering and the exact point-in-polygon test in C, so a km tile costs seconds
    rather than a Python loop over 25 M points.
    """
    import shapely
    from shapely import STRtree

    inside = np.zeros(x.size, dtype=bool)
    geom_array = np.asarray(geoms.values if hasattr(geoms, "values") else geoms, dtype=object)
    if geom_array.size == 0 or x.size == 0:
        return inside
    tree = STRtree(geom_array)
    for start in range(0, x.size, CHUNK):
        stop = min(start + CHUNK, x.size)
        pts = shapely.points(x[start:stop], y[start:stop])
        hit = tree.query(pts, predicate="intersects")
        if hit.size:
            inside[start + np.unique(hit[0])] = True
    return inside


def mask_buildings(las_path, buildings_gpkg, out_las, buffer_m: float = 1.0,
                   min_roof_fraction: float = 0.5, layer: str | None = None) -> dict:
    """Write ``out_las``: ``las_path`` with building points and roof instances removed.

    Points inside a buffered footprint get ``semantic = 3`` (building) and
    ``treeID = -1``; their ALS ``classification`` is left untouched, so the original
    ALS class of a roof point stays inspectable.  An instance with at least
    ``min_roof_fraction`` of its points inside footprints is removed completely (its
    points outside the footprints also get ``treeID = -1``, but keep their semantic).
    Surviving instances keep their ids -- nothing is renumbered.

    Returns ``{"n_points", "n_footprints", "points_masked", "instances_before",
    "instances_removed", "instances_partially_masked", "instances_after",
    "points_unassigned"}``.
    """
    las_path, out_las = Path(las_path), Path(out_las)
    las = laspy.read(str(las_path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    tree_id = np.array(las.treeID, dtype=np.int64)
    semantic = np.array(las.semantic, dtype=np.int64)

    bbox = (float(x.min()), float(y.min()), float(x.max()), float(y.max())) if x.size else None
    geoms = load_footprints(buildings_gpkg, bbox=bbox, buffer_m=buffer_m, layer=layer)
    inside = points_in_footprints(x, y, geoms)

    # Per-instance point counts via bincount rather than a loop of `tree_id == tid`
    # masks: a km tile has ~20 k instances and 25 M points, so the loop would be
    # 5e11 comparisons (measured: it did not finish in 5 minutes).
    has_tree = tree_id >= 0
    ids_before = np.unique(tree_id[has_tree])
    if ids_before.size:
        length = int(ids_before.max()) + 1
        total = np.bincount(tree_id[has_tree], minlength=length)
        n_in = np.bincount(tree_id[has_tree & inside], minlength=length)
        roof = (n_in > 0) & (n_in >= min_roof_fraction * total)
        removed = np.flatnonzero(roof)
        partial = int(np.count_nonzero((n_in > 0) & ~roof))
    else:
        removed = np.empty(0, dtype=np.int64)
        partial = 0

    drop = inside.copy()
    if removed.size:
        # `roof` is indexed by tree id, so a lookup table beats np.isin here.
        is_roof_id = np.zeros(length, dtype=bool)
        is_roof_id[removed] = True
        drop |= has_tree & is_roof_id[np.where(has_tree, tree_id, 0)]
    tree_id[drop] = -1
    semantic[inside] = SEMANTIC_BUILDING

    las.treeID = tree_id.astype(np.int32)
    las.semantic = semantic.astype(np.uint8)
    las.score = np.where(tree_id >= 0, np.asarray(las.score, dtype=np.float32), 0.0).astype(
        np.float32
    )
    out_las.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(out_las))

    return {
        "las": str(out_las),
        "n_points": int(x.size),
        "n_footprints": int(len(geoms)),
        "points_masked": int(np.count_nonzero(inside)),
        "instances_before": int(ids_before.size),
        "instances_removed": int(removed.size),
        "instances_partially_masked": partial,
        "instances_after": int(np.unique(tree_id[tree_id >= 0]).size),
        "points_unassigned": int(np.count_nonzero(drop)),
        "buffer_m": float(buffer_m),
        "min_roof_fraction": float(min_roof_fraction),
    }
