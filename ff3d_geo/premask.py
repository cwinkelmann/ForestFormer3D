"""Pre-inference polygon mask: keep roof and water points away from the model.

The Berlin ALS 2021 classification has no building class (roof points sit in the
vegetation bins 3/4/5; measured on 3dm_33_379_5826: 88.7 % of the points inside ALKIS
building footprints are class 5), so the model segments roofs as trees, and jetties and
moored boats over water survive every post-filter. :func:`ff3d_geo.buildings.mask_buildings`
repairs the result afterwards; this module removes such points BEFORE inference instead:
``las_to_ply`` leaves every point inside a (buffered) polygon out of the PLY the model
sees, and ``results_to_las`` puts them back at their original positions with
``semantic = 3`` (masked), ``treeID = -1`` and ``score = -1``, so the result LAS still holds
every source point in source order.

Polygon sources are GeoPackages, given as ``path`` or ``path:layer``; typical Berlin inputs
are the ALKIS building footprints (``benchmark/fetch_berlin_buildings.py``), the water
polygons of ``alkis:tatsaechlichenutzungflaechen`` (``bezeich`` in AX_Fliessgewaesser,
AX_StehendesGewaesser, AX_Hafenbecken) and the ``alkis:bauwerkeflaechen`` structures
(``bezbwf`` Landebrücke / Überdachung / Carport / Brücke), fetched with the same script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_polygon_spec(spec) -> tuple[Path, str | None]:
    """``path`` or ``path:layer`` -> (path, layer). A ``:`` inside the path itself is
    only taken as a layer separator when the remainder holds no path separator."""
    text = str(spec)
    if ":" in text:
        head, tail = text.rsplit(":", 1)
        if tail and "/" not in tail and "\\" not in tail and not tail.lower().endswith((".gpkg", ".shp")):
            return Path(head), tail
    return Path(text), None


def load_mask_polygons(specs, bbox=None, buffer_m: float = 1.0):
    """All polygons of ``specs`` (clipped to ``bbox``, buffered) as one GeoSeries."""
    import geopandas as gpd
    import pandas as pd

    from ff3d_geo.buildings import load_footprints

    parts = []
    for spec in specs:
        path, layer = parse_polygon_spec(spec)
        if not path.is_file():
            raise FileNotFoundError(f"mask polygons {path} do not exist")
        parts.append(load_footprints(path, bbox=bbox, buffer_m=buffer_m, layer=layer))
    if not parts:
        return gpd.GeoSeries([], dtype="geometry")
    return gpd.GeoSeries(pd.concat(parts, ignore_index=True))


def mask_points(x: np.ndarray, y: np.ndarray, geoms) -> np.ndarray:
    """Boolean array: True where the point lies inside any polygon."""
    from ff3d_geo.buildings import points_in_footprints

    if len(geoms) == 0:
        return np.zeros(x.shape[0], dtype=bool)
    return points_in_footprints(x, y, geoms)
