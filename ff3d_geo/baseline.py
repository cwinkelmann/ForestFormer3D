"""Model-free plausibility reference: tree tops from a CHM built from ALS classes."""

from __future__ import annotations

import laspy
import numpy as np

from ff3d_geo.grid import grid_extent, grid_reduce, nearest_fill

GROUND_CLASSES = (2,)
VEGETATION_CLASSES = (3, 4, 5)


def chm_local_maxima(
    las_path,
    cell_m: float = 1.0,
    window_m: float = 3.0,
    min_height_m: float = 3.0,
) -> list[tuple[float, float, float]]:
    """Local maxima of a canopy height model as ``(x, y, height)`` in LAS coordinates.

    CHM = per-cell max z of ALS classes 3/4/5 minus a DTM (per-cell min z of class 2,
    nearest-filled). Cells without vegetation count as height 0. A cell is a maximum
    when it is strictly higher than every other cell in the ``window_m`` square
    around it and at least ``min_height_m``. Returned x, y are cell centers.
    """
    las = laspy.read(str(las_path))
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    classification = np.asarray(las.classification, dtype=np.int64)

    extent = grid_extent(x, y, cell_m)
    dtm = grid_reduce(extent, x, y, z, "min", mask=np.isin(classification, GROUND_CLASSES))
    if np.isnan(dtm).all():
        dtm[:] = float(z.min())
    else:
        dtm = nearest_fill(dtm)
    dsm = grid_reduce(extent, x, y, z, "max", mask=np.isin(classification, VEGETATION_CLASSES))
    chm = dsm - dtm
    chm[np.isnan(chm)] = 0.0

    radius = max(1, int(round(window_m / cell_m)) // 2)
    padded = np.pad(chm, radius, constant_values=-np.inf)
    shifts = [
        padded[radius + dy : radius + dy + extent.ny, radius + dx : radius + dx + extent.nx]
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if (dy, dx) != (0, 0)
    ]
    neighbour_max = np.max(np.stack(shifts), axis=0)
    is_max = (chm > neighbour_max) & (chm >= min_height_m)

    maxima = []
    for iy, ix in zip(*np.nonzero(is_max)):
        cx, cy = extent.center(int(ix), int(iy))
        maxima.append((cx, cy, float(chm[iy, ix])))
    return maxima
