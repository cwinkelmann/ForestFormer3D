"""Model-free plausibility reference: tree tops from a CHM built from ALS classes."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import laspy
import numpy as np

GROUND_CLASSES = (2,)
VEGETATION_CLASSES = (3, 4, 5)


# --- Local copies of ff3d_geo.grid helpers (Task 4) -------------------------
# ff3d_geo.grid (GridExtent/grid_extent/grid_reduce/nearest_fill) does not
# exist yet as of this module's authoring. The private helpers below mirror
# the spec in .superpowers/sdd/2026-09-22-ff3d-03-tegel-geo/task-4-brief.md
# exactly, so this module can be switched to
# ``from ff3d_geo.grid import grid_extent, grid_reduce, nearest_fill`` with
# no behaviour change once Task 4 lands. Do not import these private names
# from outside this module.


@dataclass(frozen=True)
class _GridExtent:
    """Axis-aligned cell grid covering a point set. Cell (ix, iy) spans
    ``[x0 + ix*cell, x0 + (ix+1)*cell)`` and likewise in y."""

    x0: float
    y0: float
    cell: float
    nx: int
    ny: int

    def index(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Column/row indices for coordinates, clamped into the grid."""
        ix = np.clip(np.floor((np.asarray(x) - self.x0) / self.cell).astype(int), 0, self.nx - 1)
        iy = np.clip(np.floor((np.asarray(y) - self.y0) / self.cell).astype(int), 0, self.ny - 1)
        return ix, iy

    def center(self, ix: int, iy: int) -> tuple[float, float]:
        return self.x0 + (ix + 0.5) * self.cell, self.y0 + (iy + 0.5) * self.cell


def _grid_extent(x, y, cell: float) -> _GridExtent:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        raise ValueError("cannot build a grid over zero points")
    x0 = float(np.floor(x.min() / cell) * cell)
    y0 = float(np.floor(y.min() / cell) * cell)
    nx = int(np.floor((x.max() - x0) / cell)) + 1
    ny = int(np.floor((y.max() - y0) / cell)) + 1
    return _GridExtent(x0, y0, float(cell), nx, ny)


def _grid_reduce(extent: _GridExtent, x, y, values, how: str, mask=None) -> np.ndarray:
    """Per-cell min or max of ``values`` (shape (ny, nx), NaN where empty)."""
    values = np.asarray(values, dtype=np.float64)
    if mask is not None:
        x, y, values = np.asarray(x)[mask], np.asarray(y)[mask], values[mask]
    if how == "min":
        out = np.full(extent.ny * extent.nx, np.inf)
        reduce_at = np.minimum.at
    elif how == "max":
        out = np.full(extent.ny * extent.nx, -np.inf)
        reduce_at = np.maximum.at
    else:
        raise ValueError(f"how must be 'min' or 'max', got {how!r}")
    if values.size:
        ix, iy = extent.index(x, y)
        reduce_at(out, iy * extent.nx + ix, values)
    out[~np.isfinite(out)] = np.nan
    return out.reshape(extent.ny, extent.nx)


def _nearest_fill(grid: np.ndarray) -> np.ndarray:
    """Fill NaN cells from their nearest filled cells.

    Each pass fills every empty cell that touches a filled cell (8-neighbourhood)
    with the mean of those neighbours, so the search radius grows by one cell per
    pass until nothing is empty. Raises ``ValueError`` if no cell is filled.
    """
    g = np.array(grid, dtype=np.float64, copy=True)
    if np.isnan(g).all():
        raise ValueError("grid has no filled cells to propagate")
    ny, nx = g.shape
    while np.isnan(g).any():
        padded = np.pad(g, 1, constant_values=np.nan)
        shifts = [
            padded[1 + dy : 1 + dy + ny, 1 + dx : 1 + dx + nx]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dy, dx) != (0, 0)
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            neighbour_mean = np.nanmean(np.stack(shifts), axis=0)
        empty = np.isnan(g)
        g[empty] = neighbour_mean[empty]
    return g


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

    extent = _grid_extent(x, y, cell_m)
    dtm = _grid_reduce(extent, x, y, z, "min", mask=np.isin(classification, GROUND_CLASSES))
    if np.isnan(dtm).all():
        dtm[:] = float(z.min())
    else:
        dtm = _nearest_fill(dtm)
    dsm = _grid_reduce(extent, x, y, z, "max", mask=np.isin(classification, VEGETATION_CLASSES))
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
