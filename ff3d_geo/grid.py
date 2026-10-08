"""Small 2D raster helpers shared by trees.py and baseline.py (numpy only)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridExtent:
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

    def contains(self, x, y) -> np.ndarray:
        """True where ``(x, y)`` really falls inside the grid.

        :meth:`index` clamps out-of-grid coordinates to the edge cell silently, which
        is what a raster pass wants but not what a caller asking "is this point in
        this tile?" wants (``ff3d_geo.stitch`` picks the ground grid of the tile that
        holds a cross-border stem). Half-open on the upper edge, like ``index``.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return ((x >= self.x0) & (x < self.x0 + self.nx * self.cell)
                & (y >= self.y0) & (y < self.y0 + self.ny * self.cell))

    def center(self, ix: int, iy: int) -> tuple[float, float]:
        return self.x0 + (ix + 0.5) * self.cell, self.y0 + (iy + 0.5) * self.cell


def grid_extent(x, y, cell: float) -> GridExtent:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        raise ValueError("cannot build a grid over zero points")
    x0 = float(np.floor(x.min() / cell) * cell)
    y0 = float(np.floor(y.min() / cell) * cell)
    nx = int(np.floor((x.max() - x0) / cell)) + 1
    ny = int(np.floor((y.max() - y0) / cell)) + 1
    return GridExtent(x0, y0, float(cell), nx, ny)


def grid_reduce(extent: GridExtent, x, y, values, how: str, mask=None) -> np.ndarray:
    """Per-cell min or max of ``values`` (shape (ny, nx), NaN where empty)."""
    x = np.asarray(x)
    y = np.asarray(y)
    values = np.asarray(values, dtype=np.float64)
    if mask is not None:
        x, y, values = x[mask], y[mask], values[mask]
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


def nearest_fill(grid: np.ndarray) -> np.ndarray:
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
            padded[1 + dy: 1 + dy + ny, 1 + dx: 1 + dx + nx]
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
