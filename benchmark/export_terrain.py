#!/usr/bin/env python3
"""Export terrain rasters from a LAS tile: DTM, DSM and CHM as GeoTIFFs.

    python3 benchmark/export_terrain.py --las <tile>.las --out <dir> [--dtm-cell 1.0] [--chm-cell 0.5]

Writes ``<stem>_dtm_<c>.tif`` (min z of ground points, class 2, nearest-filled),
``<stem>_dsm_<c>.tif`` (max z of all points except noise, class 7) and
``<stem>_chm_<c>.tif`` (DSM minus the DTM sampled at the DSM cells, clipped at 0),
all float32, nodata -9999, EPSG:25833 (override with --epsg), north-up, LZW.
The same grid helpers as ``ff3d_geo.baseline`` are used, so the CHM here is the
one the reports' local-maxima baseline is computed from.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ff3d_geo.grid import grid_extent, grid_reduce, nearest_fill  # noqa: E402

NODATA = -9999.0
GROUND = (2,)
NOISE = (7,)


def _tag(cell: float) -> str:
    return f"{int(round(cell * 100))}cm" if cell < 1 else f"{cell:g}m"


def _write(path: Path, grid: np.ndarray, extent, epsg: int) -> None:
    import rasterio
    from rasterio.transform import from_origin

    data = np.flipud(grid).astype(np.float32)  # row 0 of grid_reduce is the south edge
    data = np.where(np.isfinite(data), data, NODATA).astype(np.float32)
    transform = from_origin(extent.x0, extent.y0 + extent.ny * extent.cell, extent.cell, extent.cell)
    with rasterio.open(
        path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1,
        dtype="float32", crs=f"EPSG:{epsg}", transform=transform, nodata=NODATA,
        compress="lzw", tiled=True,
    ) as dst:
        dst.write(data, 1)


def export_terrain(las_path: Path, out_dir: Path, dtm_cell: float = 1.0,
                   chm_cell: float = 0.5, epsg: int = 25833) -> dict:
    import laspy

    las = laspy.read(str(las_path))
    x = np.asarray(las.x); y = np.asarray(las.y); z = np.asarray(las.z)
    cls = np.asarray(las.classification)
    del las

    ext_dtm = grid_extent(x, y, dtm_cell)
    dtm = grid_reduce(ext_dtm, x, y, z, "min", mask=np.isin(cls, GROUND))
    dtm = nearest_fill(dtm)

    ext_chm = grid_extent(x, y, chm_cell)
    dsm = grid_reduce(ext_chm, x, y, z, "max", mask=~np.isin(cls, NOISE))

    # sample the DTM at every DSM cell centre
    cy, cx = np.mgrid[0:ext_chm.ny, 0:ext_chm.nx]
    px = ext_chm.x0 + (cx + 0.5) * ext_chm.cell
    py = ext_chm.y0 + (cy + 0.5) * ext_chm.cell
    ix, iy = ext_dtm.index(px.ravel(), py.ravel())
    ground = dtm[iy, ix].reshape(ext_chm.ny, ext_chm.nx)
    chm = np.where(np.isfinite(dsm), np.maximum(dsm - ground, 0.0), np.nan)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = las_path.stem
    paths = {
        "dtm": out_dir / f"{stem}_dtm_{_tag(dtm_cell)}.tif",
        "dsm": out_dir / f"{stem}_dsm_{_tag(chm_cell)}.tif",
        "chm": out_dir / f"{stem}_chm_{_tag(chm_cell)}.tif",
    }
    _write(paths["dtm"], dtm, ext_dtm, epsg)
    _write(paths["dsm"], dsm, ext_chm, epsg)
    _write(paths["chm"], chm, ext_chm, epsg)
    valid = np.isfinite(chm)
    return {
        "paths": {k: str(v) for k, v in paths.items()},
        "n_points": int(x.size),
        "dtm_range": [float(np.nanmin(dtm)), float(np.nanmax(dtm))],
        "chm_max": float(np.nanmax(chm)) if valid.any() else None,
        "chm_median_vegetated": float(np.median(chm[valid & (chm >= 2)])) if (valid & (chm >= 2)).any() else None,
        "shape_chm": [int(ext_chm.ny), int(ext_chm.nx)],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--las", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dtm-cell", type=float, default=1.0)
    ap.add_argument("--chm-cell", type=float, default=0.5)
    ap.add_argument("--epsg", type=int, default=25833)
    a = ap.parse_args(argv)
    info = export_terrain(a.las, a.out, a.dtm_cell, a.chm_cell, a.epsg)
    for k, p in info["paths"].items():
        print(f"{k}: {p}")
    print(f"points {info['n_points']:,}  DTM {info['dtm_range'][0]:.1f}..{info['dtm_range'][1]:.1f} m  "
          f"CHM max {info['chm_max']:.1f} m  CHM median (>=2 m) {info['chm_median_vegetated']:.1f} m  grid {info['shape_chm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
