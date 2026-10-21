#!/usr/bin/env python3
"""One seamless terrain model over a whole ALS mosaic: merged DTM + hillshade.

    python3 benchmark/terrain_mosaic.py --terrain <ALS_Data/berlin_terrain> \
        --out <ALS_Data/berlin_terrain_mosaic> [--threshold 0.25]

``benchmark/export_terrain.py`` writes one DTM, DSM and CHM per km tile. This
script turns the per-tile DTMs into a single product:

  <prefix>_dtm_1m.tif        the merged terrain model, float32, nodata -9999,
                             a Cloud Optimized GeoTIFF with overviews
  <prefix>_hillshade_1m.tif  shaded relief from it, uint8 (gdaldem, az 315 /
                             alt 45, --compute-edges so tile borders and the
                             mosaic's holes get no dark rim)
  <prefix>_mosaic.json       extent, cell, tiles, nodata share, z range and the
                             seam statistics below

Every tile DTM is already snapped to a whole-metre lattice in the same CRS
(``ff3d_geo.grid.grid_extent`` floors the origin to a multiple of the cell), so
the merge is a paste with no resampling -- ``mosaic_plan`` refuses the job if
that ever stops being true.

Why the seam check: ``export_terrain`` fills each tile's ground holes with
``nearest_fill`` INSIDE that tile. A hole straddling a km border -- a building,
a lake -- is therefore filled from one side in one tile and from the other side
in its neighbour, with neither knowing the other. That can leave a step exactly
along the km grid, and a hillshade is the most unforgiving way to show it: a
20 cm step reads as a bright line. So every pair of adjacent tiles is compared
along the two cell columns either side of their shared border, and the report
names the worst seam and how many exceed ``--threshold`` metres.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

NODATA = -9999.0
EPS = 1e-6


@dataclass(frozen=True)
class TileRaster:
    """One tile's raster geometry: south-west corner, size in cells, cell size."""

    name: str
    path: Path
    x0: float
    y0: float
    nx: int
    ny: int
    cell: float

    @property
    def x1(self) -> float:
        return self.x0 + self.nx * self.cell

    @property
    def y1(self) -> float:
        return self.y0 + self.ny * self.cell


# ------------------------------------------------------------------- pure arithmetic
def mosaic_plan(tiles: list[TileRaster]) -> dict:
    """The union grid of ``tiles``, or ValueError if they cannot be pasted together.

    ``covered_cells`` counts the tiles' own cells, so ``gap_fraction`` is the share
    of the merged raster that no tile reaches -- a third of the Berlin mosaic, whose
    57 tiles sit inside an 11 x 7 km box.
    """
    if not tiles:
        raise ValueError("no tiles to mosaic")
    cells = {t.cell for t in tiles}
    if len(cells) != 1:
        raise ValueError(f"tiles differ in cell size: {sorted(cells)}")
    cell = cells.pop()

    x0 = min(t.x0 for t in tiles)
    y0 = min(t.y0 for t in tiles)
    for t in tiles:
        for axis, off in (("x", t.x0 - x0), ("y", t.y0 - y0)):
            rem = abs(off / cell - round(off / cell))
            if rem > EPS:
                raise ValueError(
                    f"{t.name} is off the common lattice in {axis}: {off} m is not a "
                    f"whole number of {cell} m cells")

    nx = int(round((max(t.x1 for t in tiles) - x0) / cell))
    ny = int(round((max(t.y1 for t in tiles) - y0) / cell))
    covered = sum(t.nx * t.ny for t in tiles)
    return {"x0": float(x0), "y0": float(y0), "cell": float(cell), "nx": nx, "ny": ny,
            "n_tiles": len(tiles), "covered_cells": covered,
            "gap_fraction": 1.0 - covered / (nx * ny)}


def adjacent_pairs(tiles: list[TileRaster]) -> list[tuple[TileRaster, TileRaster, str]]:
    """Pairs of tiles sharing a border, as ``(lower, upper, axis)``.

    ``axis`` is ``"x"`` when they sit side by side (the border is a vertical line and
    ``lower`` is the western tile) and ``"y"`` when stacked (``lower`` is southern).
    Diagonal neighbours share a corner, not an edge, and are not pairs.
    """
    out = []
    for i, a in enumerate(tiles):
        for b in tiles[i + 1:]:
            for first, second in ((a, b), (b, a)):
                if abs(first.x1 - second.x0) < EPS and _overlap(
                        first.y0, first.y1, second.y0, second.y1) > EPS:
                    out.append((first, second, "x"))
                if abs(first.y1 - second.y0) < EPS and _overlap(
                        first.x0, first.x1, second.x0, second.x1) > EPS:
                    out.append((first, second, "y"))
    return out


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return min(a1, b1) - max(a0, b0)


def seam_profile(left: np.ndarray, right: np.ndarray) -> dict:
    """Height differences across one seam, over the cells where both sides have data.

    ``left``/``right`` are the cell values immediately either side of the border, in
    the same order along it. Nodata must already be NaN.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"seam strips differ in length: {left.shape} vs {right.shape}")
    both = np.isfinite(left) & np.isfinite(right)
    if not both.any():
        return {"n": 0, "mean_abs": None, "p99": None, "max": None}
    d = np.abs(left[both] - right[both])
    return {"n": int(both.sum()), "mean_abs": float(d.mean()),
            "p99": float(np.percentile(d, 99)), "max": float(d.max())}


def seam_report(stats: list[dict], threshold: float = 0.25) -> dict:
    """Aggregate per-seam profiles; a seam counts as a step when its mean |dz| exceeds
    ``threshold`` metres. Seams where no cell pair had data on both sides (a lake on
    the border) are reported but cannot be judged."""
    measured = [s for s in stats if s["n"] > 0]
    over = [s for s in measured if s["mean_abs"] > threshold]
    return {
        "n_seams": len(stats),
        "n_measured": len(measured),
        "threshold_m": threshold,
        "n_over_threshold": len(over),
        "over_threshold": [s["pair"] for s in over],
        "worst": max(measured, key=lambda s: s["mean_abs"]) if measured else None,
        "mean_abs_over_all_seams":
            float(np.mean([s["mean_abs"] for s in measured])) if measured else None,
    }


# ------------------------------------------------------------------------- raster I/O
def read_tiles(paths: list[Path]) -> list[TileRaster]:
    """Tile geometry from the rasters themselves; refuses a mixed-CRS set."""
    import rasterio

    tiles, crs = [], set()
    for p in paths:
        with rasterio.open(p) as d:
            if d.res[0] != d.res[1]:
                raise ValueError(f"{p.name}: non-square cells {d.res}")
            crs.add(d.crs.to_epsg())
            tiles.append(TileRaster(name=p.parent.name, path=p, x0=d.bounds.left,
                                    y0=d.bounds.bottom, nx=d.width, ny=d.height,
                                    cell=float(d.res[0])))
    if len(crs) > 1:
        raise ValueError(f"tiles do not share a CRS: {sorted(crs)}")
    return tiles


def seam_strips(a: TileRaster, b: TileRaster, axis: str) -> tuple[np.ndarray, np.ndarray]:
    """The one-cell strips either side of the border between ``a`` and ``b``.

    Read from the two tiles independently (they do not overlap), clipped to the part
    of the border they actually share, with nodata turned into NaN.
    """
    import rasterio

    lo = max(a.y0, b.y0) if axis == "x" else max(a.x0, b.x0)
    hi = min(a.y1, b.y1) if axis == "x" else min(a.x1, b.x1)
    n = int(round((hi - lo) / a.cell))

    def strip(t: TileRaster, edge: str) -> np.ndarray:
        with rasterio.open(t.path) as d:
            if axis == "x":
                col = t.nx - 1 if edge == "last" else 0
                row0 = int(round((t.y1 - hi) / t.cell))   # rasterio row 0 is the north edge
                win = rasterio.windows.Window(col, row0, 1, n)
            else:
                row = 0 if edge == "last" else t.ny - 1   # 'last' in y is the northern row
                col0 = int(round((lo - t.x0) / t.cell))
                win = rasterio.windows.Window(col0, row, n, 1)
            v = d.read(1, window=win, masked=True).astype(np.float64).filled(np.nan)
        return v.ravel()

    return strip(a, "last"), strip(b, "first")


def measure_seams(pairs, threshold: float) -> tuple[dict, list[dict]]:
    stats = []
    for a, b, axis in pairs:
        left, right = seam_strips(a, b, axis)
        s = seam_profile(left, right)
        s["pair"] = f"{a.name}|{b.name}"
        s["axis"] = axis
        stats.append(s)
    return seam_report(stats, threshold), stats


# ----------------------------------------------------------------------------- gdal
def run(cmd: list[str]) -> str:
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{Path(str(cmd[0])).name} failed ({r.returncode}): "
                           f"{r.stderr.strip()[:400]}")
    return r.stdout


def build_merged(gdal_bin: Path, tiles: list[TileRaster], out_tif: Path,
                 nodata: float = NODATA) -> Path:
    """gdalbuildvrt over the tiles, then one COG. The VRT is kept next to the output:
    it is how the merge can be re-run or inspected without re-listing the tiles."""
    vrt = out_tif.with_suffix(".vrt")
    run([gdal_bin / "gdalbuildvrt", "-overwrite", "-q",
         "-srcnodata", nodata, "-vrtnodata", nodata, vrt, *[t.path for t in tiles]])
    run([gdal_bin / "gdal_translate", "-q", "-of", "COG",
         "-co", "COMPRESS=DEFLATE", "-co", "PREDICTOR=3", "-co", "BIGTIFF=IF_SAFER",
         "-co", "RESAMPLING=AVERAGE", "-a_nodata", nodata, vrt, out_tif])
    return out_tif


def build_hillshade(gdal_bin: Path, dtm: Path, out_tif: Path, azimuth: float = 315.0,
                    altitude: float = 45.0, z_factor: float = 1.0) -> Path:
    """Shaded relief from the merged DTM. ``-compute_edges`` keeps the mosaic's outer
    boundary and its holes from drawing a one-cell black frame."""
    run([gdal_bin / "gdaldem", "hillshade", "-q", "-compute_edges",
         "-az", azimuth, "-alt", altitude, "-z", z_factor,
         "-of", "COG", "-co", "COMPRESS=DEFLATE", dtm, out_tif])
    return out_tif


def raster_stats(path: Path) -> dict:
    """Valid-cell count and z range, straight from the band statistics."""
    import rasterio

    with rasterio.open(path) as d:
        a = d.read(1, masked=True)
        valid = int(a.count())
        return {"shape": [d.height, d.width], "valid_cells": valid,
                "nodata_fraction": 1.0 - valid / (d.height * d.width),
                "z_min": float(a.min()) if valid else None,
                "z_max": float(a.max()) if valid else None}


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terrain", required=True, type=Path,
                    help="directory of per-tile terrain dirs (berlin_terrain)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--suffix", default="_dtm_1m.tif", help="per-tile DTM file suffix")
    ap.add_argument("--prefix", default="berlin", help="output file prefix")
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="mean |dz| across a seam that counts as a step, in metres")
    ap.add_argument("--azimuth", type=float, default=315.0)
    ap.add_argument("--altitude", type=float, default=45.0)
    ap.add_argument("--z-factor", type=float, default=1.0)
    ap.add_argument("--gdal-bin", type=Path, default=Path("/opt/local/bin"))
    ap.add_argument("--skip-seams", action="store_true",
                    help="merge only; do not read the tile borders back")
    a = ap.parse_args(argv)

    paths = sorted(d / f"{d.name}{a.suffix}" for d in sorted(a.terrain.iterdir())
                   if d.is_dir() and (d / f"{d.name}{a.suffix}").exists())
    if not paths:
        print(f"no *{a.suffix} under {a.terrain}", file=sys.stderr)
        return 2
    tiles = read_tiles(paths)
    plan = mosaic_plan(tiles)
    print(f"{plan['n_tiles']} tiles -> {plan['nx']} x {plan['ny']} cells of "
          f"{plan['cell']:g} m, {plan['gap_fraction'] * 100:.1f} % of the box is gap")

    a.out.mkdir(parents=True, exist_ok=True)
    dtm = build_merged(a.gdal_bin, tiles, a.out / f"{a.prefix}_dtm_1m.tif")
    print(f"merged  {dtm}")
    shade = build_hillshade(a.gdal_bin, dtm, a.out / f"{a.prefix}_hillshade_1m.tif",
                            a.azimuth, a.altitude, a.z_factor)
    print(f"hillshade {shade}")

    info = {"plan": plan, "dtm": str(dtm), "hillshade": str(shade),
            "tiles": [t.name for t in tiles], "raster": raster_stats(dtm)}
    print(f"DTM {info['raster']['z_min']:.1f}..{info['raster']['z_max']:.1f} m, "
          f"{info['raster']['nodata_fraction'] * 100:.1f} % nodata")

    if not a.skip_seams:
        pairs = adjacent_pairs(tiles)
        rep, stats = measure_seams(pairs, a.threshold)
        info["seams"] = rep
        info["seam_detail"] = stats
        w = rep["worst"]
        print(f"seams   {rep['n_seams']} shared borders, {rep['n_measured']} measurable, "
              f"mean |dz| {rep['mean_abs_over_all_seams']:.3f} m, "
              f"{rep['n_over_threshold']} over {a.threshold} m"
              + (f" (worst {w['pair']} {w['mean_abs']:.3f} m)" if w else ""))

    j = a.out / f"{a.prefix}_mosaic.json"
    j.write_text(json.dumps(info, indent=1))
    print(f"wrote {j}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
