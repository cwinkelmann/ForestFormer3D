#!/usr/bin/env python3
"""Overlay figure: which tree instances the ALKIS building mask removes.

DOP20 2021 orthophoto of a window of one km tile, with the ALKIS building footprints
outlined, the crown hulls of the instances that ``ff3d_geo buildings`` **removed** in
red and the crowns it **kept** in green.

The window defaults to the 250 m square with the most removed instances, which on
``3dm_33_381_5829_1_be`` is the allotment area -- the place where the missing building
class in the Berlin ALS 2021 classification shows up most clearly.

Run with the repo's CPU venv::

    .venv-cpu/bin/python benchmark/plot_berlin_buildings_overlay.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DOP_DIR = Path("/Volumes/2TB/winmol/ALS_Data/berlin_dop_2021/dop_2021_rgb")
ALS_DIR = Path("/Volumes/2TB/winmol/ALS_Data/berlin_als_2021_ff3d")
BUILDINGS = Path("/Volumes/2TB/winmol/ALS_Data/berlin_buildings/alkis_buildings.gpkg")
OUT = REPO / "docs" / "benchmarks" / "assets" / "berlin-visual" / "09-buildings.png"
DEFAULT_TILE = "3dm_33_381_5829_1_be"


def read_dop(path: Path, bounds, max_px: int = 1100):
    """RGB array for ``bounds`` (minx, miny, maxx, maxy), downsampled to ``max_px``."""
    with rasterio.open(path) as src:
        window = from_bounds(*bounds, transform=src.transform)
        height = int(min(max_px, max(1, window.height)))
        width = int(min(max_px, max(1, window.width)))
        data = src.read(indexes=[1, 2, 3], window=window, out_shape=(3, height, width),
                        resampling=Resampling.bilinear, boundless=True, fill_value=255)
    return np.transpose(data, (1, 2, 0))


def outlines(frame) -> list[np.ndarray]:
    """Exterior rings of every (multi)polygon in ``frame``, as coordinate arrays."""
    segments = []
    for geom in frame.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        for part in parts:
            if part.geom_type != "Polygon":
                continue
            segments.append(np.asarray(part.exterior.coords)[:, :2])
    return segments


def best_window(points, bounds, size: float, step: float = 25.0):
    """The ``size`` m square inside ``bounds`` holding the most of ``points``."""
    minx, miny, maxx, maxy = bounds
    if points.size == 0:
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)
    best, best_n = None, -1
    for x0 in np.arange(minx, max(minx + step, maxx - size), step):
        for y0 in np.arange(miny, max(miny + step, maxy - size), step):
            n = int(np.count_nonzero((points[:, 0] >= x0) & (points[:, 0] < x0 + size)
                                     & (points[:, 1] >= y0) & (points[:, 1] < y0 + size)))
            if n > best_n:
                best, best_n = (x0, y0, x0 + size, y0 + size), n
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", default=DEFAULT_TILE)
    ap.add_argument("--dop-dir", type=Path, default=DOP_DIR)
    ap.add_argument("--als-dir", type=Path, default=ALS_DIR)
    ap.add_argument("--buildings", type=Path, default=BUILDINGS)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--size", type=float, default=250.0, help="window edge in metres")
    ap.add_argument("--bounds", nargs=4, type=float, default=None,
                    metavar=("MINX", "MINY", "MAXX", "MAXY"))
    ap.add_argument("--dpi", type=int, default=95)
    args = ap.parse_args(argv)

    tile_dir = args.als_dir / args.tile
    crowns_all = gpd.read_file(tile_dir / f"{args.tile}_crowns.gpkg")
    crowns_kept = gpd.read_file(tile_dir / "masked" / f"{args.tile}_crowns.gpkg")
    kept_ids = set(crowns_kept["tree_id"].tolist())
    removed = crowns_all[~crowns_all["tree_id"].isin(kept_ids)]
    print(f"{args.tile}: {len(crowns_all)} crowns, {len(removed)} removed, "
          f"{len(crowns_kept)} kept")

    if args.bounds:
        bounds = tuple(args.bounds)
    else:
        centres = np.array([[g.centroid.x, g.centroid.y] for g in removed.geometry])
        bounds = best_window(centres, tuple(crowns_all.total_bounds), args.size)
    print("window", " ".join(f"{v:.0f}" for v in bounds))

    dop = read_dop(args.dop_dir / f"{args.tile}.tif", bounds)
    clip = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(
        [f"POLYGON(({bounds[0]} {bounds[1]}, {bounds[2]} {bounds[1]}, "
         f"{bounds[2]} {bounds[3]}, {bounds[0]} {bounds[3]}, {bounds[0]} {bounds[1]}))"]),
        crs=crowns_all.crs)
    footprints = gpd.read_file(args.buildings, layer="buildings", bbox=clip)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(dop, extent=(bounds[0], bounds[2], bounds[1], bounds[3]), origin="upper")
    ax.add_collection(LineCollection(outlines(footprints), colors="#ffd400",
                                     linewidths=1.4, zorder=3))
    ax.add_collection(LineCollection(outlines(gpd.clip(crowns_kept, clip)),
                                     colors="#19c219", linewidths=1.0, zorder=4))
    ax.add_collection(LineCollection(outlines(gpd.clip(removed, clip)),
                                     colors="#ff2222", linewidths=1.4, zorder=5))
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.set_xlabel("E (EPSG:25833)")
    ax.set_ylabel("N (EPSG:25833)")
    ax.set_title(f"{args.tile} — ALKIS building mask\n"
                 f"{int(args.size)} m window at E{int(bounds[0])} N{int(bounds[1])}; "
                 f"{len(removed)} of {len(crowns_all)} instances removed on the whole tile")
    ax.legend(handles=[
        Line2D([], [], color="#ffd400", lw=1.6, label="ALKIS building footprint"),
        Line2D([], [], color="#ff2222", lw=1.6, label="instance removed (roof)"),
        Line2D([], [], color="#19c219", lw=1.6, label="instance kept (tree)"),
    ], loc="upper right", framealpha=0.85)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
