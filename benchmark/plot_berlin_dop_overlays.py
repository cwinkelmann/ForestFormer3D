#!/usr/bin/env python3
"""Crown-outline overlay figures on the Berlin DOP20 2021 orthophotos.

Produces four figures into ``docs/benchmarks/assets/berlin-dop``:

  a) berlin-dop-381-5829-crowns.png   full 1 km tile, crowns coloured by height
  b) berlin-dop-381-5829-zoom.png     150 m zoom, DOP + ALS instance raster
  c) berlin-dop-381-5829-drone2025.png same 150 m window on the 2025 drone ortho
  d) berlin-dop-mosaic-density.png    mosaic of every tile with results + crown density

Run with the repo's CPU venv::

    .venv-cpu/bin/python benchmark/plot_berlin_dop_overlays.py
"""

from __future__ import annotations

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

# DOP 2021 tiles. The 3-band RGB rendering is preferred; the 4-band RGBI stack of the
# same service is the fallback, since `read_rgb` only ever reads bands 1-3 and the later
# production blocks were fetched as RGBI only.
DOP_DIRS = [
    "/Volumes/2TB/winmol/ALS_Data/berlin_dop_2021/dop_2021_rgb",
    "/Volumes/2TB/winmol/ALS_Data/berlin_dop_2021/dop_2021_rgbi",
]
DOP_DIR = DOP_DIRS[0]
ALS_DIRS = [
    "/Users/christian/work/hnee/ForestFormer3D_runs/berlin_out/berlin-2021",
    "/Volumes/2TB/winmol/ALS_Data/berlin_als_2021_ff3d",
]
DRONE = ("/Volumes/2TB/winmol/training_data/WINDWURF_Tegel/Revier_12/"
         "ortho/result_Res1.2_QGIS_preview_1to8.tif")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "docs", "benchmarks", "assets", "berlin-dop")

# The original eleven-tile benchmark set. It is only a fallback: `discover_tiles()`
# below returns every km tile that actually has ForestFormer3D results under one of
# `ALS_DIRS`, so a new production block is picked up without editing this list.
BENCHMARK_TILES = ["379_5828", "379_5829", "380_5828", "380_5829", "381_5828",
                   "381_5829", "381_5830", "382_5828", "382_5829", "383_5828",
                   "383_5829"]
MAIN = "381_5829"
ZOOM_SIZE = 150.0

TILE_RE = re.compile(r"^3dm_33_(\d+_\d+)_1_be$")


def tname(t: str) -> str:
    return f"3dm_33_{t}_1_be"


def discover_tiles():
    """Every km tile with a merged tree table in one of ALS_DIRS, sorted by E then N."""
    found = set()
    for root in ALS_DIRS:
        if not os.path.isdir(root):
            continue
        for entry in os.listdir(root):
            m = TILE_RE.match(entry)
            if m and os.path.exists(os.path.join(root, entry, f"{entry}_trees.gpkg")):
                found.add(m.group(1))
    if not found:
        found = set(BENCHMARK_TILES)
    return sorted(found, key=lambda t: (int(t.split("_")[0]), int(t.split("_")[1])))


TILES = discover_tiles()


def dop_path(t: str):
    """The DOP 2021 GeoTIFF for a tile key, RGB first then RGBI, or None."""
    for root in DOP_DIRS:
        p = os.path.join(root, tname(t) + ".tif")
        if os.path.exists(p):
            return p
    return None


def tile_bounds(t: str):
    e, n = t.split("_")
    x, y = int(e) * 1000.0, int(n) * 1000.0
    return x, y, x + 1000.0, y + 1000.0


def als_path(tile: str, suffix: str):
    for root in ALS_DIRS:
        p = os.path.join(root, tname(tile), f"{tname(tile)}_{suffix}")
        if os.path.exists(p):
            return p
    return None


def read_rgb(path, bounds=None, max_px=2400):
    with rasterio.open(path) as src:
        if bounds is None:
            win, b = None, src.bounds
            h, w = src.height, src.width
        else:
            win = from_bounds(*bounds, transform=src.transform)
            b = bounds
            h, w = int(round(win.height)), int(round(win.width))
        scale = min(1.0, max_px / max(h, w))
        oh, ow = max(1, int(h * scale)), max(1, int(w * scale))
        arr = src.read([1, 2, 3], window=win, out_shape=(3, oh, ow),
                       resampling=Resampling.average, boundless=win is not None,
                       fill_value=0)
    return np.moveaxis(arr, 0, -1), (b[0], b[2], b[1], b[3])


def stretch(img, lo=1.0, hi=99.0, gamma=0.9):
    """Mild joint percentile stretch -- one lo/hi for all bands, so the
    leaf-off canopy gains contrast without losing its colour balance."""
    b = img.astype(np.float32)
    a, c = np.percentile(b, lo), np.percentile(b, hi)
    return np.clip((b - a) / max(c - a, 1e-6), 0, 1) ** gamma


def save(fig, out, budget_mb=1.45):
    """Save at 150 dpi; palette-quantise if the PNG exceeds the budget."""
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    for colors in (256, 128, 64):
        if os.path.getsize(out) / 1e6 <= budget_mb:
            break
        im = Image.open(out).convert("RGB").quantize(
            colors=colors, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)
        im.save(out, optimize=True)
    return out


def crown_lines(gdf):
    """Exterior rings of (multi)polygons as a list of (n,2) arrays."""
    segs = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            segs.append(np.zeros((0, 2)))
            continue
        if geom.geom_type == "Polygon":
            segs.append(np.asarray(geom.exterior.coords))
        elif geom.geom_type == "MultiPolygon":
            biggest = max(geom.geoms, key=lambda g: g.area)
            segs.append(np.asarray(biggest.exterior.coords))
        else:
            segs.append(np.asarray(geom.coords).reshape(-1, 2))
    return segs


def load_crowns(tile, bounds=None):
    import geopandas as gpd
    cp = als_path(tile, "crowns.gpkg")
    if cp is None:
        return None
    kw = {"bbox": bounds} if bounds else {}
    g = gpd.read_file(cp, **kw)
    tp = als_path(tile, "trees.gpkg")
    if tp is not None and "height" not in g.columns:
        t = gpd.read_file(tp)[["tree_id", "height"]]
        g = g.merge(t, on="tree_id", how="left")
    return g


def pick_zoom(tile):
    """Densest ZOOM_SIZE window inside the tile, on a 25 m grid."""
    import geopandas as gpd
    tp = als_path(tile, "trees.gpkg")
    t = gpd.read_file(tp)
    minx, miny, maxx, maxy = tile_bounds(tile)
    best, bxy = -1, (minx + 400, miny + 400)
    xs = t.geometry.x.to_numpy()
    ys = t.geometry.y.to_numpy()
    hs = t["height"].to_numpy()
    tall = hs > 22.0
    if tall.sum() < 50:
        tall = hs > 12.0
    for x0 in np.arange(minx + 50, maxx - ZOOM_SIZE - 50, 25.0):
        for y0 in np.arange(miny + 50, maxy - ZOOM_SIZE - 50, 25.0):
            m = (xs >= x0) & (xs < x0 + ZOOM_SIZE) & (ys >= y0) & (ys < y0 + ZOOM_SIZE)
            n = int((m & tall).sum())
            if n > best:
                best, bxy = n, (x0, y0)
    return (bxy[0], bxy[1], bxy[0] + ZOOM_SIZE, bxy[1] + ZOOM_SIZE), best


def fig_a(zoom_bounds):
    path = dop_path(MAIN)
    img, ext = read_rgb(path, max_px=2200)
    g = load_crowns(MAIN)
    segs = crown_lines(g)
    h = g["height"].fillna(0).to_numpy() if "height" in g else g["top_z"].to_numpy()
    norm = Normalize(0, np.nanpercentile(h, 98))
    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    ax.imshow(stretch(img), extent=ext)
    lc = LineCollection(segs, linewidths=0.18, cmap="viridis", norm=norm)
    lc.set_array(h)
    ax.add_collection(lc)
    for v in range(100, 1000, 100):
        ax.axvline(381000 + v, color="w", lw=0.3, alpha=0.35)
        ax.axhline(5829000 + v, color="w", lw=0.3, alpha=0.35)
    x0, y0, x1, y1 = zoom_bounds
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="red", lw=1.4)
    cb = fig.colorbar(lc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("tree height (m)")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_title(f"{tname(MAIN)} — DOP20 RGB 2021 with {len(g):,} ALS crown outlines\n"
                 "white grid = 100 m sub-tile borders, red box = zoom window")
    ax.set_xlabel("E (EPSG:25833)"); ax.set_ylabel("N (EPSG:25833)")
    return save(fig, os.path.join(OUT_DIR, "berlin-dop-381-5829-crowns.png"))


def fig_b(zoom_bounds):
    img, ext = read_rgb(dop_path(MAIN),
                        bounds=zoom_bounds, max_px=1600)
    g = load_crowns(MAIN, bounds=zoom_bounds)
    segs = crown_lines(g)
    ip = als_path(MAIN, "instance_50cm.tif")
    with rasterio.open(ip) as src:
        win = from_bounds(*zoom_bounds, transform=src.transform)
        inst = src.read(1, window=win)
    rng = np.random.default_rng(7)
    lut = rng.random((int(inst.max()) + 2, 3)) * 0.75 + 0.25
    lut[0] = 0.08
    rgbi = lut[np.clip(inst + 1, 0, len(lut) - 1)]
    rgbi[inst <= 0] = 0.08
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 6.6))
    ax[0].imshow(stretch(img), extent=ext)
    ax[0].add_collection(LineCollection(segs, linewidths=0.9, colors="#ff2bd6"))
    ax[0].set_title("DOP20 RGB 2021 + crown outlines")
    ax[1].imshow(rgbi, extent=ext)
    ax[1].add_collection(LineCollection(segs, linewidths=0.6, colors="w", alpha=0.6))
    ax[1].set_title("ALS instance raster (50 cm), one colour per tree")
    for a in ax:
        a.set_xlim(ext[0], ext[1]); a.set_ylim(ext[2], ext[3])
        a.set_xlabel("E (EPSG:25833)")
    ax[0].set_ylabel("N (EPSG:25833)")
    fig.suptitle(f"{tname(MAIN)} — {int(ZOOM_SIZE)} m x {int(ZOOM_SIZE)} m zoom "
                 f"at E{int(zoom_bounds[0])} N{int(zoom_bounds[1])}, "
                 f"{len(g)} crowns")
    return save(fig, os.path.join(OUT_DIR, "berlin-dop-381-5829-zoom.png"))


def fig_c(zoom_bounds):
    import geopandas as gpd
    with rasterio.open(DRONE) as src:
        drone_crs = src.crs
    g = load_crowns(MAIN, bounds=zoom_bounds).to_crs(drone_crs)
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:25833", drone_crs, always_xy=True)
    x0, y0 = tr.transform(zoom_bounds[0], zoom_bounds[1])
    x1, y1 = tr.transform(zoom_bounds[2], zoom_bounds[3])
    db = (x0, y0, x1, y1)
    dimg, dext = read_rgb(DRONE, bounds=db, max_px=1600)
    img21, ext21 = read_rgb(dop_path(MAIN),
                            bounds=zoom_bounds, max_px=1600)
    segs21 = crown_lines(load_crowns(MAIN, bounds=zoom_bounds))
    segs = crown_lines(g)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 6.6))
    ax[0].imshow(stretch(img21), extent=ext21)
    ax[0].add_collection(LineCollection(segs21, linewidths=0.9, colors="#ff2bd6"))
    ax[0].set_title("2021 DOP20 (20 cm, leaf-off 22.02.2021)")
    ax[0].set_xlabel("E (EPSG:25833)"); ax[0].set_ylabel("N (EPSG:25833)")
    ax[1].imshow(stretch(dimg), extent=dext)
    ax[1].add_collection(LineCollection(segs, linewidths=0.9, colors="#ff2bd6"))
    ax[1].set_title("2025 drone ortho (9.6 cm, EPSG:32633)")
    ax[1].set_xlabel("E (EPSG:32633)")
    ax[0].set_xlim(ext21[0], ext21[1]); ax[0].set_ylim(ext21[2], ext21[3])
    ax[1].set_xlim(dext[0], dext[1]); ax[1].set_ylim(dext[2], dext[3])
    fig.suptitle("Same 150 m window, 2021 vs 2025 — ALS 2021 crown outlines on both")
    return save(fig, os.path.join(OUT_DIR, "berlin-dop-381-5829-drone2025.png"))


def fig_d():
    import geopandas as gpd
    present = [t for t in TILES
               if dop_path(t)]
    es = sorted({int(t.split("_")[0]) for t in present})
    ns = sorted({int(t.split("_")[1]) for t in present}, reverse=True)
    cell = 420
    mosaic = np.full((len(ns) * cell, len(es) * cell, 3), 55, dtype=np.uint8)
    minx, maxx = min(es) * 1000.0, (max(es) + 1) * 1000.0
    miny, maxy = min(ns) * 1000.0, (max(ns) + 1) * 1000.0
    have_trees = []
    for t in present:
        e, n = int(t.split("_")[0]), int(t.split("_")[1])
        r, c = ns.index(n), es.index(e)
        img, _ = read_rgb(dop_path(t), max_px=cell)
        mosaic[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = img[:cell, :cell]
        if als_path(t, "trees.gpkg"):
            have_trees.append(t)
    xs, ys = [], []
    for t in have_trees:
        g = gpd.read_file(als_path(t, "trees.gpkg"))
        xs.append(g.geometry.x.to_numpy()); ys.append(g.geometry.y.to_numpy())
    xs = np.concatenate(xs) if xs else np.array([])
    ys = np.concatenate(ys) if ys else np.array([])
    ext = (minx, maxx, miny, maxy)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.4), constrained_layout=True)
    ax[0].imshow(mosaic, extent=ext)
    ax[0].set_title(f"DOP20 RGB 2021 mosaic — {len(present)} km tiles\n(grey = no ForestFormer3D result for that km square)", fontsize=11)
    ax[1].imshow(mosaic, extent=ext, alpha=0.55)
    if xs.size:
        nb = (int((maxx - minx) // 50), int((maxy - miny) // 50))
        h, xe, ye = np.histogram2d(xs, ys, bins=nb,
                                   range=[[minx, maxx], [miny, maxy]])
        h = np.ma.masked_where(h == 0, h)
        im = ax[1].imshow(h.T[::-1], extent=ext, cmap="inferno", alpha=0.75,
                          vmax=np.percentile(h.compressed(), 99))
        cb = fig.colorbar(im, ax=ax[1], fraction=0.04, pad=0.02)
        cb.set_label("crowns per 50 m x 50 m cell")
    ax[1].set_title(f"Crown centroid density — {len(xs):,} trees\n"
                    f"over the {len(have_trees)} tiles with ForestFormer3D results",
                    fontsize=11)
    for a in ax:
        a.set_xlabel("E (EPSG:25833)")
        for e in es:
            a.axvline(e * 1000, color="w", lw=0.5, alpha=0.5)
        for n in ns:
            a.axhline(n * 1000, color="w", lw=0.5, alpha=0.5)
        a.set_xlim(minx, maxx); a.set_ylim(miny, maxy)
    ax[0].set_ylabel("N (EPSG:25833)")
    return save(fig, os.path.join(OUT_DIR, "berlin-dop-mosaic-density.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, choices=list("abcd"))
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    zb, n = pick_zoom(MAIN)
    print(f"zoom window {zb} with {n} trees >12 m")
    want = args.only or list("abcd")
    for key, fn in (("a", lambda: fig_a(zb)), ("b", lambda: fig_b(zb)),
                    ("c", lambda: fig_c(zb)), ("d", fig_d)):
        if key not in want:
            continue
        out = fn()
        print(f"{key}: {out}  {os.path.getsize(out)/1e6:.2f} MB")


if __name__ == "__main__":
    main()
