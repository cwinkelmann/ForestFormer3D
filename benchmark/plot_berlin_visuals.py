#!/usr/bin/env python3
"""Render the figures of the Berlin ALS 2021 ForestFormer3D visual report.

Reads the per-km-tile inference products written by ``ff3d_geo`` --
``<tile>.las`` (with the extra dimensions ``treeID``/``semantic``/``score``),
``<tile>_trees.gpkg``, ``<tile>_crowns.gpkg``, ``<tile>_instance_50cm.tif`` and
``<tile>_report.json`` -- and writes eight PNG figures.

Usage::

    .venv-cpu/bin/python benchmark/plot_berlin_visuals.py \
        --tile 3dm_33_381_5829_1_be \
        --data-dir ~/work/hnee/ForestFormer3D_runs/berlin_out/berlin-2021 \
        --out-dir docs/benchmarks/assets/berlin-visual

``--data-dir`` may be repeated; the first directory that holds
``<tile>/<tile>.las`` wins for the single-tile figures, while the multi-tile
overview uses every tile directory found under any of them.

Only numpy / laspy / geopandas / rasterio / scipy / matplotlib are used, and
matplotlib always runs on the ``Agg`` backend.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LightSource, Normalize
from matplotlib.lines import Line2D
from rasterio.windows import from_bounds as window_from_bounds

CELL = 0.5  # m, raster cell size for CHM / instance rasters
TILE_M = 1000.0  # km tile edge
AOI_GPKG = Path(
    "/Users/christian/data/Winmol/training_data/WINDWURF_Tegel/Revier_12/"
    "202507_Rev_12_Tegelsee.gpkg"
)
AOI_LAYER = "202507_Rev_12_Tegelsee_AOI"
PLOT_LAYERS = (
    "202507_Rev_12_Tegelsee",
    "202507_Rev_12_Tegelsee_Probekreis2",
    "202507_Rev_12_Tegelsee_Probekreis3Inhalt",
)
SEM_COLORS = {0: "#b9a179", 1: "#8c4a2f", 2: "#2f8f3e", 255: "#dddddd"}
SEM_NAMES = {0: "ground", 1: "wood", 2: "leaf", 255: "unassigned"}
GREY = "#cfcfcf"
MAX_PNG_BYTES = 1_500_000


# --------------------------------------------------------------------------
# paths and small helpers
# --------------------------------------------------------------------------
def tile_origin(tile: str) -> tuple[float, float]:
    """(x0, y0) of a ``3dm_33_<E km>_<N km>_1_be`` tile in EPSG:25833."""
    parts = tile.split("_")
    return float(parts[2]) * 1000.0, float(parts[3]) * 1000.0


def tile_dir(data_dirs: list[Path], tile: str) -> Path:
    for d in data_dirs:
        if (d / tile / f"{tile}.las").exists() or (d / tile / f"{tile}_trees.gpkg").exists():
            return d / tile
    raise SystemExit(f"tile {tile} not found under {[str(d) for d in data_dirs]}")


def find_tiles(data_dirs: list[Path]) -> dict[str, Path]:
    """Every tile directory that already carries a ``_trees.gpkg``."""
    found: dict[str, Path] = {}
    for d in data_dirs:
        if not d.is_dir():
            continue
        for sub in sorted(d.iterdir()):
            if not sub.is_dir() or sub.name.startswith("."):
                continue
            if (sub / f"{sub.name}_trees.gpkg").exists():
                found.setdefault(sub.name, sub)
    return found


def save(fig: plt.Figure, path: Path, dpi: int = 170) -> Path:
    """Save a figure, then palette-quantise it if the PNG came out too big."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if path.stat().st_size > MAX_PNG_BYTES:
        try:
            from PIL import Image

            img = Image.open(path).convert("RGB")
            img.convert("P", palette=Image.ADAPTIVE, colors=256).save(path, optimize=True)
        except Exception as exc:  # pragma: no cover - cosmetic only
            print(f"  (could not quantise {path.name}: {exc})")
    print(f"  wrote {path.name}  {path.stat().st_size / 1e6:.2f} MB")
    return path


def tree_colors(ids: np.ndarray, seed: int = 7) -> np.ndarray:
    """Random categorical RGB per tree id; -1 (no tree) becomes light grey."""
    uniq = np.unique(ids)
    rng = np.random.default_rng(seed)
    h = rng.permutation(np.linspace(0.0, 1.0, max(len(uniq), 2), endpoint=False))
    s = rng.uniform(0.55, 0.95, len(uniq))
    v = rng.uniform(0.55, 0.98, len(uniq))
    import matplotlib.colors as mcolors

    lut = mcolors.hsv_to_rgb(np.stack([h[: len(uniq)], s, v], axis=1))
    lut[uniq < 0] = matplotlib.colors.to_rgb(GREY)
    return lut[np.searchsorted(uniq, ids)]


def scale_bar(ax, length: float, label: str | None = None, loc=(0.06, 0.05), color="black") -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    sx = x0 + loc[0] * (x1 - x0)
    sy = y0 + loc[1] * (y1 - y0)
    ax.plot([sx, sx + length], [sy, sy], color=color, lw=3, solid_capstyle="butt", zorder=6)
    ax.text(
        sx + length / 2,
        sy + 0.012 * (y1 - y0),
        label or f"{length:g} m",
        ha="center",
        va="bottom",
        color=color,
        fontsize=8,
        zorder=6,
    )


def utm_axes(ax, title: str) -> None:
    ax.set_aspect("equal")
    ax.set_xlabel("easting (m, EPSG:25833)")
    ax.set_ylabel("northing (m)")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.set_title(title)


# --------------------------------------------------------------------------
# LAS pass
# --------------------------------------------------------------------------
def read_las(
    las_path: Path,
    window: tuple[float, float, float, float],
    sub_target: int,
    origin: tuple[float, float],
    cache: Path | None,
) -> dict:
    """One streaming pass over the LAS: tile subsample, window points, CHM grid."""
    if cache is not None and cache.exists():
        print(f"  cache hit {cache}")
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}

    import laspy

    wx0, wy0, wx1, wy1 = window
    x0, y0 = origin
    n_cell = int(TILE_M / CELL)
    dsm = np.full(n_cell * n_cell, -9e9, dtype=np.float32)
    dtm = np.full(n_cell * n_cell, 9e9, dtype=np.float32)

    sub: list[np.ndarray] = []
    win: list[np.ndarray] = []
    rng = np.random.default_rng(3)

    with laspy.open(las_path) as fh:
        total = fh.header.point_count
        frac = min(1.0, sub_target / float(total))
        print(f"  {las_path.name}: {total:,} points, keeping ~{frac:.3%} for the overview")
        for chunk in fh.chunk_iterator(4_000_000):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float32)
            tid = np.asarray(chunk.treeID, dtype=np.int32)
            sem = np.asarray(chunk.semantic, dtype=np.uint8)
            cls = np.asarray(chunk.classification, dtype=np.uint8)

            ix = np.clip(((x - x0) / CELL).astype(np.int64), 0, n_cell - 1)
            iy = np.clip(((y - y0) / CELL).astype(np.int64), 0, n_cell - 1)
            flat = iy * n_cell + ix
            veg = np.isin(cls, (3, 4, 5))
            np.maximum.at(dsm, flat[veg], z[veg])
            gnd = cls == 2
            np.minimum.at(dtm, flat[gnd], z[gnd])

            keep = rng.random(len(x)) < frac
            sub.append(
                np.stack(
                    [x[keep] - x0, y[keep] - y0, z[keep], tid[keep].astype(np.float32)], axis=1
                ).astype(np.float32)
            )
            m = (x >= wx0) & (x < wx1) & (y >= wy0) & (y < wy1)
            if m.any():
                win.append(
                    np.stack(
                        [
                            x[m] - x0,
                            y[m] - y0,
                            z[m],
                            tid[m].astype(np.float32),
                            sem[m].astype(np.float32),
                        ],
                        axis=1,
                    ).astype(np.float32)
                )

    dsm = dsm.reshape(n_cell, n_cell)
    dtm = dtm.reshape(n_cell, n_cell)
    dsm[dsm < -8e8] = np.nan
    dtm[dtm > 8e8] = np.nan

    # fill DTM gaps (under canopy / buildings) with the nearest ground cell
    from scipy import ndimage

    bad = ~np.isfinite(dtm)
    if bad.any():
        idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
        dtm = dtm[tuple(idx)]
    chm = np.where(np.isfinite(dsm), dsm - dtm, 0.0).astype(np.float32)
    chm = np.clip(np.nan_to_num(chm), 0.0, 60.0)

    out = {
        "sub": np.concatenate(sub) if sub else np.zeros((0, 4), np.float32),
        "win": np.concatenate(win) if win else np.zeros((0, 5), np.float32),
        "chm": chm,
        "origin": np.asarray(origin, dtype=np.float64),
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **out)
    return out


def densest_window(trees: gpd.GeoDataFrame, origin, size: float, step: float = 20.0):
    """The ``size`` x ``size`` window inside the tile holding the most trees."""
    x0, y0 = origin
    tx = trees["x"].to_numpy() - x0
    ty = trees["y"].to_numpy() - y0
    edges = np.arange(0.0, TILE_M + step, step)
    h, _, _ = np.histogram2d(tx, ty, bins=[edges, edges])
    k = int(round(size / step))
    csum = np.cumsum(np.cumsum(h, axis=0), axis=1)
    csum = np.pad(csum, ((1, 0), (1, 0)))
    box = csum[k:, k:] - csum[:-k, k:] - csum[k:, :-k] + csum[:-k, :-k]
    i, j = np.unravel_index(int(np.argmax(box)), box.shape)
    return (
        x0 + i * step,
        y0 + j * step,
        x0 + i * step + size,
        y0 + j * step + size,
        int(box[i, j]),
    )


def load_aoi(bounds=None):
    """AOI polygons (and sample-plot layers) clipped to ``bounds``, or None."""
    if not AOI_GPKG.exists():
        return None, {}
    aoi = gpd.read_file(AOI_GPKG, layer=AOI_LAYER)
    plots = {}
    for lyr in PLOT_LAYERS:
        try:
            plots[lyr] = gpd.read_file(AOI_GPKG, layer=lyr)
        except Exception:
            pass
    if bounds is not None:
        from shapely.geometry import box as shbox

        clip = shbox(*bounds)
        aoi = aoi[aoi.intersects(clip)]
        plots = {k: v[v.intersects(clip)] for k, v in plots.items()}
        plots = {k: v for k, v in plots.items() if len(v)}
    return aoi, plots


def crown_polys(crowns: gpd.GeoDataFrame) -> list[np.ndarray]:
    polys = []
    for geom in crowns.geometry:
        if geom is None or geom.is_empty or geom.geom_type != "Polygon":
            continue
        polys.append(np.asarray(geom.exterior.coords)[:, :2])
    return polys


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def fig_topdown(d, tile, origin, out: Path):
    sub = d["sub"]
    x, y, tid = sub[:, 0] + origin[0], sub[:, 1] + origin[1], sub[:, 3].astype(np.int32)
    fig, ax = plt.subplots(figsize=(9, 9))
    g = tid < 0
    ax.scatter(x[g], y[g], s=0.4, c=GREY, marker=".", linewidths=0, rasterized=True)
    t = ~g
    ax.scatter(
        x[t], y[t], s=0.9, c=tree_colors(tid[t]), marker=".", linewidths=0, rasterized=True
    )
    ax.set_xlim(origin[0], origin[0] + TILE_M)
    ax.set_ylim(origin[1], origin[1] + TILE_M)
    utm_axes(
        ax,
        f"{tile} - top-down point cloud, colour = predicted tree id\n"
        f"{len(sub):,} of the tile's points shown; grey = ground / no tree",
    )
    scale_bar(ax, 100, "100 m")
    return save(fig, out / "01-tile-topdown.png")


def fig_oblique(d, tile, origin, window, out: Path, n_points=300_000):
    win = d["win"]
    if len(win) > n_points:
        win = win[np.random.default_rng(5).choice(len(win), n_points, replace=False)]
    x, y, z, tid = win[:, 0] + origin[0], win[:, 1] + origin[1], win[:, 2], win[:, 3].astype(int)
    fig = plt.figure(figsize=(10, 8.5))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x, y, z, s=1.1, c=tree_colors(tid), marker=".", linewidths=0, depthshade=False)
    ax.view_init(elev=28, azim=-60)
    dz = float(z.max() - z.min())
    ax.set_box_aspect((1, 1, max(dz / (window[2] - window[0]), 0.25)), zoom=1.18)
    ax.set_xlabel("easting (m)")
    ax.set_ylabel("northing (m)")
    ax.set_zlabel("z (m)")
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(plt.MaxNLocator(4))
    ax.set_title(
        f"{tile} - oblique view of the densest 120 m x 120 m window\n"
        f"{len(win):,} points, colour = predicted tree id, no z exaggeration"
    )
    return save(fig, out / "02-oblique-3d.png")


def fig_strip(d, tile, origin, window, out: Path):
    win = d["win"]
    cx = (window[0] + window[2]) / 2 - origin[0]
    cy = (window[1] + window[3]) / 2 - origin[1]
    m = (
        (win[:, 0] >= cx - 50)
        & (win[:, 0] < cx + 50)
        & (win[:, 1] >= cy - 10)
        & (win[:, 1] < cy + 10)
    )
    s = win[m]
    x, z, tid, sem = s[:, 0] + origin[0], s[:, 2], s[:, 3].astype(int), s[:, 4].astype(int)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, sharey=True)
    axes[0].scatter(x, z, s=2.2, c=tree_colors(tid), marker=".", linewidths=0, rasterized=True)
    axes[0].set_title(
        f"{tile} - side elevation of a 100 m x 20 m strip (y = "
        f"{origin[1] + cy - 10:.0f}-{origin[1] + cy + 10:.0f} m), colour = predicted tree id"
    )
    cols = np.array([SEM_COLORS.get(int(v), "#ff00ff") for v in sem])
    axes[1].scatter(x, z, s=2.2, c=cols, marker=".", linewidths=0, rasterized=True)
    axes[1].set_title("same strip, colour = predicted semantic class")
    axes[1].legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=SEM_COLORS[k], label=SEM_NAMES[k])
            for k in (0, 1, 2)
        ],
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )
    for ax in axes:
        ax.set_ylabel("z (m)")
        ax.set_aspect("equal")
        ax.ticklabel_format(style="plain", useOffset=False)
    axes[1].set_xlabel("easting (m, EPSG:25833)")
    fig.tight_layout()
    return save(fig, out / "03-strip-side.png")


def fig_closeup(d, crowns, tile, origin, window, out: Path, size=60.0):
    win = d["win"]
    cx = (window[0] + window[2]) / 2 - origin[0]
    cy = (window[1] + window[3]) / 2 - origin[1]
    m = (
        (win[:, 0] >= cx - size / 2)
        & (win[:, 0] < cx + size / 2)
        & (win[:, 1] >= cy - size / 2)
        & (win[:, 1] < cy + size / 2)
    )
    s = win[m]
    x, y, tid = s[:, 0] + origin[0], s[:, 1] + origin[1], s[:, 3].astype(int)
    bounds = (x.min(), y.min(), x.max(), y.max())
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(x, y, s=5.0, c=tree_colors(tid), marker=".", linewidths=0, rasterized=True)
    from shapely.geometry import box as shbox

    sel = crowns[crowns.intersects(shbox(*bounds))]
    ax.add_collection(
        LineCollection(crown_polys(sel), colors="black", linewidths=0.7, zorder=4)
    )
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    utm_axes(
        ax,
        f"{tile} - {size:.0f} m x {size:.0f} m close-up, {len(sel)} crown hulls outlined\n"
        f"{len(s):,} points, colour = predicted tree id",
    )
    scale_bar(ax, 10, "10 m")
    return save(fig, out / "04-closeup-crowns.png")


def fig_crowns_map(d, crowns, trees, tile, origin, out: Path):
    chm = d["chm"]
    ls = LightSource(azdeg=315, altdeg=45)
    shade = ls.hillshade(chm, vert_exag=1.0, dx=CELL, dy=CELL)
    ext = (origin[0], origin[0] + TILE_M, origin[1], origin[1] + TILE_M)
    fig, ax = plt.subplots(figsize=(10, 9.4))
    ax.imshow(shade, cmap="gray", extent=ext, origin="lower", vmin=0, vmax=1.1, zorder=1)

    h = trees.set_index("tree_id")["height"]
    heights = crowns["tree_id"].map(h).to_numpy(dtype=float)
    polys, vals = [], []
    for geom, hv in zip(crowns.geometry, heights):
        if geom is None or geom.is_empty or geom.geom_type != "Polygon" or not np.isfinite(hv):
            continue
        polys.append(np.asarray(geom.exterior.coords)[:, :2])
        vals.append(hv)
    norm = Normalize(vmin=0, vmax=float(np.nanpercentile(vals, 99)))
    pc = PolyCollection(
        polys, array=np.asarray(vals), cmap="viridis", norm=norm, lw=0.1,
        edgecolors="none", alpha=0.85, zorder=2,
    )
    ax.add_collection(pc)

    aoi, _ = load_aoi(ext_bounds(ext))
    if aoi is not None and len(aoi):
        ax.add_collection(
            LineCollection(
                [np.asarray(g.exterior.coords)[:, :2] for g in aoi.geometry if g.geom_type == "Polygon"],
                colors="red", linewidths=1.6, zorder=5,
            )
        )
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    utm_axes(
        ax,
        f"{tile} - {len(polys):,} predicted crowns over the 0.5 m CHM hillshade\n"
        "fill = tree height, red = Revier 12 Tegelsee AOI",
    )
    scale_bar(ax, 100, "100 m", color="black")
    fig.colorbar(pc, ax=ax, shrink=0.75, label="tree height (m)")
    return save(fig, out / "05-crowns-chm.png")


def ext_bounds(ext):
    return (ext[0], ext[2], ext[1], ext[3])


def fig_aoi_zoom(paths, crowns, tile, origin, out: Path, size=200.0):
    tile_bounds = (origin[0], origin[1], origin[0] + TILE_M, origin[1] + TILE_M)
    aoi, plots = load_aoi(tile_bounds)
    if aoi is None or not len(aoi):
        print("  no AOI inside this tile; skipping 06")
        return None
    from shapely.geometry import box as shbox

    clip = shbox(*tile_bounds)
    inter = max((g.intersection(clip) for g in aoi.geometry), key=lambda g: g.area)
    # prefer the sample-plot layers as the zoom centre when they fall inside
    ref = None
    for lyr in PLOT_LAYERS:
        if lyr in plots and len(plots[lyr]):
            ref = plots[lyr].union_all().centroid
            break
    c = ref if ref is not None else inter.centroid
    cx = float(np.clip(c.x, tile_bounds[0] + size / 2, tile_bounds[2] - size / 2))
    cy = float(np.clip(c.y, tile_bounds[1] + size / 2, tile_bounds[3] - size / 2))
    b = (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)

    with rasterio.open(paths["instance"]) as r:
        w = window_from_bounds(*b, transform=r.transform)
        inst = r.read(1, window=w)
        wt = r.window_transform(w)
    ext = (
        wt.c,
        wt.c + inst.shape[1] * CELL,
        wt.f - inst.shape[0] * CELL,
        wt.f,
    )
    rgb = tree_colors(inst.ravel()).reshape(inst.shape + (3,))

    fig, ax = plt.subplots(figsize=(9.5, 9))
    ax.imshow(rgb, extent=ext, origin="upper", interpolation="nearest", zorder=1)
    sel = crowns[crowns.intersects(shbox(*b))]
    ax.add_collection(
        LineCollection(crown_polys(sel), colors="black", linewidths=0.6, zorder=3)
    )
    ax.add_collection(
        LineCollection(
            [np.asarray(g.exterior.coords)[:, :2] for g in aoi.geometry if g.geom_type == "Polygon"],
            colors="red", linewidths=2.0, zorder=5,
        )
    )
    for lyr, gdf in plots.items():
        ax.add_collection(
            LineCollection(
                [
                    np.asarray(g.exterior.coords)[:, :2]
                    for g in gdf.geometry
                    if g.geom_type == "Polygon"
                ],
                colors="white", linewidths=0.9, zorder=4,
            )
        )
    ax.set_xlim(b[0], b[2])
    ax.set_ylim(b[1], b[3])
    utm_axes(
        ax,
        f"{tile} - {size:.0f} m x {size:.0f} m at the Revier 12 Tegelsee AOI\n"
        f"background = 0.5 m instance raster, black = crown hulls ({len(sel)}), "
        "red = AOI, white = reference trees",
    )
    scale_bar(ax, 25, "25 m", color="black")
    return save(fig, out / "06-aoi-zoom.png")


def fig_hist(trees, report, tile, out: Path):
    h = trees["height"].to_numpy()
    a = trees["crown_area_m2"].to_numpy()
    base = report.get("chm_baseline_count")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(h, bins=np.arange(np.floor(h.min()), h.max() + 1, 1.0), color="#3b7dd8")
    axes[0].axvline(np.median(h), color="red", lw=1.2, label=f"median {np.median(h):.1f} m")
    axes[0].set_xlabel("tree height (m)")
    axes[0].set_ylabel("trees")
    axes[0].legend(fontsize=8)
    axes[0].set_title("height distribution")
    axes[1].hist(a, bins=np.arange(0, np.percentile(a, 99.5) + 2, 2.0), color="#2f8f3e")
    axes[1].axvline(np.median(a), color="red", lw=1.2, label=f"median {np.median(a):.1f} m2")
    axes[1].set_xlabel("crown area (m2, convex hull)")
    axes[1].set_ylabel("trees")
    axes[1].legend(fontsize=8)
    axes[1].set_title("crown area distribution")
    fig.suptitle(
        f"{tile} - {len(trees):,} predicted trees vs {base:,} CHM local-maxima baseline trees"
        if base
        else f"{tile} - {len(trees):,} predicted trees"
    )
    fig.tight_layout()
    return save(fig, out / "07-histograms.png")


def fig_overview(found: dict[str, Path], out: Path):
    rows, xs, ys, hs = [], [], [], []
    for tile, d in sorted(found.items()):
        t = gpd.read_file(d / f"{tile}_trees.gpkg", layer="trees")
        rp = d / f"{tile}_report.json"
        rep = json.loads(rp.read_text()) if rp.exists() else {}
        xs.append(t["x"].to_numpy())
        ys.append(t["y"].to_numpy())
        hs.append(t["height"].to_numpy())
        rows.append(
            {
                "tile": tile,
                "n_points": rep.get("n_points"),
                "n_trees": len(t),
                "chm_baseline": rep.get("chm_baseline_count"),
                "median_height": float(np.median(t["height"])),
                "median_crown_area": float(np.median(t["crown_area_m2"])),
                "ground_agreement": rep.get("ground_vs_vegetation_agreement"),
            }
        )
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    h = np.concatenate(hs)
    fig, ax = plt.subplots(figsize=(10, 9))
    sc = ax.scatter(
        x, y, c=h, s=1.3, cmap="viridis", vmin=0, vmax=float(np.percentile(h, 99)),
        marker=".", linewidths=0, rasterized=True,
    )
    for tile in sorted(found):
        ox, oy = tile_origin(tile)
        ax.add_patch(
            plt.Rectangle((ox, oy), TILE_M, TILE_M, fill=False, ec="black", lw=0.9, zorder=3)
        )
        ax.text(
            ox + TILE_M / 2, oy + TILE_M - 40, tile.replace("3dm_33_", "").replace("_1_be", ""),
            ha="center", va="top", fontsize=9, zorder=4,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5),
        )
    aoi, _ = load_aoi()
    if aoi is not None and len(aoi):
        ax.add_collection(
            LineCollection(
                [np.asarray(g.exterior.coords)[:, :2] for g in aoi.geometry if g.geom_type == "Polygon"],
                colors="red", linewidths=1.5, zorder=5,
            )
        )
    utm_axes(
        ax,
        f"Berlin ALS 2021 - {len(found)} km tiles, {len(x):,} predicted tree centroids\n"
        "colour = tree height, red = Revier 12 Tegelsee AOI",
    )
    fig.colorbar(sc, ax=ax, shrink=0.75, label="tree height (m)")
    scale_bar(ax, 500, "500 m")
    save(fig, out / "08-tiles-overview.png")
    return rows


def markdown_table(rows) -> str:
    head = (
        "| Tile | Points | Trees (model) | CHM baseline | Median height (m) | "
        "Median crown area (m2) | Ground/veg agreement |\n|---|---:|---:|---:|---:|---:|---:|"
    )
    out = [head]
    for r in rows:
        ag = r["ground_agreement"]
        out.append(
            f"| {r['tile']} | {r['n_points']:,} | {r['n_trees']:,} | "
            f"{r['chm_baseline']:,} | {r['median_height']:.1f} | "
            f"{r['median_crown_area']:.1f} | {ag * 100:.1f} % |"
            if r["n_points"] and r["chm_baseline"] and ag is not None
            else f"| {r['tile']} | - | {r['n_trees']:,} | - | {r['median_height']:.1f} | "
            f"{r['median_crown_area']:.1f} | - |"
        )
    return "\n".join(out)


# --------------------------------------------------------------------------
def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tile", default="3dm_33_381_5829_1_be")
    ap.add_argument(
        "--data-dir",
        action="append",
        default=None,
        help="directory holding <tile>/ subdirectories; repeatable",
    )
    ap.add_argument("--out-dir", default=str(repo / "docs/benchmarks/assets/berlin-visual"))
    ap.add_argument("--cache-dir", default=os.environ.get("TMPDIR", "/tmp") + "/ff3d_berlin_viz")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--sub-points", type=int, default=1_500_000)
    ap.add_argument("--window", type=float, default=120.0)
    ap.add_argument("--only", default=None, help="comma-separated figure numbers, e.g. 5,6")
    args = ap.parse_args()

    data_dirs = [
        Path(p).expanduser()
        for p in (
            args.data_dir
            or [
                "~/work/hnee/ForestFormer3D_runs/berlin_out/berlin-2021",
                "/Volumes/2TB/winmol/ALS_Data/berlin_als_2021_ff3d",
            ]
        )
    ]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    only = {int(v) for v in args.only.split(",")} if args.only else set(range(1, 9))

    tile = args.tile
    tdir = tile_dir(data_dirs, tile)
    paths = {
        "las": tdir / f"{tile}.las",
        "trees": tdir / f"{tile}_trees.gpkg",
        "crowns": tdir / f"{tile}_crowns.gpkg",
        "instance": tdir / f"{tile}_instance_50cm.tif",
        "report": tdir / f"{tile}_report.json",
    }
    origin = tile_origin(tile)
    print(f"tile {tile} at {origin} from {tdir}")

    trees = gpd.read_file(paths["trees"], layer="trees")
    report = json.loads(paths["report"].read_text()) if paths["report"].exists() else {}
    window = densest_window(trees, origin, args.window)
    print(f"  densest {args.window:.0f} m window: {window[:4]} with {window[4]} trees")

    need_las = bool(only & {1, 2, 3, 4, 5})
    d = None
    if need_las:
        cache = (
            None
            if args.no_cache
            else Path(args.cache_dir) / f"{tile}_{int(args.sub_points)}_{int(window[0])}_{int(window[1])}.npz"
        )
        d = read_las(paths["las"], window[:4], args.sub_points, origin, cache)

    crowns = None
    if only & {4, 5, 6}:
        crowns = gpd.read_file(paths["crowns"], layer="crowns")

    if 1 in only:
        fig_topdown(d, tile, origin, out)
    if 2 in only:
        fig_oblique(d, tile, origin, window, out)
    if 3 in only:
        fig_strip(d, tile, origin, window, out)
    if 4 in only:
        fig_closeup(d, crowns, tile, origin, window, out)
    if 5 in only:
        fig_crowns_map(d, crowns, trees, tile, origin, out)
    if 6 in only:
        fig_aoi_zoom(paths, crowns, tile, origin, out)
    if 7 in only:
        fig_hist(trees, report, tile, out)
    if 8 in only:
        rows = fig_overview(find_tiles(data_dirs), out)
        print()
        print(markdown_table(rows))


if __name__ == "__main__":
    main()
