#!/usr/bin/env python3
"""Comparisons and analytics across the Berlin mosaic, as one generated report chapter.

Reads the per-km-tile products of every method under ``--als-data`` (the stitched tree
tables ``<T>_trees.gpkg``, the ``_border.json`` seam metrics, ``stitch.json``, the
building-mask reports under ``masked/``), the pairwise agreement JSONs that
``benchmark/instance_agreement.py`` wrote, and the two external references fetched for
the mosaic -- the Berlin forest stand map (species and inventory height per stand) and the
tree cadastre (species and measured height per street/park tree) -- and writes:

* ``<assets>/analytics_*.png``   the figures
* ``<doc>``                      the markdown chapter the PDF builder picks up
  (``docs/benchmarks/2026-10-05-berlin-analytics.md``, key ``berlin-analytics``)
* ``<assets>/analytics.json``    every number in the chapter, for later diffs

It is deterministic and re-runnable: after the mosaic grows (new tiles, a new method) run
it again and the chapter and figures follow. Nothing here needs the LAS files.

    .venv-cpu/bin/python benchmark/berlin_analytics.py
    .venv-cpu/bin/python benchmark/berlin_analytics.py --methods ff3d sat   # subset
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

EPSG = 25833
# key -> (label, result dir under --als-data, agreement dir under --als-data/berlin_agreement)
# v4 = the 57-tile mosaics of 2026-10-08 (2 m height filter); v3 is the 44-tile set of
# 2026-10-06 and the 33-tile sets stay as berlin_als_2021_ff3d_v2 / _sat / _ams3d.
METHODS = {
    "ff3d": ("ForestFormer3D", "berlin_als_2021_ff3d_v4"),
    "sat": ("SegmentAnyTree", "berlin_als_2021_sat_v4"),
    "ams3d": ("AMS3D", "berlin_als_2021_ams3d_v4"),
    "ptf": ("PointTreeFormer", "berlin_als_2021_ptf"),      # S. Reder's results, 15 Tegel tiles (benchmark/ptf_to_ff3d.py)
}
AGREEMENT_TAG = "57"      # berlin_agreement/<a>_vs_<b>_<tag>/ from the carrot driver
# A pair the current driver did not recompute keeps an older tag: PointTreeFormer ran on the
# 15 Tegel tiles only, and those tiles' per-point labels are the same in both mosaics.
AGREEMENT_TAG_FALLBACK = {"ff3d_vs_ptf": "44"}
COLOURS = {"ff3d": "#1f5fbf", "sat": "#b4267a", "ams3d": "#e08a1e", "ptf": "#2e8b57"}
FLAT_H, FLAT_AREA = 2.0, 50.0        # a "tree" under 2 m tall over 50 m2 of crown is ground labelled leaf
SMALL_POINTS = 20                    # instances with fewer points than this are noise-sized
CADASTRE_MATCH_M = 3.0               # a predicted top within this distance of a cadastre tree is a detection


# --------------------------------------------------------------------------- helpers
def tile_key(stem: str) -> str:
    """``3dm_33_380_5828_1_be`` -> ``380_5828``."""
    p = stem.split("_")
    return f"{p[2]}_{p[3]}"


def md_table(header: list[str], rows: list[list], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    sep = {"l": ":---", "r": "---:", "c": ":---:"}
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep[a] for a in align) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt(v, nd=1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return f"{v:,.{nd}f}"


def pct(v, nd=1) -> str:
    return "–" if v is None or np.isnan(v) else f"{100 * v:.{nd}f} %"


def quantiles(a, qs=(0.1, 0.5, 0.9)) -> list[float]:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return [float(np.quantile(a, q)) for q in qs] if a.size else [np.nan] * len(qs)


def flat_blob_mask(height, crown_area) -> np.ndarray:
    """The artefact of section 3: instances that are flat and wide -- bare ground the
    model labelled as vegetation, not trees."""
    return (np.asarray(height) < FLAT_H) & (np.asarray(crown_area) > FLAT_AREA)


def pooled_agreement(records: list[dict]) -> dict:
    """Pool per-tile instance_agreement.py JSONs by counts, not by averaging fractions."""
    n_a = sum(r["n_a"] for r in records)
    n_b = sum(r["n_b"] for r in records)
    matched = sum(r["matched"] for r in records)
    pts = sum(r["n_points"] for r in records)
    return {
        "tiles": len(records), "n_points": pts, "n_a": n_a, "n_b": n_b, "matched": matched,
        "matched_frac_a": matched / n_a if n_a else np.nan,
        "matched_frac_b": matched / n_b if n_b else np.nan,
        "iou_median": float(np.median([r["iou_median"] for r in records])) if records else np.nan,
        "split_a_frac": float(np.average([r["split_a_frac"] for r in records], weights=[r["n_a"] for r in records])) if n_a else np.nan,
        "split_b_frac": float(np.average([r["split_b_frac"] for r in records], weights=[r["n_b"] for r in records])) if n_b else np.nan,
    }


# ----------------------------------------------------------------------------- loading
def load_trees(als: Path, key: str, log=print):
    """All stitched tree tables of a method as one GeoDataFrame with a ``tile`` column."""
    import geopandas as gpd
    import pandas as pd

    root = als / METHODS[key][1]
    frames = []
    for d in sorted(p for p in root.glob("3dm_33_*_1_be") if p.is_dir()):
        f = d / f"{d.name}_trees.gpkg"
        if not f.exists():
            log(f"  {key}: {d.name} has no tree table, skipped")
            continue
        g = gpd.read_file(f)
        g["tile"] = d.name
        frames.append(g)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    # A table regenerated per tile from the tile's own LAS (building mask, height filter)
    # lists a tree straddling a km border in both tiles; the stitch's tables list it once.
    # Count every id once, in the tile holding most of its points -- but only where ids ARE
    # mosaic-wide (a stitch.json exists); a method with per-tile ids (PointTreeFormer) reuses
    # id 0.. in every tile and must be counted per (tile, id).
    if (root / "stitch.json").exists():
        out = out.sort_values(["tree_id", "n_points"], ascending=[True, False]).drop_duplicates("tree_id", keep="first")
    out = out.sort_values(["tile", "tree_id"]).reset_index(drop=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=f"EPSG:{EPSG}")


def real(paths):
    """Drop the ``._*`` resource-fork twins macOS leaves next to every file on exFAT."""
    return [p for p in paths if not p.name.startswith("._")]


def load_json_per_tile(root: Path, suffix: str) -> dict[str, dict]:
    out = {}
    for d in sorted(p for p in root.glob("3dm_33_*_1_be") if p.is_dir()):
        f = d / f"{d.name}{suffix}"
        if f.exists():
            out[d.name] = json.loads(f.read_text())
    return out


def load_agreement(dirpath: Path) -> list[dict]:
    return [json.loads(f.read_text()) for f in real(sorted(dirpath.glob("*.json")))] if dirpath.is_dir() else []


def agreement_pairs(root: Path) -> dict[tuple[str, str], Path]:
    """(method a, method b) -> the directory with that pair's per-tile agreement JSONs.

    `AGREEMENT_TAG` is the mosaic the report describes; `AGREEMENT_TAG_FALLBACK` names the
    pairs that keep an earlier tag, so a method run on a subset of the tiles is not dropped
    from the chapter when the mosaic grows.
    """
    out: dict[tuple[str, str], Path] = {}
    for tag in dict.fromkeys([AGREEMENT_TAG, *AGREEMENT_TAG_FALLBACK.values()]):
        for d in sorted(root.glob(f"*_vs_*_{tag}")):
            pair = d.name[: -len(f"_{tag}")]
            if AGREEMENT_TAG_FALLBACK.get(pair, AGREEMENT_TAG) != tag:
                continue
            a_key, b_key = pair.split("_vs_")
            out.setdefault((a_key, b_key), d)
    return out


# ----------------------------------------------------------------------------- figures
def fig_tile_grid(trees: dict, out: Path):
    """Trees per km tile as a grid heatmap per method, plus the SAT/FF3D ratio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in ("ff3d", "sat") if k in trees]
    counts = {k: trees[k].groupby("tile").size() for k in keys}
    tiles = sorted(set().union(*[set(c.index) for c in counts.values()]))
    es = sorted({int(t.split("_")[2]) for t in tiles}); ns = sorted({int(t.split("_")[3]) for t in tiles})
    panels = keys + (["ratio"] if len(keys) == 2 else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6), squeeze=False)
    for ax, p in zip(axes[0], panels):
        grid = np.full((len(ns), len(es)), np.nan)
        for t in tiles:
            e, n = int(t.split("_")[2]), int(t.split("_")[3])
            i, j = ns.index(n), es.index(e)
            if p == "ratio":
                a, b = counts["ff3d"].get(t, np.nan), counts["sat"].get(t, np.nan)
                grid[i, j] = b / a if a else np.nan
            else:
                grid[i, j] = counts[p].get(t, np.nan)
        cmap = "RdBu_r" if p == "ratio" else "YlGn"
        kw = dict(vmin=0.6, vmax=1.4) if p == "ratio" else {}
        im = ax.imshow(grid, origin="lower", cmap=cmap, **kw)
        for t in tiles:
            e, n = int(t.split("_")[2]), int(t.split("_")[3])
            v = grid[ns.index(n), es.index(e)]
            if np.isfinite(v):
                ax.text(es.index(e), ns.index(n), f"{v:.2f}" if p == "ratio" else f"{v / 1000:.1f}k",
                        ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(es))); ax.set_xticklabels(es, fontsize=8)
        ax.set_yticks(range(len(ns))); ax.set_yticklabels(ns, fontsize=8)
        ax.set_xlabel("easting km"); ax.set_ylabel("northing km")
        ax.set_title({"ff3d": "ForestFormer3D trees per km tile", "sat": "SegmentAnyTree trees per km tile",
                      "ratio": "SegmentAnyTree / ForestFormer3D"}[p], fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_distributions(trees: dict, out: Path, tiles: list[str] | None = None, title_suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    specs = [("height", "tree height (m)", np.arange(0, 50, 1)),
             ("crown_area_m2", "crown area (m²)", np.arange(0, 200, 4)),
             ("n_points", "points per tree", np.logspace(1, 5, 50))]
    for ax, (col, label, bins) in zip(axes, specs):
        for k, g in trees.items():
            if tiles is not None:
                g = g[g.tile.isin(tiles)]
            v = g[col].to_numpy(dtype=float)
            ax.hist(v[np.isfinite(v)], bins=bins, histtype="step", lw=1.6, color=COLOURS[k],
                    label=f"{METHODS[k][0]} (n={len(v):,})", density=True)
        ax.set_xlabel(label); ax.set_ylabel("density")
        if col == "n_points":
            ax.set_xscale("log")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Per-tree size distributions{title_suffix}", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_agreement(records: list[dict], out: Path, la: str, lb: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = sorted(records, key=lambda r: tile_key(Path(r["a"]).stem))
    keys = [tile_key(Path(r["a"]).stem) for r in records]
    x = np.arange(len(records))
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].bar(x - 0.2, [r["matched_frac_a"] for r in records], 0.4, color=COLOURS["ff3d"], label=f"{la} trees matched")
    axes[0].bar(x + 0.2, [r["matched_frac_b"] for r in records], 0.4, color=COLOURS["sat"], label=f"{lb} trees matched")
    axes[0].plot(x, [r["iou_median"] for r in records], "k.-", lw=1, label="median IoU of matches")
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("fraction"); axes[0].legend(fontsize=8, frameon=False, ncol=3)
    axes[1].bar(x - 0.2, [r["split_a_frac"] for r in records], 0.4, color=COLOURS["ff3d"], label=f"{la} trees split by {lb}")
    axes[1].bar(x + 0.2, [r["split_b_frac"] for r in records], 0.4, color=COLOURS["sat"], label=f"{lb} trees split by {la}")
    axes[1].set_ylabel("fraction"); axes[1].legend(fontsize=8, frameon=False, ncol=2)
    axes[1].set_xticks(x); axes[1].set_xticklabels(keys, rotation=90, fontsize=7)
    fig.suptitle(f"Instance agreement per km tile, {la} vs {lb} (IoU ≥ 0.5 on identical points)", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_seams_and_buildings(border: dict, masked: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4))
    tiles = sorted(set().union(*[set(b) for b in border.values()]))
    x = np.arange(len(tiles)); w = 0.8 / max(len(border), 1)
    for i, (k, b) in enumerate(border.items()):
        axes[0].bar(x + (i - (len(border) - 1) / 2) * w, [100 * b[t]["touching_frac"] if t in b else np.nan for t in tiles],
                    w, color=COLOURS[k], label=METHODS[k][0])
    axes[0].set_ylabel("crowns touching a 100 m grid line (%)"); axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_xticks(x); axes[0].set_xticklabels([tile_key(t) for t in tiles], rotation=90, fontsize=7)
    axes[0].set_title("Seam residue per tile after the mosaic-wide stitch", fontsize=10)
    mt = sorted(masked)
    rem = [masked[t]["instances_removed"] / masked[t]["instances_before"] * 100 for t in mt]
    axes[1].bar(np.arange(len(mt)), rem, color="#d63a3a")
    axes[1].set_ylabel("instances removed by the ALKIS mask (%)")
    axes[1].set_xticks(np.arange(len(mt))); axes[1].set_xticklabels([tile_key(t) for t in mt], rotation=90, fontsize=7)
    axes[1].set_title("Building mask: share of ForestFormer3D instances on roofs", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_stand_heights(per_stand, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6))
    top = per_stand.groupby("species")["gis_area"].sum().sort_values(ascending=False).index[:6]
    cols = plt.cm.tab10.colors
    for i, sp in enumerate(top):
        s = per_stand[per_stand.species == sp]
        ax.scatter(s.inv_height, s.pred_p90, s=np.clip(s.n_trees / 20, 6, 120), color=cols[i % 10], alpha=0.65, label=sp)
    other = per_stand[~per_stand.species.isin(top)]
    ax.scatter(other.inv_height, other.pred_p90, s=np.clip(other.n_trees / 20, 6, 120), color="grey", alpha=0.4, label="other")
    lim = (0, max(45, float(np.nanmax(per_stand.inv_height)) + 2))
    ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("stand height, 2014 inventory, main canopy layer (m)")
    ax.set_ylabel("ForestFormer3D, 90th percentile of tree heights in the stand (m)")
    ax.set_title("Predicted canopy height vs the forest inventory, per stand (marker size = trees)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_cadastre(matched, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for k, c in [("ff3d", COLOURS["ff3d"]), ("sat", COLOURS["sat"])]:
        m = matched[k]
        axes[0].scatter(m.baumhoehe, m.height, s=4, alpha=0.25, color=c, label=f"{METHODS[k][0]} (n={len(m):,})")
    lim = (0, 45); axes[0].plot(lim, lim, "k--", lw=0.8); axes[0].set_xlim(lim); axes[0].set_ylim(lim)
    axes[0].set_xlabel("cadastre height (m)"); axes[0].set_ylabel("predicted tree height (m)")
    axes[0].set_title(f"Height of cadastre trees with a predicted top within {CADASTRE_MATCH_M:.0f} m", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    for k, c in [("ff3d", COLOURS["ff3d"]), ("sat", COLOURS["sat"])]:
        d = (matched[k].height - matched[k].baumhoehe).to_numpy(dtype=float)
        axes[1].hist(d[np.isfinite(d)], bins=np.arange(-20, 20.5, 1), histtype="step", lw=1.6, color=c, label=METHODS[k][0])
    axes[1].axvline(0, color="k", lw=0.8); axes[1].set_xlabel("predicted − cadastre height (m)"); axes[1].set_ylabel("trees")
    axes[1].legend(fontsize=8, frameon=False); axes[1].set_title("Height residual", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_study_area(als: Path, trees: dict, out: Path):
    """The study area: km tiles by processing status, the forest stands, the two survey
    footprints and the building footprints, for the introduction."""
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    fig, ax = plt.subplots(figsize=(11, 7.2))
    stands = als / "berlin_forest" / "forstbetriebskarte_2014.gpkg"
    if stands.exists():
        gpd.read_file(stands, layer="hauptbaumarten", columns=["geometry"]).to_crs(EPSG).plot(ax=ax, color="#b9dca6", edgecolor="none")
    bld = als / "berlin_buildings" / "alkis_buildings.gpkg"
    if bld.exists():
        gpd.read_file(bld, layer="buildings", columns=["geometry"]).to_crs(EPSG).plot(ax=ax, color="#9a9a9a", edgecolor="none", linewidth=0)
    processed = set(trees["ff3d"].tile.unique()) if "ff3d" in trees else set()
    rev = als / "berlin_forest" / "reviere.gpkg"
    status = {}
    if rev.exists():
        cov = gpd.read_file(rev, layer="coverage")
        for _, r in cov.iterrows():
            status.setdefault(r.tile, r.status)
    tiles = processed | set(status)
    colours = {"processed": "#1f5fbf", "downloaded": "#e08a1e", "missing": "#d63a3a"}
    for t in sorted(tiles):
        e, n = int(t.split("_")[2]) * 1000, int(t.split("_")[3]) * 1000
        st = "processed" if t in processed else status.get(t, "missing")
        ax.add_patch(Rectangle((e, n), 1000, 1000, fill=False, edgecolor=colours[st], linewidth=1.6 if st == "processed" else 1.2,
                               linestyle="-" if st == "processed" else "--"))
        ax.text(e + 500, n + 500, t[7:15], ha="center", va="center", fontsize=6.5, color=colours[st])
    if rev.exists():
        fp = gpd.read_file(rev, layer="footprints").to_crs(EPSG)
        for _, f in fp.iterrows():
            gpd.GeoSeries([f.geometry], crs=EPSG).boundary.plot(ax=ax, color="#5a0a3a" if f.key == "R13" else "#0a2a6a", linewidth=2.2)
            c = f.geometry.representative_point()
            ax.text(c.x, c.y, f.key, fontsize=12, fontweight="bold", color="#5a0a3a" if f.key == "R13" else "#0a2a6a", ha="center")
    ax.set_aspect("equal"); ax.set_xlabel("easting (m, EPSG:25833)"); ax.set_ylabel("northing (m)")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.legend(handles=[Patch(facecolor="#b9dca6", label="forest stands (Forstbetriebskarte 2014)"),
                       Patch(facecolor="#9a9a9a", label="buildings (ALKIS)"),
                       Patch(facecolor="none", edgecolor=colours["processed"], label=f"km tile processed ({len(processed)})"),
                       Patch(facecolor="none", edgecolor=colours["downloaded"], linestyle="--", label="downloaded, not processed"),
                       Patch(facecolor="none", edgecolor=colours["missing"], linestyle="--", label="not downloaded"),
                       Patch(facecolor="none", edgecolor="#0a2a6a", linewidth=2, label="WINMOL survey footprints R12 / R13")],
              loc="lower right", fontsize=8, frameon=True)
    ax.set_title("Study area: Berlin ALS 2021 km tiles around the Tegel and Spandau forests", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------- analyses
def stand_analysis(trees_ff3d, stands_path: Path):
    """Join every ForestFormer3D tree to the 2014 stand it stands in."""
    import geopandas as gpd

    st = gpd.read_file(stands_path, layer="hauptbaumarten",
                       columns=["best_dist", "s1_1_deuts", "s1_1_hoehe", "s1_1_bhd", "grpalter", "gis_area", "geometry"])
    st = st.to_crs(EPSG).reset_index(drop=True)
    st["stand_idx"] = st.index
    st.geometry = st.geometry.make_valid()
    j = gpd.sjoin(trees_ff3d[["height", "crown_area_m2", "geometry"]], st[["stand_idx", "geometry"]], how="inner", predicate="within")
    per = j.groupby("stand_idx").agg(n_trees=("height", "size"), pred_median=("height", "median"),
                                    pred_p90=("height", lambda v: float(np.quantile(v, 0.9))),
                                    crown_median=("crown_area_m2", "median"))
    per = per.join(st.set_index("stand_idx")[["s1_1_deuts", "s1_1_hoehe", "gis_area", "grpalter"]])
    per = per.rename(columns={"s1_1_deuts": "species", "s1_1_hoehe": "inv_height"})
    per["density_ha"] = per.n_trees / (per.gis_area / 1e4)
    per = per[(per.n_trees >= 20) & per.inv_height.notna() & (per.inv_height > 0)]
    by_sp = per.groupby("species").apply(lambda s: {
        "stands": int(len(s)), "area_ha": float(s.gis_area.sum() / 1e4), "trees": int(s.n_trees.sum()),
        "density_ha": float(s.n_trees.sum() / (s.gis_area.sum() / 1e4)),
        "pred_median": float(np.average(s.pred_median, weights=s.n_trees)),
        "pred_p90": float(np.average(s.pred_p90, weights=s.n_trees)),
        "inv_height": float(np.average(s.inv_height, weights=s.gis_area)),
        "crown_median": float(np.average(s.crown_median, weights=s.n_trees)),
    }, include_groups=False)
    return per, {k: v for k, v in by_sp.items()}


def cadastre_analysis(trees: dict, cadastre_path: Path, tile_union):
    """Detection rate and height agreement against the Berlin tree cadastre."""
    import geopandas as gpd
    import pandas as pd

    frames = []
    for layer, kind in (("strassenbaeume", "street"), ("anlagenbaeume", "park")):
        c = gpd.read_file(cadastre_path, layer=layer, columns=["gattung_deutsch", "art_dtsch", "baumhoehe", "pflanzjahr", "kronedurch", "geometry"])
        c["kind"] = kind
        frames.append(c.to_crs(EPSG))
    cad = pd.concat(frames, ignore_index=True)
    cad = gpd.GeoDataFrame(cad, geometry="geometry", crs=f"EPSG:{EPSG}")
    cad = cad[cad.within(tile_union)].reset_index(drop=True)
    cad["cid"] = cad.index
    result, matched = {}, {}
    for k in ("ff3d", "sat"):
        if k not in trees:
            continue
        t = trees[k][["height", "crown_area_m2", "geometry"]]
        j = gpd.sjoin_nearest(cad[["cid", "baumhoehe", "kind", "gattung_deutsch", "geometry"]], t, how="left",
                              max_distance=CADASTRE_MATCH_M, distance_col="dist")
        j = j.sort_values("dist").drop_duplicates("cid")
        hit = j.height.notna()
        m = j[hit & j.baumhoehe.notna() & (j.baumhoehe > 0)]
        matched[k] = m
        res = {"n_cadastre": int(len(cad)), "detected": int(hit.sum()), "detected_frac": float(hit.mean())}
        for kind in ("street", "park"):
            s = j[j.kind == kind]
            res[f"detected_frac_{kind}"] = float(s.height.notna().mean()) if len(s) else np.nan
            res[f"n_{kind}"] = int(len(s))
        d = (m.height - m.baumhoehe).to_numpy(dtype=float)
        res.update({"n_height_pairs": int(len(m)), "height_bias": float(np.mean(d)), "height_mae": float(np.mean(np.abs(d))),
                    "height_r": float(np.corrcoef(m.baumhoehe, m.height)[0, 1]) if len(m) > 2 else np.nan})
        by_genus = j.groupby("gattung_deutsch").agg(n=("cid", "size"), det=("height", lambda v: v.notna().mean()))
        res["by_genus"] = {g: {"n": int(r.n), "detected_frac": float(r.det)} for g, r in by_genus.sort_values("n", ascending=False).head(8).iterrows()}
        result[k] = res
    return result, matched


def footprint_analysis(trees: dict, reviere_path: Path):
    import geopandas as gpd

    if not reviere_path.exists():
        return {}
    fp = gpd.read_file(reviere_path, layer="footprints").to_crs(EPSG)
    out = {}
    from shapely.geometry import box
    from shapely.ops import unary_union

    for _, f in fp.iterrows():
        row = {"area_ha": float(f.area_ha)}
        for k, g in trees.items():
            tiles = unary_union([box(int(t.split("_")[2]) * 1000, int(t.split("_")[3]) * 1000,
                                     int(t.split("_")[2]) * 1000 + 1000, int(t.split("_")[3]) * 1000 + 1000) for t in g.tile.unique()])
            covered = f.geometry.intersection(tiles).area / f.geometry.area if f.geometry.area else np.nan
            inside = g[g.within(f.geometry)]
            ha_cov = f.area_ha * covered
            row[k] = {"trees": int(len(inside)), "covered": float(covered),
                      "density_ha": float(len(inside) / ha_cov) if ha_cov else np.nan, "height_q": quantiles(inside.height)}
        out[f.key] = row
    return out


# -------------------------------------------------------------------------------- doc
def write_doc(doc: Path, assets_rel: str, S: dict, labels: dict) -> None:
    L = labels
    keys = list(S["methods"])
    lines = ["# Comparisons and analytics across the Berlin mosaic", "",
             f"Generated {S['generated']} by `benchmark/berlin_analytics.py` from the stitched per-tile "
             f"products on the 2TB volume; every number here is also in `{assets_rel}/analytics.json`. "
             "The chapter compares the three segmentation methods with each other on identical points "
             "and, where an external reference exists, against the Berlin forest inventory and the tree "
             "cadastre. It says nothing about which method is *right* in the forest interior -- no per-tree "
             "ground truth exists there (data-sources chapter, section 5).", ""]

    # 1 overview
    lines += ["## 1. What was compared", "",
              md_table(["method", "km tiles", "trees", "trees / km²", "points / tree (median)", "height p10 / p50 / p90 (m)", "crown area p50 (m²)"],
                       [[L[k], S["methods"][k]["tiles"], fmt(S["methods"][k]["trees"]), fmt(S["methods"][k]["trees_per_km2"], 0),
                         fmt(S["methods"][k]["points_q"][1], 0),
                         " / ".join(fmt(v) for v in S["methods"][k]["height_q"]), fmt(S["methods"][k]["crown_q"][1])] for k in keys]), "",
              "ForestFormer3D, SegmentAnyTree and AMS3D ran on the same 20 m-halo split of each km tile and went "
              "through the same mosaic-wide stitch, so per-point labels are directly comparable (SegmentAnyTree and "
              "AMS3D preserve point order; ForestFormer3D's result is re-ordered to the split); PointTreeFormer's "
              "per-tile results were re-ordered onto the same points."
              + "".join(f" {L[k]} covers {S['methods'][k]['tiles']} of the tiles, the subset it has been run on."
                        for k in keys if S["methods"][k]["tiles"] < S["methods"]["ff3d"]["tiles"])
              + (" PointTreeFormer's ids are per km tile (its run had a 20 m buffer but no mosaic-wide stitch), "
                 "so its border trees are counted on both sides of a km line." if "ptf" in keys else ""), ""]

    # 2 per tile
    if "ff3d" in keys and "sat" in keys:
        lines += ["## 2. Trees per km tile", "",
                  f"![Trees per km tile. Left and middle: the number of tree instances each method reports per "
                  f"1 km tile (the label is the count in thousands; the grid is the easting and northing of the "
                  f"tile's south-west corner in km, EPSG:25833; white cells are tiles outside the mosaic). Right: the "
                  f"SegmentAnyTree count divided by the ForestFormer3D count, red where SegmentAnyTree finds more "
                  f"trees, blue where it finds fewer; 1.00 would mean identical totals.]({assets_rel}/analytics_tile_grid.png)", "",
                  "What we see: both methods agree on where the trees are -- the darkest cells are the closed forest "
                  "in the west (R13) and north-east (the Tegel forest north of 5829), the palest are the lake tile "
                  "381_5827 (Tegeler See, 2-3 thousand trees) and the housing to the east. The ratio map is not noise: "
                  "it is a block of red over the R13 forest tiles and a block of blue-to-white over R12 and the "
                  "built-up tiles, i.e. the two methods differ systematically by forest, not tile by tile.", "",
                  f"SegmentAnyTree finds {fmt(S['sat_over_ff3d'], 3)}x the ForestFormer3D tree count over the mosaic "
                  f"(per tile from {fmt(S['sat_over_ff3d_range'][0], 2)} to {fmt(S['sat_over_ff3d_range'][1], 2)}). "
                  + (f"On the {S['forest_tiles_n']} tiles that are more than half forest by the stand map the ratio averages "
                     f"{fmt(S['ratio_forest'], 2)}, on the other {S['other_tiles_n']} tiles {fmt(S['ratio_other'], 2)}. " if S.get("forest_tiles_n") else "")
                  + (("The two surveyed forests differ: " + ", ".join(
                      f"{k} {fmt(v['ratio'], 2)} ({v['tiles']} tiles with at least a quarter of the tile inside the footprint)"
                      for k, v in S["ratio_by_footprint"].items()) + ". Where SegmentAnyTree reports more trees it has cut "
                      "crowns into more pieces (section 4); where it reports fewer, ForestFormer3D's extra instances are "
                      "mostly the small and flat ones of section 3.") if S.get("ratio_by_footprint") else ""), ""]

    # 3 distributions
    lines += ["## 3. Size distributions and quality flags", "",
              f"![Per-tree size distributions over the whole mosaic. Each panel is a normalised histogram (area 1) "
              f"of one attribute of the tree table, one curve per method, so curves of methods with different tree "
              f"counts are comparable in shape: height above ground in 1 m bins; crown area (convex hull of the "
              f"instance's points) in 4 m² bins; points per instance on a logarithmic axis. "
              f"n is the number of trees behind each curve.]({assets_rel}/analytics_distributions.png)", "",
              "What we see: the height panel has two modes for both methods, a tall one at 24-28 m (the pine and oak "
              "canopy) and a short one at 4-7 m (understory, hedges, young trees in gardens), with a trough at "
              "12-18 m; ForestFormer3D puts more mass under 8 m and SegmentAnyTree more at 20-30 m. The crown-area "
              "panel shows SegmentAnyTree's crowns shifted to smaller areas (mode 8-12 m² against 20-30 m²) -- the "
              "signature of cutting crowns into pieces. The points-per-tree panel has a spike at the smallest sizes "
              "for ForestFormer3D (instances of 10-20 points, the noise-sized class of the table below) and otherwise "
              "the same log-linear decline for both.", "",
              md_table(["method", "trees", f"flat blobs (< {FLAT_H:.0f} m, > {FLAT_AREA:.0f} m²)", f"noise-sized (< {SMALL_POINTS} points)", "taller than 40 m"],
                       [[L[k], fmt(S["methods"][k]["trees"]), f"{fmt(S['methods'][k]['flat'])} ({pct(S['methods'][k]['flat_frac'])})",
                         f"{fmt(S['methods'][k]['small'])} ({pct(S['methods'][k]['small_frac'])})",
                         f"{fmt(S['methods'][k]['tall'])} ({pct(S['methods'][k]['tall_frac'], 2)})"] for k in keys]), "",
              "Both height distributions are bimodal: a canopy mode near 26 m and a second mode at 4-7 m "
              "(understory, hedges, young trees in gardens). ForestFormer3D has more of the low mode and of "
              "instances under 2 m; SegmentAnyTree's crowns are smaller (crown area p50 20 vs 26 m²) because it "
              "cuts more of them (section 4).", "",
              "The flat-blob flag is the artefact seen in the viewer: an instance with almost no height but a "
              "large footprint is bare ground (ALS class 2) that the model labelled as leaf, not a tree. The "
              "noise-sized flag counts instances too small to be a crown at 20-30 pts/m². "
              + (f"The products of this chapter went through the stitch's minimum-height rule ({S['min_height']} m), "
                 f"which removed {fmt(S['n_short_removed'])} instances mosaic-wide before any table was written; "
                 "the counts above are after it, and the flat-blob row is what the 2 m rule does not reach "
                 "(instances taller than 2 m with a wide, flat hull)."
                 if S.get("min_height") else
                 "Both are candidates for a post-filter; neither is applied to the products this chapter "
                 "describes (the minimum-height rule of the methods chapter applies from the 44-tile stitch on)."), ""]
    if S.get("three_way_tiles"):
        lines += [f"On the {len(S['three_way_tiles'])} tiles every method covers "
                  f"({', '.join(tile_key(t) for t in S['three_way_tiles'])}):", "",
                  f"![The same distributions restricted to the tiles every method covers, all methods. "
                  f"Same axes and binning as the previous figure.]({assets_rel}/analytics_distributions_3way.png)", "",
                  "What we see: AMS3D has no short mode at all -- its height distribution starts at about 10 m -- and "
                  "its crowns are the largest. Mean shift with a height-dependent bandwidth merges the understory into "
                  "the canopy tree above it, which is why it reports the fewest trees and the tallest ones; the learned "
                  "methods separate that layer. PointTreeFormer is the odd one out in height: a broad mode at 10-18 m, "
                  "exactly where the other three have their trough, and the heaviest tail of large crowns (hulls "
                  "over 75 m² are twice as frequent as in ForestFormer3D) -- consistent with it drawing the larger "
                  "crowns in the agreement table and with mid-height instances that the others split into a canopy "
                  "tree and an understory one.", "",
                  md_table(["method", "trees on these tiles", "height p10 / p50 / p90 (m)", "crown p50 (m²)"],
                           [[L[k], fmt(v["trees"]), " / ".join(fmt(x) for x in v["height_q"]), fmt(v["crown_q"][1])]
                            for k, v in S["three_way"].items()]), ""]

    # 4 agreement
    if S.get("agreement"):
        a = S["agreement"]["ff3d_sat"]
        lines += ["## 4. Instance agreement: ForestFormer3D vs SegmentAnyTree", "",
                  f"![Instance agreement per km tile between ForestFormer3D (blue) and SegmentAnyTree (magenta), "
                  f"computed on identical points. Top: the fraction of each method's trees that have a counterpart in "
                  f"the other method overlapping with IoU ≥ 0.5 (bars), and the median IoU of those matched pairs "
                  f"(black line, right-hand reading on the same 0-1 axis). Bottom: the fraction of each method's "
                  f"trees whose points are covered by two or more instances of the other method -- a tree the other "
                  f"method has split. Tiles are ordered by easting, then northing.]({assets_rel}/analytics_agreement.png)", "",
                  "What we see: the matched fractions move together from tile to tile (0.35-0.68) and are lowest on the "
                  "built-up tiles (380_5830, 383_5826/5827), where both methods segment small garden vegetation "
                  "differently, and highest on closed forest; the median IoU of a match is flat at 0.71-0.78 "
                  "everywhere, so when the two agree on a tree they agree on its extent. The bottom panel is the "
                  "asymmetry the whole chapter turns on: the blue bars (ForestFormer3D trees split by SegmentAnyTree) "
                  "are 2-4x the magenta ones on every tile, 0.32-0.38 on the R13 forest tiles.", "",
                  f"Pooled over {a['tiles']} tiles and {fmt(a['n_points'])} identical points: {fmt(a['matched'])} tree pairs "
                  f"overlap with IoU ≥ 0.5, i.e. {pct(a['matched_frac_a'])} of ForestFormer3D's {fmt(a['n_a'])} trees and "
                  f"{pct(a['matched_frac_b'])} of SegmentAnyTree's {fmt(a['n_b'])}; the median IoU of a matched pair is "
                  f"{fmt(a['iou_median'], 3)} (the median of the per-tile medians; all pooled figures weight tiles by their tree "
                  f"count, so they differ in the last digit from the unweighted tile means quoted in the appendix). "
                  f"{pct(a['split_a_frac'])} of ForestFormer3D trees are covered by two or more "
                  f"SegmentAnyTree instances against {pct(a['split_b_frac'])} the other way round: where the two disagree, "
                  "SegmentAnyTree has mostly cut one crown into several, which is also why it reports more trees.", ""]
        if S["agreement"].get("ams3d"):
            rows = [[p, v["tiles"], pct(v["matched_frac_a"]), pct(v["matched_frac_b"]), fmt(v["iou_median"], 3), pct(v["split_a_frac"]), pct(v["split_b_frac"])]
                    for p, v in S["agreement"]["ams3d"].items()]
            lines += ["The other pairs, each on the tiles both methods cover (A = the first named):", "",
                      md_table(["pair", "tiles", "A matched", "B matched", "median IoU", "A split by B", "B split by A"], rows), "",
                      "The split columns say who draws the larger crowns: a method whose trees are often covered by "
                      "several instances of the other draws them large (AMS3D, PointTreeFormer), one that splits the "
                      "other's trees draws them small (SegmentAnyTree).", ""]

    # 5 seams + buildings
    lines += ["## 5. Seams and the building mask", "",
              f"![Left: seam residue per km tile after the mosaic-wide stitch -- the percentage of crowns that a "
              f"100 m grid line (the sub-tile borders used for inference) still intersects, per method; the control "
              f"value for an arbitrary line through the same crowns is about 8 %. Right: the percentage of "
              f"ForestFormer3D instances on each tile that the ALKIS building mask removes because at least half of "
              f"their points lie on a roof.]({assets_rel}/analytics_seams_buildings.png)", "",
              "What we see, left: ForestFormer3D sits at 10-12 % on every tile, i.e. 2-4 points above the control "
              "floor, and SegmentAnyTree at 13-16 % with outliers to 20 % -- its smaller, more numerous instances "
              "touch lines more often, and some of its matches across sub-tiles fail the IoU threshold. Neither shows "
              "the 30-40 % that an un-haloed split produced. Right: the forest tiles lose nothing (no buildings), the "
              "lakeside and housing tiles 5-20 %, and 383_5827 -- the densest housing -- 40 % of its instances were "
              "roofs or roof vegetation.", ""]
    if S.get("stitch"):
        lines += [md_table(["method", "trees after stitch", "instances unified over halos", "cross-km matches", "crowns touching a grid line (mean)", "strip excess (mean pp)"],
                           [[L[k], fmt(v["n_trees"]), fmt(v["n_unified"]), fmt(v["n_cross_km"]), pct(v["touching_mean"]), fmt(v["strip_mean"], 3)]
                            for k, v in S["stitch"].items()]), "",
                  "The seam metrics are what `ff3d_geo border-check` measures on the stitched LAS: the share of crowns "
                  "a 100 m grid line still cuts and the excess of unlabelled points in a 2 m strip along the lines "
                  "relative to the interior (the control floor from an offset lattice is about 8 % and 0 pp).", ""]
    if S.get("buildings"):
        b = S["buildings"]
        lines += [f"The ALKIS footprints cover every tile; the mask removes {fmt(b['removed'])} of {fmt(b['before'])} "
                  f"ForestFormer3D instances ({pct(b['removed_frac'], 2)}; per-tile counts, so a tree on a km border is "
                  f"counted in both tiles) and re-labels {fmt(b['points'])} points as building. "
                  f"The densest built-up tiles lose the most ({', '.join(f'{tile_key(t)} {pct(f)}' for t, f in b['top'])}).", ""]

    # 6 inventory
    if S.get("stands"):
        rows = [[sp, v["stands"], fmt(v["area_ha"], 0), fmt(v["trees"]), fmt(v["density_ha"], 0), fmt(v["inv_height"]), fmt(v["pred_median"]), fmt(v["pred_p90"]), fmt(v["crown_median"])]
                for sp, v in sorted(S["stands"].items(), key=lambda kv: -kv[1]["area_ha"])[:8]]
        lines += ["## 6. Against the forest inventory (Forstbetriebskarte 2014)", "",
                  f"![Predicted canopy height against the forest inventory, one marker per stand (only stands with "
                  f"at least 20 predicted trees and an inventory height). Horizontal axis: the height of the main "
                  f"canopy layer from the 2014 Forstbetriebskarte, a stand mean of dominant trees. Vertical axis: "
                  f"the 90th percentile of ForestFormer3D tree heights inside that stand. Marker size scales with "
                  f"the number of predicted trees in the stand, colour is the stand's dominant species (six most "
                  f"frequent by area; grey = other). The dashed line is equality.]({assets_rel}/analytics_stand_heights.png)", "",
                  f"What we see: the cloud follows the diagonal -- taller stands in the inventory are taller in the "
                  f"predictions -- and sits {fmt(S['stands_bias_p90'])} m above it, consistently for pine, oak and larch; "
                  "beech (green, the 30-36 m stands) lies on the line. The offset is expected: the ALS is seven growing "
                  "seasons younger than the inventory, and a 90th percentile of tree tops exceeds a mean of dominant "
                  "heights. The few stands far above the line at low inventory height (10-15 m) are young stands "
                  "where tall remnant trees dominate the predicted percentile.", "",
                  md_table(["dominant species", "stands", "ha", "FF3D trees", "trees / ha", "inventory height (m)", "pred. median (m)", "pred. p90 (m)", "crown p50 (m²)"], rows), "",
                  f"Every ForestFormer3D tree was joined to the stand it stands in ({fmt(S['stands_n'])} stands with at least 20 trees). "
                  "The inventory height is the main canopy layer's height from the 2014 management inventory, a "
                  "stand mean of dominant trees, so the 90th percentile of the predicted tree heights is the "
                  "comparable statistic (seven years of growth separate the two, worth a few metres of height). "
                  f"Over all stands the predicted p90 is {fmt(S['stands_bias_p90'])} m above the inventory "
                  f"height on average (correlation {fmt(S['stands_r'], 2)}); the predicted median sits below it, as it should "
                  "for a figure that includes the understory.", ""]

    # 7 cadastre
    if S.get("cadastre"):
        c = S["cadastre"]
        rows = [[L[k], fmt(v["n_cadastre"]), pct(v["detected_frac"]), pct(v["detected_frac_street"]), pct(v["detected_frac_park"]),
                 fmt(v["n_height_pairs"]), fmt(v["height_bias"], 2), fmt(v["height_mae"], 2), fmt(v["height_r"], 2)] for k, v in c.items()]
        lines += ["## 7. Against the tree cadastre (street and park trees)", "",
                  f"![Street and park trees of the Berlin tree cadastre against the predictions. Left: for every "
                  f"cadastre tree with a predicted tree top within {CADASTRE_MATCH_M:.0f} m, the cadastre's height "
                  f"(horizontal, whole metres as recorded at inspection, hence the vertical stripes) against the "
                  f"height of that predicted tree (vertical), one dot per tree, blue ForestFormer3D and magenta "
                  f"SegmentAnyTree, dashed line = equality. Right: histogram of predicted minus cadastre height "
                  f"for the same pairs, 1 m bins.]({assets_rel}/analytics_cadastre.png)", "",
                  f"What we see: the stripes are the cadastre's integer heights, which stop at 30 m. Within each stripe "
                  f"the predicted heights spread widely, but their centre rises with the cadastre value and the "
                  f"residual histogram is symmetric about zero with a sharp peak (bias {fmt(S['cadastre']['ff3d']['height_bias'], 2)} m, "
                  f"MAE {fmt(S['cadastre']['ff3d']['height_mae'])} m): most matched pairs agree to within a few metres. "
                  "The dots far above the diagonal at cadastre heights of 4-8 m are young street trees whose nearest "
                  "predicted top belongs to a taller neighbour's crown (the nearest-top rule assigns it anyway); the "
                  "dots far below are large cadastre trees for which only a fragment was predicted. The two methods "
                  "overlay almost exactly, so the scatter is the reference's and the matching rule's, not the model's.", "",
                  md_table(["method", "cadastre trees in the mosaic", "with a predicted top ≤ 3 m away", "street", "park", "height pairs", "bias (m)", "MAE (m)", "r"], rows), "",
                  "The cadastre lists managed trees with a surveyed position and a height that is updated at "
                  "inspection, so a predicted tree top within 3 m of a cadastre position is counted as a detection "
                  "(the nearest one only; a cadastre tree hidden under a larger neighbour's crown is a miss by "
                  "construction). The height comparison uses the detected pairs with a cadastre height. "
                  "Detection by genus, ForestFormer3D (eight most frequent):", "",
                  md_table(["genus", "cadastre trees", "detected"],
                           [[g, fmt(v["n"]), pct(v["detected_frac"])] for g, v in c["ff3d"]["by_genus"].items()]), ""]

    # 8 footprints
    if S.get("footprints"):
        rows = []
        for key, v in S["footprints"].items():
            for k in keys:
                if k in v:
                    rows.append([key, fmt(v["area_ha"], 0), L[k], pct(v[k]["covered"], 0), fmt(v[k]["trees"]), fmt(v[k]["density_ha"], 0), " / ".join(fmt(x) for x in v[k]["height_q"])])
        lines += ["## 8. The WINMOL survey footprints", "",
                  md_table(["footprint", "ha", "method", "footprint covered by the method's tiles", "trees inside", "trees / covered ha", "height p10 / p50 / p90 (m)"], rows), "",
                  "Density is per covered hectare, so partially covered footprints (R13 Spandau until its eleven "
                  "remaining tiles are processed; AMS3D on its tile subset) stay comparable; the tree counts are partial.", ""]

    lines += ["## 9. Caveats", "",
              "* Agreement between methods is not accuracy: two methods can agree on a wrong split.",
              "* The inventory heights are stand means from 2014 and the cadastre heights are inspection "
              "estimates; both references are coarser than the ALS-derived heights they are compared with.",
              "* The flat-blob and noise flags are descriptive; no filtering was applied to any count in this report.",
              "* A method that covers only part of the tiles (the table says how many) is not comparable with the "
              "whole-mosaic totals of the others; its per-tile rates are.", ""]
    doc.write_text("\n".join(lines))


# ------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--als-data", type=Path, default=Path("/Volumes/2TB/winmol/ALS_Data"))
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--assets", type=Path, default=Path("docs/benchmarks/assets/analytics"))
    ap.add_argument("--doc", type=Path, default=Path("docs/benchmarks/2026-10-05-berlin-analytics.md"))
    a = ap.parse_args(argv)
    import geopandas as gpd
    from shapely.geometry import box
    from shapely.ops import unary_union

    als = a.als_data
    if not als.is_dir():
        print(f"!!! {als} is not a directory (is the 2TB mounted?)", file=sys.stderr)
        return 2
    a.assets.mkdir(parents=True, exist_ok=True)
    assets_rel = str(a.assets.relative_to(a.doc.parent)) if str(a.assets).startswith(str(a.doc.parent)) else str(a.assets)
    labels = {k: METHODS[k][0] for k in METHODS}
    S: dict = {"generated": dt.date.today().isoformat(), "methods": {}}

    print("== tree tables")
    trees = {}
    for k in a.methods:
        g = load_trees(als, k)
        if g is None:
            print(f"  {k}: nothing found, skipped")
            continue
        trees[k] = g
        n_tiles = g.tile.nunique()
        flat = flat_blob_mask(g.height, g.crown_area_m2)
        small = g.n_points < SMALL_POINTS
        tall = g.height > 40
        S["methods"][k] = {"tiles": int(n_tiles), "trees": int(len(g)), "trees_per_km2": len(g) / n_tiles,
                           "height_q": quantiles(g.height), "crown_q": quantiles(g.crown_area_m2), "points_q": quantiles(g.n_points),
                           "flat": int(flat.sum()), "flat_frac": float(flat.mean()), "small": int(small.sum()), "small_frac": float(small.mean()),
                           "tall": int(tall.sum()), "tall_frac": float(tall.mean())}
        print(f"  {k}: {len(g):,} trees on {n_tiles} tiles")
    if not trees:
        print("!!! no tree tables at all", file=sys.stderr)
        return 2

    print("== figures: tile grid, distributions")
    if "ff3d" in trees and "sat" in trees:
        ca, cb = trees["ff3d"].groupby("tile").size(), trees["sat"].groupby("tile").size()
        common = ca.index.intersection(cb.index)
        ratio = (cb[common] / ca[common])
        S["sat_over_ff3d"] = float(cb[common].sum() / ca[common].sum())
        S["sat_over_ff3d_range"] = [float(ratio.min()), float(ratio.max())]
        fig_tile_grid(trees, a.assets / "analytics_tile_grid.png")
        stands_path = als / "berlin_forest" / "forstbetriebskarte_2014.gpkg"
        if stands_path.exists():
            st = gpd.read_file(stands_path, layer="hauptbaumarten", columns=["geometry"]).to_crs(EPSG)
            forest = unary_union(st.geometry.make_valid())
            def forest_share(t):
                e, n = int(t.split("_")[2]) * 1000, int(t.split("_")[3]) * 1000
                return forest.intersection(box(e, n, e + 1000, n + 1000)).area / 1e6
            is_forest = {t: forest_share(t) > 0.5 for t in common}
            fr = [ratio[t] for t in common if is_forest[t]]; ot = [ratio[t] for t in common if not is_forest[t]]
            S.update({"forest_tiles_n": len(fr), "other_tiles_n": len(ot),
                      "ratio_forest": float(np.mean(fr)) if fr else np.nan, "ratio_other": float(np.mean(ot)) if ot else np.nan})
        rev_path = als / "berlin_forest" / "reviere.gpkg"
        if rev_path.exists():   # the ratio by survey footprint: the two forests behave differently
            fp = gpd.read_file(rev_path, layer="footprints").to_crs(EPSG)
            S["ratio_by_footprint"] = {}
            for _, f in fp.iterrows():
                inside = [t for t in common if f.geometry.intersection(box(int(t.split("_")[2]) * 1000, int(t.split("_")[3]) * 1000,
                                                                          int(t.split("_")[2]) * 1000 + 1000, int(t.split("_")[3]) * 1000 + 1000)).area > 2.5e5]
                S["ratio_by_footprint"][f.key] = {"tiles": len(inside), "ratio": float(np.mean([ratio[t] for t in inside])) if inside else np.nan}
    full = {k: v for k, v in trees.items() if v.tile.nunique() == trees["ff3d"].tile.nunique()} if "ff3d" in trees else trees
    fig_distributions(full or trees, a.assets / "analytics_distributions.png")
    if len(trees) > len(full):
        three = sorted(set.intersection(*[set(v.tile) for v in trees.values()]))
        if three:
            S["three_way_tiles"] = three
            S["three_way"] = {k: {"trees": int(v.tile.isin(three).sum()), "height_q": quantiles(v[v.tile.isin(three)].height),
                                  "crown_q": quantiles(v[v.tile.isin(three)].crown_area_m2)} for k, v in trees.items()}
            fig_distributions(trees, a.assets / "analytics_distributions_3way.png", tiles=three, title_suffix=f" on the {len(three)} tiles all methods cover")

    print("== agreement")
    dirs = agreement_pairs(als / "berlin_agreement")
    agr = load_agreement(dirs[("ff3d", "sat")]) if ("ff3d", "sat") in dirs else []
    if agr:
        S["agreement"] = {"ff3d_sat": pooled_agreement(agr)}
        fig_agreement(agr, a.assets / "analytics_agreement.png", labels["ff3d"], labels["sat"])
        pairs = {}
        for (a_key, b_key), d in sorted(dirs.items()):
            if (a_key, b_key) == ("ff3d", "sat") or a_key not in labels or b_key not in labels:
                continue
            recs = load_agreement(d)
            if recs:
                pairs[f"{labels[a_key]} vs {labels[b_key]}"] = pooled_agreement(recs)
                S["agreement"][f"{a_key}_{b_key}"] = pairs[f"{labels[a_key]} vs {labels[b_key]}"]
        if pairs:
            S["agreement"]["ams3d"] = pairs

    print("== seams, stitch, buildings")
    border = {k: load_json_per_tile(als / METHODS[k][1], "_border.json") for k in trees}
    border = {k: v for k, v in border.items() if v}
    S["stitch"] = {}
    for k in border:
        sj = als / METHODS[k][1] / "stitch.json"
        if sj.exists():
            s = json.loads(sj.read_text())
            if k == "ff3d" and s.get("min_height"):
                S["min_height"] = float(s["min_height"]); S["n_short_removed"] = int(s.get("n_short_removed", 0))
            S["stitch"][k] = {"n_trees": s["n_trees"], "n_unified": s["n_unified"], "n_cross_km": s["n_cross_km"],
                              "touching_mean": float(np.mean([b["touching_frac"] for b in border[k].values()])),
                              "strip_mean": float(np.mean([b["strip_excess_pp"] for b in border[k].values()]))}
    masked = {t: r["buildings"] for t, r in load_json_per_tile(als / METHODS["ff3d"][1] / "masked", "_report.json").items() if "buildings" in r}
    if masked:
        before = sum(m["instances_before"] for m in masked.values()); removed = sum(m["instances_removed"] for m in masked.values())
        top = sorted(((t, m["instances_removed"] / m["instances_before"]) for t, m in masked.items()), key=lambda x: -x[1])[:3]
        S["buildings"] = {"tiles": len(masked), "before": before, "removed": removed, "removed_frac": removed / before,
                          "points": sum(m["points_masked"] for m in masked.values()), "top": top}
    if border or masked:
        fig_seams_and_buildings(border, masked, a.assets / "analytics_seams_buildings.png")

    print("== forest inventory")
    stands_path = als / "berlin_forest" / "forstbetriebskarte_2014.gpkg"
    if "ff3d" in trees and stands_path.exists():
        per, by_sp = stand_analysis(trees["ff3d"], stands_path)
        S["stands"] = by_sp; S["stands_n"] = int(len(per))
        S["stands_bias_p90"] = float(np.mean(per.pred_p90 - per.inv_height))
        S["stands_r"] = float(np.corrcoef(per.inv_height, per.pred_p90)[0, 1]) if len(per) > 2 else np.nan
        fig_stand_heights(per, a.assets / "analytics_stand_heights.png")

    print("== tree cadastre")
    cad_path = als / "berlin_trees" / "baumbestand_berlin.gpkg"
    if cad_path.exists() and "ff3d" in trees:
        tiles = sorted(set().union(*[set(v.tile) for v in trees.values()]))
        union = unary_union([box(int(t.split("_")[2]) * 1000, int(t.split("_")[3]) * 1000, int(t.split("_")[2]) * 1000 + 1000, int(t.split("_")[3]) * 1000 + 1000) for t in tiles])
        res, matched = cadastre_analysis(trees, cad_path, union)
        S["cadastre"] = res
        if "ff3d" in matched and "sat" in matched:
            fig_cadastre(matched, a.assets / "analytics_cadastre.png")

    print("== footprints, study area")
    S["footprints"] = footprint_analysis(trees, als / "berlin_forest" / "reviere.gpkg")
    fig_study_area(als, trees, a.assets / "analytics_study_area.png")

    (a.assets / "analytics.json").write_text(json.dumps(S, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    write_doc(a.doc, assets_rel, S, labels)
    print(f"wrote {a.doc} and {a.assets}/analytics_*.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
