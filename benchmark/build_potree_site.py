#!/usr/bin/env python3
"""Build the data/ payload and index.html of the Berlin ForestFormer3D Potree site.

The Potree octrees themselves are produced separately with PotreeConverter 2.1.1
(see docs/benchmarks/2026-09-23-potree-viewer.md); this script only builds the
light-weight overlays that the viewer drapes on top of them:

  data/<tile>_trees.geojson     tree tops   (EPSG:25833 coordinates kept as x/y/z)
  data/<tile>_crowns.geojson    crown hulls (EPSG:25833, simplified)
  data/<tile>_chm.png|.json     0.5 m canopy height model from the LAS, viridis
  data/<tile>_instance.png|.json  instance raster, random colours, transparent nodata
  data/<tile>_dop2021.png|.json   leaf-off orthophoto  (downsampled)
  data/<tile>_dop2025.png|.json   leaf-on orthophoto   (downsampled)
  data/tiles.json               manifest read by index.html

With ``--variant sat --sat-dir <dir>`` it instead attaches a second segmentation
(SegmentAnyTree, converted to the same contract by benchmark/sat_to_ff3d.py) to tiles
already in the manifest: the octree is expected under ``pointclouds_sat/<tile>/``, and
``data/<tile>_sat_trees.geojson``, ``_sat_crowns.geojson`` and ``_sat_instance.png|.json``
are written; ``tiles.json`` gets ``variants.sat`` per tile. CHM and orthophotos are shared.

Example
-------
    .venv-cpu/bin/python benchmark/build_potree_site.py \
        --site /Volumes/2TB/winmol/ALS_Data/berlin_potree \
        --ff3d-dir /Volumes/2TB/winmol/ALS_Data/berlin_als_2021_ff3d \
        --dop2021-dir /Volumes/2TB/winmol/ALS_Data/berlin_dop_2021/dop_2021_rgb \
        --dop2025-dir /Volumes/2TB/winmol/ALS_Data/berlin_dop_2025_sommer
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ff3d_geo.grid import GridExtent, nearest_fill  # noqa: E402

GROUND_CLASSES = (2,)
CRS = "EPSG:25833"


# --------------------------------------------------------------------------- utils
def _round_coords(geom, ndigits=1):
    """Recursively round a GeoJSON coordinate structure."""
    if isinstance(geom, (list, tuple)):
        if geom and isinstance(geom[0], (int, float)):
            return [round(float(v), ndigits) for v in geom]
        return [_round_coords(g, ndigits) for g in geom]
    return geom


def _png(path: Path, rgba: np.ndarray) -> None:
    from PIL import Image

    mode = "RGBA" if rgba.shape[2] == 4 else "RGB"
    Image.fromarray(rgba, mode).save(path, optimize=True)


def _extent_json(path: Path, bounds, extra=None) -> None:
    doc = {
        "crs": CRS,
        "bounds": [round(float(v), 3) for v in bounds],  # minx, miny, maxx, maxy
    }
    if extra:
        doc.update(extra)
    path.write_text(json.dumps(doc, indent=1))


def _random_lut(n: int, seed: int = 20260923) -> np.ndarray:
    """n distinct-ish RGB colours, bright enough to read on a dark background."""
    rng = np.random.default_rng(seed)
    hsv = np.stack(
        [rng.random(n), 0.45 + 0.5 * rng.random(n), 0.55 + 0.4 * rng.random(n)], axis=1
    )
    import matplotlib.colors as mcolors

    return (mcolors.hsv_to_rgb(hsv) * 255).astype(np.uint8)


# ----------------------------------------------------------------------- overlays
def tile_bounds_from_name(las_header) -> tuple[float, float, float, float]:
    """1 km tile bounds, snapped to the km grid the Berlin tiles use."""
    x0 = math.floor(float(las_header.mins[0]) / 1000.0) * 1000.0
    y0 = math.floor(float(las_header.mins[1]) / 1000.0) * 1000.0
    return x0, y0, x0 + 1000.0, y0 + 1000.0


def build_chm(las_path: Path, bounds, cell: float, dtm_cell: float):
    """0.5 m CHM over the fixed tile grid. Returns (chm, zmin, zmax) with row 0 = south."""
    import laspy

    x0, y0, x1, y1 = bounds
    nx = int(round((x1 - x0) / cell))
    ny = int(round((y1 - y0) / cell))
    ext = GridExtent(x0, y0, cell, nx, ny)
    step = int(round(dtm_cell / cell))
    cnx, cny = nx // step, ny // step
    cext = GridExtent(x0, y0, dtm_cell, cnx, cny)

    dsm = np.full(ny * nx, -np.inf)
    dtm = np.full(cny * cnx, np.inf)
    zmin, zmax = np.inf, -np.inf

    with laspy.open(str(las_path)) as f:
        for pts in f.chunk_iterator(4_000_000):
            x = np.asarray(pts.x, dtype=np.float64)
            y = np.asarray(pts.y, dtype=np.float64)
            z = np.asarray(pts.z, dtype=np.float64)
            zmin, zmax = min(zmin, float(z.min())), max(zmax, float(z.max()))
            ix, iy = ext.index(x, y)
            np.maximum.at(dsm, iy * nx + ix, z)
            g = np.isin(np.asarray(pts.classification, dtype=np.int64), GROUND_CLASSES)
            if g.any():
                cix, ciy = cext.index(x[g], y[g])
                np.minimum.at(dtm, ciy * cnx + cix, z[g])

    dsm = dsm.reshape(ny, nx)
    dsm[~np.isfinite(dsm)] = np.nan
    dtm = dtm.reshape(cny, cnx)
    dtm[~np.isfinite(dtm)] = np.nan
    dtm = nearest_fill(dtm) if not np.isnan(dtm).all() else np.full_like(dtm, zmin)
    dtm_fine = np.repeat(np.repeat(dtm, step, axis=0), step, axis=1)[:ny, :nx]

    chm = dsm - dtm_fine
    chm[~np.isfinite(chm)] = 0.0
    np.clip(chm, 0.0, None, out=chm)
    return chm, zmin, zmax, float(np.nanmedian(dtm))


def write_chm_png(chm: np.ndarray, out_png: Path, out_json: Path, bounds, vmax, ground_z):
    import matplotlib

    try:
        cmap = matplotlib.colormaps["viridis"]
    except AttributeError:  # matplotlib < 3.5
        import matplotlib.cm as cm

        cmap = cm.get_cmap("viridis")
    v = np.clip(chm / max(vmax, 1e-6), 0.0, 1.0)
    rgba = (cmap(np.flipud(v)) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.flipud(chm) < 0.5, 0, 235)  # bare ground stays see-through
    _png(out_png, rgba)
    _extent_json(out_json, bounds, {"vmin": 0.0, "vmax": round(float(vmax), 2),
                                    "cmap": "viridis", "z": round(ground_z, 2),
                                    "units": "m above ground"})


def write_instance_png(tif: Path, out_png: Path, out_json: Path, bounds, ground_z):
    import rasterio

    with rasterio.open(tif) as r:
        a = r.read(1)
        nodata = r.nodata if r.nodata is not None else -1
    valid = a != nodata
    lut = _random_lut(4096)
    rgba = np.zeros(a.shape + (4,), dtype=np.uint8)
    idx = np.zeros(a.shape, dtype=np.int64)
    idx[valid] = (a[valid].astype(np.int64) * 2654435761) % 4096  # Knuth hash
    rgba[..., :3] = lut[idx]
    rgba[..., 3] = np.where(valid, 235, 0)
    _png(out_png, rgba)
    _extent_json(out_json, bounds, {"z": round(ground_z, 2), "nodata": int(nodata)})


def write_dop_png(tif: Path, out_png: Path, out_json: Path, bounds, px: int, ground_z):
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(tif) as r:
        n = min(3, r.count)
        a = r.read(
            list(range(1, n + 1)),
            out_shape=(n, px, px),
            resampling=Resampling.average,
        )
    rgb = np.transpose(a, (1, 2, 0)).astype(np.uint8)
    _png(out_png, rgb)
    _extent_json(out_json, bounds, {"z": round(ground_z, 2)})


def write_vectors(ff3d_dir: Path, tile: str, data_dir: Path, infix: str = ""):
    """``data/<tile><infix>_trees.geojson`` and ``_crowns.geojson`` from ``<dir>/<tile>/``."""
    import geopandas as gpd
    from shapely.geometry import mapping

    out = {}
    trees = gpd.read_file(ff3d_dir / tile / f"{tile}_trees.gpkg")
    feats = []
    for row in trees.itertuples(index=False):
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [round(float(row.x), 2), round(float(row.y), 2),
                                         round(float(row.top_z), 2)]},
            "properties": {
                "tree_id": int(row.tree_id),
                "height": round(float(row.height), 2),
                "crown_area_m2": round(float(row.crown_area_m2), 2),
                "n_points": int(row.n_points),
                "mean_score": round(float(row.mean_score), 3),
                "top_z": round(float(row.top_z), 2),
            },
        })
    p = data_dir / f"{tile}{infix}_trees.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "crs_note": CRS,
                             "features": feats}))
    out["trees"] = {"file": p.name, "count": len(feats)}

    crowns = gpd.read_file(ff3d_dir / tile / f"{tile}_crowns.gpkg")
    feats = []
    for row in crowns.itertuples(index=False):
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != "Polygon":
            continue
        g = mapping(geom.simplify(0.4, preserve_topology=False))
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": _round_coords(g["coordinates"], 1)},
            "properties": {"tree_id": int(row.tree_id), "top_z": round(float(row.top_z), 2)},
        })
    p = data_dir / f"{tile}{infix}_crowns.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "crs_note": CRS,
                             "features": feats}))
    out["crowns"] = {"file": p.name, "count": len(feats)}
    return out


def attach_variant(recs: list[dict], tile: str, name: str, sub: dict) -> dict:
    """Store ``sub`` as ``variants[name]`` of the manifest record of ``tile``.

    The top-level fields of a record stay the ForestFormer3D variant (the viewer's
    default, and what older manifests hold); a second method only ever adds to
    ``variants``. Raises ``KeyError`` when the tile is not in the manifest, because a
    variant without the base record (CHM, DOPs, bounds) cannot be shown.
    """
    for rec in recs:
        if rec["tile"] == tile:
            rec.setdefault("variants", {})[name] = sub
            return rec
    raise KeyError(f"{tile} is not in the manifest; build the ForestFormer3D variant first")


def build_variant(tile: str, name: str, args_dict: dict) -> dict:
    """The ``variants[name]`` sub-record: octree under ``pointclouds_<name>/<tile>/``,
    vectors and instance overlay from ``<variant dir>/<tile>/`` with ``_<name>_`` names."""
    a = argparse.Namespace(**args_dict)
    src_dir = Path(a.variant_dir)
    site = Path(a.site)
    data_dir = site / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    import laspy

    meta = json.loads((site / f"pointclouds_{name}" / tile / "metadata.json").read_text())
    las = src_dir / tile / f"{tile}.las"
    hdr = laspy.open(str(las)).header
    bounds = tile_bounds_from_name(hdr)
    ground_z = json.loads((data_dir / f"{tile}_chm.json").read_text())["z"]
    sub: dict = {
        "pointcloud": f"pointclouds_{name}/{tile}/metadata.json",
        "points": meta["points"],
        "attributes": {at["name"]: {"min": at.get("min"), "max": at.get("max")}
                       for at in meta["attributes"]},
        "layers": {},
    }
    inst = src_dir / tile / f"{tile}_instance_50cm.tif"
    if inst.exists():
        write_instance_png(inst, data_dir / f"{tile}_{name}_instance.png",
                           data_dir / f"{tile}_{name}_instance.json", bounds, ground_z)
        sub["layers"]["instance"] = {"png": f"data/{tile}_{name}_instance.png",
                                     "json": f"data/{tile}_{name}_instance.json"}
    vec = write_vectors(src_dir, tile, data_dir, infix=f"_{name}")
    sub["trees"] = {"file": f"data/{vec['trees']['file']}", "count": vec["trees"]["count"]}
    sub["crowns"] = {"file": f"data/{vec['crowns']['file']}", "count": vec["crowns"]["count"]}
    return sub


# --------------------------------------------------------------------------- tile
def build_tile(tile: str, args_dict: dict) -> dict:
    a = argparse.Namespace(**args_dict)
    import laspy

    ff3d_dir = Path(a.ff3d_dir)
    data_dir = Path(a.site) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((Path(a.site) / "pointclouds" / tile / "metadata.json").read_text())
    las = ff3d_dir / tile / f"{tile}.las"
    hdr = laspy.open(str(las)).header
    bounds = tile_bounds_from_name(hdr)
    rec: dict = {
        "tile": tile,
        "pointcloud": f"pointclouds/{tile}/metadata.json",
        "points": meta["points"],
        "bounds": [round(float(v), 2) for v in bounds],
        "z_range": [round(float(hdr.mins[2]), 2), round(float(hdr.maxs[2]), 2)],
        "attributes": {at["name"]: {"min": at.get("min"), "max": at.get("max")}
                       for at in meta["attributes"]},
        "layers": {},
    }

    chm, zmin, zmax, ground_z = build_chm(las, bounds, a.chm_cell, a.dtm_cell)
    rec["ground_z"] = round(ground_z, 2)
    vmax = float(np.percentile(chm[chm > 0.5], 99)) if (chm > 0.5).any() else 30.0
    vmax = max(10.0, min(45.0, vmax))
    write_chm_png(chm, data_dir / f"{tile}_chm.png", data_dir / f"{tile}_chm.json",
                  bounds, vmax, ground_z)
    rec["layers"]["chm"] = {"png": f"data/{tile}_chm.png", "json": f"data/{tile}_chm.json",
                            "vmax": round(vmax, 1)}

    inst = ff3d_dir / tile / f"{tile}_instance_50cm.tif"
    if inst.exists():
        write_instance_png(inst, data_dir / f"{tile}_instance.png",
                           data_dir / f"{tile}_instance.json", bounds, ground_z)
        rec["layers"]["instance"] = {"png": f"data/{tile}_instance.png",
                                     "json": f"data/{tile}_instance.json"}

    for key, src_dir in (("dop2021", a.dop2021_dir), ("dop2025", a.dop2025_dir)):
        if not src_dir:
            continue
        tif = Path(src_dir) / f"{tile}.tif"
        if not tif.exists():
            continue
        write_dop_png(tif, data_dir / f"{tile}_{key}.png", data_dir / f"{tile}_{key}.json",
                      bounds, a.texture_px, ground_z)
        rec["layers"][key] = {"png": f"data/{tile}_{key}.png",
                              "json": f"data/{tile}_{key}.json"}

    vec = write_vectors(ff3d_dir, tile, data_dir)
    rec["trees"] = {"file": f"data/{vec['trees']['file']}", "count": vec["trees"]["count"]}
    rec["crowns"] = {"file": f"data/{vec['crowns']['file']}", "count": vec["crowns"]["count"]}
    return rec


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", required=True, help="output site root (holds pointclouds/)")
    p.add_argument("--ff3d-dir", default=None, help="dir with <tile>/<tile>.las + gpkg + tif")
    p.add_argument("--variant", default=None, choices=["sat"],
                   help="attach a second method to tiles already in the manifest")
    p.add_argument("--variant-dir", "--sat-dir", dest="variant_dir", default=None,
                   help="dir with <tile>/<tile>.las + gpkg + tif of that method")
    p.add_argument("--dop2021-dir", default=None)
    p.add_argument("--dop2025-dir", default=None)
    p.add_argument("--tiles", nargs="*", default=None, help="subset of tile names")
    p.add_argument("--texture-px", type=int, default=1792, help="orthophoto texture size")
    p.add_argument("--chm-cell", type=float, default=0.5)
    p.add_argument("--dtm-cell", type=float, default=2.0)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--skip-existing", action="store_true",
                   help="keep tiles already present in data/tiles.json")
    p.add_argument("--no-index", action="store_true",
                   help="do not refresh <site>/index.html from benchmark/potree_index.html")
    a = p.parse_args()

    site = Path(a.site)
    manifest_path = site / "data" / "tiles.json"
    if a.variant:
        return build_variant_main(a, site, manifest_path)
    if not a.ff3d_dir:
        p.error("--ff3d-dir is required unless --variant is given")

    pc_dir = site / "pointclouds"
    tiles = a.tiles or sorted(
        d.name for d in pc_dir.iterdir()
        if (d / "metadata.json").exists() and (Path(a.ff3d_dir) / d.name).is_dir()
    )
    if not tiles:
        print(f"no tiles found under {pc_dir}", file=sys.stderr)
        return 1

    old = {}
    if a.skip_existing and manifest_path.exists():
        old = {t["tile"]: t for t in json.loads(manifest_path.read_text())["tiles"]}
        tiles = [t for t in tiles if t not in old]

    args_dict = vars(a)
    recs = list(old.values())
    if tiles:
        print(f"building {len(tiles)} tile(s) with {a.jobs} job(s)")
        with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            futs = {ex.submit(build_tile, t, args_dict): t for t in tiles}
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    recs.append(fut.result())
                    print(f"  ok   {t}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"  FAIL {t}: {exc!r}", flush=True)

    recs.sort(key=lambda r: r["tile"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"crs": CRS, "tiles": recs}, indent=1))
    print(f"wrote {manifest_path} ({len(recs)} tiles)")

    write_index(site, a.no_index)
    return 0


def write_index(site: Path, no_index: bool) -> None:
    if no_index:
        return
    tpl = Path(__file__).with_name("potree_index.html")
    (site / "index.html").write_text(tpl.read_text())
    print(f"wrote {site / 'index.html'} from {tpl}")


def build_variant_main(a, site: Path, manifest_path: Path) -> int:
    """``--variant <name>``: add that method's octrees/vectors to the existing manifest."""
    name = a.variant
    if not a.variant_dir:
        print("--variant needs --variant-dir (--sat-dir)", file=sys.stderr)
        return 1
    if not manifest_path.exists():
        print(f"{manifest_path} missing: build the ForestFormer3D site first", file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text())
    recs = manifest["tiles"]
    known = {r["tile"] for r in recs}
    pc_dir = site / f"pointclouds_{name}"
    tiles = a.tiles
    if not tiles and pc_dir.is_dir():
        tiles = sorted(
            d.name for d in pc_dir.iterdir()
            if (d / "metadata.json").exists()
            and (Path(a.variant_dir) / d.name / f"{d.name}.las").exists()
        )
    if not tiles:
        print(f"no {name} tiles found under {pc_dir}", file=sys.stderr)
        return 1
    if a.skip_existing:
        tiles = [t for t in tiles if not (next((r for r in recs if r["tile"] == t), {})
                                          .get("variants", {}).get(name))]
    args_dict = vars(a)
    print(f"building {name} variant for {len(tiles)} tile(s) with {a.jobs} job(s)")
    with ProcessPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        futs = {ex.submit(build_variant, t, name, args_dict): t for t in tiles if t in known}
        for t in set(tiles) - known:
            print(f"  SKIP {t}: not in the manifest (no ForestFormer3D variant)")
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                attach_variant(recs, t, name, fut.result())
                print(f"  ok   {t}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL {t}: {exc!r}", flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    n = sum(1 for r in recs if r.get("variants", {}).get(name))
    print(f"wrote {manifest_path} ({n} tiles carry the {name} variant)")
    write_index(site, a.no_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
